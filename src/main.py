"""ETVL pipeline orchestrator entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.common.errors import ValidationError
from src.common.cost import log_summary
from src.common.logger import configure_logging, log_errors
from src.config import Settings, get_settings
from src.extract import extract_page_image, extract_page_text
from src.load import stitch
from src.models import DictionaryEntry
from src.transform import SYSTEM_PROMPT, build_user_prompt, extract_json
from src.validate import persist_validated_page, validate

logger = logging.getLogger(__name__)


def _write_raw(out_path: Path, raw_json: str) -> None:
    """Persist a raw_json artifact, creating parent dirs as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(raw_json, encoding="utf-8")
    logger.debug("wrote raw_json %s", out_path)


@log_errors
def _extract_page(page: int, settings: Settings) -> tuple[str, str]:
    """Extract layer: render page image + read OCR text for one printed page."""
    image_b64 = extract_page_image(page)
    ocr_text = extract_page_text(page)
    logger.info(
        "page %d ok (image=%d b64, ocr=%d chars)",
        page,
        len(image_b64),
        len(ocr_text),
    )
    return image_b64, ocr_text


@log_errors
def _transform_page(
    image_b64: str,
    ocr_text: str,
    page: int,
    settings: Settings,
    attempt: int = 1,
) -> str:
    """Transform layer: compile prompt + call VLM, return raw JSON string.

    Raw-JSON persistence rules:
    - attempt 1, debug mode: written to ``raw_page_path`` here.
    - attempt 1, running mode: written to ``raw_page_path`` by the
      orchestrator only if validation fails, so failure artifacts survive.
    - attempt > 1, any mode: always written to ``raw_retry_page_path``.
    """
    raw_json = extract_json(
        image_b64,
        SYSTEM_PROMPT,
        build_user_prompt(ocr_text),
        page=page,
    )
    logger.info(
        "attempt %d ok (raw_json=%d chars)", attempt, len(raw_json)
    )

    if attempt == 1:
        if settings.mode == "debug":
            _write_raw(settings.raw_page_path(page, settings.model), raw_json)
    else:
        _write_raw(
            settings.raw_retry_page_path(page, settings.model, attempt), raw_json
        )
    return raw_json


@log_errors
def _validate_page(
    raw_json: str,
    ocr_text: str,
    image_b64: str,
    page: int,
    settings: Settings,
) -> list[DictionaryEntry]:
    """Validate layer: schema + grounding + deterministic injection."""
    entries = validate(raw_json, ocr_text, image_b64, page, settings)
    if settings.mode == "debug":
        persist_validated_page(entries, page, settings)
    logger.info("ok (%d entries)", len(entries))
    return entries


@log_errors
def _process_page_with_retries(
    page: int, settings: Settings
) -> list[DictionaryEntry]:
    """Run E once -> (T -> V) up to ``settings.max_attempts`` times for one page.

    Extraction (image + OCR) is performed once and cached across retries;
    only the transform (zero-shot VLM call) and validate layers re-run on
    each retry. Only ``ValidationError`` — from validation, or from the
    transform layer (e.g. a response with no choices) — triggers a retry;
    any other exception propagates and kills the pipeline immediately.

    Raises ``ValidationError`` if every attempt fails.
    """
    max_attempts = settings.max_attempts
    logger.info("page %d: start (mode=%s)", page, settings.mode)
    image_b64, ocr_text = _extract_page(page, settings)

    last_error: ValidationError | None = None
    for attempt in range(1, max_attempts + 1):
        raw_json: str | None = None
        try:
            raw_json = _transform_page(
                image_b64, ocr_text, page, settings, attempt=attempt
            )
            entries = _validate_page(raw_json, ocr_text, image_b64, page, settings)
        except ValidationError as e:
            last_error = e
            # In running mode, the first-attempt raw_json wasn't persisted by
            # _transform_page; preserve the failed payload here so prompt
            # failures are debuggable even outside debug mode.
            if raw_json is not None and attempt == 1 and settings.mode != "debug":
                _write_raw(settings.raw_page_path(page, settings.model), raw_json)
            print(f"[page {page}] attempt {attempt}/{max_attempts} failed: {e}")
            logger.warning(
                "page %d attempt %d/%d failed: %s",
                page,
                attempt,
                max_attempts,
                e,
            )
            continue
        logger.info("page %d: complete (%d entries, attempt=%d)",
                     page, len(entries), attempt)
        return entries

    raise ValidationError(
        f"page {page} failed after {max_attempts} attempts: {last_error}"
    )


def main() -> None:
    """Iterate over a page range, run E->T->V per page, then load-stitch.

    On the first page that fails all retries (or raises a non-
    ``ValidationError`` exception), the pipeline stops processing further
    pages, stitches every succeeded page to disk so partial progress
    survives, and exits with a non-zero status.
    """
    parser = argparse.ArgumentParser(
        description="Extract & LLM-transform a page range of a VS volume.",
    )
    parser.add_argument("--start", type=int, required=True, help="first printed page")
    parser.add_argument("--end", type=int, required=True, help="last printed page")
    parser.add_argument(
        "--mode",
        choices=["debug", "running"],
        help="execution mode (overrides VS_MODE); 'debug' persists per-page artifacts "
        "from transform and validate, 'running' keeps validated entries in memory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="parallelize the transform/validate loop across this many pages at once. "
        "If omitted, pages are processed sequentially. Must be <= total pages in range.",
    )
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be >= --start")

    total_pages = args.end - args.start + 1
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be >= 1")
        if args.batch_size > total_pages:
            parser.error(
                f"--batch-size {args.batch_size} exceeds page count {total_pages}"
            )

    settings = get_settings()
    if args.mode:
        settings = settings.model_copy(update={"mode": args.mode})

    configure_logging(
        settings.log_file,
        src_level=logging.DEBUG if settings.mode == "debug" else logging.INFO,
    )
    logger.info(
        "pipeline start: volume=%s pages=%d-%d model=%s mode=%s batch_size=%s",
        settings.volume,
        args.start,
        args.end,
        settings.model,
        settings.mode,
        args.batch_size if args.batch_size is not None else "sequential",
    )

    try:
        if args.batch_size is None:
            entries_by_page, fatal = _run_sequential(args.start, args.end, settings)
        else:
            entries_by_page, fatal = _run_parallel(
                args.start, args.end, args.batch_size, settings
            )

        stitched_path = stitch(entries_by_page, settings)
        print(f"[load] stitched {len(entries_by_page)} pages -> {stitched_path}")
        logger.info(
            "pipeline end: pages=%d-%d stitched=%s entries=%d fatal=%s",
            args.start,
            args.end,
            stitched_path,
            sum(len(v) for v in entries_by_page.values()),
            "yes" if fatal else "no",
        )

        if fatal is not None:
            sys.exit(1)
    finally:
        # Always emit the cost summary so the running tally survives even
        # if a fatal page error, stitch failure, or sys.exit short-circuits.
        if settings.track_cost:
            log_summary()


def _run_sequential(
    start: int, end: int, settings: Settings
) -> tuple[dict[int, list[DictionaryEntry]], Exception | None]:
    """Sequential page-by-page E->T->V loop; stops at the first fatal page."""
    entries_by_page: dict[int, list[DictionaryEntry]] = {}
    fatal: Exception | None = None
    for page in range(start, end + 1):
        try:
            entries = _process_page_with_retries(page, settings)
        except Exception as e:
            print(f"[page {page}] FATAL: {e}")
            logger.error("page %d fatal failure: %s", page, e)
            fatal = e
            break
        entries_by_page[page] = entries
        print(f"[page {page}] validated ({len(entries)} entries)")
    return entries_by_page, fatal


def _run_parallel(
    start: int, end: int, batch_size: int, settings: Settings
) -> tuple[dict[int, list[DictionaryEntry]], Exception | None]:
    """Parallel E->T->V loop: pages run concurrently up to ``batch_size``.

    All pages in the range are submitted up front; the ``ThreadPoolExecutor``
    bounds actual concurrency to ``batch_size``. Retries for a single page
    stay sequential (handled inside ``_process_page_with_retries``).

    On the first fatal page failure (any exception escaping
    ``_process_page_with_retries``), pending not-yet-started futures are
    cancelled, in-flight futures are allowed to finish (a thread can't be
    interrupted mid-call), their results are collected, then the pipeline
    stitches whatever succeeded and exits non-zero.
    """
    pages = list(range(start, end + 1))
    entries_by_page: dict[int, list[DictionaryEntry]] = {}
    fatal: Exception | None = None

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(_process_page_with_retries, page, settings): page
            for page in pages
        }
        for future in as_completed(futures):
            page = futures[future]
            if fatal is not None:
                # Drain: a fatal failure already happened. Collect any
                # in-flight results that succeed; swallow exceptions from
                # other pages (we're aborting anyway).
                try:
                    entries = future.result()
                except Exception:
                    continue
                entries_by_page[page] = entries
                print(f"[page {page}] validated ({len(entries)} entries)")
                continue

            try:
                entries = future.result()
            except Exception as e:
                print(f"[page {page}] FATAL: {e}")
                logger.error("page %d fatal failure: %s", page, e)
                fatal = e
                # Cancel not-yet-started futures; in-flight ones can't be
                # interrupted and will be drained above.
                for pending in futures:
                    pending.cancel()
                continue
            entries_by_page[page] = entries
            print(f"[page {page}] validated ({len(entries)} entries)")

    return entries_by_page, fatal


if __name__ == "__main__":
    main()