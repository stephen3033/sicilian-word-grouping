"""Single-pass concurrent ETV pipeline with immediate page persistence."""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.common.cost import log_summary
from src.common.logger import configure_logging, log_errors
from src.config import Settings, get_settings
from src.extract import extract_page_image, extract_page_text
from src.load import PersistenceCoordinator
from src.models import DictionaryEntry
from src.transform import SYSTEM_PROMPT, build_user_prompt, extract_json
from src.validate import persist_validated_page, validate

logger = logging.getLogger(__name__)


@log_errors
def _write_raw(out_path: Path, raw_json: str) -> None:
    """Persist a raw model response, creating parent directories as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(raw_json, encoding="utf-8")
    logger.debug("wrote raw_json %s", out_path)


@log_errors
def _extract_page(page: int, settings: Settings) -> tuple[str, str]:
    """Render the page image and load its OCR context."""
    image_b64 = extract_page_image(page)
    ocr_text = extract_page_text(page)
    logger.info(
        "page %d extract ok (image=%d b64, ocr=%d chars)",
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
) -> str:
    """Make the page's single model request and return its raw JSON text."""
    raw_json = extract_json(
        image_b64,
        SYSTEM_PROMPT,
        build_user_prompt(ocr_text),
        page=page,
    )
    logger.info("page %d transform ok (raw_json=%d chars)", page, len(raw_json))
    if settings.mode == "debug":
        _write_raw(settings.raw_page_path(page, settings.model), raw_json)
    return raw_json


@log_errors
def _validate_page(
    raw_json: str,
    ocr_text: str,
    image_b64: str,
    page: int,
    settings: Settings,
) -> list[DictionaryEntry]:
    """Enforce the page schema and annotate all entry-level findings."""
    entries = validate(raw_json, ocr_text, image_b64, page, settings)
    if settings.mode == "debug":
        persist_validated_page(entries, page, settings)
    logger.info("page %d validate ok (%d entries)", page, len(entries))
    return entries


@log_errors
def _process_page(
    page: int,
    settings: Settings,
    persistence: PersistenceCoordinator,
) -> list[DictionaryEntry]:
    """Run one complete E→T→V→persist flow exactly once.

    Every exception marks the page failed before it escapes to the executor.
    A validation failure in running mode also preserves the raw response for
    diagnosis. Successful pages are committed immediately by their worker.
    """
    logger.info("page %d start (mode=%s)", page, settings.mode)
    raw_json: str | None = None
    try:
        image_b64, ocr_text = _extract_page(page, settings)
        raw_json = _transform_page(image_b64, ocr_text, page, settings)
        entries = _validate_page(raw_json, ocr_text, image_b64, page, settings)
        persistence.insert_page(page, entries)
    except Exception:
        if raw_json is not None and settings.mode != "debug":
            try:
                _write_raw(settings.raw_page_path(page, settings.model), raw_json)
            except Exception:
                # The artifact error has already been logged. Failure-ledger
                # persistence is still attempted before the page error escapes.
                pass
        try:
            persistence.add_failure(page)
        except Exception:
            # Preserve the original provider/schema/infrastructure error; the
            # coordinator has already logged the independent ledger failure.
            pass
        raise

    logger.info("page %d complete (%d entries)", page, len(entries))
    return entries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and persist selected pages of a VS volume.",
    )
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--start", type=int, help="first printed page")
    selectors.add_argument(
        "--pages",
        type=int,
        nargs="+",
        help="explicit printed pages, for example: --pages 3 7 9",
    )
    selectors.add_argument(
        "--pages-file",
        type=Path,
        help="file containing one printed page number per line",
    )
    parser.add_argument("--end", type=int, help="last printed page (with --start)")
    parser.add_argument(
        "--mode",
        choices=["debug", "running"],
        help="override VS_MODE; debug also writes raw and per-page artifacts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="maximum concurrent page workers (default: 1)",
    )
    return parser


def _resolve_pages(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[int]:
    """Resolve any supported selector to a sorted, unique page list."""
    if args.start is not None:
        if args.end is None:
            parser.error("--start requires --end")
        if args.end < args.start:
            parser.error("--end must be >= --start")
        pages = list(range(args.start, args.end + 1))
    else:
        if args.end is not None:
            parser.error("--end can only be used with --start")
        if args.pages is not None:
            pages = args.pages
        else:
            try:
                lines = args.pages_file.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                parser.error(f"cannot read --pages-file {args.pages_file}: {exc}")
            pages = []
            for line_number, raw_line in enumerate(lines, start=1):
                value = raw_line.strip()
                if not value:
                    continue
                try:
                    pages.append(int(value))
                except ValueError:
                    parser.error(
                        f"invalid page number on line {line_number} of "
                        f"{args.pages_file}: {value!r}"
                    )

    if not pages:
        parser.error("at least one page must be selected")
    if any(page < 1 for page in pages):
        parser.error("page numbers must be >= 1")
    return sorted(set(pages))


def _run_pages(
    pages: list[int],
    batch_size: int,
    settings: Settings,
    persistence: PersistenceCoordinator,
) -> tuple[set[int], dict[int, Exception]]:
    """Keep all workers busy and collect every result without cancellation."""
    succeeded: set[int] = set()
    failed: dict[int, Exception] = {}
    if not pages:
        return succeeded, failed

    worker_count = min(batch_size, len(pages))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_process_page, page, settings, persistence): page
            for page in pages
        }
        for future in as_completed(futures):
            page = futures[future]
            try:
                entries = future.result()
            except Exception as exc:
                failed[page] = exc
                print(f"[page {page}] failed: {exc}")
                logger.info(
                    "page %d worker finished unsuccessfully; other workers continue",
                    page,
                )
                continue
            succeeded.add(page)
            print(f"[page {page}] persisted ({len(entries)} entries)")
    return succeeded, failed


def main() -> None:
    """Run all requested, incomplete pages and exit nonzero on any failure."""
    parser = _build_parser()
    args = parser.parse_args()
    pages = _resolve_pages(args, parser)
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    batch_size = args.batch_size or 1

    settings = get_settings()
    if args.mode:
        settings = settings.model_copy(update={"mode": args.mode})
    configure_logging(
        settings.log_file,
        src_level=logging.DEBUG if settings.mode == "debug" else logging.INFO,
    )
    logger.info(
        "pipeline start: volume=%s pages=%s model=%s mode=%s batch_size=%d",
        settings.volume,
        ",".join(str(page) for page in pages),
        settings.model,
        settings.mode,
        batch_size,
    )

    try:
        persistence = PersistenceCoordinator(settings)
        pending: list[int] = []
        for page in pages:
            if persistence.page_exists(page):
                print(f"[page {page}] skipped (already persisted)")
                logger.info("page %d skipped because it is already complete", page)
            else:
                pending.append(page)

        succeeded, failed = _run_pages(
            pending, batch_size, settings, persistence
        )
        logger.info(
            "pipeline end: requested=%d skipped=%d succeeded=%d failed=%d output=%s",
            len(pages),
            len(pages) - len(pending),
            len(succeeded),
            len(failed),
            persistence.dictionary_path,
        )
        if failed:
            raise SystemExit(1)
    finally:
        if settings.track_cost:
            log_summary()


if __name__ == "__main__":
    main()
