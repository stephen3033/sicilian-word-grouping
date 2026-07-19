"""ETVL pipeline orchestrator entry point."""

from __future__ import annotations

import argparse
import logging

from src.config import Settings, get_settings
from src.extract import extract_page_image, extract_page_text
from src.load import stitch
from src.common.logger import configure_logging, log_errors
from src.models import DictionaryEntry
from src.transform import SYSTEM_PROMPT, build_user_prompt, extract_json
from src.validate import persist_validated_page, validate

logger = logging.getLogger(__name__)


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
    image_b64: str, ocr_text: str, page: int, settings: Settings
) -> str:
    """Transform layer: compile prompt + call VLM, return raw JSON string."""
    raw_json = extract_json(
        image_b64,
        SYSTEM_PROMPT,
        build_user_prompt(ocr_text),
    )
    logger.info("ok (raw_json=%d chars)", len(raw_json))

    if settings.mode == "debug":
        out_path = settings.raw_page_path(page, settings.model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(raw_json, encoding="utf-8")
        logger.debug("wrote raw_json %s", out_path)
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
    entries = validate(raw_json, ocr_text, image_b64, page)
    if settings.mode == "debug":
        persist_validated_page(entries, page, settings)
    logger.info("ok (%d entries)", len(entries))
    return entries


@log_errors
def _process_page(
    page: int, settings: Settings
) -> list[DictionaryEntry]:
    """Run E -> T -> V for one printed page; persist per-page artifacts in debug."""
    logger.info("page %d: start (mode=%s)", page, settings.mode)
    image_b64, ocr_text = _extract_page(page, settings)
    raw_json = _transform_page(image_b64, ocr_text, page, settings)
    entries = _validate_page(raw_json, ocr_text, image_b64, page, settings)
    logger.info("page %d: complete (%d entries)", page, len(entries))
    return entries


def main() -> None:
    """Iterate over a page range, run E->T->V per page, then load-stitch."""
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
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be >= --start")

    settings = get_settings()
    if args.mode:
        settings = settings.model_copy(update={"mode": args.mode})

    configure_logging(
        settings.log_file,
        src_level=logging.DEBUG if settings.mode == "debug" else logging.INFO,
    )
    logger.info(
        "pipeline start: volume=%s pages=%d-%d model=%s mode=%s",
        settings.volume,
        args.start,
        args.end,
        settings.model,
        settings.mode,
    )

    entries_by_page: dict[int, list[DictionaryEntry]] = {}
    for page in range(args.start, args.end + 1):
        try:
            entries_by_page[page] = _process_page(page, settings)
            print(f"[page {page}] validated ({len(entries_by_page[page])} entries)")
        except Exception as e:
            print(f"[page {page}] failed: {e}")

    stitched_path = stitch(entries_by_page, settings)
    print(f"[load] stitched {len(entries_by_page)} pages -> {stitched_path}")
    logger.info(
        "pipeline end: pages=%d-%d stitched=%s entries=%d",
        args.start,
        args.end,
        stitched_path,
        sum(len(v) for v in entries_by_page.values()),
    )


if __name__ == "__main__":
    main()