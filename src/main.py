"""ETVL pipeline orchestrator entry point."""

from __future__ import annotations

import argparse
import logging

from src.config import Settings, get_settings
from src.extract import extract_page_image, extract_page_text
from src.common.logger import configure_logging, log_errors
from src.models import DictionaryEntry
from src.transform import SYSTEM_PROMPT, build_user_prompt, extract_json
from src.validate import validate

logger = logging.getLogger(__name__)


@log_errors
def _extract_page(page: int, settings: Settings) -> tuple[str, str]:
    """Extract layer: render page image + read OCR text for one printed page."""
    image_b64 = extract_page_image(page)
    ocr_text = extract_page_text(page)
    logger.info(
        "extract: page %d ok (image=%d b64, ocr=%d chars)",
        page,
        len(image_b64),
        len(ocr_text),
    )
    return image_b64, ocr_text


@log_errors
def _transform_page(image_b64: str, ocr_text: str, settings: Settings) -> str:
    """Transform layer: compile prompt + call VLM, return raw JSON string."""
    raw_json = extract_json(
        image_b64,
        SYSTEM_PROMPT,
        build_user_prompt(ocr_text),
    )
    logger.info("transform: ok (raw_json=%d chars)", len(raw_json))
    return raw_json


@log_errors
def _validate_page(
    raw_json: str, ocr_text: str, image_b64: str, settings: Settings
) -> list[DictionaryEntry]:
    """Validate layer: schema + grounding + layout checks; raises on failure."""
    entries = validate(raw_json, ocr_text, image_b64)
    logger.info("validate: ok (%d entries)", len(entries))
    return entries


@log_errors
def _process_page(page: int, settings: Settings) -> None:
    """Run E -> T -> (persist raw JSON) -> V for one printed page."""
    logger.info("page %d: start", page)
    image_b64, ocr_text = _extract_page(page, settings)
    raw_json = _transform_page(image_b64, ocr_text, settings)

    out_path = settings.output_dir / f"VS{settings.volume}_page_{page}_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(raw_json, encoding="utf-8")
    logger.debug("process_page: wrote %s", out_path)

    _validate_page(raw_json, ocr_text, image_b64, settings)
    logger.info("page %d: complete", page)


def main() -> None:
    """Iterate over a page range, run E->T->V per page, persist raw model output."""
    parser = argparse.ArgumentParser(
        description="Extract & LLM-transform a page range of a VS volume.",
    )
    parser.add_argument("--start", type=int, required=True, help="first printed page")
    parser.add_argument("--end", type=int, required=True, help="last printed page")
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be >= --start")

    settings = get_settings()
    configure_logging(settings.log_file)
    logger.info(
        "pipeline start: volume=%s pages=%d-%d model=%s",
        settings.volume,
        args.start,
        args.end,
        settings.model,
    )

    for page in range(args.start, args.end + 1):
        try:
            _process_page(page, settings)
            print(f"[page {page}] wrote {settings.output_dir}/VS{settings.volume}_page_{page}_output.json")
        except KeyError:
            raise
        except Exception as e:
            print(f"[page {page}] failed: {e}")

    logger.info("pipeline end: pages=%d-%d", args.start, args.end)


if __name__ == "__main__":
    main()
