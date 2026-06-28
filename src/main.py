"""ETVL pipeline orchestrator entry point."""

from __future__ import annotations

import argparse

from src.config import get_settings
from src.extract import extract_page_image, extract_page_text
from src.transform import SYSTEM_PROMPT, build_user_prompt, extract_json


def _extract_page(page: int, settings) -> None:
    """Run extract -> transform for one page and write the raw output to disk."""
    image_b64 = extract_page_image(page)
    ocr_text = extract_page_text(page)
    raw_json = extract_json(image_b64, SYSTEM_PROMPT, build_user_prompt(ocr_text))

    out_path = settings.output_dir / f"VS{settings.volume}_page_{page}_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(raw_json, encoding="utf-8")


def main() -> None:
    """Iterate over a page range, run E->T per page, persist raw model output."""
    parser = argparse.ArgumentParser(
        description="Extract & LLM-transform a page range of a VS volume.",
    )
    parser.add_argument("--start", type=int, required=True, help="first printed page")
    parser.add_argument("--end", type=int, required=True, help="last printed page")
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be >= --start")

    settings = get_settings()
    for page in range(args.start, args.end + 1):
        try:
            _extract_page(page, settings)
            print(f"[page {page}] wrote {settings.output_dir}/VS{settings.volume}_page_{page}_output.json")
        except KeyError:
            raise
        except Exception as e:
            print(f"[page {page}] failed: {e}")


if __name__ == "__main__":
    main()
