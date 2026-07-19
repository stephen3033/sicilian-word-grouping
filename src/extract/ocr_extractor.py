from __future__ import annotations

import logging
from functools import lru_cache

from src.config import get_settings
from src.common.logger import log_errors

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _build_index(ocr_path: str, _mtime: float) -> dict[int, list[str]]:
    """Build {printed_page: [raw_lines...]} by scanning the OCR txt once."""
    index: dict[int, list[str]] = {}
    with open(ocr_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            page_str = line.partition(" ")[0]
            if not page_str.isdigit():
                raise KeyError("Bad page number!!")
            index.setdefault(int(page_str), []).append(line)
    logger.debug("indexed %s -> %d pages", ocr_path, len(index))
    return index


@log_errors
def extract_page_text(page_number: int) -> str:
    """Return the OCR text block for printed page `page_number`.

    Lines in the txt are prefixed `<n> <text>`. By default the prefix is
    stripped (see `Settings.strip_ocr_prefix`). Returns "" if the page has no
    lines.
    """

    s = get_settings()
    ocr_path = s.ocr_txt_path()
    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR txt not found: {ocr_path}")

    # mtime in the key invalidates the cache when the volume or file changes.
    index = _build_index(str(ocr_path), ocr_path.stat().st_mtime)
    lines = index.get(page_number, [])
    if not lines:
        raise KeyError(f"No OCR text for page {page_number}")

    if not s.strip_ocr_prefix:
        body = "\n".join(lines)
    else:
        body = "\n".join(line.partition(" ")[2] for line in lines)
    logger.debug(
        "page %d lines=%d body=%d chars strip_prefix=%s",
        page_number,
        len(lines),
        len(body),
        s.strip_ocr_prefix,
    )
    return body