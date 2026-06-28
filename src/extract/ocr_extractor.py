from __future__ import annotations

import re
from functools import lru_cache

from src.config import Settings, get_settings

_LINE_RE = re.compile(r"^(\d+)\s+(.*)$")


@lru_cache(maxsize=8)
def _build_index(ocr_path: str, _mtime: float) -> dict[int, list[str]]:
    """Build {printed_page: [raw_lines...]} by scanning the OCR txt once."""
    index: dict[int, list[str]] = {}
    with open(ocr_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = _LINE_RE.match(line)
            if not m:
                continue
            index.setdefault(int(m.group(1)), []).append(line)
    return index


def extract_page_text(page_number: int, settings: Settings | None = None) -> str:
    """Return the OCR text block for printed page `page_number`.

    Lines in the txt are prefixed `<n> <text>`. By default the prefix is
    stripped (see `Settings.strip_ocr_prefix`). Returns "" if the page has no
    lines.
    """

    s = settings or get_settings()
    ocr_path = s.ocr_txt_path()
    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR txt not found: {ocr_path}")

    # mtime in the key invalidates the cache when the volume or file changes.
    index = _build_index(str(ocr_path), ocr_path.stat().st_mtime)
    lines = index.get(page_number, [])
    if not lines:
        return ""

    if not s.strip_ocr_prefix:
        return "\n".join(lines)
    return "\n".join(_LINE_RE.match(raw).group(2) for raw in lines)