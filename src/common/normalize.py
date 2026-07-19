"""OCR text normalization for verbatim substring grounding."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """NFC + collapse whitespace. No lowercasing or character alteration."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text)
