"""Text normalization helpers shared across pipeline stages.

The `normalization` helper is the single canonical way to prepare OCR text
and extracted fields for verbatim substring (`in`) grounding checks. Both
sides of any grounding comparison must pass through it so that layout
differences (line breaks, multiple spaces) and Unicode representation
differences (NFC vs NFD) do not defeat the check.
"""

from __future__ import annotations

import re
import unicodedata


def normalization(text: str) -> str:
    """Normalize OCR text for verbatim substring grounding.

    - Unicode Canonical Composition (NFC) to align character byte
      representations.
    - Flatten all layout line breaks (``\\n``, ``\\r``) and runs of
      consecutive whitespace into a single ASCII space so layout
      differences don't defeat ``in`` checks.

    Intentionally minimalist: no lowercasing, no punctuation stripping, no
    character alteration.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    # TODO: add targeted regex replacements here if real-world OCR data
    # requires further normalization (e.g. ligature folding, soft-hyphen
    # removal). Keep rules surgical and document each one.
    return text