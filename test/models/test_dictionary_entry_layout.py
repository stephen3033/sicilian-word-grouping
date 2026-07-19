"""Unit tests for the headword-null `validate_layout_alignment` model_validator.

Synthesizes small PNGs with PIL so the validator is exercised
deterministically without the real corpus. The underlying layout helpers
are tested in test/common/test_layout.py.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError as PydanticValidationError

from src.common.errors import ValidationError
from src.models import DictionaryEntry

_MIN_OCR = "a¹ f. e (antiq.) m. vocale e prima lettera dell'alfabeto."


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _page_with_lines(line_specs: list[tuple[int, int, int, int]]) -> bytes:
    """Build a white PNG with black-rectangle 'text lines' ((left, top, width, height))."""
    img = Image.new("RGB", (1000, 800), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for left, top, width, height in line_specs:
        draw.rectangle(
            [left, top, left + width - 1, top + height - 1], fill=(0, 0, 0)
        )
    return _png_bytes(img)


# Two lines at different X -> headword layout (|Δx|>threshold -> headword present expected).
_HEADWORD_PNG = _page_with_lines([(100, 50, 600, 40), (160, 120, 600, 40)])
# Two lines 5px apart in X -> orphan layout (|Δx|<=threshold -> headword=None expected).
_ORPHAN_PNG = _page_with_lines([(100, 50, 600, 40), (105, 120, 600, 40)])

_ENTRY = {
    "headword": "a¹",
    "trailing_text": "vocale",
    "variants": None,
    "page_numbers": [1],
    "vs_vol": 0,
}


def _ctx(
    png: bytes,
    index: int = 0,
    delta: float = 36.0,
    tolerance: float = 15.0,
) -> dict:
    return {
        "normalized_ocr": _MIN_OCR,
        "index": index,
        "image_payload": png,
        "headword_delta": delta,
        "tolerance": tolerance,
    }


class TestValidateLayoutAlignment:
    def test_indented_layout_headword_present_passes(self):
        entry = {**_ENTRY}
        out = DictionaryEntry.model_validate(entry, context=_ctx(_HEADWORD_PNG))
        assert out.headword == "a¹"

    def test_indented_layout_headword_null_raises_mismatch(self):
        entry = {**_ENTRY, "headword": None, "trailing_text": "vocale"}
        with pytest.raises(PydanticValidationError, match=r"AI extraction mismatch"):
            DictionaryEntry.model_validate(entry, context=_ctx(_HEADWORD_PNG))

    def test_aligned_layout_headword_null_passes(self):
        entry = {**_ENTRY, "headword": None, "trailing_text": "vocale"}
        out = DictionaryEntry.model_validate(entry, context=_ctx(_ORPHAN_PNG))
        assert out.headword is None

    def test_aligned_layout_headword_present_raises_mismatch(self):
        entry = {**_ENTRY}
        with pytest.raises(PydanticValidationError, match=r"AI extraction mismatch"):
            DictionaryEntry.model_validate(entry, context=_ctx(_ORPHAN_PNG))

    def test_index_not_zero_no_ops(self):
        # Second entry on a page: validator skips regardless of layout.
        entry = {**_ENTRY}
        out = DictionaryEntry.model_validate(entry, context=_ctx(_HEADWORD_PNG, index=1))
        assert out.headword == "a¹"

    def test_missing_image_payload_no_ops(self):
        entry = {**_ENTRY}
        out = DictionaryEntry.model_validate(
            entry, context={"normalized_ocr": _MIN_OCR, "index": 0}
        )
        assert out.headword == "a¹"

    def test_no_context_raises_grounding_error(self):
        # Layout validator itself skips (no image_payload); grounding
        # validators raise on missing normalized_ocr.
        entry = {**_ENTRY}
        with pytest.raises(ValidationError, match=r"normalized_ocr context required"):
            DictionaryEntry.model_validate(entry)

    def test_mismatch_message_includes_threshold_and_values(self):
        entry = {**_ENTRY, "headword": None, "trailing_text": "vocale"}
        with pytest.raises(
            PydanticValidationError,
            match=(
                r"headword=None.*expected.*headword present.*"
                r"threshold=21\.0.*headword_delta=36\.0.*tolerance=15\.0"
            ),
        ):
            DictionaryEntry.model_validate(entry, context=_ctx(_HEADWORD_PNG))
