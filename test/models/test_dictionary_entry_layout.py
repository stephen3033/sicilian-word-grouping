"""Unit tests for the layout helpers + `is_orphan_fragment` @field_validator.

Synthesizes small PNGs with PIL so the helpers and validator are exercised
deterministically without the real corpus.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError as PydanticValidationError

from src.common.errors import ValidationError
from src.models import DictionaryEntry
from src.models.dictionary_entry import (
    _analyze_first_span,
    _convert_image_to_layout_data,
)

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


# Two lines at different X -> headword layout (|Δx|>threshold -> is_orphan=False).
_HEADWORD_PNG = _page_with_lines([(100, 50, 600, 40), (160, 120, 600, 40)])
# Two lines 5px apart in X -> orphan layout (|Δx|<=threshold -> is_orphan=True).
_ORPHAN_PNG = _page_with_lines([(100, 50, 600, 40), (105, 120, 600, 40)])


class TestConvertImageToLayoutData:
    def test_three_distinct_lines_detected(self):
        data = _convert_image_to_layout_data(
            _page_with_lines(
                [(100, 50, 600, 40), (160, 120, 600, 40), (100, 190, 600, 40)]
            )
        )
        lines = data["lines"]
        assert len(lines) == 3
        assert lines[0]["top"] == 50
        assert lines[0]["left_x"] == 100
        assert lines[0]["width"] == 600
        assert lines[0]["height"] >= 10
        assert lines[1]["top"] == 120
        assert lines[1]["left_x"] == 160
        assert lines[2]["top"] == 190

    def test_stray_marks_below_min_height_dropped(self):
        data = _convert_image_to_layout_data(
            _page_with_lines([(100, 50, 100, 5), (100, 120, 600, 40)])
        )
        assert len(data["lines"]) == 1
        assert data["lines"][0]["top"] == 120

    def test_gap_over_tolerance_splits_lines(self):
        data = _convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (100, 120, 600, 40)])
        )
        assert len(data["lines"]) == 2

    def test_small_gap_keeps_lines_separate(self):
        # 6px gap (> gap_tol=4) splits; a 3px gap would merge.
        data = _convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (100, 96, 600, 40)])
        )
        assert len(data["lines"]) == 2

    def test_blank_image_returns_no_lines(self):
        img = Image.new("RGB", (500, 500), (255, 255, 255))
        data = _convert_image_to_layout_data(_png_bytes(img))
        assert data["lines"] == []
        assert data["page_width"] == 500


class TestAnalyzeFirstSpan:
    def test_large_x_difference_is_indented(self):
        data = _convert_image_to_layout_data(_HEADWORD_PNG)
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is True

    def test_small_x_difference_not_indented(self):
        data = _convert_image_to_layout_data(_ORPHAN_PNG)
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is False

    def test_zero_x_difference_not_indented(self):
        data = _convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (100, 120, 600, 40)])
        )
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is False

    def test_narrow_line_below_30px_width_ignored(self):
        # line2 (width=20) is skipped; comparison uses line1 & line3 -> |Δx|=60.
        data = _convert_image_to_layout_data(
            _page_with_lines(
                [(100, 50, 600, 40), (500, 120, 20, 40), (160, 190, 600, 40)]
            )
        )
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is True

    def test_fewer_than_two_lines_defaults_to_not_indented(self):
        data = _convert_image_to_layout_data(_page_with_lines([(100, 50, 600, 40)]))
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is False

    def test_threshold_boundary_exclusive(self):
        # |Δx|==threshold (21) is NOT indented (strict >); |Δx|=22 IS indented.
        at = _convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (121, 120, 600, 40)])
        )
        _, is_indented = _analyze_first_span(at, delta=36.0, tolerance=15.0)
        assert is_indented is False

        above = _convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (122, 120, 600, 40)])
        )
        _, is_indented = _analyze_first_span(above, delta=36.0, tolerance=15.0)
        assert is_indented is True


_ENTRY = {
    "headword": "a¹",
    "trailing_text": "vocale",
    "variants": None,
    "page_numbers": [1],
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
    def test_indented_layout_ai_false_passes(self):
        entry = {**_ENTRY, "is_orphan_fragment": False}
        out = DictionaryEntry.model_validate(entry, context=_ctx(_HEADWORD_PNG))
        assert out.is_orphan_fragment is False

    def test_indented_layout_ai_true_raises_mismatch(self):
        entry = {**_ENTRY, "is_orphan_fragment": True}
        with pytest.raises(PydanticValidationError, match=r"AI extraction mismatch"):
            DictionaryEntry.model_validate(entry, context=_ctx(_HEADWORD_PNG))

    def test_aligned_layout_ai_true_passes(self):
        entry = {**_ENTRY, "is_orphan_fragment": True}
        out = DictionaryEntry.model_validate(entry, context=_ctx(_ORPHAN_PNG))
        assert out.is_orphan_fragment is True

    def test_aligned_layout_ai_false_raises_mismatch(self):
        entry = {**_ENTRY, "is_orphan_fragment": False}
        with pytest.raises(PydanticValidationError, match=r"AI extraction mismatch"):
            DictionaryEntry.model_validate(entry, context=_ctx(_ORPHAN_PNG))

    def test_index_not_zero_no_ops(self):
        # Second entry on a page: validator skips regardless of layout.
        entry = {**_ENTRY, "is_orphan_fragment": False}
        out = DictionaryEntry.model_validate(entry, context=_ctx(_HEADWORD_PNG, index=1))
        assert out.is_orphan_fragment is False

    def test_missing_image_payload_no_ops(self):
        entry = {**_ENTRY, "is_orphan_fragment": False}
        out = DictionaryEntry.model_validate(
            entry, context={"normalized_ocr": _MIN_OCR, "index": 0}
        )
        assert out.is_orphan_fragment is False

    def test_no_context_raises_grounding_error(self):
        # Layout validator itself skips (no image_payload); grounding
        # validators raise on missing normalized_ocr.
        entry = {**_ENTRY, "is_orphan_fragment": False}
        with pytest.raises(ValidationError, match=r"normalized_ocr context required"):
            DictionaryEntry.model_validate(entry)

    def test_mismatch_message_includes_threshold_and_values(self):
        entry = {**_ENTRY, "is_orphan_fragment": True}
        with pytest.raises(
            PydanticValidationError,
            match=(
                r"is_orphan_fragment=True.*expected=False.*"
                r"threshold=21\.0.*headword_delta=36\.0.*tolerance=15\.0"
            ),
        ):
            DictionaryEntry.model_validate(entry, context=_ctx(_HEADWORD_PNG))
