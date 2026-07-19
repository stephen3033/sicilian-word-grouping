"""Unit tests for the layout-verification helpers and the
`is_orphan_fragment` @field_validator on `DictionaryEntry`.

The VS PDFs are rasterized scans (no text layer), so layout extraction is
purely pixel-based. These tests synthesize small PNG images with PIL to
exercise the helpers and the validator deterministically without touching
the real corpus.
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


def _draw_line(
    img: Image.Image, left: int, top: int, width: int, height: int
) -> None:
    """Draw a solid black rectangle simulating a text-line ink band."""
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [left, top, left + width - 1, top + height - 1], fill=(0, 0, 0)
    )


def _page_with_lines(line_specs: list[tuple[int, int, int, int]]) -> bytes:
    """Build a white PNG with black-rectangle 'text lines'.

    line_specs: list of (left, top, width, height) tuples in pixels.
    """
    img = Image.new("RGB", (1000, 800), (255, 255, 255))
    for left, top, width, height in line_specs:
        _draw_line(img, left, top, width, height)
    return _png_bytes(img)


# ---------------------------------------------------------------------------
# _convert_image_to_layout_data
# ---------------------------------------------------------------------------


class TestConvertImageToLayoutData:
    def test_three_distinct_lines_detected(self):
        # Three text-line bands at distinct Y positions, distinct X.
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (160, 120, 600, 40),
                (100, 190, 600, 40),
            ]
        )
        data = _convert_image_to_layout_data(png)
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
        # 5px-tall mark at top should be dropped; real line below kept.
        png = _page_with_lines(
            [
                (100, 50, 100, 5),
                (100, 120, 600, 40),
            ]
        )
        data = _convert_image_to_layout_data(png)
        assert len(data["lines"]) == 1
        assert data["lines"][0]["top"] == 120

    def test_gap_over_tolerance_splits_lines(self):
        # Two bands separated by 30 white rows -> two separate lines.
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (100, 120, 600, 40),
            ]
        )
        data = _convert_image_to_layout_data(png)
        assert len(data["lines"]) == 2

    def test_small_gap_keeps_lines_separate(self):
        # 6px gap (still > gap_tol=4) splits; 3px gap would merge.
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (100, 96, 600, 40),  # 6px gap
            ]
        )
        data = _convert_image_to_layout_data(png)
        assert len(data["lines"]) == 2

    def test_blank_image_returns_no_lines(self):
        img = Image.new("RGB", (500, 500), (255, 255, 255))
        data = _convert_image_to_layout_data(_png_bytes(img))
        assert data["lines"] == []
        assert data["page_width"] == 500


# ---------------------------------------------------------------------------
# _analyze_first_span
# ---------------------------------------------------------------------------


class TestAnalyzeFirstSpan:
    def test_large_x_difference_is_indented(self):
        # |Δx|=60 > threshold (36-15=21) -> indented.
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (160, 120, 600, 40),
            ]
        )
        data = _convert_image_to_layout_data(png)
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is True

    def test_small_x_difference_not_indented(self):
        # |Δx|=5 <= threshold (36-15=21) -> not indented.
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (105, 120, 600, 40),
            ]
        )
        data = _convert_image_to_layout_data(png)
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is False

    def test_zero_x_difference_not_indented(self):
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (100, 120, 600, 40),
            ]
        )
        data = _convert_image_to_layout_data(png)
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is False

    def test_narrow_line_below_30px_width_ignored(self):
        # Line1 (width=600) at x=100; line2 is narrow (width=20) at x=500,
        # line3 (width=600) at x=160. The narrow line is skipped; the
        # comparison uses line1 and line3 -> |Δx|=60 > threshold -> indented.
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (500, 120, 20, 40),
                (160, 190, 600, 40),
            ]
        )
        data = _convert_image_to_layout_data(png)
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is True

    def test_fewer_than_two_lines_defaults_to_not_indented(self):
        png = _page_with_lines([(100, 50, 600, 40)])
        data = _convert_image_to_layout_data(png)
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is False

    def test_threshold_boundary_exclusive(self):
        # |Δx| exactly equal to the threshold (delta-tolerance=21) is NOT
        # indented (strict >). |Δx|=22 (just above) IS indented.
        png_at_threshold = _page_with_lines(
            [(100, 50, 600, 40), (121, 120, 600, 40)]  # |Δx|=21 == threshold
        )
        data = _convert_image_to_layout_data(png_at_threshold)
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is False

        png_above_threshold = _page_with_lines(
            [(100, 50, 600, 40), (122, 120, 600, 40)]  # |Δx|=22 > threshold
        )
        data = _convert_image_to_layout_data(png_above_threshold)
        _, is_indented = _analyze_first_span(data, delta=36.0, tolerance=15.0)
        assert is_indented is True


# ---------------------------------------------------------------------------
# validate_layout_alignment field_validator on DictionaryEntry
# ---------------------------------------------------------------------------


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
        # Headword page: |Δx|=60 > tol -> expected is_orphan=False.
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (160, 120, 600, 40),
            ]
        )
        entry = {**_ENTRY, "is_orphan_fragment": False}
        out = DictionaryEntry.model_validate(entry, context=_ctx(png))
        assert out.is_orphan_fragment is False

    def test_indented_layout_ai_true_raises_mismatch(self):
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (160, 120, 600, 40),
            ]
        )
        entry = {**_ENTRY, "is_orphan_fragment": True}
        with pytest.raises(PydanticValidationError, match=r"AI extraction mismatch"):
            DictionaryEntry.model_validate(entry, context=_ctx(png))

    def test_aligned_layout_ai_true_passes(self):
        # Orphan page: |Δx|=5 <= tol -> expected is_orphan=True.
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (105, 120, 600, 40),
            ]
        )
        entry = {**_ENTRY, "is_orphan_fragment": True}
        out = DictionaryEntry.model_validate(entry, context=_ctx(png))
        assert out.is_orphan_fragment is True

    def test_aligned_layout_ai_false_raises_mismatch(self):
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (105, 120, 600, 40),
            ]
        )
        entry = {**_ENTRY, "is_orphan_fragment": False}
        with pytest.raises(PydanticValidationError, match=r"AI extraction mismatch"):
            DictionaryEntry.model_validate(entry, context=_ctx(png))

    def test_index_not_zero_no_ops(self):
        # Second entry on a page: validator skips regardless of layout.
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (160, 120, 600, 40),
            ]
        )
        # AI says orphan=False but layout says headword; since index=1, no
        # error and the AI value passes through.
        entry = {**_ENTRY, "is_orphan_fragment": False}
        out = DictionaryEntry.model_validate(entry, context=_ctx(png, index=1))
        assert out.is_orphan_fragment is False

    def test_missing_image_payload_no_ops(self):
        # No image bytes in context -> validator skips; only grounding runs.
        entry = {**_ENTRY, "is_orphan_fragment": False}
        out = DictionaryEntry.model_validate(
            entry, context={"normalized_ocr": _MIN_OCR, "index": 0}
        )
        assert out.is_orphan_fragment is False

    def test_no_context_no_ops(self):
        # Bare model_validate without any context: the layout validator
        # skips (no image_payload), but the grounding validators raise the
        # custom ValidationError on missing normalized_ocr. Confirm the
        # layout validator itself does not raise a layout-specific error.
        entry = {**_ENTRY, "is_orphan_fragment": False}
        with pytest.raises(ValidationError, match=r"normalized_ocr context required"):
            DictionaryEntry.model_validate(entry)

    def test_mismatch_message_includes_threshold_and_values(self):
        png = _page_with_lines(
            [
                (100, 50, 600, 40),
                (160, 120, 600, 40),
            ]
        )
        entry = {**_ENTRY, "is_orphan_fragment": True}
        with pytest.raises(
            PydanticValidationError,
            match=(
                r"is_orphan_fragment=True.*expected=False.*"
                r"threshold=21\.0.*headword_delta=36\.0.*tolerance=15\.0"
            ),
        ):
            DictionaryEntry.model_validate(entry, context=_ctx(png))
