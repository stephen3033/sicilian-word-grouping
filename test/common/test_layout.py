"""Unit tests for src.common.layout (pixel-based text-line heuristics).

Synthesizes small PNGs with PIL so the helpers are exercised
deterministically without the real corpus.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from src.common.layout import analyze_first_span, convert_image_to_layout_data


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


class TestConvertImageToLayoutData:
    def test_three_distinct_lines_detected(self):
        data = convert_image_to_layout_data(
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
        data = convert_image_to_layout_data(
            _page_with_lines([(100, 50, 100, 5), (100, 120, 600, 40)])
        )
        assert len(data["lines"]) == 1
        assert data["lines"][0]["top"] == 120

    def test_gap_over_tolerance_splits_lines(self):
        data = convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (100, 120, 600, 40)])
        )
        assert len(data["lines"]) == 2

    def test_small_gap_keeps_lines_separate(self):
        # 6px gap (> gap tolerance of 4) splits; a 3px gap would merge.
        data = convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (100, 96, 600, 40)])
        )
        assert len(data["lines"]) == 2

    def test_blank_image_returns_no_lines(self):
        img = Image.new("RGB", (500, 500), (255, 255, 255))
        data = convert_image_to_layout_data(_png_bytes(img))
        assert data["lines"] == []
        assert data["page_width"] == 500


class TestAnalyzeFirstSpan:
    def test_large_x_difference_is_indented(self):
        data = convert_image_to_layout_data(_HEADWORD_PNG)
        assert analyze_first_span(data, delta=36.0, tolerance=15.0) is True

    def test_small_x_difference_not_indented(self):
        data = convert_image_to_layout_data(_ORPHAN_PNG)
        assert analyze_first_span(data, delta=36.0, tolerance=15.0) is False

    def test_zero_x_difference_not_indented(self):
        data = convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (100, 120, 600, 40)])
        )
        assert analyze_first_span(data, delta=36.0, tolerance=15.0) is False

    def test_narrow_line_below_30px_width_ignored(self):
        # line2 (width=20) is skipped; comparison uses line1 & line3 -> |Δx|=60.
        data = convert_image_to_layout_data(
            _page_with_lines(
                [(100, 50, 600, 40), (500, 120, 20, 40), (160, 190, 600, 40)]
            )
        )
        assert analyze_first_span(data, delta=36.0, tolerance=15.0) is True

    def test_fewer_than_two_lines_defaults_to_not_indented(self):
        data = convert_image_to_layout_data(_page_with_lines([(100, 50, 600, 40)]))
        assert analyze_first_span(data, delta=36.0, tolerance=15.0) is False

    def test_threshold_boundary_exclusive(self):
        # |Δx|==threshold (21) is NOT indented (strict >); |Δx|=22 IS indented.
        at = convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (121, 120, 600, 40)])
        )
        assert analyze_first_span(at, delta=36.0, tolerance=15.0) is False

        above = convert_image_to_layout_data(
            _page_with_lines([(100, 50, 600, 40), (122, 120, 600, 40)])
        )
        assert analyze_first_span(above, delta=36.0, tolerance=15.0) is True
