"""Pixel-based page-layout heuristics for the headword-null check.

VS PDFs are rasterized scans with no text layer, so the first-entry
orphan check works on the rendered page image: find text-line bounding
boxes, then compare the left X of the first two real lines.
"""

from __future__ import annotations

import io

from PIL import Image, ImageChops

# Calibrated for VS scans rendered at 200 DPI.
_INK_THRESHOLD = 128  # grayscale value below which a pixel counts as ink
_COLUMN_STRIDE = 4  # sample every Nth column when scanning rows for ink
_GAP_TOLERANCE_PX = 4  # white rows tolerated inside one text line
_MIN_LINE_HEIGHT_PX = 10  # ink bands shorter than this are noise
_MIN_TEXT_LINE_WIDTH_PX = 30  # lines narrower than this are marginalia


def convert_image_to_layout_data(image_payload: bytes) -> dict:
    """Decode PNG bytes and extract text-line bboxes from the rendered page.

    Returns ``{"lines": [...], "page_width": w}`` in pixel coordinates.
    Lines are groups of vertically-contiguous inked rows (gaps up to
    ``_GAP_TOLERANCE_PX`` tolerated); bands shorter than
    ``_MIN_LINE_HEIGHT_PX`` are dropped.
    """
    img = Image.open(io.BytesIO(image_payload)).convert("L")
    bw = img.point(lambda p: 0 if p < _INK_THRESHOLD else 255)
    inv = ImageChops.invert(bw)
    width, height = bw.size
    px = bw.load()
    row_ink = [
        any(px[x, y] == 0 for x in range(0, width, _COLUMN_STRIDE))
        for y in range(height)
    ]

    lines: list[tuple[int, int]] = []
    y = 0
    while y < height:
        if row_ink[y]:
            start = y
            gap = 0
            while y < height and gap <= _GAP_TOLERANCE_PX:
                if row_ink[y]:
                    gap = 0
                else:
                    gap += 1
                y += 1
            end = y - gap
            if end - start >= _MIN_LINE_HEIGHT_PX:
                lines.append((start, end))
        else:
            y += 1

    out_lines = []
    for top, bottom in lines:
        bbox = inv.crop((0, top, width, bottom)).getbbox()
        if bbox is None:
            continue
        left, _, right, _ = bbox
        out_lines.append(
            {
                "top": top,
                "bottom": bottom,
                "left_x": left,
                "width": right - left,
                "height": bottom - top,
            }
        )
    return {"lines": out_lines, "page_width": width}


def analyze_first_span(layout_data: dict, delta: float, tolerance: float) -> bool:
    """Return True if the first two real text lines are indented from each other.

    Only lines wider than ``_MIN_TEXT_LINE_WIDTH_PX`` are compared. Indented
    means their left X differs by more than ``delta - tolerance`` px.
    """
    lines = layout_data.get("lines", [])
    real = [ln for ln in lines if ln["width"] > _MIN_TEXT_LINE_WIDTH_PX]
    if len(real) < 2:
        return False
    threshold = delta - tolerance
    return abs(real[0]["left_x"] - real[1]["left_x"]) > threshold
