from __future__ import annotations

import base64
import io
import logging

import pymupdf
from PIL import Image

from src.config import get_settings
from src.common.logger import log_errors

logger = logging.getLogger(__name__)


@log_errors
def extract_page_image(page_number: int) -> str:
    """Render printed page `page_number` of the active volume to a raw base64
    PNG string (the caller supplies the `image/png` media type)."""

    s = get_settings()
    pdf_path = s.pdf_path()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pymupdf.open(pdf_path) as doc:
        printed_page_count = doc.page_count // 2
        if not 1 <= page_number <= printed_page_count:
            raise ValueError(
                f"page_number {page_number} out of range [1, {printed_page_count}]"
                f" for {pdf_path.name}"
            )
        left_img = _render_page(doc, 2 * page_number - 2, s.image_dpi)
        right_img = _render_page(doc, 2 * page_number - 1, s.image_dpi)

    composite = _composite(left_img, right_img, s.column_layout)
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    logger.debug(
        "page %d/%d dpi=%d left=%dx%d right=%dx%d composite=%dx%d "
        "layout=%s png=%d bytes b64=%d chars",
        page_number,
        printed_page_count,
        s.image_dpi,
        left_img.width,
        left_img.height,
        right_img.width,
        right_img.height,
        composite.width,
        composite.height,
        s.column_layout,
        len(png_bytes),
        len(b64),
    )
    return b64


def _render_page(doc: pymupdf.Document, page_index: int, dpi: int) -> Image.Image:
    pix = doc[page_index].get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _composite(left: Image.Image, right: Image.Image, layout: str) -> Image.Image:
    if layout == "horizontal":
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        return canvas

    # vertical (default): stack left column above right to preserve the
    # dictionary's top-to-bottom reading order (left col, then right col).
    canvas = Image.new("RGB", (max(left.width, right.width), left.height + right.height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (0, left.height))
    return canvas