from __future__ import annotations

import base64
import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from src.config import Settings
from src.extract.pdf_extractor import _composite, _render_page, extract_page_image

VS1_DATA_DIR = Path("VS")
VS1_PDF = VS1_DATA_DIR / "columns" / "VS1-1col.pdf"
GOLDEN_PAGES = [1, 500, 973]
OUTPUT_DIR = Path("test/data/extract/output")

vs1_available = VS1_PDF.exists()


def _patch_settings(monkeypatch, data_dir: Path, **overrides) -> None:
    base = {"data_dir": data_dir, "volume": 1}
    base.update(overrides)
    settings = Settings(**base)
    monkeypatch.setattr(
        "src.extract.pdf_extractor.get_settings", lambda: settings
    )


def _make_tiny_pdf(path: Path, page_count: int = 4) -> None:
    """Write a minimal PDF with `page_count` A4 pages, each holding a colored rectangle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for i in range(page_count):
        page = doc.new_page(width=595, height=842)
        page.draw_rect(
            pymupdf.Rect(50, 50, 545, 792),
            color=(0.1 * i, 0.2 * i, 0.3 * i),
            fill=(0.1 * i, 0.2 * i, 0.3 * i),
        )
    doc.save(path)
    doc.close()


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "columns" / "VS1-1col.pdf"
    _make_tiny_pdf(pdf, page_count=4)
    return pdf


class TestComposite:
    def test_vertical_stacks_left_above_right(self):
        left = Image.new("RGB", (10, 20), "red")
        right = Image.new("RGB", (10, 30), "blue")
        out = _composite(left, right, "vertical")
        assert out.size == (10, 50)
        assert out.getpixel((0, 0)) == (255, 0, 0)  # left top
        assert out.getpixel((0, 20)) == (0, 0, 255)  # right top

    def test_horizontal_places_left_then_right(self):
        left = Image.new("RGB", (10, 20), "red")
        right = Image.new("RGB", (15, 20), "blue")
        out = _composite(left, right, "horizontal")
        assert out.size == (25, 20)
        assert out.getpixel((0, 0)) == (255, 0, 0)
        assert out.getpixel((10, 0)) == (0, 0, 255)

    def test_vertical_pads_narrower_column_with_white(self):
        left = Image.new("RGB", (10, 20), "red")
        right = Image.new("RGB", (5, 30), "blue")
        out = _composite(left, right, "vertical")
        assert out.size == (10, 50)
        # right column narrower; extra pixels on its row stay white
        assert out.getpixel((9, 20)) == (255, 255, 255)
        assert out.getpixel((4, 20)) == (0, 0, 255)


class TestRenderPage:
    def test_returns_rgb_image_with_page_dimensions(self, tiny_pdf: Path):
        with pymupdf.open(tiny_pdf) as doc:
            img = _render_page(doc, 0, dpi=72)
        assert img.mode == "RGB"
        # 595x842 points at 72 dpi == 595x842 pixels
        assert img.size == (595, 842)


class TestExtractPageImage:
    def test_returns_valid_base64_png(self, tiny_pdf: Path, monkeypatch):
        _patch_settings(monkeypatch, tiny_pdf.parent.parent)
        out = extract_page_image(1)
        img = Image.open(io.BytesIO(base64.b64decode(out)))
        assert img.format == "PNG"
        assert img.size[0] > 0 and img.size[1] > 0

    def test_out_of_range_raises_valueerror(self, tiny_pdf: Path, monkeypatch):
        _patch_settings(monkeypatch, tiny_pdf.parent.parent)
        # 4 PDF pages -> printed_page_count == 2
        with pytest.raises(ValueError, match="out of range"):
            extract_page_image(3)
        with pytest.raises(ValueError, match="out of range"):
            extract_page_image(0)

    def test_missing_file_raises_filenotfound(self, tmp_path: Path, monkeypatch):
        _patch_settings(monkeypatch, tmp_path)
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            extract_page_image(1)

    def test_horizontal_doubles_width_vs_single_column(
        self, tiny_pdf: Path, monkeypatch
    ):
        _patch_settings(monkeypatch, tiny_pdf.parent.parent, column_layout="vertical")
        vertical_b64 = extract_page_image(1)

        _patch_settings(monkeypatch, tiny_pdf.parent.parent, column_layout="horizontal")
        horizontal_b64 = extract_page_image(1)

        vertical = Image.open(io.BytesIO(base64.b64decode(vertical_b64)))
        horizontal = Image.open(io.BytesIO(base64.b64decode(horizontal_b64)))
        # horizontal width ~ 2x vertical width; vertical height ~ 2x horizontal
        assert horizontal.size[0] == pytest.approx(vertical.size[0] * 2, rel=0.05)
        assert vertical.size[1] == pytest.approx(horizontal.size[1] * 2, rel=0.05)


@pytest.mark.skipif(not vs1_available, reason="VS1 data not available")
@pytest.mark.parametrize("page", GOLDEN_PAGES)
def test_pdf_page_image_written_and_valid(page: int):
    """Render the page, validate it's a PNG, and write it to output/ for review."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_b64 = extract_page_image(page)

    img = Image.open(io.BytesIO(base64.b64decode(out_b64)))
    assert img.format == "PNG"
    assert img.size[0] > 0 and img.size[1] > 0

    out_path = OUTPUT_DIR / f"page_{page:03d}.png"
    img.save(out_path)
