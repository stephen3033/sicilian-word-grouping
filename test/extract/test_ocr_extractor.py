from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import Settings
from src.extract.ocr_extractor import _build_index, extract_page_text

VS1_DATA_DIR = Path("VS")
VS1_PDF = VS1_DATA_DIR / "columns" / "VS1-1col.pdf"
VS1_OCR = VS1_DATA_DIR / "OCR_cols" / "VS1-1col-googlevision.txt"
GOLDEN_PAGES = [1, 500, 973]
EXPECTED_DIR = Path("test/data/extract/expected")
OUTPUT_DIR = Path("test/data/extract/output")

vs1_available = VS1_PDF.exists() and VS1_OCR.exists()


def _write_ocr_txt(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patch_settings(monkeypatch, data_dir: Path, **overrides) -> None:
    base = {"data_dir": data_dir, "volume": 1}
    base.update(overrides)
    settings = Settings(**base)
    monkeypatch.setattr(
        "src.extract.ocr_extractor.get_settings", lambda: settings
    )


# ---------------------------------------------------------------------------
# Unit tests — synthetic OCR txt in tmp_path, no VS data required.
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_groups_lines_by_page(self, tmp_path: Path):
        txt = tmp_path / "OCR_cols" / "VS1-1col-googlevision.txt"
        _write_ocr_txt(
            txt,
            [
                "1 first line of page one",
                "1 second line of page one",
                "2 a line of page two",
                "3 line of page three",
            ],
        )
        index = _build_index(str(txt), txt.stat().st_mtime)
        assert set(index) == {1, 2, 3}
        assert index[1] == ["1 first line of page one", "1 second line of page one"]
        assert index[2] == ["2 a line of page two"]
        assert index[3] == ["3 line of page three"]

    def test_raises_on_non_numeric_prefix(self, tmp_path: Path):
        txt = tmp_path / "OCR_cols" / "VS1-1col-googlevision.txt"
        _write_ocr_txt(
            txt,
            [
                "1 a real page line",
                "garbage line with no number",
                "2 another real page line",
            ],
        )
        with pytest.raises(KeyError, match="Bad page number"):
            _build_index(str(txt), txt.stat().st_mtime)

    def test_cache_invalidates_on_mtime_change(self, tmp_path: Path):
        txt = tmp_path / "OCR_cols" / "VS1-1col-googlevision.txt"
        _write_ocr_txt(txt, ["1 original line"])
        first = _build_index(str(txt), txt.stat().st_mtime)
        assert first[1] == ["1 original line"]

        # Append a new line and force a distinct mtime so the cache key changes.
        with txt.open("a", encoding="utf-8") as fh:
            fh.write("1 appended line\n")
        new_mtime = txt.stat().st_mtime + 10
        os.utime(txt, (new_mtime, new_mtime))

        second = _build_index(str(txt), new_mtime)
        assert second[1] == ["1 original line", "1 appended line"]


class TestExtractPageText:
    def test_strips_prefix_by_default(self, tmp_path: Path, monkeypatch):
        txt = tmp_path / "OCR_cols" / "VS1-1col-googlevision.txt"
        _write_ocr_txt(txt, ["1 hello world", "1 second line"])
        _patch_settings(monkeypatch, tmp_path)
        out = extract_page_text(1)
        assert out == "hello world\nsecond line"
        assert not any(line.startswith("1 ") for line in out.splitlines())

    def test_keeps_prefix_when_setting_off(self, tmp_path: Path, monkeypatch):
        txt = tmp_path / "OCR_cols" / "VS1-1col-googlevision.txt"
        _write_ocr_txt(txt, ["1 hello world"])
        _patch_settings(monkeypatch, tmp_path, strip_ocr_prefix=False)
        out = extract_page_text(1)
        assert out == "1 hello world"

    def test_missing_page_raises(self, tmp_path: Path, monkeypatch):
        txt = tmp_path / "OCR_cols" / "VS1-1col-googlevision.txt"
        _write_ocr_txt(txt, ["1 the only page"])
        _patch_settings(monkeypatch, tmp_path)
        with pytest.raises(KeyError, match="No OCR text for page 99"):
            extract_page_text(99)

    def test_missing_file_raises_filenotfound(self, tmp_path: Path, monkeypatch):
        _patch_settings(monkeypatch, tmp_path)
        with pytest.raises(FileNotFoundError, match="OCR txt not found"):
            extract_page_text(1)

    def test_preserves_line_order(self, tmp_path: Path, monkeypatch):
        txt = tmp_path / "OCR_cols" / "VS1-1col-googlevision.txt"
        _write_ocr_txt(
            txt,
            ["1 alpha", "1 beta", "1 gamma", "2 other page", "1 delta"],
        )
        _patch_settings(monkeypatch, tmp_path)
        out = extract_page_text(1)
        assert out == "alpha\nbeta\ngamma\ndelta"


# ---------------------------------------------------------------------------
# Integration tests — require the real VS1 corpus + manually-added golden files.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not vs1_available, reason="VS1 data not available")
@pytest.mark.parametrize("page", GOLDEN_PAGES)
def test_ocr_page_matches_expected(page: int):
    """Compare extract_page_text output against the golden txt file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    actual = extract_page_text(page)

    out_path = OUTPUT_DIR / f"page_{page:03d}.txt"
    out_path.write_text(actual, encoding="utf-8")

    expected_path = EXPECTED_DIR / f"page_{page:03d}.txt"
    assert expected_path.exists(), (
        f"expected golden file missing: {expected_path}\n"
        f"add it manually with the prefix-stripped text for VS1 page {page}"
    )
    assert actual == expected_path.read_text(encoding="utf-8")
