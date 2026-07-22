from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import Settings
from src.extract.ocr_extractor import _build_index, extract_page_text

VS1_DATA_DIR = Path("VS")
VS1_PDF = VS1_DATA_DIR / "columns" / "VS1-1col.pdf"
VS1_OCR = VS1_DATA_DIR / "OCR_cols" / "VS1-1col-googlevision.txt"
GOLDEN_PAGES = [1, 2, 3, 500, 973]
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


@pytest.fixture
def ocr_txt(tmp_path: Path) -> Path:
    return tmp_path / "sanitized_ocr" / "VS1-1col-googlevision.txt"


class TestBuildIndex:
    def test_groups_lines_by_page(self, ocr_txt: Path):
        _write_ocr_txt(
            ocr_txt,
            [
                "1 first line of page one",
                "1 second line of page one",
                "2 a line of page two",
                "3 line of page three",
            ],
        )
        index = _build_index(str(ocr_txt), ocr_txt.stat().st_mtime)
        assert set(index) == {1, 2, 3}
        assert index[1] == ["1 first line of page one", "1 second line of page one"]
        assert index[2] == ["2 a line of page two"]
        assert index[3] == ["3 line of page three"]

    def test_raises_on_non_numeric_prefix(self, ocr_txt: Path):
        _write_ocr_txt(
            ocr_txt,
            [
                "1 a real page line",
                "garbage line with no number",
                "2 another real page line",
            ],
        )
        with pytest.raises(ValueError, match=r":2: line has no numeric page prefix"):
            _build_index(str(ocr_txt), ocr_txt.stat().st_mtime)

    def test_skips_blank_lines(self, ocr_txt: Path):
        # VS volumes 2-5 contain thousands of blank lines between blocks.
        _write_ocr_txt(
            ocr_txt,
            ["1 first line", "", "1 second line", "   ", "2 other page", ""],
        )
        index = _build_index(str(ocr_txt), ocr_txt.stat().st_mtime)
        assert index[1] == ["1 first line", "1 second line"]
        assert index[2] == ["2 other page"]

    def test_cache_invalidates_on_mtime_change(self, ocr_txt: Path):
        _write_ocr_txt(ocr_txt, ["1 original line"])
        first = _build_index(str(ocr_txt), ocr_txt.stat().st_mtime)
        assert first[1] == ["1 original line"]

        # Append and force a distinct mtime so the cache key changes.
        with ocr_txt.open("a", encoding="utf-8") as fh:
            fh.write("1 appended line\n")
        new_mtime = ocr_txt.stat().st_mtime + 10
        os.utime(ocr_txt, (new_mtime, new_mtime))

        second = _build_index(str(ocr_txt), new_mtime)
        assert second[1] == ["1 original line", "1 appended line"]


class TestExtractPageText:
    def test_strips_prefix_by_default(self, ocr_txt: Path, monkeypatch):
        _write_ocr_txt(ocr_txt, ["1 hello world", "1 second line"])
        _patch_settings(monkeypatch, ocr_txt.parent.parent)
        out = extract_page_text(1)
        assert out == "hello world\nsecond line"
        assert not any(line.startswith("1 ") for line in out.splitlines())

    def test_keeps_prefix_when_setting_off(self, ocr_txt: Path, monkeypatch):
        _write_ocr_txt(ocr_txt, ["1 hello world"])
        _patch_settings(monkeypatch, ocr_txt.parent.parent, strip_ocr_prefix=False)
        assert extract_page_text(1) == "1 hello world"

    def test_missing_page_raises(self, ocr_txt: Path, monkeypatch):
        _write_ocr_txt(ocr_txt, ["1 the only page"])
        _patch_settings(monkeypatch, ocr_txt.parent.parent)
        with pytest.raises(KeyError, match="No OCR text for page 99"):
            extract_page_text(99)

    def test_missing_file_raises_filenotfound(self, tmp_path: Path, monkeypatch):
        _patch_settings(monkeypatch, tmp_path)
        with pytest.raises(FileNotFoundError, match="OCR txt not found"):
            extract_page_text(1)

    def test_preserves_line_order(self, ocr_txt: Path, monkeypatch):
        _write_ocr_txt(
            ocr_txt,
            ["1 alpha", "1 beta", "1 gamma", "2 other page", "1 delta"],
        )
        _patch_settings(monkeypatch, ocr_txt.parent.parent)
        assert extract_page_text(1) == "alpha\nbeta\ngamma\ndelta"


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


SANITIZED_VS1 = VS1_DATA_DIR / "sanitized_ocr" / "VS1-1col-googlevision.txt"
sanitized_available = SANITIZED_VS1.exists()


@pytest.mark.skipif(not sanitized_available, reason="sanitized VS1 OCR not available")
class TestSanitizationApplied:
    """Smoke tests that the one-shot OCR sanitization is in place.

    `VS/sanitized_ocr/sanitize.py` splits OCR lines at embedded
    page-break markers so continuation text on page N+1 is attributed
    to N+1 (not N). Guards against `config.py` being re-pointed at
    unsanitized `OCR_cols/`.
    """

    def test_page_685_contains_chiuviri_continuation(self, monkeypatch):
        """Page 685 must contain the chiòviri continuation + variants.

        Unsanitized, the whole `chiòviri` entry (incl. continuation
        physically on page 685) was attributed to page 684.
        """
        _patch_settings(monkeypatch, VS1_DATA_DIR)
        text = extract_page_text(685)
        assert "visazzi piovere a catinelle" in text, (
            "page 685 OCR missing chiòviri continuation — sanitization "
            "not applied or config pointing at unsanitized OCR_cols/"
        )
        assert "chiuòviri" in text
        assert "ciòviri" in text

    def test_page_442_contains_bbramu_continuation(self, monkeypatch):
        """Page 442 must contain the bbramu continuation + variants.

        Unsanitized, the `D442 76) grido acuto. ... V. anche abbramu,
        bbrama² e bbrammu.` text on page 442 was attributed to page 441.
        """
        _patch_settings(monkeypatch, VS1_DATA_DIR)
        text = extract_page_text(442)
        assert "76) grido acuto" in text, (
            "page 442 OCR missing bbramu continuation — sanitization "
            "not applied or config pointing at unsanitized OCR_cols/"
        )
        assert "abbramu" in text
        assert "bbrama²" in text
        assert "bbrammu" in text

    def test_no_embedded_page_markers_remain(self):
        """Sanitized VS1 OCR must have zero remaining `page+1` markers."""
        import re
        pattern = re.compile(
            r"(?<=[a-zA-ZàèéìòùḍÀÈÉÌÒÙḌ])(\d{1,3}) "
        )
        remaining = 0
        for line in SANITIZED_VS1.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prefix = line.partition(" ")[0]
            if not prefix.isdigit():
                continue
            page = int(prefix)
            for match in pattern.finditer(line):
                if int(match.group(1)) == page + 1:
                    remaining += 1
        assert remaining == 0, (
            f"sanitized VS1 OCR has {remaining} remaining page+1 markers "
            f"— re-run VS/sanitized_ocr/sanitize.py"
        )
