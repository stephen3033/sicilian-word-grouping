"""Unit tests for src.load.load (stitch + read_pages_from_disk)."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import Settings
from src.load.load import read_pages_from_disk, stitch
from src.models import DictionaryEntry
from src.validate.validate import persist_validated_page, validate

_OCR = (
    "1 a¹ f. e (antiq.) m. vocale e prima lettera dell'alfabeto.\n"
    "a² art. femm. la. V. anche la¹. Cfr. u2.\n"
    "a³ pron. femm. la. Anche la². Cfr. u³.\n"
)
# Reuse the headword-layout image pattern from the validate tests by
# constructing one inline so this module is self-contained.
import base64
import io

from PIL import Image, ImageDraw


def _png_b64(line_specs: list[tuple[int, int, int, int]]) -> str:
    img = Image.new("RGB", (1000, 800), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for left, top, width, height in line_specs:
        draw.rectangle(
            [left, top, left + width - 1, top + height - 1], fill=(0, 0, 0)
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_HEADWORD_IMG_B64 = _png_b64([(100, 50, 600, 40), (160, 120, 600, 40)])


def _payload(entries: list[dict]) -> str:
    return json.dumps({"entries": entries})


def _entry(headword="a²", trailing_text="art. femm. la.", variants=None,
           page_numbers=(0,), vs_vol=0) -> dict:
    return {
        "headword": headword,
        "trailing_text": trailing_text,
        "variants": variants,
        "page_numbers": list(page_numbers),
        "vs_vol": vs_vol,
    }


class TestStitch:
    def test_writes_metadata_and_concatenates_entries_in_page_order(
        self, tmp_path: Path
    ):
        settings = Settings(output_dir=tmp_path)
        # Insert out of order to prove stitch sorts by page number.
        page1_raw = _payload(
            [
                _entry(headword="a¹", trailing_text="vocale"),
                _entry(headword="a²", variants=["la¹", "u2"]),
            ]
        )
        page2_raw = _payload([_entry(headword="a³")])
        entries_by_page = {
            2: validate(page2_raw, _OCR, _HEADWORD_IMG_B64, page_number=2),
            1: validate(page1_raw, _OCR, _HEADWORD_IMG_B64, page_number=1),
        }
        out_path = stitch(entries_by_page, settings)
        assert out_path == tmp_path / "vs_1_anthropic-claude-sonnet-4.6.json"
        assert out_path.exists()
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["volume"] == 1
        assert payload["model"] == "anthropic/claude-sonnet-4.6"
        assert payload["page_count"] == 2
        assert payload["entry_count"] == 3
        # Page order preserved in the flat entries list.
        assert [e["headword"] for e in payload["entries"]] == [
            "a¹", "a²", "a³",
        ]
        # Deterministic fields injected by validate() survive the stitch.
        assert all(e["vs_vol"] == 1 for e in payload["entries"])
        assert all(e["page_numbers"] in ([1], [2]) for e in payload["entries"])

    def test_empty_pages_dict_writes_empty_entries_envelope(
        self, tmp_path: Path
    ):
        settings = Settings(output_dir=tmp_path)
        out_path = stitch({}, settings)
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload == {
            "volume": 1,
            "model": "anthropic/claude-sonnet-4.6",
            "page_count": 0,
            "entry_count": 0,
            "entries": [],
        }


class TestReadPagesFromDisk:
    def test_round_trips_persist_validated_page(self, tmp_path: Path):
        settings = Settings(output_dir=tmp_path)
        entries_p1 = validate(
            _payload([_entry(headword="a²", variants=["la¹", "u2"])]),
            _OCR, _HEADWORD_IMG_B64, page_number=1,
        )
        entries_p2 = validate(
            _payload([_entry(headword="a³")]),
            _OCR, _HEADWORD_IMG_B64, page_number=2,
        )
        persist_validated_page(entries_p1, 1, settings)
        persist_validated_page(entries_p2, 2, settings)

        result = read_pages_from_disk(settings)
        assert sorted(result.keys()) == [1, 2]
        assert all(isinstance(v, list) for v in result.values())
        assert all(isinstance(e, DictionaryEntry) for e in result[1])
        assert result[1][0].headword == "a²"
        assert result[1][0].vs_vol == 1
        assert result[1][0].page_numbers == [1]
        assert result[2][0].headword == "a³"
        assert result[2][0].page_numbers == [2]

    def test_missing_pages_dir_returns_empty_dict(self, tmp_path: Path):
        settings = Settings(output_dir=tmp_path)
        # No persist calls -> dir does not exist.
        assert read_pages_from_disk(settings) == {}

    def test_unrecognized_filename_is_skipped(self, tmp_path: Path):
        settings = Settings(output_dir=tmp_path)
        entries = validate(
            _payload([_entry()]), _OCR, _HEADWORD_IMG_B64, page_number=1
        )
        persist_validated_page(entries, 1, settings)
        # Drop a junk file alongside the valid one.
        junk = settings.validated_pages_dir() / "notes.json"
        junk.write_text("{}", encoding="utf-8")
        result = read_pages_from_disk(settings)
        assert list(result.keys()) == [1]