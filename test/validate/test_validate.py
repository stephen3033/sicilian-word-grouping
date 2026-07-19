"""Unit tests for src.validate.validate (no network; pure function)."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image, ImageDraw

from src.common.errors import ValidationError
from src.models import DictionaryEntry
from src.validate.validate import validate

_OCR = (
    "1 a¹ f. e (antiq.) m. vocale e prima lettera dell'alfabeto.\n"
    "a² art. femm. la. V. anche la¹. Cfr. u2.\n"
    "a³ pron. femm. la. Anche la². Cfr. u³.\n"
)


def _png_b64(line_specs: list[tuple[int, int, int, int]]) -> str:
    """Build a white PNG with black-rectangle 'text lines' ((left, top, width, height))."""
    img = Image.new("RGB", (1000, 800), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for left, top, width, height in line_specs:
        draw.rectangle(
            [left, top, left + width - 1, top + height - 1], fill=(0, 0, 0)
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# |Δx|=60 > threshold -> headword layout (is_orphan_fragment=False expected).
_HEADWORD_IMG_B64 = _png_b64([(100, 50, 600, 40), (160, 120, 600, 40)])
# |Δx|=0 <= threshold -> orphan layout (is_orphan_fragment=True expected).
_ORPHAN_IMG_B64 = _png_b64([(100, 50, 600, 40), (100, 120, 600, 40)])


def _payload(entries: list[dict]) -> str:
    return json.dumps({"entries": entries})


def _entry(
    headword: str | None = "a²",
    trailing_text: str = "art. femm. la.",
    variants: list[str] | None = None,
    page_numbers: list[int] = (1,),
    is_orphan_fragment: bool = False,
) -> dict:
    return {
        "headword": headword,
        "trailing_text": trailing_text,
        "variants": variants,
        "page_numbers": list(page_numbers),
        "is_orphan_fragment": is_orphan_fragment,
    }


class TestValidateSuccess:
    def test_valid_headword_and_variants_returns_entries(self):
        out = validate(
            _payload([_entry(headword="a²", variants=["la¹", "u2"])]),
            _OCR,
            _HEADWORD_IMG_B64,
        )
        assert len(out) == 1
        assert isinstance(out[0], DictionaryEntry)
        assert out[0].headword == "a²"
        assert out[0].variants == ["la¹", "u2"]

    def test_orphan_entry_with_null_headword_and_null_variants_passes(self):
        out = validate(
            _payload(
                [
                    _entry(
                        headword=None,
                        trailing_text="a¹ f. e (antiq.) m. vocale",
                        variants=None,
                        is_orphan_fragment=True,
                    )
                ]
            ),
            _OCR,
            _ORPHAN_IMG_B64,
        )
        assert len(out) == 1
        assert out[0].headword is None
        assert out[0].variants is None

    def test_empty_variants_list_skips_grounding(self):
        out = validate(
            _payload([_entry(variants=[])]), _OCR, _HEADWORD_IMG_B64
        )
        assert out[0].variants == []

    def test_multiple_entries_all_valid(self):
        raw = _payload(
            [
                _entry(headword="a¹", trailing_text="vocale"),
                _entry(headword="a²", variants=["la¹", "u2"]),
                _entry(headword="a³", trailing_text="pron. femm. la.", variants=["la²", "u³"]),
            ]
        )
        out = validate(raw, _OCR, _HEADWORD_IMG_B64)
        assert [e.headword for e in out] == ["a¹", "a²", "a³"]


class TestValidateParseFailures:
    @pytest.mark.parametrize(
        ("raw", "match"),
        [
            ("{not json", "not valid JSON"),
            (json.dumps({"words": []}), "missing top-level 'entries'"),
            (json.dumps({"entries": {"headword": "a²"}}), "'entries' is not a list"),
        ],
    )
    def test_parse_failure_raises(self, raw, match):
        with pytest.raises(ValidationError, match=match):
            validate(raw, _OCR, _HEADWORD_IMG_B64)


class TestValidateSchemaConformance:
    @pytest.mark.parametrize(
        "entry_dict",
        [
            {
                "headword": "a²",
                "trailing_text": "...",
                "variants": None,
                "is_orphan_fragment": False,
            },
            {
                "headword": "a²",
                "trailing_text": "...",
                "variants": None,
                "page_numbers": "1",
                "is_orphan_fragment": False,
            },
        ],
    )
    def test_schema_failure_raises(self, entry_dict):
        with pytest.raises(ValidationError, match="schema conformance"):
            validate(_payload([entry_dict]), _OCR, _HEADWORD_IMG_B64)

    def test_schema_failure_skips_grounding_check(self):
        # headword 'a²' is valid against OCR, but page_numbers is missing so
        # the schema check trips first and never reaches grounding.
        raw = _payload(
            [
                {
                    "headword": "a²",
                    "trailing_text": "...",
                    "variants": None,
                    "is_orphan_fragment": False,
                }
            ]
        )
        with pytest.raises(ValidationError, match="schema conformance"):
            validate(raw, _OCR, _HEADWORD_IMG_B64)


class TestValidateGrounding:
    def test_headword_not_in_ocr_raises(self):
        with pytest.raises(ValidationError, match="headword 'xyzzy' not found"):
            validate(
                _payload([_entry(headword="xyzzy")]), _OCR, _HEADWORD_IMG_B64
            )

    def test_variant_not_in_ocr_raises(self):
        raw = _payload([_entry(variants=["la¹", "nope"])])
        with pytest.raises(ValidationError, match="variant 'nope' not found"):
            validate(raw, _OCR, _HEADWORD_IMG_B64)

    def test_headword_grounding_check_happens_before_variants(self):
        with pytest.raises(ValidationError, match="headword 'xyzzy' not found"):
            validate(
                _payload([_entry(headword="xyzzy", variants=["also_missing"])]),
                _OCR,
                _HEADWORD_IMG_B64,
            )

    def test_entries_checked_in_order_second_fails(self):
        raw = _payload([_entry(headword="a²"), _entry(headword="xyzzy")])
        with pytest.raises(ValidationError, match="entry 1 headword 'xyzzy'"):
            validate(raw, _OCR, _HEADWORD_IMG_B64)


class TestValidateTrailingText:
    def test_trailing_text_present_in_ocr_passes(self):
        out = validate(_payload([_entry()]), _OCR, _HEADWORD_IMG_B64)
        assert out[0].trailing_text == "art. femm. la."

    def test_trailing_text_not_in_ocr_raises(self):
        with pytest.raises(
            ValidationError, match=r"entry 0 trailing_text 'xyzzy' not found"
        ):
            validate(
                _payload([_entry(trailing_text="xyzzy")]),
                _OCR,
                _HEADWORD_IMG_B64,
            )

    @pytest.mark.parametrize(
        "entry_dict",
        [
            {
                "headword": "a²",
                "trailing_text": None,
                "variants": None,
                "page_numbers": [1],
                "is_orphan_fragment": False,
            },
            {
                "headword": "a²",
                "variants": None,
                "page_numbers": [1],
                "is_orphan_fragment": False,
            },
        ],
    )
    def test_trailing_text_schema_failure_raises(self, entry_dict):
        with pytest.raises(ValidationError, match=r"schema conformance"):
            validate(_payload([entry_dict]), _OCR, _HEADWORD_IMG_B64)

    def test_trailing_text_normalized_whitespace_match_passes(self):
        # Raw `in` would miss the embedded newline; normalization() collapses
        # whitespace runs so "vocale e prima" is found inside the OCR text.
        out = validate(
            _payload([_entry(headword="a¹", trailing_text="vocale\ne prima")]),
            _OCR,
            _HEADWORD_IMG_B64,
        )
        assert out[0].trailing_text == "vocale\ne prima"

    def test_trailing_text_normalized_unicode_match_passes(self):
        # _OCR uses NFC "é"; trailing_text uses NFD "e\u0301"; normalization
        # NFC-composes both sides so the substring match succeeds.
        ocr = "a² art. femm. la. caff\xe9.\n"
        out = validate(
            _payload(
                [_entry(trailing_text="art. femm. la. caffe\u0301.")]
            ),
            ocr,
            _HEADWORD_IMG_B64,
        )
        assert out[0].trailing_text == "art. femm. la. caffe\u0301."

    def test_headword_check_runs_before_trailing_text(self):
        # Both invalid; model validators run in definition order, so headword wins.
        with pytest.raises(
            ValidationError, match=r"entry 0 headword 'xyzzy' not found"
        ):
            validate(
                _payload([_entry(headword="xyzzy", trailing_text="plugh")]),
                _OCR,
                _HEADWORD_IMG_B64,
            )


class TestValidateGroundingContext:
    """Grounding requires a non-empty `normalized_ocr` in the validation context."""

    _ENTRY = {
        "headword": "a²",
        "trailing_text": "art. femm. la.",
        "variants": None,
        "page_numbers": [1],
        "is_orphan_fragment": False,
    }

    @pytest.mark.parametrize(
        "context",
        [
            None,
            {},
            {"normalized_ocr": ""},
            {"normalized_ocr": "   "},
        ],
    )
    def test_model_validate_with_bad_context_raises(self, context):
        with pytest.raises(ValidationError, match=r"normalized_ocr context required"):
            DictionaryEntry.model_validate(self._ENTRY, context=context)

    @pytest.mark.parametrize("ocr", ["", "   "])
    def test_validate_with_empty_ocr_raises(self, ocr):
        with pytest.raises(ValidationError, match=r"normalized_ocr context required"):
            validate(_payload([self._ENTRY]), ocr, _HEADWORD_IMG_B64)
