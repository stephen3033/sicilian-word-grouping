"""Unit tests for src.validate.validate (no network; pure function)."""

from __future__ import annotations

import json

import pytest

from src.models import DictionaryEntry
from src.validate import ValidationError, validate

_OCR = (
    "1 a¹ f. e (antiq.) m. vocale e prima lettera dell'alfabeto.\n"
    "a² art. femm. la. V. anche la¹. Cfr. u2.\n"
    "a³ pron. femm. la. Anche la². Cfr. u³.\n"
)


def _payload(entries: list[dict]) -> str:
    return json.dumps({"entries": entries})


class TestValidateSuccess:
    def test_valid_headword_and_variants_returns_entries(self):
        raw = _payload(
            [
                {
                    "headword": "a²",
                    "trailing_text": "art. femm. la. V. anche la¹. Cfr. u2.",
                    "variants": ["la¹", "u2"],
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                }
            ]
        )
        out = validate(raw, _OCR)
        assert len(out) == 1
        assert isinstance(out[0], DictionaryEntry)
        assert out[0].headword == "a²"
        assert out[0].variants == ["la¹", "u2"]

    def test_orphan_entry_with_null_headword_and_null_variants_passes(self):
        raw = _payload(
            [
                {
                    "headword": None,
                    "trailing_text": "a¹ f. e (antiq.) m. vocale ...",
                    "variants": None,
                    "page_numbers": [1],
                    "is_orphan_fragment": True,
                }
            ]
        )
        out = validate(raw, _OCR)
        assert len(out) == 1
        assert out[0].headword is None
        assert out[0].variants is None

    def test_empty_variants_list_skips_grounding(self):
        raw = _payload(
            [
                {
                    "headword": "a²",
                    "trailing_text": "...",
                    "variants": [],
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                }
            ]
        )
        out = validate(raw, _OCR)
        assert out[0].variants == []

    def test_multiple_entries_all_valid(self):
        raw = _payload(
            [
                {
                    "headword": "a¹",
                    "trailing_text": "vocale",
                    "variants": None,
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                },
                {
                    "headword": "a²",
                    "trailing_text": "art. femm. la.",
                    "variants": ["la¹", "u2"],
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                },
                {
                    "headword": "a³",
                    "trailing_text": "pron. femm. la.",
                    "variants": ["la²", "u³"],
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                },
            ]
        )
        out = validate(raw, _OCR)
        assert len(out) == 3
        assert [e.headword for e in out] == ["a¹", "a²", "a³"]


class TestValidateParseFailures:
    def test_malformed_json_raises(self):
        with pytest.raises(ValidationError, match="not valid JSON"):
            validate("{not json", _OCR)

    def test_missing_entries_key_raises(self):
        with pytest.raises(ValidationError, match="missing top-level 'entries'"):
            validate(json.dumps({"words": []}), _OCR)

    def test_entries_not_a_list_raises(self):
        with pytest.raises(ValidationError, match="'entries' is not a list"):
            validate(json.dumps({"entries": {"headword": "a²"}}), _OCR)


class TestValidateSchemaConformance:
    def test_entry_missing_required_page_numbers_raises(self):
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
            validate(raw, _OCR)

    def test_entry_wrong_type_raises(self):
        raw = _payload(
            [
                {
                    "headword": "a²",
                    "trailing_text": "...",
                    "variants": None,
                    "page_numbers": "1",
                    "is_orphan_fragment": False,
                }
            ]
        )
        with pytest.raises(ValidationError, match="schema conformance"):
            validate(raw, _OCR)

    def test_schema_failure_skips_grounding_check(self):
        # headword 'a²' is valid against OCR, but page_numbers is missing so
        # the schema check must trip first and never reach grounding.
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
            validate(raw, _OCR)


class TestValidateGrounding:
    def test_headword_not_in_ocr_raises(self):
        raw = _payload(
            [
                {
                    "headword": "xyzzy",
                    "trailing_text": "...",
                    "variants": None,
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                }
            ]
        )
        with pytest.raises(ValidationError, match="headword 'xyzzy' not found"):
            validate(raw, _OCR)

    def test_variant_not_in_ocr_raises(self):
        raw = _payload(
            [
                {
                    "headword": "a²",
                    "trailing_text": "...",
                    "variants": ["la¹", "nope"],
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                }
            ]
        )
        with pytest.raises(ValidationError, match="variant 'nope' not found"):
            validate(raw, _OCR)

    def test_headword_grounding_check_happens_before_variants(self):
        raw = _payload(
            [
                {
                    "headword": "xyzzy",
                    "trailing_text": "...",
                    "variants": ["also_missing"],
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                }
            ]
        )
        with pytest.raises(ValidationError, match="headword 'xyzzy' not found"):
            validate(raw, _OCR)

    def test_entries_checked_in_order_second_fails(self):
        raw = _payload(
            [
                {
                    "headword": "a²",
                    "trailing_text": "...",
                    "variants": None,
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                },
                {
                    "headword": "xyzzy",
                    "trailing_text": "...",
                    "variants": None,
                    "page_numbers": [1],
                    "is_orphan_fragment": False,
                },
            ]
        )
        with pytest.raises(ValidationError, match="entry 1 headword 'xyzzy'"):
            validate(raw, _OCR)
