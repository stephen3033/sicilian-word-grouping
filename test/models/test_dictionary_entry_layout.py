"""Schema tests for model-owned and persisted dictionary entries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import DictionaryEntry, LLMEntry, ReviewStatus


def test_llm_entry_schema_contains_only_model_owned_fields():
    schema = LLMEntry.model_json_schema()
    assert set(schema["properties"]) == {"headword", "trailing_text", "variants"}
    assert set(schema["required"]) == {"headword", "trailing_text", "variants"}
    assert schema["additionalProperties"] is False


def test_dictionary_entry_is_a_flat_llm_entry_subclass():
    schema = DictionaryEntry.model_json_schema()
    assert issubclass(DictionaryEntry, LLMEntry)
    assert set(schema["properties"]) == {
        "headword",
        "trailing_text",
        "variants",
        "page_numbers",
        "vs_vol",
        "is_review_needed",
        "review_reason",
    }
    assert set(schema["required"]) == set(schema["properties"])


def test_review_status_values_are_stable():
    assert [status.value for status in ReviewStatus] == [
        "passed",
        "machine",
        "human",
    ]


def test_dictionary_review_fields_are_required():
    with pytest.raises(ValidationError):
        DictionaryEntry(
            headword="a",
            trailing_text="body",
            variants=None,
            page_numbers=[1],
            vs_vol=1,
        )
