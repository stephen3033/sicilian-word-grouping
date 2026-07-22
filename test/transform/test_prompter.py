from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.models import LLMEntry
from src.transform.prompter import DEFAULT_USER_PREAMBLE, build_user_prompt


@pytest.fixture(autouse=True)
def _default_model(monkeypatch):
    monkeypatch.setattr("src.transform.prompter.get_settings", lambda: Settings())


def test_prompt_uses_only_llm_entry_schema():
    prompt = build_user_prompt("ignored")
    schema_block = prompt.split("json_schema:\n", 1)[1].split(
        "\n\nocr_page_text:", 1
    )[0]
    schema = json.loads(schema_block)
    assert schema == LLMEntry.model_json_schema()
    assert set(schema["properties"]) == {"headword", "trailing_text", "variants"}
    assert "page_numbers" not in prompt
    assert "vs_vol" not in prompt
    assert "is_review_needed" not in prompt
    assert "review_reason" not in prompt


def test_page_text_is_verbatim_at_end_after_schema():
    page_text = "line one\nline two with weird chars: àèì"
    prompt = build_user_prompt(page_text)
    assert prompt.endswith(page_text)
    assert prompt.index("json_schema:") < prompt.index("ocr_page_text:")


def test_model_agnostic_quality_preamble_is_retained():
    prompt = build_user_prompt("text")
    assert DEFAULT_USER_PREAMBLE in prompt
    assert "### HEADWORD AND VARIANT CHARACTER FIDELITY:" in prompt
    assert "### MANDATORY PRE-OUTPUT AUDIT:" in prompt
    assert "Never summarize, shorten, paraphrase, or omit body content" in prompt


def test_no_retry_reminder_or_retry_api_remains():
    prompt = build_user_prompt("text")
    assert "RETRY QUALITY PASS" not in prompt
    with pytest.raises(TypeError):
        build_user_prompt("text", retry=True)
