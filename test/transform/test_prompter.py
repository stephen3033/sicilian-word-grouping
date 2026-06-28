from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.models import DictionaryEntry
from src.transform.prompter import _USER_PREAMBLES, build_user_prompt


def _patch_model(monkeypatch, model: str) -> None:
    s = Settings()
    s.model = model
    monkeypatch.setattr("src.transform.prompter.get_settings", lambda: s)


class TestBuildUserPrompt:
    def test_schema_block_is_valid_jsonschema_for_dictionary_entry(
        self, monkeypatch
    ):
        _patch_model(monkeypatch, "anthropic/claude-sonnet-4.6")
        out = build_user_prompt("ignored")
        schema_block = out.split("json_schema:\n", 1)[1].split(
            "\n\nocr_page_text:", 1
        )[0]
        assert json.loads(schema_block) == DictionaryEntry.model_json_schema()

    def test_page_text_present_verbatim_at_end(self, monkeypatch):
        _patch_model(monkeypatch, "anthropic/claude-sonnet-4.6")
        page_text = "line one\nline two with weird chars: \xe0\xe8\xec"
        assert build_user_prompt(page_text).endswith(page_text)

    def test_schema_precedes_ocr_text(self, monkeypatch):
        _patch_model(monkeypatch, "anthropic/claude-sonnet-4.6")
        out = build_user_prompt("some ocr text")
        assert out.index("json_schema:") < out.index("ocr_page_text:")

    def test_unknown_model_raises_keyerror(self, monkeypatch):
        _patch_model(monkeypatch, "some/unknown-model")
        with pytest.raises(KeyError):
            build_user_prompt("text")

    def test_known_model_preamble_present(self, monkeypatch):
        _patch_model(monkeypatch, "anthropic/claude-sonnet-4.6")
        assert _USER_PREAMBLES["anthropic/claude-sonnet-4.6"] in build_user_prompt(
            "text"
        )

    def test_preamble_precedes_schema(self, monkeypatch):
        _patch_model(monkeypatch, "anthropic/claude-sonnet-4.6")
        out = build_user_prompt("text")
        assert out.index(
            _USER_PREAMBLES["anthropic/claude-sonnet-4.6"]
        ) < out.index("json_schema:")
