from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.models import DictionaryEntry
from src.transform.prompter import _DEFAULT_PREAMBLE, _USER_PREAMBLES, build_user_prompt

_DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"


@pytest.fixture(autouse=True)
def _default_model(monkeypatch):
    s = Settings()
    s.model = _DEFAULT_MODEL
    monkeypatch.setattr("src.transform.prompter.get_settings", lambda: s)


def _set_model(monkeypatch, model: str) -> None:
    s = Settings()
    s.model = model
    monkeypatch.setattr("src.transform.prompter.get_settings", lambda: s)


class TestBuildUserPrompt:
    def test_schema_block_is_valid_jsonschema_for_dictionary_entry(self):
        out = build_user_prompt("ignored")
        schema_block = out.split("json_schema:\n", 1)[1].split(
            "\n\nocr_page_text:", 1
        )[0]
        assert json.loads(schema_block) == DictionaryEntry.model_json_schema()

    def test_page_text_present_verbatim_at_end(self):
        page_text = "line one\nline two with weird chars: \xe0\xe8\xec"
        assert build_user_prompt(page_text).endswith(page_text)

    def test_schema_precedes_ocr_text(self):
        out = build_user_prompt("some ocr text")
        assert out.index("json_schema:") < out.index("ocr_page_text:")

    def test_unknown_model_falls_back_to_default(self, monkeypatch):
        _set_model(monkeypatch, "some/unknown-model")
        assert _DEFAULT_PREAMBLE in build_user_prompt("text")

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4.6",
            "google/gemini-3.5-flash",
            "openai/gpt-5.4",
            "moonshotai/kimi-k2.7-code",
        ],
    )
    def test_known_model_preamble_present(self, monkeypatch, model):
        _set_model(monkeypatch, model)
        assert _USER_PREAMBLES[model] in build_user_prompt("text")

    def test_preamble_precedes_schema(self):
        out = build_user_prompt("text")
        assert out.index(_USER_PREAMBLES[_DEFAULT_MODEL]) < out.index("json_schema:")
