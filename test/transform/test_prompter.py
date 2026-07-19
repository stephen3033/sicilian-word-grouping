from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.models import DictionaryEntry
from src.transform.prompter import DEFAULT_USER_PREAMBLE, build_user_prompt


@pytest.fixture(autouse=True)
def _default_model(monkeypatch):
    s = Settings()
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

    def test_default_preamble_present(self):
        assert DEFAULT_USER_PREAMBLE in build_user_prompt("text")

    def test_preamble_precedes_schema(self):
        out = build_user_prompt("text")
        assert out.index(DEFAULT_USER_PREAMBLE) < out.index("json_schema:")

    def test_preamble_same_regardless_of_model(self, monkeypatch):
        s1 = Settings()
        s1.model = "anthropic/claude-sonnet-4.6"
        s2 = Settings()
        s2.model = "some/other-model"
        monkeypatch.setattr("src.transform.prompter.get_settings", lambda: s1)
        p1 = build_user_prompt("text")
        monkeypatch.setattr("src.transform.prompter.get_settings", lambda: s2)
        p2 = build_user_prompt("text")
        assert p1 == p2