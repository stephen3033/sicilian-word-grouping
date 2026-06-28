"""Unit tests for src.transform.client (no network; OpenAI is faked)."""

from __future__ import annotations

from types import SimpleNamespace

from src.config import Settings
from src.transform.client import _get_client, extract_json

_CANNED_CONTENT = '{"headword": "parrinu", "page_numbers": [42]}'

_SYSTEM = "you are a lexicographer"
_USER = "json_schema:\n...\nocr_page_text:\nfoo"


class _FakeOpenAI:
    """Records constructor args and the last `create` kwargs; returns canned text."""

    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.last_create_kwargs: dict = {}
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )
        _FakeOpenAI.instances.append(self)

    def _create(self, **kwargs):
        self.last_create_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=_CANNED_CONTENT))]
        )


def _patch(monkeypatch) -> Settings:
    s = Settings()
    s.openai_api_key = type(s.openai_api_key)("test-key")
    monkeypatch.setattr("src.transform.client.get_settings", lambda: s)
    _get_client.cache_clear()
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr("src.transform.client.OpenAI", _FakeOpenAI)
    return s


class TestExtractJson:
    def test_returns_raw_message_content_verbatim(self, monkeypatch):
        _patch(monkeypatch)
        assert extract_json("b64abc", _SYSTEM, _USER) == _CANNED_CONTENT

    def test_sends_system_then_vision_user_message(self, monkeypatch):
        s = _patch(monkeypatch)
        extract_json("b64abc", _SYSTEM, _USER)

        assert len(_FakeOpenAI.instances) == 1
        inst = _FakeOpenAI.instances[0]
        assert inst.kwargs == {
            "base_url": s.openai_base_url,
            "api_key": "test-key",
        }

        kwargs = inst.last_create_kwargs
        assert kwargs["model"] == s.model
        messages = kwargs["messages"]
        assert len(messages) == 2

        assert messages[0] == {"role": "system", "content": _SYSTEM}

        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert len(content) == 2
        assert content[0] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,b64abc"},
        }
        assert content[1] == {"type": "text", "text": _USER}
