"""Unit tests for src.transform.client (no network; OpenAI is faked)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.common import cost as cost_module
from src.common.errors import ValidationError
from src.config import Settings
from src.transform.client import _get_client, extract_json

_CANNED_CONTENT = '{"headword": "parrinu", "page_numbers": [42]}'

_SYSTEM = "you are a lexicographer"

_USER = "json_schema:\n...\nocr_page_text:\nfoo"


class _FakeRawResponse:
    """Mimics openai's `with_raw_response.create(...)` payload: `.parse()`
    returns the typed ChatCompletion; `.headers` carries provider headers."""

    def __init__(self, completion: SimpleNamespace, headers: dict[str, str]) -> None:
        self._completion = completion
        self.headers = headers

    def parse(self) -> SimpleNamespace:
        return self._completion


class _FakeOpenAI:
    """Records constructor args and the last `create` kwargs; returns canned text."""

    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.last_create_kwargs: dict = {}
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=self._create)
            )
        )
        _FakeOpenAI.instances.append(self)

    def _create(self, **kwargs):
        self.last_create_kwargs = kwargs
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=_CANNED_CONTENT))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        return _FakeRawResponse(completion, headers={"x-or-cost-total": "0.0123"})


def _patch(monkeypatch) -> Settings:
    s = Settings()
    s.openai_api_key = type(s.openai_api_key)("test-key")
    monkeypatch.setattr("src.transform.client.get_settings", lambda: s)
    _get_client.cache_clear()
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr("src.transform.client.OpenAI", _FakeOpenAI)
    cost_module.reset()
    return s


class TestExtractJson:
    def test_returns_raw_message_content_verbatim(self, monkeypatch):
        _patch(monkeypatch)
        assert extract_json("b64abc", _SYSTEM, _USER, page=7) == _CANNED_CONTENT

    def test_sends_system_then_vision_user_message(self, monkeypatch):
        s = _patch(monkeypatch)
        extract_json("b64abc", _SYSTEM, _USER, page=7)

        assert len(_FakeOpenAI.instances) == 1
        inst = _FakeOpenAI.instances[0]
        assert inst.kwargs == {
            "base_url": s.openai_base_url,
            "api_key": "test-key",
            "max_retries": 0,
            "timeout": 120.0,
        }

        kwargs = inst.last_create_kwargs
        assert kwargs["model"] == s.model
        # Cost is requested explicitly so OpenRouter includes usage.cost.
        assert kwargs["extra_body"] == {"usage": {"include": True}}
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

    def test_empty_choices_raises_page_validation_error(self, monkeypatch):
        class _NoChoicesFakeOpenAI(_FakeOpenAI):
            def _create(self, **kwargs):
                raw = super()._create(**kwargs)
                raw._completion.choices = []
                return raw

        _patch(monkeypatch)
        monkeypatch.setattr("src.transform.client.OpenAI", _NoChoicesFakeOpenAI)
        _get_client.cache_clear()

        with pytest.raises(ValidationError, match="no choices"):
            extract_json("b64abc", _SYSTEM, _USER, page=3)

    def test_sends_configured_reasoning_effort(self, monkeypatch):
        s = _patch(monkeypatch)
        s.reasoning_effort = "high"

        extract_json("b64abc", _SYSTEM, _USER, page=7)

        assert _FakeOpenAI.instances[0].last_create_kwargs["extra_body"] == {
            "usage": {"include": True},
            "reasoning": {"effort": "high"},
        }

    def test_sends_configured_request_timeout_and_disables_sdk_retries(
        self, monkeypatch
    ):
        s = _patch(monkeypatch)
        s.request_timeout_seconds = 17.5

        extract_json("b64abc", _SYSTEM, _USER, page=7)

        assert _FakeOpenAI.instances[0].kwargs["timeout"] == 17.5
        assert _FakeOpenAI.instances[0].kwargs["max_retries"] == 0


class TestCostTracking:
    def test_records_openrouter_cost_header(self, monkeypatch):
        _patch(monkeypatch)
        extract_json("b64abc", _SYSTEM, _USER, page=7)

        assert cost_module._totals.calls == 1
        assert cost_module._totals.total_cost == pytest.approx(0.0123)
        assert cost_module._totals.by_page[7] == pytest.approx(0.0123)

    def test_records_no_cost_when_header_missing(self, monkeypatch):
        class _NoCostFakeOpenAI(_FakeOpenAI):
            def _create(self, **kwargs):
                raw = super()._create(**kwargs)
                raw.headers = {}
                return raw

        s = _patch(monkeypatch)
        monkeypatch.setattr("src.transform.client.OpenAI", _NoCostFakeOpenAI)
        _get_client.cache_clear()
        extract_json("b64abc", _SYSTEM, _USER, page=1)

        assert cost_module._totals.calls == 1
        assert cost_module._totals.total_cost == 0.0  # counted as 0
        assert cost_module._totals.by_page[1] == 0.0

    def test_accumulates_across_multiple_calls(self, monkeypatch):
        _patch(monkeypatch)
        extract_json("b64abc", _SYSTEM, _USER, page=1)
        extract_json("b64abc", _SYSTEM, _USER, page=2)
        extract_json("b64abc", _SYSTEM, _USER, page=2)  # repeated attribution

        assert cost_module._totals.calls == 3
        assert cost_module._totals.total_cost == pytest.approx(0.0369)
        assert cost_module._totals.by_page[1] == pytest.approx(0.0123)
        assert cost_module._totals.by_page[2] == pytest.approx(0.0246)

    def test_skips_recording_when_track_cost_false(self, monkeypatch):
        s = _patch(monkeypatch)
        s.track_cost = False
        monkeypatch.setattr("src.transform.client.get_settings", lambda: s)
        extract_json("b64abc", _SYSTEM, _USER, page=5)

        assert cost_module._totals.calls == 0
        assert cost_module._totals.total_cost == 0.0
