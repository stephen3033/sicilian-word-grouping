"""OpenRouter/OpenAI-compatible vision-language client."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from openai import OpenAI

from src.common.cost import record_call
from src.common.logger import log_errors
from src.config import get_settings

logger = logging.getLogger(__name__)

# OpenRouter exposes the billed cost of a request via response headers
# (the exact header name has shifted across API versions) and, on some
# plans, via a ``cost`` field inside the response body's ``usage`` block.
# We try the known header names in order, then fall back to body fields.
_OR_COST_HEADERS = ("x-or-cost-total", "x-or-cost", "x-or-step-cost-total")
_OR_USAGE_COST_FIELDS = ("cost", "cost_total")


@lru_cache
def _get_client() -> OpenAI:
    s = get_settings()
    return OpenAI(
        base_url=s.openai_base_url,
        api_key=s.openai_api_key.get_secret_value(),
    )


def _parse_or_cost(headers: Any, usage: Any) -> float | None:
    """Extract the USD cost of one call from OpenRouter's response.

    Returns ``None`` if no cost field is present / parseable so the
    tracker can log the gap instead of silently zeroing it.
    """
    if headers is not None:
        get = getattr(headers, "get", None)
        if callable(get):
            for key in _OR_COST_HEADERS:
                raw = get(key)
                if raw:
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        continue

    if usage is not None:
        for attr in _OR_USAGE_COST_FIELDS:
            value = getattr(usage, attr, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(attr)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
    return None


@log_errors
def extract_json(
    base64_image: str,
    system_prompt: str,
    user_prompt: str,
    *,
    page: int,
) -> str:
    """Send image + system/user prompts to the VLM; return raw response text.

    ``page`` is keyword-only so the per-call cost reported by OpenRouter
    can be attributed to the right page in the running tally maintained
    by ``src.common.cost``.
    """
    s = get_settings()
    raw = _get_client().chat.completions.with_raw_response.create(
        model=s.model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
    )
    response = raw.parse()
    content = response.choices[0].message.content or ""

    if s.track_cost:
        cost = _parse_or_cost(getattr(raw, "headers", None), getattr(response, "usage", None))
        record_call(page, cost)
    else:
        cost = None

    logger.debug(
        "model=%s image=%d chars prompt=%d chars response=%d chars cost=%s",
        s.model,
        len(base64_image),
        len(user_prompt),
        len(content),
        "n/a" if cost is None else f"${cost:.6f}",
    )
    return content