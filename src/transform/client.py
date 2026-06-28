"""OpenRouter/OpenAI-compatible vision-language client."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from src.config import get_settings


@lru_cache
def _get_client() -> OpenAI:
    s = get_settings()
    return OpenAI(
        base_url=s.openai_base_url,
        api_key=s.openai_api_key.get_secret_value(),
    )


def extract_json(base64_image: str, system_prompt: str, user_prompt: str) -> str:
    """Send image + system/user prompts to the VLM; return raw response text."""
    s = get_settings()
    response = _get_client().chat.completions.create(
        model=s.model,
        response_format={"type": "json_object"},  # Hard rail: Forces raw JSON output
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
    )
    # Fallback to empty string if content returns None
    return response.choices[0].message.content or ""
