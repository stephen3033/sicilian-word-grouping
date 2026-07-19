"""Validation layer: parse + unwrap + schema/grounding quality gate.

Pydantic `mode="after"` validators run after field-level validation, so
schema conformance strictly precedes grounding. Failures raise
`ValidationError`.

Deterministic fields (`vs_vol`, `page_numbers`) are placeholder `0`
emitted by the model and injected here in the same per-entry loop pass,
so the entries list never gets iterated a second time.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from src.common.errors import ValidationError
from src.common.logger import log_errors
from src.common.normalize import normalize
from src.config import Settings, get_settings
from src.models import DictionaryEntry

logger = logging.getLogger(__name__)


@log_errors
def validate(
    raw_json: str,
    ocr_text: str,
    image_b64: str,
    page_number: int,
    settings: Settings | None = None,
) -> list[DictionaryEntry]:
    """Validate `{"entries": [...]}` against schema + OCR grounding.

    `ocr_text` is the prefix-stripped page OCR (ground truth for substring
    checks). `image_b64` is decoded once and threaded as `image_payload`
    for the headword-null layout heuristic on the first entry.
    `page_number` is the printed page number, injected into each entry's
    `page_numbers` (overriding the model's `[0]` placeholder) along with
    `vs_vol` (overriding `0`) in the same per-entry pass.

    Raises `ValidationError` on any parse, schema, or grounding failure.
    """
    if settings is None:
        settings = get_settings()
    logger.debug(
        "raw_json=%d chars ocr=%d chars image=%d b64 page=%d",
        len(raw_json),
        len(ocr_text),
        len(image_b64),
        page_number,
    )

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValidationError(f"raw payload is not valid JSON: {e}") from e

    if not isinstance(payload, dict) or "entries" not in payload:
        raise ValidationError("payload missing top-level 'entries' key")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValidationError(
            f"'entries' is not a list (got {type(raw_entries).__name__})"
        )
    logger.debug("unwrapped %d entries", len(raw_entries))

    normalized_ocr = normalize(ocr_text)
    image_payload = base64.b64decode(image_b64)
    entries: list[DictionaryEntry] = []
    for i, raw_entry in enumerate(raw_entries):
        try:
            entry = DictionaryEntry.model_validate(
                raw_entry,
                context={
                    "normalized_ocr": normalized_ocr,
                    "index": i,
                    "image_payload": image_payload,
                    "headword_delta": settings.headword_delta,
                    "tolerance": settings.layout_tolerance,
                },
            )
        except PydanticValidationError as e:
            raise ValidationError(
                f"entry {i} failed schema conformance: {e}"
            ) from e
        entries.append(
            entry.model_copy(
                update={"vs_vol": settings.volume, "page_numbers": [page_number]}
            )
        )
        logger.debug("entry %d schema + grounding ok (vs_vol=%d page=%d)",
                     i, settings.volume, page_number)

    logger.info("%d entries passed (page=%d)", len(entries), page_number)
    return entries


@log_errors
def persist_validated_page(
    entries: list[DictionaryEntry], page_number: int, settings: Settings
) -> Path:
    """Write a page's validated entries to disk (debug mode only).

    Returns the path written. Parents are created as needed.
    """
    out_path = settings.validated_page_path(page_number, settings.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": [e.model_dump() for e in entries]}
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.debug("wrote %d entries to %s", len(entries), out_path)
    return out_path
