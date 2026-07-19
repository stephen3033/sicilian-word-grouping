"""Validation layer: parse + unwrap + schema/grounding quality gate.

Pydantic `mode="after"` validators run after field-level validation, so
schema conformance strictly precedes grounding. Failures raise
`ValidationError`.
"""

from __future__ import annotations

import base64
import json
import logging

from pydantic import ValidationError as PydanticValidationError

from src.common.errors import ValidationError
from src.common.logger import log_errors
from src.common.normalize import normalization
from src.config import get_settings
from src.models import DictionaryEntry

logger = logging.getLogger(__name__)


@log_errors
def validate(
    raw_json: str, ocr_text: str, image_b64: str
) -> list[DictionaryEntry]:
    """Validate `{"entries": [...]}` against schema + OCR grounding.

    `ocr_text` is the prefix-stripped page OCR (ground truth for substring
    checks). `image_b64` is decoded once and threaded as `image_payload`
    for the `is_orphan_fragment` layout heuristic on the first entry.

    Raises `ValidationError` on any parse, schema, or grounding failure.
    """
    logger.debug(
        "raw_json=%d chars ocr=%d chars image=%d b64",
        len(raw_json),
        len(ocr_text),
        len(image_b64),
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

    normalized_ocr = normalization(ocr_text)
    image_payload = base64.b64decode(image_b64)
    s = get_settings()
    tolerance = s.layout_tolerance
    headword_delta = s.headword_delta
    entries: list[DictionaryEntry] = []
    for i, raw_entry in enumerate(raw_entries):
        try:
            entries.append(
                DictionaryEntry.model_validate(
                    raw_entry,
                    context={
                        "normalized_ocr": normalized_ocr,
                        "index": i,
                        "image_payload": image_payload,
                        "headword_delta": headword_delta,
                        "tolerance": tolerance,
                    },
                )
            )
            logger.debug("entry %d schema + grounding ok", i)
        except PydanticValidationError as e:
            raise ValidationError(
                f"entry {i} failed schema conformance: {e}"
            ) from e

    logger.info("%d entries passed", len(entries))
    return entries
