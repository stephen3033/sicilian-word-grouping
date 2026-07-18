"""Validation layer of the ETVL pipeline.

The validate step is a runtime quality gate. It takes the raw JSON string
emitted by the transform layer, parses it, and runs two ordered phases:

1. Parse & unwrap - the raw JSON is parsed and the top-level ``entries``
   list is extracted. A failure here kills the attempt immediately.
2. Schema conformance + grounding - each entry is validated against the
   `DictionaryEntry` pydantic model via `model_validate`, with the page's
   OCR text (pre-normalized once via `normalization()`) and the entry
   index threaded through the validation context. Pydantic runs the
   model's `@model_validator(mode="after")` checks in definition order
   - `headword`, then `variants`, then `trailing_text` - each grounding
   its field against the normalized OCR text. A failure in any check
   kills the attempt.

`mode="after"` validators only run once all field-level validation
(types, required fields) has passed, so schema conformance strictly
precedes grounding - enforced by Pydantic, not by code ordering.

Failures are logged to the configured logfile and raise `ValidationError`.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError as PydanticValidationError

from src.common.errors import ValidationError
from src.common.logger import log_errors
from src.common.normalize import normalization
from src.models import DictionaryEntry

logger = logging.getLogger(__name__)


@log_errors
def validate(raw_json: str, ocr_text: str) -> list[DictionaryEntry]:
    """Validate a transformed payload against the schema and OCR grounding.

    Args:
        raw_json: Raw JSON string returned by the transform layer, shaped
            as `{"entries": [ {DictionaryEntry}, ... ]}`.
        ocr_text: The page's OCR text (prefix-stripped) used as the ground
            truth for headword / variant / trailing_text substring checks.

    Returns:
        The list of schema-valid, ground-truth-checked `DictionaryEntry`
        instances on success.

    Raises:
        ValidationError: on any parse, schema, or grounding failure.
    """
    logger.debug(
        "validate: raw_json=%d chars ocr=%d chars", len(raw_json), len(ocr_text)
    )

    # --- 1. JSON parse -------------------------------------------------
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.error("validate: raw payload is not valid JSON: %s", e)
        raise ValidationError(f"raw payload is not valid JSON: {e}") from e

    # --- 2. Unwrap entries ---------------------------------------------
    if not isinstance(payload, dict) or "entries" not in payload:
        logger.error("validate: payload missing top-level 'entries' key")
        raise ValidationError("payload missing top-level 'entries' key")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        logger.error(
            "validate: 'entries' is not a list (got %s)",
            type(raw_entries).__name__,
        )
        raise ValidationError("'entries' is not a list")
    logger.debug("validate: unwrapped %d entries", len(raw_entries))

    # --- 3. Schema conformance + grounding -----------------------------
    # The page's OCR text is normalized once here; every per-entry
    # `@model_validator(mode="after")` grounding check receives it via the
    # validation context and normalizes only the field side before the
    # verbatim substring (`in`) check. Pydantic runs the model validators
    # in definition order - headword, variants, trailing_text - and only
    # after all field-level validation has passed, so schema conformance
    # strictly precedes grounding.
    normalized_ocr = normalization(ocr_text)
    entries: list[DictionaryEntry] = []
    for i, raw_entry in enumerate(raw_entries):
        try:
            entries.append(
                DictionaryEntry.model_validate(
                    raw_entry,
                    context={"normalized_ocr": normalized_ocr, "index": i},
                )
            )
            logger.debug("validate: entry %d schema + grounding ok", i)
        except PydanticValidationError as e:
            logger.error(
                "validate: entry %d failed schema conformance: %s", i, e
            )
            raise ValidationError(
                f"entry {i} failed schema conformance: {e}"
            ) from e

    logger.info("validate: %d entries passed", len(entries))
    return entries
