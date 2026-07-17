"""Validation layer of the ETVL pipeline.

The validate step is a runtime quality gate. It takes the raw JSON string
emitted by the transform layer, parses it, and runs two ordered checks:

1. Schema conformance - each entry is validated against the
   `DictionaryEntry` pydantic model. A failure here kills the attempt
   immediately (no attribute checks run).
2. Attribute grounding - for each schema-valid entry, the `headword` (when
   present) and every element of `variants` (when present) must appear as
   an exact substring of the page's OCR text. A failure kills the attempt.

Failures are logged to the configured logfile and raise `ValidationError`.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError as PydanticValidationError

from src.common.logger import log_errors
from src.models import DictionaryEntry

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a transformed payload fails validation."""


@log_errors
def validate(raw_json: str, ocr_text: str) -> list[DictionaryEntry]:
    """Validate a transformed payload against the schema and OCR grounding.

    Args:
        raw_json: Raw JSON string returned by the transform layer, shaped
            as `{"entries": [ {DictionaryEntry}, ... ]}`.
        ocr_text: The page's OCR text (prefix-stripped) used as the ground
            truth for headword / variant substring checks.

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

    # --- 3. Schema conformance (first check) ---------------------------
    entries: list[DictionaryEntry] = []
    for i, raw_entry in enumerate(raw_entries):
        try:
            entries.append(DictionaryEntry.model_validate(raw_entry))
            logger.debug("validate: entry %d schema ok", i)
        except PydanticValidationError as e:
            logger.error(
                "validate: entry %d failed schema conformance: %s", i, e
            )
            raise ValidationError(
                f"entry {i} failed schema conformance: {e}"
            ) from e

    # --- 4. Attribute grounding (second check) -------------------------
    for i, entry in enumerate(entries):
        if entry.headword is not None and entry.headword not in ocr_text:
            logger.error(
                "validate: entry %d headword %r not found in OCR text",
                i,
                entry.headword,
            )
            raise ValidationError(
                f"entry {i} headword {entry.headword!r} not found in OCR text"
            )
        if entry.variants:
            for v in entry.variants:
                if v not in ocr_text:
                    logger.error(
                        "validate: entry %d variant %r not found in OCR text",
                        i,
                        v,
                    )
                    raise ValidationError(
                        f"entry {i} variant {v!r} not found in OCR text"
                    )
        logger.debug("validate: entry %d grounding ok", i)

    logger.info("validate: %d entries passed", len(entries))
    return entries
