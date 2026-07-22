"""Parse model output, enforce its schema, and annotate entry quality."""

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
from src.models import DictionaryEntry, LLMEntry, ReviewStatus, annotate_entry

logger = logging.getLogger(__name__)


@log_errors
def validate(
    raw_json: str,
    ocr_text: str,
    image_b64: str,
    page_number: int,
    settings: Settings | None = None,
) -> list[DictionaryEntry]:
    """Validate a complete page envelope, then annotate every entry.

    Malformed JSON/envelopes, any ``LLMEntry`` schema failure, and missing OCR
    context fail the whole page. OCR grounding and layout findings never alter
    model text and never fail the page; they set the persisted review fields.
    Schema validation is completed for the full list before annotation starts,
    so a page can never return a partial result.
    """
    if settings is None:
        settings = get_settings()
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        raise ValidationError(
            "normalized_ocr context required for grounding (missing, empty, "
            "or whitespace-only)"
        )

    logger.debug(
        "raw_json=%d chars ocr=%d chars image=%d b64 page=%d",
        len(raw_json),
        len(ocr_text),
        len(image_b64),
        page_number,
    )

    try:
        payload = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValidationError(f"raw payload is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or "entries" not in payload:
        raise ValidationError("payload missing top-level 'entries' key")
    if set(payload) != {"entries"}:
        extra_keys = sorted(str(key) for key in set(payload) - {"entries"})
        raise ValidationError(
            f"payload has invalid top-level keys: {', '.join(extra_keys)}"
        )
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValidationError(
            f"'entries' is not a list (got {type(raw_entries).__name__})"
        )
    logger.debug("unwrapped %d entries", len(raw_entries))

    # Finish schema validation for every item before creating any persisted
    # entry. A bad item therefore excludes the complete page.
    llm_entries: list[LLMEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        try:
            llm_entries.append(LLMEntry.model_validate(raw_entry))
        except PydanticValidationError as exc:
            raise ValidationError(
                f"entry {index} failed schema conformance: {exc}"
            ) from exc

    normalized_ocr = normalize(ocr_text)
    layout_error: str | None = None
    try:
        image_payload = base64.b64decode(image_b64, validate=True)
    except Exception as exc:
        image_payload = b""
        layout_error = f"Structural layout parsing failure: {exc}"

    entries = [
        annotate_entry(
            entry,
            index=index,
            page_number=page_number,
            volume=settings.volume,
            normalized_ocr=normalized_ocr,
            image_payload=image_payload,
            headword_delta=settings.headword_delta,
            layout_tolerance=settings.layout_tolerance,
            grounding_threshold=settings.grounding_threshold,
            grounding_min_tokens=settings.grounding_min_tokens,
            layout_error=layout_error if index == 0 else None,
        )
        for index, entry in enumerate(llm_entries)
    ]

    counts = {status: 0 for status in ReviewStatus}
    for index, entry in enumerate(entries):
        counts[entry.is_review_needed] += 1
        logger.debug(
            "entry %d annotated status=%s findings=%d (vs_vol=%d page=%d)",
            index,
            entry.is_review_needed.value,
            0 if not entry.review_reason else len(entry.review_reason.splitlines()),
            entry.vs_vol,
            page_number,
        )
    logger.info(
        "%d entries accepted (page=%d passed=%d machine=%d human=%d)",
        len(entries),
        page_number,
        counts[ReviewStatus.PASSED],
        counts[ReviewStatus.MACHINE],
        counts[ReviewStatus.HUMAN],
    )
    return entries


@log_errors
def persist_validated_page(
    entries: list[DictionaryEntry], page_number: int, settings: Settings
) -> Path:
    """Write one annotated page artifact in debug mode."""
    out_path = settings.validated_page_path(page_number, settings.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": [entry.model_dump(mode="json") for entry in entries]
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.debug("wrote %d entries to %s", len(entries), out_path)
    return out_path
