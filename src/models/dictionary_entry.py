import logging

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from src.common.errors import ValidationError
from src.common.layout import analyze_first_span, convert_image_to_layout_data
from src.common.normalize import normalize

logger = logging.getLogger(__name__)


def _require_normalized_ocr(info: ValidationInfo) -> str:
    """Return the pre-normalized OCR text from the validation context.

    Raises `ValidationError` if missing, absent, or whitespace-only.
    """
    ctx = info.context or {}
    normalized_ocr = ctx.get("normalized_ocr")
    if not isinstance(normalized_ocr, str) or not normalized_ocr.strip():
        raise ValidationError(
            "normalized_ocr context required for grounding (missing, empty, "
            "or whitespace-only)"
        )
    return normalized_ocr


def _entry_prefix(info: ValidationInfo) -> str:
    """Build the `entry {i} ` prefix for grounding error messages."""
    ctx = info.context or {}
    index = ctx.get("index")
    return f"entry {index} " if index is not None else ""


class DictionaryEntry(BaseModel):
    headword: str | None = Field(
        None,
        description="The canonical lemma/headword; None when a page begins mid-definition.",
    )
    trailing_text: str = Field(
        ...,
        description="Body text following the headword line: definitions, citations, sub-senses.",
    )
    variants: list[str] | None = Field(
        None,
        description="Alternate spellings / dialectal variants explicitly listed in the entry.",
    )
    page_numbers: list[int] = Field(
        description="Printed page number(s) this entry's text was drawn from; multi-element if it spans pages.",
    )
    vs_vol: int = Field(
        description="VS volume number this entry was drawn from; placeholder "
        "0 emitted by the model, real value injected programmatically downstream.",
    )

    @model_validator(mode="after")
    def validate_layout_alignment(self, info: ValidationInfo) -> "DictionaryEntry":
        """Verify the headword's null-state against the page's physical layout.

        Only the first entry on a page can be an orphan, so the check is
        gated on ``context["index"] == 0`` and an ``image_payload`` (raw PNG
        bytes) in the context; otherwise no-ops so standalone
        ``model_validate`` calls without image bytes still work.

        Heuristic on the first two real text lines (pixel-based; VS PDFs are
        rasterized scans with no text layer):

        - ``|Δx| > (headword_delta - tolerance)`` -> headword present ->
          expected ``self.headword is not None``.
        - ``|Δx| ≤ (headword_delta - tolerance)`` -> lines aligned ->
          continuation -> expected ``self.headword is None``.

        `headword_delta` and `tolerance` come from `Settings` via the
        validation context. Disagreement raises `ValueError`.
        """
        ctx = info.context or {}
        if "image_payload" not in ctx:
            return self
        if ctx.get("index", 0) != 0:
            return self

        payload = ctx["image_payload"]
        delta = ctx.get("headword_delta", 36.0)
        tolerance = ctx.get("tolerance", 15.0)

        try:
            layout_data = convert_image_to_layout_data(payload)
            is_indented = analyze_first_span(layout_data, delta, tolerance)
            expected_is_orphan = not is_indented
            if (self.headword is None) != expected_is_orphan:
                raise ValueError(
                    f"AI extraction mismatch. Model flagged page start with "
                    f"headword={'None' if self.headword is None else repr(self.headword)}, "
                    f"but physical layout heuristics expected "
                    f"{'headword=None (orphan)' if expected_is_orphan else 'headword present'} "
                    f"(|Δx| threshold={delta - tolerance} from headword_delta={delta} - "
                    f"tolerance={tolerance})."
                )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Structural layout parsing failure: {e}")

        return self

    @model_validator(mode="after")
    def _validate_headword(self, info: ValidationInfo) -> "DictionaryEntry":
        """Ground `headword` against the normalized OCR text; skip if None."""
        normalized_ocr = _require_normalized_ocr(info)
        if self.headword is None:
            return self
        if normalize(self.headword) not in normalized_ocr:
            raise ValidationError(
                f"{_entry_prefix(info)}headword {self.headword!r} "
                "not found in OCR text"
            )
        return self

    @model_validator(mode="after")
    def _validate_variants(self, info: ValidationInfo) -> "DictionaryEntry":
        """Ground each element of `variants` against the normalized OCR text."""
        normalized_ocr = _require_normalized_ocr(info)
        if not self.variants:
            return self
        for v in self.variants:
            if normalize(v) not in normalized_ocr:
                raise ValidationError(
                    f"{_entry_prefix(info)}variant {v!r} not found in OCR text"
                )
        return self

    @model_validator(mode="after")
    def _validate_trailing_text(
        self, info: ValidationInfo
    ) -> "DictionaryEntry":
        """Ground `trailing_text` against the normalized OCR text."""
        normalized_ocr = _require_normalized_ocr(info)
        if normalize(self.trailing_text) not in normalized_ocr:
            raise ValidationError(
                f"{_entry_prefix(info)}trailing_text {self.trailing_text!r} "
                "not found in OCR text"
            )
        return self
