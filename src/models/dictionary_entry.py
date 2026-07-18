from pydantic import BaseModel, Field, ValidationInfo, model_validator

from src.common.errors import ValidationError
from src.common.normalize import normalization


def _require_normalized_ocr(info: ValidationInfo) -> str:
    """Return the pre-normalized OCR text from the validation context.

    A grounding check with nothing to ground against is a pipeline setup
    error, not a silent skip: raises `ValidationError` if the context is
    missing, if `normalized_ocr` is absent, or if it is empty or
    whitespace-only.
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
    """Build the `entry {i} ` prefix for grounding error messages.

    The index is threaded in via the validation `context`; when absent
    (e.g. a standalone `model_validate` call that did supply OCR), the
    prefix is empty so the message just describes the failed field.
    """
    ctx = info.context or {}
    index = ctx.get("index")
    return f"entry {index} " if index is not None else ""


class DictionaryEntry(BaseModel):
    headword: str | None = Field(
        None,
        description=(
            "The canonical lemma/headword of this dictionary entry. None when "
            "a page begins mid-definition (continuation of the previous page's "
            "final entry)."
        ),
    )
    trailing_text: str | None = Field(
        None,
        description=(
            "Remaining body text of the entry that follows the headword line "
            "- definitions, citations, sub-senses."
        ),
    )
    variants: list[str] | None = Field(
        None,
        description=(
            "Alternate spellings / dialectal variants of the headword "
            "explicitly listed in the entry."
        ),
    )
    page_numbers: list[int] = Field(
        description=(
            "Printed page number(s) this entry's text was drawn from. "
            "Multi-element when a single headword spans consecutive pages."
        ),
    )
    is_orphan_fragment: bool = Field(
        False,
        description=(
            "True when this text is a continuation of the previous page's "
            "final headword (no new headword begins on this page). False when "
            "the text starts a fresh entry."
        ),
    )

    @model_validator(mode="after")
    def _validate_headword(self, info: ValidationInfo) -> "DictionaryEntry":
        """Ground `headword` against the normalized OCR text.

        Raises `ValidationError` if the `normalized_ocr` context is
        missing, empty, or whitespace-only. Skips silently when
        `headword` is None (orphan fragment).
        """
        normalized_ocr = _require_normalized_ocr(info)
        if self.headword is None:
            return self
        if normalization(self.headword) not in normalized_ocr:
            raise ValidationError(
                f"{_entry_prefix(info)}headword {self.headword!r} "
                "not found in OCR text"
            )
        return self

    @model_validator(mode="after")
    def _validate_variants(self, info: ValidationInfo) -> "DictionaryEntry":
        """Ground each element of `variants` against the normalized OCR text.

        Raises `ValidationError` if the `normalized_ocr` context is
        missing, empty, or whitespace-only. Skips silently when
        `variants` is None or empty (falsy).
        """
        normalized_ocr = _require_normalized_ocr(info)
        if not self.variants:
            return self
        for v in self.variants:
            if normalization(v) not in normalized_ocr:
                raise ValidationError(
                    f"{_entry_prefix(info)}variant {v!r} not found in OCR text"
                )
        return self

    @model_validator(mode="after")
    def _validate_trailing_text(
        self, info: ValidationInfo
    ) -> "DictionaryEntry":
        """Ground `trailing_text` against the normalized OCR text.

        The page's OCR text (pre-normalized once by the validate stage)
        and the entry's position in the payload are threaded in via the
        validation `context`. `trailing_text` itself passes through
        `normalization()` before the verbatim substring (`in`) check so
        that layout whitespace and Unicode representation differences do
        not defeat grounding.

        Raises `ValidationError` if the `normalized_ocr` context is
        missing, empty, or whitespace-only. Skips silently when
        `trailing_text` is None.
        """
        normalized_ocr = _require_normalized_ocr(info)
        if self.trailing_text is None:
            return self
        if normalization(self.trailing_text) not in normalized_ocr:
            raise ValidationError(
                f"{_entry_prefix(info)}trailing_text {self.trailing_text!r} "
                "not found in OCR text"
            )
        return self
