import io
import logging

from PIL import Image, ImageChops
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from src.common.errors import ValidationError
from src.common.normalize import normalization

logger = logging.getLogger(__name__)


def _convert_image_to_layout_data(image_payload: bytes) -> dict:
    """Decode PNG bytes and extract text-line bboxes from the rendered page.

    Returns: ``{"lines": [{"top", "bottom", "left_x", "width", "height"}, ...]}``
    in pixel coordinates. Lines are groups of vertically-contiguous inked
    rows (gap <= 4 white rows tolerated); bands shorter than 10px are
    dropped as stray marks. The image is binarized at grayscale threshold
    128 before row-ink detection.

    The VS PDFs are rasterized scans with no text layer, so layout
    extraction is purely pixel-based. The page image is the composite PNG
    produced by ``src.extract.pdf_extractor.extract_page_image`` (two
    columns stitched, rendered at ``Settings.image_dpi``).
    """
    img = Image.open(io.BytesIO(image_payload)).convert("L")
    bw = img.point(lambda p: 0 if p < 128 else 255)
    inv = ImageChops.invert(bw)
    width, height = bw.size
    px = bw.load()
    # Sample every 4th column for speed; row-ink detection does not need
    # full horizontal resolution.
    row_ink = [
        any(px[x, y] == 0 for x in range(0, width, 4)) for y in range(height)
    ]

    lines: list[tuple[int, int]] = []
    gap_tol = 4
    min_height = 10
    y = 0
    while y < height:
        if row_ink[y]:
            start = y
            gap = 0
            while y < height and gap <= gap_tol:
                if row_ink[y]:
                    gap = 0
                else:
                    gap += 1
                y += 1
            end = y - gap
            if end - start >= min_height:
                lines.append((start, end))
        else:
            y += 1

    out_lines = []
    for top, bottom in lines:
        bbox = inv.crop((0, top, width, bottom)).getbbox()
        if bbox is None:
            continue
        left, _, right, _ = bbox
        out_lines.append(
            {
                "top": top,
                "bottom": bottom,
                "left_x": left,
                "width": right - left,
                "height": bottom - top,
            }
        )
    return {"lines": out_lines, "page_width": width}


def _analyze_first_span(
    layout_data: dict, delta: float, tolerance: float
) -> tuple[bool, bool]:
    """Compare the first two real text lines and return ``(is_bold, is_indented)``.

    ``is_indented`` is ``True`` when the left X of the first two text lines
    (each wider than 30px to drop marginalia/noise) differ by more than
    ``delta - tolerance`` pixels. ``delta`` is the calibrated min |Δx|
    observed on headword pages (200 DPI); ``tolerance`` is the fudge factor
    subtracted from it to account for scan skew/tilt. This relative
    comparison (rather than an absolute X coordinate) accommodates the
    varying scan margins across VS volumes: a headword page has an
    outdented headword token whose X differs from the body's X by
    ~36-60px at 200 DPI, while an orphan-continuation page has all lines
    aligned within ~5px.

    ``is_bold`` is always ``True`` (no-op axis). Bold detection via ink
    density was empirically unreliable at 200 DPI for the VS font (bold
    density 0.15-0.26 overlaps body 0.12-0.18); the indent signal alone
    correctly classifies all 32 validated pages. The axis is retained for
    signature parity with the reference template's truth table.
    """
    lines = layout_data.get("lines", [])
    real = [ln for ln in lines if ln["width"] > 30]
    if len(real) < 2:
        return True, False
    threshold = delta - tolerance
    is_indented = abs(real[0]["left_x"] - real[1]["left_x"]) > threshold
    return True, is_indented


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
    trailing_text: str = Field(
        ...,
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

    @field_validator("is_orphan_fragment")
    @classmethod
    def validate_layout_alignment(cls, v: bool, info: ValidationInfo) -> bool:
        """Verify ``is_orphan_fragment`` against the page's physical layout.

        Only the first entry on a page can be a true orphan fragment, so the
        check is gated on ``context["index"] == 0`` and the presence of an
        ``image_payload`` (raw PNG bytes) in the validation context. When
        either is missing, the validator no-ops (returns ``v`` unchanged) so
        standalone ``model_validate`` calls without image bytes continue to
        work.

        The heuristic compares the left X of the first two real text lines
        on the rendered page image (pixel-based; the VS PDFs are rasterized
        scans with no text layer):

        - ``|Δx| > (headword_delta - tolerance)`` -> lines at different X ->
          a headword token is present -> expected ``is_orphan_fragment = False``.
        - ``|Δx| ≤ (headword_delta - tolerance)`` -> lines aligned ->
          continuation/orphan -> expected ``is_orphan_fragment = True``.

        ``headword_delta`` is the calibrated min |Δx| observed on headword
        pages (200 DPI); ``tolerance`` is the fudge factor subtracted from
        it to account for scan skew/tilt. Both are sourced from
        ``Settings`` and threaded through the validation context.

        If the AI-supplied value disagrees with the layout-derived
        expectation, a ``ValueError`` is raised detailing the mismatch. The
        bold axis from the reference template is dropped (ink density is
        unreliable at 200 DPI for the VS font); see
        ``_analyze_first_span`` for the empirical justification.
        """
        ctx = info.context or {}
        if "image_payload" not in ctx:
            return v
        if ctx.get("index", 0) != 0:
            return v

        payload = ctx["image_payload"]
        delta = ctx.get("headword_delta", 36.0)
        tolerance = ctx.get("tolerance", 15.0)

        try:
            layout_data = _convert_image_to_layout_data(payload)
            _, is_indented = _analyze_first_span(layout_data, delta, tolerance)

            if is_indented:
                expected_is_orphan = False
            else:
                expected_is_orphan = True

            if v != expected_is_orphan:
                raise ValueError(
                    f"AI extraction mismatch. Model flagged page start as "
                    f"is_orphan_fragment={v}, but physical layout heuristics "
                    f"expected={expected_is_orphan} (|Δx| threshold="
                    f"{delta - tolerance} from headword_delta={delta} - "
                    f"tolerance={tolerance})."
                )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Structural layout parsing failure: {e}")

        return v

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
        missing, empty, or whitespace-only. `trailing_text` is required;
        a missing or null value is rejected by field-level validation
        before this `mode="after"` check runs.
        """
        normalized_ocr = _require_normalized_ocr(info)
        if normalization(self.trailing_text) not in normalized_ocr:
            raise ValidationError(
                f"{_entry_prefix(info)}trailing_text {self.trailing_text!r} "
                "not found in OCR text"
            )
        return self
