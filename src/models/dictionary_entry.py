import io
import logging

from PIL import Image, ImageChops
from pydantic import BaseModel, Field, ValidationInfo, model_validator

from src.common.errors import ValidationError
from src.common.normalize import normalization

logger = logging.getLogger(__name__)


def _convert_image_to_layout_data(image_payload: bytes) -> dict:
    """Decode PNG bytes and extract text-line bboxes from the rendered page.

    Returns ``{"lines": [...], "page_width": w}`` in pixel coordinates. Lines
    are groups of vertically-contiguous inked rows (gap <= 4 white rows
    tolerated); bands < 10px are dropped. Binarized at grayscale threshold
    128; every 4th column is sampled for row-ink detection.
    """
    img = Image.open(io.BytesIO(image_payload)).convert("L")
    bw = img.point(lambda p: 0 if p < 128 else 255)
    inv = ImageChops.invert(bw)
    width, height = bw.size
    px = bw.load()
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
    """Compare the first two real text lines; return ``(is_bold, is_indented)``.

    Only lines wider than 30px (drops marginalia) are compared. `is_indented`
    is True when their left X differs by more than `delta - tolerance` px.
    `is_bold` is always True: bold detection via ink density was unreliable
    at 200 DPI for the VS font; the indent axis alone classifies all
    validated pages. Retained for truth-table signature parity.
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
            layout_data = _convert_image_to_layout_data(payload)
            _, is_indented = _analyze_first_span(layout_data, delta, tolerance)
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
        if normalization(self.headword) not in normalized_ocr:
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
            if normalization(v) not in normalized_ocr:
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
        if normalization(self.trailing_text) not in normalized_ocr:
            raise ValidationError(
                f"{_entry_prefix(info)}trailing_text {self.trailing_text!r} "
                "not found in OCR text"
            )
        return self
