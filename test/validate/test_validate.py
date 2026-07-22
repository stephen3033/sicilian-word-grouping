"""Page-envelope validation and entry-review classification tests."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image, ImageDraw

from src.common.errors import ValidationError
from src.config import Settings
from src.models import ReviewStatus
from src.validate import validate


def _image_b64(*, indented: bool = True) -> str:
    image = Image.new("RGB", (1000, 800), "white")
    draw = ImageDraw.Draw(image)
    second_left = 160 if indented else 100
    draw.rectangle((100, 50, 700, 90), fill="black")
    draw.rectangle((second_left, 120, second_left + 600, 160), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


HEADWORD_IMAGE = _image_b64(indented=True)
ORPHAN_IMAGE = _image_b64(indented=False)
OCR = "a2 art. femm. la. V. anche la1. Cfr. u2."


def _entry(
    *,
    headword: str | None = "a²",
    trailing_text: str = "art. femm. la.",
    variants: list[str] | None = None,
) -> dict:
    return {
        "headword": headword,
        "trailing_text": trailing_text,
        "variants": variants,
    }


def _payload(*entries: dict) -> str:
    return json.dumps({"entries": list(entries)})


def test_no_findings_passes_with_empty_reason_and_injected_metadata():
    result = validate(_payload(_entry()), OCR, HEADWORD_IMAGE, 42)
    assert result[0].model_dump(mode="json") == {
        "headword": "a²",
        "trailing_text": "art. femm. la.",
        "variants": None,
        "page_numbers": [42],
        "vs_vol": 1,
        "is_review_needed": "passed",
        "review_reason": "",
    }


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        (_entry(headword="missing"), "headword 'missing' not found"),
        (_entry(variants=["missing"]), "variant 'missing' not found"),
        (_entry(trailing_text="missing body"), "trailing_text 'missing body' not found"),
    ],
)
def test_exactly_one_text_finding_is_machine_review(entry, reason):
    result = validate(_payload(entry), OCR, HEADWORD_IMAGE, 1)[0]
    assert result.is_review_needed is ReviewStatus.MACHINE
    assert reason in result.review_reason


def test_two_text_findings_are_human_and_each_invalid_variant_counts():
    result = validate(
        _payload(_entry(variants=["missing-one", "missing-two"])),
        OCR,
        HEADWORD_IMAGE,
        1,
    )[0]
    assert result.is_review_needed is ReviewStatus.HUMAN
    assert result.review_reason.splitlines() == [
        "entry 0 variant 'missing-one' not found in OCR text",
        "entry 0 variant 'missing-two' not found in OCR text",
    ]


def test_findings_follow_layout_headword_variant_trailing_validator_order():
    result = validate(
        _payload(
            _entry(
                headword="bad, combined",
                variants=["variant-one", "variant-two"],
                trailing_text="bad trailing",
            )
        ),
        OCR,
        ORPHAN_IMAGE,
        1,
    )[0]
    reasons = result.review_reason.splitlines()
    assert reasons[0].startswith("AI extraction mismatch.")
    assert "headword 'bad, combined' contains combined alternatives" in reasons[1]
    assert "variant 'variant-one' not found" in reasons[2]
    assert "variant 'variant-two' not found" in reasons[3]
    assert "trailing_text 'bad trailing' not found" in reasons[4]
    assert result.is_review_needed is ReviewStatus.HUMAN


def test_layout_aligned_orphan_is_preserved_without_a_finding():
    text = "continuation from the prior page"
    result = validate(
        _payload(_entry(headword=None, trailing_text=text)),
        text,
        ORPHAN_IMAGE,
        9,
    )[0]
    assert result.headword is None
    assert result.trailing_text == text
    assert result.is_review_needed is ReviewStatus.PASSED
    assert result.review_reason == ""


def test_layout_mismatch_is_human_even_without_text_findings():
    result = validate(_payload(_entry()), OCR, ORPHAN_IMAGE, 1)[0]
    assert result.is_review_needed is ReviewStatus.HUMAN
    assert result.review_reason.startswith("AI extraction mismatch.")


def test_layout_analysis_failure_is_human_not_page_failure():
    invalid_image = base64.b64encode(b"not a png").decode("ascii")
    result = validate(_payload(_entry()), OCR, invalid_image, 1)[0]
    assert result.is_review_needed is ReviewStatus.HUMAN
    assert "Structural layout parsing failure" in result.review_reason


def test_quality_findings_never_change_extracted_text():
    raw_entry = _entry(
        headword="missing",
        trailing_text="missing trailing",
        variants=["missing variant"],
    )
    result = validate(_payload(raw_entry), OCR, HEADWORD_IMAGE, 1)[0]
    assert result.headword == raw_entry["headword"]
    assert result.trailing_text == raw_entry["trailing_text"]
    assert result.variants == raw_entry["variants"]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{not json", "not valid JSON"),
        (json.dumps({"words": []}), "missing top-level 'entries'"),
        (json.dumps({"entries": {}, "other": 1}), "invalid top-level keys"),
        (json.dumps({"entries": {}}), "'entries' is not a list"),
    ],
)
def test_invalid_page_envelopes_fail_the_whole_page(raw, message):
    with pytest.raises(ValidationError, match=message):
        validate(raw, OCR, HEADWORD_IMAGE, 1)


def test_any_llm_entry_schema_failure_excludes_the_whole_page():
    bad = {"headword": "b", "trailing_text": "body"}
    with pytest.raises(ValidationError, match="entry 1 failed schema conformance"):
        validate(_payload(_entry(), bad), OCR, HEADWORD_IMAGE, 1)


def test_deterministic_placeholders_are_rejected_as_extra_llm_fields():
    with pytest.raises(ValidationError, match="schema conformance"):
        validate(
            _payload({**_entry(), "page_numbers": [0], "vs_vol": 0}),
            OCR,
            HEADWORD_IMAGE,
            1,
        )


@pytest.mark.parametrize("ocr", ["", "   ", None])
def test_missing_ocr_context_fails_the_page(ocr):
    with pytest.raises(ValidationError, match="normalized_ocr context required"):
        validate(_payload(_entry()), ocr, HEADWORD_IMAGE, 1)


def test_volume_is_injected_from_settings():
    settings = Settings(volume=4)
    result = validate(_payload(_entry()), OCR, HEADWORD_IMAGE, 3, settings)[0]
    assert result.vs_vol == 4
    assert result.page_numbers == [3]
