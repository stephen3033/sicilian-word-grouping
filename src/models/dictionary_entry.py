"""Dictionary-entry schemas and deterministic review annotation."""

from __future__ import annotations

import logging
import re
import unicodedata
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.common.layout import analyze_first_span, convert_image_to_layout_data
from src.common.normalize import (
    HEADWORD_FUZZY_CUTOFF,
    best_token_ratio,
    is_hyphen_prefix_match,
    is_standalone_token,
    normalize,
    strip_trailing_digits,
    token_overlap_ratio,
)

logger = logging.getLogger(__name__)

_CROSS_REFERENCE_PROSE_RE = re.compile(
    r"(?:^|\s)(?:v\.|V\.|Cfr\.|Anche|anche)(?:\s|$)"
)
_EDGE_PUNCTUATION = ".,;:!?\"“”‘’"
_ROMAN_ARABIC_ONE_RE = re.compile(r"(?<!\w)[I1](?!\w)")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")


class ReviewStatus(str, Enum):
    """Disposition assigned by the deterministic entry-quality checks."""

    PASSED = "passed"
    MACHINE = "machine"
    HUMAN = "human"


class LLMEntry(BaseModel):
    """The complete and only schema emitted by the language model."""

    model_config = ConfigDict(extra="forbid")

    headword: str | None = Field(
        ...,
        description=(
            "The canonical lemma/headword; null only when a page begins "
            "mid-definition."
        ),
    )
    trailing_text: str = Field(
        ...,
        description=(
            "Body text following the headword: definitions, citations, and "
            "sub-senses."
        ),
    )
    variants: list[str] | None = Field(
        ...,
        description=(
            "Alternate spellings or dialectal variants explicitly listed in "
            "the entry."
        ),
    )


class DictionaryEntry(LLMEntry):
    """A flat persisted entry with pipeline-owned provenance and review data."""

    page_numbers: list[int] = Field(
        ...,
        description=(
            "Printed page number(s) from which this entry's text was drawn."
        ),
    )
    vs_vol: int = Field(
        ...,
        description="VS volume number from which this entry was drawn.",
    )
    is_review_needed: ReviewStatus = Field(
        ...,
        description="Deterministic review disposition: passed, machine, or human.",
    )
    review_reason: str = Field(
        ...,
        description="Newline-delimited quality findings in validator order.",
    )


def _prefix(index: int | None) -> str:
    return f"entry {index} " if index is not None else ""


def _atomic_headword_finding(value: str, index: int | None) -> str | None:
    """Return the existing atomic-headword error message, if applicable."""
    if "," in value or ";" in value:
        reason = "combined alternatives"
    elif "(" in value or ")" in value:
        reason = "locality/source parentheses"
    elif _CROSS_REFERENCE_PROSE_RE.search(value):
        reason = "cross-reference prose"
    else:
        return None
    return f"{_prefix(index)}headword {value!r} contains {reason}"


def _hyphen_mismatch_token(value: str, normalized_ocr: str) -> str | None:
    """Find an otherwise exact OCR token with different hyphen placement."""
    value_token = normalize(value).strip(_EDGE_PUNCTUATION)
    value_without_hyphens = value_token.replace("-", "")
    if not value_without_hyphens:
        return None

    for raw_token in normalized_ocr.split():
        ocr_token = raw_token.strip(_EDGE_PUNCTUATION)
        if "-" not in ocr_token:
            continue
        if ocr_token.endswith("-") and ocr_token.count("-") == 1:
            continue
        if (
            ocr_token != value_token
            and ocr_token.replace("-", "") == value_without_hyphens
        ):
            return ocr_token
    return None


def _normalized_ocr_tokens(normalized_ocr: str) -> set[str]:
    return {
        raw_token.strip(_EDGE_PUNCTUATION)
        for raw_token in normalized_ocr.split()
        if raw_token.strip(_EDGE_PUNCTUATION)
    }


def _collapsed_doubled_dotted_d_token(
    value: str, normalized_ocr: str
) -> str | None:
    value_token = normalize(value).strip(_EDGE_PUNCTUATION)
    for ocr_token in _normalized_ocr_tokens(normalized_ocr):
        if "ḍḍ" in ocr_token and ocr_token.replace("ḍḍ", "ḍ") == value_token:
            return ocr_token
    return None


def _omitted_homonym_digit_token(
    value: str, normalized_ocr: str
) -> str | None:
    value_token = normalize(value).strip(_EDGE_PUNCTUATION)
    if not value_token or _TRAILING_DIGITS_RE.search(value_token):
        return None
    ocr_tokens = _normalized_ocr_tokens(normalized_ocr)
    if value_token in ocr_tokens:
        return None
    for ocr_token in ocr_tokens:
        if (
            _TRAILING_DIGITS_RE.search(ocr_token)
            and _TRAILING_DIGITS_RE.sub("", ocr_token) == value_token
        ):
            return ocr_token
    return None


def _numeral_skeleton(value: str) -> str:
    return _ROMAN_ARABIC_ONE_RE.sub("#", value)


def _has_roman_arabic_one_substitution(
    normalized_value: str, normalized_ocr: str
) -> bool:
    numeral_positions = [
        match.start() for match in _ROMAN_ARABIC_ONE_RE.finditer(normalized_value)
    ]
    if not numeral_positions:
        return False

    value_skeleton = _numeral_skeleton(normalized_value)
    ocr_skeleton = _numeral_skeleton(normalized_ocr)
    start = ocr_skeleton.find(value_skeleton)
    while start != -1:
        if any(
            normalized_value[position] != normalized_ocr[start + position]
            for position in numeral_positions
        ):
            return True
        start = ocr_skeleton.find(value_skeleton, start + 1)
    return False


def _has_compatible_ocr_diacritic_loss(
    normalized_value: str, normalized_ocr: str
) -> bool:
    parts: list[str] = []
    has_diacritic = False
    for character in normalized_value:
        decomposed = unicodedata.normalize("NFD", character)
        if len(decomposed) > 1 and any(
            unicodedata.combining(mark) for mark in decomposed[1:]
        ):
            has_diacritic = True
            parts.append(
                f"(?:{re.escape(character)}|{re.escape(decomposed[0])})"
            )
        else:
            parts.append(re.escape(character))
    if not has_diacritic:
        return False
    return re.search("".join(parts), normalized_ocr) is not None


def _short_field_finding(
    value: str,
    field_label: str,
    normalized_ocr: str,
    index: int | None,
) -> str | None:
    """Return the first finding from the established short-field gate."""
    normalized_value = normalize(value)
    ocr_token = _hyphen_mismatch_token(value, normalized_ocr)
    if ocr_token is not None:
        return (
            f"{_prefix(index)}{field_label} {value!r} has omitted or "
            f"displaced hyphen; OCR contains {ocr_token!r}"
        )

    doubled_token = _collapsed_doubled_dotted_d_token(value, normalized_ocr)
    if doubled_token is not None:
        return (
            f"{_prefix(index)}{field_label} {value!r} collapses printed "
            f"'ḍḍ' to 'ḍ'; OCR contains {doubled_token!r}"
        )

    numbered_token = _omitted_homonym_digit_token(value, normalized_ocr)
    if numbered_token is not None:
        return (
            f"{_prefix(index)}{field_label} {value!r} omits attached "
            f"homonym numeral; OCR contains {numbered_token!r}"
        )

    if normalized_value in normalized_ocr:
        return None

    root = strip_trailing_digits(normalized_value)
    if root and root != normalized_value and is_standalone_token(root, normalized_ocr):
        logger.debug(
            "%s%s root-token-accepted (value=%r root=%r)",
            _prefix(index),
            field_label,
            value,
            root,
        )
        return None

    if is_hyphen_prefix_match(normalized_value, normalized_ocr):
        logger.debug(
            "%s%s hyphen-prefix-accepted (value=%r)",
            _prefix(index),
            field_label,
            value,
        )
        return None

    ratio = best_token_ratio(normalized_value, normalized_ocr)
    if ratio >= HEADWORD_FUZZY_CUTOFF:
        logger.debug(
            "%s%s fuzzy-token-accepted (value=%r ratio=%.2f)",
            _prefix(index),
            field_label,
            value,
            ratio,
        )
        return None

    return f"{_prefix(index)}{field_label} {value!r} not found in OCR text"


def _trailing_text_finding(
    value: str,
    normalized_ocr: str,
    index: int | None,
    *,
    grounding_min_tokens: int,
    grounding_threshold: float,
) -> str | None:
    normalized_trailing = normalize(value)
    if normalized_trailing in normalized_ocr:
        return None

    if _has_roman_arabic_one_substitution(normalized_trailing, normalized_ocr):
        return (
            f"{_prefix(index)}trailing_text contains Roman I/Arabic 1 "
            "substitution"
        )

    if _has_compatible_ocr_diacritic_loss(normalized_trailing, normalized_ocr):
        logger.debug(
            "%strailing_text compatible OCR-diacritic-loss accepted",
            _prefix(index),
        )
        return None

    token_count = len(normalized_trailing.split())
    if token_count >= grounding_min_tokens:
        ratio = token_overlap_ratio(normalized_ocr, normalized_trailing)
        if ratio >= grounding_threshold:
            logger.debug(
                "%strailing_text fuzzy-accepted (tokens=%d ratio=%.4f "
                "threshold=%.2f)",
                _prefix(index),
                token_count,
                ratio,
                grounding_threshold,
            )
            return None

    return f"{_prefix(index)}trailing_text {value!r} not found in OCR text"


def annotate_entry(
    entry: LLMEntry,
    *,
    index: int,
    page_number: int,
    volume: int,
    normalized_ocr: str,
    image_payload: bytes,
    headword_delta: float,
    layout_tolerance: float,
    grounding_threshold: float,
    grounding_min_tokens: int,
    layout_error: str | None = None,
) -> DictionaryEntry:
    """Collect all quality findings and return an unchanged annotated entry.

    Finding order is layout, headword, variants in source order, then
    trailing text. Text-only findings are machine-reviewable when exactly one
    is present and human-reviewable when two or more are present. Any
    orphan/layout mismatch or layout-analysis finding promotes the entry
    directly to human review.
    """
    findings: list[str] = []
    has_layout_finding = False

    if index == 0:
        if layout_error is not None:
            findings.append(layout_error)
            has_layout_finding = True
        else:
            try:
                layout_data = convert_image_to_layout_data(image_payload)
                is_indented = analyze_first_span(
                    layout_data, headword_delta, layout_tolerance
                )
                expected_is_orphan = not is_indented
                if (entry.headword is None) != expected_is_orphan:
                    findings.append(
                        "AI extraction mismatch. Model flagged page start with "
                        f"headword={'None' if entry.headword is None else repr(entry.headword)}, "
                        "but physical layout heuristics expected "
                        f"{'headword=None (orphan)' if expected_is_orphan else 'headword present'} "
                        f"(|Δx| threshold={headword_delta - layout_tolerance} "
                        f"from headword_delta={headword_delta} - "
                        f"tolerance={layout_tolerance})."
                    )
                    has_layout_finding = True
            except Exception as exc:
                findings.append(f"Structural layout parsing failure: {exc}")
                has_layout_finding = True

    text_findings: list[str] = []
    if entry.headword is not None:
        headword_finding = _atomic_headword_finding(entry.headword, index)
        if headword_finding is None:
            headword_finding = _short_field_finding(
                entry.headword, "headword", normalized_ocr, index
            )
        if headword_finding is not None:
            text_findings.append(headword_finding)

    if entry.variants:
        for variant in entry.variants:
            finding = _short_field_finding(
                variant, "variant", normalized_ocr, index
            )
            if finding is not None:
                text_findings.append(finding)

    trailing_finding = _trailing_text_finding(
        entry.trailing_text,
        normalized_ocr,
        index,
        grounding_min_tokens=grounding_min_tokens,
        grounding_threshold=grounding_threshold,
    )
    if trailing_finding is not None:
        text_findings.append(trailing_finding)

    findings.extend(text_findings)
    if has_layout_finding or len(text_findings) >= 2:
        status = ReviewStatus.HUMAN
    elif len(text_findings) == 1:
        status = ReviewStatus.MACHINE
    else:
        status = ReviewStatus.PASSED

    return DictionaryEntry(
        **entry.model_dump(),
        page_numbers=[page_number],
        vs_vol=volume,
        is_review_needed=status,
        review_reason="\n".join(findings),
    )
