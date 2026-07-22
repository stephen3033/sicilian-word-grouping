"""Unit tests for src.common.normalize (OCR normalization + fuzzy overlap)."""

from __future__ import annotations

import pytest

from src.common.normalize import (
    HEADWORD_FUZZY_CUTOFF,
    MIN_HYPHEN_PREFIX_LEN,
    best_token_ratio,
    is_hyphen_prefix_match,
    is_standalone_token,
    normalize,
    strip_trailing_digits,
    token_overlap_ratio,
)


class TestNormalizeSuperscriptFold:
    """Superscript digits fold to ASCII digits; regular digits unchanged."""

    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            ("u²", "u2"),
            ("abbachiari²", "abbachiari2"),
            ("abbaḍḍuliari²", "abbaḍḍuliari2"),
            ("a¹ a² a³", "a1 a2 a3"),
            ("⁴⁵⁶⁷⁸⁹⁰", "4567890"),
            ("abbacari2", "abbacari2"),
            ("plain text", "plain text"),
        ],
    )
    def test_superscript_folds(self, inp, expected):
        assert normalize(inp) == expected

    def test_superscript_inside_longer_text_still_folds(self):
        text = "Cfr. abbaḍḍuliari² e abbachiari³."
        assert normalize(text) == "Cfr. abbaḍḍuliari2 e abbachiari3."

    def test_preserves_dotted_ded_and_ligature(self):
        """NFKC would fold ḍ and ﬁ; our targeted table must not."""
        assert "ḍ" in normalize("abbaḍḍuliari²")
        assert normalize("ﬁne") == "ﬁne"

    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            ("3ª", "3a"),
            ("1º", "1o"),
            ("3ª:", "3a:"),
            ("pres. ind. 3ª: abbastimìa", "pres. ind. 3a: abbastimìa"),
        ],
    )
    def test_ordinal_indicators_fold(self, inp, expected):
        """Feminine/masculine ordinal indicators (ª/º) fold to plain a/o."""
        assert normalize(inp) == expected


class TestNormalizeHyphenSpace:
    """Whitespace around hyphens collapsed only when on BOTH sides.

    One-sided hyphens (e.g. column-break truncation `azzacca-`) are
    preserved so `is_hyphen_prefix_match` can detect them.
    """

    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            ("m - pedi", "m-pedi"),
            ("nu - llazzu", "nu-llazzu"),
            ("m  -  pedi", "m-pedi"),
            ("m-pedi", "m-pedi"),
            ("nu-llazzu", "nu-llazzu"),
            ("a - b - c", "a-b-c"),
        ],
    )
    def test_hyphen_space_both_sides_collapsed(self, inp, expected):
        assert normalize(inp) == expected

    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            # Column-break truncation: space after hyphen only — preserved.
            ("azzacca- 2.", "azzacca- 2."),
            ("azzacca-\n2.", "azzacca- 2."),
            ("v. inonda-", "v. inonda-"),
            # Tight hyphen with no spaces — preserved.
            ("a-6", "a-6"),
            ("a-⁶", "a-6"),
        ],
    )
    def test_one_sided_hyphen_preserved(self, inp, expected):
        assert normalize(inp) == expected


class TestNormalizeBaseBehavior:
    """NFC + whitespace collapse still works; idempotent."""

    def test_nfc_composition(self):
        assert normalize("cafe\u0301") == "café"

    def test_whitespace_collapsed(self):
        assert normalize("a\n\nb   c\td") == "a b c d"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize("  hello  ") == "hello"

    def test_idempotent(self):
        text = "u² m - pedi  cafe\u0301"
        once = normalize(text)
        twice = normalize(once)
        assert once == twice


class TestTokenOverlapRatio:
    """Needle-coverage: fraction of needle tokens found in haystack."""

    def test_full_overlap_returns_one(self):
        haystack = "the quick brown fox jumps over the lazy dog"
        needle = "the quick brown fox"
        assert token_overlap_ratio(haystack, needle) == 1.0

    def test_partial_overlap_returns_ratio(self):
        haystack = "a b c d e"
        needle = "a b x y z"
        # a, b match; x, y, z don't -> 2/5
        assert token_overlap_ratio(haystack, needle) == pytest.approx(0.4)

    def test_empty_needle_returns_zero(self):
        assert token_overlap_ratio("some haystack", "") == 0.0

    def test_empty_haystack_returns_zero(self):
        assert token_overlap_ratio("", "needle text here") == 0.0

    def test_inserted_garbage_in_haystack_does_not_lower_ratio(self):
        """The page-5 case: OCR has extra garbage tokens the model dropped."""
        needle = "fari a supra un trittari o fari a supra nu llazzu"
        haystack = (
            "fari a supra un trittari o 1 15 palau (RG 7) fari a supra nu llazzu"
        )
        assert token_overlap_ratio(haystack, needle) == 1.0

    def test_superscript_needle_matches_plain_haystack(self):
        """normalize() folds superscripts on both sides before comparison."""
        haystack = "abbacari2 intr. diminuire"
        needle = "abbacari²"
        assert token_overlap_ratio(haystack, needle) == 1.0

    def test_fuzzy_token_match_handles_minor_ocr_drift(self):
        """rapidfuzz fallback: single-glyph drift still counts as a hit."""
        haystack = "abbacari intr. diminuire scemare"
        needle = "abbacaro intr. diminuire scemare"
        # 'abbacari' vs 'abbacaro' differ by one char; fuzz.ratio ~87 >= 85
        assert token_overlap_ratio(haystack, needle) == 1.0

    def test_fuzzy_token_below_cutoff_does_not_count(self):
        """Tokens too dissimilar don't count as matches."""
        haystack = "abbacari intr. diminuire"
        needle = "xyzzyabc intr. diminuire"
        # 'xyzzyabc' vs 'abbacari' very different; ratio 1/3
        assert token_overlap_ratio(haystack, needle) == pytest.approx(2 / 3)

    def test_truncated_ocr_token_completes_to_needle_token(self):
        """Column-break truncation: OCR `azzacca-` → needle `azzaccanari2`.

        The third membership tier (hyphen-prefix) catches this where
        exact and fuzzy both miss. needle-coverage = 1.0.
        """
        haystack = "azzaccaniari (CT 38) v. azzacca- 2."
        needle = "(CT 38) v. azzaccanari2 2."
        # 6 needle tokens; (CT, 38), v., 2. match exactly; azzaccanari2
        # matches via hyphen-prefix (stem 'azzacca', len 7 >= 5).
        assert token_overlap_ratio(haystack, needle) == 1.0

    def test_truncated_short_stem_does_not_count(self):
        """Hyphen-prefix below MIN_HYPHEN_PREFIX_LEN doesn't match."""
        haystack = "ab- xyzzy"
        needle = "abbacari xyzzy"
        # 'abbacari' fails exact, fuzzy (vs 'ab-' and 'xyzzy'), and
        # hyphen-prefix (stem 'ab' too short). 1/2.
        assert token_overlap_ratio(haystack, needle) == pytest.approx(0.5)


class TestIsHyphenPrefixMatch:
    """Hyphen-truncated OCR token's stem is a prefix of the needle token."""

    def test_page345_reproduction(self):
        """OCR `azzacca-` (stem len 7) is a prefix of `azzaccanari2`."""
        haystack = "azzaccaniari (CT 38) v. azzacca- 2."
        assert is_hyphen_prefix_match("azzaccanari2", haystack) is True

    def test_length5_boundary_matches(self):
        """Stem of exactly 5 chars matches."""
        haystack = "v. inonda- e altro"
        assert is_hyphen_prefix_match("inondazione", haystack) is True

    def test_length4_stem_does_not_match_default_threshold(self):
        """Stem of 4 chars < MIN_HYPHEN_PREFIX_LEN (5) → no match."""
        haystack = "v. cian- e altro"
        assert is_hyphen_prefix_match("cianografico", haystack) is False

    def test_length4_stem_matches_with_lowered_threshold(self):
        """Explicit min_prefix_len=4 lets a 4-char stem match."""
        haystack = "v. cian- e altro"
        assert (
            is_hyphen_prefix_match("cianografico", haystack, min_prefix_len=4)
            is True
        )

    def test_no_hyphen_token_in_haystack_returns_false(self):
        assert is_hyphen_prefix_match("azzaccanari2", "v. azzaccanari 2.") is False

    def test_empty_needle_returns_false(self):
        assert is_hyphen_prefix_match("", "v. azzacca-") is False

    def test_needle_identical_to_stem_does_not_match(self):
        """Strict-shorter guard: `xyzzy` vs OCR `xyzzy-` must NOT match.

        The fallback is for *completions*; if the needle equals the stem
        the model didn't complete anything, so this would be a false
        positive. Guard: `len(stem) < len(needle)`.
        """
        assert is_hyphen_prefix_match("xyzzy", "v. xyzzy-") is False

    def test_needle_shorter_than_stem_does_not_match(self):
        """Needle shorter than stem can't be a completion of it."""
        # stem 'azzacca' (7) vs needle 'azzac' (5) — needle shorter than stem.
        assert is_hyphen_prefix_match("azzac", "v. azzacca-") is False

    def test_multiple_hyphen_tokens_one_matches(self):
        """Multiple truncated OCR tokens; at least one stem matches."""
        haystack = "v. inonda- e v. azzacca-"
        assert is_hyphen_prefix_match("azzaccanari2", haystack) is True

    def test_min_hyphen_prefix_len_constant(self):
        assert MIN_HYPHEN_PREFIX_LEN == 5


class TestStripTrailingDigits:
    """`strip_trailing_digits` removes trailing ASCII digits after normalize()."""

    @pytest.mark.parametrize(
        ("inp", "expected"),
        [
            ("a4", "a"),
            ("abbachiari2", "abbachiari"),
            ("abbaḍḍuliari²", "abbaḍḍuliari"),
            ("u2", "u"),
            ("a-6", "a-"),
            ("a", "a"),
            ("a¹ a²", "a1 a"),  # only the trailing run is stripped
            ("123", ""),
            ("", ""),
            ("a4a4", "a4a"),  # only trailing digits stripped
        ],
    )
    def test_strip(self, inp, expected):
        assert strip_trailing_digits(inp) == expected

    def test_superscript_folds_before_strip(self):
        # normalize() folds ²→2 first, then strip removes the 2.
        assert strip_trailing_digits("abbachiari²") == "abbachiari"

    def test_no_trailing_digit_returns_unchanged(self):
        assert strip_trailing_digits("abbachiari") == "abbachiari"


class TestIsStandaloneToken:
    """Token-set membership — not substring."""

    def test_present_as_token_returns_true(self):
        assert is_standalone_token("a", "a b c d") is True

    def test_substring_but_not_token_returns_false(self):
        """The page-1 risk case: `a` must not match inside `abbacari`."""
        assert is_standalone_token("a", "abbacari intr. diminuire") is False

    def test_empty_token_returns_false(self):
        assert is_standalone_token("", "any haystack") is False

    def test_empty_haystack_returns_false(self):
        assert is_standalone_token("a", "") is False

    def test_punctuation_distinct_token(self):
        # "a" is not "a." — token set is whitespace-delimited.
        assert is_standalone_token("a", "a. b c") is False
        assert is_standalone_token("a.", "a. b c") is True


class TestBestTokenRatio:
    """Best per-token fuzz.ratio between needle and haystack tokens."""

    def test_exact_token_match_returns_100(self):
        assert best_token_ratio("abbacari", "abbacari intr. diminuire") == 100.0

    def test_close_token_above_cutoff(self):
        # 'abbacari' vs 'abbacaro' — 7/8 chars match → ratio 87.5.
        # With default cutoff 88, this is just below; pass a lower cutoff.
        ratio = best_token_ratio(
            "abbacaro", "abbacari intr. diminuire", score_cutoff=80.0
        )
        assert ratio >= 80.0

    def test_default_cutoff_rejects_87_5(self):
        # 'abbacari' vs 'abbacaro' → 87.5 < HEADWORD_FUZZY_CUTOFF (88).
        assert best_token_ratio("abbacaro", "abbacari intr.") == 0.0

    def test_no_close_token_returns_zero(self):
        assert best_token_ratio("xyzzyabc", "abbacari intr. diminuire") == 0.0

    def test_empty_needle_returns_zero(self):
        assert best_token_ratio("", "abbacari intr.") == 0.0

    def test_empty_haystack_returns_zero(self):
        assert best_token_ratio("abbacari", "") == 0.0

    def test_headword_fuzzy_cutoff_constant(self):
        # Sanity check on the module-level constant the validator depends on.
        assert HEADWORD_FUZZY_CUTOFF == 88.0

    def test_single_char_drift_on_long_token_passes_default_cutoff(self):
        # 'abbaḍḍuliari' vs 'abbaḍḍuliaro' — 13/14 chars → ratio ~92.9 >= 88.
        ratio = best_token_ratio(
            "abbaḍḍuliari", "abbaḍḍuliaro v. bbaḍḍuliari."
        )
        assert ratio >= 88.0
