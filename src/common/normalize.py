"""OCR text normalization for verbatim substring grounding.

`normalize()` is applied symmetrically to OCR text and the model field
before any comparison, so it must be idempotent and lossless for
grounding. Layered on top of NFC + whitespace-collapse:

1. **Superscript digits → ASCII digits** via a targeted `str.translate`
   table. NFKC is deliberately avoided to preserve `ﬁ` and `ḍḍ` verbatim.
   VS OCR renders superscript lemma suffixes as ASCII (`abbacari2`); the
   model transcribes the printed typography (`abbacari²`); the table
   reconciles them.

2. **Hyphen-space collapse, only when whitespace is on BOTH sides**
   (`m - pedi` → `m-pedi`). One-sided hyphens are preserved — VS OCR
   ends truncated words with `azzacca-` + space, and collapsing that
   would destroy the truncation signal `is_hyphen_prefix_match` needs.

`token_overlap_ratio()` is the fuzzy fallback for long `trailing_text`
where OCR may insert/drop a few tokens while the model is otherwise
faithful. It's needle-coverage: the fraction of the model's tokens
present in the OCR, so extra OCR tokens never penalize the ratio.

`strip_trailing_digits()`, `is_standalone_token()`, `best_token_ratio()`
are the headword/variants fallbacks for OCR defect classes strict
substring can't absorb:

- **Digit omission/misread** — OCR drops a superscript suffix (`a` vs
  `a⁴`) or misreads it (`abbachiari1` vs `abbachiari²`). `normalize()`
  folds the superscript to ASCII; `strip_trailing_digits` yields the
  root; `is_standalone_token` checks the OCR token set.

- **Character-level drift** — a single-glyph misread on a short field
  (`abbacoari` vs `abbachiari`) where token-overlap is too coarse.
  `best_token_ratio` returns the best per-token `fuzz.ratio` against
  the OCR token set, gated by `HEADWORD_FUZZY_CUTOFF` (88 — tighter than
  the trailing_text cutoff; headwords are short, precision matters).

`is_hyphen_prefix_match()` grounds VS column-break truncations: OCR
`azzacca-` vs model `azzaccanari2`. A hyphen-truncated OCR token's stem
must be a prefix of the needle, with `len(stem) >= MIN_HYPHEN_PREFIX_LEN`
(5) and strictly shorter than the needle (the needle is a *completion*,
not identical — guards `xyzzy` vs `xyzzy-`). Used by `_ground_short_field`
and `token_overlap_ratio`.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz, process

_SUPERSCRIPT_TABLE = str.maketrans(
    {
        # VS OCR: ASCII; model: Unicode superscripts.
        "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
        "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁰": "0",
        # VS prints `3ª`/`1º`; OCR transcribes as `3a`/`1o`.
        "ª": "a", "º": "o",
    }
)

_HYPHEN_SPACE_RE = re.compile(r"\s+-\s+")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFC + superscript-digit fold + hyphen-space collapse + whitespace collapse.

    No lowercasing. Idempotent.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_SUPERSCRIPT_TABLE)
    text = _HYPHEN_SPACE_RE.sub("-", text)
    return _WS_RE.sub(" ", text).strip()


_TOKEN_MATCH_CUTOFF = 85.0
HEADWORD_FUZZY_CUTOFF = 88.0
MIN_HYPHEN_PREFIX_LEN = 5

_TRAILING_DIGITS_RE = re.compile(r"\d+$")


def token_overlap_ratio(haystack: str, needle: str) -> float:
    """Needle-coverage token overlap ratio in `[0.0, 1.0]`.

    Both inputs are `normalize()`-d first. Returns the fraction of
    needle tokens present in the haystack — extra haystack tokens never
    lower the ratio (needle-coverage, not Jaccard).

    Three-stage membership: exact set match, then `fuzz.ratio` fallback
    with per-token cutoff (absorbs minor char drift), then hyphen-prefix
    fallback for column-break truncations (`azzacca-` vs `azzaccanari2`).
    """
    needle_tokens = normalize(needle).split()
    if not needle_tokens:
        return 0.0
    haystack_token_set = set(normalize(haystack).split())
    # Hyphen-truncated tokens handled exclusively by the hyphen-prefix tier.
    fuzzy_candidates = [t for t in haystack_token_set if not t.endswith("-")]
    hits = 0
    for nt in needle_tokens:
        if nt in haystack_token_set:
            hits += 1
            continue
        match = process.extractOne(
            nt, fuzzy_candidates, scorer=fuzz.ratio,
            score_cutoff=_TOKEN_MATCH_CUTOFF,
        )
        if match is not None:
            hits += 1
            continue
        if _hyphen_prefix_match_in_set(nt, haystack_token_set):
            hits += 1
    return hits / len(needle_tokens)


def is_hyphen_prefix_match(
    needle_token: str,
    normalized_haystack: str,
    min_prefix_len: int = MIN_HYPHEN_PREFIX_LEN,
) -> bool:
    """True if any haystack token is a hyphen-truncated prefix of `needle_token`.

    A hyphen-truncated OCR token ends with `-`; its stem (without `-`)
    must be a prefix of `needle_token` with `len(stem) >= min_prefix_len`
    AND `len(stem) < len(needle_token)` (the needle is a *completion*;
    guards `xyzzy` vs OCR `xyzzy-`). `normalized_haystack` is
    `normalize()`-d here for safety.
    """
    if not needle_token:
        return False
    needle_norm = normalize(needle_token)
    for tok in set(normalized_haystack.split()):
        if not tok.endswith("-"):
            continue
        stem = tok[:-1]
        if len(stem) < min_prefix_len or len(stem) >= len(needle_norm):
            continue
        if needle_norm.startswith(stem):
            return True
    return False


def _hyphen_prefix_match_in_set(
    needle_token: str, haystack_token_set: set[str]
) -> bool:
    """Fast-path variant of `is_hyphen_prefix_match` over a pre-split token set.

    Used by `token_overlap_ratio` to avoid re-normalizing/re-splitting
    the haystack per needle token. Inputs assumed already `normalize()`-d.
    """
    if not needle_token:
        return False
    for tok in haystack_token_set:
        if not tok.endswith("-"):
            continue
        stem = tok[:-1]
        if len(stem) < MIN_HYPHEN_PREFIX_LEN or len(stem) >= len(needle_token):
            continue
        if needle_token.startswith(stem):
            return True
    return False


def strip_trailing_digits(text: str) -> str:
    """Strip all trailing ASCII digits from `text`.

    `normalize()`-d first, then `\\d+$` removed. Returns "" if digits-only.
    Headword/variants root-token fallback: `a⁴` → `a4` → `a`,
    `abbachiari²` → `abbachiari2` → `abbachiari`.
    """
    normalized = normalize(text)
    return _TRAILING_DIGITS_RE.sub("", normalized)


def is_standalone_token(token: str, normalized_haystack: str) -> bool:
    """True iff `token` appears as a whitespace-delimited token in the haystack.

    `token` is used verbatim (caller should pre-`normalize()` it);
    haystack is `normalize()`-d here. Token-set membership avoids the
    false positives a substring check would invite (root `a` must not
    match inside `abbacari`).
    """
    if not token:
        return False
    return token in set(normalized_haystack.split())


def best_token_ratio(
    needle: str, haystack: str, score_cutoff: float = HEADWORD_FUZZY_CUTOFF
) -> float:
    """Best `fuzz.ratio` between `needle` and any token in `haystack`.

    Both inputs `normalize()`-d first. Returns the best score in
    `[0.0, 100.0]`, or `0.0` if no token clears `score_cutoff`.
    Headword/variants fuzzy fallback for char-level OCR drift on short
    fields. Hyphen-truncated tokens (ending `-`) are excluded — they're
    column-break markers handled by `is_hyphen_prefix_match`.
    """
    needle_norm = normalize(needle)
    if not needle_norm:
        return 0.0
    haystack_tokens = [
        t for t in set(normalize(haystack).split()) if not t.endswith("-")
    ]
    if not haystack_tokens:
        return 0.0
    match = process.extractOne(
        needle_norm, haystack_tokens, scorer=fuzz.ratio,
        score_cutoff=score_cutoff,
    )
    return match[1] if match is not None else 0.0
