"""Prompt compilation for the transform layer."""

from __future__ import annotations

import json
import logging
import textwrap

from src.config import get_settings
from src.common.logger import log_errors
from src.models import LLMEntry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a high-precision, non-conversational data extraction engine for structured lexicographical parsing. 
    Your sole task is to process a document page and transform its text and layout features into a single valid JSON object.

    STRUCTURAL OUTPUT RULE:
    The JSON Schema supplied in the user prompt defines the structure for a SINGLE dictionary entry (one word). 
    Because a single page contains multiple distinct entries, you must return a root JSON object containing a list of these word extractions under an "entries" key, structured exactly like this:
    {
      "entries": [
        {... word object matching the provided schema ...},
        {... word object matching the provided schema ...}
      ]
    }

    CRITICAL RULES OF ENGAGEMENT:
    1. LITERAL ACCURACY: Use the provided raw user text block as your absolute single source of truth for character data, spelling, diacritics, and specialized symbols. Do not alter spellings.
    2. VISUAL GROUNDING: Use the provided page image strictly as a structural map. Rely on the image to locate entry splits, column switches, nested paragraph configurations, and typographic shifts (e.g., font shifts, bolding, or italics) that indicate field changes.
    3. HEADWORD BOUNDARY HARD RAIL: A headword is exactly one bold lemma span. Stop at the end of that bold span. Never combine it with comma-separated alternates, a second bold lemma, parenthetical locality/source codes, POS text, or cross-reference prose.
    4. CHARACTER HARD RAIL: Never add, remove, or normalize a headword/variant hyphen or a Roman/Arabic numeral. OCR spaces around a hyphen do not remove the hyphen. Roman `I` and Arabic `1` are different characters and must never be interchanged in any field.
    5. NO WRAPPERS: Output your extraction directly as a raw JSON string. Do not include markdown code block formatting (```json) or conversational text.
""")

DEFAULT_USER_PREAMBLE = textwrap.dedent("""\
    Return a root JSON object containing an array of entry blocks under the "entries" key. Use the OCR text for literal characters and the image for typography/layout boundaries.

    ### TYPOGRAPHIC MAPPING RULES:
    - headword (str|null): The **bolded** lemma starting an entry. Copy the complete lemma exactly, including every hyphen, apostrophe, diacritic, doubled letter, and homonym numeral. Set null only if the page starts mid-definition.
    - paired headwords: If two **bolded** lemmas are separated by "e" and share one body, generate a separate entry for each and assign shared variants to both. A second bold lemma always gets its own entry even when OCR places both on one line. Every trailing_text must remain one contiguous OCR substring: for the first entry retain the connecting "e", the second lemma, its qualifier, and the shared body; for the second entry begin immediately after its own bold lemma. Never synthesize a cleaned trailing_text by deleting the second lemma from the middle of the OCR span, never leave the first entry with only its qualifier, and never omit the second entry.
    - trailing_text (str): Copy the complete regular or *italic* body text (definitions, POS, senses, citations, and examples) bounded between **bolded** lemma words. Never summarize, shorten, paraphrase, or omit body content. Ignore **bolded numbers** within body text as entry boundaries; only **bolded words** signal a new entry boundary. Start with the first non-whitespace character after the headword and end with the final non-whitespace character before the next headword; never include outer whitespace.
    - variants (list[str]|null): Capture every *italicized* alternate or referenced lemma governed by "v.", "V.", "anche", "Anche", "Cfr.", or "v. Anche" inside the entry. Parse comma/"e" separated italic sequences into separate clean array elements. Preserve each lemma exactly, but exclude the marker, locality/source labels, POS labels, surrounding prose, and sense selectors that follow a referenced lemma. A lemma after `pp. di` is grammatical body text, not a variant; likewise, an ordinary definition word is never a variant merely because it resembles another lemma. Use JSON null—not an empty array—only after checking the entire entry for every marker.
    ### HEADWORD AND VARIANT CHARACTER FIDELITY:
    - Audit every headword and every variants element character by character against the OCR and its bold/italic span in the image. Never silently normalize, autocomplete, simplify, or fuzzy-correct these fields.
    - Preserve hyphens even when OCR inserts spaces around them. Exact required mappings from this source: OCR `a - 6 prefisso` -> headword `a-⁶`, never `a⁶`; OCR `â - 3.` -> headword `â-³`, never `â³`. The hyphen is part of the bold lemma, while OCR whitespace around it is not.
    - Preserve every doubled/marked letter exactly. Printed `ḍḍ` must not become `dḍ` and must never become single `ḍ`. Exact required headwords include `abbaḍḍari²`, `abbaḍḍariari`, `abbaḍḍariatu`, `abbaḍḍàrisi`, `abbaḍḍatu`, `abbaḍḍiatu`, `abbaḍḍuniari`, `abbaḍḍuttari`, `abbaḍḍuttuliari`, `abbamminiḍḍatu`, and `abbamminiḍḍutu`.
    - Preserve homonym numerals attached to a lemma, including their identity and printed superscript form: `bbadàgghiu¹` must not become `bbadàgghiu` or `bbadàgghiu²`. In OCR, a digit attached directly to a lemma is its homonym numeral and must be emitted as the corresponding superscript (`bbabbalà1` -> `bbabbalà¹`, `bbabbaluccu2` -> `bbabbaluccu²`, `bbabbu1` -> `bbabbu¹`). A numeral separated from a lemma by whitespace is a sense selector and stays out of variants.
    - A headword must contain only one bold lemma. For OCR `abbaciù, abbaçiù (CT II), abbaciurri (RG 7) v. abbaciurru 1.`, output headword `abbaciù`; the other referenced forms belong in variants. Never output the combined string `abbaciù, abbaçiù (CT II), abbaciurri (RG 7)` as a headword.
    - Keep sense selectors out of variants. In `(CT II) v. abbivirari 1.`, the variant is `abbivirari`; the selector `1` remains only in trailing_text. Never produce a variants element such as `abbivirari 1` or `abbauttiri 2 e 3`.
    - For a shared OCR span such as **lemma A** `(X) e` **lemma B** `(Y) v. reference.`, both entries receive `reference` in variants. Entry A retains contiguous trailing_text `(X) e lemma B (Y) v. reference.`; entry B receives contiguous trailing_text `(Y) v. reference.`.
    - Required same-line boundaries in this source: `aba ... abba ...` is two entries; `abbadagghiari e (S. C.) abbadagliari ...` is two entries; `abbadàgghiu e (S. C.) abbadàgliu ...` is two entries. Preserve the shared contiguous text as described above, but emit every bold lemma as its own entry.

    ### OCR CORRECTION:
    - The OCR scan is imperfect: fix only an obvious OCR character failure confirmed by the page image (malformed glyph, broken diacritic, clearly wrong letter). Do not rewrite or clean OCR wholesale, and do not use this permission to alter a plausible headword or variant spelling.
    - NUMERAL FIDELITY IN ALL FIELDS: VS cross-references mix Roman and Arabic sense numerals on the same page (e.g. `v. custumi I, 3 e 4.`, `(CT II) v. abbivirari 1.`). Transcribe whichever numeral form is printed verbatim. Do NOT normalize Roman to Arabic (`I` → `1`) or Arabic to Roman. Treat `V.` (the "Vedi" citation marker) as a citation keyword, not a numeral. Treat lowercase `i`/`ii` (the Sicilian pronoun, as in `i iurnati`) as words, not numerals.
    - Exact required Roman-numeral example: OCR `(Man.) v. abbabbasuniri I.` must remain `(Man.) v. abbabbasuniri I.` in trailing_text. Outputting `(Man.) v. abbabbasuniri 1.` is incorrect. Its variants value contains `abbabbasuniri` without the selector.

    ### MANDATORY PRE-OUTPUT AUDIT:
    1. Walk the page in reading order and verify that every bolded lemma has exactly one entry, except the single null-headword continuation allowed at the page start.
    2. Reject and rebuild any headword that absorbs a comma-separated alternate, parenthetical locality/source code, or second lemma. Recheck every remaining headword character, especially hyphens, diacritics, every `ḍḍ` pair, and attached superscript numerals. Count bold lemma spans independently from the JSON entries; the counts must agree exactly.
    3. Sweep every entry for every `v.`/`V.`/`Anche`/`anche`/`Cfr.` marker; verify that all and only the governed italic lemma strings appear in variants, with no lost attached homonym numeral and no trailing sense selector. Remove anything sourced only from `pp. di` or ordinary definition prose, and replace every empty variants array with null.
    4. Search the OCR for spaced lemma hyphens and for Roman `I`; verify the output retained each hyphen and never substituted `1` for `I`. Verify that trailing_text contains the complete body through the next bolded lemma boundary and is always a contiguous OCR substring.
    5. Return only the JSON object after all four checks pass.

""")

_USER_TEMPLATE = textwrap.dedent("""\
    {preamble}

    json_schema:
    {schema_json}

    ocr_page_text:
    {page_text}""")


@log_errors
def build_user_prompt(page_text: str) -> str:
    """Compile the shared prompt with the model-owned ``LLMEntry`` schema."""
    s = get_settings()
    prompt = _USER_TEMPLATE.format(
        preamble=DEFAULT_USER_PREAMBLE,
        schema_json=json.dumps(LLMEntry.model_json_schema(), indent=2),
        page_text=page_text,
    )
    logger.debug(
        "model=%s preamble=%d chars prompt=%d chars ocr=%d chars",
        s.model,
        len(DEFAULT_USER_PREAMBLE),
        len(prompt),
        len(page_text),
    )
    return prompt
