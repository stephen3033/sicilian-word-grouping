"""Prompt compilation for the transform layer."""

from __future__ import annotations

import json
import logging
import textwrap

from src.config import get_settings
from src.common.logger import log_errors
from src.models import DictionaryEntry

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
    3. NO WRAPPERS: Output your extraction directly as a raw JSON string. Do not include markdown code block formatting (```json) or conversational text.
""")

DEFAULT_USER_PREAMBLE = textwrap.dedent("""\
    Return a root JSON object containing an array of entry blocks under the "entries" key. Use the OCR text for literal characters and the image for typography/layout boundaries.

    ### TYPOGRAPHIC MAPPING RULES:
    - headword (str|null): The **bolded** lemma starting an entry. If two **bolded** words are separated by "e", generate separate entry objects for each, duplicating their shared trailing_text. Set null if page starts mid-definition.
    - trailing_text (str): Regular or *italic* body text (definitions, POS, senses) bounded between **bolded** words. Note: Ignore **bolded numbers** within body text; only **bolded words** signal a new entry boundary.
    - variants (list[str]|null): *Italicized* alternate spellings inside trailing_text preceded by markers: "v.", "anche", "Cfr.", or "v. Anche". Parse comma/ "e" separated italic sequences into clean array elements. Else null.
    - page_numbers (list[int]): Placeholder only. Always populate `page_numbers` with `[0]` for every entry. Do not infer page numbers from the OCR text or the image; the real page number is injected programmatically downstream.
    - vs_vol (int): Placeholder only. Always populate `vs_vol` with `0` for every entry. Do not infer the volume from the OCR text or the image; the real volume number is injected programmatically downstream.

    ### OCR CORRECTION:
    - The OCR scan is imperfect: fix only obvious character failures (malformed glyph, broken diacritic, clearly wrong letter) as a side effect. Do not rewrite or clean OCR wholesale.

    ### INJECTION CONTEXT:
""")

_USER_TEMPLATE = textwrap.dedent("""\
    {preamble}

    json_schema:
    {schema_json}

    ocr_page_text:
    {page_text}""")


@log_errors
def build_user_prompt(page_text: str) -> str:
    """Compile the shared user-prompt preamble + DictionaryEntry schema + OCR text."""
    s = get_settings()
    prompt = _USER_TEMPLATE.format(
        preamble=DEFAULT_USER_PREAMBLE,
        schema_json=json.dumps(DictionaryEntry.model_json_schema(), indent=2),
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