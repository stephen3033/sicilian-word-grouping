from pydantic import BaseModel, Field


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
