"""Which document elements get narrated.

This module is the whole answer to "skip the diagrams and tables". Docling gives
every element a semantic label, so exclusion is a set membership test rather
than a regex over rendered markdown - a table that happens to be formatted with
spaces instead of pipes is still labelled ``table`` and still gets dropped.
"""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel

#: Prose. Always narrated.
CORE_SPEAKABLE: frozenset[DocItemLabel] = frozenset(
    {
        DocItemLabel.TITLE,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.TEXT,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.LIST_ITEM,
    }
)

#: Opt-in via ``--captions``. Figure and table captions are often useful prose
#: ("Figure 3 shows the decline in..."), but read out of context they interrupt
#: the narrative, so they are off by default.
CAPTION_LABELS: frozenset[DocItemLabel] = frozenset({DocItemLabel.CAPTION})

#: Opt-in via ``--footnotes``. Footnotes are almost always citations or asides
#: that destroy narration flow when spliced inline.
FOOTNOTE_LABELS: frozenset[DocItemLabel] = frozenset({DocItemLabel.FOOTNOTE})

#: Never narrated, and worth being explicit about rather than relying on
#: "everything not in CORE_SPEAKABLE". Listing them documents the intent and
#: makes it obvious what a new Docling label would default to (excluded).
ALWAYS_EXCLUDED: frozenset[DocItemLabel] = frozenset(
    {
        DocItemLabel.TABLE,  # the requirement
        DocItemLabel.PICTURE,  # the requirement
        DocItemLabel.CHART,  # a picture by another name
        DocItemLabel.FORMULA,  # LaTeX read aloud is noise
        DocItemLabel.CODE,
        DocItemLabel.DOCUMENT_INDEX,  # ToC - the audio has chapters instead
        DocItemLabel.PAGE_HEADER,  # running heads repeat every page
        DocItemLabel.PAGE_FOOTER,  # page numbers
        DocItemLabel.REFERENCE,  # bibliography entries
        DocItemLabel.FORM,
        DocItemLabel.KEY_VALUE_REGION,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.HANDWRITTEN_TEXT,
        DocItemLabel.EMPTY_VALUE,
        DocItemLabel.MARKER,
        DocItemLabel.GRADING_SCALE,
        DocItemLabel.FIELD_HEADING,
        DocItemLabel.FIELD_HINT,
        DocItemLabel.FIELD_ITEM,
        DocItemLabel.FIELD_KEY,
        DocItemLabel.FIELD_REGION,
        DocItemLabel.FIELD_VALUE,
    }
)


def speakable_labels(
    *, include_captions: bool = False, include_footnotes: bool = False
) -> set[DocItemLabel]:
    """Build the label set passed to ``DoclingDocument.export_to_markdown``."""
    labels = set(CORE_SPEAKABLE)
    if include_captions:
        labels |= CAPTION_LABELS
    if include_footnotes:
        labels |= FOOTNOTE_LABELS
    return labels
