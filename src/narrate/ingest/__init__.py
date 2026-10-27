"""Ingest dispatch: any supported file in, a :class:`Book` out.

Nothing downstream of this package knows what a PDF is.
"""

from __future__ import annotations

from pathlib import Path

from narrate.document import Book
from narrate.ingest.docling_ingest import NoTextLayerError, ingest_with_docling
from narrate.ingest.markdown import ingest_markdown

#: Formats routed through Docling.
DOCLING_FORMATS = frozenset({".pdf", ".epub", ".docx", ".html", ".htm", ".pptx"})

#: Formats read directly - no parser needed, and going through Docling would
#: only risk mangling text that is already clean.
PLAINTEXT_FORMATS = frozenset({".md", ".markdown", ".txt", ".text"})

SUPPORTED_FORMATS = DOCLING_FORMATS | PLAINTEXT_FORMATS

__all__ = [
    "Book",
    "NoTextLayerError",
    "UnsupportedFormatError",
    "SUPPORTED_FORMATS",
    "ingest",
]


class UnsupportedFormatError(ValueError):
    pass


def ingest(
    path: Path,
    *,
    split_level: int = 2,
    include_captions: bool = False,
    include_footnotes: bool = False,
    ocr: bool = False,
    force_empty: bool = False,
) -> Book:
    """Ingest ``path`` into a :class:`Book` of speakable markdown."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = path.suffix.lower()

    if suffix in PLAINTEXT_FORMATS:
        return ingest_markdown(path, split_level=split_level)

    if suffix in DOCLING_FORMATS:
        return ingest_with_docling(
            path,
            split_level=split_level,
            include_captions=include_captions,
            include_footnotes=include_footnotes,
            ocr=ocr,
            force_empty=force_empty,
        )

    raise UnsupportedFormatError(
        f"Cannot ingest {path.name!r}. Supported formats: "
        f"{', '.join(sorted(SUPPORTED_FORMATS))}"
    )
