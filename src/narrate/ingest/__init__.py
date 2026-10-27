"""Ingest dispatch: any supported file in, a :class:`Book` out.

Nothing downstream of this package knows what a PDF is.
"""

from __future__ import annotations

from pathlib import Path

from narrate.document import Book
from narrate.ingest.chapterize import split_into_chapters
from narrate.ingest.docling_ingest import NoTextLayerError, ingest_with_docling

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
        return _ingest_plaintext(path, split_level=split_level)

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


def _ingest_plaintext(path: Path, *, split_level: int) -> Book:
    """Read .md/.txt directly.

    Markdown tables are stripped here because there is no Docling label tree to
    filter - this is the one place where structural exclusion has to be textual.
    """
    import re

    text = path.read_text(encoding="utf-8", errors="replace")

    # Pipe tables: a row of cells followed by a separator row.
    text = re.sub(r"^\|.*\|[ \t]*$\n?", "", text, flags=re.MULTILINE)
    # Images: ![alt](src) carries no narratable content.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Fenced code blocks.
    text = re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    title = path.stem.replace("_", " ").replace("-", " ").strip() or "Untitled"
    heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if heading:
        title = heading.group(1).strip()

    chapters = split_into_chapters(text, max_level=split_level, fallback_title=title)
    return Book(title=title, chapters=chapters, source_path=path)
