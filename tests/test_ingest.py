"""Ingest correctness: the primary requirement.

Tables and diagrams must not reach the narrator; the prose around them must.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from narrate.ingest import ingest
from narrate.ingest.chapterize import split_into_chapters
from narrate.ingest.docling_ingest import NoTextLayerError, _is_glyph_noise

FIXTURE_DIR = Path(__file__).parent / "fixtures"

PROSE_SENTINEL = "quintessential"
TABLE_SENTINEL = "zbxqtable"
FIGURE_SENTINEL = "zbxqfigure"


@pytest.fixture(scope="session", autouse=True)
def fixtures_exist() -> None:
    if (FIXTURE_DIR / "sample.pdf").exists() and (FIXTURE_DIR / "sample.epub").exists():
        return
    subprocess.run(
        [sys.executable, str(FIXTURE_DIR / "make_fixtures.py")], check=True
    )


@pytest.fixture(scope="session")
def pdf_book():
    return ingest(FIXTURE_DIR / "sample.pdf")


@pytest.fixture(scope="session")
def epub_book():
    return ingest(FIXTURE_DIR / "sample.epub")


def _full_text(book) -> str:
    return "\n".join(chapter.markdown for chapter in book.chapters)


# --- the requirement ------------------------------------------------------


@pytest.mark.parametrize("book_fixture", ["pdf_book", "epub_book"])
def test_prose_survives(book_fixture, request):
    book = request.getfixturevalue(book_fixture)
    assert PROSE_SENTINEL in _full_text(book)


@pytest.mark.parametrize("book_fixture", ["pdf_book", "epub_book"])
def test_table_contents_are_dropped(book_fixture, request):
    """Regression guard for do_table_structure.

    Turning that pipeline stage off makes Docling emit table cells as ordinary
    paragraphs, and this assertion is the only thing that catches it.
    """
    book = request.getfixturevalue(book_fixture)
    assert TABLE_SENTINEL not in _full_text(book)


@pytest.mark.parametrize("book_fixture", ["pdf_book", "epub_book"])
def test_figure_caption_is_dropped(book_fixture, request):
    book = request.getfixturevalue(book_fixture)
    assert FIGURE_SENTINEL not in _full_text(book)


@pytest.mark.parametrize("book_fixture", ["pdf_book", "epub_book"])
def test_no_image_placeholders(book_fixture, request):
    book = request.getfixturevalue(book_fixture)
    assert "<!-- image -->" not in _full_text(book)


# --- structure and metadata -----------------------------------------------


def test_html_entities_are_decoded():
    """Regression: '&amp;' reached the narrator as 'and amp;'.

    The normalizer expands a bare '&' to 'and', so an undecoded entity produces
    an audible nonsense word rather than a visible glitch.
    """
    from narrate.ingest.docling_ingest import _clean

    assert _clean("Gods &amp; Monsters") == "Gods & Monsters"
    assert _clean("caf&eacute; &nbsp;bar").startswith("café")


def test_epub_metadata_comes_from_opf(epub_book):
    """Docling drops EPUB package metadata, so it is read from the OPF."""
    assert epub_book.title == "The Migration of Starlings"
    assert epub_book.author == "Elena Ramirez"


def test_epub_preserves_spine_order(epub_book):
    assert [c.title for c in epub_book.chapters] == [
        "Chapter One: Departure",
        "Chapter Two: Arrival",
    ]


def test_pdf_title_falls_back_to_first_heading(pdf_book):
    """Filename is 'sample'; the real title is only in the document body."""
    assert pdf_book.title == "The Migration of Starlings"


@pytest.mark.parametrize("book_fixture", ["pdf_book", "epub_book"])
def test_chapters_are_indexed_from_one(book_fixture, request):
    book = request.getfixturevalue(book_fixture)
    assert [c.index for c in book.chapters] == list(range(1, len(book.chapters) + 1))


def test_chapter_filenames_sort_correctly(epub_book):
    names = [c.filename("mp3") for c in epub_book.chapters]
    assert names == sorted(names)
    assert names[0].startswith("001 - ")


# --- captions and footnotes are opt-in ------------------------------------


def test_captions_can_be_opted_into():
    book = ingest(FIXTURE_DIR / "sample.epub", include_captions=True)
    assert FIGURE_SENTINEL in _full_text(book)


# --- plaintext path -------------------------------------------------------


def test_plaintext_strips_tables_and_images(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text(
        "# Notes\n\n"
        "Real prose here.\n\n"
        "| Region | Count |\n"
        "|--------|-------|\n"
        f"| North  | {TABLE_SENTINEL} |\n\n"
        f"![{FIGURE_SENTINEL}](chart.png)\n\n"
        "More prose after.\n"
    )
    book = ingest(source)
    text = _full_text(book)

    assert "Real prose here." in text
    assert "More prose after." in text
    assert TABLE_SENTINEL not in text
    assert FIGURE_SENTINEL not in text


def test_unsupported_format_is_rejected(tmp_path):
    from narrate.ingest import UnsupportedFormatError

    bogus = tmp_path / "audio.flac"
    bogus.write_bytes(b"not a document")
    with pytest.raises(UnsupportedFormatError):
        ingest(bogus)


def test_scanned_pdf_fails_loudly(tmp_path):
    """A PDF with no text layer must raise, not narrate silence."""
    from reportlab.pdfgen import canvas as pdf_canvas

    empty = tmp_path / "scanned.pdf"
    page = pdf_canvas.Canvas(str(empty))
    page.drawString(100, 700, ".")
    page.save()

    with pytest.raises(NoTextLayerError, match="scanned"):
        ingest(empty)


# --- chapterization units -------------------------------------------------


def test_short_sections_merge_into_previous():
    markdown = (
        "# Real Chapter\n\n" + "word " * 60 + "\n\n"
        "# Stub\n\nToo short.\n\n"
        "# Another Real\n\n" + "word " * 60
    )
    chapters = split_into_chapters(markdown)
    assert [c.title for c in chapters] == ["Real Chapter", "Another Real"]
    assert "Too short." in chapters[0].markdown


def test_preamble_is_not_lost():
    markdown = "Front matter sentence.\n\n# First\n\n" + "word " * 60
    chapters = split_into_chapters(markdown)
    assert "Front matter sentence." in chapters[0].markdown


def test_deep_headings_stay_inline():
    markdown = "# Chapter\n\n" + "word " * 60 + "\n\n### Subsection\n\nmore text"
    chapters = split_into_chapters(markdown, max_level=2)
    assert len(chapters) == 1
    assert "### Subsection" in chapters[0].markdown


def test_document_with_no_headings_becomes_one_chapter():
    chapters = split_into_chapters("Just a wall of text.", fallback_title="Doc")
    assert len(chapters) == 1
    assert chapters[0].title == "Doc"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("■-", True),
        ("▪ ▪ ▪", True),
        ("--- ---", True),
        ("## Real Heading", False),
        ("The starlings gathered at dusk.", False),
        ("A1", False),
    ],
)
def test_glyph_noise_detection(line, expected):
    assert _is_glyph_noise(line) is expected
