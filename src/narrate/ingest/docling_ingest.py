"""Docling-backed ingest for PDF, EPUB, DOCX and HTML.

Docling preserves EPUB spine order correctly, so no separate EPUB content path
is needed. It does *not* surface EPUB package metadata, so title and author are
read from the OPF directly - see :func:`_epub_metadata`.
"""

from __future__ import annotations

import html
import logging
import re
import zipfile
from pathlib import Path

from narrate.document import Book
from narrate.ingest.chapterize import split_into_chapters
from narrate.ingest.labels import speakable_labels

log = logging.getLogger(__name__)

#: Docling emits this for dropped images when image_mode is PLACEHOLDER.
_IMAGE_PLACEHOLDER = re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.MULTILINE)

#: A scanned page yields a handful of stray characters at most. Below this
#: ratio we assume there is no text layer rather than narrating silence.
_MIN_CHARS_PER_PAGE = 100


class NoTextLayerError(RuntimeError):
    """Raised for scanned PDFs, which need OCR that this version does not do."""


def ingest_with_docling(
    path: Path,
    *,
    split_level: int = 2,
    include_captions: bool = False,
    include_footnotes: bool = False,
    ocr: bool = False,
    force_empty: bool = False,
) -> Book:
    """Convert a document to a :class:`Book`, keeping only speakable elements."""
    converter = _build_converter(ocr=ocr)

    log.info("Converting %s with Docling (ocr=%s)", path.name, ocr)
    doc = converter.convert(str(path)).document

    markdown = doc.export_to_markdown(
        labels=speakable_labels(
            include_captions=include_captions,
            include_footnotes=include_footnotes,
        ),
        image_placeholder="",
    )
    markdown = _clean(markdown)

    if not force_empty:
        _guard_missing_text_layer(path, doc, markdown)

    title, author = _extract_metadata(doc, path, markdown)
    chapters = split_into_chapters(markdown, max_level=split_level, fallback_title=title)

    return Book(title=title, author=author, chapters=chapters, source_path=path)


def _build_converter(*, ocr: bool):
    """Build a converter with OCR off by default.

    Docling enables OCR for PDFs out of the box, which downloads ~30 MB of
    RapidOCR weights and adds seconds per document. Born-digital PDFs - the
    supported case - gain nothing from it, and running OCR would defeat the
    scanned-document guard by filling in text that was never in the file.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = ocr
    # do_table_structure MUST stay on. It looks like pure waste - the label
    # filter discards every table it produces - but it is the stage that *marks*
    # a region as a table. With it off, Docling emits the cell text as ordinary
    # paragraphs and the table gets narrated. Verified by test_ingest.py.
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _clean(markdown: str) -> str:
    """Remove residue that survives label filtering."""
    # EPUB and HTML sources carry entities like &amp; and &nbsp;, and Docling
    # escapes HTML on export. Left encoded, "&amp;" reaches the narrator as
    # "and amp;" because the normalizer expands the bare & to "and".
    markdown = html.unescape(markdown)
    markdown = _IMAGE_PLACEHOLDER.sub("", markdown)
    markdown = "\n".join(
        line for line in markdown.splitlines() if not _is_glyph_noise(line)
    )
    # Label filtering leaves gaps where excluded elements were; collapse them so
    # the segmenter does not read them as paragraph breaks.
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def _is_glyph_noise(line: str) -> bool:
    """True for lines that are mostly non-speakable glyphs.

    PDF vector graphics and table rules leak through as short runs of box
    characters, bullets and dashes (``■-``, ``▪ ▪ ▪``). They are labelled as
    text, so the label filter cannot catch them, and a TTS engine either reads
    them as noise or chokes on them.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if len(stripped) > 40:
        return False
    speakable = sum(char.isalnum() for char in stripped)
    return speakable / len(stripped) < 0.5


def _guard_missing_text_layer(path: Path, doc, markdown: str) -> None:
    """Fail loudly on scanned PDFs instead of narrating near-silence."""
    page_count = len(getattr(doc, "pages", {}) or {}) or 1
    if len(markdown) >= _MIN_CHARS_PER_PAGE * page_count:
        return

    raise NoTextLayerError(
        f"{path.name}: extracted only {len(markdown)} characters across "
        f"{page_count} page(s). This looks like a scanned document with no text "
        f"layer. Re-run with --ocr to let Docling read it, or --force-empty to "
        f"narrate what little was found."
    )


def _extract_metadata(doc, path: Path, markdown: str) -> tuple[str, str | None]:
    """Best-effort title and author.

    Precedence: EPUB package metadata, then a document-level TITLE element, then
    the first heading, then the filename stem. A wrong title only affects
    directory naming and ID3 tags, both overridable from the CLI.
    """
    if path.suffix.lower() == ".epub":
        title, author = _epub_metadata(path)
        if title:
            return _tidy_title(title), author

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        if label is not None and str(getattr(label, "value", label)) == "title":
            text = (getattr(item, "text", "") or "").strip()
            if text:
                return _tidy_title(text), None
            break

    # Docling labels the leading heading of many PDFs as a section header rather
    # than a title, so the first heading beats the filename.
    first_heading = re.search(r"^#{1,3}\s+(.+)$", markdown, re.MULTILINE)
    if first_heading:
        return _tidy_title(first_heading.group(1)), None

    return _tidy_title(path.stem), None


def _epub_metadata(path: Path) -> tuple[str | None, str | None]:
    """Read dc:title and dc:creator straight from the EPUB package document.

    Docling parses EPUB content correctly but discards package metadata, and
    ebooklib refuses some otherwise-valid files, so the OPF is read directly.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            container = archive.read("META-INF/container.xml").decode("utf-8")
            opf_match = re.search(r'full-path="([^"]+)"', container)
            if not opf_match:
                return None, None
            opf = archive.read(opf_match.group(1)).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        log.debug("Could not read EPUB metadata from %s: %s", path.name, exc)
        return None, None

    def first(tag: str) -> str | None:
        match = re.search(rf"<dc:{tag}[^>]*>(.*?)</dc:{tag}>", opf, re.DOTALL)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else None

    return first("title"), first("creator")


def _tidy_title(title: str) -> str:
    title = re.sub(r"[_\-]+", " ", title)
    return re.sub(r"\s+", " ", title).strip() or "Untitled"
