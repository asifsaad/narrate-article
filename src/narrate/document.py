"""The intermediate document model.

Everything between ingest and synthesis speaks in these types. Ingest backends
(Docling, EPUB, plaintext) all converge on ``Book``; the normalizer, segmenter
and TTS layers only ever see ``Book`` and ``Chapter``, so a new input format
never reaches past ``narrate.ingest``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def slugify(text: str, max_length: int = 60) -> str:
    """Filesystem-safe ASCII slug. Used for work/output directory names."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w\s-]", "", ascii_only).strip().lower()
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug[:max_length].strip("-") or "untitled"


def safe_filename(text: str, max_length: int = 80) -> str:
    """Human-readable but filesystem-safe. Preserves case and spaces."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:max_length].strip() or "Untitled"


@dataclass
class Chapter:
    """One unit of narration, producing one output audio file."""

    index: int
    """1-based position in the book. Drives output file ordering."""

    title: str
    markdown: str
    """Speakable markdown. Tables, figures and other excluded elements are
    already gone by the time a Chapter exists."""

    def __post_init__(self) -> None:
        self.title = self.title.strip()
        self.markdown = self.markdown.strip()

    @property
    def is_empty(self) -> bool:
        """A chapter that is only a heading has nothing to narrate.

        Common on PDFs with part-title pages, and after label filtering strips a
        chapter that was entirely a table or figure.
        """
        body = re.sub(r"^#{1,6}\s+.*$", "", self.markdown, flags=re.MULTILINE)
        return not body.strip()

    @property
    def word_count(self) -> int:
        return len(self.markdown.split())

    def filename(self, extension: str) -> str:
        """e.g. ``001 - The Golden Bowl.mp3`` — sorts correctly in any player."""
        return f"{self.index:03d} - {safe_filename(self.title)}.{extension.lstrip('.')}"


@dataclass
class Book:
    """A whole ingested document."""

    title: str
    chapters: list[Chapter]
    author: str | None = None
    source_path: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return slugify(self.title)

    @property
    def word_count(self) -> int:
        return sum(chapter.word_count for chapter in self.chapters)

    def narratable_chapters(self) -> list[Chapter]:
        return [chapter for chapter in self.chapters if not chapter.is_empty]


@dataclass
class Chunk:
    """A single unit handed to the TTS engine.

    Sized to sit under the engine's budget and never split mid-sentence. The
    trailing silence is decided at segmentation time, where paragraph structure
    is still visible — by synthesis time it is gone.
    """

    text: str
    """Normalized, fully speakable. No markdown, no digits, no abbreviations."""

    chapter_index: int
    order: int
    """0-based position within the chapter."""

    trailing_silence_ms: int = 0
    """Silence appended after this chunk's audio."""

    def cache_key(self, engine: str, voice: str, speed: float) -> str:
        """Content hash over everything that affects the rendered audio.

        Deliberately excludes chapter_index and order so that moving a paragraph
        does not invalidate its audio.
        """
        import hashlib

        payload = f"{engine}|{voice}|{speed:.3f}|{self.text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
