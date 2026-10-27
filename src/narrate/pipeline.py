"""Orchestration: file in, narrated audio out.

Stages are separable on purpose. ``extract`` alone produces the editable
markdown checkpoint; ``narrate`` resumes from that checkpoint. The two are
connected only through files on disk, so a user can edit between them - or run
extraction on one machine and synthesis on another.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from narrate.audio import (
    build_m4b,
    concat_with_silence,
    extract_cover,
    format_duration,
    write_chapter,
)
from narrate.document import Book, Chapter
from narrate.ingest import ingest
from narrate.ingest.chapterize import split_into_chapters
from narrate.normalize import Lexicon, get_normalizer
from narrate.segment import segment_chapter
from narrate.tts import AudioCache, get_engine

log = logging.getLogger(__name__)

WORK_DIR = Path("work")
OUTPUT_DIR = Path("out")

_CHAPTER_FILE = re.compile(r"^(\d{3})[-_ ]+(.*)$")


@dataclass
class NarrationConfig:
    """Everything that affects the rendered result."""

    engine: str = "kokoro"
    voice: str = "af_heart"
    speed: float = 1.0
    audio_format: str = "mp3"
    bitrate: str = "128k"
    normalize_loudness: bool = True

    split_level: int = 2
    include_captions: bool = False
    include_footnotes: bool = False
    keep_citations: bool = False
    ocr: bool = False
    force_empty: bool = False

    lexicon_path: Path | None = None
    work_dir: Path = WORK_DIR
    output_dir: Path = OUTPUT_DIR
    use_cache: bool = True
    parallel_encode: bool = True
    """Encode chapter N while chapter N+1 synthesizes.

    Synthesis is GPU-bound and encoding is CPU-bound, so they overlap for free -
    measured at -1.5% (noise) on an M3 Pro. Set False to debug ffmpeg errors
    against a specific chapter, or on a machine with very few cores.
    """

    chapters: set[int] | None = None
    """1-based chapter numbers to render. ``None`` means all of them."""


@dataclass
class ChapterResult:
    chapter: Chapter
    path: Path | None
    duration_seconds: float
    chunk_count: int
    audio: object | None = None
    """Raw samples, retained only while bundling an m4b, then released."""


@dataclass
class NarrationResult:
    book: Book
    chapters: list[ChapterResult] = field(default_factory=list)
    cache_summary: str = ""
    elapsed_seconds: float = 0.0
    bundle: Path | None = None
    """The single-file .m4b, when one was requested."""

    @property
    def total_duration(self) -> float:
        return sum(c.duration_seconds for c in self.chapters)


ProgressCallback = Callable[[str, int, int], None]


# --- stage 1: extract -----------------------------------------------------


def extract(source: Path, config: NarrationConfig) -> tuple[Book, Path]:
    """Ingest to markdown and write the editable checkpoint."""
    book = ingest(
        source,
        split_level=config.split_level,
        include_captions=config.include_captions,
        include_footnotes=config.include_footnotes,
        ocr=config.ocr,
        force_empty=config.force_empty,
    )

    checkpoint = write_checkpoint(book, config.work_dir)
    return book, checkpoint


def write_checkpoint(book: Book, work_dir: Path) -> Path:
    """Write one markdown file per chapter, plus a manifest.

    Filenames carry the chapter number so ordering survives a round trip
    through the filesystem, and so a user can reorder by renaming.
    """
    directory = Path(work_dir) / book.slug
    chapters_dir = directory / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale chapter files from a previous run with different settings;
    # leaving them would silently narrate content the user thought was gone.
    for stale in chapters_dir.glob("*.md"):
        stale.unlink()

    for chapter in book.chapters:
        path = chapters_dir / chapter.filename("md")
        path.write_text(chapter.markdown + "\n", encoding="utf-8")

    manifest = [
        f"# {book.title}",
        "",
        f"- Author: {book.author or 'unknown'}",
        f"- Source: {book.source_path.name if book.source_path else 'unknown'}",
        f"- Chapters: {len(book.chapters)}",
        f"- Words: {book.word_count:,}",
        "",
        "Edit the files in `chapters/`, then re-run with `--from-markdown`.",
        "Only edited chunks are re-synthesized.",
        "",
    ]
    for chapter in book.chapters:
        flag = "  (empty)" if chapter.is_empty else ""
        manifest.append(f"{chapter.index:3d}. {chapter.title} — {chapter.word_count:,}w{flag}")

    (directory / "README.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return directory


def load_checkpoint(directory: Path, *, split_level: int = 2) -> Book:
    """Rebuild a Book from an edited checkpoint directory."""
    directory = Path(directory)
    chapters_dir = directory / "chapters"
    if not chapters_dir.is_dir():
        raise FileNotFoundError(f"No chapters/ directory under {directory}")

    files = sorted(chapters_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown files in {chapters_dir}")

    title = directory.name.replace("-", " ").title()
    author = None
    readme = directory / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        if heading := re.search(r"^#\s+(.+)$", text, re.MULTILINE):
            title = heading.group(1).strip()
        if creator := re.search(r"^- Author:\s*(.+)$", text, re.MULTILINE):
            author = None if creator.group(1).strip() == "unknown" else creator.group(1).strip()

    chapters: list[Chapter] = []
    for position, path in enumerate(files, start=1):
        markdown = path.read_text(encoding="utf-8").strip()
        if not markdown:
            continue

        stem_match = _CHAPTER_FILE.match(path.stem)
        chapter_title = stem_match.group(2).strip() if stem_match else path.stem

        # A user who pasted several chapters into one file gets them split
        # again rather than one oversized output track.
        parts = split_into_chapters(
            markdown, max_level=split_level, fallback_title=chapter_title
        )
        if len(parts) <= 1:
            chapters.append(
                Chapter(index=position, title=chapter_title, markdown=markdown)
            )
        else:
            chapters.extend(parts)

    for position, chapter in enumerate(chapters, start=1):
        chapter.index = position

    return Book(title=title, author=author, chapters=chapters, source_path=directory)


# --- stage 2: narrate -----------------------------------------------------


def narrate(
    book: Book,
    config: NarrationConfig,
    *,
    progress: ProgressCallback | None = None,
) -> NarrationResult:
    """Normalize, segment, synthesize and encode every selected chapter."""
    started = time.time()

    engine = get_engine(config.engine, lang_code=_lang_code_for(config))
    normalizer = get_normalizer(
        lexicon=Lexicon.load(config.lexicon_path),
        keep_citations=config.keep_citations,
    )
    cache = AudioCache(
        Path(config.work_dir) / book.slug / "cache",
        sample_rate=engine.sample_rate,
        enabled=config.use_cache,
    )

    destination = Path(config.output_dir) / book.slug
    destination.mkdir(parents=True, exist_ok=True)

    selected = _select_chapters(book, config)
    result = NarrationResult(book=book)

    # An m4b is one file, so its chapters are muxed at the end rather than
    # written out individually. They are staged to disk as raw WAV rather than
    # held in memory - a 28-hour book is ~10 GB of float32, and concatenating
    # in memory would double that.
    bundling = config.audio_format == "m4b"
    stage_dir = Path(config.work_dir) / book.slug / "stage"
    tracks: list[tuple[str, Path, int]] = []

    # Synthesis saturates the GPU on its own - running chapters concurrently
    # measured *slower* (0.81x at four threads). Encoding is CPU work on a
    # separate resource, so that is what gets overlapped instead.
    encoder = _ChapterEncoder(
        enabled=config.parallel_encode and not bundling,
        book=book,
        config=config,
        engine=engine,
        destination=destination,
    )

    try:
        for position, chapter in enumerate(selected, start=1):
            if progress:
                progress(chapter.title, position, len(selected))

            rendered = _render_chapter(
                chapter,
                engine=engine,
                normalizer=normalizer,
                cache=cache,
                config=config,
                book=book,
                destination=destination,
            )
            if rendered is None:
                continue

            result.chapters.append(rendered)
            if bundling:
                tracks.append(_stage_chapter(rendered, stage_dir, engine.sample_rate))
            else:
                encoder.submit(rendered)

        encoder.finish()
    finally:
        encoder.shutdown()

    if bundling and tracks:
        try:
            result.bundle = _bundle_m4b(
                tracks, book=book, config=config, engine=engine, destination=destination
            )
        finally:
            # Staged WAVs are ~10 GB for a long book; never leave them behind.
            for _title, path, _samples in tracks:
                path.unlink(missing_ok=True)
            with suppress(OSError):
                stage_dir.rmdir()

    result.cache_summary = cache.summary()
    result.elapsed_seconds = time.time() - started
    return result


def _stage_chapter(
    rendered: ChapterResult, stage_dir: Path, sample_rate: int
) -> tuple[str, Path, int]:
    """Write one chapter's samples to disk and release them from memory.

    Staged as float32 WAV - the same samples the encoder would have received,
    so the eventual m4b is still a single lossy encode rather than a transcode.
    """
    import soundfile as sf

    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / f"{rendered.chapter.index:03d}.wav"

    audio = rendered.audio
    sf.write(path, audio, sample_rate, format="WAV", subtype="FLOAT")
    samples = int(getattr(audio, "size", len(audio)))

    rendered.audio = None
    return rendered.chapter.title, path, samples


class _ChapterEncoder:
    """Runs chapter encoding off the synthesis thread.

    ffmpeg is a subprocess doing CPU work while Kokoro holds the GPU, so the two
    overlap essentially for free. Encoding is ~4x faster than synthesizing the
    same chapter, so the queue drains faster than it fills and never grows -
    but ``MAX_IN_FLIGHT`` bounds it anyway, because each queued chapter holds
    its float32 samples in memory until written.
    """

    MAX_IN_FLIGHT = 3

    def __init__(self, *, enabled, book, config, engine, destination):
        self.enabled = enabled
        self.book = book
        self.config = config
        self.engine = engine
        self.destination = destination
        self._pending: list[tuple[object, ChapterResult]] = []
        self._pool = None

        if enabled:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="encode")

    def submit(self, rendered: ChapterResult) -> None:
        if not self.enabled:
            self._encode(rendered)
            return

        while len(self._pending) >= self.MAX_IN_FLIGHT:
            self._drain_one()

        self._pending.append((self._pool.submit(self._encode, rendered), rendered))

    def finish(self) -> None:
        while self._pending:
            self._drain_one()

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    def _drain_one(self) -> None:
        future, _ = self._pending.pop(0)
        # Re-raises inside the caller's thread, so an ffmpeg failure surfaces
        # as a normal error rather than vanishing into the pool.
        future.result()

    def _encode(self, rendered: ChapterResult) -> None:
        chapter = rendered.chapter
        path = self.destination / chapter.filename(self.config.audio_format)

        write_chapter(
            rendered.audio,
            path,
            sample_rate=self.engine.sample_rate,
            audio_format=self.config.audio_format,
            bitrate=self.config.bitrate,
            normalize_loudness=self.config.normalize_loudness,
            tags={
                "album": self.book.title,
                "artist": self.book.author or "Unknown",
                "albumartist": self.book.author or "Unknown",
                "title": chapter.title,
                "tracknumber": str(chapter.index),
                "genre": "Audiobook",
            },
        )

        rendered.path = path
        rendered.audio = None  # a book's worth of float32 is gigabytes


def _bundle_m4b(tracks, *, book: Book, config: NarrationConfig, engine, destination: Path) -> Path:
    cover = extract_cover(book.source_path, destination) if book.source_path else None

    path = destination / f"{book.slug}.m4b"
    build_m4b(
        tracks,
        path,
        sample_rate=engine.sample_rate,
        bitrate=config.bitrate,
        normalize_loudness=config.normalize_loudness,
        tags={
            "title": book.title,
            "album": book.title,
            "artist": book.author or "Unknown",
            "album_artist": book.author or "Unknown",
            "genre": "Audiobook",
        },
        cover=cover,
    )

    if cover and cover.is_file():
        cover.unlink(missing_ok=True)

    return path


def _lang_code_for(config: NarrationConfig) -> str:
    if config.engine != "kokoro":
        return "a"
    from narrate.tts.kokoro_engine import lang_code_for_voice

    return lang_code_for_voice(config.voice)


def _select_chapters(book: Book, config: NarrationConfig) -> list[Chapter]:
    chapters = book.narratable_chapters()
    if config.chapters is not None:
        chapters = [c for c in chapters if c.index in config.chapters]
    return chapters


def _render_chapter(
    chapter: Chapter,
    *,
    engine,
    normalizer,
    cache: AudioCache,
    config: NarrationConfig,
    book: Book,
    destination: Path,
) -> ChapterResult | None:
    text = normalizer.normalize(chapter.markdown)
    chunks = segment_chapter(
        text, chapter_index=chapter.index, max_chars=engine.max_chars
    )
    if not chunks:
        log.warning("Chapter %d (%s) produced no speakable text", chapter.index, chapter.title)
        return None

    pieces = []
    for chunk in chunks:
        audio = cache.get(
            chunk, engine=engine.name, voice=config.voice, speed=config.speed
        )
        if audio is None:
            audio = engine.synthesize(
                chunk.text, voice=config.voice, speed=config.speed
            )
            cache.put(
                chunk, audio, engine=engine.name, voice=config.voice, speed=config.speed
            )
        pieces.append((audio, chunk.trailing_silence_ms))

    track = concat_with_silence(pieces, sample_rate=engine.sample_rate)
    if track.size == 0:
        log.warning("Chapter %d rendered to silence; skipping", chapter.index)
        return None

    # Writing is the caller's job: it either hands the samples to the encoder
    # thread or muxes them into an m4b. Keeping synthesis free of file I/O is
    # what lets the two overlap.
    return ChapterResult(
        chapter=chapter,
        path=None,
        duration_seconds=len(track) / engine.sample_rate,
        chunk_count=len(chunks),
        audio=track,
    )


# --- convenience ----------------------------------------------------------


def run(
    source: Path,
    config: NarrationConfig,
    *,
    progress: ProgressCallback | None = None,
) -> NarrationResult:
    """Full pipeline: extract then narrate."""
    book, _ = extract(source, config)
    return narrate(book, config, progress=progress)


__all__ = [
    "NarrationConfig",
    "NarrationResult",
    "ChapterResult",
    "extract",
    "narrate",
    "run",
    "load_checkpoint",
    "write_checkpoint",
    "format_duration",
]
