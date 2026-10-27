"""Short documents: one markdown file in, one audio file out.

The book pipeline is shaped by hours of audio - an editable checkpoint between
stages, one file per chapter, m4b bundling. None of that helps a 2,000-word
note. Here the source file *is* the editable text, and the result is a single
track you can drop on a phone.

Everything below the document level is shared with books: the same
normalizer, segmenter, engine and content-addressed cache. A note sounds
exactly like a chapter, and re-running after an edit re-synthesizes only the
sentences that changed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from narrate.audio import concat_with_silence, write_chapter
from narrate.document import Book, Chapter, Chunk, slugify
from narrate.ingest import PLAINTEXT_FORMATS
from narrate.ingest.markdown import ingest_markdown
from narrate.normalize import Lexicon, get_normalizer
from narrate.pipeline import NarrationConfig, _lang_code_for, plan_chapter, synthesize_chunks
from narrate.tts import AudioCache, get_engine

#: Kokoro's speaking rate at speed 1.0, as measured in the README. Drives only
#: the estimate shown before synthesis; the real duration is reported after.
WORDS_PER_MINUTE = 200

#: Pause where one section ends and the next begins. Longer than a paragraph
#: break (600 ms) so the change of topic is audible.
SECTION_SILENCE_MS = 1200

#: Past this, one long file is the wrong shape - chapters you can skip between
#: matter more. The command still runs, but suggests ``narrate speak``.
LONG_ARTICLE_MINUTES = 30

#: m4b is left out: it is a container for chapters, and an article is one track.
ARTICLE_FORMATS = ("mp3", "m4a", "opus", "wav")


@dataclass
class Section:
    """One heading's worth of the document, already cut into engine chunks."""

    chapter: Chapter
    chunks: list[Chunk]

    @property
    def word_count(self) -> int:
        # Counted after normalization, so "$4.2M" counts as the five words
        # that will actually be spoken.
        return sum(len(chunk.text.split()) for chunk in self.chunks)


@dataclass
class ArticlePlan:
    """Exactly what will be read, decided before any audio exists."""

    book: Book
    sections: list[Section]
    engine: object
    speed: float

    @property
    def word_count(self) -> int:
        return sum(section.word_count for section in self.sections)

    @property
    def chunk_count(self) -> int:
        return sum(len(section.chunks) for section in self.sections)

    def estimated_seconds(self, section: Section | None = None) -> float:
        words = section.word_count if section else self.word_count
        return words / (WORDS_PER_MINUTE * self.speed) * 60

    def script(self) -> str:
        """The engine's input: one chunk per line, a blank line between sections."""
        return "\n\n".join(
            "\n".join(chunk.text for chunk in section.chunks) for section in self.sections
        ) + "\n"


@dataclass
class ArticleResult:
    path: Path
    duration_seconds: float
    elapsed_seconds: float
    cache_summary: str


def load_article(path: Path) -> Book:
    """Ingest a markdown or text file. Anything else is a book."""
    path = Path(path)
    if path.suffix.lower() not in PLAINTEXT_FORMATS:
        raise ValueError(
            f"{path.name}: `narrate article` reads markdown or text "
            f"({', '.join(sorted(PLAINTEXT_FORMATS))}). For a PDF or EPUB, use "
            f"`narrate speak`, or `narrate extract` and point this command at "
            f"one of the chapter files it writes."
        )
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    return ingest_markdown(path)


def resolve_output(
    source: Path,
    *,
    output: Path | None,
    audio_format: str | None,
    output_dir: Path,
) -> tuple[Path, str]:
    """Settle the output path and format from whichever of the two was given.

    ``-o talk.m4a`` implies m4a; ``--format opus`` alone writes
    ``out/<name>.opus``; neither gives ``out/<name>.mp3``. An existing
    directory given as ``-o`` is where the file goes, not its name.
    """
    if output is not None and output.is_dir():
        output_dir, output = output, None

    if output is not None and output.suffix:
        implied = output.suffix.lstrip(".").lower()
        if audio_format is None:
            audio_format = implied
        elif audio_format.lower() != implied:
            raise ValueError(
                f"--format {audio_format} does not match the output file {output.name}"
            )

    audio_format = (audio_format or "mp3").lower()
    if audio_format not in ARTICLE_FORMATS:
        raise ValueError(
            f"Cannot write an article as {audio_format!r}. "
            f"Choose one of: {', '.join(ARTICLE_FORMATS)}"
        )

    if output is None:
        output = Path(output_dir) / f"{Path(source).stem}.{audio_format}"
    elif not output.suffix:
        output = output.with_suffix(f".{audio_format}")
    return output, audio_format


def plan_article(book: Book, config: NarrationConfig, *, engine=None) -> ArticlePlan:
    """Normalize and segment every section without touching the model.

    Model-free on purpose: ``--dry-run`` needs the estimate and the script on a
    machine where the engine is not even installed.
    """
    engine = engine or get_engine(config.engine, lang_code=_lang_code_for(config))
    normalizer = get_normalizer(
        lexicon=Lexicon.load(config.lexicon_path),
        keep_citations=config.keep_citations,
    )

    sections = []
    for chapter in book.narratable_chapters():
        chunks = plan_chapter(chapter, engine=engine, normalizer=normalizer)
        if chunks:
            sections.append(Section(chapter=chapter, chunks=chunks))

    return ArticlePlan(book=book, sections=sections, engine=engine, speed=config.speed)


def render_article(
    plan: ArticlePlan,
    config: NarrationConfig,
    *,
    output: Path,
    on_chunk: Callable[[], None] | None = None,
) -> ArticleResult:
    """Synthesize the plan into one file at ``output``."""
    if not plan.sections:
        raise ValueError(f"{plan.book.title}: nothing to narrate")

    started = time.time()
    engine = plan.engine
    cache = AudioCache(
        Path(config.work_dir) / _work_name(plan.book) / "cache",
        sample_rate=engine.sample_rate,
        enabled=config.use_cache,
    )

    tracks = [
        synthesize_chunks(
            section.chunks, engine=engine, cache=cache, config=config, on_chunk=on_chunk
        )
        for section in plan.sections
    ]
    # The segmenter leaves no silence after a section's last chunk, so the
    # section break is added here - and not after the final one, where it
    # would be dead air.
    pieces = [(track, SECTION_SILENCE_MS) for track in tracks if track.size]
    if not pieces:
        raise ValueError(f"{plan.book.title}: rendered to silence")
    pieces[-1] = (pieces[-1][0], 0)
    audio = concat_with_silence(pieces, sample_rate=engine.sample_rate)

    write_chapter(
        audio,
        output,
        sample_rate=engine.sample_rate,
        audio_format=config.audio_format,
        bitrate=config.bitrate,
        normalize_loudness=config.normalize_loudness,
        tags={
            "title": plan.book.title,
            "album": plan.book.title,
            "artist": plan.book.author or "Unknown",
            "genre": "Speech",
        },
    )

    return ArticleResult(
        path=output,
        duration_seconds=len(audio) / engine.sample_rate,
        elapsed_seconds=time.time() - started,
        cache_summary=cache.summary(),
    )


def _work_name(book: Book) -> str:
    """Cache directory name, from the file rather than the title.

    A title edited in front matter would otherwise move the cache and
    re-synthesize everything.
    """
    if book.source_path is not None:
        return slugify(Path(book.source_path).stem)
    return book.slug


__all__ = [
    "ARTICLE_FORMATS",
    "ArticlePlan",
    "ArticleResult",
    "Section",
    "load_article",
    "plan_article",
    "render_article",
    "resolve_output",
]
