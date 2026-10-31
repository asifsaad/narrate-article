"""``narrate read``: a document in; audio, timings and a player page out.

Planning is ``article``'s - the same normalizer, segmenter, engine and cache -
so a note read along sounds exactly like the same note as an article, and the
two share every cached chunk. What differs is that each chunk is rendered
through the timed path, and that the encoded file is checked against the
timeline before the page is written.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from narrate.article import (
    ArticlePlan,
    article_cache,
    article_tags,
    build_normalizer,
    join_sections,
)
from narrate.audio import concat_with_silence, write_chapter
from narrate.pipeline import NarrationConfig, synthesize_chunk_with_timings
from narrate.readalong.build import BuildStats, SectionText, build_timings
from narrate.readalong.encode import READALONG_FORMATS, EncodeCheck, check_encoded
from narrate.readalong.page import EMBED_WARN_BYTES, audio_data_uri, audio_link, render_page
from narrate.readalong.schema import validate_timings
from narrate.readalong.timeline import RenderedChunk, build_timeline
from narrate.segment import paragraph_sentences
from narrate.tts.timing_estimate import with_estimated_timings

log = logging.getLogger(__name__)

PAGE_NAME = "index.html"


@dataclass(frozen=True)
class ReadAlongPaths:
    directory: Path
    audio: Path
    timings: Path
    page: Path


@dataclass
class ReadAlongResult:
    paths: ReadAlongPaths
    duration_seconds: float
    elapsed_seconds: float
    cache_summary: str
    stats: BuildStats
    check: EncodeCheck
    timing_source: str
    warnings: list[str] = field(default_factory=list)


def readalong_paths(source: Path, *, output_dir: Path, audio_format: str) -> ReadAlongPaths:
    """``out/<name>/<name>.<fmt>``, ``<name>.timings.json`` and ``index.html``."""
    if audio_format not in READALONG_FORMATS:
        raise ValueError(
            f"Read-along audio must be one of: {', '.join(READALONG_FORMATS)} "
            f"(got {audio_format!r})"
        )
    name = Path(source).stem
    directory = Path(output_dir) / name
    return ReadAlongPaths(
        directory=directory,
        audio=directory / f"{name}.{audio_format}",
        timings=directory / f"{name}.timings.json",
        page=directory / PAGE_NAME,
    )


def render_readalong(
    plan: ArticlePlan,
    config: NarrationConfig,
    *,
    paths: ReadAlongPaths,
    embed_audio: bool = False,
    on_chunk: Callable[[], None] | None = None,
) -> ReadAlongResult:
    """Synthesize ``plan`` and write the audio, ``timings.json`` and the page."""
    if not plan.sections:
        raise ValueError(f"{plan.book.title}: nothing to narrate")

    started = time.time()
    engine = plan.engine
    sample_rate = engine.sample_rate
    cache = article_cache(plan.book, config, sample_rate=sample_rate)
    warnings: list[str] = []

    sections: list[list[RenderedChunk]] = []
    estimated = 0
    for index, section in enumerate(plan.sections):
        rendered = []
        for chunk in section.chunks:
            result = synthesize_chunk_with_timings(
                chunk, engine=engine, cache=cache, config=config
            )
            if result.words is None:
                result = with_estimated_timings(result, chunk.text)
                estimated += 1
            rendered.append(RenderedChunk(chunk=chunk, result=result, section=index))
            if on_chunk:
                on_chunk()
        sections.append(rendered)

    # Assembled exactly as ``render_article`` assembles it.
    tracks = [
        concat_with_silence(
            [(r.result.audio, r.chunk.trailing_silence_ms) for r in rendered],
            sample_rate=sample_rate,
        )
        for rendered in sections
    ]
    audio = join_sections(tracks, sample_rate=sample_rate)
    if not audio.size:
        raise ValueError(f"{plan.book.title}: rendered to silence")

    timeline = build_timeline(sections, sample_rate=sample_rate)
    if timeline.total_samples != audio.size:
        raise RuntimeError(
            f"Timeline is {timeline.total_samples} samples but the audio is {audio.size}; "
            f"word times would drift. This is a bug."
        )

    paths.directory.mkdir(parents=True, exist_ok=True)
    write_chapter(
        audio,
        paths.audio,
        sample_rate=sample_rate,
        audio_format=config.audio_format,
        bitrate=config.bitrate,
        normalize_loudness=config.normalize_loudness,
        tags=article_tags(plan.book),
        output_sample_rate=sample_rate,
    )

    check = check_encoded(
        paths.audio, audio, sample_rate=sample_rate, audio_format=config.audio_format
    )
    for problem in check.problems():
        warnings.append(f"{paths.audio.name}: {problem}; the highlight may drift.")

    normalizer = build_normalizer(config)
    texts = [
        SectionText(
            chapter=section.chapter,
            sentences=[
                sentence
                for _paragraph, sentences in paragraph_sentences(
                    normalizer.normalize(section.chapter.markdown)
                )
                for sentence in sentences
            ],
        )
        for section in plan.sections
    ]

    flat = [item for rendered in sections for item in rendered]
    source = "engine" if not estimated else ("estimated" if estimated == len(flat) else "mixed")
    timings, stats = build_timings(
        title=plan.book.title,
        audio_name=paths.audio.name,
        voice=config.voice,
        speed=config.speed,
        sections=texts,
        rendered=flat,
        timeline=timeline,
        audio_offset_s=check.applied_offset_s,
        timing_source=source,
    )
    if estimated:
        warnings.append(
            f"{estimated} of {len(flat)} chunks have estimated word timings; "
            f"the highlight will be approximate there."
        )

    errors = validate_timings(timings)
    if errors:
        raise RuntimeError("Generated timings are invalid (a bug): " + "; ".join(errors[:5]))

    paths.timings.write_text(dump_timings(timings), encoding="utf-8")

    if embed_audio:
        size = paths.audio.stat().st_size
        if size > EMBED_WARN_BYTES:
            warnings.append(
                f"Embedding {size / 1_048_576:.0f} MB of audio; the page will be slow to open."
            )
        audio_src = audio_data_uri(paths.audio)
    else:
        audio_src = audio_link(paths.audio)
    paths.page.write_text(render_page(timings, audio_src=audio_src), encoding="utf-8")

    return ReadAlongResult(
        paths=paths,
        duration_seconds=audio.size / sample_rate,
        elapsed_seconds=time.time() - started,
        cache_summary=cache.summary(),
        stats=stats,
        check=check,
        timing_source=source,
        warnings=warnings,
    )


def dump_timings(data: dict) -> str:
    """Pretty at the top level, one record per line below it.

    Readable, and diffable: re-rendering after an edit changes the lines of
    the words that moved, not one enormous line.
    """
    lines = ["{"]
    items = list(data.items())
    for position, (key, value) in enumerate(items):
        comma = "," if position < len(items) - 1 else ""
        if isinstance(value, list) and value and isinstance(value[0], dict):
            lines.append(f"  {json.dumps(key)}: [")
            for i, record in enumerate(value):
                tail = "," if i < len(value) - 1 else ""
                lines.append("    " + json.dumps(record, ensure_ascii=False) + tail)
            lines.append(f"  ]{comma}")
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


__all__ = [
    "PAGE_NAME",
    "ReadAlongPaths",
    "ReadAlongResult",
    "dump_timings",
    "readalong_paths",
    "render_readalong",
]
