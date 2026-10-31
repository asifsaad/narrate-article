"""Assemble ``timings.json`` from a rendered document.

Inputs are what rendering already produced - each section's chapter, its
chunks and their engine results, and the timeline built from them. Nothing
here touches audio or the engine, so all of it is exercised by stub tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from narrate.document import Chapter
from narrate.readalong.align import is_word, locate_tokens, naive_display_spans, sentence_ids
from narrate.readalong.display import block_at, markup_spans, parse_blocks, visible_mask
from narrate.readalong.schema import SCHEMA_VERSION
from narrate.readalong.timeline import RenderedChunk, Timeline

#: Joins sections' Markdown into one display text, as the chapters were split.
SECTION_SEPARATOR = "\n\n"

#: Characters a sentence's display span extends over after its last word:
#: closing punctuation, and emphasis markers the page hides anyway.
_SENTENCE_TAIL = ".!?…\"'”’)]*_"


@dataclass(frozen=True)
class SectionText:
    """One section's display Markdown and its spoken sentences."""

    chapter: Chapter
    sentences: list[str]
    """Exactly the sentences the segmenter packed this section's chunks from."""


@dataclass
class BuildStats:
    words: int = 0
    shared_display_spans: int = 0
    """Words with no direct match in the display text, which shared the span
    between their neighbours (``$4.2M`` read as five words)."""
    token_misses: int = 0
    """Engine word tokens not found in the chunk's spoken text."""
    warnings: list[str] = field(default_factory=list)


def _r(value: float) -> float:
    return round(value, 3)


def build_timings(
    *,
    title: str,
    audio_name: str,
    voice: str,
    speed: float,
    sections: Sequence[SectionText],
    rendered: Sequence[RenderedChunk],
    timeline: Timeline,
    audio_offset_s: float = 0.0,
    timing_source: str = "engine",
) -> tuple[dict, BuildStats]:
    """The ``timings.json`` document for one rendered read-along."""
    stats = BuildStats()

    display = SECTION_SEPARATOR.join(section.chapter.markdown for section in sections)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for section in sections:
        ranges.append((cursor, cursor + len(section.chapter.markdown)))
        cursor += len(section.chapter.markdown) + len(SECTION_SEPARATOR)

    blocks = parse_blocks(display)
    markup = markup_spans(display, blocks)
    visible = visible_mask(len(display), markup)

    # Which spoken characters each token covers, and which sentence each
    # character of each chunk belongs to.
    token_spans: list[list[tuple[int, int] | None]] = []
    for item in rendered:
        spans, misses = locate_tokens(item.chunk.text, [w.text for w in item.result.words or []])
        token_spans.append(spans)
        stats.token_misses += misses

    char_sentences: list[list[int]] = [[] for _ in rendered]
    for index, section in enumerate(sections):
        members = [i for i, item in enumerate(rendered) if item.section == index]
        ids = sentence_ids([rendered[i].chunk.text for i in members], section.sentences)
        for i, per_char in zip(members, ids):
            char_sentences[i] = per_char

    # Spoken words in playback order, each with its sentence key.
    entries: list[dict] = []
    last_key: tuple[int, int] = (0, 0)
    for token in timeline.tokens:
        if not is_word(token.text):
            continue
        item = rendered[token.rendered]
        spoken = token_spans[token.rendered][token.token]
        per_char = char_sentences[token.rendered]
        if spoken is not None and spoken[0] < len(per_char):
            last_key = (item.section, per_char[spoken[0]])
        entries.append(
            {"token": token, "section": item.section, "key": last_key, "src": None}
        )

    for index, (start, end) in enumerate(ranges):
        members = [entry for entry in entries if entry["section"] == index]
        spans, shared = naive_display_spans(
            display, visible, [entry["token"].text for entry in members], start=start, end=end
        )
        stats.shared_display_spans += shared
        for entry, src in zip(members, spans):
            entry["src"] = src

    # Sentences, numbered in order of first appearance; only those with words.
    order: dict[tuple[int, int], int] = {}
    for entry in entries:
        order.setdefault(entry["key"], len(order))

    groups: list[list[dict]] = [[] for _ in order]
    for entry in entries:
        groups[order[entry["key"]]].append(entry)

    sentences_json = []
    for sid, group in enumerate(groups):
        a = min(entry["src"][0] for entry in group)
        b = max(entry["src"][1] for entry in group)
        while b < len(display) and display[b] in _SENTENCE_TAIL:
            b += 1
        block = block_at(blocks, a)
        sentences_json.append(
            {
                "id": sid,
                "block": block.id if block else 0,
                "src": [a, b],
                "start": _r(min(entry["token"].start for entry in group)),
                "end": _r(max(entry["token"].end for entry in group)),
            }
        )

    words_json = []
    for entry in sorted(entries, key=lambda e: e["token"].start):
        token = entry["token"]
        start = _r(token.start)
        words_json.append(
            {
                "spoken": token.text,
                "start": start,
                "end": max(start, _r(token.end)),
                "src": list(entry["src"]),
                "sentence": order[entry["key"]],
            }
        )
    stats.words = len(words_json)

    data = {
        "version": SCHEMA_VERSION,
        "title": title,
        "audio": audio_name,
        "duration_s": _r(timeline.duration),
        "audio_offset_s": _r(audio_offset_s),
        "voice": voice,
        "speed": speed,
        "timing_source": timing_source,
        "display_text": display,
        "markup": [list(span) for span in markup],
        "blocks": [block.to_json() for block in blocks],
        "sentences": sentences_json,
        "words": words_json,
    }
    return data, stats


__all__ = ["BuildStats", "SECTION_SEPARATOR", "SectionText", "build_timings"]
