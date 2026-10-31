"""Place every engine token on the timeline of the finished audio.

A token's absolute time is its chunk-relative time plus everything before its
chunk: earlier chunks' audio, their trailing pauses (250 / 600 / 800 ms) and
the section pauses (1200 ms). All of it is counted in integer samples with the
same helpers the audio is assembled with, and converted to seconds once - never
measured back from the encoded file, which would inherit codec padding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from narrate.article import section_starts
from narrate.audio import silence_samples
from narrate.document import Chunk
from narrate.tts.base import SynthesisResult


@dataclass(frozen=True)
class RenderedChunk:
    """One chunk as it was synthesized, with its position in the document."""

    chunk: Chunk
    result: SynthesisResult
    section: int
    """0-based index of the section (article heading) the chunk belongs to."""


@dataclass(frozen=True)
class PlacedToken:
    """An engine token at its absolute time in the finished audio."""

    text: str
    start: float
    end: float
    rendered: int
    """Index into the flat list of rendered chunks."""
    token: int
    """Index of the token within its chunk's ``result.words``."""


@dataclass(frozen=True)
class Timeline:
    tokens: list[PlacedToken]
    chunk_starts: list[int]
    """Sample at which each rendered chunk's audio begins."""
    total_samples: int
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.total_samples / self.sample_rate


def section_track_samples(chunks: Sequence[RenderedChunk], *, sample_rate: int) -> list[int]:
    """Where each chunk starts within its section track, plus the track length.

    Mirrors ``concat_with_silence``: a chunk's audio, then its trailing pause.
    """
    offsets = []
    cursor = 0
    for rendered in chunks:
        offsets.append(cursor)
        cursor += rendered.result.audio.size
        cursor += silence_samples(rendered.chunk.trailing_silence_ms, sample_rate=sample_rate)
    return [*offsets, cursor]


def build_timeline(sections: Sequence[Sequence[RenderedChunk]], *, sample_rate: int) -> Timeline:
    """Absolute times for every token of every chunk, in playback order.

    ``sections`` are in document order, each a list of its rendered chunks -
    exactly what was joined into the audio by ``article.join_sections``.
    """
    per_section = [section_track_samples(chunks, sample_rate=sample_rate) for chunks in sections]
    starts = section_starts([offsets[-1] for offsets in per_section], sample_rate=sample_rate)

    tokens: list[PlacedToken] = []
    chunk_starts: list[int] = []
    total = 0
    flat = 0
    for chunks, offsets, section_start in zip(sections, per_section, starts):
        for position, rendered in enumerate(chunks):
            begin = (section_start or 0) + offsets[position]
            chunk_starts.append(begin)
            if section_start is not None:
                audio_end = (begin + rendered.result.audio.size) / sample_rate
                for index, word in enumerate(rendered.result.words or []):
                    start = min(begin / sample_rate + word.start, audio_end)
                    end = min(begin / sample_rate + word.end, audio_end)
                    tokens.append(PlacedToken(word.text, start, max(start, end), flat, index))
            flat += 1
        if section_start is not None:
            total = section_start + offsets[-1]

    return Timeline(tokens=tokens, chunk_starts=chunk_starts, total_samples=total, sample_rate=sample_rate)


__all__ = ["PlacedToken", "RenderedChunk", "Timeline", "build_timeline", "section_track_samples"]
