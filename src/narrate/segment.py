"""Split normalized text into chunks sized for the TTS engine.

Two rules drive everything here:

1. **Never split mid-sentence.** Prosody is planned per utterance, so a sentence
   cut in half produces a falling intonation in the middle of a clause and a
   cold start on the remainder. It is the most audible defect in generated
   narration.
2. **Silence is decided here, not at synthesis time.** Paragraph structure is
   still visible at this stage; by the time a chunk reaches the engine it is
   just a string.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

from narrate.document import Chunk

log = logging.getLogger(__name__)

#: Pause after a sentence inside a paragraph.
SENTENCE_SILENCE_MS = 250

#: Pause between paragraphs. Roughly double the sentence gap - enough to hear
#: as a topic shift without sounding like a fault.
PARAGRAPH_SILENCE_MS = 600

#: Pause after a heading, which usually introduces a new section.
HEADING_SILENCE_MS = 800

#: Clause boundaries, used only when a single sentence exceeds the engine
#: budget and something has to give.
_CLAUSE_BOUNDARY = re.compile(r"(?<=[;:,])\s+")

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@lru_cache(maxsize=4)
def _segmenter(language: str = "en"):
    import pysbd

    return pysbd.Segmenter(language=language, clean=False)


def split_sentences(text: str, *, language: str = "en") -> list[str]:
    """Sentence-split with pysbd.

    pysbd is rule-based and handles the abbreviation cases that break naive
    ``.``-splitting. The normalizer has already expanded ``Dr.`` and ``e.g.``
    by this point, but titles and initials in names still occur.
    """
    return [s.strip() for s in _segmenter(language).segment(text) if s.strip()]


def segment_chapter(
    text: str,
    *,
    chapter_index: int,
    max_chars: int,
    language: str = "en",
) -> list[Chunk]:
    """Turn one chapter's normalized text into ordered :class:`Chunk` objects."""
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]

    for paragraph in paragraphs:
        sentences = split_sentences(paragraph, language=language)
        packed = _pack(sentences, max_chars=max_chars)

        for i, body in enumerate(packed):
            is_last = i == len(packed) - 1
            chunks.append(
                Chunk(
                    text=body,
                    chapter_index=chapter_index,
                    order=len(chunks),
                    trailing_silence_ms=(
                        _paragraph_gap(paragraph) if is_last else SENTENCE_SILENCE_MS
                    ),
                )
            )

    if chunks:
        # Trailing silence on the final chunk becomes dead air at the end of the
        # file, where the player already provides a gap.
        chunks[-1].trailing_silence_ms = 0

    return chunks


def _paragraph_gap(paragraph: str) -> int:
    """Headings get a longer pause than ordinary paragraphs.

    By this point markdown is gone, so a heading is identified by shape: a short
    line with no terminal punctuation beyond the full stop the normalizer added.
    """
    words = paragraph.split()
    if len(words) <= 8 and paragraph.count(".") <= 1:
        return HEADING_SILENCE_MS
    return PARAGRAPH_SILENCE_MS


def _pack(sentences: list[str], *, max_chars: int) -> list[str]:
    """Greedily fill chunks up to ``max_chars`` on sentence boundaries.

    Greedy rather than balanced: an engine's prosody depends on the text it is
    given, so re-flowing sentences to equalise chunk sizes would change how the
    audio sounds for no benefit.
    """
    packed: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_split_oversized(sentence, max_chars=max_chars))
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            packed.append(current)
            current = sentence

    if current:
        packed.append(current)

    return packed


def _split_oversized(sentence: str, *, max_chars: int) -> list[str]:
    """Break a sentence that cannot fit, least-bad option first.

    Clause boundaries are tried before a hard word split, because a pause at a
    comma is one a listener would expect anyway.
    """
    log.debug("Sentence exceeds %d chars, splitting: %.60s...", max_chars, sentence)

    parts = _CLAUSE_BOUNDARY.split(sentence)
    if all(len(p) <= max_chars for p in parts) and len(parts) > 1:
        return _pack(parts, max_chars=max_chars)

    # No usable clause boundary: pack whole words to the limit. Only reachable
    # with pathological input (an unpunctuated wall of text).
    out: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                out.append(current)
            current = word[:max_chars]
    if current:
        out.append(current)
    return out
