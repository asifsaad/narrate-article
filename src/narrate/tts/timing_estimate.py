"""Word timings for engines that cannot report their own.

A safety net, not a substitute: spreading a chunk's duration across its words
by length keeps a read-along highlight roughly in place - typically within a
word or two over a sentence - but it cannot know where the engine paused or
lingered. Kokoro reports real timings and never needs this.
"""

from __future__ import annotations

import re

from narrate.tts.base import SynthesisResult, WordTiming

#: Whitespace-separated words; punctuation stays attached ("dollars,"), which
#: read-along alignment tolerates because it searches for the token text.
_WORD = re.compile(r"\S+")


def estimate_word_timings(text: str, duration: float) -> list[WordTiming]:
    """Spread ``duration`` seconds over the words of ``text`` by length.

    Each word's share is proportional to its character count, so "a" is brief
    and "consequently" is long. Words tile the chunk with no gaps: the first
    starts at 0 and the last ends at ``duration``.
    """
    words = _WORD.findall(text)
    if not words or duration <= 0:
        return []

    total = sum(len(word) for word in words)
    timings: list[WordTiming] = []
    elapsed = 0
    for word in words:
        start = duration * elapsed / total
        elapsed += len(word)
        timings.append(WordTiming(text=word, start=start, end=duration * elapsed / total))
    return timings


def with_estimated_timings(result: SynthesisResult, text: str) -> SynthesisResult:
    """``result`` unchanged if it has timings; otherwise estimated ones."""
    if result.words is not None:
        return result
    return SynthesisResult(
        audio=result.audio,
        sample_rate=result.sample_rate,
        words=estimate_word_timings(text, result.duration),
    )


__all__ = ["estimate_word_timings", "with_estimated_timings"]
