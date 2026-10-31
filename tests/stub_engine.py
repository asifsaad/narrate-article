"""A deterministic engine that reports word timings, for tests.

Not a test module (no ``test_`` prefix), so pytest imports it only when a test
does. The existing ``StubEngine`` copies in ``test_pipeline.py`` and
``test_article.py`` are left untouched: they have no timings, which is exactly
the engine shape the fallback paths need covering for.
"""

from __future__ import annotations

import re

import numpy as np

from narrate.tts.base import SynthesisResult, Voice, WordTiming

SAMPLE_RATE = 24_000

#: Kokoro-like tokens: words (with inner apostrophes and hyphens) and single
#: punctuation marks, each its own token.
_TOKEN = re.compile(r"[\w'’-]+|[^\w\s]")


def token_samples(token: str, *, speed: float = 1.0, sample_rate: int = SAMPLE_RATE) -> int:
    """100 ms per five characters, slowed or sped like a real engine."""
    return round(sample_rate * 0.1 * len(token) / 5 / speed)


class TimedStubEngine:
    """Audio is the tokens laid end to end; timings are exact sample counts.

    Each token is ``100 ms x len / 5`` long, divided by ``speed``, with no
    gaps, so the expected time of any token is a plain cumulative sum. Every
    token's audio has its own level, which lets tests check that the bytes
    ``synthesize`` and ``synthesize_with_timings`` return are the same.
    """

    name = "stub-timed"
    sample_rate = SAMPLE_RATE
    max_chars = 200

    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_voices(self) -> list[Voice]:
        return [Voice("test", "Test")]

    def synthesize(self, text: str, *, voice: str = "test", speed: float = 1.0) -> np.ndarray:
        return self.synthesize_with_timings(text, voice=voice, speed=speed).audio

    def synthesize_with_timings(
        self, text: str, *, voice: str = "test", speed: float = 1.0
    ) -> SynthesisResult:
        self.calls.append(text)
        parts: list[np.ndarray] = []
        words: list[WordTiming] = []
        cursor = 0
        for i, token in enumerate(_TOKEN.findall(text)):
            samples = token_samples(token, speed=speed)
            parts.append(np.full(samples, 0.05 + 0.001 * (i % 50), dtype=np.float32))
            words.append(
                WordTiming(
                    text=token,
                    start=cursor / self.sample_rate,
                    end=(cursor + samples) / self.sample_rate,
                )
            )
            cursor += samples
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate, words=words)


class UntimedStubEngine:
    """The shape of an engine that cannot report timings."""

    name = "stub-untimed"
    sample_rate = SAMPLE_RATE
    max_chars = 200

    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_voices(self) -> list[Voice]:
        return [Voice("test", "Test")]

    def synthesize(self, text: str, *, voice: str = "test", speed: float = 1.0) -> np.ndarray:
        self.calls.append(text)
        return np.full(max(len(text) * 10, 1), 0.1, dtype=np.float32)
