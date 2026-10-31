"""The TTS engine seam.

Everything model-specific lives behind this protocol. Swapping Kokoro for
Qwen3-TTS (voice cloning) or Piper (fast previews) means adding one module here
and nothing else - the pipeline, cache and audio assembly are engine-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Voice:
    """A selectable voice."""

    id: str
    """Passed back to :meth:`SynthesisEngine.synthesize`."""

    name: str
    language: str = "en-us"
    gender: str | None = None
    notes: str = ""


class SynthesisError(RuntimeError):
    """Raised when an engine cannot render a chunk."""


class EngineUnavailableError(RuntimeError):
    """Raised when an engine's dependencies or weights are missing.

    Separate from SynthesisError because it is a setup problem with an
    actionable fix, not a failure on particular input.
    """


@runtime_checkable
class SynthesisEngine(Protocol):
    """Text in, mono float32 audio out."""

    name: str
    sample_rate: int

    max_chars: int
    """Longest chunk the engine handles well. The segmenter packs to this.

    A budget rather than a hard limit: engines that exceed it usually degrade
    (prosody drift, dropped clauses) rather than erroring, which is exactly the
    failure the segmenter exists to prevent.
    """

    def list_voices(self) -> list[Voice]: ...

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> np.ndarray:
        """Render ``text`` to mono float32 samples in [-1, 1]."""
        ...


# --- word timings ---------------------------------------------------------


@dataclass(frozen=True)
class WordTiming:
    """One engine token and when it is heard.

    Tokens are what the engine saw, punctuation included: a comma is its own
    token and its time is the pause it produced. Read-along keeps them as
    alignment anchors and never highlights them.
    """

    text: str
    """Token text exactly as the engine saw it."""

    start: float
    """Seconds from the start of this chunk's audio."""

    end: float


@dataclass(frozen=True, eq=False)
class SynthesisResult:
    """Audio plus, where the engine can say, when each token is spoken.

    ``eq=False`` because comparing two numpy arrays has no single truth value.
    """

    audio: np.ndarray
    """Mono float32, exactly what :meth:`SynthesisEngine.synthesize` returns."""

    sample_rate: int

    words: list[WordTiming] | None
    """Token timings, in order, relative to the start of ``audio``. ``None``
    when the engine cannot report them - see ``tts.timing_estimate``."""

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0


@runtime_checkable
class TimedSynthesisEngine(SynthesisEngine, Protocol):
    """An engine that can also say when each word is spoken.

    Optional. ``synthesize_with_timings`` must return exactly the audio
    ``synthesize`` would, so that cached chunks are interchangeable between
    the two and existing output never changes.
    """

    def synthesize_with_timings(
        self, text: str, *, voice: str, speed: float = 1.0
    ) -> SynthesisResult: ...


def synthesize_with_timings(
    engine: SynthesisEngine, text: str, *, voice: str, speed: float = 1.0
) -> SynthesisResult:
    """Render ``text`` with timings if ``engine`` can report them.

    Engines that only implement :meth:`SynthesisEngine.synthesize` get
    ``words=None``; nothing is estimated here, so a cache built from this
    result records what the engine actually knew.
    """
    method = getattr(engine, "synthesize_with_timings", None)
    if callable(method):
        return method(text, voice=voice, speed=speed)
    audio = engine.synthesize(text, voice=voice, speed=speed)
    return SynthesisResult(audio=audio, sample_rate=engine.sample_rate, words=None)
