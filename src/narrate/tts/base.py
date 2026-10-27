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
