"""Kokoro-82M.

Apache-2.0, 82M parameters, and competitive on blind-preference leaderboards
with models eighty times its size. Chosen over higher-ranked open-weights models
because most of those ship under non-commercial licences or impose 20-30 second
clip caps that make long-form narration impossible.
"""

from __future__ import annotations

import logging
import os
import warnings

import numpy as np

from narrate.tts.base import EngineUnavailableError, SynthesisError, Voice

log = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24_000

#: Kokoro's internal budget is ~510 phoneme tokens per forward pass; beyond it
#: the model silently truncates. Characters are a proxy for phonemes, so this
#: sits well under the limit - the cost of a chunk that is too small is one
#: extra pause, while too large loses words outright.
KOKORO_MAX_CHARS = 400

#: Kokoro voice ids encode language and gender: ``af_heart`` is American
#: female, ``bm_george`` British male. Only the voices worth narrating with are
#: listed; quality varies noticeably across the full set of 54.
_VOICES: tuple[Voice, ...] = (
    Voice("af_heart", "Heart", "en-us", "female", "warm, best overall for narration"),
    Voice("af_bella", "Bella", "en-us", "female", "expressive"),
    Voice("af_nicole", "Nicole", "en-us", "female", "soft, close-mic feel"),
    Voice("af_sarah", "Sarah", "en-us", "female", "neutral"),
    Voice("af_sky", "Sky", "en-us", "female", "bright"),
    Voice("am_michael", "Michael", "en-us", "male", "steady, good for long-form"),
    Voice("am_adam", "Adam", "en-us", "male", "deeper"),
    Voice("am_puck", "Puck", "en-us", "male", "lively"),
    Voice("bf_emma", "Emma", "en-gb", "female", "British, strong for narration"),
    Voice("bf_isabella", "Isabella", "en-gb", "female", "British"),
    Voice("bm_george", "George", "en-gb", "male", "British, measured"),
    Voice("bm_lewis", "Lewis", "en-gb", "male", "British, deeper"),
)

DEFAULT_VOICE = "af_heart"


class KokoroEngine:
    """Kokoro-82M via the ``kokoro`` package."""

    name = "kokoro"
    sample_rate = KOKORO_SAMPLE_RATE
    max_chars = KOKORO_MAX_CHARS

    def __init__(self, *, device: str | None = None, lang_code: str = "a") -> None:
        """``lang_code`` selects the G2P: ``a`` American, ``b`` British.

        It must match the voice prefix - a British voice driven by American
        phonemes produces a subtly wrong accent rather than an error.
        """
        self._lang_code = lang_code
        self._device = device or _default_device()
        self._pipeline = None

    # -- lifecycle ---------------------------------------------------------

    def _ensure_loaded(self):
        """Load lazily so --list-voices and --extract-only stay instant."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise EngineUnavailableError(
                "Kokoro is not installed. Run: uv sync --extra kokoro"
            ) from exc

        log.info("Loading Kokoro on %s (lang_code=%r)", self._device, self._lang_code)
        with warnings.catch_warnings():
            # The RNN dropout and weight-norm warnings come from Kokoro's own
            # architecture and are not actionable here.
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                self._pipeline = KPipeline(
                    lang_code=self._lang_code, device=self._device
                )
            except Exception as exc:
                raise EngineUnavailableError(
                    f"Could not load Kokoro on device {self._device!r}: {exc}. "
                    f"If this is an espeak-ng error, install it with "
                    f"`brew install espeak-ng`."
                ) from exc

        return self._pipeline

    # -- SynthesisEngine ---------------------------------------------------

    def list_voices(self) -> list[Voice]:
        return list(_VOICES)

    def synthesize(
        self, text: str, *, voice: str = DEFAULT_VOICE, speed: float = 1.0
    ) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        pipeline = self._ensure_loaded()

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                segments = list(pipeline(text, voice=voice, speed=speed))
        except Exception as exc:
            raise SynthesisError(
                f"Kokoro failed on {text[:60]!r}...: {exc}"
            ) from exc

        audio = _collect(segments)
        if audio.size == 0:
            raise SynthesisError(f"Kokoro returned no audio for {text[:60]!r}")
        return audio


def _collect(segments) -> np.ndarray:
    """Flatten KPipeline output into one mono float32 array.

    KPipeline yields per-sentence results and may split internally, so a single
    call can produce several segments that must be joined in order.
    """
    parts: list[np.ndarray] = []

    for segment in segments:
        # Result objects expose .audio; older versions yield (gs, ps, audio).
        chunk = getattr(segment, "audio", None)
        if chunk is None and isinstance(segment, tuple) and len(segment) >= 3:
            chunk = segment[2]
        if chunk is None:
            continue

        if hasattr(chunk, "detach"):  # torch.Tensor
            chunk = chunk.detach().cpu().numpy()

        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size:
            parts.append(chunk)

    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def _default_device() -> str:
    """Prefer MPS on Apple Silicon, then CUDA, then CPU.

    NARRATE_DEVICE overrides, which matters because MPS occasionally regresses
    on specific torch releases and CPU is a viable fallback for an 82M model.
    """
    override = os.environ.get("NARRATE_DEVICE")
    if override:
        return override

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def lang_code_for_voice(voice: str) -> str:
    """Derive the G2P language code from a voice id.

    ``bf_emma`` and ``bm_george`` need British phonemes; everything else uses
    American. Getting this wrong is inaudible in testing but wrong in aggregate.
    """
    return "b" if voice.startswith(("bf_", "bm_")) else "a"
