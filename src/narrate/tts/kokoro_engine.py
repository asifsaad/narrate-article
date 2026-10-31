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
from collections.abc import Sequence

import numpy as np

from narrate.tts.base import (
    EngineUnavailableError,
    SynthesisError,
    SynthesisResult,
    Voice,
    WordTiming,
)

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

#: Kokoro's duration predictor counts frames of 600 samples (25 ms at 24 kHz).
#: Its timestamps are kept in half-frames so a space can be split between the
#: words either side of it - hence 80 half-frames per second.
HALF_FRAMES_PER_SECOND = 80


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

        audio = _collect(self._run(text, voice=voice, speed=speed))
        if audio.size == 0:
            raise SynthesisError(f"Kokoro returned no audio for {text[:60]!r}")
        return audio

    def synthesize_with_timings(
        self, text: str, *, voice: str = DEFAULT_VOICE, speed: float = 1.0
    ) -> SynthesisResult:
        """Exactly :meth:`synthesize`'s audio, plus per-token timings.

        The same pipeline call, so the samples are identical - the tokens it
        yields alongside the audio are simply kept instead of discarded. A
        failure to read them costs only the timings, never the audio.
        """
        text = text.strip()
        if not text:
            return SynthesisResult(np.zeros(0, dtype=np.float32), self.sample_rate, [])

        results = self._run(text, voice=voice, speed=speed)
        audio = _collect(results)
        if audio.size == 0:
            raise SynthesisError(f"Kokoro returned no audio for {text[:60]!r}")

        vocab = getattr(getattr(self._pipeline, "model", None), "vocab", None)
        try:
            words = token_timings(results, sample_rate=self.sample_rate, vocab=vocab)
        except Exception as exc:  # timings are a bonus; the audio is the product
            log.warning("Could not read Kokoro timings for %r: %s", text[:60], exc)
            words = None
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate, words=words)

    def _run(self, text: str, *, voice: str, speed: float) -> list:
        """One pipeline call, every result kept. Shared by both synthesize paths."""
        pipeline = self._ensure_loaded()

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                return list(pipeline(text, voice=voice, speed=speed))
        except Exception as exc:
            raise SynthesisError(
                f"Kokoro failed on {text[:60]!r}...: {exc}"
            ) from exc


def _samples(segment) -> np.ndarray | None:
    """One pipeline result's audio as mono float32, or None if it has none."""
    # Result objects expose .audio; older versions yield (gs, ps, audio).
    chunk = getattr(segment, "audio", None)
    if chunk is None and isinstance(segment, tuple) and len(segment) >= 3:
        chunk = segment[2]
    if chunk is None:
        return None

    if hasattr(chunk, "detach"):  # torch.Tensor
        chunk = chunk.detach().cpu().numpy()

    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    return chunk if chunk.size else None


def _collect(segments) -> np.ndarray:
    """Flatten KPipeline output into one mono float32 array.

    KPipeline yields per-sentence results and may split internally, so a single
    call can produce several segments that must be joined in order.
    """
    parts = [chunk for chunk in map(_samples, segments) if chunk is not None]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


# --- timings --------------------------------------------------------------


def token_timings(results, *, sample_rate: int, vocab=None) -> list[WordTiming] | None:
    """Every token the pipeline yielded, timed against the joined audio.

    The pipeline splits its input on newlines and again at 510 phonemes, and
    each result's times start from zero, so each is offset by the samples of
    the results before it - counted exactly as :func:`_collect` joins them.
    ``None`` if no result carried tokens (an old kokoro, or a non-English G2P).
    """
    words: list[WordTiming] = []
    any_tokens = False
    cursor = 0

    for result in results:
        audio = _samples(result)
        if audio is None:
            continue
        tokens = getattr(result, "tokens", None) or []
        if tokens:
            any_tokens = True
            offset = cursor / sample_rate
            duration = audio.size / sample_rate
            times = _token_times(
                tokens,
                getattr(result, "phonemes", "") or "",
                getattr(result, "pred_dur", None),
                vocab,
            )
            for token, (start, end) in zip(tokens, _fill_gaps(times, duration)):
                words.append(WordTiming(text=token.text, start=offset + start, end=offset + end))
        cursor += audio.size

    return words if any_tokens else None


def _token_times(
    tokens, phonemes: str, pred_dur, vocab
) -> list[tuple[float | None, float | None]]:
    """Token times from the duration predictor, falling back to upstream's.

    Upstream's ``join_timestamps`` is used only when there is no ``pred_dur``
    to recompute from, or the recomputation fails.
    """
    upstream = [(getattr(t, "start_ts", None), getattr(t, "end_ts", None)) for t in tokens]
    if pred_dur is None:
        return upstream
    try:
        durations = pred_dur.tolist() if hasattr(pred_dur, "tolist") else list(pred_dur)
        return times_from_durations(tokens, phonemes, [int(d) for d in durations], vocab)
    except Exception as exc:
        log.debug("Falling back to upstream timestamps: %s", exc)
        return upstream


def times_from_durations(
    tokens, phonemes: str, durations: Sequence[int], vocab=None
) -> list[tuple[float | None, float | None]]:
    """Recompute token times from ``pred_dur`` frames, in seconds.

    Mirrors ``KPipeline.join_timestamps`` - same half-frame convention, same
    ``-3`` frame lead-in, word boundaries in the middle of the space between
    words - and fixes two things it gets wrong:

    - A token with no phonemes but trailing whitespace (``~``, ``★``) makes it
      advance its index twice, so every later token is given the next
      phoneme's duration and the last ones run off the end as ``None``.
    - It assumes one model input per phoneme character, but
      ``KModel.forward`` drops characters missing from the vocab.

    Positions follow how the pipeline built ``phonemes`` (``tokens_to_ps``:
    phonemes plus a space per whitespace, stripped); indices follow the vocab
    filter. Tokens past a 510-phoneme truncation, and tokens with no phonemes,
    get ``None``.
    """
    built = "".join((t.phonemes or "") + (" " if t.whitespace else "") for t in tokens)
    lead = len(built) - len(built.lstrip())

    # Model input index of every phoneme character; BOS is index 0.
    index_of: list[int | None] = []
    position = 1
    for char in phonemes:
        if vocab is None or char in vocab:
            index_of.append(position)
            position += 1
        else:
            index_of.append(None)

    def frames(a: int, b: int) -> int | None:
        if a < 0 or b > len(index_of):
            return None
        return sum(durations[i] for i in index_of[a:b] if i is not None and i < len(durations))

    if not durations:
        return [(None, None)] * len(tokens)

    left = right = 2 * max(0, durations[0] - 3)
    cursor = -lead
    times: list[tuple[float | None, float | None]] = []

    for token in tokens:
        size = len(token.phonemes or "")
        space = 1 if token.whitespace else 0

        if not size:
            gap = frames(cursor, cursor + space) if space else 0
            if gap:
                left = right + gap
                right = left + gap
            times.append((None, None))
            cursor += space
            continue

        token_frames = frames(cursor, cursor + size)
        if token_frames is None:
            times.append((None, None))
            cursor += size + space
            continue

        space_frames = (frames(cursor + size, cursor + size + space) or 0) if space else 0
        start = left
        left = right + 2 * token_frames + space_frames
        right = left + space_frames
        times.append((start / HALF_FRAMES_PER_SECOND, left / HALF_FRAMES_PER_SECOND))
        cursor += size + space

    return times


def _fill_gaps(
    times: list[tuple[float | None, float | None]], duration: float
) -> list[tuple[float, float]]:
    """Give every token a time, ordered and inside ``[0, duration]``.

    A missing start takes the previous token's end; a missing end takes the
    next known start, or ``duration`` for the last token. Starts never go
    backwards and no token outlasts its audio.
    """
    next_start: list[float | None] = [None] * len(times)
    upcoming: float | None = None
    for i in range(len(times) - 1, -1, -1):
        next_start[i] = upcoming
        if times[i][0] is not None:
            upcoming = times[i][0]

    filled: list[tuple[float, float]] = []
    previous_start = previous_end = 0.0
    for i, (start, end) in enumerate(times):
        if start is None:
            start = previous_end
        start = min(max(start, previous_start, 0.0), duration)
        if end is None:
            end = next_start[i] if next_start[i] is not None else duration
        end = min(max(end, start), duration)
        filled.append((start, end))
        previous_start, previous_end = start, end
    return filled


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
