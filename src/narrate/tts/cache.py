"""Content-addressed audio cache.

The reason the editable-markdown checkpoint is usable in practice. Chunks are
keyed by a hash of their text plus render settings, so fixing one sentence in a
300-page book re-synthesizes one chunk instead of all of them.

Each entry is ``<key>.wav``, plus a ``<key>.timings.json`` sidecar holding the
engine's word timings for read-along. The key is the same for both, so the
audio a ``speak`` run cached is the audio a ``read`` run reuses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from narrate.document import Chunk
from narrate.tts.base import SynthesisResult, WordTiming

log = logging.getLogger(__name__)

#: Sidecar suffix, appended to the key in place of ``.wav``.
TIMINGS_SUFFIX = ".timings.json"

#: Bumped if the sidecar layout changes; other versions read as missing.
TIMINGS_VERSION = 1

#: A sidecar whose last word ends this far past its audio is from another
#: render of the same key and cannot be trusted.
_TIMING_SLACK_S = 0.05


class AudioCache:
    """Stores rendered chunks as WAV files under a cache directory.

    WAV rather than a compressed format: these are intermediates that get
    concatenated and re-encoded, so lossy round-trips would compound.
    """

    def __init__(self, directory: Path, *, sample_rate: int, enabled: bool = True):
        self.directory = Path(directory)
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.stale = 0
        """Entries with audio but no timings, re-synthesized by the timed path."""
        self._warned_stale = False

        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, chunk: Chunk, *, engine: str, voice: str, speed: float) -> Path:
        # Two-character prefix directory keeps any single directory from holding
        # tens of thousands of entries, which some filesystems handle poorly.
        key = chunk.cache_key(engine, voice, speed)
        return self.directory / key[:2] / f"{key}.wav"

    def timings_path_for(
        self, chunk: Chunk, *, engine: str, voice: str, speed: float
    ) -> Path:
        audio = self.path_for(chunk, engine=engine, voice=voice, speed=speed)
        return audio.with_name(audio.stem + TIMINGS_SUFFIX)

    def get(
        self, chunk: Chunk, *, engine: str, voice: str, speed: float
    ) -> np.ndarray | None:
        """Cached audio, whether or not timings were stored with it."""
        if not self.enabled:
            return None

        audio = self._read_audio(self.path_for(chunk, engine=engine, voice=voice, speed=speed))
        if audio is None:
            self.misses += 1
            return None

        self.hits += 1
        return audio

    def get_with_timings(
        self, chunk: Chunk, *, engine: str, voice: str, speed: float
    ) -> SynthesisResult | None:
        """Cached audio *and* timings, or ``None`` to re-synthesize.

        Audio cached before timings existed counts as a miss: re-rendering one
        chunk costs a second, while estimated timings would put the highlight
        visibly off for good. A sidecar recording ``"words": null`` is a hit -
        that engine simply cannot report timings, and asking again won't help.
        """
        if not self.enabled:
            return None

        audio_path = self.path_for(chunk, engine=engine, voice=voice, speed=speed)
        timings_path = self.timings_path_for(chunk, engine=engine, voice=voice, speed=speed)

        if audio_path.is_file() and not timings_path.is_file():
            self.stale += 1
            if not self._warned_stale:
                log.warning(
                    "Cached audio from before word timings existed; "
                    "re-synthesizing those chunks once."
                )
                self._warned_stale = True
            self.misses += 1
            return None

        audio = self._read_audio(audio_path)
        if audio is None:
            self.misses += 1
            return None

        words = self._read_timings(timings_path, duration=len(audio) / self.sample_rate)
        if words is False:
            self.misses += 1
            return None

        self.hits += 1
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate, words=words)

    def _read_audio(self, path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None

        try:
            import soundfile as sf

            audio, rate = sf.read(path, dtype="float32", always_2d=False)
        except Exception as exc:
            # A truncated file from an interrupted run must not be fatal.
            log.warning("Discarding unreadable cache entry %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
            return None

        if rate != self.sample_rate:
            log.debug("Cache entry %s has wrong sample rate; ignoring", path.name)
            return None

        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _read_timings(self, path: Path, *, duration: float) -> list[WordTiming] | None | bool:
        """Parsed sidecar words, ``None`` for "engine has none", ``False`` if unusable."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != TIMINGS_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            raw = data["words"]
            words = None if raw is None else [
                WordTiming(text=str(text), start=float(start), end=float(end))
                for text, start, end in raw
            ]
        except Exception as exc:
            log.warning("Discarding unreadable timings %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
            return False

        if words and words[-1].end > duration + _TIMING_SLACK_S:
            log.warning("Discarding timings %s that outlast their audio", path.name)
            path.unlink(missing_ok=True)
            return False
        return words

    def put(
        self,
        chunk: Chunk,
        audio: np.ndarray,
        *,
        engine: str,
        voice: str,
        speed: float,
    ) -> None:
        """Cache audio alone. Any timings stored for this key are removed,
        since they described a different render."""
        if not self.enabled or audio.size == 0:
            return

        if self._write_audio(chunk, audio, engine=engine, voice=voice, speed=speed):
            self.timings_path_for(chunk, engine=engine, voice=voice, speed=speed).unlink(
                missing_ok=True
            )

    def put_result(
        self,
        chunk: Chunk,
        result: SynthesisResult,
        *,
        engine: str,
        voice: str,
        speed: float,
    ) -> None:
        """Cache audio and its timings together.

        The sidecar is written after the audio, so an interrupted run leaves
        audio without timings - which the timed path re-synthesizes - and
        never timings without the audio they describe.
        """
        if not self.enabled or result.audio.size == 0:
            return

        if not self._write_audio(chunk, result.audio, engine=engine, voice=voice, speed=speed):
            return

        path = self.timings_path_for(chunk, engine=engine, voice=voice, speed=speed)
        words = None if result.words is None else [
            [word.text, word.start, word.end] for word in result.words
        ]
        payload = json.dumps(
            {"version": TIMINGS_VERSION, "words": words}, ensure_ascii=False, separators=(",", ":")
        )
        staging = path.with_name(path.name + ".partial")
        try:
            staging.write_text(payload, encoding="utf-8")
            staging.replace(path)
        except Exception as exc:
            log.warning("Could not cache timings %s: %s", path.name, exc)
            staging.unlink(missing_ok=True)

    def _write_audio(
        self, chunk: Chunk, audio: np.ndarray, *, engine: str, voice: str, speed: float
    ) -> bool:
        import soundfile as sf

        path = self.path_for(chunk, engine=engine, voice=voice, speed=speed)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temporary name and rename, so an interrupted run cannot
        # leave a half-written file that later reads as a valid cache hit.
        staging = path.with_suffix(".partial")
        try:
            # format is explicit: soundfile infers it from the extension, and
            # ".partial" is not a format it recognises.
            sf.write(staging, audio, self.sample_rate, format="WAV", subtype="FLOAT")
            staging.replace(path)
        except Exception as exc:
            log.warning("Could not cache %s: %s", path.name, exc)
            staging.unlink(missing_ok=True)
            return False
        return True

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def summary(self) -> str:
        return f"{self.hits} cached, {self.misses} synthesized"
