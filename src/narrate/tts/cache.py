"""Content-addressed audio cache.

The reason the editable-markdown checkpoint is usable in practice. Chunks are
keyed by a hash of their text plus render settings, so fixing one sentence in a
300-page book re-synthesizes one chunk instead of all of them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from narrate.document import Chunk

log = logging.getLogger(__name__)


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

        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, chunk: Chunk, *, engine: str, voice: str, speed: float) -> Path:
        # Two-character prefix directory keeps any single directory from holding
        # tens of thousands of entries, which some filesystems handle poorly.
        key = chunk.cache_key(engine, voice, speed)
        return self.directory / key[:2] / f"{key}.wav"

    def get(
        self, chunk: Chunk, *, engine: str, voice: str, speed: float
    ) -> np.ndarray | None:
        if not self.enabled:
            return None

        path = self.path_for(chunk, engine=engine, voice=voice, speed=speed)
        if not path.is_file():
            self.misses += 1
            return None

        try:
            import soundfile as sf

            audio, rate = sf.read(path, dtype="float32", always_2d=False)
        except Exception as exc:
            # A truncated file from an interrupted run must not be fatal.
            log.warning("Discarding unreadable cache entry %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None

        if rate != self.sample_rate:
            log.debug("Cache entry %s has wrong sample rate; ignoring", path.name)
            self.misses += 1
            return None

        self.hits += 1
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def put(
        self,
        chunk: Chunk,
        audio: np.ndarray,
        *,
        engine: str,
        voice: str,
        speed: float,
    ) -> None:
        if not self.enabled or audio.size == 0:
            return

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

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def summary(self) -> str:
        return f"{self.hits} cached, {self.misses} synthesized"
