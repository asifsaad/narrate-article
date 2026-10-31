"""Encode read-along audio, then prove the timeline survived the encoder.

Word times are computed from the PCM that went *into* ffmpeg. They only line
up with what a browser plays if the encoded file has the same length, the same
sample rate and no unaccounted start delay. Each of those is checked by
decoding the file back, rather than trusted.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from narrate.audio import require_ffmpeg

log = logging.getLogger(__name__)

#: m4a is the default: ffmpeg writes an edit list for AAC's priming samples,
#: and browsers honour it. mp3 relies on the LAME header for the same thing,
#: which not every browser reads.
READALONG_FORMATS = ("m4a", "mp3")

#: One codec frame, the granularity at which an encoder may pad the end.
CODEC_FRAME_SAMPLES = {"m4a": 1024, "mp3": 1152}

#: Below this, a measured start offset is noise and is not applied.
OFFSET_THRESHOLD_S = 0.020

#: How much of the start is compared. Enough to contain speech; short enough
#: that the FFT stays small for an hour-long file.
CORRELATION_WINDOW_S = 5.0

#: Largest offset searched for. Encoder priming is ~20-50 ms; anything near
#: this bound means the measurement failed, not that the file is late.
MAX_LAG_S = 0.25

#: Normalized correlation below this means the two signals were not compared
#: meaningfully (silence, or a heavily processed start), so no offset is used.
MIN_CORRELATION = 0.5


@dataclass(frozen=True)
class EncodeCheck:
    """What decoding the encoded file back revealed."""

    audio_format: str
    pcm_samples: int
    encoded_samples: int
    expected_sample_rate: int
    encoded_sample_rate: int
    measured_offset_s: float | None
    """Start delay of the encoded audio relative to the PCM; ``None`` if the
    correlation was too weak to say."""
    correlation: float

    @property
    def length_difference(self) -> int:
        return self.encoded_samples - self.pcm_samples

    @property
    def length_ok(self) -> bool:
        return abs(self.length_difference) <= CODEC_FRAME_SAMPLES.get(self.audio_format, 1152)

    @property
    def rate_ok(self) -> bool:
        return self.encoded_sample_rate == self.expected_sample_rate

    @property
    def applied_offset_s(self) -> float:
        """The offset to write into ``timings.json``: zero unless it matters."""
        if self.measured_offset_s is None or abs(self.measured_offset_s) <= OFFSET_THRESHOLD_S:
            return 0.0
        return round(self.measured_offset_s, 3)

    def problems(self) -> list[str]:
        out = []
        if not self.rate_ok:
            out.append(
                f"encoded at {self.encoded_sample_rate} Hz, expected {self.expected_sample_rate} Hz"
            )
        if not self.length_ok:
            out.append(
                f"encoded length differs from the synthesized audio by "
                f"{self.length_difference} samples"
            )
        return out


def probe_sample_rate(path: Path) -> int:
    """Sample rate of the first audio stream, from ffprobe."""
    ffprobe = str(Path(require_ffmpeg()).with_name("ffprobe"))
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate", "-print_format", "json", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {result.stderr.strip()}")
    return int(json.loads(result.stdout)["streams"][0]["sample_rate"])


def decode_pcm(path: Path) -> np.ndarray:
    """Decode to mono float32 at the file's own rate, as a player would.

    ffmpeg applies the m4a edit list and the mp3 LAME delay while decoding, so
    the samples come out where a conforming player puts them.
    """
    ffmpeg = require_ffmpeg()
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "pipe:1"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not decode {path.name}: {result.stderr.decode(errors='replace').strip()}"
        )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def measure_offset(
    reference: np.ndarray, decoded: np.ndarray, *, sample_rate: int
) -> tuple[float | None, float]:
    """How late ``decoded`` starts relative to ``reference``, in seconds.

    Normalized cross-correlation over the first few seconds, searched within
    ±``MAX_LAG_S``. Positive means the encoded audio is delayed, so a word
    timed at ``t`` is heard at ``t + offset``. Loudness normalization changes
    the level, not the timing, which correlation ignores.
    """
    window = int(CORRELATION_WINDOW_S * sample_rate)
    max_lag = int(MAX_LAG_S * sample_rate)
    ref = np.asarray(reference[:window], dtype=np.float64)
    dec = np.asarray(decoded[: window + max_lag], dtype=np.float64)
    if ref.size < sample_rate // 10 or dec.size < sample_rate // 10:
        return None, 0.0

    ref = ref - ref.mean()
    dec = dec - dec.mean()
    norm = float(np.linalg.norm(ref) * np.linalg.norm(dec[: ref.size]))
    if norm == 0.0:
        return None, 0.0

    size = 1 << int(np.ceil(np.log2(ref.size + dec.size)))
    spectrum = np.fft.rfft(dec, size) * np.conj(np.fft.rfft(ref, size))
    correlation = np.fft.irfft(spectrum, size)
    lags = np.arange(-max_lag, max_lag + 1)
    values = correlation[lags % size]

    best = int(np.argmax(values))
    peak = float(values[best] / norm)
    if peak < MIN_CORRELATION:
        return None, peak
    return lags[best] / sample_rate, peak


def check_encoded(
    path: Path, pcm: np.ndarray, *, sample_rate: int, audio_format: str
) -> EncodeCheck:
    """Decode ``path`` and compare it with the PCM it was encoded from."""
    encoded_rate = probe_sample_rate(path)
    decoded = decode_pcm(path)

    if encoded_rate == sample_rate:
        offset, peak = measure_offset(pcm, decoded, sample_rate=sample_rate)
        encoded_samples = decoded.size
    else:
        # Wrong rate is reported as its own problem; compare lengths in the
        # PCM's terms so the two numbers mean the same thing.
        offset, peak = None, 0.0
        encoded_samples = round(decoded.size * sample_rate / encoded_rate)

    check = EncodeCheck(
        audio_format=audio_format,
        pcm_samples=int(pcm.size),
        encoded_samples=int(encoded_samples),
        expected_sample_rate=sample_rate,
        encoded_sample_rate=encoded_rate,
        measured_offset_s=offset,
        correlation=peak,
    )
    log.debug("Encode check for %s: %s", path.name, check)
    return check


__all__ = [
    "CODEC_FRAME_SAMPLES",
    "EncodeCheck",
    "READALONG_FORMATS",
    "check_encoded",
    "decode_pcm",
    "measure_offset",
    "probe_sample_rate",
]
