"""Assemble rendered chunks into finished per-chapter audio files."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: EBU R128 targets. -19 LUFS mono is the audiobook convention (ACX asks for
#: -23 to -18 with -3 dBTP ceiling); music mastering targets are far louder and
#: sound fatiguing over hours of speech.
TARGET_LUFS = -19.0
TARGET_TRUE_PEAK = -3.0
TARGET_LRA = 11.0


class FFmpegNotFoundError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError(
            "ffmpeg is required for encoding. Install it with: brew install ffmpeg"
        )
    return path


def silence_samples(milliseconds: int, *, sample_rate: int) -> int:
    """Length of a generated pause, in samples.

    The single definition of pause arithmetic: assembly uses it to insert
    silence and read-along uses it to place words, so the two cannot drift.
    """
    return int(sample_rate * milliseconds / 1000) if milliseconds > 0 else 0


def concat_with_silence(
    pieces: list[tuple[np.ndarray, int]], *, sample_rate: int
) -> np.ndarray:
    """Join ``(audio, trailing_silence_ms)`` pairs into one track.

    Silence is generated rather than taken from the model, which would otherwise
    have to render pauses as audio and could hallucinate breath or room tone
    into them.
    """
    if not pieces:
        return np.zeros(0, dtype=np.float32)

    out: list[np.ndarray] = []
    for audio, silence_ms in pieces:
        if audio.size:
            out.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if silence_ms > 0:
            samples = silence_samples(silence_ms, sample_rate=sample_rate)
            out.append(np.zeros(samples, dtype=np.float32))

    if not out:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(out)


def write_chapter(
    audio: np.ndarray,
    destination: Path,
    *,
    sample_rate: int,
    audio_format: str = "mp3",
    bitrate: str = "128k",
    normalize_loudness: bool = True,
    tags: dict[str, str] | None = None,
    output_sample_rate: int | None = None,
) -> Path:
    """Encode one chapter and tag it.

    The WAV intermediate is written first because piping raw samples through
    ffmpeg's stdin makes failures much harder to diagnose, and the file is
    removed immediately afterwards.

    ``output_sample_rate`` pins the encoded rate. Without it, ``loudnorm``'s
    internal 192 kHz resampling leaks into the output (96 kHz AAC, 48 kHz
    mp3). Read-along passes the engine's rate so the encoded timeline is the
    synthesized one; existing commands pass nothing and are unchanged.
    """
    import soundfile as sf

    destination.parent.mkdir(parents=True, exist_ok=True)

    if audio.size == 0:
        raise ValueError(f"Refusing to write empty audio to {destination.name}")

    audio = _guard_clipping(audio)

    staging = destination.with_suffix(".staging.wav")
    sf.write(staging, audio, sample_rate)

    try:
        if audio_format == "wav" and not normalize_loudness:
            staging.replace(destination)
        else:
            _encode(
                staging,
                destination,
                audio_format=audio_format,
                bitrate=bitrate,
                normalize_loudness=normalize_loudness,
                output_sample_rate=output_sample_rate,
            )
    finally:
        staging.unlink(missing_ok=True)

    if tags and audio_format == "mp3":
        _write_tags(destination, tags)

    return destination


def _guard_clipping(audio: np.ndarray) -> np.ndarray:
    """Scale down if the concatenation exceeds full scale.

    Individual chunks are in range, but loudnorm runs after encoding starts, so
    an out-of-range buffer would clip in the WAV intermediate first.
    """
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1.0:
        log.debug("Scaling audio down from peak %.3f", peak)
        return (audio / peak * 0.99).astype(np.float32)
    return audio


def _encode(
    source: Path,
    destination: Path,
    *,
    audio_format: str,
    bitrate: str,
    normalize_loudness: bool,
    output_sample_rate: int | None = None,
) -> None:
    ffmpeg = require_ffmpeg()

    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(source)]

    if normalize_loudness:
        # Single-pass loudnorm. Two-pass measures first for a more exact result,
        # but doubles the ffmpeg work for a difference inaudible on speech.
        command += [
            "-af",
            f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA={TARGET_LRA}",
        ]

    if output_sample_rate is not None:
        command += ["-ar", str(output_sample_rate)]

    if audio_format == "mp3":
        command += ["-codec:a", "libmp3lame", "-b:a", bitrate]
    elif audio_format == "wav":
        command += ["-codec:a", "pcm_s16le"]
    elif audio_format in {"m4a", "aac"}:
        command += ["-codec:a", "aac", "-b:a", bitrate]
    elif audio_format == "opus":
        command += ["-codec:a", "libopus", "-b:a", bitrate]
    else:
        raise ValueError(f"Unsupported audio format: {audio_format!r}")

    command.append(str(destination))

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed encoding {destination.name}:\n{result.stderr.strip()}"
        )


def _write_tags(path: Path, tags: dict[str, str]) -> None:
    """Write ID3 tags so players show a real book, not a list of files."""
    try:
        from mutagen.easyid3 import EasyID3
        from mutagen.mp3 import MP3
    except ImportError:
        log.debug("mutagen not available; skipping tags")
        return

    try:
        audio = MP3(path, ID3=EasyID3)
        if audio.tags is None:
            audio.add_tags()
        for key, value in tags.items():
            if value and key in EasyID3.valid_keys:
                audio.tags[key] = str(value)
        audio.save()
    except Exception as exc:
        # Tagging is cosmetic; a failure must not discard rendered audio.
        log.warning("Could not tag %s: %s", path.name, exc)


def build_m4b(
    tracks: list[tuple[str, Path, int]],
    destination: Path,
    *,
    sample_rate: int,
    bitrate: str = "64k",
    normalize_loudness: bool = True,
    tags: dict[str, str] | None = None,
    cover: Path | None = None,
) -> Path:
    """Mux staged chapter WAVs into one .m4b with chapter markers.

    ``tracks`` is ``(chapter_title, wav_path, sample_count)`` in playback order.

    Chapters arrive as files rather than arrays because a long audiobook does
    not fit in memory: 28 hours at 24 kHz float32 is ~10 GB, and concatenating
    in memory doubles that. ffmpeg's concat demuxer streams them instead, so
    peak memory is independent of book length.

    The audio is still encoded exactly once - the staged files are the raw
    samples, not an intermediate lossy format. Loudness normalization is applied
    across the whole book rather than per chapter, so the level does not shift
    at a chapter boundary.
    """
    import tempfile

    ffmpeg = require_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not tracks:
        raise ValueError("Cannot build an m4b with no chapters")

    with tempfile.TemporaryDirectory(prefix="narrate-m4b-") as scratch:
        scratch_dir = Path(scratch)

        # Chapter boundaries come from sample counts, not from the encoded file:
        # reading them back would inherit AAC's frame padding and drift a few
        # milliseconds per chapter, accumulating across a book.
        marks: list[tuple[str, int, int]] = []
        cursor_ms = 0
        listing: list[str] = []

        for title, wav_path, samples in tracks:
            if samples <= 0:
                continue
            duration_ms = int(round(samples / sample_rate * 1000))
            marks.append((title, cursor_ms, cursor_ms + duration_ms))
            cursor_ms += duration_ms
            # Paths must be absolute: the concat demuxer resolves relative
            # entries against the *list file's* directory, which is a temp dir,
            # not the working directory. Single quotes also need escaping.
            escaped = str(Path(wav_path).resolve()).replace("'", r"'\''")
            listing.append(f"file '{escaped}'")

        if not marks:
            raise ValueError("Cannot build an m4b with no audio")

        concat_list = scratch_dir / "inputs.txt"
        concat_list.write_text("\n".join(listing) + "\n", encoding="utf-8")

        metadata = scratch_dir / "chapters.txt"
        metadata.write_text(_ffmetadata(marks, tags or {}), encoding="utf-8")

        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-i", str(metadata),
        ]
        if cover and cover.is_file():
            command += ["-i", str(cover)]

        command += ["-map_metadata", "1", "-map", "0:a"]
        if cover and cover.is_file():
            # attached_pic marks the stream as cover art rather than video, so
            # players show it instead of trying to play it.
            command += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v", "attached_pic"]

        if normalize_loudness:
            command += [
                "-af",
                f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA={TARGET_LRA}",
            ]

        command += ["-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart"]
        command.append(str(destination))

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed building {destination.name}:\n{result.stderr.strip()}"
            )

    return destination


def _ffmetadata(marks: list[tuple[str, int, int]], tags: dict[str, str]) -> str:
    """Render ffmpeg's metadata format, which is how m4b chapters are set."""
    lines = [";FFMETADATA1"]

    for key, value in tags.items():
        if value:
            lines.append(f"{key}={_escape_meta(str(value))}")

    for title, start_ms, end_ms in marks:
        lines += [
            "",
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={_escape_meta(title)}",
        ]

    return "\n".join(lines) + "\n"


def _escape_meta(value: str) -> str:
    """``=``, ``;``, ``#``, ``\\`` and newlines are structural in ffmetadata."""
    for char in ("\\", "=", ";", "#"):
        value = value.replace(char, "\\" + char)
    return value.replace("\n", " ")


def extract_cover(source: Path, destination_dir: Path) -> Path | None:
    """Pull cover art out of an EPUB, if it has any.

    Best-effort: a missing cover is normal and must never fail a render.
    """
    if source is None or source.suffix.lower() != ".epub" or not source.is_file():
        return None

    import zipfile

    try:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            candidates = [
                n
                for n in names
                if "cover" in n.lower() and n.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if not candidates:
                candidates = [n for n in names if n.lower().endswith((".jpg", ".jpeg"))]
            if not candidates:
                return None

            # Largest file is the full-size cover rather than a thumbnail.
            best = max(candidates, key=lambda n: archive.getinfo(n).file_size)
            destination_dir.mkdir(parents=True, exist_ok=True)
            out = destination_dir / f"cover{Path(best).suffix.lower()}"
            out.write_bytes(archive.read(best))
            return out
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        log.debug("No cover extracted from %s: %s", source.name, exc)
        return None


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
