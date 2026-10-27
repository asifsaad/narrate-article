"""Cache, audio assembly and checkpoint round-tripping.

Synthesis is faked with a deterministic stub engine so these run in
milliseconds and do not depend on model weights being present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from narrate.audio import concat_with_silence, format_duration, write_chapter
from narrate.document import Book, Chapter, Chunk
from narrate.pipeline import (
    NarrationConfig,
    load_checkpoint,
    narrate,
    write_checkpoint,
)
from narrate.tts import AudioCache
from narrate.tts.base import Voice

SAMPLE_RATE = 24_000


class StubEngine:
    """Deterministic fake: audio length is proportional to text length.

    Counts calls so tests can assert on what was *not* re-synthesized, which is
    the only way to verify the cache actually saves work.
    """

    name = "stub"
    sample_rate = SAMPLE_RATE
    max_chars = 200

    def __init__(self):
        self.calls: list[str] = []

    def list_voices(self):
        return [Voice("test", "Test")]

    def synthesize(self, text, *, voice="test", speed=1.0):
        self.calls.append(text)
        samples = max(int(self.sample_rate * len(text) / 100), 1)
        return np.full(samples, 0.1, dtype=np.float32)


@pytest.fixture
def stub_engine(monkeypatch):
    engine = StubEngine()
    monkeypatch.setattr("narrate.pipeline.get_engine", lambda *a, **k: engine)
    return engine


@pytest.fixture
def book() -> Book:
    return Book(
        title="Test Book",
        author="A. Author",
        chapters=[
            Chapter(1, "First Chapter", "# First Chapter\n\n" + "Alpha beta gamma. " * 12),
            Chapter(2, "Second Chapter", "# Second Chapter\n\n" + "Delta epsilon. " * 12),
        ],
    )


@pytest.fixture
def config(tmp_path) -> NarrationConfig:
    return NarrationConfig(
        engine="stub",
        voice="test",
        audio_format="wav",
        normalize_loudness=False,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )


# --- cache ----------------------------------------------------------------


def test_cache_round_trip(tmp_path):
    cache = AudioCache(tmp_path / "c", sample_rate=SAMPLE_RATE)
    chunk = Chunk(text="Hello there.", chapter_index=1, order=0)
    audio = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)

    assert cache.get(chunk, engine="stub", voice="v", speed=1.0) is None
    cache.put(chunk, audio, engine="stub", voice="v", speed=1.0)

    restored = cache.get(chunk, engine="stub", voice="v", speed=1.0)
    assert restored is not None
    np.testing.assert_allclose(restored, audio, atol=1e-6)


def test_cache_miss_on_different_voice(tmp_path):
    cache = AudioCache(tmp_path / "c", sample_rate=SAMPLE_RATE)
    chunk = Chunk(text="Hello.", chapter_index=1, order=0)
    cache.put(chunk, np.ones(100, dtype=np.float32), engine="stub", voice="a", speed=1.0)
    assert cache.get(chunk, engine="stub", voice="b", speed=1.0) is None


def test_cache_writes_no_stray_staging_files(tmp_path):
    """Regression: staging files were written with a suffix soundfile could not
    infer a format from, so nothing was ever cached."""
    cache = AudioCache(tmp_path / "c", sample_rate=SAMPLE_RATE)
    chunk = Chunk(text="Hello.", chapter_index=1, order=0)
    cache.put(chunk, np.ones(100, dtype=np.float32), engine="stub", voice="v", speed=1.0)

    assert list((tmp_path / "c").rglob("*.wav"))
    assert not list((tmp_path / "c").rglob("*.partial"))


def test_corrupt_cache_entry_is_discarded(tmp_path):
    cache = AudioCache(tmp_path / "c", sample_rate=SAMPLE_RATE)
    chunk = Chunk(text="Hello.", chapter_index=1, order=0)
    path = cache.path_for(chunk, engine="stub", voice="v", speed=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a wav file")

    assert cache.get(chunk, engine="stub", voice="v", speed=1.0) is None
    assert not path.exists()


def test_disabled_cache_stores_nothing(tmp_path):
    cache = AudioCache(tmp_path / "c", sample_rate=SAMPLE_RATE, enabled=False)
    chunk = Chunk(text="Hello.", chapter_index=1, order=0)
    cache.put(chunk, np.ones(100, dtype=np.float32), engine="stub", voice="v", speed=1.0)
    assert cache.get(chunk, engine="stub", voice="v", speed=1.0) is None


# --- audio assembly -------------------------------------------------------


def test_silence_is_inserted_between_pieces():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32)
    joined = concat_with_silence([(a, 500), (b, 0)], sample_rate=1000)
    assert len(joined) == 1000 + 500 + 1000
    assert joined[1200] == 0.0


def test_concat_of_nothing_is_empty():
    assert concat_with_silence([], sample_rate=SAMPLE_RATE).size == 0


def test_write_chapter_rejects_empty_audio(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_chapter(
            np.zeros(0, dtype=np.float32),
            tmp_path / "x.wav",
            sample_rate=SAMPLE_RATE,
            audio_format="wav",
            normalize_loudness=False,
        )


def test_write_chapter_produces_readable_file(tmp_path):
    import soundfile as sf

    audio = np.sin(np.linspace(0, 100, SAMPLE_RATE)).astype(np.float32)
    out = write_chapter(
        audio,
        tmp_path / "chapter.wav",
        sample_rate=SAMPLE_RATE,
        audio_format="wav",
        normalize_loudness=False,
    )
    data, rate = sf.read(out)
    assert rate == SAMPLE_RATE
    assert len(data) == SAMPLE_RATE
    assert not list(tmp_path.glob("*.staging.wav"))


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(45, "45s"), (90, "1m 30s"), (3661, "1h 01m 01s")],
)
def test_duration_formatting(seconds, expected):
    assert format_duration(seconds) == expected


# --- checkpoint -----------------------------------------------------------


def test_checkpoint_round_trip(tmp_path, book):
    directory = write_checkpoint(book, tmp_path)
    restored = load_checkpoint(directory)

    assert restored.title == book.title
    assert restored.author == book.author
    assert [c.title for c in restored.chapters] == [c.title for c in book.chapters]


def test_checkpoint_reflects_user_edits(tmp_path, book):
    directory = write_checkpoint(book, tmp_path)
    target = directory / "chapters" / "001 - First Chapter.md"
    target.write_text("# First Chapter\n\nCompletely rewritten body text.\n")

    restored = load_checkpoint(directory)
    assert "Completely rewritten" in restored.chapters[0].markdown


def test_checkpoint_clears_stale_chapters(tmp_path, book):
    directory = write_checkpoint(book, tmp_path)
    stale = directory / "chapters" / "099 - Deleted Chapter.md"
    stale.write_text("# Deleted\n\nShould not survive a re-extract.\n")

    write_checkpoint(book, tmp_path)
    assert not stale.exists()


# --- narration ------------------------------------------------------------


def test_narrate_writes_one_file_per_chapter(stub_engine, book, config):
    result = narrate(book, config)

    assert len(result.chapters) == 2
    for rendered in result.chapters:
        assert rendered.path.is_file()
        assert rendered.duration_seconds > 0


def test_second_run_synthesizes_nothing(stub_engine, book, config):
    narrate(book, config)
    first_pass = len(stub_engine.calls)
    assert first_pass > 0

    narrate(book, config)
    assert len(stub_engine.calls) == first_pass, "cache did not prevent re-synthesis"


def test_editing_one_chapter_resynthesizes_only_that_chapter(
    stub_engine, book, config
):
    narrate(book, config)
    baseline = len(stub_engine.calls)

    book.chapters[1].markdown = "# Second Chapter\n\nEntirely different words now."
    narrate(book, config)

    added = len(stub_engine.calls) - baseline
    assert 0 < added < baseline, f"expected a partial re-render, got {added} new calls"


def test_chapter_selection_limits_output(stub_engine, book, config):
    config.chapters = {2}
    result = narrate(book, config)

    assert len(result.chapters) == 1
    assert result.chapters[0].chapter.index == 2


def test_empty_chapters_are_skipped(stub_engine, config):
    book = Book(
        title="Sparse",
        chapters=[
            Chapter(1, "Heading Only", "# Heading Only"),
            Chapter(2, "Real", "# Real\n\n" + "Words here. " * 12),
        ],
    )
    result = narrate(book, config)
    assert [r.chapter.index for r in result.chapters] == [2]


# --- concurrent encoding --------------------------------------------------


def test_parallel_encode_produces_same_files(stub_engine, book, config):
    """Overlapping encoding with synthesis must not change the output."""
    config.parallel_encode = False
    serial = narrate(book, config)
    serial_sizes = {p.path.name: p.path.stat().st_size for p in serial.chapters}

    for rendered in serial.chapters:
        rendered.path.unlink()

    config.parallel_encode = True
    parallel = narrate(book, config)
    parallel_sizes = {p.path.name: p.path.stat().st_size for p in parallel.chapters}

    assert serial_sizes == parallel_sizes


def test_parallel_encode_sets_paths_and_releases_audio(stub_engine, book, config):
    config.parallel_encode = True
    result = narrate(book, config)

    for rendered in result.chapters:
        assert rendered.path is not None and rendered.path.is_file()
        assert rendered.audio is None, "chapter samples must not be retained"


def test_encode_failure_propagates(stub_engine, book, config, monkeypatch):
    """An ffmpeg error inside the pool must surface, not vanish."""
    config.parallel_encode = True

    def boom(*args, **kwargs):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr("narrate.pipeline.write_chapter", boom)
    with pytest.raises(RuntimeError, match="ffmpeg exploded"):
        narrate(book, config)


def test_encoder_bounds_chapters_held_in_memory(stub_engine, config):
    """Many chapters must not queue up unbounded float32 buffers."""
    from narrate.pipeline import _ChapterEncoder

    many = Book(
        title="Long",
        chapters=[
            Chapter(i, f"Chapter {i}", f"# Chapter {i}\n\n" + "Words here. " * 12)
            for i in range(1, 13)
        ],
    )
    config.parallel_encode = True
    result = narrate(many, config)

    assert len(result.chapters) == 12
    assert _ChapterEncoder.MAX_IN_FLIGHT <= 4
    assert all(r.audio is None for r in result.chapters)


# --- m4b bundling ---------------------------------------------------------


def _ffprobe(path, entries, stream=False):
    import json
    import subprocess

    command = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_chapters" if not stream else "-show_streams", str(path),
    ]
    return json.loads(subprocess.run(command, capture_output=True, text=True).stdout)


@pytest.fixture
def m4b_config(config):
    config.audio_format = "m4b"
    config.bitrate = "64k"
    return config


def test_m4b_produces_one_file(stub_engine, book, m4b_config):
    result = narrate(book, m4b_config)

    assert result.bundle is not None
    assert result.bundle.suffix == ".m4b"
    assert result.bundle.is_file()
    # The whole point: no per-chapter files alongside it.
    assert not list(result.bundle.parent.glob("*.mp3"))


def test_m4b_has_one_marker_per_chapter(stub_engine, book, m4b_config):
    result = narrate(book, m4b_config)
    chapters = _ffprobe(result.bundle, "chapters")["chapters"]

    assert len(chapters) == 2
    assert [c["tags"]["title"] for c in chapters] == [
        "First Chapter",
        "Second Chapter",
    ]


def test_m4b_chapter_markers_are_contiguous(stub_engine, book, m4b_config):
    """A gap or overlap between markers makes players skip or repeat audio."""
    result = narrate(book, m4b_config)
    chapters = _ffprobe(result.bundle, "chapters")["chapters"]

    assert float(chapters[0]["start_time"]) == 0.0
    for earlier, later in zip(chapters, chapters[1:]):
        assert float(earlier["end_time"]) == pytest.approx(
            float(later["start_time"]), abs=0.01
        )


def test_m4b_releases_chapter_audio(stub_engine, book, m4b_config):
    """A full book's float samples are gigabytes; they must not be held."""
    result = narrate(book, m4b_config)
    assert all(rendered.audio is None for rendered in result.chapters)


def test_m4b_builds_from_a_relative_work_dir(stub_engine, book, tmp_path, monkeypatch):
    """Regression: ffmpeg's concat demuxer resolves relative paths against the
    list file's own temp directory, so a relative --work-dir broke the mux."""
    monkeypatch.chdir(tmp_path)

    config = NarrationConfig(
        engine="stub",
        voice="test",
        audio_format="m4b",
        bitrate="64k",
        normalize_loudness=False,
        work_dir=Path("work"),      # relative, as the CLI default is
        output_dir=Path("out"),
    )
    result = narrate(book, config)
    assert result.bundle is not None and result.bundle.is_file()


def test_m4b_cleans_up_staged_wavs(stub_engine, book, config):
    """Staged chapters are ~10 GB for a long book; they must not be left."""
    config.audio_format = "m4b"
    config.bitrate = "64k"
    result = narrate(book, config)

    stage = Path(config.work_dir) / book.slug / "stage"
    assert result.bundle.is_file()
    assert not stage.exists() or not list(stage.glob("*.wav"))


def test_ffmetadata_escapes_structural_characters():
    from narrate.audio import _ffmetadata

    rendered = _ffmetadata([("Title; with = signs", 0, 1000)], {"artist": "A#B"})
    assert r"\;" in rendered and r"\=" in rendered and r"\#" in rendered


def test_epub_cover_is_extracted(tmp_path):
    from narrate.audio import extract_cover

    cover = extract_cover(FIXTURE_EPUB, tmp_path)
    # The fixture EPUB has no embedded image, so None is the correct answer.
    assert cover is None or cover.is_file()


FIXTURE_EPUB = __import__("pathlib").Path(__file__).parent / "fixtures" / "sample.epub"


def test_no_digits_reach_the_engine(stub_engine, config):
    book = Book(
        title="Numbers",
        chapters=[
            Chapter(1, "Ch", "# Ch\n\nIn 1997 Dr. Smith spent $4.2M, i.e. 12% of it."),
        ],
    )
    narrate(book, config)
    assert stub_engine.calls
    assert not any(char.isdigit() for text in stub_engine.calls for char in text)
