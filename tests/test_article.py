"""Short documents: one markdown file in, one audio file out.

Synthesis is faked with a stub engine, as in test_pipeline.py, so the sample
arithmetic below is exact and nothing depends on model weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from narrate.article import (
    SECTION_SILENCE_MS,
    WORDS_PER_MINUTE,
    load_article,
    plan_article,
    render_article,
    resolve_output,
)
from narrate.pipeline import NarrationConfig
from narrate.tts.base import Voice

SAMPLE_RATE = 24_000


class StubEngine:
    name = "stub"
    sample_rate = SAMPLE_RATE
    max_chars = 200

    def __init__(self):
        self.calls: list[str] = []

    def list_voices(self):
        return [Voice("test", "Test")]

    def synthesize(self, text, *, voice="test", speed=1.0):
        self.calls.append(text)
        return np.full(max(len(text) * 10, 1), 0.1, dtype=np.float32)


@pytest.fixture
def stub_engine(monkeypatch):
    engine = StubEngine()
    monkeypatch.setattr("narrate.article.get_engine", lambda *a, **k: engine)
    return engine


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


PARAGRAPH = "Sleep is when the brain files away the day and clears out the rest. " * 4


@pytest.fixture
def note(tmp_path) -> Path:
    path = tmp_path / "sleep.md"
    path.write_text(
        "---\ntitle: Why Sleep Matters\nauthor: Jane Doe\n---\n\n"
        f"## Introduction\n\n{PARAGRAPH}\n\n"
        f"## The basics\n\n{PARAGRAPH}\n\n- Light sleep\n- Deep sleep\n\n"
        f"## Habits\n\n{PARAGRAPH}\n",
        encoding="utf-8",
    )
    return path


# --- planning -------------------------------------------------------------


def test_plan_has_one_section_per_heading(stub_engine, note, config):
    plan = plan_article(load_article(note), config)
    # The front-matter title is spoken as a heading; too short to stand alone,
    # it absorbs the introduction and keeps its own name.
    assert [s.chapter.title for s in plan.sections] == [
        "Why Sleep Matters",
        "The basics",
        "Habits",
    ]


def test_planning_synthesizes_nothing(stub_engine, note, config):
    """--dry-run must work without the model, so planning cannot touch it."""
    plan_article(load_article(note), config)
    assert stub_engine.calls == []


def test_estimate_follows_word_count_and_speed(stub_engine, tmp_path, config):
    path = tmp_path / "long.md"
    path.write_text("Word " * (WORDS_PER_MINUTE * 10 - 1) + "end.\n")
    book = load_article(path)

    assert plan_article(book, config).estimated_seconds() == pytest.approx(600, rel=0.01)
    config.speed = 2.0
    assert plan_article(book, config).estimated_seconds() == pytest.approx(300, rel=0.01)


def test_script_is_exactly_what_the_engine_reads(stub_engine, note, config):
    config.use_cache = False  # the repeated paragraph would otherwise be a cache hit
    plan = plan_article(load_article(note), config)
    render_article(plan, config, output=Path(config.output_dir) / "sleep.wav")

    assert plan.script().split() == " ".join(stub_engine.calls).split()
    assert not any(char.isdigit() or char in "#-*[]" for char in plan.script())


# --- rendering ------------------------------------------------------------


def test_renders_exactly_one_file(stub_engine, note, config):
    output = Path(config.output_dir) / "sleep.wav"
    result = render_article(plan_article(load_article(note), config), config, output=output)

    assert result.path == output and output.is_file()
    assert [p.name for p in Path(config.output_dir).iterdir()] == ["sleep.wav"]
    assert result.duration_seconds > 0


def test_sections_are_joined_with_a_section_pause(stub_engine, note, config):
    import soundfile as sf

    config.use_cache = False  # every chunk's audio must pass through calls
    plan = plan_article(load_article(note), config)
    output = Path(config.output_dir) / "sleep.wav"
    render_article(plan, config, output=output)

    def samples(ms: int) -> int:
        return int(SAMPLE_RATE * ms / 1000)

    speech = sum(len(text) * 10 for text in stub_engine.calls)
    chunk_pauses = sum(samples(c.trailing_silence_ms) for s in plan.sections for c in s.chunks)
    section_pauses = samples(SECTION_SILENCE_MS) * (len(plan.sections) - 1)

    data, _ = sf.read(output)
    assert len(data) == speech + chunk_pauses + section_pauses
    # No dead air: the file ends on speech, not on a section pause.
    assert data[-1] != 0.0


def test_second_run_synthesizes_nothing(stub_engine, note, config):
    output = Path(config.output_dir) / "sleep.wav"
    render_article(plan_article(load_article(note), config), config, output=output)
    first = len(stub_engine.calls)

    result = render_article(plan_article(load_article(note), config), config, output=output)
    assert len(stub_engine.calls) == first
    assert result.cache_summary.endswith("0 synthesized")


def test_editing_one_sentence_resynthesizes_only_that_chunk(stub_engine, note, config):
    output = Path(config.output_dir) / "sleep.wav"
    render_article(plan_article(load_article(note), config), config, output=output)
    before = len(stub_engine.calls)

    note.write_text(note.read_text().replace("- Deep sleep", "- Dreaming"))
    render_article(plan_article(load_article(note), config), config, output=output)

    assert len(stub_engine.calls) - before == 1


def test_retitling_keeps_the_cache(stub_engine, note, config):
    """The cache lives under the file's name, not its title."""
    output = Path(config.output_dir) / "sleep.wav"
    render_article(plan_article(load_article(note), config), config, output=output)
    before = len(stub_engine.calls)

    note.write_text(note.read_text().replace("title: Why Sleep Matters", "title: Sleep"))
    render_article(plan_article(load_article(note), config), config, output=output)

    # Only the spoken title changed, so only the title chunk is new.
    assert len(stub_engine.calls) - before == 1


def test_mp3_is_tagged(stub_engine, note, config):
    from mutagen.easyid3 import EasyID3

    config.audio_format = "mp3"
    output = Path(config.output_dir) / "sleep.mp3"
    render_article(plan_article(load_article(note), config), config, output=output)

    tags = EasyID3(output)
    assert tags["title"] == ["Why Sleep Matters"]
    assert tags["artist"] == ["Jane Doe"]


def test_empty_document_is_refused(stub_engine, tmp_path, config):
    path = tmp_path / "empty.md"
    path.write_text("---\ntags: [x]\n---\n<!-- nothing here -->\n")
    plan = plan_article(load_article(path), config)

    assert plan.sections == []
    with pytest.raises(ValueError, match="nothing to narrate"):
        render_article(plan, config, output=tmp_path / "x.wav")


def test_books_are_pointed_elsewhere(tmp_path):
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="narrate speak"):
        load_article(path)


# --- output resolution ----------------------------------------------------


@pytest.mark.parametrize(
    ("output", "audio_format", "expected_path", "expected_format"),
    [
        (None, None, "out/sleep.mp3", "mp3"),
        (None, "opus", "out/sleep.opus", "opus"),
        ("talk.m4a", None, "talk.m4a", "m4a"),
        ("talk.M4A", "m4a", "talk.M4A", "m4a"),
        ("talk", "wav", "talk.wav", "wav"),
    ],
)
def test_output_and_format_resolve_together(output, audio_format, expected_path, expected_format):
    path, fmt = resolve_output(
        Path("notes/sleep.md"),
        output=Path(output) if output else None,
        audio_format=audio_format,
        output_dir=Path("out"),
    )
    assert (path, fmt) == (Path(expected_path), expected_format)


def test_output_directory_keeps_the_source_name(tmp_path):
    path, fmt = resolve_output(
        Path("sleep.md"), output=tmp_path, audio_format="m4a", output_dir=Path("out")
    )
    assert (path, fmt) == (tmp_path / "sleep.m4a", "m4a")


@pytest.mark.parametrize(("output", "audio_format"), [("talk.m4a", "mp3"), ("talk.m4b", None)])
def test_conflicting_or_unsupported_formats_are_refused(output, audio_format):
    with pytest.raises(ValueError):
        resolve_output(
            Path("sleep.md"), output=Path(output), audio_format=audio_format, output_dir=Path("out")
        )


# --- command line ---------------------------------------------------------


def _cli(*args):
    from narrate.cli import app

    return CliRunner().invoke(app, ["article", *map(str, args)])


def test_cli_writes_one_file(stub_engine, note, tmp_path):
    result = _cli(
        note, "--format", "wav", "--no-loudnorm",
        "--work-dir", tmp_path / "work", "--output-dir", tmp_path / "out",
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "sleep.wav").is_file()
    assert "Why Sleep Matters" in result.output


def test_cli_dry_run_writes_the_script_and_no_audio(stub_engine, note, tmp_path):
    result = _cli(note, "--dry-run", "--output-dir", tmp_path / "out")

    assert result.exit_code == 0, result.output
    assert stub_engine.calls == []
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["sleep.script.txt"]
    assert "Habits" in result.output


def test_cli_warns_when_a_document_is_book_length(stub_engine, tmp_path):
    path = tmp_path / "long.md"
    path.write_text("Word " * (WORDS_PER_MINUTE * 31) + "end.\n")
    result = _cli(path, "--dry-run", "--output-dir", tmp_path / "out")

    assert result.exit_code == 0, result.output
    assert "long for one file" in result.output
