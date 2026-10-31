"""Read-along: timeline, alignment, timings.json, the page and `narrate read`.

Synthesis is the deterministic ``TimedStubEngine``: every token lasts
``100 ms x len / 5`` and tokens are laid end to end, so every expected time is
a cumulative sum of token lengths and the generated pauses. Encoding tests need
ffmpeg, like the rest of the suite; the browser test needs Node + Playwright
and skips without them.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from stub_engine import SAMPLE_RATE, TimedStubEngine, UntimedStubEngine, token_samples
from typer.testing import CliRunner

from narrate.article import (
    SECTION_SILENCE_MS,
    join_sections,
    load_article,
    plan_article,
    render_article,
)
from narrate.audio import concat_with_silence, silence_samples, write_chapter
from narrate.document import Chapter, Chunk
from narrate.normalize import Lexicon
from narrate.pipeline import NarrationConfig
from narrate.readalong.align import locate_tokens, naive_display_spans, sentence_ids
from narrate.readalong.build import SectionText, build_timings
from narrate.readalong.display import markup_spans, parse_blocks, visible_mask
from narrate.readalong.encode import (
    CODEC_FRAME_SAMPLES,
    check_encoded,
    measure_offset,
    probe_sample_rate,
)
from narrate.readalong.page import inline_json
from narrate.readalong.render import dump_timings, readalong_paths, render_readalong
from narrate.readalong.schema import validate_timings
from narrate.readalong.timeline import RenderedChunk, build_timeline
from narrate.segment import (
    HEADING_SILENCE_MS,
    PARAGRAPH_SILENCE_MS,
    SENTENCE_SILENCE_MS,
)
from narrate.tts.base import SynthesisResult, WordTiming

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is needed to encode read-along audio",
)


@pytest.fixture
def engine(monkeypatch) -> TimedStubEngine:
    stub = TimedStubEngine()
    monkeypatch.setattr("narrate.article.get_engine", lambda *a, **k: stub)
    return stub


@pytest.fixture
def config(tmp_path) -> NarrationConfig:
    return NarrationConfig(
        engine="stub",
        voice="test",
        audio_format="m4a",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
    )


NOTE = """\
---
title: Why Sleep Matters
author: Jane Doe
---

# Why Sleep Matters

The deal was $4.2M, per Dr. Smith [12]. The **FBI** knew it in 1939-1945, on the 21st of May.

Sleep is when the brain files away the day, see https://example.com/sleep for more.

## Habits

- Keep a regular bedtime.
- Avoid screens late at night; read [the guide](https://example.com/guide) instead.

> A quote, with a wrapped
> second line.
"""


@pytest.fixture
def note(tmp_path) -> Path:
    path = tmp_path / "sleep.md"
    path.write_text(NOTE, encoding="utf-8")
    return path


def _render(note: Path, config: NarrationConfig, **kwargs):
    book = load_article(note)
    plan = plan_article(book, config)
    paths = readalong_paths(note, output_dir=config.output_dir, audio_format=config.audio_format)
    return plan, render_readalong(plan, config, paths=paths, **kwargs)


def _timings(result) -> dict:
    return json.loads(result.paths.timings.read_text(encoding="utf-8"))


def _shown(data: dict, word: dict) -> str:
    a, b = word["src"]
    return data["display_text"][a:b]


# --- timeline -------------------------------------------------------------


def _rendered(text: str, silence_ms: int, section: int) -> RenderedChunk:
    result = TimedStubEngine().synthesize_with_timings(text)
    chunk = Chunk(text=text, chapter_index=section + 1, order=0, trailing_silence_ms=silence_ms)
    return RenderedChunk(chunk=chunk, result=result, section=section)


def test_word_times_are_cumulative_sums_of_audio_and_every_pause():
    """250, 600 and 800 ms chunk pauses and the 1200 ms section pause, by hand."""
    sections = [
        [
            _rendered("Title.", HEADING_SILENCE_MS, 0),
            _rendered("One two.", SENTENCE_SILENCE_MS, 0),
            _rendered("Three.", PARAGRAPH_SILENCE_MS, 0),
            _rendered("Four.", 0, 0),
        ],
        [],  # renders to nothing: no audio and no section pause
        [_rendered("Five six.", 0, 2)],
    ]
    timeline = build_timeline(sections, sample_rate=SAMPLE_RATE)

    def ms(value: int) -> int:
        return silence_samples(value, sample_rate=SAMPLE_RATE)

    expected: list[tuple[str, float]] = []
    cursor = 0
    for text, pause in [
        ("Title.", ms(HEADING_SILENCE_MS)),
        ("One two.", ms(SENTENCE_SILENCE_MS)),
        ("Three.", ms(PARAGRAPH_SILENCE_MS)),
        ("Four.", ms(SECTION_SILENCE_MS)),
        ("Five six.", 0),
    ]:
        for token in re.findall(r"\w+|[^\w\s]", text):
            expected.append((token, cursor / SAMPLE_RATE))
            cursor += token_samples(token)
        cursor += pause

    placed = [(token.text, token.start) for token in timeline.tokens]
    assert [text for text, _ in placed] == [text for text, _ in expected]
    for (text, got), (_, want) in zip(placed, expected):
        assert got == pytest.approx(want, abs=1e-9), text

    tracks = [
        concat_with_silence(
            [(r.result.audio, r.chunk.trailing_silence_ms) for r in chunks], sample_rate=SAMPLE_RATE
        )
        for chunks in sections
    ]
    assert timeline.total_samples == join_sections(tracks, sample_rate=SAMPLE_RATE).size
    assert timeline.total_samples == cursor


@requires_ffmpeg
def test_words_are_strictly_ordered_and_inside_the_audio(engine, note, config):
    _, result = _render(note, config)
    data = _timings(result)
    starts = [w["start"] for w in data["words"]]

    assert starts == sorted(starts)
    assert all(a < b for a, b in itertools.pairwise(starts)), "two words start together"
    assert all(0 <= w["start"] <= w["end"] <= data["duration_s"] for w in data["words"])


@requires_ffmpeg
def test_timings_follow_speed(engine, note, config):
    """Timings come from the rendered audio, so --speed shows up in them."""
    _, normal = _render(note, config)
    a = _timings(normal)["words"]  # read now: the next render reuses the folder
    config.speed = 0.9
    _, slow = _render(note, config)
    b = _timings(slow)["words"]

    assert len(a) == len(b)
    assert slow.duration_seconds > normal.duration_seconds
    for x, y in zip(a, b):
        # Speech stretches by 1/0.9; the generated pauses do not.
        assert y["end"] - y["start"] == pytest.approx((x["end"] - x["start"]) / 0.9, abs=0.003)


# --- spoken alignment -----------------------------------------------------


def test_tokens_are_located_in_order_with_punctuation():
    text = "The F. B. I. knew, “roughly.”"
    spans, misses = locate_tokens(text, ["The", "F.", "B.", "I.", "knew", ",", '"', "roughly", ".", "”"])
    assert misses == 0
    assert [text[a:b] for a, b in spans] == ["The", "F.", "B.", "I.", "knew", ",", "“", "roughly", ".", "”"]


def test_a_token_that_does_not_match_takes_the_next_word():
    spans, misses = locate_tokens("alpha beta gamma", ["alpha", "BEETA", "gamma"])
    assert misses == 1
    assert spans == [(0, 5), (6, 10), (11, 16)]


def test_missing_punctuation_is_not_a_miss():
    spans, misses = locate_tokens("no stop here", ["no", ";", "stop", "here"])
    assert misses == 0 and spans[1] is None


def test_clause_split_sentence_keeps_one_sentence_id():
    """A long sentence split across chunks is still one sentence to underline."""
    sentences = ["Short one.", "A very long sentence, split at a clause, keeps going."]
    chunks = ["Short one. A very long sentence,", "split at a clause, keeps going."]
    ids = sentence_ids(chunks, sentences)
    assert set(ids[0][:10]) == {0}
    assert set(ids[0][11:]) == {1}
    assert set(ids[1]) == {1}


# --- display alignment ----------------------------------------------------


def _naive(display: str, words: list[str]) -> list[str]:
    blocks = parse_blocks(display)
    visible = visible_mask(len(display), markup_spans(display, blocks))
    spans, _ = naive_display_spans(display, visible, words, start=0, end=len(display))
    return [display[a:b] for a, b in spans]


def test_rewritten_words_share_the_display_text_they_came_from():
    display = "The deal was $4.2M, per Dr. Smith. The FBI knew."
    spoken = "The deal was four point two million dollars per Doctor Smith The F. B. I. knew".split()
    shown = _naive(display, spoken)
    assert shown[3:8] == ["$4.2M"] * 5
    assert shown[9] == "Dr."
    assert shown[12:15] == ["FBI"] * 3


def test_a_common_word_later_on_does_not_derail_alignment():
    """Regression: a greedy matcher sent "for" (from "e.g.") to a later "for"."""
    display = "It ran in 1939-1945, e.g. on May.\n\n- Read the guide for more."
    spoken = "It ran in nineteen thirty-nine to nineteen forty-five for example on May Read the guide for more".split()
    shown = _naive(display, spoken)
    assert shown[spoken.index("on")] == "on"
    assert shown[-2:] == ["for", "more"]


def test_link_urls_and_markup_never_receive_words():
    display = "See [the guide](https://example.com/guide) or **bold** text."
    shown = _naive(display, ["See", "the", "guide", "or", "bold", "text"])
    assert shown == ["See", "the", "guide", "or", "bold", "text"]


def test_markup_is_hidden_in_every_block():
    """Regression: markers were only found in the first block, because ``^``
    does not anchor at a match position in Python."""
    display = "# One\n\nPara **bold** [link](http://x.y).\n\n## Two ##\n\n- item\n\n1. first\n\n> quote\n> more"
    hidden = {i for a, b in markup_spans(display, parse_blocks(display)) for i in range(a, b)}
    visible = "".join(c for i, c in enumerate(display) if i not in hidden)
    assert visible == "One\n\nPara bold link.\n\nTwo\n\nitem\n\nfirst\n\nquote\nmore"


def test_blocks_are_parsed_by_kind():
    display = "# One\n\nA paragraph\nwrapped.\n\n- item one\n  continued\n- item two\n\n3. third\n\n> quoted"
    kinds = [(b.kind, b.level, b.number) for b in parse_blocks(display)]
    assert kinds == [
        ("heading", 1, None),
        ("paragraph", None, None),
        ("list_item", None, None),
        ("list_item", None, None),
        ("list_item", None, 3),
        ("quote", None, None),
    ]


# --- timings.json ---------------------------------------------------------


@requires_ffmpeg
def test_timings_validate_and_point_inside_the_display_text(engine, note, config):
    _, result = _render(note, config)
    data = _timings(result)

    assert validate_timings(data) == []
    length = len(data["display_text"])
    for item in data["words"] + data["sentences"] + data["blocks"]:
        a, b = item["src"]
        assert 0 <= a < b <= length
    used = {w["sentence"] for w in data["words"]}
    assert used == set(range(len(data["sentences"])))


@requires_ffmpeg
def test_display_text_is_the_speakable_markdown(engine, note, config):
    """What `extract` would show: furniture gone, citations and URLs kept."""
    plan, result = _render(note, config)
    data = _timings(result)
    assert data["display_text"] == "\n\n".join(s.chapter.markdown for s in plan.sections)
    assert "title: Why Sleep Matters" not in data["display_text"]
    assert "[12]" in data["display_text"]


@requires_ffmpeg
def test_normalized_items_highlight_their_source(engine, note, config):
    _, result = _render(note, config)
    data = _timings(result)
    shown = {w["spoken"]: _shown(data, w) for w in data["words"]}

    assert shown["dollars"] == "$4.2M"
    assert shown["Doctor"] == "Dr."
    assert shown["twenty-first"] == "21st"
    assert [_shown(data, w) for w in data["words"] if w["spoken"] in ("F", "B", "I")] == ["FBI"] * 3


@requires_ffmpeg
def test_citations_and_urls_produce_no_words(engine, note, config):
    _, result = _render(note, config)
    data = _timings(result)
    text = data["display_text"]
    skipped = [text.index("[12]"), text.index("https://example.com/sleep"), text.index("https://example.com/guide")]
    for word in data["words"]:
        a, b = word["src"]
        assert not any(a <= position < b for position in skipped), word


@requires_ffmpeg
def test_lexicon_drops_are_never_highlighted(engine, tmp_path, config):
    path = tmp_path / "drop.md"
    path.write_text("# Notes\n\nThe end is near. All rights reserved. Truly the end.\n")
    config.lexicon_path = tmp_path / "lexicon.yaml"
    config.lexicon_path.write_text('drop:\n  - "All rights reserved."\n')

    _, result = _render(path, config)
    data = _timings(result)
    start = data["display_text"].index("All rights")
    end = start + len("All rights reserved.")
    assert all(w["src"][1] <= start or w["src"][0] >= end for w in data["words"])
    assert Lexicon.load(config.lexicon_path).drop == {"All rights reserved."}


@requires_ffmpeg
def test_heading_words_are_highlighted_and_their_pause_is_measured(engine, note, config):
    _, result = _render(note, config)
    data = _timings(result)
    heading = next(b for b in data["blocks"] if b["kind"] == "heading" and b["level"] == 2)
    in_heading = [w for w in data["words"] if data["sentences"][w["sentence"]]["block"] == heading["id"]]
    assert [w["spoken"] for w in in_heading] == ["Habits"]

    # Between the heading's last word and the next word: its full stop (a
    # token of its own) and the 800 ms heading pause.
    following = data["words"][data["words"].index(in_heading[-1]) + 1]
    gap = HEADING_SILENCE_MS / 1000 + token_samples(".") / SAMPLE_RATE
    assert following["start"] - in_heading[-1]["end"] == pytest.approx(gap, abs=0.002)


@requires_ffmpeg
def test_long_sentence_split_across_chunks_is_one_sentence(engine, tmp_path, config):
    clause = "this clause runs on for a while, "
    path = tmp_path / "long.md"
    path.write_text("# Long\n\n" + "Here " + clause * 12 + "and it ends.\n")
    plan, result = _render(path, config)
    data = _timings(result)

    assert sum(len(s.chunks) for s in plan.sections) > 2, "fixture must split the sentence"
    body = [s for s in data["sentences"] if data["blocks"][s["block"]]["kind"] == "paragraph"]
    assert len(body) == 1
    assert data["display_text"][body[0]["src"][0] : body[0]["src"][1]].endswith("and it ends.")


def test_validator_reports_broken_documents():
    good = {
        "version": 1, "title": "t", "audio": "a.m4a", "duration_s": 2.0, "audio_offset_s": 0.0,
        "voice": "v", "speed": 1.0, "timing_source": "engine", "display_text": "Hello there.",
        "markup": [], "blocks": [{"id": 0, "kind": "paragraph", "src": [0, 12]}],
        "sentences": [{"id": 0, "block": 0, "src": [0, 12], "start": 0.0, "end": 1.0}],
        "words": [
            {"spoken": "Hello", "start": 0.0, "end": 0.5, "src": [0, 5], "sentence": 0},
            {"spoken": "there", "start": 0.5, "end": 1.0, "src": [6, 11], "sentence": 0},
        ],
    }
    assert validate_timings(good) == []

    def broken(mutate) -> list[str]:
        data = json.loads(json.dumps(good))
        mutate(data)
        return validate_timings(data)

    assert broken(lambda d: d.update(version=2))
    assert broken(lambda d: d["words"].reverse())
    assert broken(lambda d: d["words"][0].update(src=[0, 99]))
    assert broken(lambda d: d["words"][0].update(start=0.0001))
    assert broken(lambda d: d["sentences"].append({"id": 1, "block": 0, "src": [0, 5], "start": 1.0, "end": 1.0}))
    assert broken(lambda d: d["blocks"][0].update(kind="table"))
    assert broken(lambda d: d.pop("words"))


def test_dumped_timings_round_trip():
    data = {"version": 1, "words": [{"spoken": "a", "src": [0, 1]}], "markup": [[0, 1]], "title": "</x>"}
    assert json.loads(dump_timings(data)) == data


# --- encoding -------------------------------------------------------------


def _speechy(seconds: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(SAMPLE_RATE * seconds)
    envelope = (np.sin(np.linspace(0, 40 * np.pi, n)) > 0).astype(np.float32)
    return (0.3 * envelope * rng.standard_normal(n)).astype(np.float32)


@requires_ffmpeg
@pytest.mark.parametrize("audio_format", ["m4a", "mp3"])
@pytest.mark.parametrize("loudnorm", [True, False])
def test_encoded_audio_keeps_length_rate_and_start(tmp_path, audio_format, loudnorm):
    pcm = _speechy()
    path = write_chapter(
        pcm,
        tmp_path / f"x.{audio_format}",
        sample_rate=SAMPLE_RATE,
        audio_format=audio_format,
        normalize_loudness=loudnorm,
        output_sample_rate=SAMPLE_RATE,
    )
    check = check_encoded(path, pcm, sample_rate=SAMPLE_RATE, audio_format=audio_format)

    assert check.encoded_sample_rate == SAMPLE_RATE
    assert abs(check.length_difference) <= CODEC_FRAME_SAMPLES[audio_format]
    assert check.measured_offset_s is not None and abs(check.measured_offset_s) <= 0.02
    assert check.applied_offset_s == 0.0
    assert check.problems() == []


@requires_ffmpeg
def test_existing_encodes_are_left_at_their_old_rate(tmp_path):
    """Without the opt-in, loudnorm still lifts m4a to 96 kHz, exactly as before."""
    path = write_chapter(_speechy(1.0), tmp_path / "x.m4a", sample_rate=SAMPLE_RATE, audio_format="m4a")
    assert probe_sample_rate(path) == 96_000


def test_offset_measurement_finds_a_known_delay():
    pcm = _speechy()
    delayed = np.concatenate([np.zeros(int(0.030 * SAMPLE_RATE), np.float32), pcm * 0.5])
    offset, correlation = measure_offset(pcm, delayed, sample_rate=SAMPLE_RATE)
    assert offset == pytest.approx(0.030, abs=1 / SAMPLE_RATE)
    assert correlation > 0.9


def test_offset_measurement_declines_on_silence():
    silence = np.zeros(SAMPLE_RATE, np.float32)
    assert measure_offset(silence, silence, sample_rate=SAMPLE_RATE)[0] is None


# --- the page -------------------------------------------------------------


def _inline_timings(page: str) -> dict:
    match = re.search(r'<script id="timings" type="application/json">(.*?)</script>', page, re.DOTALL)
    assert match, "timings script missing"
    return json.loads(match.group(1))


@requires_ffmpeg
def test_page_inlines_the_timings_and_reaches_nothing_external(engine, note, config):
    _, result = _render(note, config)
    page = result.paths.page.read_text(encoding="utf-8")

    assert _inline_timings(page) == _timings(result)
    outside_data = re.sub(r'<script id="timings".*?</script>', "", page, flags=re.DOTALL)
    for marker in ("http://", "https://", "//cdn", "fetch("):
        assert marker not in outside_data, marker
    assert f'src="{result.paths.audio.name}"' in page


def test_document_text_cannot_break_out_of_the_script():
    data = {"display_text": "</script><script>alert(1)</script> <!-- x"}
    inline = inline_json(data)
    assert "<" not in inline
    assert json.loads(inline) == data


@requires_ffmpeg
def test_embedded_audio_makes_one_self_contained_file(engine, note, config):
    _, result = _render(note, config, embed_audio=True)
    page = result.paths.page.read_text(encoding="utf-8")
    assert 'src="data:audio/mp4;base64,' in page


# --- caching and edits ----------------------------------------------------


@requires_ffmpeg
def test_article_and_read_share_every_chunk(engine, note, config):
    config.audio_format = "mp3"
    book = load_article(note)
    render_article(plan_article(book, config), config, output=Path(config.output_dir) / "a.mp3")
    rendered = len(engine.calls)

    config.audio_format = "m4a"
    _, result = _render(note, config)
    assert len(engine.calls) == rendered
    assert result.cache_summary.endswith("0 synthesized")


@requires_ffmpeg
def test_editing_one_sentence_resynthesizes_one_chunk_and_updates_timings(engine, note, config):
    _render(note, config)
    before = len(engine.calls)

    note.write_text(note.read_text().replace("Keep a regular bedtime.", "Keep a steady bedtime."))
    _, result = _render(note, config)
    data = _timings(result)

    assert len(engine.calls) - before == 1
    assert "steady" in data["display_text"] and "regular" not in data["display_text"]
    steady = next(w for w in data["words"] if w["spoken"] == "steady")
    assert _shown(data, steady) == "steady"
    assert validate_timings(data) == []


@requires_ffmpeg
def test_engine_without_timings_falls_back_to_estimates(monkeypatch, note, config):
    monkeypatch.setattr("narrate.article.get_engine", lambda *a, **k: UntimedStubEngine())
    _, result = _render(note, config)
    data = _timings(result)

    assert data["timing_source"] == "estimated"
    assert validate_timings(data) == []
    assert any("estimated" in warning for warning in result.warnings)


def test_empty_document_is_refused(engine, tmp_path, config):
    path = tmp_path / "empty.md"
    path.write_text("---\ntags: [x]\n---\n<!-- nothing -->\n\n```\ncode only\n```\n")
    plan = plan_article(load_article(path), config)
    paths = readalong_paths(path, output_dir=config.output_dir, audio_format="m4a")
    with pytest.raises(ValueError, match="nothing to narrate"):
        render_readalong(plan, config, paths=paths)


def test_build_timings_handles_estimated_chunks_without_tokens_in_text():
    """Estimated words carry punctuation ("dollars,"); they still place."""
    chapter = Chapter(1, "T", "# T\n\nIt cost $5, they said.")
    chunk = Chunk(text="It cost five dollars, they said.", chapter_index=1, order=0)
    audio = np.zeros(SAMPLE_RATE, np.float32)
    words = [WordTiming(w, i * 0.1, i * 0.1 + 0.1) for i, w in enumerate(chunk.text.split())]
    rendered = [RenderedChunk(chunk, SynthesisResult(audio, SAMPLE_RATE, words), 0)]
    timeline = build_timeline([rendered], sample_rate=SAMPLE_RATE)

    data, stats = build_timings(
        title="T", audio_name="t.m4a", voice="v", speed=1.0,
        sections=[SectionText(chapter, ["It cost five dollars, they said."])],
        rendered=rendered, timeline=timeline, timing_source="estimated",
    )
    assert validate_timings(data) == []
    assert stats.token_misses == 0
    assert [_shown(data, w) for w in data["words"]][2:4] == ["$5"] * 2


# --- command line ---------------------------------------------------------


def _cli(*args):
    from narrate.cli import app

    return CliRunner().invoke(app, ["read", *map(str, args)])


@requires_ffmpeg
def test_cli_writes_audio_timings_and_page(engine, note, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    result = _cli(note, "--work-dir", tmp_path / "work", "--output-dir", tmp_path / "out", "--open")

    assert result.exit_code == 0, result.output
    folder = tmp_path / "out" / "sleep"
    assert sorted(p.name for p in folder.iterdir()) == ["index.html", "sleep.m4a", "sleep.timings.json"]
    assert opened and opened[0].startswith("file://") and opened[0].endswith("index.html")


@requires_ffmpeg
def test_cli_warns_that_mp3_sync_can_drift(engine, note, tmp_path):
    result = _cli(note, "--format", "mp3", "--work-dir", tmp_path / "w", "--output-dir", tmp_path / "o")
    assert result.exit_code == 0, result.output
    assert "m4a is exact" in result.output
    assert (tmp_path / "o" / "sleep" / "sleep.mp3").is_file()


def test_cli_refuses_formats_without_reliable_sync(engine, note, tmp_path):
    result = _cli(note, "--format", "wav", "--output-dir", tmp_path / "o")
    assert result.exit_code == 1
    assert "m4a, mp3" in result.output


def test_cli_explains_an_empty_document(engine, tmp_path):
    path = tmp_path / "empty.md"
    path.write_text("<!-- only a comment -->\n")
    result = _cli(path, "--output-dir", tmp_path / "o", "--work-dir", tmp_path / "w")
    assert result.exit_code == 1
    assert "Nothing to narrate" in result.output


def test_cli_points_books_elsewhere_for_now(engine, tmp_path):
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF")
    result = _cli(path, "--output-dir", tmp_path / "o")
    assert result.exit_code == 1


# --- in a browser (optional) ----------------------------------------------

CHECK_SCRIPT = Path(__file__).parent / "browser" / "readalong_check.js"


def _node_playwright() -> str | None:
    node = shutil.which("node")
    if node is None:
        return None
    probe = subprocess.run([node, "-e", "require('playwright')"], capture_output=True)
    return node if probe.returncode == 0 else None


@requires_ffmpeg
def test_page_plays_in_a_browser(engine, tmp_path, config):
    """Plays a stub render in headless Chromium: the highlight follows the
    clock, clicks seek, keys step, and the page scrolls - unless the reader
    just scrolled away."""
    node = _node_playwright()
    if node is None:
        pytest.skip("needs Node.js with Playwright")

    paragraphs = "\n\n".join(
        f"Paragraph {i} talks about sleep for a while. It has a second sentence too."
        for i in range(1, 16)
    )
    path = tmp_path / "long.md"
    # The last sentence is long, so the manual-scroll check below lands inside it.
    ending = "The last words are here, and they go on for quite a while longer than the rest."
    path.write_text(f"# Long Read\n\n{paragraphs}\n\n## Ending\n\n{ending}\n")
    _, result = _render(path, config)

    run = subprocess.run(
        [node, str(CHECK_SCRIPT), str(result.paths.page.resolve())],
        capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)

    assert report["errors"] == []
    assert not report["layout"]["overflowX"], "page scrolls sideways at phone width"
    assert "#" not in report["layout"]["text"]
    assert report["moved"] >= 3, "highlight did not move during playback"
    assert report["outOfStep"] == 0
    assert report["afterClick"]["litText"] == report["targetText"]
    assert report["afterNext"]["sentence"] == report["before"] + 1
    assert report["afterParagraph"]["sentence"] > report["afterNext"]["sentence"]
    assert report["scrolled"]["visible"] and report["scrolled"]["scrollY"] > 0
    assert report["suspended"]["sentence"] == report["suspended"]["last"]
    assert report["suspended"]["scrollY"] < 50, "auto-scroll ignored a manual scroll"
