"""Engine word timings and the cache sidecars that store them.

Everything here runs without model weights. Kokoro's extraction is exercised
with hand-built pipeline results whose expected times are worked out by hand,
and the end-to-end paths use deterministic stub engines. The one test that
needs real weights is marked ``kokoro`` and skips when they are not cached.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from stub_engine import SAMPLE_RATE, TimedStubEngine, UntimedStubEngine, token_samples

from narrate.audio import concat_with_silence
from narrate.document import Book, Chapter, Chunk
from narrate.pipeline import (
    NarrationConfig,
    narrate,
    plan_chapter,
    synthesize_chunk_with_timings,
    synthesize_chunks,
)
from narrate.tts import AudioCache, SynthesisResult, WordTiming, synthesize_with_timings
from narrate.tts.cache import TIMINGS_SUFFIX
from narrate.tts.kokoro_engine import (
    KokoroEngine,
    _fill_gaps,
    times_from_durations,
    token_timings,
)
from narrate.tts.timing_estimate import estimate_word_timings, with_estimated_timings


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


def _cache(tmp_path, **kwargs) -> AudioCache:
    return AudioCache(tmp_path / "cache", sample_rate=SAMPLE_RATE, **kwargs)


def _chunk(text: str = "Hello there.") -> Chunk:
    return Chunk(text=text, chapter_index=1, order=0)


KEY = {"engine": "stub", "voice": "v", "speed": 1.0}


# --- the engine contract --------------------------------------------------


def test_untimed_engine_reports_no_words():
    engine = UntimedStubEngine()
    result = synthesize_with_timings(engine, "Hello there.", voice="test")

    assert result.words is None
    assert result.sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(result.audio, engine.synthesize("Hello there.", voice="test"))


def test_timed_engine_reports_tokens_including_punctuation():
    result = synthesize_with_timings(TimedStubEngine(), "Hi, Doctor Smith.", voice="test")
    assert [w.text for w in result.words] == ["Hi", ",", "Doctor", "Smith", "."]
    assert result.words[0].start == 0.0
    assert result.words[-1].end == pytest.approx(result.duration)


def test_timed_and_plain_synthesis_return_the_same_samples():
    engine = TimedStubEngine()
    text = "The deal was four point two million dollars."
    np.testing.assert_array_equal(
        engine.synthesize(text), engine.synthesize_with_timings(text).audio
    )


def test_slower_speed_stretches_the_timings():
    """Timings come from the rendered audio, so --speed is reflected in them."""
    engine = TimedStubEngine()
    normal = engine.synthesize_with_timings("Sleep matters a great deal.", speed=1.0)
    slow = engine.synthesize_with_timings("Sleep matters a great deal.", speed=0.9)

    assert slow.duration == pytest.approx(normal.duration / 0.9, rel=1e-3)
    for a, b in zip(normal.words, slow.words):
        assert b.start == pytest.approx(a.start / 0.9, abs=1e-3)


# --- fallback estimator ---------------------------------------------------


def test_estimate_is_proportional_to_word_length():
    words = estimate_word_timings("a bb cccc", 0.7)
    assert [(w.text, round(w.start, 6), round(w.end, 6)) for w in words] == [
        ("a", 0.0, 0.1),
        ("bb", 0.1, 0.3),
        ("cccc", 0.3, 0.7),
    ]


def test_estimate_tiles_the_chunk_without_gaps():
    words = estimate_word_timings("Sleep is when the brain files away the day.", 3.2)
    assert words[0].start == 0.0
    assert words[-1].end == pytest.approx(3.2)
    for earlier, later in itertools.pairwise(words):
        assert earlier.end == pytest.approx(later.start)
        assert earlier.start < earlier.end


@pytest.mark.parametrize(("text", "duration"), [("", 1.0), ("   ", 1.0), ("words", 0.0)])
def test_estimate_of_nothing_is_empty(text, duration):
    assert estimate_word_timings(text, duration) == []


def test_estimate_only_fills_missing_timings():
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    real = SynthesisResult(audio, SAMPLE_RATE, [WordTiming("Hi", 0.2, 0.4)])
    assert with_estimated_timings(real, "Hi") is real

    filled = with_estimated_timings(SynthesisResult(audio, SAMPLE_RATE, None), "Hi there")
    assert [w.text for w in filled.words] == ["Hi", "there"]
    assert filled.words[-1].end == pytest.approx(1.0)
    assert filled.audio is audio


# --- Kokoro: times from pred_dur ------------------------------------------


def _token(text: str, phonemes: str, whitespace: str = " ", start=None, end=None):
    """Shaped like misaki's MToken, which is all the extraction reads."""
    return SimpleNamespace(
        text=text, phonemes=phonemes, whitespace=whitespace, start_ts=start, end_ts=end
    )


def _clean_tokens():
    return [_token("Hello", "abcde"), _token("world", "fgh", ""), _token(".", ".", "")]


#: BOS, then one entry per character of "abcde fgh.", then EOS.
CLEAN_DURATIONS = [5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]

#: Worked by hand in half-frames (80 per second). The lead-in is 2*(5-3) = 4.
#: Hello: starts at 4, spans 2*(1+2+3+4+5) plus half of the 6-frame space
#: on either side -> ends at 4+30+6 = 40. world: 40 -> 46+2*24 = 94.
#: ".": 94 -> 94+2*10 = 114.
CLEAN_EXPECTED = [(4 / 80, 40 / 80), (40 / 80, 94 / 80), (94 / 80, 114 / 80)]


def test_times_from_durations_matches_hand_computed_values():
    times = times_from_durations(_clean_tokens(), "abcde fgh.", CLEAN_DURATIONS)
    assert times == pytest.approx(CLEAN_EXPECTED)


def test_times_from_durations_agrees_with_kokoro_on_clean_input():
    """On input without phoneme-less tokens, upstream's walk is correct."""
    kokoro = pytest.importorskip("kokoro")
    torch = pytest.importorskip("torch")

    tokens = _clean_tokens()
    kokoro.KPipeline.join_timestamps(tokens, torch.tensor(CLEAN_DURATIONS))
    ours = times_from_durations(_clean_tokens(), "abcde fgh.", CLEAN_DURATIONS)
    assert ours == pytest.approx([(t.start_ts, t.end_ts) for t in tokens])


def _junk_tokens():
    return [
        _token("Hello", "abcde"),
        _token("~", ""),  # no phonemes, but trailing whitespace
        _token("world", "fgh", ""),
        _token(".", ".", ""),
    ]


#: "abcde  fgh." - the junk token's space is its own model input (7th char).
JUNK_DURATIONS = [5, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 1]


def test_phonemeless_token_does_not_shift_later_tokens():
    """Upstream advances its index twice here; every later token then gets
    the next phoneme's duration and the last runs off the end as None."""
    times = times_from_durations(_junk_tokens(), "abcde  fgh.", JUNK_DURATIONS)
    # Hello as before. "~" consumes its 6-frame space: 46+6 = 52, right 58.
    # world: 52 -> 58+2*24 = 106. ".": 106 -> 106+2*10 = 126.
    assert times[0] == pytest.approx((4 / 80, 40 / 80))
    assert times[1] == (None, None)
    assert times[2] == pytest.approx((52 / 80, 106 / 80))
    assert times[3] == pytest.approx((106 / 80, 126 / 80))


def test_upstream_loses_the_last_token_after_a_phonemeless_one():
    """Documents the upstream behaviour the recomputation exists to avoid."""
    kokoro = pytest.importorskip("kokoro")
    torch = pytest.importorskip("torch")

    tokens = _junk_tokens()
    kokoro.KPipeline.join_timestamps(tokens, torch.tensor(JUNK_DURATIONS))
    assert tokens[-1].start_ts is None


def test_characters_outside_the_vocab_have_no_duration_entry():
    """KModel.forward drops them, so they must not consume a pred_dur index."""
    tokens = [_token("Hello", "abXcde"), _token("world", "fgh", ""), _token(".", ".", "")]
    vocab = set("abcdefgh. ")
    times = times_from_durations(tokens, "abXcde fgh.", CLEAN_DURATIONS, vocab)
    assert times == pytest.approx(CLEAN_EXPECTED)


def test_tokens_past_truncation_have_no_time():
    """The pipeline truncates past 510 phonemes; tokens beyond that went unspoken."""
    times = times_from_durations(_clean_tokens(), "abcde fg", CLEAN_DURATIONS[:9] + [1])
    assert times[0] == pytest.approx(CLEAN_EXPECTED[0])
    assert times[1:] == [(None, None), (None, None)]


def test_missing_times_are_filled_in_order():
    filled = _fill_gaps([(0.0, 0.2), (None, None), (0.3, None), (None, 0.5)], duration=1.0)
    # A missing start takes the previous end; a missing end the next start.
    assert filled[0] == (0.0, 0.2)
    assert filled[1] == (0.2, 0.3)
    assert filled[2] == (0.3, 1.0)  # the next start is unknown too
    assert filled[3] == (1.0, 1.0)


def test_filled_times_never_go_backwards_or_past_the_audio():
    filled = _fill_gaps([(0.5, 0.9), (0.4, 0.45), (1.2, 1.6)], duration=1.0)
    starts = [start for start, _ in filled]
    assert starts == sorted(starts)
    assert all(0.0 <= start <= end <= 1.0 for start, end in filled)


def test_a_final_missing_end_runs_to_the_end_of_the_chunk():
    assert _fill_gaps([(0.1, None)], duration=0.8) == [(0.1, 0.8)]


def _result(samples: int, tokens, phonemes: str = "", pred_dur=None):
    return SimpleNamespace(
        audio=np.full(samples, 0.1, dtype=np.float32),
        tokens=tokens,
        phonemes=phonemes,
        pred_dur=pred_dur,
        graphemes="",
    )


def test_each_internal_result_is_offset_by_the_audio_before_it():
    """The pipeline splits on newlines and at 510 phonemes; each result's
    times start from zero."""
    first = _result(2400, [_token("One", "x", "", 0.0, 0.05), _token(".", ".", "", 0.05, 0.1)])
    second = _result(4800, [_token("Two", "y", "", 0.0, 0.15)])
    words = token_timings([first, second], sample_rate=SAMPLE_RATE)

    assert [w.text for w in words] == ["One", ".", "Two"]
    assert words[2].start == pytest.approx(0.1)
    assert words[2].end == pytest.approx(0.25)


def test_results_without_audio_do_not_advance_the_clock():
    empty = SimpleNamespace(audio=None, tokens=[_token("Lost", "x", "", 0.0, 0.1)])
    second = _result(2400, [_token("Kept", "y", "", 0.02, 0.08)])
    words = token_timings([empty, second], sample_rate=SAMPLE_RATE)
    assert [(w.text, w.start) for w in words] == [("Kept", pytest.approx(0.02))]


def test_old_tuple_results_carry_no_timings():
    legacy = [("Hello.", "həlˈO.", np.ones(2400, dtype=np.float32))]
    assert token_timings(legacy, sample_rate=SAMPLE_RATE) is None


def test_pred_dur_is_preferred_over_upstream_timestamps():
    tokens = _clean_tokens()
    for token in tokens:
        token.start_ts, token.end_ts = 9.0, 9.5  # deliberately wrong
    words = token_timings(
        [_result(2 * SAMPLE_RATE, tokens, "abcde fgh.", CLEAN_DURATIONS)],
        sample_rate=SAMPLE_RATE,
    )
    assert [(w.start, w.end) for w in words] == pytest.approx(CLEAN_EXPECTED)


class _FakePipeline:
    """Stands in for a loaded KPipeline: same call signature, canned results."""

    def __init__(self, results):
        self.results = results
        self.model = SimpleNamespace(vocab=None)
        self.calls = 0

    def __call__(self, text, voice=None, speed=1.0):
        self.calls += 1
        yield from self.results


def _engine_with(results) -> KokoroEngine:
    engine = KokoroEngine(device="cpu")
    engine._pipeline = _FakePipeline(results)
    return engine


def test_kokoro_timed_audio_is_the_plain_audio():
    results = [
        _result(24_000, _clean_tokens(), "abcde fgh.", CLEAN_DURATIONS),
        _result(12_000, [_token("Again", "z", "", 0.0, 0.3)]),
    ]
    engine = _engine_with(results)

    plain = engine.synthesize("Hello world.", voice="af_heart")
    timed = engine.synthesize_with_timings("Hello world.", voice="af_heart")

    np.testing.assert_array_equal(plain, timed.audio)
    assert [w.text for w in timed.words] == ["Hello", "world", ".", "Again"]
    assert timed.words[3].start == pytest.approx(1.0)


def test_broken_tokens_cost_the_timings_not_the_audio(caplog):
    bad = SimpleNamespace(
        audio=np.ones(2400, dtype=np.float32), tokens=[object()], phonemes="", pred_dur=None
    )
    engine = _engine_with([bad])

    with caplog.at_level(logging.WARNING):
        result = engine.synthesize_with_timings("Hello.", voice="af_heart")

    assert result.words is None
    assert result.audio.size == 2400
    assert "timings" in caplog.text


def test_kokoro_empty_text_needs_no_model():
    engine = KokoroEngine(device="cpu")
    result = engine.synthesize_with_timings("   ", voice="af_heart")
    assert result.audio.size == 0 and result.words == []


# --- cache sidecars -------------------------------------------------------


def _timed(seconds: float = 0.1, words=None) -> SynthesisResult:
    audio = np.linspace(-0.5, 0.5, int(SAMPLE_RATE * seconds), dtype=np.float32)
    if words is None:
        words = [WordTiming("Hello", 0.0, seconds / 2), WordTiming("there", seconds / 2, seconds)]
    return SynthesisResult(audio, SAMPLE_RATE, words)


def test_cache_keys_are_unchanged():
    """Timings must not invalidate anyone's cache: these keys predate them."""
    assert _chunk().cache_key("kokoro", "af_heart", 1.0) == (
        "7a906e40610cacf779c07e8cb39b6d432b89391cd4e7b850e98ab3e40717981b"
    )
    chunk = Chunk(text="Four point two million dollars.", chapter_index=3, order=7)
    assert chunk.cache_key("stub", "test", 0.9) == (
        "fd649551b7e1fff2b5e4b0fdc4ef4072110c7db28520f086808891a918b23a07"
    )


def test_sidecar_sits_next_to_the_audio(tmp_path):
    cache = _cache(tmp_path)
    audio_path = cache.path_for(_chunk(), **KEY)
    sidecar = cache.timings_path_for(_chunk(), **KEY)

    assert sidecar.parent == audio_path.parent
    assert sidecar.name == audio_path.stem + TIMINGS_SUFFIX


def test_timings_round_trip(tmp_path):
    cache = _cache(tmp_path)
    original = _timed()
    cache.put_result(_chunk(), original, **KEY)

    restored = cache.get_with_timings(_chunk(), **KEY)
    assert restored is not None
    np.testing.assert_allclose(restored.audio, original.audio, atol=1e-6)
    assert restored.words == original.words
    assert cache.hits == 1


def test_engine_without_timings_is_cached_as_such(tmp_path):
    """Not a miss: re-asking an engine that cannot time words won't help."""
    cache = _cache(tmp_path)
    cache.put_result(_chunk(), SynthesisResult(_timed().audio, SAMPLE_RATE, None), **KEY)

    restored = cache.get_with_timings(_chunk(), **KEY)
    assert restored is not None and restored.words is None
    assert json.loads(cache.timings_path_for(_chunk(), **KEY).read_text())["words"] is None


def test_audio_cached_before_timings_is_a_timed_miss(tmp_path, caplog):
    cache = _cache(tmp_path)
    for text in ("One.", "Two."):
        cache.put(_chunk(text), _timed().audio, **KEY)

    with caplog.at_level(logging.WARNING, logger="narrate.tts.cache"):
        assert cache.get_with_timings(_chunk("One."), **KEY) is None
        assert cache.get_with_timings(_chunk("Two."), **KEY) is None

    assert cache.stale == 2
    assert caplog.text.count("before word timings") == 1, "should log once"
    # The plain path still reuses it - speak and article never re-render.
    assert cache.get(_chunk("One."), **KEY) is not None


@pytest.mark.parametrize(
    "content",
    ["not json", '{"version": 99, "words": []}', '{"version": 1}', '{"version": 1, "words": [[1]]}'],
)
def test_unreadable_sidecar_is_discarded(tmp_path, content):
    cache = _cache(tmp_path)
    cache.put_result(_chunk(), _timed(), **KEY)
    sidecar = cache.timings_path_for(_chunk(), **KEY)
    sidecar.write_text(content)

    assert cache.get_with_timings(_chunk(), **KEY) is None
    assert not sidecar.exists()


def test_timings_that_outlast_their_audio_are_discarded(tmp_path):
    cache = _cache(tmp_path)
    cache.put_result(_chunk(), _timed(0.1, [WordTiming("Hello", 0.0, 0.5)]), **KEY)
    assert cache.get_with_timings(_chunk(), **KEY) is None


def test_plain_put_removes_timings_for_a_different_render(tmp_path):
    cache = _cache(tmp_path)
    cache.put_result(_chunk(), _timed(), **KEY)
    cache.put(_chunk(), _timed(0.2).audio, **KEY)

    assert not cache.timings_path_for(_chunk(), **KEY).exists()
    assert cache.get_with_timings(_chunk(), **KEY) is None


def test_put_result_leaves_no_staging_files(tmp_path):
    cache = _cache(tmp_path)
    cache.put_result(_chunk(), _timed(), **KEY)
    assert not list((tmp_path / "cache").rglob("*.partial"))
    assert len(list((tmp_path / "cache").rglob("*" + TIMINGS_SUFFIX))) == 1


def test_disabled_cache_stores_no_timings(tmp_path):
    cache = _cache(tmp_path, enabled=False)
    cache.put_result(_chunk(), _timed(), **KEY)
    assert cache.get_with_timings(_chunk(), **KEY) is None
    assert not (tmp_path / "cache").exists()


def test_empty_audio_is_not_cached(tmp_path):
    cache = _cache(tmp_path)
    cache.put_result(_chunk(), SynthesisResult(np.zeros(0, np.float32), SAMPLE_RATE, []), **KEY)
    assert not list((tmp_path / "cache").rglob("*"))


# --- pipeline -------------------------------------------------------------


def _chunks(engine) -> list[Chunk]:
    from narrate.normalize import RuleNormalizer

    chapter = Chapter(1, "Ch", "# Ch\n\nThe deal was $4.2M. Dr. Smith agreed.\n\nIt closed in 1997.")
    return plan_chapter(chapter, engine=engine, normalizer=RuleNormalizer())


def test_synthesize_chunks_output_is_unchanged(tmp_path, config):
    """Byte-identical to joining the engine's own output, as before timings."""
    engine = UntimedStubEngine()
    chunks = _chunks(engine)
    cache = _cache(tmp_path)

    track = synthesize_chunks(chunks, engine=engine, cache=cache, config=config)

    reference = UntimedStubEngine()
    expected = concat_with_silence(
        [(reference.synthesize(c.text), c.trailing_silence_ms) for c in chunks],
        sample_rate=SAMPLE_RATE,
    )
    np.testing.assert_array_equal(track, expected)
    assert engine.calls == [c.text for c in chunks]


def test_synthesize_chunks_stores_timings_alongside(tmp_path, config):
    engine = TimedStubEngine()
    chunks = _chunks(engine)
    cache = _cache(tmp_path)
    synthesize_chunks(chunks, engine=engine, cache=cache, config=config)

    for chunk in chunks:
        stored = cache.get_with_timings(chunk, engine=engine.name, voice="test", speed=1.0)
        assert stored is not None
        assert stored.words == engine.synthesize_with_timings(chunk.text).words


def test_timed_chunk_is_synthesized_once(tmp_path, config):
    engine = TimedStubEngine()
    chunk = _chunk("Four point two million dollars.")
    cache = _cache(tmp_path)

    first = synthesize_chunk_with_timings(chunk, engine=engine, cache=cache, config=config)
    second = synthesize_chunk_with_timings(chunk, engine=engine, cache=cache, config=config)

    assert len(engine.calls) == 1
    assert first.words == second.words
    np.testing.assert_allclose(first.audio, second.audio, atol=1e-6)


def test_old_cache_entry_is_re_synthesized_exactly_once(tmp_path, config):
    engine = TimedStubEngine()
    chunk = _chunk("Four point two million dollars.")
    cache = _cache(tmp_path)
    cache.put(chunk, engine.synthesize(chunk.text), engine=engine.name, voice="test", speed=1.0)
    engine.calls.clear()

    for _ in range(3):
        result = synthesize_chunk_with_timings(chunk, engine=engine, cache=cache, config=config)

    assert engine.calls == [chunk.text]
    assert result.words is not None


def test_untimed_engine_result_has_no_words_through_the_cache(tmp_path, config):
    engine = UntimedStubEngine()
    chunk = _chunk()
    cache = _cache(tmp_path)

    for _ in range(2):
        result = synthesize_chunk_with_timings(chunk, engine=engine, cache=cache, config=config)
    assert result.words is None
    assert len(engine.calls) == 1


def test_speak_then_timed_path_reuses_every_chunk(monkeypatch, config):
    """A book narrated with `speak` needs no re-render for read-along."""
    engine = TimedStubEngine()
    monkeypatch.setattr("narrate.pipeline.get_engine", lambda *a, **k: engine)
    book = Book(
        title="Shared",
        chapters=[Chapter(1, "One", "# One\n\n" + "Alpha beta gamma delta. " * 12)],
    )
    narrate(book, config)
    rendered = len(engine.calls)

    cache = AudioCache(
        Path(config.work_dir) / book.slug / "cache", sample_rate=SAMPLE_RATE
    )
    from narrate.normalize import RuleNormalizer

    for chunk in plan_chapter(book.chapters[0], engine=engine, normalizer=RuleNormalizer()):
        synthesize_chunk_with_timings(chunk, engine=engine, cache=cache, config=config)

    assert len(engine.calls) == rendered
    assert cache.stale == 0


def test_stub_word_times_are_cumulative_sums():
    """The property later timeline tests lean on."""
    result = TimedStubEngine().synthesize_with_timings("Hi, Doctor Smith.")
    cursor = 0
    for word in result.words:
        assert word.start == cursor / SAMPLE_RATE
        cursor += token_samples(word.text)
        assert word.end == cursor / SAMPLE_RATE
    assert cursor == result.audio.size


# --- real Kokoro (optional) -----------------------------------------------

PARAGRAPH = (
    "The deal was four point two million dollars, per Doctor Smith. "
    "The F. B. I. knew; roughly twelve percent was never read aloud."
)


@pytest.fixture
def kokoro_offline(monkeypatch):
    """Skip unless Kokoro and its weights are already in the local HF cache.

    Forced offline, so a missing file is a skip rather than a download.
    """
    pytest.importorskip("kokoro")
    hub = pytest.importorskip("huggingface_hub")
    needed = ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/bf_emma.pt")
    if not all(
        isinstance(hub.try_to_load_from_cache("hexgrad/Kokoro-82M", name), str) for name in needed
    ):
        pytest.skip("Kokoro-82M weights are not in the local Hugging Face cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True, raising=False)


@pytest.mark.kokoro
@pytest.mark.parametrize("voice", ["af_heart", "bf_emma"])
def test_real_kokoro_reports_ordered_word_timings(kokoro_offline, voice):
    """British voices use a different G2P; timings must come through for both."""
    from narrate.tts.kokoro_engine import lang_code_for_voice

    engine = KokoroEngine(lang_code=lang_code_for_voice(voice), device="cpu")
    result = engine.synthesize_with_timings(PARAGRAPH, voice=voice)

    words = [w for w in result.words if any(c.isalnum() for c in w.text)]
    assert [w.text for w in words][:4] == ["The", "deal", "was", "four"]
    assert all(0 <= w.start <= w.end <= result.duration for w in words)
    assert [w.start for w in words] == sorted(w.start for w in words)
    assert words[0].start < 0.5


@pytest.mark.kokoro
def test_real_kokoro_timings_follow_speed(kokoro_offline):
    engine = KokoroEngine(device="cpu")
    normal = engine.synthesize_with_timings(PARAGRAPH, voice="af_heart", speed=1.0)
    slow = engine.synthesize_with_timings(PARAGRAPH, voice="af_heart", speed=0.9)

    assert slow.duration > normal.duration
    assert slow.words[-1].end > normal.words[-1].end
