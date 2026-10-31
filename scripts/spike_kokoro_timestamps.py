"""Phase 1 spike: are Kokoro's token timestamps good enough to drive read-along?

Throwaway. It renders one paragraph through the *same* call that
``KokoroEngine.synthesize`` makes - ``KPipeline(text, voice=, speed=)`` - and
reads the ``tokens`` that path already returns, instead of discarding them as
``_collect`` does. It then:

1. prints ``text, start, end`` per token, with each internal result offset by
   the audio before it;
2. checks the timestamps against the audio itself: speech onset and offset, and
   whether the pauses an energy detector finds line up with token boundaries;
3. writes ``spike.wav``, ``spike.json`` and ``index.html``, a hard-coded page
   that highlights the paragraph word by word, with a nudge slider for judging
   the offset by ear.

Usage::

    uv run --extra kokoro python scripts/spike_kokoro_timestamps.py
    uv run --extra kokoro python scripts/spike_kokoro_timestamps.py --voice bf_emma --speed 0.9
    uv run --extra kokoro python scripts/spike_kokoro_timestamps.py --g2p-only   # no weights needed

Open ``out/spike/index.html`` in a browser afterwards.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24_000

#: Samples per duration-predictor frame. ``join_timestamps`` divides
#: half-frames by 80, i.e. 40 frames per second at 24 kHz.
SAMPLES_PER_FRAME = 600

DEFAULT_TEXT = (
    "The deal was $4.2M, per Dr. Smith [12]. The FBI reviewed it between "
    "1939-1945, e.g. in the 21st volume; roughly 12% of Chapter IV was never "
    "read aloud. “Quietly,” she said — and then, after a long pause, "
    "nothing more."
)

PAGE_TEMPLATE = Path(__file__).with_name("spike_player.html")


# --- synthesis ------------------------------------------------------------


def synthesize(spoken: str, *, voice: str, speed: float):
    """Run the exact pipeline call the engine makes and keep every result.

    Returns the results and the model's phoneme vocab, which decides how
    phoneme characters map onto ``pred_dur`` entries.
    """
    from narrate.tts.kokoro_engine import KokoroEngine, lang_code_for_voice

    engine = KokoroEngine(lang_code=lang_code_for_voice(voice))
    pipeline = engine._ensure_loaded()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        results = list(pipeline(spoken, voice=voice, speed=speed))
    return results, getattr(pipeline.model, "vocab", None)


def rederive_times(tokens, phonemes: str, pred_dur, vocab) -> list[tuple[float | None, float | None]]:
    """Token times recomputed from ``pred_dur`` - the engine's own code path.

    Upstream ``join_timestamps`` advances twice past a token with no phonemes
    but trailing whitespace, and ignores the vocab filter in ``KModel.forward``.
    ``narrate.tts.kokoro_engine.times_from_durations`` fixes both; the spike
    calls it so a run measures exactly what read-along will use.
    """
    from narrate.tts.kokoro_engine import times_from_durations

    return times_from_durations(tokens, phonemes, [int(d) for d in pred_dur.tolist()], vocab)


def flatten(results, vocab=None) -> tuple[list[dict], np.ndarray, list[dict]]:
    """Tokens with absolute times, the joined audio, and per-result checks.

    A chunk can come back as several results (the pipeline splits on newlines
    and at 510 phonemes). Each result's timestamps start at zero, so they are
    offset by the samples of the results before it.
    """
    rows: list[dict] = []
    parts: list[np.ndarray] = []
    checks: list[dict] = []
    cursor = 0

    for index, result in enumerate(results):
        audio = result.audio
        if audio is None:
            continue
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        pred_dur = getattr(result, "pred_dur", None)
        frames = int(pred_dur.sum().item()) if pred_dur is not None else None
        checks.append(
            {
                "result": index,
                "graphemes": result.graphemes,
                "samples": int(audio.size),
                "pred_dur_samples": None if frames is None else frames * SAMPLES_PER_FRAME,
            }
        )

        offset = cursor / SAMPLE_RATE
        tokens = result.tokens or []
        fixed = (
            rederive_times(tokens, result.phonemes, pred_dur, vocab)
            if pred_dur is not None and tokens
            else [(None, None)] * len(tokens)
        )
        for token, (start, end) in zip(tokens, fixed):
            rows.append(
                {
                    "result": index,
                    "offset": round(offset, 4),
                    "text": token.text,
                    "phonemes": token.phonemes,
                    "whitespace": token.whitespace,
                    # Upstream join_timestamps, relative to this result.
                    "start_ts": token.start_ts,
                    "end_ts": token.end_ts,
                    # Re-derived, absolute. These drive the page.
                    "start": None if start is None else round(offset + start, 4),
                    "end": None if end is None else round(offset + end, 4),
                }
            )

        parts.append(audio)
        cursor += audio.size

    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return rows, audio, checks


def is_word(text: str) -> bool:
    return any(char.isalnum() for char in text)


# --- measurement ----------------------------------------------------------

#: Energy-detector settings. Coarse on purpose: the question is whether the
#: highlight lands on the right word, which tolerates tens of milliseconds.
HOP_S = 0.010
MIN_PAUSE_S = 0.060
THRESHOLD_BELOW_PEAK_DB = 30.0


def speech_frames(audio: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Boolean per 10 ms hop: is there speech energy here?"""
    hop = int(sample_rate * HOP_S)
    count = len(audio) // hop
    if count == 0:
        return np.zeros(0, dtype=bool)
    frames = audio[: count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    threshold = max(rms.max() * 10 ** (-THRESHOLD_BELOW_PEAK_DB / 20), 1e-5)
    return rms > threshold


def pauses(active: np.ndarray) -> list[tuple[float, float]]:
    """Silent runs of at least MIN_PAUSE_S strictly inside the speech."""
    if not active.any():
        return []
    first, last = np.flatnonzero(active)[[0, -1]]
    runs: list[tuple[float, float]] = []
    start = None
    for i in range(first, last + 1):
        if not active[i] and start is None:
            start = i
        elif active[i] and start is not None:
            if (i - start) * HOP_S >= MIN_PAUSE_S:
                runs.append((start * HOP_S, i * HOP_S))
            start = None
    return runs


def measure(rows: list[dict], audio: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> dict:
    """Compare token times with where the audio actually has speech."""
    words = [r for r in rows if is_word(r["text"])]
    timed = [r for r in words if r["start"] is not None and r["end"] is not None]
    punctuation = [r for r in rows if not is_word(r["text"])]
    duration = len(audio) / sample_rate

    active = speech_frames(audio, sample_rate=sample_rate)
    stats: dict = {
        "duration_s": round(duration, 3),
        "tokens": len(rows),
        "word_tokens": len(words),
        "punctuation_tokens": len(punctuation),
        "words_missing_start": sum(r["start"] is None for r in words),
        "words_missing_end": sum(r["end"] is None for r in words),
        "punctuation_missing_times": sum(r["start"] is None for r in punctuation),
        "punctuation_durations_ms": [
            [r["text"], round((r["end"] - r["start"]) * 1000)]
            for r in punctuation
            if r["start"] is not None and r["end"] is not None
        ],
        "words_per_minute": round(len(words) / duration * 60) if duration else None,
        "upstream_missing_times": sum(r["start_ts"] is None for r in rows),
        "upstream_disagrees_ms": _upstream_disagreement(rows),
    }
    if not timed or not active.any():
        return stats

    onset = float(np.flatnonzero(active)[0] * HOP_S)
    offset = float((np.flatnonzero(active)[-1] + 1) * HOP_S)
    stats["speech_onset_s"] = round(onset, 3)
    stats["first_word_start_s"] = timed[0]["start"]
    stats["onset_error_ms"] = round((timed[0]["start"] - onset) * 1000)
    stats["speech_offset_s"] = round(offset, 3)
    stats["last_word_end_s"] = timed[-1]["end"]
    stats["offset_error_ms"] = round((timed[-1]["end"] - offset) * 1000)
    stats["non_monotonic_words"] = sum(
        b["start"] < a["start"] for a, b in itertools.pairwise(timed)
    )

    # A pause the detector hears should begin where a word ends and finish
    # where the next word starts. The distance to the nearest boundary is the
    # error a listener would perceive as the highlight running early or late.
    ends = np.array([r["end"] for r in timed])
    starts = np.array([r["start"] for r in timed])
    errors_end, errors_start = [], []
    for silent_from, silent_to in pauses(active):
        errors_end.append((ends - silent_from)[np.argmin(np.abs(ends - silent_from))])
        errors_start.append((starts - silent_to)[np.argmin(np.abs(starts - silent_to))])
    if errors_end:
        stats["pauses_detected"] = len(errors_end)
        stats["pause_vs_word_end_ms"] = _summary(errors_end)
        stats["pause_vs_next_start_ms"] = _summary(errors_start)
    return stats


def _upstream_disagreement(rows: list[dict]) -> dict:
    """How far upstream's join_timestamps is from the re-derived times.

    Compared relative to each result. Non-zero only when a token had no
    phonemes, or a phoneme character was missing from the vocab.
    """
    diffs = [
        abs(row["start"] - row["offset"] - row["start_ts"]) * 1000
        for row in rows
        if row["start"] is not None and row["start_ts"] is not None
    ]
    if not diffs:
        return {"compared": 0}
    return {"compared": len(diffs), "over_1ms": sum(d > 1 for d in diffs), "max": round(max(diffs))}


def _summary(errors: list[float]) -> dict:
    ms = np.array(errors) * 1000
    return {
        "median": round(float(np.median(ms))),
        "median_abs": round(float(np.median(np.abs(ms)))),
        "max_abs": round(float(np.max(np.abs(ms)))),
    }


# --- output ---------------------------------------------------------------


def write_outputs(
    out_dir: Path,
    *,
    display: str,
    spoken: str,
    voice: str,
    speed: float,
    rows: list[dict],
    audio: np.ndarray,
    checks: list[dict],
    stats: dict,
) -> Path:
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "spike.wav", audio, SAMPLE_RATE)

    data = {
        "display": display,
        "spoken": spoken,
        "voice": voice,
        "speed": speed,
        "sample_rate": SAMPLE_RATE,
        "audio": "spike.wav",
        "results": checks,
        "tokens": rows,
        "stats": stats,
    }
    (out_dir / "spike.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    # Inline rather than fetch(): fetch() of a local file fails on file:// in
    # Chrome. "</" is escaped so the text cannot close the script element.
    inline = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = PAGE_TEMPLATE.read_text(encoding="utf-8").replace("/*__SPIKE_DATA__*/null", inline)
    target = out_dir / "index.html"
    target.write_text(page, encoding="utf-8")
    return target


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


def print_tokens(rows: list[dict]) -> None:
    """``start``/``end`` are re-derived and absolute; ``ts`` columns are upstream's."""
    print(
        f"{'res':>3}  {'text':<16} {'phonemes':<18} {'start':>7} {'end':>7}"
        f"  {'start_ts':>8} {'end_ts':>8}  note"
    )
    for row in rows:
        note = "" if is_word(row["text"]) else "punct"
        print(
            f"{row['result']:>3}  {row['text']:<16.16} {row['phonemes'] or '':<18.18} "
            f"{_fmt(row['start']):>7} {_fmt(row['end']):>7}  "
            f"{_fmt(row['start_ts']):>8} {_fmt(row['end_ts']):>8}  {note}"
        )


def g2p_report(spoken: str, *, voice: str) -> int:
    """Tokenize without the model: shows what alignment will have to handle."""
    from kokoro import KPipeline

    from narrate.tts.kokoro_engine import lang_code_for_voice

    pipeline = KPipeline(
        lang_code=lang_code_for_voice(voice), repo_id="hexgrad/Kokoro-82M", model=False
    )
    for result in pipeline(spoken):
        print(f"\nresult {result.text_index}: {result.graphemes!r}")
        for token in result.tokens:
            timed = "yes" if token.phonemes else "NO (empty phonemes -> start_ts/end_ts None)"
            kind = "word " if is_word(token.text) else "punct"
            print(f"  {kind} {token.text!r:<16} {token.phonemes!r:<20} ws={token.whitespace!r:<4} timed={timed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Display text (normalized first)")
    parser.add_argument("--raw", action="store_true", help="Send --text to Kokoro un-normalized")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("out/spike"))
    parser.add_argument("--g2p-only", action="store_true", help="Tokenize only; needs no weights")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    from narrate.normalize import RuleNormalizer

    spoken = args.text if args.raw else RuleNormalizer().normalize(args.text)
    print(f"display: {args.text}\nspoken:  {spoken}")

    if args.g2p_only:
        return g2p_report(spoken, voice=args.voice)

    from narrate.tts import EngineUnavailableError

    try:
        results, vocab = synthesize(spoken, voice=args.voice, speed=args.speed)
    except EngineUnavailableError as exc:
        # Most often the weights could not be fetched from huggingface.co.
        print(f"Kokoro unavailable: {exc}\nTry --g2p-only, which needs no weights.", file=sys.stderr)
        return 2
    rows, audio, checks = flatten(results, vocab)
    if audio.size == 0:
        print("Kokoro returned no audio", file=sys.stderr)
        return 1

    print()
    print_tokens(rows)
    print("\nper-result checks (samples should equal pred_dur * 600):")
    for check in checks:
        print(f"  {check}")

    stats = measure(rows, audio)
    print("\nstats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    page = write_outputs(
        args.out,
        display=args.text,
        spoken=spoken,
        voice=args.voice,
        speed=args.speed,
        rows=rows,
        audio=audio,
        checks=checks,
        stats=stats,
    )
    print(f"\nwrote {args.out / 'spike.wav'}, {args.out / 'spike.json'}, {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
