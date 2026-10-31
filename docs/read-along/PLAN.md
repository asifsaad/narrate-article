# Read-along mode — plan

`narrate read notes.md` → `out/notes/{notes.m4a, notes.timings.json, index.html}`:
audio plus a self-contained page that highlights the word being spoken.

This file records what the code actually looks like, where it differs from the
task brief, the design, and a log entry per phase.

## What I found (verified against the code, not the brief)

**Repository and branch.** The working repo is `asifsaad/narrate-article`, not
`text-to-audio-v1`. It has no PR #1 and no `claude/practical-carson-nj21lt`
branch; `main` (`6b92022`) already contains `article.py` and the Markdown
cleanup (`ingest/markdown.py`). This session must push to
`claude/funny-mccarthy-elkxf9`, which is identical to `main`, so that branch
stands in for `feature/read-along`. Nothing is merged into `main`.

**Baseline.** `uv run --group dev pytest tests/ -q`: 198 passed, 6 failed and 1
error. All 7 are PDF ingest tests (`tests/test_ingest.py`) that need Docling's
layout models from `huggingface.co`, which this environment's network policy
blocks. They are not regressions, and they pass wherever Hugging Face is
reachable.

**Kokoro already reports timings on the path we use.** `KokoroEngine.synthesize`
calls `KPipeline.__call__(text, voice=, speed=)`. In kokoro 0.9.4 that path
calls `KModel.forward(..., return_output=True)`, then `join_timestamps(tokens,
pred_dur)`, then yields `Result(graphemes, phonemes, tokens, output)`.
`_collect()` keeps `.audio` and throws the tokens away. So **the engine path does
not need to change**: reading `result.tokens` alongside `result.audio` leaves
the audio byte-identical.

- Timings come from the duration predictor (`pred_dur`, in frames): 1 frame =
  600 samples = 25 ms at 24 kHz, and timestamps are quantised to half-frames
  (12.5 ms). They are the model's own alignment, not an ASR estimate.
- The start offset is a heuristic: `2 * (pred_dur[0] - 3)` half-frames, with an
  upstream `TODO: Is -3 an appropriate offset?`. The spike measures this.
- **Several results per chunk.** `__call__` splits its input on `\n+` (its
  `split_pattern`), and `en_tokenize` splits again at 510 phonemes. Each
  `Result` has its own audio, with times starting at 0, so a result's times
  must be offset by the audio length of the results before it.
- **`None` times.** `join_timestamps` skips tokens with empty phonemes:
  out-of-vocabulary words when the espeak fallback is missing (`unk=''`),
  currency symbols, and junk characters. It also stops early if `pred_dur` is
  shorter than the phoneme string. That happens when `KModel.forward` filters
  phoneme characters missing from the vocab, while `join_timestamps` still
  assumes one input id per phoneme character. Every token after that point
  gets `None`.
- **Tokens, from a real G2P run** (misaki + spaCy, no model weights; see
  Phase 1). `.text` is an exact, in-order substring of the input. Punctuation
  marks (`, . ; — “ ”`) are separate tokens whose phonemes are the character
  itself, so they *do* get (short) times. `F. B. I.` becomes three tokens,
  `F.` `B.` `I.`. Hyphenated number words stay single tokens (`thirty-nine`,
  `twenty-first`), and so do possessives (`Four's`). British G2P (`lang_code
  'b'`) produces the same token structure.

**Normalizer pass order (from the code).** markdown → citations/URLs/emails →
currency → numbers → abbreviations → **all-caps-run title-casing** → acronyms →
lexicon (drops, *then* pronunciations) → tidy. This differs from the brief in two
ways: the brief leaves out the caps-run pass, and lexicon drops run last
together with pronunciations. Note also that the `rules.py` module docstring
says drops run second, which is out of date. The port follows the code.

**Loudness normalisation changes the sample rate.** ffmpeg's `loudnorm`
resamples to 192 kHz internally, and `_encode` sets no `-ar`. Today's outputs are
therefore **96 kHz for m4a and 48 kHz for mp3**. Only `--no-loudnorm` output stays at
24 kHz. Read-along needs 24 kHz, so it
passes `-ar 24000` through a new opt-in keyword. Existing commands keep their
current behaviour, because the brief requires byte-identical output. The same
fix is probably worth making for `speak`/`article` (smaller files, less CPU),
but as a separate change.

**Encoder delay.** ffprobe reports `start_time` 0.000 for m4a (ffmpeg writes an
edit list) and 0.023–0.046 s for mp3 (LAME priming). This supports making m4a
the read-along default.

**Smaller things.**
- `Chunk` is a plain dataclass, and `cache_key` hashes only engine, voice,
  speed and text. An optional mapping field leaves cache keys unchanged.
- The segmenter packs several sentences into one chunk and does not keep the
  boundaries. Long sentences are split at clauses (`_split_oversized`).
- Silences: 250 / 600 / 800 ms are set per chunk (`segment.py`), and 1200 ms
  between sections (`article.py`). Silence length is
  `int(sample_rate * ms / 1000)` in `concat_with_silence`. m4b chapter marks
  count samples and round to ms per chapter.
- `segment._paragraph_gap` detects headings by shape (≤ 8 words, ≤ 1 period).
  Read-along takes block kinds from the display Markdown instead.
- Both test files carry their own copy of the same `StubEngine`.
- **Kokoro weights are unavailable here.** `KModel` and the voice packs download
  from `huggingface.co`, which is blocked, so no real audio can be rendered in
  this environment. G2P works (spaCy's `en_core_web_sm` installed from GitHub).

## Design

No new runtime dependencies. Playwright (pre-installed in some environments) is
used only by one optional test, through `pytest.importorskip`.

**Engine (`tts/base.py`, `tts/kokoro_engine.py`).** Add the frozen dataclasses
`WordTiming(text, start, end)` and `SynthesisResult(audio, sample_rate, words)`.
`synthesize()` stays exactly as it is. Engines may add
`synthesize_with_timings()`. A module-level helper,
`synthesize_with_timings(engine, text, ...)`, calls it when it exists and
otherwise wraps `synthesize()` with `words=None`. For Kokoro, both methods share
a private `_run()` that returns the results. `synthesize` is still
`_collect(results)`. The timing method also walks `result.tokens`, offsets
each result by the samples before it, and fills in `None` times: a missing
start takes the previous end; a missing end takes the next start, or the chunk
duration if there is none. Punctuation tokens are kept, because they help
alignment, and filtered out when words are built.
`tts/timing_estimate.py` spreads the duration across words by character
length, for engines without timings.

**Cache (`tts/cache.py`).** Add a `<key>.timings.json` sidecar next to
`<key>.wav`; the key does not change. `get_with_timings()` treats a WAV without
a sidecar as a miss, so the chunk is re-synthesized rather than estimated, and
logs this once per cache. `put(..., words=None)` writes the sidecar only when
timings exist, so `speak`/`article` write exactly what they write today.

**Receipts (`normalize/mapped.py`).** `MappedText(text, starts, ends)` stores the
spans in two `array('l')` columns instead of a list of tuples, which saves
memory on books. `mapped_sub(mt, pattern, repl)` works like `re.sub`:
- Literal replacement characters, and the result of a callable, map to the
  whole match's span.
- Characters copied in through `\1`-style group references keep their own
  spans, so `[link text](url)` still highlights word by word.
- A callable may return a `MappedText` when it needs finer control.

Each pass gets a mapped twin. `RuleNormalizer.normalize()` stays on the current
string path, which carries zero risk for existing commands. The new
`normalize_mapped()` is checked against it by an equality test over every
existing test input and fixture. If they ever diverge, the test fails, which
also guarantees that read-along and `article` share cache entries.

**Segmentation.** An internal `_segment()` returns, for each chunk, the
sentences it was packed from. `segment_chapter()` returns exactly the chunks it
returns today. A mapped wrapper aligns each chunk's text back into the
`MappedText` (whitespace-tolerant cursor) and sets the new optional field
`Chunk.spoken_map`, plus spoken sentence spans. Clause-split parts keep their
parent sentence id, so the underline spans the whole sentence.

**Alignment (`readalong/align.py`).** For each token, search case-insensitively
from a moving cursor in the chunk's spoken text, tolerating quote and whitespace
variants. If the search fails, skip to the next word boundary and count a
warning. The token's display span is the union of the spans of its spoken
characters. Tokens without letters or digits are dropped. Adjacent words with
the same `src` form one highlight unit, which covers both `$4.2M` and `F. B. I.`.

**Display text and blocks.** `display_text` is the chapters' speakable Markdown
joined with `"\n\n"`, which is what `extract` would show. Blocks (heading,
paragraph, list item, quote) come from scanning that Markdown line by line. One
addition to the brief's schema: a `markup` list of spans removed by the
Markdown pass (`#`, `**`, `- `, link URLs). The player hides these, so the page
reads as prose rather than raw Markdown. Removed citations and lexicon drops
are not markup: they stay visible but are never highlighted.

**Timeline (`readalong/timeline.py`).** Times are counted in integer samples:
each chunk's audio, its trailing silence, and the section silence, using the
same `silence_samples()` helper that `concat_with_silence` will be refactored to
call (identical arithmetic). Seconds are computed once at the end, never from
the encoded file. Encoding passes `-ar 24000`. Afterwards the file is decoded
back to PCM to check two things: the sample count must be within one codec
frame (1024 for AAC, 1152 for mp3), and cross-correlating the first 5 s against
the pre-encode PCM must give an offset of at most 20 ms. A larger offset is
written to `audio_offset_s`.

**Player (`readalong/player.html`).** One template with inline CSS and JS. The
timings are inlined as `<script type="application/json">`, with `</` escaped.
The audio is referenced by relative path, or embedded as base64 with
`--embed-audio`. Sync is a rAF loop with a binary search, and it touches the DOM
only when the highlighted unit changes. All other behaviour follows brief §4.7.

**CLI and reuse.** `narrate read`. `render_article` is split so `article` and
`read` share the plan → synthesize step. `read` additionally keeps per-chunk
results and sample counts. For `article`, the audio stays identical.

**Future work (out of scope).** Highlighting on rendered PDF pages, using
Docling provenance boxes plus PyMuPDF word boxes. EPUB 3 Media Overlays: SMIL
`<par>` elements generated from `sentences`/`words`.

**Phase 6 (live mode).** Design only, to be written up after Phase 5.

## Log

### Phase 0 — reading and baseline
Read the modules and tests listed in the brief, plus `ingest/__init__.py`,
`chapterize.py` and `docling_ingest.py`. Recorded the baseline above.

### Phase 1 — spike

**What changed.** `scripts/spike_kokoro_timestamps.py` renders one paragraph
through the exact `KPipeline(text, voice=, speed=)` call the engine makes. It
prints `text, start, end` per token, offsetting each result by the audio before
it, next to upstream's `start_ts`/`end_ts`. It writes `out/spike/spike.wav`,
`spike.json` and `index.html`. `scripts/spike_player.html` is the throwaway
page: inline JSON, rAF + binary-search highlighting, click-to-seek, and a
±300 ms nudge slider for judging the offset by ear. `--g2p-only` tokenizes
without weights.

```bash
uv run --extra kokoro python scripts/spike_kokoro_timestamps.py
uv run --extra kokoro python scripts/spike_kokoro_timestamps.py --voice bf_emma --speed 0.9
```

**Measured here.** The Kokoro weights could not be downloaded in this
environment, so the following come from the real G2P plus deterministic or
synthetic inputs. None of it is model output.

- *Engine path.* No change is needed. Tokens arrive on the call we already
  make, so the audio is untouched.
- *Punctuation.* Separate tokens (`, . ; — … ( ) “ ”`) with their own short
  times. An ASCII `'` is a token with phoneme `“`. Alignment keeps them as
  anchors; they never become words.
- *`None`.* Tokens with no phonemes (`~`, `★`, and OOD words when espeak is
  missing) get `None`.
- *Upstream bug.* `join_timestamps` mishandles such a token when it is
  followed by a space: it advances its duration index twice. Every later
  token then gets the next phoneme's duration, and the last ones run off the
  end as `None`. With synthetic durations and `"Hello ~ world and more."`,
  upstream gives `.` no time and shifts `and`/`more` by one phoneme.
- *Fix.* `rederive_times()` recomputes the times from `pred_dur` with the same
  half-frame convention: word boundaries sit mid-way through the space between
  words, and start = `2·(pred_dur[0]−3)` half-frames. It reproduces upstream
  exactly on clean input and fixes the junk case. It also respects the vocab
  filter in `KModel.forward`.
- *Multi-result chunks.* Confirmed for text containing `\n` (one result per
  line, each timed from 0). The spike offsets results by their real sample
  counts.
- *Tooling.* The energy-based checks (speech onset/offset, and pause edges
  against word boundaries) report 0 ms error on synthetic tone bursts with
  known boundaries. In headless Chromium the spike page highlights the right
  word at each sample time, stays lit through 120 ms gaps, clears after the
  last word, and seeks on click, with no console errors.

**Not yet measured: real-audio accuracy.** That needs one run with weights.
The run prints the following:
- `onset_error_ms` and `offset_error_ms`, which test the `−3` frame heuristic;
- `pause_vs_word_end_ms` and `pause_vs_next_start_ms`, which show whether the
  highlight runs early or late at audible pauses;
- `upstream_disagrees_ms`, i.e. how often the upstream bug fires on real text;
- per result, whether `samples == pred_dur · 600`.

What would change the plan: a consistent onset bias over ~50 ms, which would
become a constant correction in the engine, or `samples != pred_dur · 600`,
which would mean times must be scaled to the real audio length.

**Decision for Phase 2.** Keep the call. Read `result.tokens` and
`result.pred_dur`. Use the re-derived times, with upstream's only as a
cross-check, and fill the remaining `None` values as in the design.

**Open issues.**
1. Real-audio numbers are pending; see above.
2. Should `speak`/`article` also get `-ar 24000`? Today `loudnorm` outputs
   96/48 kHz.
