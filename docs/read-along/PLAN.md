# Read-along mode — plan

`narrate read notes.md` → `out/notes/{notes.m4a, notes.timings.json, index.html}`:
audio plus a self-contained page that highlights the word being spoken.

This file records what the code actually looks like, where it differs from the
task brief, the design, and a log entry per phase.

## Status

| Phase | State |
|---|---|
| 0 — read the code, baseline, this plan | done |
| 1 — Kokoro timestamp spike | done; real-audio accuracy numbers still pending (needs weights) |
| 2 — engine timings + cache sidecars | done |
| 3 — timeline + player for Markdown/text | **done** (verified with stub audio in headless Chromium; not yet with real Kokoro) |
| 4 — normalizer receipts | next |
| 5 — polish, all inputs, README section | not started |
| 6 — live mode | design only, needs a go-ahead |

**`narrate read notes.md`** now writes `out/notes/notes.m4a`,
`notes.timings.json` and `index.html`. It supports `.md`, `.markdown`, `.txt`
and `.text` files. What exists:

- **Command** (`cli.py`): `narrate read` with `--voice`, `--speed`, `--format
  {m4a,mp3}`, `--embed-audio`, `--open`, `--output-dir`, `--work-dir`,
  `--keep-citations`, `--lexicon`, `--no-loudnorm`, `--no-cache`, `--bitrate`
  and `--engine`.
- **Renderer** (`readalong/render.py`): planned with `article`'s planner,
  synthesized through the timed cache path, and assembled with the same
  helpers as `article`. It encodes at 24 kHz, checks the encoded file, then
  writes the timings and the page.
- **Timeline** (`readalong/timeline.py`): absolute token times from integer
  sample counts.
- **Encode checks** (`readalong/encode.py`): decodes the file back to compare
  its length and rate, and cross-correlates the start to measure any offset.
- **Display** (`readalong/display.py`): blocks (heading, paragraph, list
  item, quote) and the Markdown-syntax spans the page hides.
- **Alignment** (`readalong/align.py`): tokens located in the spoken text,
  sentence ids, and the *temporary* spoken → display word alignment.
- **`timings.json`** (`readalong/build.py`, `readalong/schema.py`): schema
  v1 plus a hand-written validator.
- **Player** (`readalong/player.html`, filled in by `readalong/page.py`):
  - word highlight, sentence underline and auto-scroll;
  - click-to-seek, Space, ←/→ for sentences and Shift+←/→ for paragraphs;
  - a progress bar and a light/dark theme.
- **Engine timings and cache sidecars** from Phase 2 (`tts/`, `pipeline.py`).
- **Shared helpers** factored out of existing code, with output
  byte-identical:
  - `audio.silence_samples`;
  - `article.join_sections`, `section_starts`, `article_cache`,
    `article_tags` and `build_normalizer`;
  - `segment.paragraph_sentences`.
- **Tests**:
  - `tests/test_timings.py`: 47 tests, plus 3 real-Kokoro tests that skip
    without weights;
  - `tests/test_readalong.py`: 42 tests, including one that plays a page in
    headless Chromium (`tests/browser/readalong_check.js`) and skips without
    Node + Playwright.
- **Spike**: `scripts/spike_kokoro_timestamps.py` and
  `scripts/spike_player.html`.

**Not yet:**
- **Exact highlighting for normalizer rewrites.** These wait for Phase 4
  receipts. Until then, a run of consecutive rewrites shares one coarse
  highlight, e.g. `1939-1945, e.g.`.
- **Player extras** (Phase 5): step modes, speed control, resume, sentence
  ticks on the progress bar, and a chapter sidebar.
- **Other formats:** PDF/EPUB/DOCX/HTML/PPTX input (Phase 5).

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

**Cache (`tts/cache.py`), as built in Phase 2.** A `<key>.timings.json`
sidecar sits next to `<key>.wav`; the key does not change.
- *Format:* `{"version": 1, "words": [[text, start, end], ...]}`. `"words":
  null` means the engine cannot report timings.
- *Write:* `put_result()` writes the audio first, then the sidecar, so an
  interrupted run never leaves timings without audio. The plain `put()`
  deletes any sidecar for its key, because that sidecar described a
  different render.
- *Read:* `get()` is unchanged and ignores sidecars. `get_with_timings()`
  treats audio without a sidecar as a miss, so the chunk is re-synthesized
  rather than estimated; it counts these in `cache.stale` and logs once per
  cache. It also discards sidecars that are unreadable, from another format
  version, or that outlast their audio by more than 50 ms.
- *Deviation from the earlier draft:* `speak` and `article` now *do* write the
  sidecar, as the brief allows. Their output audio is unchanged. In exchange,
  a later `narrate read` reuses every chunk they rendered.

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

**Segmentation, as built in Phase 3.** `segment_chapter()` is unchanged in
output. The paragraph → sentence split it packs from is now a public function,
`paragraph_sentences()`, and `segment_chapter` calls it.
`align.sentence_ids()` walks a section's chunks and its sentences together,
matching non-space characters, so every chunk character gets a sentence id.
Clause-split parts of a long sentence therefore share one id, and the
underline spans the whole sentence. This replaces the `_segment()` /
`Chunk.spoken_map` idea from the first draft, which isn't needed until Phase
4.

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

**Timeline (`readalong/timeline.py`), as built in Phase 3.**
- *Counting.* Times are counted in integer samples: each chunk's audio, its
  trailing pause, and the 1200 ms section pause. The arithmetic is shared with
  assembly: `silence_samples()` for every pause, and
  `article.section_starts()` next to `join_sections()`.
- *Consistency.* The renderer raises if the timeline length differs from the
  assembled audio by even one sample. Seconds are computed once; nothing is
  measured from the encoded file.
- *Encoding.* `write_chapter(..., output_sample_rate=24000)` adds `-ar 24000`
  only for read-along.
- *Encoded check* (`readalong/encode.py`). Measured with ffmpeg here:
  - the sample rate stays 24 kHz with loudnorm;
  - mp3 decodes to exactly the PCM length, and m4a is +148 samples of end
    padding (under one 1024-sample frame);
  - cross-correlating the first 5 s gives an offset of 0 ms for both formats
    (correlation 0.995 or higher).

  An offset over 20 ms would be written to `audio_offset_s`.

**Display ↔ spoken alignment, Phase 3 stand-in.** The brief suggested naive
sequential matching. A greedy windowed matcher derailed in testing: "for"
from "e.g." latched onto a later "for", and every word after it shifted. So
Phase 3 instead aligns the two word sequences with the standard library's
`difflib.SequenceMatcher`.
- Matches stay in order, and a far match is never preferred, because it
  would cost every match in between.
- Unmatched spoken words share the display text between their matched
  neighbours: `$4.2M`, `Dr.`, `FBI`, `21st`.
- Only *runs* of consecutive rewrites (`1939-1945, e.g.`) end up coarse.

Phase 4 replaces this with exact receipts.

**`timings.json` additions to the brief's schema** (all validated):
- `timing_source` (`engine` / `estimated` / `mixed`);
- `markup` (spans the page hides);
- `number` on ordered list items.

The file is written one record per line, so a re-render after an edit gives a
readable diff.

**Player, as built in Phase 3.** A single template
(`readalong/player.html`); `page.py` inlines the JSON with every `<` escaped,
so document text cannot close the script.
- *Rendering.* Blocks become `h1`–`h6`, `p`, `ul`/`ol > li` and
  `blockquote`. Adjacent words with the same span form one `.w` unit, and
  each sentence is a `.s` span.
- *Sync.* A requestAnimationFrame loop plus a binary search; it touches the
  DOM only when the unit or sentence changes.
- *Highlight rule.* A word stays lit until the next word starts. It clears
  in gaps over 1 s (section pauses) and after the last word.
- *Heading pause.* The brief's 1 s rule alone would keep a heading's last
  word lit through its 800 ms pause. The player also clears the highlight
  when a heading's last word ends and the next word is in another block, as
  §7 asks.
- *Auto-scroll* keeps the current sentence near the upper third, and pauses
  for 4 s after a wheel, touch, scrollbar or scroll-key action.
- *Missing audio* shows a banner naming the file.

**CLI and reuse, as built.** `narrate read` takes `article`'s planner,
cache directory (`work/<file-stem>/cache`), normalizer and tags, so `article`
and `read` on the same file share every chunk. The output folder is
`out/<file-stem>/`. `--format mp3` prints a sync warning; `wav` is refused.

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

### Phase 2 — engine timings and cache sidecars

**What changed.**

- **`tts/base.py`.** Adds `WordTiming(text, start, end)` and
  `SynthesisResult(audio, sample_rate, words)` (`eq=False`, because arrays
  have no single truth value), plus a `duration` property. Adds the
  `TimedSynthesisEngine` protocol and the `synthesize_with_timings()` helper.
  `synthesize()` and the `SynthesisEngine` protocol are untouched.
- **`tts/kokoro_engine.py`.**
  - `synthesize()` and the new `synthesize_with_timings()` share `_run()`,
    which makes the same single `KPipeline(text, voice=, speed=)` call as
    before.
  - `_collect()` is unchanged in behaviour; `_samples()` was split out of it
    so the timing code counts samples exactly the way the audio is joined.
  - `token_timings()` offsets each internal result by the samples before it.
  - `times_from_durations()` is the corrected duration walk from Phase 1.
  - `_fill_gaps()` gives every token a time. A missing start takes the
    previous end; a missing end takes the next start, or the end of the audio.
    Starts never go backwards and no token outlasts its audio.
  - If the tokens can't be read, the result keeps its audio, `words=None` is
    returned, and a warning is logged.
  - Punctuation tokens are kept. Phase 3/4 alignment filters them out as
    words.
- **`tts/timing_estimate.py`.** `estimate_word_timings(text, duration)` gives
  each whitespace-separated word a share proportional to its length, with no
  gaps. `with_estimated_timings(result, text)` fills in `words` only when
  they are `None`.
- **`tts/cache.py`.** Sidecars, as described in the Design section. Adds
  `timings_path_for`, `get_with_timings` and `put_result`; `cache.stale`
  counts entries that had audio but no sidecar.
- **`pipeline.py`.** `synthesize_chunks()` renders misses through
  `synthesize_with_timings()` and `put_result()`; hits still use `get()`. Adds
  `synthesize_chunk_with_timings()` for read-along.
- **`pyproject.toml`.** Registers the `kokoro` pytest marker.
- **Tests.** `tests/stub_engine.py` has `TimedStubEngine` (each token lasts
  100 ms × length / 5 ÷ speed, laid end to end, so the expected times are
  plain cumulative sums) and `UntimedStubEngine`. `tests/test_timings.py`
  covers:
  - the contract and speed scaling;
  - the estimator;
  - the duration walk against hand-worked values, and against Kokoro's own
    walk on clean input when `kokoro` is installed;
  - the junk-token, vocab-filter and truncation cases;
  - gap filling and multi-result offsets;
  - audio staying identical between the two Kokoro methods, using a fake
    pipeline;
  - every cache sidecar case;
  - unchanged cache keys (golden hashes);
  - the pipeline paths, including `speak` → timed path reusing every chunk.

  The real-Kokoro tests (`-m kokoro`, both voices, speed 0.9) run offline and
  skip when the weights aren't cached.

**Verified.**

- Full suite: 245 passed, 3 skipped (real Kokoro). The 6 failures and 1 error
  are the same Hugging Face-blocked PDF ingest tests as the baseline.
- The new file runs in about 8 s warm, most of it the one-time `kokoro`
  import. The first run after a package rebuild is slower.
- **Regression check:** the same note was rendered with the pre-Phase-2 code
  (stashed) and the new code, as `article` in wav, mp3 and m4a with and
  without loudnorm, plus one `speak` mp3. All 7 output files have identical
  MD5s. The only difference on disk is one small `.timings.json` per cached
  chunk.

**What's next (Phase 3).**

- `narrate read` for `.md`/`.txt`, reusing `article`'s plan → render step.
- `readalong/timeline.py`: sample-exact times across chunk and section
  silences.
- 24 kHz encoding for read-along, plus the encoded-length and encoder-offset
  checks.
- `timings.json` v1 and `readalong/player.html` with sync, click-to-seek,
  auto-scroll and basic controls.
- Naive sequential word → display matching until Phase 4.

**Open issues.**

1. Real-audio accuracy is still unmeasured. If the spike shows a constant
   onset bias, it becomes one constant in `times_from_durations`. If
   `samples != pred_dur · 600`, the times will need scaling to the real audio
   length. For now `_fill_gaps` only clamps them to it.
2. An entry first cached by an older version is re-rendered once by the timed
   path, and the new render replaces the cached WAV. Kokoro's decoder draws
   random phase and noise (`torch.rand`/`randn_like` in `istftnet.py`), so
   that render is not bit-identical to the old one. The difference is
   inaudible, and from then on `speak`, `article` and `read` share the
   entry.
3. The 96/48 kHz `loudnorm` question from Phase 1 is still open.

### Phase 3 — `narrate read` for Markdown and text

**What changed.**

- **New package `src/narrate/readalong/`.**
  - `render.py` orchestrates: plan → timed synthesis → assembly → timeline →
    encode at 24 kHz → encoded check → `timings.json` → page.
  - `timeline.py`, `encode.py`, `display.py`, `align.py`, `build.py`,
    `schema.py` and `page.py` are described under Design.
  - `player.html` is the page template, shipped in the wheel.
- **`cli.py`.** New `read` command. It refuses `wav`, warns for `mp3`,
  explains empty documents, and prints any encode/timing warnings.
- **`article.py`.** `render_article` now uses the extracted `join_sections`,
  `article_cache`, `article_tags` and `build_normalizer`. `section_starts`
  is new.
- **`audio.py`.** Adds `silence_samples()`, now used by
  `concat_with_silence`. `write_chapter`/`_encode` gain the opt-in
  `output_sample_rate`.
- **`segment.py`.** `paragraph_sentences()` is extracted;
  `segment_chapter` calls it.
- **Tests.**
  - `tests/stub_engine.py`: `TimedStubEngine` now emits tone bursts instead
    of flat levels, so codecs and the offset correlation see real signal.
    Its timings are unchanged.
  - `tests/test_readalong.py` (42 tests):
    - *Timeline:* hand-computed cumulative sums across all four pause types
      and an empty section; strict ordering; speed 0.9.
    - *Alignment:* token location, punctuation, misses, clause-split
      sentences; display alignment for `$4.2M` / `Dr.` / `FBI` and the
      derailment regression; markup hiding in every block (regression);
      block parsing.
    - *timings.json:* schema validity, spans inside the text, sentences with
      words, display text = speakable Markdown, citations/URLs/lexicon drops
      never highlighted, heading words and their pause, long sentence = one
      sentence, validator negatives, writer round-trip.
    - *Encoding:* m4a/mp3 × loudnorm keep length, rate and start; the old
      96 kHz behaviour is untouched without the opt-in; offset detection
      finds a known 30 ms delay and declines on silence.
    - *Page:* inline JSON equals the file, nothing external, script
      break-out escaped, `--embed-audio`.
    - *Cache:* `article` → `read` shares every chunk; an edit re-renders one
      chunk and updates the timings exactly; an untimed engine falls back to
      estimates.
    - *CLI:* files, `--open`, the mp3 warning, wav refusal, empty document,
      PDF refusal.
    - *Browser:* `tests/browser/readalong_check.js` plays a page in headless
      Chromium at phone width. It checks that the lit unit tracks the clock
      at every sample, click-to-seek, →, Shift+→, auto-scroll into view,
      auto-scroll suspension after a manual scroll, no horizontal overflow,
      and no console errors.

**Verified.**

- Full suite: 287 passed, 3 skipped (real Kokoro). The only failures are the
  same 7 Hugging Face-blocked PDF ingest tests as the baseline.
- The browser test passed in 3 of 3 repeat runs (about 9 s each).
- **Regression:** the same 7-file `article`/`speak` render (wav/mp3/m4a with
  and without loudnorm, plus a `speak` mp3) has identical MD5s before and
  after Phase 3. They also match the pre-Phase-2 code.
- Screenshots of the stub page in light (820 px) and dark (390 px) themes
  checked by eye: highlight, sentence underline, hidden Markdown syntax,
  visible citation.

**Not verified here.** There has been no end-to-end run with real Kokoro
audio, because the weights can't be downloaded in this environment. The brief's
"real Markdown note plays with visibly correct sync" has so far been checked
with stub audio, where the timings are exact by construction. With Kokoro, sync
also depends on the timestamp accuracy that the Phase 1 spike measures.

**What's next (Phase 4).**

- `MappedText` and mapped twins of every normalizer pass, with a string
  equality test against today's normalizer.
- Per-chunk spoken → display maps.
- Replace `naive_display_spans` with receipt-based spans, so a run like
  `1939-1945, e.g.` splits into its exact parts.

**Open issues.**

1. Real-audio accuracy and an end-to-end Kokoro run (see Phase 1).
2. mp3 sync in browsers that ignore the LAME delay header is not measurable
   here. The CLI warns, and m4a is the default.
3. A `--format` switch leaves the other format's file in the folder; only the
   one named in `timings.json` is used.
4. The 96/48 kHz question for `speak`/`article` is still open.
