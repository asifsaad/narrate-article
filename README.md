# narrate

Turn PDFs, EPUBs and text into narrated audio, entirely locally, using
open-weights models.

Tables, diagrams, formulas, page headers and footnotes are dropped; the prose
around them is read aloud.

## Install

```bash
uv sync --extra kokoro     # TTS engine (pulls torch, ~2 GB)
brew install ffmpeg        # encoding and loudness normalization
```

`espeak-ng` is optional — Kokoro's G2P falls back to it for unusual words:

```bash
brew install espeak-ng
```

## Use

```bash
narrate speak book.epub                      # -> out/<book>/001 - Chapter.mp3
narrate speak paper.pdf --voice bf_emma
narrate voices                               # list available voices
```

The two-stage flow, which is the point of the design:

```bash
narrate extract book.pdf                     # -> work/<book>/chapters/*.md
$EDITOR work/<book>/chapters/003*.md         # fix whatever the parser got wrong
narrate speak work/<book> --from-markdown    # only edited chunks re-synthesize
```

Useful flags:

| Flag | Effect |
|---|---|
| `--chapters 1-3,7` | Render a subset — use this to audition settings |
| `--speed 0.9` | Kokoro reads at ~200 wpm; 0.9 is closer to audiobook pace |
| `--format wav` | Skip mp3 encoding (also `m4a`, `opus`) |
| `--captions` / `--footnotes` | Include figure captions / footnotes |
| `--keep-citations` | Keep `[12]` and `(Smith, 2020)` markers |
| `--lexicon config/lexicon.yaml` | Pronunciation overrides |
| `--ocr` | For scanned PDFs (otherwise they fail loudly) |
| `--no-cache` | Re-synthesize everything |

### Short documents

A note, blog post or essay wants one file, not a folder of chapters:

```bash
narrate article notes.md                     # -> out/notes.mp3
narrate article notes.md --dry-run           # estimate + exact script, no synthesis
narrate article post.md -o post.m4a --speed 0.9
```

At ~200 wpm, 10-20 minutes of audio is roughly 2,000-4,000 words. The dry run
prints the estimate per section, so you can trim before rendering; past 30
minutes the command suggests `speak --format m4b` instead.

There is no checkpoint - the markdown file *is* the editable text, and
re-running after an edit re-synthesizes only the sentences that changed.
Sections are separated by a longer pause than paragraphs.

Markdown furniture is not read aloud: front matter (its `title` and `author`
become the tags), HTML comments and tags, code blocks, tables, math, footnotes,
link definitions and task-list boxes. Callouts are announced ("Note."), and list
items are read as separate items rather than one run-on phrase.

For one chapter of a book, extract first and point `article` at the file:
`narrate article work/<book>/chapters/003*.md`.

## How it works

```
ingest -> chapterize -> normalize -> segment -> synthesize -> assemble
```

1. **Ingest** (`ingest/`) — Docling parses the document into a labelled tree.
   Exclusion happens by semantic label (`TABLE`, `PICTURE`, `FORMULA`, …), not
   by pattern-matching rendered markdown, so a table laid out with spaces is
   still recognised as a table. See `ingest/labels.py`.
2. **Chapterize** — split on headings, merging undersized sections so a PDF
   with noisy heading detection does not explode into hundreds of files.
3. **Normalize** (`normalize/`) — expand numbers, currency, dates, ordinals,
   roman numerals, abbreviations and acronyms; strip citations and URLs. A
   digit reaching the engine is a bug, and a test enforces that.
4. **Segment** (`segment.py`) — sentence-aware chunking under the engine's
   budget. Never splits mid-sentence; assigns inter-sentence and
   inter-paragraph pauses while paragraph structure is still visible.
5. **Synthesize** (`tts/`) — Kokoro-82M behind a `SynthesisEngine` protocol.
   Every chunk is cached by content hash, as `<hash>.wav` plus a small
   `<hash>.timings.json` holding when each word is spoken (see
   [Read-along](#read-along-in-progress)). The timings ride along on the same
   model call, so the audio is unchanged by them.
6. **Assemble** (`audio.py`) — concatenate with generated silence, apply EBU
   R128 loudness normalization (-19 LUFS, -3 dBTP), encode and tag.

## Read-along (in progress)

Listen to a note while the page highlights each word as it is spoken:

```bash
narrate read notes.md --open                 # -> out/notes/index.html
narrate read notes.md --voice bf_emma --speed 0.9
```

`out/notes/` holds three files:

| File | Contents |
|---|---|
| `notes.m4a` | The narration, encoded at the engine's 24 kHz |
| `notes.timings.json` | The text, its blocks and sentences, and when each word is spoken |
| `index.html` | A self-contained player: no network, works opened straight from disk |

On the page, the current word is highlighted and its sentence underlined, and
the page scrolls to keep up. It stops following for a few seconds if you
scroll yourself.

| Control | Effect |
|---|---|
| Click a word | Jump there |
| Space | Play / pause |
| ← / → | Previous / next sentence |
| Shift+← / Shift+→ | Previous / next paragraph |

The page follows your system's light or dark theme and works at phone width.

Useful flags:

| Flag | Effect |
|---|---|
| `--format mp3` | mp3 instead of m4a. Some browsers ignore mp3's encoder delay, so the highlight may run slightly early |
| `--embed-audio` | Put the audio inside `index.html`, giving one file to share (slow to open above ~20 MB) |
| `--open` | Open the page when done |
| `--keep-citations`, `--lexicon`, `--no-loudnorm`, `--no-cache` | As for `article` |

`read` and `article` share their chunk cache. Narrating a file one way and
then the other synthesizes nothing new, and after an edit only the changed
sentences are re-rendered.

How the timings work:
- **Where they come from.** Kokoro reports when each token is spoken, from its
  duration predictor, on the same model call that produces the audio. Engines
  that cannot report timings get an estimate (`tts/timing_estimate.py`).
- **Placing words.** Words are placed by counting samples through every chunk
  and pause, never by measuring the encoded file.
- **Checking the encode.** After encoding, the file is decoded back and
  checked for its length, its sample rate and any start delay.

Current limits:
- **Inputs:** Markdown and text files only for now. PDF, EPUB, DOCX, HTML and
  PPTX come later.
- **Rewritten text:** a word the normalizer rewrote (`$4.2M` → "four point
  two million dollars", `Dr.`, `FBI`) shares one highlight with its spoken
  words. Several rewrites in a row (`1939-1945, e.g.`) currently share one
  coarse highlight; exact mapping is the next step.
- **Player features still to come:** step-by-sentence/paragraph modes, a
  speed control, "resume where you left off", and a chapter sidebar.

The plan, the findings and a log of each phase are in
[`docs/read-along/PLAN.md`](docs/read-along/PLAN.md). To see Kokoro's raw
timestamps for a paragraph:

```bash
uv run --extra kokoro python scripts/spike_kokoro_timestamps.py
```

## Model choices

**TTS: Kokoro-82M (Apache-2.0).** Most of the models ranked above it on
blind-preference leaderboards are unusable for this: Fish Audio S2 Pro and
Higgs Audio V3 are non-commercial, Voxtral TTS is CC-BY-NC, and Step Audio
EditX and NVIDIA Magpie impose 30- and 20-second clip caps that make long-form
narration impossible. Kokoro is within noise of that tier, runs at ~19×
realtime on an M3 Pro, and is genuinely permissive.

`tts/base.py` is the seam for adding **Qwen3-TTS 1.7B** (Apache-2.0, better
long-form prosody, 3-second voice cloning) or **Chatterbox** (MIT) later.

**Ingest: Docling (MIT).** Marker v2 scores higher on olmOCR-bench (76.0 vs
50.3), but that margin is concentrated in tables, formulas and forms —
precisely the content this tool discards. Docling covers PDF, EPUB, DOCX and
HTML through one API and exposes the labelled tree the exclusion depends on.
Marker is GPL-3.0 with RAIL-M weights and a revenue threshold.

**Normalization: rule-based.** `nemo-text-processing` is the obvious choice and
does not install cleanly on macOS — its `pynini`/OpenFst dependency needs
conda-forge. The `Normalizer` protocol leaves room for it on Linux.

## Performance

Measured on an M3 Pro (36 GB), Kokoro on MPS:

| Stage | Share of runtime | Throughput |
|---|---|---|
| Synthesis (GPU) | ~78% | 18.4× realtime |
| ffmpeg encode + loudnorm (CPU) | ~21% | 63× realtime |
| Normalize / segment | ~2% | — |

Because synthesis is GPU-bound and encoding is CPU-bound, chapter *N* is encoded
while chapter *N+1* synthesizes. Measured on three chapters of *Alice*:

```
serial      161.6s   11.6x realtime
pipelined   110.9s   16.9x realtime     <- 1.46x faster
```

That is 92% of the pure-synthesis ceiling, so little remains on the table.
Disable with `--no-parallel-encode` when debugging ffmpeg.

**Parallelising chapters does not help.** One synthesis stream already saturates
the GPU; running several concurrently measured *slower* (0.84× at two threads,
0.81× at four). The useful parallelism is across *resource types*, not chapters.

**A GPU is not the bottleneck you might expect.** Kokoro-82M is too small to
benefit much, and it has no batched forward pass — `KModel.forward()` takes a
single phoneme string. A datacentre GPU would speed up the 78%, but the CPU-bound
21% is unchanged, capping realistic end-to-end gains near 1.6×. Cloud GPUs are
worth it for *larger models* (Qwen3-TTS 1.7B, Chatterbox), not for Kokoro.

## Known limits

- **Scanned PDFs** fail with a clear error unless `--ocr` is passed. OCR
  quality has not been tuned.
- **Kokoro reads fast** (~200 wpm against a typical 155 wpm for audiobooks).
  Use `--speed 0.9`.
- **PDF heading detection** is imperfect — the weak spot of every parser. The
  markdown checkpoint exists so you can fix it by hand.
- **No ASR verification.** Nothing automatically checks that the audio matches
  the text. The hook point is between synthesis and assembly in `pipeline.py`.
- **Voice cloning is not implemented.** The engine interface is ready for it.

## Tests

```bash
uv run --group dev pytest tests/ -q
```

`tests/fixtures/make_fixtures.py` generates a PDF and an EPUB that each embed
three sentinel strings — one in prose, one only inside table cells, one only in
a figure caption. The suite asserts the first survives and the other two do
not.

Synthesis is faked with deterministic stub engines (`tests/stub_engine.py`),
so the suite needs no model weights. One read-along test plays a generated page
in headless Chromium through Node.js and Playwright, and skips if they are not
installed. A few tests marked `kokoro` render real audio; they run offline and
skip unless Kokoro-82M is already in the local Hugging Face cache:

```bash
uv run --group dev --extra kokoro pytest tests/ -m kokoro -v
```

The PDF tests need Docling's layout models from Hugging Face, so they fail on
a machine that cannot reach it.
