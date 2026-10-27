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
   Every chunk is cached by content hash.
6. **Assemble** (`audio.py`) — concatenate with generated silence, apply EBU
   R128 loudness normalization (-19 LUFS, -3 dBTP), encode and tag.

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
