# autobook

Convert EPUB ebooks into narrated audiobooks using neural text-to-speech.

Autobook parses your EPUB file chapter by chapter, cleans and preprocesses the text, and synthesizes high-quality speech using [Kokoro](https://github.com/hexgrad/kokoro) — an 82M-parameter neural TTS model that runs entirely on your machine. A cloud-based fallback using Microsoft's [edge-tts](https://github.com/rany2/edge-tts) is also available for systems without enough disk space for the local model.

Output is one WAV file per chapter, with an optional single combined M4B audiobook.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- ffmpeg *(optional — only needed for `--combine` M4B output)*

To install ffmpeg on macOS:

```bash
brew install ffmpeg
```

---

## Setup

```bash
git clone <repo-url>
cd autobook
uv sync
```

`uv sync` creates a virtual environment and installs all dependencies. On the first run, Kokoro will also download its ~300 MB voice model from Hugging Face (cached for subsequent runs).

---

## Usage

Autobook has two subcommands: `narrate` to synthesize speech, and `combine` to merge WAVs into an M4B.

### narrate

```bash
uv run autobook narrate book.epub
```

Output is written to `output/<book-title>/`, one WAV file per chapter.

#### Preview chapters without generating audio

```bash
uv run autobook narrate book.epub --list-chapters
```

```
Found 32 chapter(s):

    1. Preface                                                                 (1,203 words)
    2. Chapter I — In the Beginning                                            (4,871 words)
    3. Chapter II — The Road Goes Ever On                                      (5,102 words)
  ...
```

#### Choose a voice

```bash
# American English voices
uv run autobook narrate book.epub --voice af_bella          # female
uv run autobook narrate book.epub --voice am_michael        # male

# British English voices
uv run autobook narrate book.epub --voice bm_george --lang b
uv run autobook narrate book.epub --voice bf_emma   --lang b
```

**All Kokoro voices:**

| Code | Gender | Accent |
|---|---|---|
| `af_heart` *(default)* | Female | American |
| `af_bella` | Female | American |
| `af_nicole` | Female | American |
| `af_sarah` | Female | American |
| `af_sky` | Female | American |
| `am_adam` | Male | American |
| `am_michael` | Male | American |
| `bf_emma` | Female | British |
| `bf_isabella` | Female | British |
| `bm_george` | Male | British |
| `bm_lewis` | Male | British |

#### Adjust speed

```bash
uv run autobook narrate book.epub --speed 0.9    # slightly slower
uv run autobook narrate book.epub --speed 1.15   # slightly faster
```

#### Narrate specific chapters

```bash
uv run autobook narrate book.epub --chapters 1-5      # chapters 1 through 5
uv run autobook narrate book.epub --chapters 2,4,6    # specific chapters
uv run autobook narrate book.epub --chapters 3        # a single chapter
```

#### Narrate and combine in one step

```bash
uv run autobook narrate book.epub --combine
```

Requires ffmpeg. Produces `output/<book-title>/audiobook.m4b` alongside the individual chapter WAVs.

#### Custom output directory

```bash
uv run autobook narrate book.epub --output ~/Audiobooks/my-book/
```

---

### combine

Merge previously narrated chapter WAVs into a single M4B — useful if you forgot to install ffmpeg before running `narrate`, or want to re-combine a subset of chapters.

```bash
# Basic — writes audiobook.m4b inside the same directory
uv run autobook combine output/my-book/

# Custom output path
uv run autobook combine output/my-book/ --output ~/Desktop/my-book.m4b

# Override the title stored in M4B metadata
uv run autobook combine output/my-book/ --title "My Book"
```

If ffmpeg isn't installed, `combine` exits immediately with clear installation instructions rather than silently failing.

---

## Hardware acceleration

Autobook automatically uses the best available hardware. The detection order is:

1. **CUDA** — NVIDIA GPU (Linux / Windows)
2. **MPS** — Apple Silicon GPU (M1/M2/M3/M4 Mac)
3. **CPU** — fallback on any machine

The active device is printed at startup:

```
Loading kokoro TTS engine...
  Device: MPS (Apple Silicon GPU)
Engine ready (mps).
```

To override the automatic selection:

```bash
uv run autobook narrate book.epub --device mps    # force Apple Silicon GPU
uv run autobook narrate book.epub --device cuda   # force NVIDIA GPU
uv run autobook narrate book.epub --device cpu    # force CPU
```

---

## Cloud TTS fallback (edge-tts)

If you'd rather not download the Kokoro model, the `edge` engine uses Microsoft's neural voices over the internet — no local model required.

```bash
uv run autobook narrate book.epub --engine edge
uv run autobook narrate book.epub --engine edge --voice en-GB-SoniaNeural
uv run autobook narrate book.epub --engine edge --voice en-US-GuyNeural
```

To see all available edge-tts voices:

```bash
uv run edge-tts --list-voices
```

> **Note:** edge-tts requires an active internet connection and ignores `--device` and `--speed`.

---

## Output structure

```
output/
└── my_book_title/
    ├── chapter_01_preface.wav
    ├── chapter_02_chapter_i.wav
    ├── chapter_03_chapter_ii.wav
    ├── ...
    └── audiobook.m4b          ← --combine or `autobook combine`
```

Each chapter WAV is:
- 24 kHz, 16-bit PCM (WAV)
- Peak-normalized to −18 dBFS for consistent loudness
- Prefaced with the spoken chapter title, followed by a natural pause

---

## All options

```
autobook narrate <epub> [options]

  --engine {kokoro,edge}        TTS backend (default: kokoro)
  --voice VOICE                 Voice name for the chosen engine
  --speed SPEED                 Speech speed multiplier, kokoro only (default: 1.0)
  --lang LANG                   Kokoro language: 'a' American, 'b' British (default: a)
  --device {auto,cuda,mps,cpu}  Compute device for Kokoro (default: auto)
  --output OUTPUT               Output directory (default: ./output/<book-title>/)
  --chapters CHAPTERS           Narrate a subset, e.g. '1-5' or '2,4,6'
  --chunk-words CHUNK_WORDS     Max words per TTS chunk (default: 400)
  --combine                     Also merge into audiobook.m4b (requires ffmpeg)
  --list-chapters               Print detected chapters and exit

autobook combine <directory> [options]

  --title TITLE                 M4B metadata title (default: derived from directory name)
  --output OUTPUT               Output .m4b path (default: <directory>/audiobook.m4b)
```

---

## Project structure

```
autobook/
├── autobook.py          # CLI entry point and orchestration
├── epub_parser.py       # EPUB chapter extraction
├── text_processor.py    # Text cleaning and TTS-safe chunking
├── tts_engine.py        # Kokoro and edge-tts backends, device detection
├── audio_processor.py   # Audio assembly, normalization, M4B export
├── pyproject.toml
└── uv.lock
```
