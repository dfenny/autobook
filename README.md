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

### Basic

```bash
uv run autobook book.epub
```

Output is written to `output/<book-title>/`, one WAV file per chapter.

### Preview chapters without generating audio

```bash
uv run autobook book.epub --list-chapters
```

```
Found 32 chapter(s):

    1. Preface                                                                 (1,203 words)
    2. Chapter I — In the Beginning                                            (4,871 words)
    3. Chapter II — The Road Goes Ever On                                      (5,102 words)
  ...
```

### Choose a voice

```bash
# American English voices
uv run autobook book.epub --voice af_bella          # female
uv run autobook book.epub --voice am_michael        # male

# British English voices
uv run autobook book.epub --voice bm_george --lang b
uv run autobook book.epub --voice bf_emma   --lang b
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

### Adjust speed

```bash
uv run autobook book.epub --speed 0.9    # slightly slower
uv run autobook book.epub --speed 1.15   # slightly faster
```

### Narrate specific chapters

```bash
uv run autobook book.epub --chapters 1-5      # chapters 1 through 5
uv run autobook book.epub --chapters 2,4,6    # specific chapters
uv run autobook book.epub --chapters 3        # a single chapter
```

### Combine into a single audiobook file

```bash
uv run autobook book.epub --combine
```

Requires ffmpeg. Produces `output/<book-title>/audiobook.m4b` in addition to the individual chapter WAVs.

### Custom output directory

```bash
uv run autobook book.epub --output ~/Audiobooks/my-book/
```

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
uv run autobook book.epub --device mps    # force Apple Silicon GPU
uv run autobook book.epub --device cuda   # force NVIDIA GPU
uv run autobook book.epub --device cpu    # force CPU
```

---

## Cloud TTS fallback (edge-tts)

If you'd rather not download the Kokoro model, the `edge` engine uses Microsoft's neural voices over the internet — no local model required.

```bash
uv run autobook book.epub --engine edge
uv run autobook book.epub --engine edge --voice en-GB-SoniaNeural
uv run autobook book.epub --engine edge --voice en-US-GuyNeural
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
    └── audiobook.m4b          ← only with --combine
```

Each chapter WAV is:
- 24 kHz, 16-bit PCM (WAV)
- Peak-normalized to −18 dBFS for consistent loudness
- Prefaced with the spoken chapter title, followed by a natural pause

---

## All options

```
usage: autobook [-h] [--engine {kokoro,edge}] [--voice VOICE] [--speed SPEED]
                [--lang LANG] [--output OUTPUT] [--combine]
                [--chapters CHAPTERS] [--chunk-words CHUNK_WORDS]
                [--device {auto,cuda,mps,cpu}] [--list-chapters]
                epub

positional arguments:
  epub                        Path to the .epub file

options:
  --engine {kokoro,edge}      TTS backend (default: kokoro)
  --voice VOICE               Voice name for the chosen engine
  --speed SPEED               Speech speed multiplier, kokoro only (default: 1.0)
  --lang LANG                 Kokoro language: 'a' American, 'b' British (default: a)
  --output OUTPUT             Output directory (default: ./output/<book-title>/)
  --combine                   Merge chapters into audiobook.m4b (requires ffmpeg)
  --chapters CHAPTERS         Narrate a subset, e.g. '1-5' or '2,4,6'
  --chunk-words CHUNK_WORDS   Max words per TTS chunk (default: 400)
  --device {auto,cuda,mps,cpu}  Compute device for Kokoro (default: auto)
  --list-chapters             Print detected chapters and exit
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
