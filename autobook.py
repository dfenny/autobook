#!/usr/bin/env python3
"""
autobook — Convert an EPUB ebook into a narrated audiobook.

Subcommands:
  narrate   Synthesize speech for each chapter of an EPUB.
  combine   Merge existing chapter WAVs into a single audiobook.m4b.

Examples:
  uv run autobook narrate book.epub
  uv run autobook narrate book.epub --voice af_bella --speed 0.95
  uv run autobook narrate book.epub --combine        # narrate + merge in one step
  uv run autobook combine  output/my-book/           # merge previously narrated WAVs
  uv run autobook combine  output/my-book/ --title "My Book" --output ~/Desktop/my-book.m4b
"""

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

from epub_parser import Chapter, parse_epub
from text_processor import clean_text, split_into_chunks
from audio_processor import combine_to_m4b, write_chapter_audio, _ffmpeg_available
from tts_engine import Backend, Device, make_engine


# ---------------------------------------------------------------------------
# Helpers shared by both subcommands
# ---------------------------------------------------------------------------

def parse_chapter_range(spec: str, total: int) -> list[int]:
    """Parse '3', '1-5', or '2,4,6' into a sorted list of 0-based indices."""
    indices: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            indices.update(range(lo - 1, hi))
        elif part.isdigit():
            indices.add(int(part) - 1)
        else:
            raise ValueError(f"Invalid chapter spec: {part!r}")
    return sorted(i for i in indices if 0 <= i < total)


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s-]+", "_", text).strip("_")[:60]


def collect_chapter_wavs(directory: Path) -> list[Path]:
    """Return chapter WAVs from a directory, sorted by chapter number."""
    def chapter_num(p: Path) -> int:
        m = re.match(r"chapter_(\d+)_", p.name)
        return int(m.group(1)) if m else 0

    return sorted(directory.glob("chapter_*.wav"), key=chapter_num)


def run_combine(chapter_files: list[Path], output_path: Path, title: str) -> None:
    """Shared combine logic used by both subcommands."""
    if not chapter_files:
        print("No chapter WAV files found.", file=sys.stderr)
        sys.exit(1)

    if not _ffmpeg_available():
        print(
            "Error: ffmpeg not found in PATH.\n"
            "Install it and re-run:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Combining {len(chapter_files)} chapter(s) → {output_path.name}...")
    for f in chapter_files:
        print(f"  {f.name}")

    success = combine_to_m4b(chapter_files, output_path, title=title)
    if success:
        size_mb = output_path.stat().st_size / 1_048_576
        print(f"\nAudiobook written: {output_path} ({size_mb:.1f} MB)")
    else:
        print("Error: ffmpeg failed. Check the files and try again.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# narrate subcommand
# ---------------------------------------------------------------------------

def narrate_chapter(
    chapter: Chapter,
    engine,
    output_dir: Path,
    max_words: int = 400,
) -> Path | None:
    print(f"\n  [{chapter.index + 1}] {chapter.title}")

    cleaned = clean_text(chapter.text)
    if not cleaned.strip():
        print("      (empty after cleaning, skipping)")
        return None

    chunks_text = split_into_chunks(cleaned, max_words=max_words)
    audio_chunks = []

    title_audio = engine.synthesize(chapter.title)
    if isinstance(title_audio, tuple):
        title_audio, _ = title_audio
    audio_chunks.append(title_audio)

    with tqdm(total=len(chunks_text), desc="      chunks", unit="chunk", leave=False) as pbar:
        for chunk in chunks_text:
            result = engine.synthesize(chunk)
            if isinstance(result, tuple):
                result, _ = result
            audio_chunks.append(result)
            pbar.update(1)

    stem = f"chapter_{chapter.index + 1:03d}_{slugify(chapter.title)}"
    out_path = output_dir / f"{stem}.wav"
    write_chapter_audio(audio_chunks, out_path, sample_rate=engine.sample_rate)
    print(f"      → {out_path.name}")
    return out_path


def cmd_narrate(args: argparse.Namespace) -> None:
    epub_path = Path(args.epub).expanduser().resolve()
    if not epub_path.exists():
        print(f"Error: file not found: {epub_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {epub_path.name}...")
    chapters = parse_epub(str(epub_path))

    if not chapters:
        print("No readable chapters found in this EPUB.", file=sys.stderr)
        sys.exit(1)

    if args.list_chapters:
        print(f"\nFound {len(chapters)} chapter(s):\n")
        for ch in chapters:
            words = len(ch.text.split())
            print(f"  {ch.index + 1:3d}. {ch.title[:70]:<70s} ({words:,} words)")
        return

    chapters_to_narrate = (
        [chapters[i] for i in parse_chapter_range(args.chapters, len(chapters))]
        if args.chapters
        else chapters
    )

    total_words = sum(len(ch.text.split()) for ch in chapters_to_narrate)
    print(
        f"Found {len(chapters)} chapter(s). Narrating {len(chapters_to_narrate)} "
        f"({total_words:,} words)."
    )

    output_dir = Path(args.output) if args.output else Path("output") / slugify(epub_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print(f"\nLoading {args.engine} TTS engine...")
    if args.engine == "kokoro":
        engine_kwargs: dict = {"voice": args.voice or "af_heart", "speed": args.speed, "device": args.device, "lang": args.lang}
    else:
        engine_kwargs = {"voice": args.voice or "en-US-JennyNeural"}
    engine = make_engine(args.engine, **engine_kwargs)
    device_label = getattr(engine, "device", "cloud")
    print(f"Engine ready ({device_label}).\n")

    chapter_files = []
    for chapter in chapters_to_narrate:
        out_path = narrate_chapter(chapter, engine, output_dir, max_words=args.chunk_words)
        if out_path:
            chapter_files.append(out_path)

    print(f"\n{len(chapter_files)} chapter file(s) written to {output_dir}/")

    if args.combine:
        m4b_path = output_dir / "audiobook.m4b"
        run_combine(chapter_files, m4b_path, title=epub_path.stem)

    print("\nDone.")


# ---------------------------------------------------------------------------
# combine subcommand
# ---------------------------------------------------------------------------

def cmd_combine(args: argparse.Namespace) -> None:
    directory = Path(args.directory).expanduser().resolve()
    if not directory.is_dir():
        print(f"Error: not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    chapter_files = collect_chapter_wavs(directory)
    if not chapter_files:
        print(f"Error: no chapter_*.wav files found in {directory}", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else directory / "audiobook.m4b"
    )
    title = args.title or directory.name.replace("_", " ").title()

    run_combine(chapter_files, output_path, title=title)
    print("\nDone.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autobook",
        description="Convert an EPUB ebook into a narrated audiobook.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # -- narrate -------------------------------------------------------------
    narrate = subparsers.add_parser(
        "narrate",
        help="Synthesize speech for each chapter of an EPUB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    narrate.add_argument("epub", help="Path to the .epub file")
    narrate.add_argument(
        "--engine", choices=["kokoro", "edge"], default="kokoro",
        help="TTS backend (default: kokoro). kokoro downloads a ~300 MB model on first run.",
    )
    narrate.add_argument(
        "--voice", default=None,
        help="Voice name. Kokoro: af_heart, af_bella, am_adam, am_michael, bm_george, bf_emma. "
             "Edge: en-US-JennyNeural, en-GB-SoniaNeural, etc.",
    )
    narrate.add_argument(
        "--speed", type=float, default=1.0,
        help="Speech speed multiplier, kokoro only (default: 1.0)",
    )
    narrate.add_argument(
        "--lang", default="a",
        help="Kokoro language: 'a' American English, 'b' British English (default: a)",
    )
    narrate.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
        help="Compute device for Kokoro (default: auto → CUDA > MPS > CPU)",
    )
    narrate.add_argument(
        "--output", default=None,
        help="Output directory (default: ./output/<book-title>/)",
    )
    narrate.add_argument(
        "--chapters", default=None,
        help="Narrate a subset of chapters, e.g. '1-5' or '2,4,6'",
    )
    narrate.add_argument(
        "--chunk-words", type=int, default=400, dest="chunk_words",
        help="Max words per TTS chunk (default: 400)",
    )
    narrate.add_argument(
        "--combine", action="store_true",
        help="Also merge all chapters into audiobook.m4b after narrating (requires ffmpeg)",
    )
    narrate.add_argument(
        "--list-chapters", action="store_true",
        help="List detected chapters and exit without generating audio",
    )

    # -- combine -------------------------------------------------------------
    combine = subparsers.add_parser(
        "combine",
        help="Merge existing chapter WAVs into a single audiobook.m4b",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    combine.add_argument(
        "directory",
        help="Directory containing chapter_*.wav files (e.g. output/my-book/)",
    )
    combine.add_argument(
        "--title", default=None,
        help="Audiobook title stored in M4B metadata (default: derived from directory name)",
    )
    combine.add_argument(
        "--output", default=None,
        help="Output .m4b path (default: <directory>/audiobook.m4b)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "narrate":
        cmd_narrate(args)
    elif args.command == "combine":
        cmd_combine(args)


if __name__ == "__main__":
    main()
