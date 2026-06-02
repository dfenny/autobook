#!/usr/bin/env python3
"""
autobook — Convert an EPUB ebook into a narrated audiobook.

Usage:
  python autobook.py book.epub
  python autobook.py book.epub --engine kokoro --voice af_bella --speed 0.95
  python autobook.py book.epub --engine edge --voice en-GB-SoniaNeural
  python autobook.py book.epub --combine          # merge chapters → audiobook.m4b
  python autobook.py book.epub --chapters 1-5     # narrate a range only
"""

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

from epub_parser import Chapter, parse_epub
from text_processor import clean_text, split_into_chunks
from audio_processor import combine_to_m4b, write_chapter_audio
from tts_engine import Backend, Device, make_engine


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


def narrate_chapter(
    chapter: Chapter,
    engine,
    output_dir: Path,
    max_words: int = 400,
) -> Path | None:
    """Synthesize one chapter and write it to a WAV file. Returns the path."""
    print(f"\n  [{chapter.index + 1}] {chapter.title}")

    cleaned = clean_text(chapter.text)
    if not cleaned.strip():
        print("      (empty after cleaning, skipping)")
        return None

    chunks_text = split_into_chunks(cleaned, max_words=max_words)

    audio_chunks = []

    # Synthesize title first
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

    stem = f"chapter_{chapter.index + 1:02d}_{slugify(chapter.title)}"
    out_path = output_dir / f"{stem}.wav"
    write_chapter_audio(audio_chunks, out_path, sample_rate=engine.sample_rate)
    print(f"      → {out_path.name}")
    return out_path


def build_kokoro_kwargs(args) -> dict:
    kwargs: dict = {
        "voice": args.voice or "af_heart",
        "speed": args.speed,
        "device": args.device,
    }
    return kwargs


def build_edge_kwargs(args) -> dict:
    kwargs = {}
    if args.voice:
        kwargs["voice"] = args.voice
    else:
        kwargs["voice"] = "en-US-JennyNeural"
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an EPUB ebook into a chapter-by-chapter narrated audiobook."
    )
    parser.add_argument("epub", help="Path to the .epub file")
    parser.add_argument(
        "--engine",
        choices=["kokoro", "edge"],
        default="kokoro",
        help="TTS backend (default: kokoro). 'kokoro' downloads a ~300 MB model on first run.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice name for the chosen engine. Kokoro: af_heart, af_bella, am_adam, am_michael, "
             "bm_george, bf_emma. Edge: en-US-JennyNeural, en-GB-SoniaNeural, etc.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (kokoro only, default 1.0)",
    )
    parser.add_argument(
        "--lang",
        default="a",
        help="Kokoro language code: 'a' = American English, 'b' = British English (default: a)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./output/<book-title>/)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all chapters into a single audiobook.m4b (requires ffmpeg)",
    )
    parser.add_argument(
        "--chapters",
        default=None,
        help="Narrate only specific chapters, e.g. '1-5' or '2,4,6'",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=400,
        dest="chunk_words",
        help="Max words per TTS chunk (default: 400)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        dest="device",
        help="Compute device for Kokoro (default: auto). "
             "auto selects CUDA → MPS → CPU in priority order.",
    )
    parser.add_argument(
        "--list-chapters",
        action="store_true",
        help="List detected chapters and exit without generating audio",
    )
    args = parser.parse_args()

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

    # Determine which chapters to narrate
    if args.chapters:
        indices = parse_chapter_range(args.chapters, len(chapters))
        chapters_to_narrate = [chapters[i] for i in indices]
    else:
        chapters_to_narrate = chapters

    total_words = sum(len(ch.text.split()) for ch in chapters_to_narrate)
    print(f"Found {len(chapters)} chapter(s). Narrating {len(chapters_to_narrate)} "
          f"({total_words:,} words).")

    # Set up output directory
    book_slug = slugify(epub_path.stem)
    output_dir = Path(args.output) if args.output else Path("output") / book_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Build TTS engine
    print(f"\nLoading {args.engine} TTS engine...")
    engine_kwargs = (
        {**build_kokoro_kwargs(args), "lang": args.lang}
        if args.engine == "kokoro"
        else build_edge_kwargs(args)
    )
    engine = make_engine(args.engine, **engine_kwargs)
    device_label = getattr(engine, "device", "cloud")
    print(f"Engine ready ({device_label}).\n")

    # Narrate chapters
    chapter_files = []
    for chapter in chapters_to_narrate:
        out_path = narrate_chapter(
            chapter, engine, output_dir, max_words=args.chunk_words
        )
        if out_path:
            chapter_files.append(out_path)

    print(f"\n{len(chapter_files)} chapter file(s) written to {output_dir}/")

    # Optionally combine into M4B
    if args.combine:
        m4b_path = output_dir / "audiobook.m4b"
        print(f"\nCombining into {m4b_path.name}...")
        success = combine_to_m4b(chapter_files, m4b_path, title=epub_path.stem)
        if success:
            size_mb = m4b_path.stat().st_size / 1_048_576
            print(f"Audiobook written: {m4b_path} ({size_mb:.1f} MB)")
        else:
            print("Warning: ffmpeg not found or failed. Skipping M4B creation.")
            print("Install ffmpeg to enable combined output.")

    print("\nDone.")


if __name__ == "__main__":
    main()
