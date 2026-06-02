"""
Assembles per-chapter audio from TTS chunks, normalizes volume, and
optionally combines everything into a single M4B audiobook.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from tqdm import tqdm


SAMPLE_RATE = 24000  # target sample rate for all output


def write_chapter_audio(
    audio_chunks: List[np.ndarray],
    chapter_path: Path,
    sample_rate: int = SAMPLE_RATE,
    pause_between_chunks: float = 0.5,
    pause_after_title: float = 1.0,
) -> None:
    """Concatenate chunks with natural pauses and write a WAV file."""
    if not audio_chunks:
        return

    segments: List[np.ndarray] = []
    title_pause = _silence(int(sample_rate * pause_after_title), sample_rate)
    chunk_pause = _silence(int(sample_rate * pause_between_chunks), sample_rate)

    # First chunk is the chapter title narration
    segments.append(audio_chunks[0])
    segments.append(title_pause)

    for i, chunk in enumerate(audio_chunks[1:], start=1):
        segments.append(chunk)
        if i < len(audio_chunks) - 1:
            segments.append(chunk_pause)

    audio = np.concatenate(segments)
    audio = normalize(audio)
    sf.write(str(chapter_path), audio, sample_rate, subtype="PCM_16")


def normalize(audio: np.ndarray, target_dBFS: float = -18.0) -> np.ndarray:
    """Peak-normalize to target dBFS."""
    peak = np.abs(audio).max()
    if peak < 1e-6:
        return audio
    target_peak = 10 ** (target_dBFS / 20)
    return (audio / peak * target_peak).clip(-1.0, 1.0).astype(np.float32)


def combine_to_m4b(
    chapter_files: List[Path],
    output_path: Path,
    title: str = "",
    chapter_titles: List[str] | None = None,
) -> bool:
    """
    Merge all chapter WAV files into a single M4B audiobook using ffmpeg.
    Returns True on success. Requires ffmpeg in PATH.
    """
    if not chapter_files:
        return False

    if not _ffmpeg_available():
        return False

    durations = [sf.info(str(f)).duration for f in chapter_files]
    total_seconds = sum(durations)

    concat_list = output_path.parent / "_concat.txt"
    with concat_list.open("w") as f:
        for p in chapter_files:
            f.write(f"file '{p.resolve()}'\n")

    # Build ffmetadata file with chapter markers
    meta_path = output_path.parent / "_meta.txt"
    with meta_path.open("w") as f:
        f.write(";FFMETADATA1\n")
        if title:
            f.write(f"title={title}\n")
        offset_ms = 0
        for i, (dur, path) in enumerate(zip(durations, chapter_files)):
            chapter_title = (chapter_titles[i] if chapter_titles and i < len(chapter_titles)
                             else _title_from_path(path))
            end_ms = offset_ms + int(dur * 1000)
            f.write("\n[CHAPTER]\nTIMEBASE=1/1000\n")
            f.write(f"START={offset_ms}\nEND={end_ms}\ntitle={chapter_title}\n")
            offset_ms = end_ms

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-i", str(meta_path),
        "-map_metadata", "1",
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        "-nostats",
    ]
    cmd.append(str(output_path))

    success = False
    with tqdm(
        total=int(total_seconds),
        unit="s",
        unit_scale=True,
        desc="  Encoding",
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}]",
    ) as pbar:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        elapsed = 0.0
        for line in proc.stdout:
            secs = _parse_out_time(line.strip())
            if secs is not None:
                pbar.update(secs - elapsed)
                elapsed = secs
        proc.wait()
        pbar.update(int(total_seconds) - elapsed)  # snap to 100% on success
        success = proc.returncode == 0

    concat_list.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return success


def _title_from_path(path: Path) -> str:
    """Derive a human-readable chapter title from a chapter_NNN_slug.wav filename."""
    name = path.stem  # e.g. "chapter_01_the_beginning"
    parts = name.split("_", 2)
    if len(parts) == 3:
        return parts[2].replace("_", " ").title()
    return name.replace("_", " ").title()


def _parse_out_time(line: str) -> float | None:
    """Parse 'out_time=HH:MM:SS.ffffff' from ffmpeg -progress output → seconds."""
    if not line.startswith("out_time="):
        return None
    time_str = line.split("=", 1)[1].strip()
    try:
        h, m, s = time_str.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, IndexError):
        return None


def _silence(n_samples: int, sample_rate: int) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.float32)


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
