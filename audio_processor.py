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


def combine_to_m4b(chapter_files: List[Path], output_path: Path, title: str = "") -> bool:
    """
    Merge all chapter WAV files into a single M4B audiobook using ffmpeg.
    Returns True on success. Requires ffmpeg in PATH.
    """
    if not chapter_files:
        return False

    if not _ffmpeg_available():
        return False

    concat_list = output_path.parent / "_concat.txt"
    with concat_list.open("w") as f:
        for p in chapter_files:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
    ]
    if title:
        cmd += ["-metadata", f"title={title}"]
    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True)
    concat_list.unlink(missing_ok=True)
    return result.returncode == 0


def _silence(n_samples: int, sample_rate: int) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.float32)


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
