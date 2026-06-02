"""
TTS backends: Kokoro (local, high quality) and edge-tts (cloud, no model download).

Kokoro voices (American English, lang_code='a'):
  af_heart, af_bella, af_nicole, af_sarah, af_sky  – female
  am_adam, am_michael                               – male

Kokoro voices (British English, lang_code='b'):
  bf_emma, bf_isabella – female
  bm_george, bm_lewis  – male

edge-tts voice list: run `edge-tts --list-voices`
"""

from __future__ import annotations

import asyncio
import io
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

Backend = Literal["kokoro", "edge"]
Device = Literal["auto", "cuda", "mps", "cpu"]

KOKORO_SAMPLE_RATE = 24000


def detect_device(preference: Device = "auto") -> str:
    """
    Return the best available torch device string.

    Priority when preference='auto': CUDA → MPS → CPU.
    A specific preference is validated and returned if available,
    falling back to CPU with a warning if not.
    """
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  Device: CUDA ({name})")
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("  Device: MPS (Apple Silicon GPU)")
            return "mps"
        print("  Device: CPU")
        return "cpu"

    if preference == "cuda":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  Device: CUDA ({name})")
            return "cuda"
        print("  Warning: CUDA requested but not available — falling back to CPU.")
        return "cpu"

    if preference == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("  Device: MPS (Apple Silicon GPU)")
            return "mps"
        print("  Warning: MPS requested but not available — falling back to CPU.")
        return "cpu"

    # cpu or unknown
    print("  Device: CPU")
    return "cpu"


class KokoroEngine:
    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "a",
        device: Device = "auto",
    ):
        from kokoro import KPipeline  # deferred import — triggers model download

        self.voice = voice
        self.speed = speed
        self._device = detect_device(device)
        with warnings.catch_warnings():
            # Kokoro's LSTM config uses dropout=0.2 with num_layers=1, which
            # PyTorch flags even though it's harmless in inference-only mode.
            warnings.filterwarnings(
                "ignore",
                message=".*dropout option adds dropout after all but last recurrent layer.*",
                category=UserWarning,
            )
            # torch.nn.utils.weight_norm is deprecated; Kokoro hasn't migrated yet.
            warnings.filterwarnings(
                "ignore",
                message=".*torch.nn.utils.weight_norm.*",
                category=FutureWarning,
            )
            self.pipeline = KPipeline(
                lang_code=lang,
                repo_id="hexgrad/Kokoro-82M",
                device=self._device,
            )

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        """Return float32 audio array at 24 kHz. Pass voice to override the engine default."""
        effective_voice = voice or self.voice
        chunks: list[np.ndarray] = []
        with warnings.catch_warnings():
            # Kokoro's iSTFT vocoder triggers a PyTorch deprecation notice about
            # output tensor resizing (fixed upstream but not yet released).
            warnings.filterwarnings(
                "ignore",
                message=".*resized since it had shape.*",
                category=UserWarning,
            )
            for _gs, _ps, audio in self.pipeline(text, voice=effective_voice, speed=self.speed):
                chunks.append(audio)

        if not chunks:
            return np.zeros(0, dtype=np.float32)

        silence = np.zeros(int(KOKORO_SAMPLE_RATE * 0.4), dtype=np.float32)
        out: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            out.append(chunk)
            if i < len(chunks) - 1:
                out.append(silence)
        return np.concatenate(out)

    @property
    def sample_rate(self) -> int:
        return KOKORO_SAMPLE_RATE

    @property
    def device(self) -> str:
        return self._device


class EdgeEngine:
    """Uses Microsoft edge-tts (requires internet, no local model)."""

    def __init__(self, voice: str = "en-US-JennyNeural", rate: str = "+0%"):
        self.voice = voice
        self.rate = rate

    def synthesize(self, text: str, voice: str | None = None) -> np.ndarray:
        import edge_tts
        effective_voice = voice or self.voice

        async def _run() -> bytes:
            communicate = edge_tts.Communicate(text, effective_voice, rate=self.rate)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()

        raw = asyncio.run(_run())
        # edge-tts returns MP3 bytes; decode via soundfile
        audio, sr = sf.read(io.BytesIO(raw))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), sr

    def synthesize_to_file(self, text: str, path: Path) -> None:
        import edge_tts

        async def _run():
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            await communicate.save(str(path))

        asyncio.run(_run())


def make_engine(backend: Backend, **kwargs):
    if backend == "kokoro":
        return KokoroEngine(**kwargs)
    elif backend == "edge":
        return EdgeEngine(**kwargs)
    raise ValueError(f"Unknown backend: {backend!r}")
