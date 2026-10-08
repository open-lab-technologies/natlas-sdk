"""Speech to text with the official N-ATLAS speech models (Whisper-small fine-tunes).

The models are gated on Hugging Face: accept the terms on each model page and
log in with `hf auth login` before first use.
"""

from __future__ import annotations

import io
import threading
from typing import Any

from .languages import get_language

SAMPLE_RATE = 16_000


class SpeechRecogniser:
    """Loads one pipeline per language on first use and keeps it."""

    def __init__(self, device: str | None = None):
        self._device = device
        self._pipelines: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _pipeline(self, code: str):
        if code not in self._pipelines:
            import torch
            from transformers import pipeline

            device = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
            self._pipelines[code] = pipeline(
                "automatic-speech-recognition",
                model=get_language(code).asr_model,
                device=device,
            )
        return self._pipelines[code]

    @staticmethod
    def load_audio(data: bytes):
        """Decode any format ffmpeg or libsndfile understands (WhatsApp .ogg/.opus included)."""
        import librosa

        audio, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
        return audio

    def transcribe(self, data: bytes, language: str) -> str:
        code = get_language(language).code
        audio = self.load_audio(data)
        with self._lock:
            pipe = self._pipeline(code)
            result = pipe(
                {"raw": audio, "sampling_rate": SAMPLE_RATE},
                chunk_length_s=30,
                return_timestamps=False,
            )
        return result["text"].strip()
