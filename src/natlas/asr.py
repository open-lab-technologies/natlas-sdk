"""Speech to text with the official N-ATLAS speech models (Whisper-small fine-tunes).

The models are gated on Hugging Face: accept the terms on each model page and
log in with `hf auth login` before first use.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
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
        """Decode audio bytes to 16 kHz mono float32.

        libsndfile handles wav, mp3, ogg and opus (WhatsApp voice notes) in memory.
        It cannot read m4a (iPhone recordings) or webm (browser recorders), so those
        go through ffmpeg, which must then be installed.
        """
        import librosa
        import numpy as np
        import soundfile

        try:
            audio, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
            return audio
        except soundfile.LibsndfileError:
            pass
        if shutil.which("ffmpeg") is None:
            raise ValueError("Unsupported audio format; install ffmpeg to read m4a, webm and others")
        # A temporary file rather than stdin: m4a keeps its index at the end of the
        # file, and ffmpeg cannot seek back to it in a pipe.
        with tempfile.NamedTemporaryFile(suffix=".audio") as src:
            src.write(data)
            src.flush()
            proc = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", src.name, "-f", "f32le", "-ac", "1",
                 "-ar", str(SAMPLE_RATE), "pipe:1"],
                capture_output=True, check=False,
            )
        if proc.returncode != 0 or not proc.stdout:
            raise ValueError(f"Could not decode audio: {proc.stderr.decode(errors='replace')[:200]}")
        return np.frombuffer(proc.stdout, dtype=np.float32)

    def transcribe(self, data: bytes, language: str) -> str:
        code = get_language(language).code
        audio = self.load_audio(data)
        with self._lock:
            pipe = self._pipeline(code)
            text = self._run(pipe, audio, {})
            if looks_looped(text):
                # The speech models, Igbo especially, sometimes fall into a loop
                # ("onye onye onye ...") that turns one sentence into hundreds of words.
                # Banning repeated 4-grams stops it. It is only applied on a retry
                # because it also blocks genuine repeats and costs accuracy elsewhere.
                text = self._run(pipe, audio, {"no_repeat_ngram_size": 4})
        return text

    @staticmethod
    def _run(pipe, audio, generate_kwargs: dict) -> str:
        result = pipe(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            chunk_length_s=30,
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        return result["text"].strip()


def looks_looped(text: str, max_repeats: int = 3, max_n: int = 4) -> bool:
    """True if any 1- to `max_n`-word phrase repeats back to back more than `max_repeats` times."""
    words = text.split()
    for n in range(1, max_n + 1):
        for i in range(len(words) - n):
            phrase, repeats = words[i:i + n], 1
            while words[i + repeats * n:i + (repeats + 1) * n] == phrase:
                repeats += 1
                if repeats > max_repeats:
                    return True
    return False
