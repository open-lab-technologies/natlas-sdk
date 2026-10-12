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
from typing import Any, Protocol

from ._extras import require
from .languages import get_language

SAMPLE_RATE = 16_000
WHISPER_WINDOW_SECONDS = 30


class Transcriber(Protocol):
    """What `pynatlas serve` needs from a speech backend."""

    def transcribe(self, data: bytes, language: str) -> str:
        """Audio file bytes in any supported format to text, for a language code."""
        ...


class AudioDecodeError(ValueError):
    pass


def load_audio(data: bytes) -> Any:
    """Decode audio file bytes to a 16 kHz mono float32 numpy array.

    libsndfile reads wav, flac, mp3, ogg and opus (WhatsApp voice notes) in memory.
    It cannot read m4a (iPhone recordings) or webm (browser recorders); those go
    through ffmpeg, which must then be installed.
    """
    np = require("numpy", "asr")
    soundfile = require("soundfile", "asr")
    try:
        audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except soundfile.LibsndfileError:
        return _ffmpeg_decode(data, np)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = require("soxr", "asr").resample(audio, rate, SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _ffmpeg_decode(data: bytes, np: Any) -> Any:
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError(
            "Unsupported audio format; install ffmpeg to read m4a, webm and others")
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
        detail = proc.stderr.decode(errors="replace")[:200]
        raise AudioDecodeError(f"Could not decode audio: {detail}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


class SpeechRecogniser:
    """`Transcriber` backed by local Hugging Face pipelines, one per language,
    loaded on first use and kept."""

    def __init__(self, device: str | None = None):
        self._device = device
        self._pipelines: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _pipeline(self, code: str) -> Any:
        if code not in self._pipelines:
            torch = require("torch", "asr")
            transformers = require("transformers", "asr")
            device = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
            self._pipelines[code] = transformers.pipeline(
                "automatic-speech-recognition",
                model=get_language(code).asr_model,
                device=device,
            )
        return self._pipelines[code]

    def transcribe(self, data: bytes, language: str) -> str:
        code = get_language(language).code
        audio = load_audio(data)
        with self._lock:
            pipe = self._pipeline(code)
            text = self._run(pipe, audio, {})
            if looks_looped(text):
                # The speech models, Igbo especially, can repeat one word hundreds of
                # times. Banning repeated 4-grams stops that, but it also blocks
                # genuine repeats and costs accuracy, so it is only used on a retry.
                text = self._run(pipe, audio, {"no_repeat_ngram_size": 4})
        return text

    @staticmethod
    def _run(pipe: Any, audio: Any, generate_kwargs: dict[str, Any]) -> str:
        result = pipe(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            chunk_length_s=WHISPER_WINDOW_SECONDS,
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        return str(result["text"]).strip()


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
