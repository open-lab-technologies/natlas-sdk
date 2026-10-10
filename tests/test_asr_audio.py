import shutil
import subprocess

import pytest

pytest.importorskip("soundfile")
pytest.importorskip("soxr")
from natlas.asr import SAMPLE_RATE, AudioDecodeError, load_audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _tone(path, codec: list[str]) -> bytes:
    """Ten seconds of 440 Hz tone, written to a real file so containers like m4a keep
    their index at the end, as phone recordings do."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
         *codec, str(path)],
        capture_output=True, check=True,
    )
    return path.read_bytes()


@pytest.mark.parametrize(("name", "codec"), [
    ("a.wav", []),
    ("a.ogg", ["-c:a", "libopus"]),     # WhatsApp voice notes
    ("a.m4a", ["-c:a", "aac"]),         # iPhone recordings
    ("a.webm", ["-c:a", "libopus"]),    # browser MediaRecorder
    ("a.mp3", []),
])
def test_load_audio_decodes_common_voice_formats(tmp_path, name, codec):
    audio = load_audio(_tone(tmp_path / name, codec))
    assert abs(len(audio) / SAMPLE_RATE - 10.0) < 0.2


def test_load_audio_rejects_garbage():
    with pytest.raises(AudioDecodeError, match="Could not decode audio"):
        load_audio(b"not audio at all")
