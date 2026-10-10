"""End-to-end tests against a running `natlas serve`. Skipped unless NATLAS_E2E_URL is set:

    natlas serve --asr --api-key test-key
    NATLAS_E2E_URL=http://127.0.0.1:8000/v1 NATLAS_E2E_API_KEY=test-key pytest tests/test_live.py

Speech tests also need ffmpeg and use FLEURS clips downloaded by `natlas bench`.
"""

import os
import shutil
import subprocess
import time

import httpx
import pytest

from natlas import NAtlas, NAtlasError

URL = os.getenv("NATLAS_E2E_URL")
KEY = os.getenv("NATLAS_E2E_API_KEY")
pytestmark = pytest.mark.skipif(not URL, reason="set NATLAS_E2E_URL to run live tests")


@pytest.fixture(scope="module")
def client():
    with NAtlas(base_url=URL, api_key=KEY, timeout=300) as c:
        yield c


@pytest.fixture(scope="module")
def health():
    resp = httpx.get(URL.removesuffix("/v1") + "/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def test_models_listed(client):
    assert "n-atlas" in client.models()


@pytest.mark.parametrize(("prompt", "expected"), [
    ("What is the capital of Nigeria? Answer in one word.", "abuja"),
    ("Kí ni olú ìlú Nàìjíríà? Dáhùn ní ọ̀rọ̀ kan.", "abuja"),
    ("Mene ne babban birnin Najeriya? Amsa da kalma daya.", "abuja"),
    ("Kedu isi obodo Naịjirịa? Zaa n'otu okwu.", "abuja"),
])
def test_chat_answers_in_every_language(client, prompt, expected):
    reply = client.chat(prompt, temperature=0.0, max_tokens=40)
    assert expected in reply.text.lower()
    assert reply.usage.completion_tokens > 0


def test_stream_matches_chat(client):
    prompt = "Count from one to five in English, separated by commas."
    streamed = "".join(client.stream(prompt, temperature=0.0, max_tokens=30)).strip()
    assert streamed == client.chat(prompt, temperature=0.0, max_tokens=30).text


def test_reply_does_not_depend_on_previous_request(client):
    prompt = "Count from one to five in English, separated by commas."
    first = client.chat(prompt, temperature=0.0, max_tokens=30).text
    client.chat("Tell me about rivers in Nigeria.", temperature=0.0, max_tokens=20)
    assert client.chat(prompt, temperature=0.0, max_tokens=30).text == first
    assert client.chat(prompt, temperature=0.0, max_tokens=30).text == first


@pytest.mark.parametrize("lang", ["ha", "ig", "yo"])
def test_translate_changes_the_text(client, lang):
    source = "Good morning, how is your family?"
    out = client.translate(source, to=lang)
    assert out and out.lower() != source.lower()


def test_server_recovers_when_a_client_hangs_up_mid_stream(client):
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    body = {"model": "n-atlas", "stream": True, "max_tokens": 400,
            "messages": [{"role": "user", "content": "Write a long story about Lagos."}]}
    with httpx.Client(timeout=60) as http, \
            http.stream("POST", f"{URL}/chat/completions", json=body, headers=headers) as resp:
        for _ in resp.iter_lines():
            break
    start = time.monotonic()
    assert client.chat("Say yes.", max_tokens=5).text
    assert time.monotonic() - start < 30


def test_prompt_longer_than_context_is_rejected(client):
    with pytest.raises(NAtlasError) as err:
        client.chat("word " * 20000, max_tokens=10)
    assert err.value.status_code == 400


def test_wrong_api_key_rejected():
    if not KEY:
        pytest.skip("server started without --api-key")
    with NAtlas(base_url=URL, api_key="wrong") as bad, pytest.raises(NAtlasError) as err:
        bad.chat("hi")
    assert err.value.status_code == 401


def test_official_openai_client():
    openai = pytest.importorskip("openai")
    oa = openai.OpenAI(base_url=URL, api_key=KEY or "unused")
    reply = oa.chat.completions.create(
        model="n-atlas",
        messages=[{"role": "developer", "content": "Answer in one word."},
                  {"role": "user", "content": [{"type": "text",
                                                "text": "What is the capital of Nigeria?"}]}],
        temperature=0,
    )
    assert "abuja" in reply.choices[0].message.content.lower()
    chunks = oa.chat.completions.create(
        model="n-atlas", messages=[{"role": "user", "content": "Say hello."}], stream=True,
    )
    assert "".join(c.choices[0].delta.content or "" for c in chunks).strip()


def _fleurs_clip(code: str):
    from natlas.bench.data import fleurs

    item = fleurs(code, 1)[0]
    return item.meta["audio"], item.gold


@pytest.fixture(scope="module")
def asr_enabled(health):
    if not health["asr"]:
        pytest.skip("server started without --asr")


@pytest.mark.usefixtures("asr_enabled")
@pytest.mark.parametrize("lang", ["en", "ha", "ig", "yo"])
def test_transcribe_real_speech(client, lang):
    from natlas.bench.scoring import wer

    audio, reference = _fleurs_clip(lang)
    text = client.transcribe(audio, language=lang)
    assert wer([text], [reference], ignore_tones=True) < 0.6


@pytest.mark.usefixtures("asr_enabled")
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("suffix", [".ogg", ".m4a", ".webm"])
def test_transcribe_phone_and_browser_formats(client, tmp_path, suffix):
    wav, _ = _fleurs_clip("ha")
    out = tmp_path / f"note{suffix}"
    codec = ["-c:a", "aac"] if suffix == ".m4a" else ["-c:a", "libopus"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", wav, *codec, str(out)], check=True)
    assert client.transcribe(out.read_bytes(), language="ha")


@pytest.mark.usefixtures("asr_enabled")
def test_openai_client_transcription_by_model_name():
    openai = pytest.importorskip("openai")
    oa = openai.OpenAI(base_url=URL, api_key=KEY or "unused")
    wav, _ = _fleurs_clip("yo")
    with open(wav, "rb") as f:
        result = oa.audio.transcriptions.create(model="NCAIR1/Yoruba-ASR", file=f)
    assert result.text.strip()
