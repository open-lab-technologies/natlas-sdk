import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from natlas.client import DEFAULT_SYSTEM_PROMPT
from natlas.engine import Chunk
from natlas.server import create_app


class FakeEngine:
    model_path = Path("fake.gguf")

    def __init__(self):
        self.calls = []

    def render(self, messages):
        return " ".join(m["content"] for m in messages)

    def count_tokens(self, text):
        return len(text.split()) + 1

    def generate(self, messages, **params):
        self.calls.append((messages, params))
        yield Chunk("Abu")
        yield Chunk("ja", "stop")


class FakeASR:
    def transcribe(self, data, language):
        return f"{language}:{len(data)}"


@pytest.fixture
def engine():
    return FakeEngine()


def test_chat_completion_shape_and_injected_system_prompt(engine):
    client = TestClient(create_app(engine))
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "capital?"}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Abuja"
    assert body["choices"][0]["finish_reason"] == "stop"
    messages, params = engine.calls[0]
    assert messages[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert params["repetition_penalty"] == 1.12


def test_streaming_emits_openai_chunks(engine):
    client = TestClient(create_app(engine))
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "capital?"}], "stream": True,
    })
    frames = [l[6:] for l in resp.text.splitlines() if l.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    deltas = [json.loads(f)["choices"][0]["delta"].get("content", "") for f in frames[:-1]]
    assert "".join(deltas) == "Abuja"
    assert json.loads(frames[-2])["choices"][0]["finish_reason"] == "stop"


def test_empty_messages_rejected(engine):
    resp = TestClient(create_app(engine)).post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400


def test_transcription_disabled_without_asr(engine):
    resp = TestClient(create_app(engine)).post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", b"x")}, data={"language": "yo"}
    )
    assert resp.status_code == 501


def test_transcription_with_asr(engine):
    client = TestClient(create_app(engine, asr=FakeASR()))
    resp = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", b"abc")}, data={"language": "Igbo"}
    )
    assert resp.json() == {"text": "ig:3", "language": "ig"}
    bad = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", b"abc")}, data={"language": "fr"}
    )
    assert bad.status_code == 400
