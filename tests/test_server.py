import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from natlas.defaults import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from natlas.engine import Chunk
from natlas.server import create_app


class FakeEngine:
    name = "fake.gguf"

    def __init__(self):
        self.calls = []

    def count_prompt_tokens(self, messages):
        return sum(len(m["content"].split()) for m in messages)

    def count_tokens(self, text):
        return len(text.split())

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


def test_accepts_content_parts_and_developer_role(engine):
    client = TestClient(create_app(engine))
    resp = client.post("/v1/chat/completions", json={"messages": [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "capital"},
                                     {"type": "text", "text": "of Nigeria?"}]},
    ]})
    assert resp.status_code == 200
    messages, _ = engine.calls[0]
    assert messages[0] == {"role": "system", "content": "Be brief."}
    assert messages[1] == {"role": "user", "content": "capital\nof Nigeria?"}


def test_errors_use_openai_shape(engine):
    resp = TestClient(create_app(engine)).post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "messages must not be empty"


class OverflowEngine(FakeEngine):
    def generate(self, messages, **params):
        raise ValueError("Requested tokens (9000) exceed context window of 8192")
        yield  # pragma: no cover


def test_context_overflow_is_a_client_error():
    resp = TestClient(create_app(OverflowEngine())).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
    )
    assert resp.status_code == 400
    assert "context window" in resp.json()["error"]["message"]


def test_api_key_required_when_set(engine):
    client = TestClient(create_app(engine, api_key="s3cret"))
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert client.post("/v1/chat/completions", json=body, headers=wrong).status_code == 401
    ok = {"Authorization": "Bearer s3cret"}
    assert client.post("/v1/chat/completions", json=body, headers=ok).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for probes


def test_transcription_language_from_model_name(engine):
    client = TestClient(create_app(engine, asr=FakeASR()))
    resp = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"abcd")},
                       data={"model": "NCAIR1/Yoruba-ASR"})
    assert resp.json()["text"] == "yo:4"
    missing = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"abcd")},
                          data={"model": "whisper-1"})
    assert missing.status_code == 400
    assert "language" in missing.json()["error"]["message"]


def test_transcription_text_response_format(engine):
    client = TestClient(create_app(engine, asr=FakeASR()))
    resp = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"ab")},
                       data={"language": "ha", "response_format": "text"})
    assert resp.text == "ha:2"


class FailingEngine(FakeEngine):
    def generate(self, messages, **params):
        yield Chunk("Abu")
        raise RuntimeError("GPU fell over")


def test_engine_failure_mid_stream_is_reported_in_the_stream():
    resp = TestClient(create_app(FailingEngine())).post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "x"}], "stream": True,
    })
    frames = [json.loads(line[6:]) for line in resp.text.splitlines()
              if line.startswith("data: {")]
    assert frames[1]["choices"][0]["delta"]["content"] == "Abu"
    assert frames[-1]["error"]["message"] == "GPU fell over"
    assert resp.text.rstrip().endswith("data: [DONE]")


def test_engine_failure_without_stream_is_a_500():
    client = TestClient(create_app(FailingEngine()), raise_server_exceptions=False)
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert resp.status_code == 500
    assert resp.json()["error"]["message"] == "RuntimeError: GPU fell over"
