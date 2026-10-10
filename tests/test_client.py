import json

import httpx
import pytest
import respx

from natlas import NAtlas, NAtlasError, get_language
from natlas.client import DEFAULT_SYSTEM_PROMPT

BASE = "http://test/v1"


def completion(text: str) -> dict:
    return {
        "model": "n-atlas",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


@respx.mock
def test_chat_injects_system_prompt_and_defaults():
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(" Abuja. "))
    )
    reply = NAtlas(base_url=BASE).chat("Capital of Nigeria?")

    assert reply.text == "Abuja."
    assert reply.usage["total_tokens"] == 7
    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"][0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert sent["temperature"] == 0.1
    assert sent["repetition_penalty"] == 1.12


@respx.mock
def test_chat_keeps_caller_system_prompt():
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("ok"))
    )
    NAtlas(base_url=BASE).chat([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ])
    sent = json.loads(route.calls[0].request.content)
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"] == "Be brief."


@respx.mock
def test_api_key_sent_as_bearer():
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("ok"))
    )
    NAtlas(base_url=BASE, api_key="secret").chat("hi")
    assert route.calls[0].request.headers["authorization"] == "Bearer secret"


@respx.mock
def test_error_status_raises_with_detail():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(400, json={"detail": "messages must not be empty"})
    )
    with pytest.raises(NAtlasError, match="messages must not be empty") as err:
        NAtlas(base_url=BASE).chat("hi")
    assert err.value.status_code == 400


@respx.mock
def test_unreachable_server_raises_natlas_error():
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(NAtlasError, match="Could not reach"):
        NAtlas(base_url=BASE).chat("hi")


@respx.mock
def test_stream_yields_deltas_until_done():
    body = "".join(
        f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n"
        for d in ({"role": "assistant"}, {"content": "San"}, {"content": "nu"}, {})
    ) + "data: [DONE]\n\n"
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, text=body))
    assert list(NAtlas(base_url=BASE).stream("hello")) == ["San", "nu"]


@respx.mock
def test_translate_names_both_languages():
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("Ẹ kú àárọ̀"))
    )
    assert NAtlas(base_url=BASE).translate("Good morning", to="Yoruba") == "Ẹ kú àárọ̀"
    prompt = json.loads(route.calls[0].request.content)["messages"][-1]["content"]
    assert "English text into Yoruba" in prompt
    assert json.loads(route.calls[0].request.content)["temperature"] == 0.0


@respx.mock
def test_transcribe_sends_language_and_model(tmp_path):
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"fake")
    route = respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": " Sannu "})
    )
    assert NAtlas(base_url=BASE).transcribe(audio, language="hausa") == "Sannu"
    body = route.calls[0].request.content
    assert b"NCAIR1/Hausa-ASR" in body and b"note.ogg" in body


def test_unknown_language_lists_supported():
    with pytest.raises(ValueError, match="yo \\(Yoruba\\)"):
        get_language("pidgin")


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("NATLAS_BASE_URL", "http://remote:9000/v1/")
    assert NAtlas().base_url == "http://remote:9000/v1"
