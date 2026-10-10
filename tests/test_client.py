import json

import httpx
import pytest
import respx

from natlas import (
    APIConnectionError,
    APITimeoutError,
    AsyncNAtlas,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NAtlas,
    NAtlasError,
    RateLimitError,
    get_language,
)
from natlas.defaults import SYSTEM_PROMPT

BASE = "http://test/v1"
CHAT = f"{BASE}/chat/completions"


def completion(text: str) -> dict:
    return {
        "model": "n-atlas",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def sse(*deltas) -> str:
    frames = [{"choices": [{"delta": d}]} for d in deltas]
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"


@pytest.fixture
def client():
    # Zero-delay retries keep the tests fast without changing the retry logic.
    c = NAtlas(base_url=BASE)
    c._retry_delay = lambda attempt, resp, _orig=c._retry_delay: (
        None if _orig(attempt, resp) is None else 0)
    return c


@respx.mock
def test_chat_sends_only_standard_fields(client):
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=completion(" Abuja. ")))
    reply = client.chat("Capital of Nigeria?")

    assert reply.text == "Abuja."
    assert reply.usage.total_tokens == 7
    sent = json.loads(route.calls[0].request.content)
    assert set(sent) == {"model", "messages", "stream", "temperature", "max_tokens"}
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}


@respx.mock
def test_provider_specific_options_pass_through(client):
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=completion("ok")))
    client.chat("hi", model="NCAIR1/N-ATLaS", repetition_penalty=1.12, temperature=None)
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "NCAIR1/N-ATLaS"
    assert sent["repetition_penalty"] == 1.12
    assert "temperature" not in sent


@respx.mock
def test_caller_system_or_developer_message_is_kept(client):
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=completion("ok")))
    client.chat([{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "hi"}])
    roles = [m["role"] for m in json.loads(route.calls[0].request.content)["messages"]]
    assert roles == ["developer", "user"]


def test_system_prompt_can_be_disabled():
    assert NAtlas(base_url=BASE, system_prompt=None)._messages("hi", None) == [
        {"role": "user", "content": "hi"}]


def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv("NATLAS_BASE_URL", "http://remote:9000/v1/")
    monkeypatch.setenv("NATLAS_MODEL", "atlas:latest")
    monkeypatch.setenv("NATLAS_API_KEY", "k")
    c = NAtlas()
    assert (c.base_url, c.model, c.api_key) == ("http://remote:9000/v1", "atlas:latest", "k")


@respx.mock
def test_api_key_sent_as_bearer():
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=completion("ok")))
    NAtlas(base_url=BASE, api_key="secret").chat("hi")
    assert route.calls[0].request.headers["authorization"] == "Bearer secret"


@pytest.mark.parametrize(("status", "error"), [
    (400, BadRequestError), (401, AuthenticationError), (429, RateLimitError),
    (503, InternalServerError),
])
@respx.mock
def test_status_codes_map_to_error_types(client, status, error):
    respx.post(CHAT).mock(return_value=httpx.Response(
        status, json={"error": {"message": "nope"}}))
    with pytest.raises(error, match="nope") as err:
        client.chat("hi")
    assert err.value.status_code == status
    assert isinstance(err.value, NAtlasError)


@respx.mock
def test_retryable_errors_are_retried_then_succeed(client):
    route = respx.post(CHAT).mock(side_effect=[
        httpx.ConnectError("refused"),
        httpx.Response(503),
        httpx.Response(200, json=completion("ok")),
    ])
    assert client.chat("hi").text == "ok"
    assert route.call_count == 3


@respx.mock
def test_retries_stop_at_max_retries(client):
    route = respx.post(CHAT).mock(return_value=httpx.Response(500))
    with pytest.raises(InternalServerError):
        client.chat("hi")
    assert route.call_count == 1 + client.max_retries


@respx.mock
def test_client_errors_are_not_retried(client):
    route = respx.post(CHAT).mock(return_value=httpx.Response(400, json={"detail": "bad"}))
    with pytest.raises(BadRequestError, match="bad"):
        client.chat("hi")
    assert route.call_count == 1


def test_retry_after_header_is_respected():
    c = NAtlas(base_url=BASE)
    resp = httpx.Response(429, headers={"retry-after": "3"})
    assert c._retry_delay(0, resp) == 3.0
    assert c._retry_delay(c.max_retries, resp) is None


@respx.mock
def test_unreachable_and_timeout_errors(client):
    respx.post(CHAT).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(APIConnectionError, match="Could not reach"):
        client.chat("hi")
    respx.post(CHAT).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(APITimeoutError):
        client.chat("hi")


@respx.mock
def test_stream_yields_deltas_until_done(client):
    respx.post(CHAT).mock(return_value=httpx.Response(
        200, text=sse({"role": "assistant"}, {"content": "San"}, {"content": "nu"}, {})))
    assert list(client.stream("hello")) == ["San", "nu"]


@respx.mock
def test_stream_retries_opening_then_raises_on_error_frame(client):
    body = "data: " + json.dumps({"error": {"message": "engine crashed"}}) + "\n\n"
    respx.post(CHAT).mock(side_effect=[httpx.Response(503), httpx.Response(200, text=body)])
    with pytest.raises(NAtlasError, match="engine crashed"):
        list(client.stream("hello"))


@respx.mock
def test_translate_names_both_languages(client):
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=completion("Ẹ kú àárọ̀")))
    assert client.translate("Good morning", to="Yoruba") == "Ẹ kú àárọ̀"
    sent = json.loads(route.calls[0].request.content)
    assert "English text into Yoruba" in sent["messages"][-1]["content"]
    assert sent["temperature"] == 0.0


@respx.mock
def test_transcribe_sends_language_and_model(client, tmp_path):
    audio = tmp_path / "note.ogg"
    audio.write_bytes(b"fake")
    route = respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": " Sannu "}))
    assert client.transcribe(audio, language="hausa") == "Sannu"
    body = route.calls[0].request.content
    assert b"NCAIR1/Hausa-ASR" in body and b"note.ogg" in body


def test_unknown_language_lists_supported():
    with pytest.raises(ValueError, match="yo \\(Yoruba\\)"):
        get_language("pidgin")


@respx.mock
async def test_async_client_matches_sync_behaviour():
    respx.post(CHAT).mock(side_effect=[
        httpx.Response(503),
        httpx.Response(200, json=completion("Abuja")),
        httpx.Response(200, text=sse({"content": "San"}, {"content": "nu"})),
        httpx.Response(401, json={"error": {"message": "bad key"}}),
    ])
    async with AsyncNAtlas(base_url=BASE) as c:
        c._retry_delay = lambda attempt, resp, _orig=c._retry_delay: (
            None if _orig(attempt, resp) is None else 0)
        assert (await c.chat("hi")).text == "Abuja"
        assert [t async for t in c.stream("hi")] == ["San", "nu"]
        with pytest.raises(AuthenticationError, match="bad key"):
            await c.chat("hi")
