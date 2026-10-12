import json

import httpx
import pytest
import respx

from pynatlas import (
    AsyncNAtlas,
    NAtlas,
    NAtlasError,
    StructuredOutputError,
    VoiceReply,
    detect_language,
)

BASE = "http://test/v1"
CHAT = f"{BASE}/chat/completions"
AUDIO = f"{BASE}/audio/transcriptions"


def reply(content):
    return httpx.Response(200, json={"model": "n-atlas", "choices": [
        {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})


def sent(route, i=0):
    return json.loads(route.calls[i].request.content)


@pytest.mark.parametrize(("text", "code"), [
    ("Sannu, yaya kake?", "ha"),
    ("Kedu ka ị mere?", "ig"),
    ("Báwo ni o ṣe wà?", "yo"),
    ("Bawo ni o se wa?", "yo"),  # typed without tone marks
    ("How are you today?", "en"),
    ("", None),
    ("12345 !!", None),
])
def test_detect_language(text, code):
    assert detect_language(text) == code


@respx.mock
def test_summarize_answers_in_the_detected_language():
    route = respx.post(CHAT).mock(return_value=reply("Takaitaccen bayani."))
    assert NAtlas(base_url=BASE).summarize("Gwamnati ta sanar da sabon tsari a yau.") == (
        "Takaitaccen bayani.")
    assert "in Hausa" in sent(route)["messages"][-1]["content"]


@respx.mock
def test_classify_enforces_the_label_set():
    route = respx.post(CHAT).mock(return_value=reply('{"label": "complaint"}'))
    label = NAtlas(base_url=BASE).classify("Kayan sun lalace.", ["complaint", "praise"])
    assert label == "complaint"
    schema = sent(route)["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["label"]["enum"] == ["complaint", "praise"]


@respx.mock
def test_classify_rejects_a_label_it_did_not_offer():
    respx.post(CHAT).mock(return_value=reply('{"label": "spam"}'))
    with pytest.raises(StructuredOutputError, match="spam"):
        NAtlas(base_url=BASE).classify("x", ["complaint", "praise"])


def test_classify_needs_two_labels():
    with pytest.raises(ValueError, match="two distinct"):
        NAtlas(base_url=BASE).classify("x", ["only"])


@respx.mock
def test_fenced_json_is_recovered_without_a_retry():
    route = respx.post(CHAT).mock(return_value=reply('Here you go:\n```json\n{"n": 3}\n```'))
    assert NAtlas(base_url=BASE).chat("x", response_format={"type": "object"}).parsed == {"n": 3}
    assert route.call_count == 1


@respx.mock
def test_mismatch_is_retried_once_with_the_error_shown():
    route = respx.post(CHAT).mock(side_effect=[reply("no json here"), reply('{"n": 3}')])
    r = NAtlas(base_url=BASE).chat("x", response_format={"type": "object"})
    assert r.parsed == {"n": 3}
    retry = sent(route, 1)["messages"]
    assert retry[-2] == {"role": "assistant", "content": "no json here"}
    assert "did not match the required format" in retry[-1]["content"]


@respx.mock
def test_format_retries_zero_raises_immediately():
    route = respx.post(CHAT).mock(return_value=reply("no json"))
    with pytest.raises(StructuredOutputError):
        NAtlas(base_url=BASE).chat("x", response_format={"type": "object"}, format_retries=0)
    assert route.call_count == 1


@respx.mock
def test_voice_chat_transcribes_then_answers_in_that_language():
    respx.post(AUDIO).mock(return_value=httpx.Response(200, json={"text": "Sannu"}))
    route = respx.post(CHAT).mock(return_value=reply("Sannu da zuwa!"))
    v = NAtlas(base_url=BASE).voice_chat(b"audio", language="ha")
    assert v == VoiceReply("Sannu", "ha", v.reply)
    assert str(v) == "Sannu da zuwa!"
    assert "Reply in Hausa" in sent(route)["messages"][0]["content"]


@respx.mock
def test_voice_chat_refuses_to_answer_silence():
    respx.post(AUDIO).mock(return_value=httpx.Response(200, json={"text": "  "}))
    with pytest.raises(NAtlasError, match="Nothing was heard"):
        NAtlas(base_url=BASE).voice_chat(b"audio", language="yo")


@respx.mock
async def test_async_helpers():
    respx.post(CHAT).mock(side_effect=[reply("Summary."), reply('{"label": "praise"}')])
    async with AsyncNAtlas(base_url=BASE) as c:
        assert await c.summarize("The market opened early today.") == "Summary."
        assert await c.classify("Great service", ["complaint", "praise"]) == "praise"
