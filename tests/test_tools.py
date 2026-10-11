import enum
import json
from typing import Literal

import httpx
import pytest
import respx

from natlas import (
    AsyncNAtlas,
    NAtlas,
    StructuredOutputError,
    ToolCall,
    ToolLoopError,
    function_tool,
)
from natlas._toolcalls import NO_TOOL, call_schema, parse_call, to_template_messages

BASE = "http://test/v1"
CHAT = f"{BASE}/chat/completions"


class Unit(enum.Enum):
    C = "celsius"
    F = "fahrenheit"


def get_weather(city: str, unit: Unit = Unit.C, days: int | None = None) -> dict:
    """Current weather for a city.

    Args:
        city: City name, for example Kano.
        days (int): Forecast length.
    """
    return {"city": city, "temp_c": 34}


def reply(content=None, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={"model": "n-atlas", "choices": [
        {"message": message, "finish_reason": finish}]})


def call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


# function_tool

def test_function_tool_builds_schema_from_signature_and_docstring():
    spec = function_tool(get_weather)["function"]
    assert spec["name"] == "get_weather"
    assert spec["description"] == "Current weather for a city."
    props = spec["parameters"]["properties"]
    assert props["city"] == {"type": "string", "description": "City name, for example Kano."}
    assert props["unit"] == {"enum": ["celsius", "fahrenheit"]}
    assert props["days"] == {"type": "integer", "description": "Forecast length."}
    assert spec["parameters"]["required"] == ["city"]


def test_function_tool_handles_literal_lists_and_pydantic():
    pydantic = pytest.importorskip("pydantic")

    class Item(pydantic.BaseModel):
        name: str

    def order(items: list[Item], speed: Literal["fast", "slow"], notes: dict) -> None:
        """Place an order."""

    props = function_tool(order)["function"]["parameters"]["properties"]
    assert props["items"]["type"] == "array"
    assert props["items"]["items"]["properties"]["name"]["type"] == "string"
    assert props["speed"] == {"enum": ["fast", "slow"]}
    assert props["notes"] == {"type": "object"}


def test_function_tool_rejects_undescribable_types():
    def bad(x: complex) -> None:
        """Nope."""

    with pytest.raises(TypeError, match="complex"):
        function_tool(bad)


# client: tools and structured output

@respx.mock
def test_tools_sent_and_tool_calls_parsed():
    route = respx.post(CHAT).mock(return_value=reply(
        tool_calls=[call("get_weather", {"city": "Kano"})], finish="tool_calls"))
    r = NAtlas(base_url=BASE).chat("Weather in Kano?", tools=[get_weather], tool_choice="get_weather")
    assert r.tool_calls == (ToolCall("call_1", "get_weather", {"city": "Kano"}),)
    assert r.text == ""
    sent = json.loads(route.calls[0].request.content)
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_tool_call_round_trip_messages():
    c = ToolCall("call_9", "get_weather", {"city": "Kano"})
    assert c.result({"temp_c": 34}) == {"role": "tool", "tool_call_id": "call_9",
                                         "content": '{"temp_c": 34}'}


@respx.mock
def test_response_format_with_pydantic_model():
    pydantic = pytest.importorskip("pydantic")

    class Analysis(pydantic.BaseModel):
        language: str
        sentiment: Literal["positive", "negative"]

    route = respx.post(CHAT).mock(return_value=reply('{"language": "ha", "sentiment": "positive"}'))
    r = NAtlas(base_url=BASE).chat("Analyse", response_format=Analysis)
    assert r.parsed == Analysis(language="ha", sentiment="positive")
    fmt = json.loads(route.calls[0].request.content)["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["properties"]["sentiment"]["enum"] == ["positive", "negative"]


@respx.mock
def test_response_format_with_plain_schema_and_json_object():
    respx.post(CHAT).mock(return_value=reply('{"n": 3}'))
    c = NAtlas(base_url=BASE)
    assert c.chat("x", response_format={"type": "object"}).parsed == {"n": 3}
    assert c.chat("x", response_format={"type": "json_object"}).parsed == {"n": 3}


@respx.mock
def test_reply_not_matching_format_raises_with_text():
    pydantic = pytest.importorskip("pydantic")

    class Strict(pydantic.BaseModel):
        n: int

    respx.post(CHAT).mock(return_value=reply('{"n": "three"}'))
    with pytest.raises(StructuredOutputError) as err:
        NAtlas(base_url=BASE).chat("x", response_format=Strict)
    assert err.value.text == '{"n": "three"}'


def test_stream_rejects_tools_and_formats():
    with pytest.raises(TypeError, match="use chat"):
        list(NAtlas(base_url=BASE).stream("x", tools=[get_weather]))


# client: run_tools

@respx.mock
def test_run_tools_executes_calls_until_text_answer():
    route = respx.post(CHAT).mock(side_effect=[
        reply(tool_calls=[call("get_weather", {"city": "Kano"})], finish="tool_calls"),
        reply("It is 34 degrees in Kano."),
    ])
    r = NAtlas(base_url=BASE).run_tools("Weather in Kano?", [get_weather])
    assert r.text == "It is 34 degrees in Kano."
    history = json.loads(route.calls[1].request.content)["messages"]
    assert history[-2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert history[-1] == {"role": "tool", "tool_call_id": "call_1",
                           "content": '{"city": "Kano", "temp_c": 34}'}


@respx.mock
def test_run_tools_returns_tool_errors_to_the_model():
    def broken(city: str) -> None:
        """Always fails."""
        raise RuntimeError("service down")

    route = respx.post(CHAT).mock(side_effect=[
        reply(tool_calls=[call("broken", {"city": "Kano"})], finish="tool_calls"),
        reply(tool_calls=[call("missing", {}, "call_2")], finish="tool_calls"),
        reply("Sorry, the weather service is down."),
    ])
    NAtlas(base_url=BASE).run_tools("Weather?", [broken])
    second = json.loads(route.calls[1].request.content)["messages"][-1]
    third = json.loads(route.calls[2].request.content)["messages"][-1]
    assert json.loads(second["content"]) == {"error": "RuntimeError: service down"}
    assert json.loads(third["content"]) == {"error": "Unknown tool 'missing'"}


@respx.mock
def test_run_tools_stops_after_max_rounds():
    respx.post(CHAT).mock(return_value=reply(
        tool_calls=[call("get_weather", {"city": "Kano"})], finish="tool_calls"))
    with pytest.raises(ToolLoopError):
        NAtlas(base_url=BASE).run_tools("x", [get_weather], max_rounds=2)


@respx.mock
async def test_async_run_tools_awaits_coroutine_tools():
    async def lookup(city: str) -> dict:
        """Async lookup."""
        return {"city": city}

    respx.post(CHAT).mock(side_effect=[
        reply(tool_calls=[call("lookup", {"city": "Abuja"})], finish="tool_calls"),
        reply("Done."),
    ])
    async with AsyncNAtlas(base_url=BASE) as c:
        assert (await c.run_tools("x", [lookup])).text == "Done."


# server-side translation

def test_call_schema_modes():
    tools = [function_tool(get_weather)]
    auto, allow_none = call_schema(tools, "auto")
    assert allow_none and auto["anyOf"][-1]["properties"]["name"]["const"] == NO_TOOL
    required, allow_none = call_schema(tools, "required")
    assert not allow_none and len(required["anyOf"]) == 1
    with pytest.raises(ValueError, match="not in tools"):
        call_schema(tools, {"type": "function", "function": {"name": "nope"}})


def test_parse_call():
    assert parse_call('{"name": "get_weather", "parameters": {"city": "Kano"}}') == (
        "get_weather", {"city": "Kano"})
    assert parse_call('{"name": "none", "parameters": {}}') is None


def test_template_messages_split_parallel_calls_and_decode_arguments():
    out = to_template_messages([
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None,
         "tool_calls": [call("a", {"k": 1}), call("b", {}, "call_2")]},
        {"role": "tool", "content": "1"},
    ])
    assert [m["role"] for m in out] == ["user", "assistant", "assistant", "tool"]
    assert out[1]["tool_calls"][0]["function"] == {"name": "a", "arguments": {"k": 1}}
