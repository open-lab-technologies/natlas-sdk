"""Clients for any OpenAI-compatible endpoint serving N-ATLAS.

Works against `natlas serve`, vLLM, llama.cpp's server, Ollama or a hosted
N-ATLAS API: anything that speaks /v1/chat/completions. Requests carry only
standard OpenAI fields unless the caller adds others.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx

from . import defaults
from .errors import (
    APIConnectionError,
    APITimeoutError,
    NAtlasError,
    StructuredOutputError,
    ToolLoopError,
    status_error,
)
from .languages import get_language
from .settings import Settings, validate_base_url
from .tools import ToolSpec, to_tool
from .types import ChatResponse, ToolCall, Usage

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

Message = dict[str, Any]
Prompt = str | Sequence[Message]
Audio = str | Path | bytes
Decoder = Callable[[str], Any]

log = logging.getLogger("natlas")


def _tool_call(data: dict[str, Any]) -> ToolCall:
    function = data["function"]
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    return ToolCall(id=data.get("id") or "", name=function["name"], arguments=arguments)


def _invoke(functions: dict[str, Callable[..., Any]], call: ToolCall) -> Any:
    fn = functions.get(call.name)
    if fn is None:
        return {"error": f"Unknown tool {call.name!r}"}
    try:
        return fn(**call.arguments)
    except Exception as e:  # noqa: BLE001  the model gets the error and can recover
        log.info("tool %s raised %s", call.name, e)
        return {"error": f"{type(e).__name__}: {e}"}


def _reject_for_stream(extra: dict[str, Any]) -> None:
    for name in ("tools", "tool_choice", "response_format"):
        if name in extra:
            raise TypeError(f"stream() does not support {name}; use chat()")


def _tool_functions(tools: Sequence[Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
    functions = {}
    for fn in tools:
        if not callable(fn) or isinstance(fn, dict):
            raise TypeError("run_tools needs Python functions, not tool definitions")
        functions[fn.__name__] = fn
    return functions


class _Base:
    """Configuration, request building and response parsing shared by both clients."""

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        model: str | None,
        system_prompt: str | None,
        timeout: float | None,
        max_retries: int | None,
        settings: Settings | None,
    ):
        settings = settings or Settings.from_env()
        self.base_url = validate_base_url(base_url) if base_url else settings.base_url
        self.api_key = api_key or settings.api_key
        # Servers name the model differently: "n-atlas" for `natlas serve`, the
        # Hugging Face id for vLLM, the tag you pulled for Ollama.
        self.model = model or settings.model
        self.system_prompt = system_prompt
        self.timeout = settings.timeout if timeout is None else timeout
        self.max_retries = settings.max_retries if max_retries is None else max_retries
        self._headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _messages(self, prompt: Prompt, system: str | None) -> list[Message]:
        if isinstance(prompt, str):
            messages: list[Message] = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)
        system = system if system is not None else self.system_prompt
        if system and not any(m.get("role") in ("system", "developer") for m in messages):
            messages.insert(0, {"role": "system", "content": system})
        return messages

    def _chat_body(
        self,
        prompt: Prompt,
        *,
        system: str | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        extra: dict[str, Any],
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: Any = None,
    ) -> tuple[dict[str, Any], Decoder | None]:
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._messages(prompt, system),
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = [to_tool(t) for t in tools]
        if tool_choice is not None:
            body["tool_choice"] = (
                {"type": "function", "function": {"name": tool_choice}}
                if tool_choice not in ("auto", "none", "required") and isinstance(tool_choice, str)
                else tool_choice
            )
        decoder = None
        if response_format is not None:
            body["response_format"], decoder = self._response_format(response_format)
        body.update(extra)
        return body, decoder

    @staticmethod
    def _response_format(fmt: Any) -> tuple[dict[str, Any], Decoder | None]:
        """Wire format and decoder for a Pydantic model class, a JSON schema, or an
        OpenAI `response_format` dict."""
        if isinstance(fmt, type) and hasattr(fmt, "model_json_schema"):
            wire = {"type": "json_schema", "json_schema": {
                "name": fmt.__name__, "schema": fmt.model_json_schema(), "strict": True}}
            return wire, fmt.model_validate_json
        if not isinstance(fmt, dict):
            raise TypeError("response_format must be a Pydantic model class or a dict")
        if fmt.get("type") == "text":
            return fmt, None
        if fmt.get("type") in ("json_object", "json_schema"):
            return fmt, json.loads
        wire = {"type": "json_schema", "json_schema": {"name": "response", "schema": fmt}}
        return wire, json.loads

    @staticmethod
    def _translate_prompt(text: str, to: str, source: str) -> str:
        src, dst = get_language(source), get_language(to)
        return (
            f"Translate the following {src.name} text into {dst.name}. "
            f"Reply with the {dst.name} translation only.\n\n{text}"
        )

    @staticmethod
    def _audio_form(audio: Audio, language: str, model: str | None) -> tuple[dict, dict]:
        lang = get_language(language)
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            content, filename = path.read_bytes(), path.name
        else:
            content, filename = audio, "audio"
        data = {"model": model or lang.asr_model, "language": lang.code}
        return data, {"file": (filename, content)}

    @staticmethod
    def _parse_chat(data: dict[str, Any], model: str, decoder: Decoder | None = None,
                    ) -> ChatResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = (message.get("content") or "").strip()
            calls = tuple(_tool_call(c) for c in message.get("tool_calls") or ())
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise NAtlasError(f"Unexpected response shape: {data!r}") from e
        parsed = None
        if decoder and not calls:
            try:
                parsed = decoder(text)
            except ValueError as e:  # json and Pydantic validation errors both subclass it
                raise StructuredOutputError(
                    f"Reply does not match the requested format: {e}", text) from e
        return ChatResponse(
            text=text,
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason"),
            usage=Usage.from_dict(data.get("usage")),
            tool_calls=calls,
            parsed=parsed,
            raw=data,
        )

    @staticmethod
    def _parse_stream_line(line: str) -> str | None:
        """Text delta from one SSE line; None for keep-alives and [DONE]."""
        if not line.startswith("data:"):
            return None
        payload = line[5:].strip()
        if payload == "[DONE]":
            return None
        data = json.loads(payload)
        if "error" in data:
            err = data["error"]
            message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise NAtlasError(f"Error during streaming: {message}")
        choices = data.get("choices") or [{}]
        return choices[0].get("delta", {}).get("content") or ""

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.is_success:
            return
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        detail = body
        if isinstance(body, dict):
            err = body.get("error")
            server_message = err.get("message") if isinstance(err, dict) else err
            detail = server_message or body.get("detail") or body
        raise status_error(resp.status_code,
                           f"{resp.status_code} from {resp.request.url}: {detail}", body)

    def _retry_delay(self, attempt: int, resp: httpx.Response | None) -> float | None:
        """Seconds to wait before retrying, or None to stop."""
        if attempt >= self.max_retries:
            return None
        if resp is not None:
            if resp.status_code not in RETRYABLE_STATUS:
                return None
            retry_after = resp.headers.get("retry-after")
            if retry_after and retry_after.replace(".", "", 1).isdigit():
                return min(float(retry_after), 60.0)
        return min(0.5 * 2**attempt, 8.0) * (0.75 + random.random() / 2)

    @staticmethod
    def _connection_error(e: httpx.HTTPError, base_url: str) -> APIConnectionError:
        if isinstance(e, httpx.TimeoutException):
            return APITimeoutError(f"Request to {base_url} timed out: {e}")
        return APIConnectionError(f"Could not reach {base_url}: {e}")


class NAtlas(_Base):
    """Synchronous client.

    >>> client = NAtlas()  # NATLAS_BASE_URL and NATLAS_API_KEY are read if set
    >>> client.chat("Kí ni olú ìlú Nàìjíríà?").text
    'Olú ìlú Nàìjíríà ni Abuja.'

    Failed connections and 408, 409, 429 and 5xx responses are retried with
    backoff up to `max_retries` times.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        system_prompt: str | None = defaults.SYSTEM_PROMPT,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        settings: Settings | None = None,
    ):
        super().__init__(base_url, api_key, model, system_prompt, timeout, max_retries, settings)
        self._http = http_client or httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> NAtlas:  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url, attempt = f"{self.base_url}{path}", 0
        while True:
            try:
                resp = self._http.request(method, url, headers=self._headers, **kwargs)
            except httpx.HTTPError as e:
                delay = self._retry_delay(attempt, None)
                if delay is None:
                    raise self._connection_error(e, self.base_url) from e
                log.info("%s %s failed (%s), retrying in %.1fs", method, path, e, delay)
            else:
                delay = None if resp.is_success else self._retry_delay(attempt, resp)
                if delay is None:
                    self._raise_for_status(resp)
                    return resp
                log.info("%s %s returned %s, retrying in %.1fs", method, path,
                         resp.status_code, delay)
            time.sleep(delay)
            attempt += 1

    def chat(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = defaults.TEMPERATURE,
        max_tokens: int | None = defaults.MAX_TOKENS,
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: Any = None,
        **extra: Any,
    ) -> ChatResponse:
        """Send a prompt, or a full message list, and return the reply.

        `tools` takes Python functions or OpenAI tool definitions; calls the model
        makes come back in `.tool_calls`. `tool_choice` is "auto", "none",
        "required" or a function name. `response_format` takes a Pydantic model
        class, a JSON schema, or an OpenAI `response_format` dict, and the decoded
        reply comes back in `.parsed`.

        Other keyword arguments go into the request body unchanged, for options a
        provider supports beyond the OpenAI API, such as `repetition_penalty` on vLLM.
        """
        body, decoder = self._chat_body(
            prompt, system=system, model=model, temperature=temperature,
            max_tokens=max_tokens, stream=False, extra=extra, tools=tools,
            tool_choice=tool_choice, response_format=response_format)
        resp = self._request("POST", "/chat/completions", json=body)
        return self._parse_chat(resp.json(), body["model"], decoder)

    def run_tools(
        self,
        prompt: Prompt,
        tools: Sequence[Callable[..., Any]],
        *,
        max_rounds: int = 5,
        **chat_kwargs: Any,
    ) -> ChatResponse:
        """Chat, running each tool the model calls and returning its results, until
        the model answers in text. Returns that final reply.

        An exception raised by a tool is returned to the model as an error result,
        so it can recover or explain; it is not raised here.
        """
        functions = _tool_functions(tools)
        messages = self._messages(prompt, chat_kwargs.pop("system", None))
        for _ in range(max_rounds):
            reply = self.chat(messages, tools=list(tools), **chat_kwargs)
            if not reply.tool_calls:
                return reply
            messages.append(reply.message)
            for call in reply.tool_calls:
                messages.append(call.result(_invoke(functions, call)))
        raise ToolLoopError(f"Model was still calling tools after {max_rounds} rounds")

    def stream(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = defaults.TEMPERATURE,
        max_tokens: int | None = defaults.MAX_TOKENS,
        **extra: Any,
    ) -> Iterator[str]:
        """Yield the reply piece by piece as the server produces it.

        Only opening the stream is retried; a stream that fails part-way raises.
        """
        _reject_for_stream(extra)
        body, _ = self._chat_body(prompt, system=system, model=model, temperature=temperature,
                                  max_tokens=max_tokens, stream=True, extra=extra)
        request = self._http.build_request("POST", f"{self.base_url}/chat/completions",
                                           json=body, headers=self._headers)
        attempt = 0
        while True:
            try:
                resp = self._http.send(request, stream=True)
            except httpx.HTTPError as e:
                delay = self._retry_delay(attempt, None)
                if delay is None:
                    raise self._connection_error(e, self.base_url) from e
            else:
                if resp.is_success:
                    break
                resp.read()
                resp.close()
                delay = self._retry_delay(attempt, resp)
                if delay is None:
                    self._raise_for_status(resp)
            time.sleep(delay)
            attempt += 1
        try:
            for line in resp.iter_lines():
                delta = self._parse_stream_line(line)
                if delta:
                    yield delta
        except httpx.HTTPError as e:
            raise self._connection_error(e, self.base_url) from e
        finally:
            resp.close()

    def translate(self, text: str, *, to: str, source: str = "en", **extra: Any) -> str:
        """Translate between English, Hausa, Igbo and Yoruba."""
        extra.setdefault("temperature", 0.0)
        return self.chat(self._translate_prompt(text, to, source), **extra).text

    def transcribe(self, audio: Audio, *, language: str, model: str | None = None) -> str:
        """Speech to text with the N-ATLAS speech model for `language`.

        `audio` is a file path or the file's bytes. Needs an endpoint with
        /v1/audio/transcriptions, such as `natlas serve --asr`.
        """
        data, files = self._audio_form(audio, language, model)
        resp = self._request("POST", "/audio/transcriptions", data=data, files=files)
        return resp.json()["text"].strip()

    def models(self) -> list[str]:
        resp = self._request("GET", "/models")
        return [m["id"] for m in resp.json().get("data", [])]


class AsyncNAtlas(_Base):
    """Asynchronous client with the same methods as `NAtlas`, as coroutines.

    >>> async with AsyncNAtlas() as client:
    ...     reply = await client.chat("Sannu!")
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        system_prompt: str | None = defaults.SYSTEM_PROMPT,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.AsyncClient | None = None,
        settings: Settings | None = None,
    ):
        super().__init__(base_url, api_key, model, system_prompt, timeout, max_retries, settings)
        self._http = http_client or httpx.AsyncClient(timeout=self.timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncNAtlas:  # noqa: PYI034
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url, attempt = f"{self.base_url}{path}", 0
        while True:
            try:
                resp = await self._http.request(method, url, headers=self._headers, **kwargs)
            except httpx.HTTPError as e:
                delay = self._retry_delay(attempt, None)
                if delay is None:
                    raise self._connection_error(e, self.base_url) from e
                log.info("%s %s failed (%s), retrying in %.1fs", method, path, e, delay)
            else:
                delay = None if resp.is_success else self._retry_delay(attempt, resp)
                if delay is None:
                    self._raise_for_status(resp)
                    return resp
                log.info("%s %s returned %s, retrying in %.1fs", method, path,
                         resp.status_code, delay)
            await asyncio.sleep(delay)
            attempt += 1

    async def chat(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = defaults.TEMPERATURE,
        max_tokens: int | None = defaults.MAX_TOKENS,
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: Any = None,
        **extra: Any,
    ) -> ChatResponse:
        body, decoder = self._chat_body(
            prompt, system=system, model=model, temperature=temperature,
            max_tokens=max_tokens, stream=False, extra=extra, tools=tools,
            tool_choice=tool_choice, response_format=response_format)
        resp = await self._request("POST", "/chat/completions", json=body)
        return self._parse_chat(resp.json(), body["model"], decoder)

    async def run_tools(
        self,
        prompt: Prompt,
        tools: Sequence[Callable[..., Any]],
        *,
        max_rounds: int = 5,
        **chat_kwargs: Any,
    ) -> ChatResponse:
        """As `NAtlas.run_tools`. Tools may be plain functions or coroutine functions."""
        functions = _tool_functions(tools)
        messages = self._messages(prompt, chat_kwargs.pop("system", None))
        for _ in range(max_rounds):
            reply = await self.chat(messages, tools=list(tools), **chat_kwargs)
            if not reply.tool_calls:
                return reply
            messages.append(reply.message)
            for call in reply.tool_calls:
                result = _invoke(functions, call)
                if asyncio.iscoroutine(result):
                    try:
                        result = await result
                    except Exception as e:  # noqa: BLE001  returned to the model, as in _invoke
                        result = {"error": f"{type(e).__name__}: {e}"}
                messages.append(call.result(result))
        raise ToolLoopError(f"Model was still calling tools after {max_rounds} rounds")

    async def stream(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = defaults.TEMPERATURE,
        max_tokens: int | None = defaults.MAX_TOKENS,
        **extra: Any,
    ) -> AsyncIterator[str]:
        _reject_for_stream(extra)
        body, _ = self._chat_body(prompt, system=system, model=model, temperature=temperature,
                                  max_tokens=max_tokens, stream=True, extra=extra)
        request = self._http.build_request("POST", f"{self.base_url}/chat/completions",
                                           json=body, headers=self._headers)
        attempt = 0
        while True:
            try:
                resp = await self._http.send(request, stream=True)
            except httpx.HTTPError as e:
                delay = self._retry_delay(attempt, None)
                if delay is None:
                    raise self._connection_error(e, self.base_url) from e
            else:
                if resp.is_success:
                    break
                await resp.aread()
                await resp.aclose()
                delay = self._retry_delay(attempt, resp)
                if delay is None:
                    self._raise_for_status(resp)
            await asyncio.sleep(delay)
            attempt += 1
        try:
            async for line in resp.aiter_lines():
                delta = self._parse_stream_line(line)
                if delta:
                    yield delta
        except httpx.HTTPError as e:
            raise self._connection_error(e, self.base_url) from e
        finally:
            await resp.aclose()

    async def translate(self, text: str, *, to: str, source: str = "en", **extra: Any) -> str:
        extra.setdefault("temperature", 0.0)
        return (await self.chat(self._translate_prompt(text, to, source), **extra)).text

    async def transcribe(self, audio: Audio, *, language: str, model: str | None = None) -> str:
        data, files = self._audio_form(audio, language, model)
        resp = await self._request("POST", "/audio/transcriptions", data=data, files=files)
        return resp.json()["text"].strip()

    async def models(self) -> list[str]:
        resp = await self._request("GET", "/models")
        return [m["id"] for m in resp.json().get("data", [])]
