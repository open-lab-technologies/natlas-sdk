"""Clients for any OpenAI-compatible endpoint serving N-ATLAS.

Works against `natlas serve`, vLLM, llama.cpp's server, Ollama or a hosted
N-ATLAS API: anything that speaks /v1/chat/completions. Requests carry only
standard OpenAI fields unless the caller adds others.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from . import defaults
from .errors import APIConnectionError, APITimeoutError, NAtlasError, status_error
from .languages import get_language

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "n-atlas"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 2
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

Message = dict[str, Any]
Prompt = str | Sequence[Message]
Audio = str | Path | bytes

log = logging.getLogger("natlas")


@dataclass
class ChatResponse:
    text: str
    model: str
    finish_reason: str | None
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        return self.text


class _Base:
    """Configuration, request building and response parsing shared by both clients."""

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        model: str | None,
        system_prompt: str | None,
        timeout: float,
        max_retries: int,
    ):
        self.base_url = (base_url or os.getenv("NATLAS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("NATLAS_API_KEY")
        # Servers name the model differently: "n-atlas" for `natlas serve`, the
        # Hugging Face id for vLLM, the tag you pulled for Ollama.
        self.model = model or os.getenv("NATLAS_MODEL") or DEFAULT_MODEL
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _messages(self, prompt: Prompt, system: str | None) -> list[Message]:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
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
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._messages(prompt, system),
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body.update(extra)
        return body

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
    def _parse_chat(data: dict[str, Any], model: str) -> ChatResponse:
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise NAtlasError(f"Unexpected response shape: {data!r}") from e
        return ChatResponse(
            text=text.strip(),
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
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
            detail = (err.get("message") if isinstance(err, dict) else err) or body.get("detail") or body
        raise status_error(resp.status_code, f"{resp.status_code} from {resp.request.url}: {detail}",
                           body)

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
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
    ):
        super().__init__(base_url, api_key, model, system_prompt, timeout, max_retries)
        self._http = http_client or httpx.Client(timeout=timeout)

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
        max_tokens: int | None = 512,
        **extra: Any,
    ) -> ChatResponse:
        """Send a prompt, or a full message list, and return the reply.

        Extra keyword arguments go into the request body unchanged, for options a
        provider supports beyond the OpenAI API, such as `repetition_penalty` on vLLM.
        """
        body = self._chat_body(prompt, system=system, model=model, temperature=temperature,
                               max_tokens=max_tokens, stream=False, extra=extra)
        resp = self._request("POST", "/chat/completions", json=body)
        return self._parse_chat(resp.json(), body["model"])

    def stream(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = defaults.TEMPERATURE,
        max_tokens: int | None = 512,
        **extra: Any,
    ) -> Iterator[str]:
        """Yield the reply piece by piece as the server produces it.

        Only opening the stream is retried; a stream that fails part-way raises.
        """
        body = self._chat_body(prompt, system=system, model=model, temperature=temperature,
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
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
    ):
        super().__init__(base_url, api_key, model, system_prompt, timeout, max_retries)
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

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
        max_tokens: int | None = 512,
        **extra: Any,
    ) -> ChatResponse:
        body = self._chat_body(prompt, system=system, model=model, temperature=temperature,
                               max_tokens=max_tokens, stream=False, extra=extra)
        resp = await self._request("POST", "/chat/completions", json=body)
        return self._parse_chat(resp.json(), body["model"])

    async def stream(
        self,
        prompt: Prompt,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = defaults.TEMPERATURE,
        max_tokens: int | None = 512,
        **extra: Any,
    ) -> AsyncIterator[str]:
        body = self._chat_body(prompt, system=system, model=model, temperature=temperature,
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
