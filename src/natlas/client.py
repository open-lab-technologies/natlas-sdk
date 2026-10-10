"""HTTP client for any OpenAI-compatible endpoint serving N-ATLAS.

Works against `natlas serve`, vLLM, llama.cpp's server, Ollama or a hosted
N-ATLAS API: anything that speaks /v1/chat/completions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .languages import get_language

# The system prompt and sampling settings from the N-ATLAS model card. The model
# was tuned with this prompt; without it, answers drift towards generic Llama-3.
DEFAULT_SYSTEM_PROMPT = (
    "you are a large language model trained by Awarri AI technologies. "
    "You are a friendly assistant and you are here to help."
)
DEFAULT_TEMPERATURE = 0.1
DEFAULT_REPETITION_PENALTY = 1.12
DEFAULT_BASE_URL = "http://localhost:8000/v1"

Message = dict[str, str]


class NAtlasError(Exception):
    """Raised when the endpoint returns an error or an unusable response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ChatResponse:
    text: str
    model: str
    finish_reason: str | None
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        return self.text


class NAtlas:
    """Synchronous N-ATLAS client.

    >>> client = NAtlas()
    >>> client.chat("Kí ni olú ìlú Nàìjíríà?").text
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "n-atlas",
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = (base_url or os.getenv("NATLAS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("NATLAS_API_KEY")
        self.model = model
        self.system_prompt = system_prompt
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._http = http_client or httpx.Client(timeout=timeout)
        self._headers = headers

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> NAtlas:  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _messages(self, prompt: str | Sequence[Message], system: str | None) -> list[Message]:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        system = system if system is not None else self.system_prompt
        if system and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system})
        return messages

    def _payload(
        self,
        prompt: str | Sequence[Message],
        system: str | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": self._messages(prompt, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repetition_penalty": DEFAULT_REPETITION_PENALTY,
            "stream": stream,
            **extra,
        }

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.is_success:
            return
        try:
            detail = resp.json()
            detail = detail.get("error", {}).get("message") or detail.get("detail") or detail
        except ValueError:
            detail = resp.text
        raise NAtlasError(f"{resp.status_code} from {resp.request.url}: {detail}", resp.status_code)

    def chat(
        self,
        prompt: str | Sequence[Message],
        *,
        system: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 512,
        **extra: Any,
    ) -> ChatResponse:
        """Send a prompt (or a full message list) and return the reply."""
        payload = self._payload(prompt, system, temperature, max_tokens, False, extra)
        try:
            resp = self._http.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self._headers
            )
        except httpx.HTTPError as e:
            raise NAtlasError(f"Could not reach {self.base_url}: {e}") from e
        self._raise_for_status(resp)
        data = resp.json()
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise NAtlasError(f"Unexpected response shape: {data!r}") from e
        return ChatResponse(
            text=text.strip(),
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
            raw=data,
        )

    def stream(
        self,
        prompt: str | Sequence[Message],
        *,
        system: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 512,
        **extra: Any,
    ) -> Iterator[str]:
        """Yield the reply token by token as the server produces it."""
        payload = self._payload(prompt, system, temperature, max_tokens, True, extra)
        try:
            with self._http.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers
            ) as resp:
                if not resp.is_success:
                    resp.read()
                    self._raise_for_status(resp)
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as e:
            raise NAtlasError(f"Could not reach {self.base_url}: {e}") from e

    def translate(self, text: str, *, to: str, source: str = "en", **extra: Any) -> str:
        """Translate between English, Hausa, Igbo and Yoruba."""
        src, dst = get_language(source), get_language(to)
        prompt = (
            f"Translate the following {src.name} text into {dst.name}. "
            f"Reply with the {dst.name} translation only.\n\n{text}"
        )
        return self.chat(prompt, temperature=0.0, **extra).text

    def transcribe(self, audio: str | Path | bytes, *, language: str) -> str:
        """Speech to text with the official N-ATLAS speech model for `language`.

        Needs a server that exposes /v1/audio/transcriptions, such as `natlas serve --asr`.
        """
        lang = get_language(language)
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            content, filename = path.read_bytes(), path.name
        else:
            content, filename = audio, "audio.wav"
        try:
            resp = self._http.post(
                f"{self.base_url}/audio/transcriptions",
                data={"model": lang.asr_model, "language": lang.code},
                files={"file": (filename, content)},
                headers=self._headers,
            )
        except httpx.HTTPError as e:
            raise NAtlasError(f"Could not reach {self.base_url}: {e}") from e
        self._raise_for_status(resp)
        return resp.json()["text"].strip()

    def models(self) -> list[str]:
        resp = self._http.get(f"{self.base_url}/models", headers=self._headers)
        self._raise_for_status(resp)
        return [m["id"] for m in resp.json().get("data", [])]
