"""OpenAI-compatible HTTP server in front of an `Engine` and, optionally, a `Transcriber`.

Any OpenAI client can point at it by changing the base URL. The N-ATLAS system
prompt and sampling settings apply when a request does not set its own, so
drop-in clients get the behaviour the model was tuned for.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, Literal

from ._extras import require

require("fastapi", "server")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, field_validator

from . import __version__, defaults
from ._toolcalls import (
    AUTO_INSTRUCTION,
    ToolChoiceError,
    call_schema,
    parse_call,
    to_template_messages,
)
from .asr import Transcriber
from .engine import Chunk, Engine
from .languages import LANGUAGES, get_language

MODEL_ID = "n-atlas"


class ChatMessage(BaseModel):
    # "developer" is what newer OpenAI SDKs send in place of "system".
    role: Literal["system", "developer", "user", "assistant", "tool"]
    # OpenAI clients may send content as a list of typed parts instead of a
    # string, and assistant messages that call a tool may have no content.
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    @field_validator("content")
    @classmethod
    def _flatten(cls, value: str | list[dict[str, Any]] | None) -> str | None:
        if value is None or isinstance(value, str):
            return value
        parts = []
        for part in value:
            if part.get("type") != "text":
                raise ValueError(
                    f"Unsupported content part {part.get('type')!r}; N-ATLAS is text-only")
            parts.append(part.get("text", ""))
        return "\n".join(parts)

    def as_dict(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "system" if self.role == "developer" else self.role,
            "content": self.content or "",
        }
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    temperature: float = defaults.TEMPERATURE
    top_p: float = 0.95
    max_tokens: int | None = defaults.MAX_TOKENS
    repetition_penalty: float = defaults.REPETITION_PENALTY
    stop: str | list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


def _json_schema(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response_format or response_format.get("type") == "text":
        return None
    if response_format.get("type") == "json_object":
        return {"type": "object"}
    if response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise HTTPException(400, "response_format.json_schema.schema must be a JSON schema")
        return schema
    raise HTTPException(400, f"Unsupported response_format type {response_format.get('type')!r}")


def _error(status: int, message: str) -> JSONResponse:
    """OpenAI's error body, so OpenAI SDKs surface the message instead of a raw payload."""
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status)


def _completion(completion_id: str, created: int, message: dict[str, Any], finish: str,
                prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _sse(body: dict[str, Any]) -> str:
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def _chunk(completion_id: str, created: int, delta: dict[str, Any], finish: str | None) -> str:
    return _sse({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                 "model": MODEL_ID,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})


async def _tool_call_events(completion_id: str, created: int,
                            tool_call: dict[str, Any]) -> AsyncIterator[str]:
    yield _chunk(completion_id, created, {"role": "assistant", "content": None}, None)
    yield _chunk(completion_id, created, {"tool_calls": [{"index": 0, **tool_call}]}, None)
    yield _chunk(completion_id, created, {}, "tool_calls")
    yield "data: [DONE]\n\n"


async def _text_events(completion_id: str, created: int, generation: _Generation,
                       first: Chunk | None) -> AsyncIterator[str]:
    finish, chunk = "stop", first
    try:
        yield _chunk(completion_id, created, {"role": "assistant"}, None)
        while chunk is not None:
            if chunk.text:
                yield _chunk(completion_id, created, {"content": chunk.text}, None)
            finish = chunk.finish_reason or finish
            chunk = await generation.next()
        yield _chunk(completion_id, created, {}, finish)
    except Exception as e:  # noqa: BLE001  headers are sent; report it in the stream
        yield _sse({"error": {"message": str(e), "type": "server_error"}})
    finally:
        # Runs when the client hangs up too, so the engine is released.
        generation.stop()
    yield "data: [DONE]\n\n"


class _Generation:
    """Runs an engine generator on a worker thread and hands chunks to the event loop.

    Engines hold a lock while generating. If a client hangs up mid-stream and the
    generator is simply abandoned, that lock is never released and every later
    request waits forever. `stop()` makes the worker stop at the next chunk and
    close the generator on its own thread, which releases the engine.
    """

    _DONE = object()

    def __init__(self, chunks: Iterator[Chunk]):
        self._chunks = chunks
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._stopped = threading.Event()
        threading.Thread(target=self._work, name="pynatlas-generate", daemon=True).start()

    def _put(self, item: Any) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:  # event loop already closed during shutdown
            pass

    def _work(self) -> None:
        try:
            for chunk in self._chunks:
                if self._stopped.is_set():
                    break
                self._put(chunk)
        except Exception as e:  # noqa: BLE001  any engine failure goes to the caller
            self._put(e)
        finally:
            close = getattr(self._chunks, "close", None)
            if close:
                close()
            self._put(self._DONE)

    async def next(self) -> Chunk | None:
        """The next chunk, None when generation has finished, or the engine's error."""
        item = await self._queue.get()
        if item is self._DONE:
            return None
        if isinstance(item, Exception):
            raise item
        chunk: Chunk = item
        return chunk

    def stop(self) -> None:
        self._stopped.set()


def create_app(
    engine: Engine,
    asr: Transcriber | None = None,
    inject_system_prompt: bool = True,
    api_key: str | None = None,
) -> FastAPI:
    app = FastAPI(title="N-ATLAS", version=__version__)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        return _error(500, f"{type(exc).__name__}: {exc}")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(400, f"{where}: {first.get('msg')}" if where else first.get("msg", ""))

    @app.middleware("http")
    async def require_key(request: Request,
                          call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # /health stays open so load balancers and scripts can probe the server.
        if api_key and request.url.path != "/health":
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(supplied, api_key):
                return _error(401, "Missing or invalid API key")
        return await call_next(request)

    def prepare(req: ChatRequest) -> list[dict[str, Any]]:
        if not req.messages:
            raise HTTPException(400, "messages must not be empty")
        try:
            messages = to_template_messages([m.as_dict() for m in req.messages])
        except (KeyError, ValueError) as e:
            raise HTTPException(400, f"Malformed tool call in messages: {e}") from e
        if inject_system_prompt and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": defaults.SYSTEM_PROMPT})
        return messages

    async def start(chunks: Iterator[Chunk]) -> tuple[_Generation, Chunk | None]:
        """Begin a generation and pull its first chunk before any response is sent,
        so a prompt that cannot fit or a schema that cannot be enforced is a 400
        even on a streaming request."""
        generation = _Generation(chunks)
        try:
            return generation, await generation.next()
        except ValueError as e:
            generation.stop()
            raise HTTPException(400, str(e)) from e
        except BaseException:
            generation.stop()
            raise

    async def collect(generation: _Generation, first: Chunk | None) -> tuple[str, str]:
        parts, finish, chunk = [], "stop", first
        try:
            while chunk is not None:
                parts.append(chunk.text)
                finish = chunk.finish_reason or finish
                chunk = await generation.next()
        finally:
            generation.stop()
        return "".join(parts).strip(), finish

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": engine.name, "asr": asr is not None,
                "attribution": defaults.ATTRIBUTION}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        data = [{"id": MODEL_ID, "object": "model", "owned_by": "NCAIR1"}]
        if asr is not None:
            data += [{"id": l.asr_model, "object": "model", "owned_by": "NCAIR1"}
                     for l in LANGUAGES.values()]
        return {"object": "list", "data": data}

    async def decide_tool(req: ChatRequest, messages: list[dict[str, Any]],
                          params: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        try:
            schema, allow_none = call_schema(req.tools or [], req.tool_choice)
        except ToolChoiceError as e:
            raise HTTPException(400, str(e)) from e
        if allow_none:
            messages = [{**messages[0], "content": messages[0]["content"] + AUTO_INSTRUCTION},
                        *messages[1:]]
        generation, first = await start(engine.generate(
            messages, **params, tools=req.tools, json_schema=schema))
        decision, _ = await collect(generation, first)
        try:
            return parse_call(decision)
        except (ValueError, KeyError) as e:  # reply was cut off by max_tokens
            raise HTTPException(400, f"Tool call was incomplete; raise max_tokens ({e})") from e

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(req: ChatRequest) -> Response | dict[str, Any]:
        messages = prepare(req)
        params: dict[str, Any] = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens or defaults.MAX_TOKENS,
            "repetition_penalty": req.repetition_penalty,
            "stop": [req.stop] if isinstance(req.stop, str) else (req.stop or []),
        }
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        answer_schema = _json_schema(req.response_format)

        call = None
        if req.tools and req.tool_choice != "none":
            call = await decide_tool(req, messages, params)
        if call is not None:
            name, arguments = call
            arguments_json = json.dumps(arguments, ensure_ascii=False)
            tool_call = {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                         "function": {"name": name, "arguments": arguments_json}}
            if req.stream:
                return StreamingResponse(_tool_call_events(completion_id, created, tool_call),
                                         media_type="text/event-stream")
            return _completion(completion_id, created,
                               {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                               "tool_calls", engine.count_prompt_tokens(messages),
                               engine.count_tokens(arguments_json))

        generation, first = await start(engine.generate(messages, **params,
                                                         json_schema=answer_schema))
        if req.stream:
            return StreamingResponse(_text_events(completion_id, created, generation, first),
                                     media_type="text/event-stream")
        text, finish = await collect(generation, first)
        return _completion(completion_id, created, {"role": "assistant", "content": text}, finish,
                           engine.count_prompt_tokens(messages), engine.count_tokens(text))

    @app.post("/v1/audio/transcriptions", response_model=None)
    def transcriptions(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        model: str | None = Form(None),
        response_format: str = Form("json"),
    ) -> Response:
        if asr is None:
            raise HTTPException(501, "Speech recognition is off. Start the server with --asr.")
        # The OpenAI SDK sends `model` but `language` is optional there, so the
        # language can also come from naming one of the N-ATLAS speech models.
        if language is None and model:
            language = next((l.code for l in LANGUAGES.values() if l.asr_model == model), None)
        if language is None:
            names = ", ".join(l.asr_model for l in LANGUAGES.values())
            raise HTTPException(400, f"Set language (en, ha, ig, yo) or model to one of {names}")
        try:
            lang = get_language(language)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        data = file.file.read()
        if not data:
            raise HTTPException(400, "Empty audio file")
        try:
            text = asr.transcribe(data, lang.code)
        except Exception as e:  # undecodable audio surfaces here in many exception types
            raise HTTPException(422, f"Could not transcribe audio: {e}") from e
        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text, "language": lang.code})

    return app
