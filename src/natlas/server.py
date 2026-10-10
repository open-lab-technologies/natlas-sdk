"""OpenAI-compatible HTTP server in front of an `Engine` and, optionally, a `Transcriber`.

Any OpenAI client can point at it by changing the base URL. The N-ATLAS system
prompt and sampling settings apply when a request does not set its own, so
drop-in clients get the behaviour the model was tuned for.
"""

from __future__ import annotations

import hmac
import itertools
import json
import time
import uuid
from typing import Any, Literal

from ._extras import require

require("fastapi", "server")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from . import __version__, defaults
from .asr import Transcriber
from .engine import Engine
from .languages import LANGUAGES, get_language

MODEL_ID = "n-atlas"


class ChatMessage(BaseModel):
    # "developer" is what newer OpenAI SDKs send in place of "system".
    role: Literal["system", "developer", "user", "assistant"]
    # OpenAI clients may send content as a list of typed parts instead of a string.
    content: str | list[dict[str, Any]]

    @field_validator("content")
    @classmethod
    def _flatten(cls, value: str | list[dict[str, Any]]) -> str:
        if isinstance(value, str):
            return value
        parts = []
        for part in value:
            if part.get("type") != "text":
                raise ValueError(f"Unsupported content part {part.get('type')!r}; N-ATLAS is text-only")
            parts.append(part.get("text", ""))
        return "\n".join(parts)

    def as_dict(self) -> dict[str, str]:
        return {"role": "system" if self.role == "developer" else self.role,
                "content": self.content}


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    temperature: float = defaults.TEMPERATURE
    top_p: float = 0.95
    max_tokens: int | None = 512
    repetition_penalty: float = defaults.REPETITION_PENALTY
    stop: str | list[str] | None = None
    stream: bool = False


def _error(status: int, message: str) -> JSONResponse:
    """OpenAI's error body, so OpenAI SDKs surface the message instead of a raw payload."""
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status)


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

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(400, f"{where}: {first.get('msg')}" if where else first.get("msg", ""))

    @app.middleware("http")
    async def require_key(request: Request, call_next):
        # /health stays open so load balancers and scripts can probe the server.
        if api_key and request.url.path != "/health":
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not hmac.compare_digest(supplied, api_key):
                return _error(401, "Missing or invalid API key")
        return await call_next(request)

    def start(messages: list[dict[str, str]], params: dict[str, Any]):
        """Begin generating, turning a prompt that cannot fit into a 400.

        The first chunk is pulled before any response is sent, so an over-long
        prompt fails cleanly even on a streaming request.
        """
        stream = engine.generate(messages, **params)
        try:
            first = next(stream, None)
        except ValueError as e:  # llama.cpp: "Requested tokens ... exceed context window"
            raise HTTPException(400, str(e)) from e
        return first, stream

    def prepare(req: ChatRequest) -> list[dict[str, str]]:
        if not req.messages:
            raise HTTPException(400, "messages must not be empty")
        messages = [m.as_dict() for m in req.messages]
        if inject_system_prompt and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": defaults.SYSTEM_PROMPT})
        return messages

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": engine.name, "asr": asr is not None}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        data = [{"id": MODEL_ID, "object": "model", "owned_by": "NCAIR1"}]
        if asr is not None:
            data += [{"id": l.asr_model, "object": "model", "owned_by": "NCAIR1"}
                     for l in LANGUAGES.values()]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        messages = prepare(req)
        stop = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
        params = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens or 512,
            "repetition_penalty": req.repetition_penalty,
            "stop": stop,
        }
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        first, rest = start(messages, params)
        chunks = itertools.chain([first] if first else [], rest)

        if req.stream:
            def events():
                def frame(delta: dict, finish: str | None) -> str:
                    body = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

                yield frame({"role": "assistant"}, None)
                finish = "stop"
                for chunk in chunks:
                    if chunk.text:
                        yield frame({"content": chunk.text}, None)
                    finish = chunk.finish_reason or finish
                yield frame({}, finish)
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        parts, finish = [], "stop"
        for chunk in chunks:
            parts.append(chunk.text)
            finish = chunk.finish_reason or finish
        text = "".join(parts).strip()
        prompt_tokens = engine.count_prompt_tokens(messages)
        completion_tokens = engine.count_tokens(text)
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        model: str | None = Form(None),
        response_format: str = Form("json"),
    ):
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
        except Exception as e:  # decoding errors from librosa/ffmpeg surface here
            raise HTTPException(422, f"Could not transcribe audio: {e}") from e
        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text, "language": lang.code})

    return app
