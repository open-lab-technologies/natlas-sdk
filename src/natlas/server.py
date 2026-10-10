"""OpenAI-compatible HTTP server for a local N-ATLAS model.

Any OpenAI client, LangChain, LlamaIndex or plain curl can point at it by
changing the base URL. N-ATLAS sampling defaults and system prompt are applied
when the caller does not set them, so drop-in clients get the tuned behaviour.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .client import DEFAULT_REPETITION_PENALTY, DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE
from .languages import LANGUAGES, get_language

MODEL_ID = "n-atlas"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = 0.95
    max_tokens: int | None = 512
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    stop: str | list[str] | None = None
    stream: bool = False


def create_app(engine: Any, asr: Any | None = None, inject_system_prompt: bool = True) -> FastAPI:
    app = FastAPI(title="N-ATLAS", version="0.1.0")

    def prepare(req: ChatRequest) -> list[dict[str, str]]:
        if not req.messages:
            raise HTTPException(400, "messages must not be empty")
        messages = [m.model_dump() for m in req.messages]
        if inject_system_prompt and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        return messages

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": engine.model_path.name, "asr": asr is not None}

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
                for chunk in engine.generate(messages, **params):
                    if chunk.text:
                        yield frame({"content": chunk.text}, None)
                    finish = chunk.finish_reason or finish
                yield frame({}, finish)
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        parts, finish = [], "stop"
        for chunk in engine.generate(messages, **params):
            parts.append(chunk.text)
            finish = chunk.finish_reason or finish
        text = "".join(parts).strip()
        prompt_tokens = engine.count_tokens(engine.render(messages))
        completion_tokens = engine.count_tokens(text) - 1
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
        language: str = Form(...),
        model: str | None = Form(None),
    ):
        if asr is None:
            raise HTTPException(501, "Speech recognition is off. Start the server with --asr.")
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
        return JSONResponse({"text": text, "language": lang.code})

    return app
