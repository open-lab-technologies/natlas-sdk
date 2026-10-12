"""Inference backends for `pynatlas serve`.

The server talks to any object that satisfies `Engine`. `LlamaCppEngine` runs
GGUF builds of N-ATLAS through llama.cpp; other backends only need to
implement the same three methods.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ._extras import require

Messages = list[dict[str, Any]]


@dataclass
class Chunk:
    text: str
    finish_reason: str | None = None


class Engine(Protocol):
    name: str

    def generate(
        self,
        messages: Messages,
        *,
        temperature: float,
        max_tokens: int,
        repetition_penalty: float,
        top_p: float,
        stop: list[str],
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Iterator[Chunk]:
        """Stream the reply.

        `tools` are OpenAI tool definitions to show the model. `json_schema`, when
        set, must constrain the output to JSON matching it. Raise ValueError if the
        prompt cannot fit the context or the schema cannot be enforced.
        """
        ...

    def count_prompt_tokens(self, messages: Messages) -> int: ...

    def count_tokens(self, text: str) -> int: ...


class LlamaCppEngine:
    EOT = "<|eot_id|>"
    # Compiling a grammar from a schema takes a noticeable fraction of a second,
    # and clients usually send the same few schemas over and over.
    GRAMMAR_CACHE_SIZE = 64

    def __init__(self, model_path: str | Path, n_ctx: int = 8192, n_gpu_layers: int = -1,
                 date: str | None = None):
        """`date` pins the date written into every prompt, as "10 Oct 2026". Replies
        depend on it, so benchmark runs are only comparable with the same date."""
        llama_cpp = require("llama_cpp", "server")
        sandbox = require("jinja2.sandbox", "server")

        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}. Run `pynatlas download` first.")
        self.name = path.name
        self.date = date
        self.llm = llama_cpp.Llama(
            model_path=str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
        )
        template = self.llm.metadata.get("tokenizer.chat_template")
        if not template:
            raise ValueError(f"{path.name} has no chat template in its metadata")
        env = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self._template = env.from_string(template)
        self._grammar_cls = llama_cpp.LlamaGrammar
        self._grammars: dict[str, Any] = {}
        # llama.cpp is not safe to call from several threads at once.
        self._lock = threading.Lock()

    def render(self, messages: Messages, tools: list[dict[str, Any]] | None = None) -> str:
        """Apply the model's chat template.

        Rendered here rather than by llama.cpp because the N-ATLAS template stamps
        today's date into the system header, and llama.cpp's formatter leaves it at
        the template's placeholder. Tools go in the system message: in the user
        message, where the template puts them by default, the model calls a tool
        for almost every request.
        """
        prompt: str = self._template.render(
            messages=messages,
            tools=[t.get("function", t) for t in tools] if tools else None,
            tools_in_user_message=False,
            add_generation_prompt=True,
            bos_token="",  # llama.cpp adds BOS itself when tokenising
            date_string=self.date or datetime.now().astimezone().strftime("%d %b %Y"),
        )
        return prompt

    def _grammar(self, schema: dict[str, Any]) -> Any:
        key = json.dumps(schema, sort_keys=True)
        if key not in self._grammars:
            try:
                self._grammars[key] = self._grammar_cls.from_json_schema(key, verbose=False)
            except Exception as e:  # llama.cpp raises several types here
                raise ValueError(f"Cannot enforce this JSON schema: {e}") from e
            if len(self._grammars) > self.GRAMMAR_CACHE_SIZE:
                self._grammars.pop(next(iter(self._grammars)))
        return self._grammars[key]

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode(), add_bos=False, special=True))

    def count_prompt_tokens(self, messages: Messages) -> int:
        return self.count_tokens(self.render(messages)) + 1  # plus BOS

    def generate(
        self,
        messages: Messages,
        *,
        temperature: float,
        max_tokens: int,
        repetition_penalty: float,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Iterator[Chunk]:
        prompt = self.render(messages, tools)
        grammar = self._grammar(json_schema) if json_schema is not None else None
        if grammar is not None:
            # The penalty discourages tokens already used, and JSON repeats quotes,
            # commas and braces by design, so it fights the grammar.
            repetition_penalty = 1.0
        with self._lock:
            # Start from an empty cache. Reusing the previous prompt's cached prefix
            # changes the arithmetic enough to flip near-tied tokens, which would make
            # a temperature-0 reply depend on the request before it.
            self.llm.reset()
            for out in self.llm.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                stop=[self.EOT, *(stop or [])],
                grammar=grammar,
                stream=True,
            ):
                choice = out["choices"][0]
                yield Chunk(choice.get("text") or "", choice.get("finish_reason"))
