"""Local inference for N-ATLAS GGUF builds through llama.cpp."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# The N-ATLAS template is Llama-3.1's, which stamps today's date into the system
# header. llama.cpp's built-in formatter leaves the date at the template default,
# so the template is rendered here instead.
EOT = "<|eot_id|>"


@dataclass
class Chunk:
    text: str
    finish_reason: str | None = None


class LlamaCppEngine:
    def __init__(self, model_path: str | Path, n_ctx: int = 8192, n_gpu_layers: int = -1):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from llama_cpp import Llama

        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}. Run `natlas download` first.")
        self.model_path = path
        self.llm = Llama(
            model_path=str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
        )
        template = self.llm.metadata.get("tokenizer.chat_template")
        if not template:
            raise ValueError(f"{path.name} has no chat template in its metadata")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self._template = env.from_string(template)
        # llama.cpp is not safe to call from several threads at once.
        self._lock = threading.Lock()

    def render(self, messages: list[dict[str, str]]) -> str:
        return self._template.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token="",  # llama.cpp adds BOS itself when tokenising
            date_string=datetime.now().astimezone().strftime("%d %b %Y"),
        )

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode(), add_bos=True, special=True))

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        repetition_penalty: float,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> Iterator[Chunk]:
        prompt = self.render(messages)
        with self._lock:
            # llama.cpp reuses the cached prefix of the previous prompt. Evaluating
            # only the remainder changes the floating-point results enough to flip
            # near-tied tokens, so the same request at temperature 0 answered
            # differently depending on what came before it. Clearing the cache makes
            # replies, and benchmark scores, independent of request order.
            self.llm.reset()
            for out in self.llm.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repetition_penalty,
                stop=[EOT, *(stop or [])],
                stream=True,
            ):
                choice = out["choices"][0]
                yield Chunk(choice.get("text") or "", choice.get("finish_reason"))
