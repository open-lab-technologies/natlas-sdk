"""Response types returned by the clients."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Usage:
        data = data or {}
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
        )


@dataclass(frozen=True)
class ToolCall:
    """A function the model asked to call. `arguments` is the decoded JSON object."""

    id: str
    name: str
    arguments: dict[str, Any]

    def result(self, value: Any) -> dict[str, Any]:
        """The message that returns `value` to the model as this call's result."""
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return {"role": "tool", "tool_call_id": self.id, "content": content}


@dataclass(frozen=True)
class ChatResponse:
    """A chat reply.

    `text` is empty when the model called a tool instead of answering. `parsed`
    holds the decoded reply when a `response_format` was requested.
    """

    text: str
    model: str
    finish_reason: str | None
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()
    parsed: Any = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def message(self) -> dict[str, Any]:
        """This reply as an assistant message, to append to the conversation history."""
        if not self.tool_calls:
            return {"role": "assistant", "content": self.text}
        return {
            "role": "assistant",
            "content": self.text or None,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name,
                              "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                for c in self.tool_calls
            ],
        }

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class VoiceReply:
    """What `voice_chat` heard, and the model's reply to it."""

    transcript: str
    language: str
    reply: ChatResponse

    def __str__(self) -> str:
        return self.reply.text
