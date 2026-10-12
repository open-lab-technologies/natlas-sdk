"""Translation between OpenAI tool calling and the N-ATLAS (Llama-3.1) format.

N-ATLAS calls a tool by replying `{"name": ..., "parameters": {...}}`, one call
per turn. Left to itself it wraps that in code fences and calls a tool for
almost every request, invented ones included. So every tool decision is
generated under a JSON schema listing only the real tools; in "auto" mode the
schema also offers `{"name": "none"}` for requests that need no tool.
"""

from __future__ import annotations

import json
from typing import Any

NO_TOOL = "none"

AUTO_INSTRUCTION = (
    "\n\nOnly call a function when the request needs live data or an action it "
    f'provides. Otherwise respond with {{"name": "{NO_TOOL}", "parameters": {{}}}}.'
)


class ToolChoiceError(ValueError):
    pass


def _function(tool: dict[str, Any]) -> dict[str, Any]:
    function: dict[str, Any] = tool.get("function", tool)
    return function


def call_schema(tools: list[dict[str, Any]], tool_choice: Any) -> tuple[dict[str, Any], bool]:
    """JSON schema for the model's tool decision, and whether "none" is allowed."""
    functions = [_function(t) for t in tools]
    if isinstance(tool_choice, dict):
        wanted = tool_choice.get("function", {}).get("name")
        functions = [f for f in functions if f["name"] == wanted]
        if not functions:
            raise ToolChoiceError(f"tool_choice names {wanted!r}, which is not in tools")
        allow_none = False
    elif tool_choice == "required":
        allow_none = False
    elif tool_choice in (None, "auto"):
        allow_none = True
    else:
        raise ToolChoiceError(f"Unsupported tool_choice {tool_choice!r}")

    options = [
        {"type": "object",
         "properties": {"name": {"const": f["name"]},
                        "parameters": f.get("parameters") or {"type": "object", "properties": {}}},
         "required": ["name", "parameters"]}
        for f in functions
    ]
    if allow_none:
        options.append({"type": "object",
                        "properties": {"name": {"const": NO_TOOL},
                                       "parameters": {"type": "object", "properties": {}}},
                        "required": ["name", "parameters"]})
    return {"anyOf": options}, allow_none


def parse_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """(name, arguments) from schema-constrained output, or None for "no tool"."""
    call = json.loads(text)
    if call["name"] == NO_TOOL:
        return None
    return call["name"], call.get("parameters") or {}


def to_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI messages in the shape the N-ATLAS chat template expects.

    The template takes one tool call per assistant message with decoded
    arguments, so a message with several calls becomes several messages.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        calls = m.get("tool_calls")
        if m["role"] == "assistant" and calls:
            for call in calls:
                fn = call["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                out.append({"role": "assistant", "content": "",
                            "tool_calls": [{"function": {"name": fn["name"], "arguments": args}}]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out
