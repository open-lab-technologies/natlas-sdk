"""Tool definitions from plain Python functions.

    def get_weather(city: str, unit: Literal["C", "F"] = "C") -> dict:
        '''Current weather for a city.

        Args:
            city: City name, for example Kano.
        '''

    client.chat("Yaya yanayin yau a Kano?", tools=[get_weather])
"""

from __future__ import annotations

import enum
import inspect
import re
import types
import typing
from collections.abc import Callable
from typing import Any, Literal, Union

ToolSpec = dict[str, Any] | Callable[..., Any]

_SCALARS: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation in _SCALARS:
        return {"type": _SCALARS[annotation]}
    if hasattr(annotation, "model_json_schema"):  # Pydantic model
        return annotation.model_json_schema()
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {"enum": [m.value for m in annotation]}
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is Literal:
        return {"enum": list(args)}
    if origin in (Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        if len(options) == 1:
            return _schema(options[0])
        return {"anyOf": [_schema(a) for a in options]}
    if origin in (list, tuple, set, frozenset) or annotation in (list, tuple, set):
        return {"type": "array", "items": _schema(args[0])} if args else {"type": "array"}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    raise TypeError(f"Cannot describe parameter type {annotation!r} as JSON schema")


def _param_docs(doc: str) -> dict[str, str]:
    """Descriptions from a Google-style "Args:" section."""
    section = r"^\s*(?:Args|Arguments|Parameters):\s*\n(.*?)(?:\n\s*\n|\Z)"
    match = re.search(section, doc, re.DOTALL | re.MULTILINE)
    if not match:
        return {}
    found: dict[str, str] = {}
    for line in match.group(1).splitlines():
        item = re.match(r"\s*(\w+)(?:\s*\([^)]*\))?:\s*(.+)", line)
        if item:
            found[item.group(1)] = item.group(2).strip()
    return found


def function_tool(fn: Callable[..., Any]) -> dict[str, Any]:
    """An OpenAI tool definition built from a function's signature and docstring."""
    doc = inspect.getdoc(fn) or ""
    descriptions = _param_docs(doc)
    hints = typing.get_type_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        prop = _schema(hints.get(name, param.annotation))
        if name in descriptions:
            prop["description"] = descriptions[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    summary = doc.split("\n\n", 1)[0].strip()
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": summary,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def to_tool(spec: ToolSpec) -> dict[str, Any]:
    """Accept a function, an OpenAI tool definition, or a bare function definition."""
    if callable(spec) and not isinstance(spec, dict):
        return function_tool(spec)
    if spec.get("type") == "function" and "function" in spec:
        return spec
    if "name" in spec:
        return {"type": "function", "function": spec}
    raise ValueError(f"Not a tool definition: {spec!r}")
