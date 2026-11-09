"""Recover a schema-shaped object from a model's structured-output reply.

Adapters ask for structured output natively (``response_format`` for
openai-compatible, a forced tool for anthropic, ``output_format`` for
claude_code), but an openai-compatible server is free to ignore
``response_format``: e.g. GLM at ollama.com answers in ordinary content, with
the JSON fenced in ````` ```json ````` and missing the schema's wrapper object
(a bare ``[...]`` where the schema says ``{"symbols": [...]}``). So the reply
is parsed leniently here, then checked against the schema, and any failure
raises :class:`StructuredOutputError` with the reason. Callers must never
mistake "nothing usable came back" for success.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n(.*?)\n?```", re.DOTALL)
_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


class StructuredOutputError(ValueError):
    """The reply held no JSON matching the requested schema."""


def extract_json(text: str) -> Any:
    """The JSON value in ``text``: the whole text, else the first fenced code
    block that parses, else the first ``{``/``[`` value found in prose."""
    s = text.strip()
    if not s:
        raise StructuredOutputError("empty reply")
    try:
        return json.loads(s)
    except ValueError:
        pass
    for m in _FENCE.finditer(s):
        try:
            return json.loads(m.group(1))
        except ValueError:
            continue
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                return dec.raw_decode(s, i)[0]
            except ValueError:
                continue
    raise StructuredOutputError(f"no JSON value in reply: {s[:120]!r}")


def _sole_array_property(schema: dict[str, Any]) -> str | None:
    """The one array-typed property of an object schema, if it has exactly
    one property (or exactly one required one) and that property is an array."""
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    names = list(props) if len(props) == 1 else list(required)
    if len(names) == 1 and (props.get(names[0]) or {}).get("type") == "array":
        return names[0]
    return None


def _check(value: Any, schema: dict[str, Any], path: str) -> None:
    """A JSON Schema subset: type, enum, required, properties, items. Enough to
    tell a schema-shaped reply from a wrong one, without a jsonschema dep."""
    t = schema.get("type")
    types = [t] if isinstance(t, str) else (t or [])
    if types:
        ok = any(
            isinstance(value, _TYPES.get(name, ()))
            # bool is an int subclass; JSON keeps them apart.
            and not (isinstance(value, bool) and name in ("integer", "number"))
            for name in types
        )
        if not ok:
            raise StructuredOutputError(
                f"{path or 'reply'}: expected {'/'.join(types)}, "
                f"got {type(value).__name__}"
            )
    if "enum" in schema and value not in schema["enum"]:
        raise StructuredOutputError(f"{path or 'reply'}: {value!r} not in enum")
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                raise StructuredOutputError(f"{path or 'reply'}: missing {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value and isinstance(sub, dict):
                _check(value[key], sub, f"{path}.{key}" if path else key)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{i}]")


def conform(value: Any, schema: dict[str, Any] | None) -> Any:
    """``value`` checked against ``schema``. A bare array where the schema is an
    object with a single array property is wrapped into that property first
    (the shape models produce when they skip the wrapper). Raises
    :class:`StructuredOutputError` if it still doesn't fit."""
    if not schema:
        return value
    if isinstance(value, list) and schema.get("type") == "object":
        key = _sole_array_property(schema)
        if key is not None:
            value = {key: value}
    _check(value, schema, "")
    return value


def parse_structured(text: str, schema: dict[str, Any] | None) -> Any:
    """Parse a structured-output reply and check it against ``schema``."""
    return conform(extract_json(text), schema)
