"""Recover a schema-shaped object from a model's structured-output reply.

Adapters ask for structured output natively (``response_format`` for
openai-compatible, a forced tool for anthropic, ``output_format`` for
claude_code), but an openai-compatible server is free to ignore
``response_format``: e.g. GLM at ollama.com answers in ordinary content, with
the JSON fenced in ````` ```json ````` and missing the schema's wrapper object
(a bare ``[...]`` where the schema says ``{"symbols": [...]}``). Verified
against ollama.com: ``json_schema`` (strict or not), ``json_object`` and
Ollama's native ``format`` are all ignored.

Two defences, both here:

* :func:`schema_instruction` puts the schema in the prompt, so a model whose
  server drops ``response_format`` still sees the shape (with it, GLM answers
  with the exact object; without it, it guesses keys).
* :func:`parse_structured` parses the reply leniently and checks it against
  the schema; any failure raises :class:`StructuredOutputError` with the
  reason. Callers must never mistake "nothing usable came back" for success.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
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


_DECODER = json.JSONDecoder()


class StructuredOutputError(ValueError):
    """The reply held no JSON matching the requested schema."""


def schema_instruction(schema: dict[str, Any]) -> str:
    """Prompt text carrying ``schema``, for adapters whose server may not
    enforce it. Appended to the system prompt."""
    return (
        "Respond with ONLY a JSON value that conforms to this JSON Schema - "
        "no prose, no code fences:\n" + json.dumps(schema)
    )


def _candidates(text: str) -> Iterator[Any]:
    """Every top-level JSON value ``text`` might mean, most likely first: the
    whole text (if it parses, it is the only candidate), then each fenced code
    block, then each ``{``/``[`` value in prose. Never a value nested inside
    another, so ``{"wrong": [...]}`` can't yield its inner array."""
    s = text.strip()
    if not s:
        return
    whole, end = _decode(s, 0)
    if end == len(s):
        yield whole
        return
    for m in _FENCE.finditer(s):
        value, end = _decode(m.group(1).strip(), 0)
        if end:
            yield value
    i = 0
    while i < len(s):
        if s[i] in "{[":
            value, end = _decode(s, i)
            if end:
                yield value
                i = end
                continue
        i += 1


def _decode(s: str, start: int) -> tuple[Any, int]:
    """The JSON value at ``s[start:]`` and the index just past it, or
    ``(None, 0)`` if none starts there."""
    try:
        return _DECODER.raw_decode(s, start)
    except ValueError:
        return None, 0


def extract_json(text: str) -> Any:
    """The most likely JSON value in ``text`` (see :func:`_candidates`)."""
    if not text.strip():
        raise StructuredOutputError("empty reply")
    for value in _candidates(text):
        return value
    raise StructuredOutputError(f"no JSON value in reply: {text.strip()[:120]!r}")


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


def merge_tool_arguments(args: list[str], schema: dict[str, Any] | None) -> str:
    """One structured reply from several tool calls' arguments. A model may
    split an answer across parallel calls (one ``{"symbols": [x]}`` each);
    when the schema is an object with a single array property, those arrays
    are concatenated into one object. Otherwise the calls are returned one per
    line and :func:`parse_structured` takes the first that fits."""
    if len(args) == 1:
        return args[0]
    key = _sole_array_property(schema) if schema and schema.get("type") == "object" else None
    if key is not None:
        merged: list[Any] = []
        for a in args:
            value, end = _decode(a.strip(), 0)
            if not end:
                break
            items = value.get(key) if isinstance(value, dict) else value
            if not isinstance(items, list):
                break
            merged.extend(items)
        else:
            return json.dumps({key: merged})
    return "\n".join(args)


def parse_structured(text: str, schema: dict[str, Any] | None) -> Any:
    """The first JSON value in a structured-output reply that fits ``schema``
    (so a stray ``[...]`` in prose or leaked reasoning can't shadow the real
    answer). Raises with the most-likely candidate's mismatch if none fits."""
    first: StructuredOutputError | None = None
    for value in _candidates(text):
        try:
            return conform(value, schema)
        except StructuredOutputError as e:
            first = first or e
    if first is not None:
        raise first
    return conform(extract_json(text), schema)  # raises: empty / no JSON
