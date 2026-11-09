"""``chat_structured`` against a server that ignores ``response_format`` and
answers in content (GLM at ollama.com does): fenced JSON, missing the schema's
wrapper object. A usable reply parses; an unusable one is a failure, never
``(0, None)``."""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

import pytest

from llmkit.bridge import ChatRequest, Provider, chat_structured
from llmkit.bridge.structured import (
    StructuredOutputError,
    conform,
    extract_json,
    parse_structured,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "symbols": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "kind"],
            },
        }
    },
    "required": ["symbols"],
}
SYMBOLS = [{"name": "f", "kind": "function", "keywords": ["a", "b"]}]


def _stub(content: str) -> tuple[str, list[dict[str, Any]]]:
    """An openai-compatible endpoint that streams ``content`` as ordinary
    deltas, whatever the request asked for. Returns (endpoint, request bodies)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    bodies: list[dict[str, Any]] = []

    def serve() -> None:
        conn, _ = srv.accept()
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(65536)
        head, _, body = data.partition(b"\r\n\r\n")
        length = next(
            int(line.split(b":", 1)[1])
            for line in head.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        )
        while len(body) < length:
            body += conn.recv(65536)
        bodies.append(json.loads(body))
        conn.sendall(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            b"connection: close\r\n\r\n"
        )
        for piece in (content[: len(content) // 2], content[len(content) // 2 :]):
            chunk = {
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": piece}}],
            }
            conn.sendall(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        conn.sendall(b"data: [DONE]\n\n")
        conn.close()
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return f"http://127.0.0.1:{srv.getsockname()[1]}/v1", bodies


def _structured(content: str) -> tuple[int, Any, list[dict[str, Any]]]:
    endpoint, bodies = _stub(content)
    code, data = chat_structured(
        Provider(model="m", endpoint=endpoint, timeout=5),
        ChatRequest(user="describe", schema=SCHEMA, schema_name="describe_symbols"),
    )
    return code, data, bodies


def test_fenced_bare_array_in_content() -> None:
    """The GLM-at-ollama reply: response_format ignored, fenced bare array."""
    reply = "```json\n" + json.dumps(SYMBOLS, indent=2) + "\n```"
    code, data, bodies = _structured(reply)
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert (code, data) == (0, {"symbols": SYMBOLS})


def test_plain_object_in_content() -> None:
    code, data, _ = _structured(json.dumps({"symbols": SYMBOLS}))
    assert (code, data) == (0, {"symbols": SYMBOLS})


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "I could not describe these symbols.",
        '{"wrong": []}',
        '{"symbols": [{"name": "f"}]}',  # item missing required "kind"
    ],
    ids=["empty", "prose", "wrong-shape", "missing-required"],
)
def test_unusable_reply_is_a_failure(
    reply: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, data, _ = _structured(reply)
    assert (code, data) == (1, None)
    assert "llmkit: structured output:" in capsys.readouterr().err


def test_extract_json() -> None:
    assert extract_json(' {"a": 1} ') == {"a": 1}
    assert extract_json('Here you go:\n```\n[1, 2]\n```\nDone.') == [1, 2]
    assert extract_json('Sure! {"a": [1]} hope that helps') == {"a": [1]}
    with pytest.raises(StructuredOutputError):
        extract_json("```json\nnot json\n```")


def test_conform() -> None:
    assert conform([1], {"type": "object", "properties": {"xs": {"type": "array"}}}) == {
        "xs": [1]
    }
    # No single array property to wrap into: a bare array doesn't fit.
    with pytest.raises(StructuredOutputError):
        conform([1], {"type": "object", "properties": {"a": {}, "b": {}}})
    with pytest.raises(StructuredOutputError, match="expected integer"):
        conform(True, {"type": "integer"})
    with pytest.raises(StructuredOutputError, match="not in enum"):
        conform("c", {"enum": ["a", "b"]})
    assert parse_structured("[]", None) == []
