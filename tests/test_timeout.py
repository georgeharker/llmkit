"""``Provider.timeout`` bounds a whole call: a stalled read (per-read client
timeout) and a slow trickle (total deadline between chunks)."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest

from llmkit.bridge import ChatRequest, Provider, chat_to_str
from llmkit.bridge.client import Deadline, build_client, timeout_kwargs
from llmkit.bridge.config import ConfigParser


def test_toml_and_client_kwargs() -> None:
    p = ConfigParser().parse({"providers": {"x": {"timeout": 90}}}).resolve("x")
    assert p.timeout == 90.0
    assert timeout_kwargs(p) == {"timeout": 90.0, "max_retries": 0}
    assert timeout_kwargs(Provider()) == {}
    client = build_client(Provider(endpoint="http://127.0.0.1:1/v1", timeout=7))
    assert client.timeout == 7 and client.max_retries == 0


def test_deadline() -> None:
    Deadline(None).check()
    d = Deadline(0.05)
    d.check()
    time.sleep(0.06)
    with pytest.raises(TimeoutError):
        d.check()


def _sse_server(chunks: Iterator[float]) -> tuple[str, threading.Event]:
    """An openai-compatible endpoint that streams one SSE chunk per delay in
    ``chunks`` (sleeping that long first), never sending [DONE]."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    done = threading.Event()

    def serve() -> None:
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            b"connection: close\r\n\r\n"
        )
        try:
            for delay in chunks:
                if done.wait(delay):
                    break
                conn.sendall(
                    b'data: {"id":"1","object":"chat.completion.chunk","created":0,'
                    b'"model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'
                )
        except OSError:
            pass
        finally:
            conn.close()
            srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return f"http://127.0.0.1:{srv.getsockname()[1]}/v1", done


@pytest.mark.parametrize(
    "chunks",
    [
        iter([0.0, 30.0]),  # one chunk, then a stalled read → per-read timeout
        iter([0.1] * 300),  # a steady trickle → total deadline between chunks
    ],
    ids=["stalled-read", "trickle"],
)
def test_openai_stream_is_bounded(chunks: Iterator[float]) -> None:
    endpoint, done = _sse_server(chunks)
    t0 = time.monotonic()
    code, text = chat_to_str(
        Provider(model="m", endpoint=endpoint, timeout=1.0), ChatRequest(user="hi")
    )
    elapsed = time.monotonic() - t0
    done.set()
    assert code == 1
    assert text.startswith("x")
    assert elapsed < 2.5
