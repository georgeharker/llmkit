"""The dataclass-first bridge API.

The reusable surface: callers build a :class:`~llmkit.bridge.config.Provider`
(however they like — from TOML, env, hardcoded) plus a request, and call
:func:`chat` / :func:`complete`. The flag-based CLI (:mod:`llmkit.bridge.cli`)
is just one adapter over this; it owns no logic the API doesn't.

Two adapters back :func:`chat`: the default ``openai-compatible`` HTTP path
and ``claude_code`` (Claude Agent SDK, chat-only). A "producer" pumps
reasoning/content deltas into an :class:`_Emitter`; :func:`_run_chat` owns
the sink/splitter/status lifecycle so every adapter shares identical
failure semantics.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, TextIO

from .client import build_client
from .config import Provider
from .sinks import close_sink, open_target
from .stream import StreamSplitter

DEFAULT_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.2


@dataclass
class ChatRequest:
    """A single chat turn. ``enable_thinking`` (auto|true|false), if set,
    overrides the provider's configured value for this call."""

    user: str
    system: Optional[str] = None
    enable_thinking: Optional[str] = None


@dataclass
class CompleteRequest:
    """A text-completion (FIM) request. ``openai-compatible`` only."""

    prompt: str
    suffix: str = ""
    stop: tuple[str, ...] = ()


def with_defaults(provider: Provider) -> Provider:
    """Fill the hard defaults a producer needs onto an otherwise-partial
    provider. Returns a base :class:`Provider` (producers read base fields
    only). Idempotent."""
    return Provider(
        model=provider.model,
        adapter=provider.adapter or "openai-compatible",
        endpoint=provider.endpoint or DEFAULT_ENDPOINT,
        api_key=provider.api_key,
        api_key_env=provider.api_key_env,
        max_tokens=provider.max_tokens
        if provider.max_tokens is not None
        else DEFAULT_MAX_TOKENS,
        temperature=provider.temperature
        if provider.temperature is not None
        else DEFAULT_TEMPERATURE,
        enable_thinking=provider.enable_thinking or "auto",
        stop=provider.stop,
        extra=provider.extra,
    )


# ── status events ───────────────────────────────────────────────────────────
def _open_status(path: str) -> Optional[TextIO]:
    """Open a status file (regular file OR fifo) for line-buffered writes.
    O_RDWR | O_APPEND | O_NONBLOCK so it works against either kind without
    blocking on fifo connect (and without truncating regular files).
    Returns None if disabled or open fails — status writes are best-effort."""
    if not path:
        return None
    try:
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NONBLOCK)
        return os.fdopen(fd, "w", buffering=1)
    except OSError:
        return None


def _write_status(fp: Optional[TextIO], event: str) -> None:
    """Best-effort status line write. Never raises."""
    if fp is None:
        return
    try:
        fp.write(event + "\n")
        fp.flush()
    except OSError:
        pass


class _Emitter:
    """Routes a producer's content/reasoning deltas into the splitter and
    fires the one-shot ``streaming`` status event on the first sign of life
    from the backend (reasoning OR content). Producers call
    :meth:`signal_streaming` directly when a chunk arrives that carries
    neither."""

    def __init__(self, splitter: StreamSplitter, status_fp: Optional[TextIO]) -> None:
        self._splitter = splitter
        self._status_fp = status_fp
        self._started = False

    def signal_streaming(self) -> None:
        if not self._started:
            _write_status(self._status_fp, "streaming")
            self._started = True

    def feed_content(self, text: str) -> None:
        if text:
            self.signal_streaming()
            self._splitter.feed_content_delta(text)

    def feed_reasoning(self, text: str) -> None:
        if text:
            self.signal_streaming()
            self._splitter.feed_reasoning_delta(text)


# A producer drives the backend and pumps deltas into the emitter. It raises
# on error (caught by _run_chat) and returns None on success.
Producer = Callable[[Provider, ChatRequest, _Emitter], None]


def _run_chat(
    provider: Provider,
    request: ChatRequest,
    content: str,
    thinking: str,
    status_file: str,
    producer: Producer,
) -> int:
    """Own the sink/splitter/status lifecycle and run ``producer`` inside it.
    Shared by every adapter so the failure semantics (status events, exit
    codes, guaranteed ``splitter.finish()``) are identical."""
    content_sink = open_target(content)
    if content_sink == "inline":
        print(
            'llmkit: content sink cannot be "inline" '
            "(that's a thinking-sink mode only)",
            file=sys.stderr,
        )
        return 2
    thinking_sink = open_target(thinking)
    splitter = StreamSplitter(content_sink, thinking_sink)
    status_fp = _open_status(status_file)
    emitter = _Emitter(splitter, status_fp)

    rc = 0
    try:
        producer(provider, request, emitter)
        _write_status(status_fp, "complete")
    except KeyboardInterrupt:
        _write_status(status_fp, "interrupted")
        rc = 130
    except Exception as e:
        _write_status(status_fp, "error")
        print(f"llmkit: {e}", file=sys.stderr)
        rc = 1
    finally:
        # ALWAYS flush — interrupts and exceptions still get the close wrap
        # emitted, so downstream readers see a complete document.
        try:
            splitter.finish()
        except Exception:
            pass
        close_sink(content_sink)
        close_sink(thinking_sink)
        if status_fp is not None:
            try:
                status_fp.close()
            except OSError:
                pass
    return rc


def chat(
    provider: Provider,
    request: ChatRequest,
    *,
    content: str = "-",
    thinking: str = "none",
    status_file: str = "",
) -> int:
    """Run one chat turn. ``content`` / ``thinking`` are sink specs (``-``,
    a path, ``none``, or — for thinking — ``inline``); ``status_file`` is an
    optional file/fifo path for ``streaming``/``complete``/``error`` events.

    Adapter is chosen from ``provider.adapter``. The request's
    ``enable_thinking`` overrides the provider's configured value."""
    provider = with_defaults(provider)
    eff_thinking = request.enable_thinking or provider.enable_thinking or "auto"
    provider = dataclasses.replace(provider, enable_thinking=eff_thinking)

    if provider.adapter == "claude_code":
        # Imported lazily so the openai path (and --help) never pays for an
        # optional dependency that may not be installed.
        from .adapters.claude_code import stream_claude_code

        producer: Producer = stream_claude_code
    else:
        from .adapters.openai import stream_openai

        producer = stream_openai
    return _run_chat(provider, request, content, thinking, status_file, producer)


def complete(
    provider: Provider,
    request: CompleteRequest,
    *,
    out: Optional[TextIO] = None,
) -> int:
    """Run a text-completion (FIM) request, streaming text to ``out``
    (default stdout). ``openai-compatible`` only."""
    out = out if out is not None else sys.stdout
    provider = with_defaults(provider)
    if provider.adapter != "openai-compatible":
        print(
            f"llmkit: FIM (complete) supports only the openai-compatible "
            f"adapter, not '{provider.adapter}'. Point the provider at an "
            f"openai-compatible backend.",
            file=sys.stderr,
        )
        return 2

    from typing import Any

    client = build_client(provider)
    kwargs: dict[str, Any] = dict(
        model=provider.model,
        prompt=request.prompt,
        max_tokens=provider.max_tokens,
        temperature=provider.temperature,
    )
    if request.suffix:
        kwargs["suffix"] = request.suffix
    if request.stop:
        kwargs["stop"] = list(request.stop)
    kwargs["stream"] = True
    try:
        for chunk in client.completions.create(**kwargs):
            t = chunk.choices[0].text if chunk.choices else ""
            if t:
                out.write(t)
                out.flush()
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"llmkit: {e}", file=sys.stderr)
        return 1
    return 0
