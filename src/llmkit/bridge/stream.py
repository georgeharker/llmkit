"""Two-stream splitter for OpenAI-compatible chat responses.

Unifies the two server conventions for "reasoning":

  1. ``delta.reasoning_content`` — split-out API field (vLLM with
     ``--reasoning-parser``, deepseek API, …)
  2. ``<think>…</think>`` tags inside ``delta.content`` (llama.cpp
     default, ollama, anything that passes raw model output through)

Routes the unified streams to two sinks:

  content   the final answer text
  thinking  a writable (TextIO), ``None`` to drop, or the sentinel
            ``"inline"`` — which merges reasoning into the content
            sink, separated from real content by a blank line. No
            tags, no prefixes; the markdown renderer treats it as
            ordinary prose

Robust to ``<think>`` tags straddling chunk boundaries — the open or
close marker can be split across two ``feed_content_delta`` calls.
"""
from __future__ import annotations

from typing import Optional, TextIO, cast

from .sinks import Sink


class StreamSplitter:
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, content_sink: Sink, thinking_sink: Sink) -> None:
        self.content: Optional[TextIO] = (
            None if content_sink in (None, "inline")
            else cast(TextIO, content_sink)
        )
        self.inline = thinking_sink == "inline"
        # Effective separate-sink thinking output: inline routes through
        # content; None drops; otherwise a TextIO.
        self._think_out: Optional[TextIO]
        if self.inline:
            self._think_out = self.content
        elif thinking_sink is None:
            self._think_out = None
        else:
            self._think_out = cast(TextIO, thinking_sink)

        # Tag-parser state (case 2: <think> inside delta.content).
        self._buf = ""
        self._in_tag_think = False
        # Inline-mode: True while we're emitting thinking text into the
        # content sink, so transitions back to real content know to
        # emit a blank-line separator.
        self._inline_in_thinking = False

    # ── public ──────────────────────────────────────────────────────────
    def feed_content_delta(self, text: str) -> None:
        if not text:
            return
        # API-field reasoning has ended — close the inline-thinking
        # paragraph before content takes over.
        self._end_inline_thinking()
        self._buf += text
        self._consume_buf()

    def feed_reasoning_delta(self, text: str) -> None:
        if not text:
            return
        self._begin_inline_thinking()
        self._emit_thinking(text)

    def finish(self) -> None:
        if self._buf:
            if self._in_tag_think:
                self._emit_thinking(self._buf)
            else:
                self._emit_content(self._buf)
            self._buf = ""
        # Close any still-open inline-thinking paragraph.
        self._end_inline_thinking()

    # ── inline-mode helpers ─────────────────────────────────────────────
    def _begin_inline_thinking(self) -> None:
        """Inline mode only: emit a leading newline before the first
        thinking chunk so the reasoning sits in its own paragraph."""
        if self.inline and not self._inline_in_thinking:
            self._emit_content("\n")
            self._inline_in_thinking = True

    def _end_inline_thinking(self) -> None:
        """Inline mode only: terminate the thinking paragraph with a
        blank line so subsequent content renders as a fresh paragraph."""
        if self.inline and self._inline_in_thinking:
            self._emit_content("\n\n")
            self._inline_in_thinking = False

    # ── low-level emitters ──────────────────────────────────────────────
    def _emit_content(self, s: str) -> None:
        if s and self.content is not None:
            self.content.write(s)
            self.content.flush()

    def _emit_thinking(self, s: str) -> None:
        """Inline mode routes through the content sink (separators are
        bracketed by _begin/_end_inline_thinking). Otherwise writes to
        the dedicated thinking sink."""
        if not s:
            return
        if self.inline:
            self._emit_content(s)
        elif self._think_out is not None:
            self._think_out.write(s)
            self._think_out.flush()

    # ── tag parser for <think>…</think> in delta.content ────────────────
    def _consume_buf(self) -> None:
        while self._buf:
            if not self._in_tag_think:
                idx = self._buf.find(self.OPEN)
                if idx != -1:
                    pre = self._buf[:idx]
                    self._buf = self._buf[idx + len(self.OPEN):]
                    self._emit_content(pre)
                    self._in_tag_think = True
                    self._begin_inline_thinking()
                    continue
                tail_keep = self._partial_tail(self.OPEN)
                if tail_keep > 0:
                    self._emit_content(self._buf[:-tail_keep])
                    self._buf = self._buf[-tail_keep:]
                else:
                    self._emit_content(self._buf)
                    self._buf = ""
                return
            else:
                idx = self._buf.find(self.CLOSE)
                if idx != -1:
                    inner = self._buf[:idx]
                    self._buf = self._buf[idx + len(self.CLOSE):]
                    self._emit_thinking(inner)
                    self._in_tag_think = False
                    self._end_inline_thinking()
                    continue
                tail_keep = self._partial_tail(self.CLOSE)
                if tail_keep > 0:
                    self._emit_thinking(self._buf[:-tail_keep])
                    self._buf = self._buf[-tail_keep:]
                else:
                    self._emit_thinking(self._buf)
                    self._buf = ""
                return

    def _partial_tail(self, marker: str) -> int:
        """Length of trailing buffer that could be the start of ``marker``."""
        buf = self._buf
        max_n = min(len(marker) - 1, len(buf))
        for n in range(max_n, 0, -1):
            if marker.startswith(buf[-n:]):
                return n
        return 0
