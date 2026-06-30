"""Flag-based CLI — a thin adapter over the dataclass API in
:mod:`llmkit.bridge.bridge`.

Two subcommands:

  chat       chat-completions (openai-compatible or claude_code)
  complete   text-completions with FIM ``suffix`` (openai-compatible only)

This is the shape a host like a shell plugin drives: it resolves its own
config (zstyle, env, …) and passes the result as flags. Everything here just
maps those flags onto a :class:`Provider` + request and calls ``chat`` /
``complete``; no logic lives here that the API doesn't already own.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import bridge
from .config import Provider


def _add_common(s: argparse.ArgumentParser) -> None:
    s.add_argument("--model", required=True)
    s.add_argument("--max-tokens", type=int, default=1024)
    s.add_argument("--temperature", type=float, default=0.2)
    s.add_argument(
        "--endpoint",
        default=os.environ.get("LLMKIT_ENDPOINT", "http://localhost:11434/v1"),
    )
    s.add_argument("--api-key", default="")
    s.add_argument("--api-key-env", default="")
    s.add_argument(
        "--adapter",
        choices=["openai-compatible", "claude_code", "anthropic", "google"],
        default=os.environ.get("LLMKIT_ADAPTER", "openai-compatible"),
        help="transport backend: openai-compatible (HTTP endpoint, default), "
        "anthropic (native Messages API), google (native Gemini API), or "
        "claude_code (Claude Agent SDK; chat only). Point --endpoint at a "
        "gateway (e.g. OpenCode Zen) to route a native protocol through it. "
        "complete/FIM is openai-compatible-only.",
    )


def _provider_from_args(args: argparse.Namespace) -> Provider:
    """Map the connection-level flags onto a Provider. enable_thinking is a
    per-call (request) concern, so it's not set here."""
    return Provider(
        model=args.model,
        adapter=args.adapter,
        endpoint=args.endpoint,
        api_key=args.api_key or None,
        api_key_env=args.api_key_env or None,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmkit",
        description="Streaming LLM bridge "
        "(openai-compatible / anthropic / google / claude_code).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="chat-completions endpoint")
    _add_common(chat)
    chat.add_argument("--system", default="")
    chat.add_argument("--user", required=True)
    chat.add_argument(
        "--enable-thinking",
        choices=["auto", "true", "false"],
        default="auto",
        help="maps to chat_template_kwargs.enable_thinking (or the SDK's "
        "thinking config for claude_code)",
    )
    chat.add_argument(
        "--content",
        default="-",
        help="content sink: -, PATH, or none (default: -)",
    )
    chat.add_argument(
        "--thinking",
        default="none",
        help="thinking sink: -, PATH, inline, or none (default: none). "
        "inline merges reasoning into the content stream, separated from "
        "real content by a blank line",
    )
    chat.add_argument(
        "--status-file",
        default="",
        help="path (file or fifo) to receive status lines: 'streaming' on "
        "first chunk, 'complete' on clean exit, 'error' on exception, "
        "'interrupted' on ^C.",
    )

    comp = sub.add_parser("complete", help="text-completions endpoint (FIM)")
    _add_common(comp)
    comp.add_argument("--prompt", required=True)
    comp.add_argument("--suffix", default="")
    comp.add_argument(
        "--stop",
        action="append",
        default=[],
        help="stop token (repeatable)",
    )

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    provider = _provider_from_args(args)
    if args.cmd == "chat":
        request = bridge.ChatRequest(
            user=args.user,
            system=args.system or None,
            enable_thinking=args.enable_thinking,
        )
        return bridge.chat(
            provider,
            request,
            content=args.content,
            thinking=args.thinking,
            status_file=args.status_file,
        )
    if args.cmd == "complete":
        creq = bridge.CompleteRequest(
            prompt=args.prompt,
            suffix=args.suffix,
            stop=tuple(args.stop),
        )
        return bridge.complete(provider, creq)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
