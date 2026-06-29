"""``claude_code`` chat producer — routes the query through the Claude Agent
SDK instead of an OpenAI-compatible endpoint.

Plain-chat only: every built-in tool is disabled and the agent runs a single
turn, so it behaves like an ordinary chat completion (no file or shell
access). Authentication piggybacks on the Claude Code CLI — the user's
existing ``claude`` login (subscription) or ``ANTHROPIC_API_KEY``, exactly as
the CLI resolves it. There is no FIM equivalent, so ``complete`` rejects this
adapter.

The SDK is an OPTIONAL dependency, imported here so the default
openai-compatible path never requires it. The provider's ``endpoint`` /
``api_key`` / ``max_tokens`` / ``temperature`` are ignored — the Claude Code
CLI governs those itself. Only ``model``, the request's ``system``, and
``enable_thinking`` carry over.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from ..bridge import ChatRequest, _Emitter
from ..config import Provider


def _require_sdk() -> Any:
    """Import the SDK or exit(2) with an actionable install hint."""
    try:
        import claude_agent_sdk as sdk  # type: ignore[import-not-found]
    except ImportError:
        print(
            "llmkit: the claude_code adapter needs the Claude Agent SDK. "
            "Install it with `pip install llmkit[claude]` (or `pip install "
            "claude-agent-sdk`), and make sure the `claude` CLI is on PATH.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return sdk


def _thinking_budget() -> int:
    """Token budget for ``enable_thinking = true`` → the SDK's
    ``--max-thinking-tokens``. A CAP on extended thinking, not a floor — the
    model still thinks only as much as the question needs. Decoupled from
    max_tokens (claude_code ignores that for output). Override via
    ``$LLMKIT_CLAUDE_THINKING_BUDGET``; floored at the API minimum 1024."""
    try:
        n = int(os.environ.get("LLMKIT_CLAUDE_THINKING_BUDGET", "") or 0)
    except ValueError:
        n = 0
    return n if n >= 1024 else 8192


def _build_options(sdk: Any, provider: Provider, request: ChatRequest) -> Any:
    """Assemble ClaudeAgentOptions for a tool-free single-turn chat."""
    opts: dict[str, Any] = dict(
        tools=[],  # no built-in tools — plain chat, no file/shell access
        allowed_tools=[],
        disallowed_tools=[],
        max_turns=1,
        permission_mode="bypassPermissions",
        include_partial_messages=True,  # token-level streaming deltas
        setting_sources=[],  # ignore project/user CLAUDE.md, settings, etc.
    )
    if provider.model:
        opts["model"] = provider.model
    if request.system:
        opts["system_prompt"] = request.system

    # Map the bridge's auto|true|false onto the SDK thinking config:
    #   false → disabled (truly off — never "forced on")
    #   true  → enabled with an explicit budget cap (see _thinking_budget)
    #   auto  → adaptive: the model scales thinking to the question's difficulty
    # NB: Claude Code returns SUMMARISED thinking (no raw chain-of-thought),
    # so expect it terser than a local reasoning model. It does stream token
    # by token via the partial-message events.
    flag = provider.enable_thinking or "auto"
    if flag == "false":
        opts["thinking"] = {"type": "disabled"}
    elif flag == "true":
        opts["thinking"] = {"type": "enabled", "budget_tokens": _thinking_budget()}
    else:
        opts["thinking"] = {"type": "adaptive"}

    return sdk.ClaudeAgentOptions(**opts)


def _handle_event(ev: Any, emitter: _Emitter) -> bool:
    """Translate one raw Anthropic stream event into emitter deltas. Returns
    True if it carried any text/thinking."""
    if not isinstance(ev, dict) or ev.get("type") != "content_block_delta":
        return False
    delta = ev.get("delta") or {}
    dt = delta.get("type")
    if dt == "text_delta":
        emitter.feed_content(delta.get("text", ""))
        return True
    if dt == "thinking_delta":
        emitter.feed_reasoning(delta.get("thinking", ""))
        return True
    return False


def stream_claude_code(
    provider: Provider, request: ChatRequest, emitter: _Emitter
) -> None:
    """Synchronous producer entry point — runs the async query to completion,
    raising on any SDK/CLI error for :func:`_run_chat` to report."""
    sdk = _require_sdk()
    import anyio  # bundled with the SDK

    anyio.run(_run, sdk, provider, request, emitter)


async def _run(
    sdk: Any, provider: Provider, request: ChatRequest, emitter: _Emitter
) -> None:
    options = _build_options(sdk, provider, request)
    saw_delta = False
    async for message in sdk.query(prompt=request.user, options=options):
        if isinstance(message, sdk.StreamEvent):
            if _handle_event(message.event, emitter):
                saw_delta = True
        elif isinstance(message, sdk.AssistantMessage) and not saw_delta:
            # Fallback for a CLI that doesn't emit partial-message stream
            # events: emit the assembled blocks once, non-incrementally.
            for block in message.content:
                if isinstance(block, sdk.ThinkingBlock):
                    emitter.feed_reasoning(getattr(block, "thinking", "") or "")
                elif isinstance(block, sdk.TextBlock):
                    emitter.feed_content(getattr(block, "text", "") or "")
