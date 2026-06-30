"""``anthropic`` chat producer — native Anthropic Messages API, streamed.

Distinct from ``claude_code`` (which routes through the Claude Agent SDK / the
``claude`` CLI's auth): this talks straight to the Messages API via the
``anthropic`` Python SDK, so it honours the provider's ``api_key`` /
``api_key_env`` and ``max_tokens`` directly.

Format differences from the openai-compatible path this copes with:

* ``system`` is a top-level request field, not a ``role: system`` message.
* Reasoning arrives as ``thinking`` content-block deltas, not a
  ``reasoning_content`` field on the chat delta.
* Sampling params are NOT forwarded — current Anthropic models (Opus 4.8/4.7,
  Sonnet 5, Fable 5) reject ``temperature`` / ``top_p`` with a 400, and govern
  sampling internally. Steer behaviour through the prompt instead.
* Thinking is configured via the ``thinking`` request param (adaptive only on
  current models — ``budget_tokens`` is removed), not ``chat_template_kwargs``.

The SDK is an OPTIONAL dependency, imported here so the default
openai-compatible path never requires it.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

from ..bridge import DEFAULT_ENDPOINT, ChatRequest, _Emitter
from ..config import Provider


def _require_sdk() -> Any:
    """Import the SDK or exit(2) with an actionable install hint."""
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        print(
            "llmkit: the anthropic adapter needs the Anthropic SDK. "
            "Install it with `pip install llmkit[anthropic]` (or `pip install "
            "anthropic`).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return anthropic


def _resolve_api_key(provider: Provider) -> Optional[str]:
    """Resolve an explicit key, or None to let the SDK read its own env
    (``ANTHROPIC_API_KEY`` / an ``ant auth login`` profile)."""
    if provider.api_key_env:
        return os.environ.get(provider.api_key_env) or None
    return provider.api_key or None


def _thinking_config(flag: Optional[str]) -> dict[str, Any]:
    """Map the bridge's auto|true|false onto the Messages API ``thinking`` param.

    Matches ``claude_code``'s mapping rather than the openai adapter's: the
    native API's only on-mode is adaptive, so both ``auto`` and ``true`` enable
    adaptive thinking — ``auto`` does NOT collapse to off. ``display`` is
    ``summarized`` so reasoning actually streams (current models default it to
    ``omitted`` — empty thinking text). ``false`` disables it.

    NB: adaptive requires a 4.6+ model — don't pair it with an older one.
    """
    if flag == "false":
        return {"type": "disabled"}
    return {"type": "adaptive", "display": "summarized"}


def stream_anthropic(
    provider: Provider, request: ChatRequest, emitter: _Emitter
) -> None:
    anthropic = _require_sdk()

    kwargs: dict[str, Any] = dict(
        api_key=_resolve_api_key(provider),
    )
    # The bridge fills endpoint with the openai/ollama default; only forward an
    # endpoint the user actually customised (an Anthropic-compatible gateway).
    if provider.endpoint and provider.endpoint != DEFAULT_ENDPOINT:
        kwargs["base_url"] = provider.endpoint
    client = anthropic.Anthropic(**kwargs)

    create: dict[str, Any] = dict(
        model=provider.model,
        max_tokens=provider.max_tokens,
        messages=[{"role": "user", "content": request.user}],
    )
    if request.system:
        create["system"] = request.system
    create["thinking"] = _thinking_config(provider.enable_thinking)

    with client.messages.stream(**create) as stream:
        for event in stream:
            # Any event is a sign of life — fire TTFT before the first non-empty
            # delta (mirrors the openai adapter's explicit signal).
            emitter.signal_streaming()
            if event.type != "content_block_delta":
                continue
            delta = event.delta
            dt = getattr(delta, "type", None)
            if dt == "text_delta":
                emitter.feed_content(getattr(delta, "text", "") or "")
            elif dt == "thinking_delta":
                emitter.feed_reasoning(getattr(delta, "thinking", "") or "")
