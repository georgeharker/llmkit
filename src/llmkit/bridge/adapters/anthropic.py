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
* ``endpoint`` follows the SAME ``/v1``-style convention as the openai adapter
  (e.g. ``https://opencode.ai/zen/v1/messages`` from a gateway's docs, or
  ``http://host/v1``), but the SDK owns the ``/v1/messages`` suffix — it appends
  it to ``base_url``. So the ``/v1...`` tail is stripped here before handing the
  host base to the SDK (else it would double to ``.../v1/v1/messages``). See
  :func:`_base_url`.

The SDK is an OPTIONAL dependency, imported here so the default
openai-compatible path never requires it.
"""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from ..bridge import DEFAULT_ENDPOINT, ChatRequest, _Emitter
from ..client import resolved_headers
from ..config import Provider


def _base_url(endpoint: str) -> str:
    """Derive the anthropic SDK ``base_url`` from a configured endpoint.

    The SDK appends ``/v1/messages`` to ``base_url``, so a ``/v1``-style
    endpoint — the openai-adapter convention, and what gateway docs list (e.g.
    OpenCode Zen's ``https://opencode.ai/zen/v1/messages``) — must have its
    ``/v1...`` tail dropped or the request URL doubles to ``.../v1/v1/messages``.
    Keep scheme, host, and any path prefix *before* the first ``v1`` path
    segment; an endpoint with no ``v1`` segment (``https://api.anthropic.com``)
    is returned unchanged. ``v1`` in the host (``https://v1.example.com``) is
    untouched — only path segments are considered.
    """
    parts = urlsplit(endpoint)
    segs = parts.path.split("/")
    path = "/".join(segs[: segs.index("v1")]) if "v1" in segs else parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _require_sdk() -> Any:
    """Import the SDK, or raise ImportError with an actionable install hint.

    Catchable exception, never ``SystemExit`` — see the note in
    ``claude_code._require_sdk``."""
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "llmkit: the anthropic adapter needs the Anthropic SDK. "
            "Install it with `pip install llmkit[anthropic]` (or `pip install "
            "anthropic`)."
        ) from e
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
    # endpoint the user actually customised (an Anthropic-compatible gateway),
    # normalised to the host base the SDK wants (see _base_url).
    if provider.endpoint and provider.endpoint != DEFAULT_ENDPOINT:
        kwargs["base_url"] = _base_url(provider.endpoint)
    headers = resolved_headers(provider)
    if headers:
        kwargs["default_headers"] = headers
    client = anthropic.Anthropic(**kwargs)

    create: dict[str, Any] = dict(
        model=provider.model,
        max_tokens=provider.max_tokens,
        messages=[{"role": "user", "content": request.user}],
    )
    if request.system:
        create["system"] = request.system
    # Structured output → a single forced tool whose input_schema IS the caller's
    # schema; the model must fill it. Its ``input_json_delta`` stream is routed to
    # the content sink, so downstream capture (chat_to_str/chat_structured) sees the
    # conformant JSON where free-form text would otherwise be. Forced tool-use and
    # extended thinking are mutually exclusive, so thinking is disabled here.
    structured = request.schema is not None
    if structured:
        create["tools"] = [{
            "name": request.schema_name,
            "description": request.schema_description or "Return the result.",
            "input_schema": request.schema,
        }]
        create["tool_choice"] = {"type": "tool", "name": request.schema_name}
        create["thinking"] = {"type": "disabled"}
    else:
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
            elif dt == "input_json_delta":   # forced-tool structured output
                emitter.feed_content(getattr(delta, "partial_json", "") or "")
            elif dt == "thinking_delta":
                emitter.feed_reasoning(getattr(delta, "thinking", "") or "")
