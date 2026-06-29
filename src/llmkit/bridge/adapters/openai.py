"""Default chat producer: OpenAI-compatible chat-completions, streamed."""

from __future__ import annotations

from typing import Any

from ..client import build_client, enable_thinking_extra
from ..config import Provider

# Imported under TYPE_CHECKING-free runtime: bridge imports this lazily, so a
# direct import here is fine and avoids a cycle (bridge → adapters, not back).
from ..bridge import ChatRequest, _Emitter


def stream_openai(provider: Provider, request: ChatRequest, emitter: _Emitter) -> None:
    client = build_client(provider)

    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.user})

    # Any-typed so the conditional appends + the **kwargs unpack into openai's
    # heavily-overloaded create() don't trigger a wall of variance complaints.
    kwargs: dict[str, Any] = dict(
        model=provider.model,
        messages=messages,
        max_tokens=provider.max_tokens,
        temperature=provider.temperature,
    )
    extra = enable_thinking_extra(provider.enable_thinking)
    if extra:
        kwargs["extra_body"] = extra
    kwargs["stream"] = True

    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        # Fire "streaming" on the FIRST chunk that has choices — even a
        # role-only/empty delta: TTFT mitigation cares about *any* sign of
        # life. feed_content / feed_reasoning ALSO signal, but relying on
        # them alone would defer this to the first NON-empty delta. So this
        # explicit call is load-bearing, not redundant.
        emitter.signal_streaming()
        delta = chunk.choices[0].delta
        r = getattr(delta, "reasoning_content", None) or ""
        c = getattr(delta, "content", None) or ""
        if r:
            emitter.feed_reasoning(r)
        if c:
            emitter.feed_content(c)
