"""Default chat producer: OpenAI-compatible chat-completions, streamed."""

from __future__ import annotations

from typing import Any

from ..client import Deadline, build_client, enable_thinking_extra
from ..config import Provider
from ..structured import merge_tool_arguments, schema_instruction

# Imported under TYPE_CHECKING-free runtime: bridge imports this lazily, so a
# direct import here is fine and avoids a cycle (bridge → adapters, not back).
from ..bridge import ChatRequest, _Emitter


def stream_openai(provider: Provider, request: ChatRequest, emitter: _Emitter) -> None:
    client = build_client(provider)
    # Structured output channel (``Provider.structured``): ``response_format``
    # (default), or ``tool``, a single forced tool whose parameters are the
    # schema, for servers that ignore response_format but honour tool calls
    # (Ollama Cloud: see llmkit.bridge.structured).
    tool_mode = request.schema is not None and provider.structured == "tool"

    # A server may ignore response_format (ollama.com does), so the schema also
    # rides in the system prompt: the model must see the shape to produce it.
    # Not in tool mode: the tool definition carries the schema, and "respond
    # with only JSON" would invite an answer in content instead of the call.
    system = request.system
    if request.schema is not None and not tool_mode:
        hint = schema_instruction(request.schema)
        system = f"{system}\n\n{hint}" if system else hint
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": request.user})

    # Any-typed so the conditional appends + the **kwargs unpack into openai's
    # heavily-overloaded create() don't trigger a wall of variance complaints.
    kwargs: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "max_tokens": provider.max_tokens,
        "temperature": provider.temperature,
    }
    extra = enable_thinking_extra(provider.enable_thinking)
    if extra:
        kwargs["extra_body"] = extra
    # Structured output → constrain decoding to the caller's JSON Schema. The
    # filled JSON lands in the content sink either way (response_format: as
    # ordinary content deltas; tool: the call's arguments, emitted at the end),
    # so capture is unchanged. Thinking is dropped when structured (mirrors the
    # anthropic path).
    if tool_mode:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": request.schema_name,
                    "description": request.schema_description or "Return the result.",
                    "parameters": request.schema,
                },
            }
        ]
        kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": request.schema_name},
        }
        kwargs.pop("extra_body", None)
    elif request.schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": request.schema_name, "schema": request.schema},
        }
        kwargs.pop("extra_body", None)
    kwargs["stream"] = True

    # Tool-call argument fragments, by call index. A model may split one answer
    # across several parallel calls (GLM at ollama.com sometimes does), so they
    # are collected and merged once the stream ends.
    calls: dict[int, list[str]] = {}
    deadline = Deadline(provider.timeout)
    for chunk in client.chat.completions.create(**kwargs):
        deadline.check()
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
        for call in getattr(delta, "tool_calls", None) or []:
            fn = getattr(call, "function", None)
            args = getattr(fn, "arguments", None) or ""
            calls.setdefault(getattr(call, "index", 0) or 0, []).append(args)
    if calls:
        emitter.feed_content(
            merge_tool_arguments(
                ["".join(calls[i]) for i in sorted(calls)], request.schema
            )
        )
