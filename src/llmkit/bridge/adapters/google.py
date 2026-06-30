"""``google`` chat producer — native Google Gemini API, streamed, via the
official ``google-genai`` SDK.

Format differences from the openai-compatible path this copes with:

* ``system`` is the ``system_instruction`` config field, not a message.
* The user turn is ``contents``; tuning lives in a ``GenerateContentConfig``.
* Reasoning ("thought summaries") arrives as content parts flagged
  ``thought=True``, interleaved with answer parts — not a separate field.
* ``max_tokens`` is ``max_output_tokens``; thinking is a ``thinking_config``.
  Unlike current Anthropic models, Gemini accepts ``temperature``, so it IS
  forwarded here.

Point ``endpoint`` at a Gemini-compatible gateway (e.g. an OpenCode Zen base
URL) to route through it; leave it at the bridge default to hit Google directly.
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
        from google import genai  # type: ignore[import-untyped,import-not-found]
        from google.genai import types  # type: ignore[import-untyped,import-not-found]
    except ImportError:
        print(
            "llmkit: the google adapter needs the Google GenAI SDK. "
            "Install it with `pip install llmkit[google]` (or `pip install "
            "google-genai`).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return genai, types


def _resolve_api_key(provider: Provider) -> Optional[str]:
    """Resolve an explicit key, or None to let the SDK read its own env
    (``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``)."""
    if provider.api_key_env:
        return os.environ.get(provider.api_key_env) or None
    return provider.api_key or None


def _thinking_config(types: Any, flag: Optional[str]) -> Optional[Any]:
    """Map the bridge's auto|true|false onto a ``ThinkingConfig``.

    ``auto`` omits it (model default), matching the openai adapter. ``true``
    requests dynamic thinking (``thinking_budget=-1``) with thought summaries so
    reasoning streams. ``false`` disables it (``thinking_budget=0`` — honoured by
    Flash/Flash-Lite; Pro clamps to its minimum)."""
    if flag == "false":
        return types.ThinkingConfig(thinking_budget=0)
    if flag == "true":
        return types.ThinkingConfig(include_thoughts=True, thinking_budget=-1)
    return None  # auto → let the model decide


def stream_google(provider: Provider, request: ChatRequest, emitter: _Emitter) -> None:
    genai, types = _require_sdk()

    client_kwargs: dict[str, Any] = dict(api_key=_resolve_api_key(provider))
    # The bridge fills endpoint with the openai/ollama default; only forward an
    # endpoint the user actually customised (a Gemini-compatible gateway).
    if provider.endpoint and provider.endpoint != DEFAULT_ENDPOINT:
        client_kwargs["http_options"] = types.HttpOptions(base_url=provider.endpoint)
    client = genai.Client(**client_kwargs)

    config = types.GenerateContentConfig(
        system_instruction=request.system or None,
        max_output_tokens=provider.max_tokens,
        temperature=provider.temperature,
        thinking_config=_thinking_config(types, provider.enable_thinking),
    )

    for chunk in client.models.generate_content_stream(
        model=provider.model, contents=request.user, config=config
    ):
        emitter.signal_streaming()
        for candidate in getattr(chunk, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None) or ""
                if not text:
                    continue
                if getattr(part, "thought", False):
                    emitter.feed_reasoning(text)
                else:
                    emitter.feed_content(text)
