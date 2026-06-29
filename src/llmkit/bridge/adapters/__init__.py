"""Chat producers — one per transport backend.

Each exposes a ``stream_*(provider, request, emitter)`` callable matching
:data:`llmkit.bridge.bridge.Producer`. Imported lazily by
:func:`llmkit.bridge.bridge.chat` so the default openai path never pays for
the optional claude_code dependency.
"""
