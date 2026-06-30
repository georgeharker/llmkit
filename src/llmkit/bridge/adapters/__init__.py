"""Chat producers — one per transport backend.

Each exposes a ``stream_*(provider, request, emitter)`` callable matching
:data:`llmkit.bridge.bridge.Producer`, translating the bridge's generic
request into that backend's native message/streaming format. Imported lazily
by :func:`llmkit.bridge.bridge.chat` so the default openai path never pays for
an optional dependency (anthropic / google-genai / claude-agent-sdk).

Backends: ``openai`` (openai-compatible HTTP), ``anthropic`` (native Messages
API), ``google`` (native Gemini API), ``claude_code`` (Claude Agent SDK). The
native HTTP adapters take a custom ``endpoint``, so any provider reachable over
one of these protocols — e.g. an OpenCode Zen gateway — is supported by
pointing the provider's endpoint at it.
"""
