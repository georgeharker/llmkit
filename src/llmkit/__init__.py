"""llmkit — a streaming LLM bridge + markdown rendering toolkit.

Two independent subpackages, disjoint dependency graphs:

  :mod:`llmkit.bridge`  streaming chat/complete over an OpenAI-compatible
                        endpoint (or the Claude Agent SDK), with a
                        reasoning/content stream splitter, output sinks,
                        and a typed provider/profile config parser. Install
                        with ``llmkit[bridge]`` (+ ``[claude]`` for the
                        claude_code adapter).
  :mod:`llmkit.md`      render markdown to a terminal: one-shot, live
                        streaming (rich), or a scrollable modal (textual).
                        Install with ``llmkit[md]``.

Extracted from the zsh-ai plugin; nothing here knows about zsh, zle, or
any host shell — the surface is plain argv/stdin in, streamed bytes out,
plus a dataclass API underneath.
"""

__version__ = "0.1.0"
