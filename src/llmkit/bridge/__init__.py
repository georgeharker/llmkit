"""llmkit.bridge — streaming chat/complete bridge + typed config.

Talks to any OpenAI-compatible HTTP endpoint (via the openai SDK), natively to
the Anthropic Messages API / Google Gemini API, or the Claude Agent SDK. A
stream splitter separates reasoning ("thinking") from content so callers can
route them to different consumers.

The reusable surface is the dataclass API: build a :class:`Provider`, then
call :func:`chat` / :func:`complete`. The flag CLI (``python -m
llmkit.bridge``) is one adapter over it. Adapter capabilities are queryable —
:func:`adapter_supports_complete` / :func:`provider_supports_complete` /
:func:`profile_supports_complete` answer "is FIM available here?" up front, so
a consumer can disable a FIM widget for chat-only adapters rather than failing
at call time.
"""

from .bridge import (
    ADAPTERS,
    ChatRequest,
    CompleteRequest,
    adapter_supports_complete,
    chat,
    complete,
    profile_supports_complete,
    provider_supports_complete,
    with_defaults,
)
from .config import Config, ConfigParser, Provider, TomlValue, load

__all__ = [
    # config
    "Config",
    "ConfigParser",
    "Provider",
    "TomlValue",
    "load",
    # runtime
    "ChatRequest",
    "CompleteRequest",
    "chat",
    "complete",
    "with_defaults",
    # adapter capabilities (queryable up front)
    "ADAPTERS",
    "adapter_supports_complete",
    "provider_supports_complete",
    "profile_supports_complete",
]
