"""llmkit.bridge — streaming chat/complete bridge + typed config.

Talks to any OpenAI-compatible HTTP endpoint (via the openai SDK) or the
Claude Agent SDK. A stream splitter separates reasoning ("thinking") from
content so callers can route them to different consumers.

The reusable surface is the dataclass API: build a :class:`Provider`, then
call :func:`chat` / :func:`complete`. The flag CLI (``python -m
llmkit.bridge``) is one adapter over it.
"""

from .bridge import ChatRequest, CompleteRequest, chat, complete, with_defaults
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
]
