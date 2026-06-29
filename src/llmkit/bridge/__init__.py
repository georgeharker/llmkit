"""llmkit.bridge — streaming chat/complete bridge + typed config.

Talks to any OpenAI-compatible HTTP endpoint (via the openai SDK) or the
Claude Agent SDK. A stream splitter separates reasoning ("thinking") from
content so callers can route them to different consumers.

Currently exported: the typed config layer (:mod:`llmkit.bridge.config`).
The dataclass-first runtime API (``chat``/``complete``, adapters, CLI) is
ported in a follow-up step.
"""

from .config import Config, ConfigParser, Provider, TomlValue, load

__all__ = ["Config", "ConfigParser", "Provider", "TomlValue", "load"]
