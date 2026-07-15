"""OpenAI client construction + small shared helpers.

Kept tiny on purpose — the SDK does the heavy lifting; this module just
resolves the endpoint, API key, and the ``enable_thinking`` extra-body
the same way for every subcommand. Works off a resolved :class:`Provider`
(see :func:`llmkit.bridge.bridge.with_defaults`).
"""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Callable, Dict, Optional

from .config import Provider

# Dynamic ``${...}`` tokens usable in a header VALUE, generated fresh each time
# headers are resolved (once per bridge process, i.e. per widget invocation — so
# ${uuid} is unique per call). Env vars are expanded after these.
_HEADER_TOKENS: Dict[str, Callable[[], str]] = {
    "uuid": lambda: uuid.uuid4().hex,
    "epoch": lambda: str(int(time.time())),
}
_TOKEN_RE = re.compile(r"\$\{(\w+)\}")


def _expand_header_value(value: str) -> str:
    """Expand a header value: ``${uuid}``/``${epoch}`` dynamic tokens first
    (braces required — ``$uuid`` is NOT a token), then ``$VAR`` / ``${VAR}``
    from the environment (unknown vars left intact). There is no escape for a
    literal ``$``: a value whose ``$…`` happens to name an env var WILL be
    substituted."""
    value = _TOKEN_RE.sub(
        lambda m: _HEADER_TOKENS[m.group(1)]() if m.group(1) in _HEADER_TOKENS else m.group(0),
        value,
    )
    return os.path.expandvars(value)


def resolved_headers(provider: Provider) -> Optional[Dict[str, str]]:
    """The provider's custom headers with each value expanded (see
    :func:`_expand_header_value`), or ``None`` when there are none — the shape the
    SDKs' ``default_headers`` want. Shared by all three HTTP adapters."""
    if not provider.headers:
        return None
    return {k: _expand_header_value(v) for k, v in provider.headers.items()}


def build_client(provider: Provider) -> Any:
    """Construct the OpenAI client. Imports inside so ``--help`` works even
    before the dependency is installed. Return type is ``Any`` to avoid a
    hard import of ``openai.OpenAI`` at module load."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "llmkit: the openai package is not installed. "
            "Install it with `pip install llmkit[bridge]` (or `pip install "
            "openai`)."
        ) from e
    return OpenAI(
        base_url=provider.endpoint,
        api_key=_resolve_api_key(provider),
        default_headers=resolved_headers(provider),
    )


def _resolve_api_key(provider: Provider) -> str:
    if provider.api_key_env:
        return os.environ.get(provider.api_key_env, "") or "placeholder"
    if provider.api_key:
        return provider.api_key
    # The SDK requires a non-empty string; local servers don't check it.
    return "placeholder"


def enable_thinking_extra(flag: Optional[str]) -> Dict[str, Any]:
    """Map ``enable_thinking`` to the ``chat_template_kwargs`` extra body."""
    if flag == "true":
        return {"chat_template_kwargs": {"enable_thinking": True}}
    if flag == "false":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}
