"""OpenAI client construction + small shared helpers.

Kept tiny on purpose — the SDK does the heavy lifting; this module just
resolves the endpoint, API key, and the ``enable_thinking`` extra-body
the same way for every subcommand. Works off a resolved :class:`Provider`
(see :func:`llmkit.bridge.bridge.with_defaults`).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from .config import Provider


def build_client(provider: Provider) -> Any:
    """Construct the OpenAI client. Imports inside so ``--help`` works even
    before the dependency is installed. Return type is ``Any`` to avoid a
    hard import of ``openai.OpenAI`` at module load."""
    try:
        from openai import OpenAI
    except ImportError:
        print(
            "llmkit: the openai package is not installed. "
            "Install it with `pip install llmkit[bridge]` (or `pip install "
            "openai`).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return OpenAI(
        base_url=provider.endpoint, api_key=_resolve_api_key(provider)
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
