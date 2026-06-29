"""Output sinks for content / thinking streams.

A "sink" is the destination for one of the two logical streams that
:class:`llmkit.bridge.stream.StreamSplitter` produces. A caller picks each
via a string spec (e.g. the ``--content`` / ``--thinking`` flags of the
CLI); this module normalises that spec into something the splitter and the
chat command can use.
"""
from __future__ import annotations

import sys
from typing import TextIO, Union

# `Sink` is the resolved form. ``'inline'`` is a thinking-side sentinel
# meaning "emit reasoning into the content sink, separated from real
# content by a blank line". Renderers then render thinking as ordinary
# prose; no wrapping tags or prefixes to confuse them.
Sink = Union[None, str, TextIO]


def open_target(spec: str) -> Sink:
    """Resolve a ``--content`` / ``--thinking`` spec to a sink.

    Returns:
      ``None`` for ``none`` (drop entirely),
      the literal sentinel ``'inline'`` (the StreamSplitter interprets it),
      ``sys.stdout`` for ``-``,
      an open writable file for any other value (treated as a path).

    Files are opened line-buffered so a downstream reader sees output
    promptly.
    """
    if spec == "none":
        return None
    if spec == "-":
        return sys.stdout
    if spec == "inline":
        return "inline"
    return open(spec, "w", buffering=1)


def close_sink(sink: Sink) -> None:
    """Close a sink if it owns its file handle."""
    if sink is None or sink == "inline":
        return
    if sink in (sys.stdout, sys.stderr):
        return
    try:
        sink.close()  # type: ignore[union-attr]
    except Exception:
        pass
