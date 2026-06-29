"""llmkit.md — render markdown to a terminal.

Two entry points, one rich/textual pipeline:

  :mod:`llmkit.md.render`  one-shot or live-streaming render to stdout
                           (``python -m llmkit.md.render``).
  :mod:`llmkit.md.view`    scrollable, follow-tailing modal viewer
                           (``python -m llmkit.md.view``).

The streaming renderer (:class:`llmkit.md.render.stream.LiveMarkdownStream`)
mirrors textual's ``Markdown.get_stream`` API, so callers can swap a rich
Console for a textual widget without restructuring.
"""
