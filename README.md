# llmkit

A streaming LLM bridge + terminal markdown renderer, in two independent
subpackages with disjoint dependencies. Nothing here knows about any host
shell — it's plain argv/stdin in, streamed bytes out, with a dataclass API
underneath. Extracted from [zsh-ai](https://github.com/georgeharker/zsh-ai).

| Subpackage | What | Install |
|---|---|---|
| `llmkit.bridge` | Streaming chat/complete over an OpenAI-compatible endpoint (or the Claude Agent SDK), with a reasoning/content stream splitter, output sinks, and a typed provider/profile config parser. | `llmkit[bridge]` (+`[claude]`) |
| `llmkit.md` | Render markdown to a terminal: one-shot, live streaming (rich), or a scrollable modal (textual). | `llmkit[md]` |

## Bridge

```python
from llmkit.bridge import Provider, ChatRequest, chat

chat(Provider(model="qwen2.5-coder:7b", endpoint="http://localhost:11434/v1"),
     ChatRequest(user="explain mmap in one line"),
     content="-", thinking="inline")
```

Or as a CLI: `python -m llmkit.bridge chat --model … --user "…"`.

The reusable surface is the dataclass API; the flag CLI is one adapter over
it. Config is typed and extensible — `[defaults]` / `[providers.*]` /
`[profiles.*]` parse into dataclasses, and you extend the schema by
subclassing `ConfigParser`/`Config` (the return type tracks your subclass,
no `Any`). Profile keys are opaque, so a consumer assigns their meaning.

## Markdown

```sh
python -m llmkit.md.render --stream < stream.md   # live re-render to stdout
python -m llmkit.md.view  -                        # scrollable modal, follows stdin
```

`LiveMarkdownStream` mirrors textual's `Markdown.get_stream` API, so the same
code drives a rich `Console` or a textual widget.
