# llmkit

A streaming LLM bridge + markdown rendering toolkit for terminal apps.
Extracted from [zsh-ai](https://github.com/) — nothing here knows about
any host shell. Two independent subpackages with disjoint dependencies:

| Subpackage | What | Install |
|---|---|---|
| `llmkit.bridge` | Streaming chat/complete over an OpenAI-compatible endpoint (or the Claude Agent SDK), with a reasoning/content stream splitter, output sinks, and a typed provider/profile config parser. | `llmkit[bridge]` (+`[claude]`) |
| `llmkit.md` | Render markdown to a terminal: one-shot, live streaming (rich), or a scrollable modal (textual). | `llmkit[md]` |

## Config (`llmkit.bridge.config`)

Typed and extensible. Three generic tables — `[defaults]`, `[providers.*]`,
`[profiles.*]` — parse into dataclasses. Profiles map *opaque* selector
keys to provider names; the library never interprets the keys, so a
consumer gives them meaning (e.g. zsh-ai's ask/modify/question/fim
"widgets"). Extend by subclassing the parser, not by stuffing untyped
values into a bag:

```python
from llmkit.bridge.config import Config, ConfigParser, Provider

@dataclass(frozen=True)
class MyProvider(Provider):
    ...                       # extra typed fields

class MyParser(ConfigParser[MyProvider]):
    provider_cls = MyProvider
    def build(self, defaults, providers, profiles, data):
        return MyConfig(defaults, providers, profiles, ...)  # extra typed sections
```

The parser's return type tracks your subclass — no `Any`. The only
escape hatch for unknown *keys* is `Provider.extra`, typed as the closed
`TomlValue` union (TOML's value grammar).

## Markdown (`llmkit.md`)

```sh
python -m llmkit.md.render --stream < stream.md     # live re-render to stdout
python -m llmkit.md.view  -                          # scrollable modal, follows stdin
```
