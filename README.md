# llmkit

A streaming LLM bridge + terminal markdown renderer, in two independent
subpackages with disjoint dependencies. Nothing here knows about any host
shell — it's plain argv/stdin in, streamed bytes out, with a dataclass API
underneath. Extracted from [zsh-ai](https://github.com/georgeharker/zsh-ai).

| Subpackage | What | Install |
|---|---|---|
| `llmkit.bridge` | Streaming chat/complete over an OpenAI-compatible endpoint — or natively over the Anthropic Messages API / Google Gemini API / Claude Agent SDK — with a reasoning/content stream splitter, output sinks, and a typed provider/profile config parser. | `llmkit[bridge]` (+`[anthropic]`/`[google]`/`[claude]`) |
| `llmkit.md` | Render markdown to a terminal: one-shot, live streaming (rich), or a scrollable modal (textual). | `llmkit[md]` |

## Bridge

```python
from llmkit.bridge import Provider, ChatRequest, chat

chat(Provider(model="qwen2.5-coder:7b", endpoint="http://localhost:11434/v1"),
     ChatRequest(user="explain mmap in one line"),
     content="-", thinking="inline")
```

Or as a CLI: `python -m llmkit.bridge chat --model … --user "…"`.

Pick a backend with `adapter` / `--adapter`: `openai-compatible` (default),
`anthropic`, `google`, or `claude_code`. The native HTTP adapters honour the
provider's `endpoint` + `api_key`, so pointing one at a gateway (e.g. an
[OpenCode Zen](https://opencode.ai/docs/zen/) base URL) routes that protocol
through it. Anthropic's current models govern sampling internally, so the
`anthropic` adapter doesn't forward `temperature`; Gemini accepts it, so
`google` does. `complete` (FIM) is `openai-compatible`-only, and that's
queryable up front — `adapter_supports_complete` / `provider_supports_complete`
/ `profile_supports_complete` let a consumer disable a FIM widget for a
chat-only provider or profile instead of failing at call time.

The reusable surface is the dataclass API; the flag CLI is one adapter over
it. Config is typed and extensible — `[defaults]` / `[providers.*]` /
`[profiles.*]` parse into dataclasses, and you extend the schema by
subclassing `ConfigParser`/`Config` (the return type tracks your subclass,
no `Any`). Profile keys are opaque, so a consumer assigns their meaning.

### Provider config

```toml
[defaults]                      # merged beneath every provider
temperature = 1.0

[providers.ollama-glm]
adapter     = "openai-compatible"
endpoint    = "https://ollama.com/v1"
api_key_env = "OLLAMA_API_KEY"
model       = "glm-5.3-flash"
max_tokens  = 8192
timeout     = 90
structured  = "tool"

[profiles.bulk]                 # opaque key -> provider name
summarise = "ollama-glm"
```

`load(path).resolve("ollama-glm")` gives the provider with `[defaults]`
beneath it; `select(profile, key)` maps a profile key to a provider name.
Every key is optional. The bridge fills a few defaults at call time (shown).

| Key | Meaning |
|---|---|
| `adapter` | `openai-compatible` (default), `anthropic`, `google`, or `claude_code`. |
| `model` | Model name, passed through. For `claude_code`, anything the `claude` CLI accepts. |
| `endpoint` | Base URL (default `http://localhost:11434/v1`). `anthropic` takes the same `/v1`-style URL and strips the tail the SDK adds itself. Ignored by `claude_code`. |
| `api_key` / `api_key_env` | A literal key, or the env var holding it. `api_key_env` wins. Unset: `anthropic`/`google` read their SDK's own env vars; `openai-compatible` sends a placeholder (local servers don't check). |
| `max_tokens` | Output cap (default 1024). Ignored by `claude_code`. |
| `temperature` | Default 0.2. Not sent by `anthropic` (current models reject it) or `claude_code`. |
| `enable_thinking` | `auto` (default) / `true` / `false`. Per adapter: `chat_template_kwargs` for openai-compatible, adaptive thinking for anthropic, `thinking_config` for google, the SDK thinking config for claude_code (`true` caps it at `$LLMKIT_CLAUDE_THINKING_BUDGET`, default 8192). Forced off for structured output. |
| `headers` | Extra HTTP headers for the HTTP adapters, as a table or a list of `KEY=VALUE`. Names are lowercased; values expand `$VAR`/`${VAR}` plus `${uuid}`/`${epoch}`. Merges per key over `[defaults]`. |
| `timeout` | Seconds; bounds the whole call. See [Timeouts](#timeouts). |
| `structured` | How `openai-compatible` asks for structured output: `response_format` (default) or `tool`. See [Structured output](#structured-output). Any other value is a config error. |
| `stop` | Parsed and carried for consumers. The bridge doesn't send it: FIM stop tokens come from `CompleteRequest.stop`. |

Unknown keys are kept in `Provider.extra`, so a config written for a newer
llmkit still loads on an older one.

### Structured output

Give `ChatRequest` a JSON Schema and call `chat_structured`:

```python
code, data = chat_structured(provider, ChatRequest(
    user=source, system="Describe each top-level symbol.",
    schema={"type": "object", "properties": {"symbols": {"type": "array"}},
            "required": ["symbols"]},
    schema_name="describe_symbols"))
```

The return contract: `(0, data)` where `data` fits the schema, or
`(non-zero, None)` with the reason on stderr. A reply with nothing usable is
always a failure, never `(0, None)`, so a caller can count failures and back
off. Thinking is turned off for structured calls.

Each adapter asks for the schema through its native channel:

| Adapter | Channel |
|---|---|
| `openai-compatible` | `response_format` json_schema, with the schema also appended to the system prompt. With `structured = "tool"`: one forced tool call whose parameters are the schema. |
| `anthropic` | One forced `tool_use`. |
| `google` | JSON response mode, the schema natively (where the installed SDK supports it), and the schema in the system prompt. |
| `claude_code` | The Agent SDK's `output_format`. |

A server can accept a structured-output request with HTTP 200 and ignore
it. **Ollama Cloud does exactly that**: its docs say it "currently does not
support structured outputs" ([ollama/ollama#12362](https://github.com/ollama/ollama/issues/12362)),
and `response_format`, `json_object` and the native `format` all come back as
prose. Forced tool calls do work there, so set `structured = "tool"` on an
Ollama Cloud provider. Some local Ollama setups drop `format` too (e.g. on
the MLX backend, or with `think=false` on some models).

So llmkit never trusts the channel. It parses the reply leniently (the whole
text, else a fenced code block, else a JSON value in prose) and keeps the
first top-level value that fits the schema. If the schema is an object with
a single array property, a bare array is wrapped into it. If a model splits
one answer across parallel tool calls, their arrays are concatenated. The
schema check covers `type`, `enum`, `required`, `properties` and `items`.

### Timeouts

`timeout` (seconds; also `Provider(timeout=…)` or `--timeout`) bounds one
call end to end:

- `openai-compatible` / `anthropic`: the SDK client timeout (connect, and
  each socket read, so a stalled stream raises), with SDK retries off so a
  retry can't restart the clock. `google`: the HTTP timeout.
- All HTTP adapters also check a total deadline between stream chunks, so a
  slow trickle is bounded too.
- `claude_code`: the query is cancelled, which stops the CLI.

A read that stalls just before the deadline can overshoot it by up to one
more `timeout`. Unset, each SDK keeps its own default: 600 s per read for
openai and anthropic. A host's own `asyncio.wait_for` around a call can only
abandon it; the worker thread keeps waiting until this timeout fires.

### Embedding in a long-lived host

llmkit is meant to run inside daemons, not only as a CLI:

- **No `fork()`.** `claude_code` starts the `claude` CLI with `posix_spawn`.
  On macOS, once a process has used Network.framework (e.g. through an HTTP
  client like DuckDB's), its fork handler crashes the child of any later
  `fork()` before it can exec. The spawn helper (`llmkit.bridge._spawn`)
  refuses options that would force a fork. Keep the host's fds 0-2 open
  (`/dev/null` is fine): if a pipe lands on one, CPython falls back to fork.
- **Catchable errors.** A missing optional SDK raises `ImportError`, never
  `SystemExit`. Call failures come back as exit codes from `chat` and
  `chat_structured`.

## Markdown

```sh
python -m llmkit.md.render --stream < stream.md   # live re-render to stdout
python -m llmkit.md.view  -                        # scrollable modal, follows stdin
```

`LiveMarkdownStream` mirrors textual's `Markdown.get_stream` API, so the same
code drives a rich `Console` or a textual widget.
