# Thalamus LLM

[![CI](https://github.com/suryajirajebhosale/thalamus-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/suryajirajebhosale/thalamus-llm/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Policy-driven LLM routing: every task goes to the right model, with automatic
fallback when a provider fails.**

In the brain, the thalamus is the relay station. Almost every incoming signal passes
through it and is sent on to the right region. Thalamus does the same for the LLM
calls in your application.

---

## What it is

Thalamus is a small Python library and CLI that sits between your code and your LLM
providers. Your code never names a model. It names a **task**:

```python
reply = await router.complete("chat_reply", "How do I reset my password?")
```

One YAML file decides, for every task:

- **which provider** serves it (OpenAI, Claude, Gemini, a local Ollama or vLLM model…)
- **which model**, and with what **timeout**, **reasoning effort**, **tool calling**
  and options
- **where to fall back** when that provider is down, slow or erroring

Moving `summarize_thread` from GPT-4o to a free local model, or sending `chat_reply`
to Claude, is then a one-line config edit. You don't touch code or redeploy.

## What it does

| | |
|---|---|
| **Task → model routing** | A YAML policy maps each task id to a provider, model, timeout, reasoning effort, tool calling and extra options (`max_tokens`, …). |
| **Fallback chains** | Per task, or as a default for every task. When the primary fails, the next provider is tried with *its own* model. |
| **Circuit breakers** | One per provider, using a rolling failure-rate window. Once a provider is failing, calls skip straight to the fallback instead of each waiting out a timeout. After a cool-down, a single probe call checks whether it has recovered. |
| **Built-in providers** | `openai` (OpenAI and every OpenAI-compatible API: Azure, Gemini, Groq, OpenRouter, Together, DeepSeek, Mistral, Ollama, vLLM, LM Studio, LiteLLM), `anthropic` (Claude), and `echo` (no keys, for trying things out). |
| **One request/response shape** | Send OpenAI-style messages (plus optional tools) and get back a `Completion` with `text`, `tool_calls`, `stop_reason` and `usage` from any provider. |
| **Capability gating** | Tools are sent only to providers that support them. Claude gets effort as `output_config.effort` and has the sampling parameters it rejects removed. |
| **Observability hook** | One `RouteEvent` per attempt (provider, model, outcome, latency, *why* it was chosen, e.g. `fallback:circuit_open`), ready for logs or metrics. |
| **CLI** | `thalamus ask` routes a real call from the terminal, `thalamus check` validates config in CI, and `thalamus explain` shows how a task resolves. |
| **Keys stay out of config** | The YAML holds only the *name* of each key's environment variable. |

---

## Try it in 60 seconds (no API keys)

```bash
git clone https://github.com/suryajirajebhosale/thalamus-llm.git
cd thalamus-llm
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

thalamus ask examples/demo.yaml chat_reply "hello"
# [thalamus] claude/claude-opus-5-5 ok (policy, 0 ms)
# [claude/claude-opus-5-5] hello

thalamus explain examples/demo.yaml summarize_thread
python examples/quickstart.py      # watch a flaky provider trip its breaker and fall back
```

`examples/demo.yaml` uses the `echo` provider, which replies with the prompt tagged by
the provider and model that served it. You can see every routing decision without
spending a token.

## Route real LLM calls

```bash
pip install -e ".[all]"            # openai + anthropic SDKs + python-dotenv
cp .env.example .env               # paste in the keys for the providers you use
thalamus ask examples/thalamus.yaml chat_reply "Give me one tip for clear commit messages"
```

[`examples/thalamus.yaml`](examples/thalamus.yaml) is a complete starting config with
OpenAI, Claude, Gemini and a local model. Copy it into your project and delete the
providers you don't use. If you only have some keys, add `--skip-unavailable`: providers
without a key are skipped, and their tasks fall back to the ones you do have.

---

## Use it in your code

```python
from thalamus import TaskRouter

router = TaskRouter.from_yaml("thalamus.yaml")           # providers + policy, built once at start-up

# Plain chat: a string, or a list of OpenAI-style messages
reply = await router.complete("chat_reply", "Summarise this ticket: ...", system="Be concise.")
print(reply.text, reply.provider, reply.model, reply.usage)

# Tool calling: tools go only to providers that support them
result = await router.complete("plan_steps", {
    "messages": [{"role": "user", "content": "Do I need an umbrella in Pune?"}],
    "tools": [{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
})
for call in result.tool_calls:
    print(call.name, call.arguments)

# No event loop (scripts, notebooks)
reply = router.complete_sync("chat_reply", "hello")
```

**Observability:** pass `on_event` to receive every attempt:

```python
router = TaskRouter.from_yaml("thalamus.yaml", on_event=lambda e: log.info(
    "llm task=%s provider=%s model=%s outcome=%s reason=%s ms=%.0f",
    e.task_id, e.provider, e.model, e.outcome, e.reason, e.latency_ms))
```

**Failure:** if every provider in the chain fails, `AllProvidersFailed` is raised, and
its `.events` list says what happened at each step.

**Your own provider:** any `async (request, route) -> result` function works. Register
it under a name the policy uses. It can also replace a YAML-declared provider:

```python
async def my_provider(request, route):           # route.model, route.timeout, route.options, ...
    return await my_internal_client.generate(model=route.model, messages=request)

router.register("internal", my_provider, tool_calling=False)
```

A full runnable app example is in [`examples/app.py`](examples/app.py).

---

## How a call is routed

```
router.complete("chat_reply", messages)
  └─ policy: chat_reply → provider=claude, model=claude-opus-5-5, fallback=[openai]
       └─ for provider in [claude, openai]:
            ├─ no key / not registered?   → skip       (event: skipped)
            ├─ circuit breaker open?      → skip       (event: circuit_open)
            ├─ call with the task timeout
            │    ├─ success               → return Completion        (event: ok)
            │    └─ error or timeout      → record failure, next provider (event: error | timeout)
       └─ nothing succeeded → AllProvidersFailed(events)
```

---

## Configuration reference

One YAML file, three sections:

```yaml
providers:            # what you can call (names are yours to choose)
  openai:
    type: openai
    api_key_env: OPENAI_API_KEY

defaults:             # applies to every task unless the task overrides it
  provider: openai
  timeout: 30
  fallback: [openai]
  models:
    openai: gpt-4o-mini

tasks:                # what your code calls; group names become each task's `tier`
  user_facing:
    chat_reply:
      provider: openai
      model: gpt-4o
```

### `providers`

| Key | Meaning |
|---|---|
| `type` | **Required.** `openai` (OpenAI and compatible APIs), `anthropic` (Claude) or `echo` (no keys) |
| `api_key_env` | *Name* of the environment variable holding the key. Never put the key itself here. |
| `base_url` / `base_url_env` | Endpoint (or the env var holding it) for OpenAI-compatible servers |
| `headers` | Extra HTTP headers (e.g. OpenRouter's `HTTP-Referer`) |
| `tool_calling` | Whether this provider may receive tools. Defaults to true for `openai`/`anthropic`. |
| `circuit_breaker` | `false` to disable the breaker for this provider |
| `default_model` | Model to use when neither the task nor `defaults.models` names one |
| `max_tokens` | `anthropic` only: default `max_tokens` (16000) when a task doesn't set one |

**Common OpenAI-compatible endpoints**, all with `type: openai`:

| Provider | `base_url` | Key env var (suggested) |
|---|---|---|
| OpenAI | *(omit)* | `OPENAI_API_KEY` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| Together | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` |
| DeepSeek | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| Ollama (local) | `http://localhost:11434/v1` | *(none)* |
| vLLM (self-hosted) | `http://<host>:8000/v1` | *(none, or your gateway's key)* |
| LM Studio (local) | `http://localhost:1234/v1` | *(none)* |

### `defaults` and `tasks`

| Key | Where | Meaning |
|---|---|---|
| `provider` | task, defaults | Provider name from the `providers` section |
| `model` | task | Pins a model for this task. Otherwise `defaults.models.<provider>` is used. |
| `models` | defaults | `{provider: model_id}`. Each model id lives in one place. |
| `timeout` | task, defaults | Seconds per attempt |
| `fallback` | task, defaults | Provider name or list, tried in order |
| `tier` | task | Free-form label. Defaults to the group name. |
| `reasoning_effort` | task | `none` `minimal` `low` `medium` `high` `xhigh` `max`. Sent to OpenAI as `reasoning_effort`, and to Claude as `output_config.effort` (`none`/`minimal` become `low`). |
| `tool_calling` | task | Ask for native tool binding (only honoured on providers that support it) |
| *anything else* | task | Passed to the provider as-is (`max_tokens`, `temperature`, `thinking`, …) |

**Claude notes:** current Claude models reject `temperature`, `top_p` and `top_k`, so the
Anthropic adapter drops them. That keeps one task config usable across providers, for
example `temperature: 0` for an OpenAI fallback. Check `Completion.stop_reason` for
`"refusal"`; the text is empty when Claude declines a request.

A task id that isn't in the policy doesn't crash. It resolves to `defaults`.
`thalamus check` (or `policy.validate(required_tasks=[...])`) catches missing entries in CI.

---

## Setup and API keys

### Requirements

| What | Needed for | Version |
|---|---|---|
| Python | everything | **3.10+** |
| `pyyaml` | reading config (installed automatically) | ≥ 6 |
| `openai` SDK | `type: openai` providers (OpenAI, Gemini, Groq, Ollama, vLLM, …) | ≥ 1.40 (`pip install -e ".[openai]"`) |
| `anthropic` SDK | `type: anthropic` providers (Claude) | ≥ 0.40 (`pip install -e ".[anthropic]"`) |
| `python-dotenv` | loading keys from a `.env` file (optional; plain env vars work too) | ≥ 1.0 (included in every extra) |
| `pytest` | running the tests | ≥ 8 (`pip install -e ".[dev]"`) |

```bash
pip install -e .                     # core + echo provider: enough for the demo
pip install -e ".[openai]"           # + OpenAI-compatible providers
pip install -e ".[anthropic]"        # + Claude
pip install -e ".[all]"              # everything
pip install -e ".[all,dev]"          # everything + tests
```

### Where the keys go

1. `cp .env.example .env`. `.env` is git-ignored, so never commit it.
2. Fill in only the keys for providers you use:

| Variable | Used by | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | `openai` provider | <https://platform.openai.com/api-keys> |
| `ANTHROPIC_API_KEY` | `claude` provider | <https://console.anthropic.com/settings/keys> |
| `GEMINI_API_KEY` | `gemini` provider | <https://aistudio.google.com/apikey> |
| `LOCAL_LLM_BASE_URL` | `local` provider (no key) | Your Ollama / vLLM / LM Studio URL, e.g. `http://localhost:11434/v1` |

3. The variable names are whatever your YAML's `api_key_env` / `base_url_env` say, so
   you can rename them as long as both files agree.

The `thalamus` CLI loads `.env` from the current directory automatically. In your own
app, either call `load_dotenv()` before `TaskRouter.from_yaml(...)`, or export the
variables however you manage secrets. Thalamus reads them from `os.environ` at
start-up. If a key is missing, start-up fails with a clear message naming the
variable, unless you pass `skip_unavailable=True`.

---

## CLI

```bash
thalamus ask     thalamus.yaml chat_reply "prompt" [--system "..."] [--skip-unavailable]
thalamus check   thalamus.yaml                 # exit 1 on problems; use in CI
thalamus explain thalamus.yaml chat_reply      # resolved provider, model, fallbacks, timeout, options
```

## Development

```bash
pip install -e ".[all,dev]"
pytest       # 42 tests: policy, breaker, router, config, CLI, and both adapters
             # run against the real openai/anthropic SDKs over a mocked HTTP layer
```

## License

MIT
