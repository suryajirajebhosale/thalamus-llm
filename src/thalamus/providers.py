"""Built-in provider adapters.

Every adapter takes the same request and returns the same :class:`Completion`,
so a task can move between providers with a one-line policy change.

Request: either a list of OpenAI-style chat messages::

    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

or a dict with ``messages`` plus optional ``tools`` (OpenAI function-tool
format). Tools are sent only when the route allows tool calling.

Adapters:

- :func:`openai_compatible` covers OpenAI and anything that speaks its Chat
  Completions API: Azure OpenAI, Google Gemini (OpenAI endpoint), Groq,
  Together, OpenRouter, DeepSeek, Mistral, Ollama, vLLM, LM Studio, LiteLLM.
- :func:`anthropic` covers Claude models through the Anthropic Messages API.
- :func:`echo` echoes the prompt back. It needs no keys, which makes it handy
  for trying a policy, tests and CI.

SDKs are imported lazily. Install only what you use:
``pip install "thalamus-llm[openai]"`` / ``"thalamus-llm[anthropic]"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .router import ProviderHandler, Route

Messages = list[dict[str, Any]]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Completion:
    """Provider-neutral result of one routed call."""

    text: str
    provider: str
    model: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None  # provider's own value, e.g. "stop", "end_turn", "tool_use", "refusal"
    usage: dict[str, int] = field(default_factory=dict)  # input_tokens / output_tokens
    raw: Any = field(default=None, repr=False)  # the SDK's own response object


def split_request(request: Any) -> tuple[Messages, list[dict[str, Any]]]:
    """Accept a messages list, a {"messages", "tools"} dict, or a bare prompt string."""
    if isinstance(request, str):
        return [{"role": "user", "content": request}], []
    if isinstance(request, Mapping):
        return list(request["messages"]), list(request.get("tools") or [])
    return list(request), []


# ── OpenAI-compatible ───────────────────────────────────────────────────────


def openai_compatible(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    default_headers: Mapping[str, str] | None = None,
    client: Any = None,
) -> ProviderHandler:
    """Adapter for the OpenAI Chat Completions API and compatible servers."""
    if client is None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError('OpenAI provider needs the SDK: pip install "thalamus-llm[openai]"') from exc
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)

    async def handler(request: Any, route: Route) -> Completion:
        messages, tools = split_request(request)
        kwargs: dict[str, Any] = {"model": route.model, "messages": messages, **route.options}
        if route.reasoning_effort:
            kwargs["reasoning_effort"] = route.reasoning_effort
        if tools and route.tool_calling:
            kwargs["tools"] = tools
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        calls = [
            ToolCall(c.id, c.function.name, _json_args(c.function.arguments))
            for c in (getattr(message, "tool_calls", None) or [])
        ]
        usage = getattr(response, "usage", None)
        return Completion(
            text=message.content or "",
            provider=route.provider,
            model=getattr(response, "model", route.model),
            tool_calls=calls,
            stop_reason=getattr(choice, "finish_reason", None),
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            }
            if usage
            else {},
            raw=response,
        )

    return handler


# ── Anthropic ───────────────────────────────────────────────────────────────


# Current Claude models reject sampling parameters (and anthropic>=1.0 removed
# them). A task's options are shared with its fallback providers, where
# `temperature: 0` is meaningful, so they are dropped here rather than failing.
_ANTHROPIC_UNSUPPORTED = frozenset({"temperature", "top_p", "top_k"})

# Claude's effort levels are low..max; the OpenAI-only "none"/"minimal" map to the lowest.
_ANTHROPIC_EFFORT = {"none": "low", "minimal": "low"}


def anthropic(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    default_max_tokens: int = 16000,
    client: Any = None,
) -> ProviderHandler:
    """Adapter for Claude via the Anthropic Messages API.

    - System messages are lifted into the ``system`` parameter.
    - OpenAI-format tools are converted to Anthropic tools.
    - ``reasoning_effort`` becomes ``output_config.effort`` (``none``/``minimal`` -> ``low``).
    - ``max_tokens`` (required by Anthropic) defaults to ``default_max_tokens``.
    - ``temperature``/``top_p``/``top_k`` are dropped: current Claude models reject them.
    - Other policy options (``thinking``, ``output_config``, ``metadata``...) pass through.
    - Check ``Completion.stop_reason`` for ``"refusal"``; the text is empty in that case.
    """
    if client is None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError('Anthropic provider needs the SDK: pip install "thalamus-llm[anthropic]"') from exc
        client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    async def handler(request: Any, route: Route) -> Completion:
        messages, tools = split_request(request)
        system = "\n\n".join(str(m["content"]) for m in messages if m["role"] == "system")
        options = {k: v for k, v in route.options.items() if k not in _ANTHROPIC_UNSUPPORTED}
        if route.reasoning_effort:
            effort = _ANTHROPIC_EFFORT.get(route.reasoning_effort, route.reasoning_effort)
            options["output_config"] = {**options.get("output_config", {}), "effort": effort}
        kwargs: dict[str, Any] = {
            "model": route.model,
            "messages": [m for m in messages if m["role"] != "system"],
            "max_tokens": options.pop("max_tokens", default_max_tokens),
            **options,
        }
        if system:
            kwargs["system"] = system
        if tools and route.tool_calling:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        response = await client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"]
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            provider=route.provider,
            model=getattr(response, "model", route.model),
            tool_calls=calls,
            stop_reason=getattr(response, "stop_reason", None),
            usage={"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens} if usage else {},
            raw=response,
        )

    return handler


def _to_anthropic_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    fn = tool.get("function", tool)
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _json_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw": arguments}


# ── echo (no keys) ──────────────────────────────────────────────────────────


def echo() -> ProviderHandler:
    """Returns the last user message, tagged with the route. For demos and tests."""

    async def handler(request: Any, route: Route) -> Completion:
        messages, _ = split_request(request)
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return Completion(text=f"[{route.provider}/{route.model}] {last}", provider=route.provider, model=route.model)

    return handler
