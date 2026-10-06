import asyncio
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from thalamus import Completion, ConfigError, TaskRouter, build_router
from thalamus.__main__ import main as cli
from thalamus.providers import anthropic, openai_compatible

EXAMPLES = Path(__file__).parent.parent / "examples"

WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "description": "weather", "parameters": {"type": "object"}},
}


def run(coro):
    return asyncio.run(coro)


class FakeOpenAI:
    def __init__(self, tool_call=False):
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))
        self.tool_call = tool_call

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        calls = [NS(id="c1", function=NS(name="get_weather", arguments='{"city": "Pune"}'))] if self.tool_call else None
        return NS(
            model=kwargs["model"] + "-2026",
            choices=[NS(message=NS(content="hi from openai", tool_calls=calls))],
            usage=NS(prompt_tokens=7, completion_tokens=3),
        )


class FakeAnthropic:
    def __init__(self):
        self.calls = []
        self.messages = NS(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return NS(
            model=kwargs["model"],
            content=[NS(type="text", text="hi from claude"),
                     NS(type="tool_use", id="t1", name="get_weather", input={"city": "Pune"})],
            usage=NS(input_tokens=11, output_tokens=5),
        )


def router_with(**providers):
    doc = {
        "defaults": {"provider": "a", "models": {"a": "model-a"}},
        "tasks": {
            "chat": {"provider": "a", "temperature": 0.2, "reasoning_effort": "low"},
            "tools": {"provider": "a", "tool_calling": True},
        },
    }
    router = TaskRouter(build_router(doc).policy)
    for name, (handler, tools) in providers.items():
        router.register(name, handler, tool_calling=tools)
    return router


# ── adapters ────────────────────────────────────────────────────────────────


def test_openai_adapter_maps_route_and_normalises_response():
    client = FakeOpenAI()
    router = router_with(a=(openai_compatible(client=client), True))
    result = run(router.complete("chat", "hello", system="be brief"))

    sent = client.calls[0]
    assert sent["model"] == "model-a"
    assert sent["temperature"] == 0.2 and sent["reasoning_effort"] == "low"
    assert sent["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
    assert "tools" not in sent
    assert isinstance(result, Completion)
    assert (result.text, result.provider, result.model) == ("hi from openai", "a", "model-a-2026")
    assert result.usage == {"input_tokens": 7, "output_tokens": 3}


def test_openai_tools_sent_only_when_route_allows_and_calls_parsed():
    client = FakeOpenAI(tool_call=True)
    request = {"messages": [{"role": "user", "content": "umbrella?"}], "tools": [WEATHER_TOOL]}

    result = run(router_with(a=(openai_compatible(client=client), True)).complete("tools", request))
    assert client.calls[-1]["tools"] == [WEATHER_TOOL]
    assert result.tool_calls[0].name == "get_weather" and result.tool_calls[0].arguments == {"city": "Pune"}

    run(router_with(a=(openai_compatible(client=client), False)).complete("tools", request))
    assert "tools" not in client.calls[-1]  # provider registered without tool support


def test_anthropic_adapter_lifts_system_converts_tools_and_defaults_max_tokens():
    client = FakeAnthropic()
    router = router_with(a=(anthropic(client=client, default_max_tokens=321), True))
    request = {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "umbrella?"}],
               "tools": [WEATHER_TOOL]}
    result = run(router.complete("tools", request))

    sent = client.calls[0]
    assert sent["system"] == "be brief"
    assert sent["messages"] == [{"role": "user", "content": "umbrella?"}]
    assert sent["max_tokens"] == 321
    assert sent["tools"] == [{"name": "get_weather", "description": "weather", "input_schema": {"type": "object"}}]
    assert "reasoning_effort" not in sent and "output_config" not in sent
    assert result.text == "hi from claude"
    assert result.tool_calls[0].arguments == {"city": "Pune"}
    assert result.usage == {"input_tokens": 11, "output_tokens": 5}


def test_anthropic_maps_effort_and_drops_sampling_params():
    client = FakeAnthropic()
    router = router_with(a=(anthropic(client=client), True))
    run(router.complete("chat", "hello"))  # task has temperature: 0.2 and reasoning_effort: low
    sent = client.calls[0]
    assert sent["output_config"] == {"effort": "low"}
    assert "temperature" not in sent and "reasoning_effort" not in sent
    assert sent["max_tokens"] == 16000


# ── config ──────────────────────────────────────────────────────────────────


def test_demo_config_routes_out_of_the_box():
    router = TaskRouter.from_yaml(str(EXAMPLES / "demo.yaml"))
    result = router.complete_sync("chat_reply", "hello")
    assert result.text == "[claude/claude-opus-5-5] hello"
    fallback = router.plan("summarize_thread")
    assert [r.provider for r in fallback] == ["hosted", "claude", "openai"]


def test_missing_api_key_is_a_clear_error_or_skipped():
    doc = {
        "providers": {"openai": {"type": "openai", "api_key_env": "MY_OPENAI_KEY"}, "backup": {"type": "echo"}},
        "defaults": {"provider": "openai", "fallback": ["backup"], "models": {"openai": "m", "backup": "e"}},
        "tasks": {"t": {}},
    }
    with pytest.raises(ConfigError, match="MY_OPENAI_KEY is not set"):
        build_router(doc, env={})

    router = build_router(doc, env={}, skip_unavailable=True)
    assert run(router.complete("t", "hi")).text == "[backup/e] hi"  # fell past the missing provider


def test_key_present_builds_real_adapter():
    pytest.importorskip("openai")
    doc = {"providers": {"openai": {"type": "openai", "api_key_env": "K"}},
           "defaults": {"provider": "openai", "models": {"openai": "m"}}}
    router = build_router(doc, env={"K": "sk-test"})
    assert router.plan("anything")[0].tool_calling is False  # no task asked for tools


def test_keyless_local_server_and_unset_base_url_env():
    pytest.importorskip("openai")
    doc = {"providers": {"local": {"type": "openai", "base_url_env": "LOCAL_URL"},
                         "backup": {"type": "echo"}},
           "defaults": {"provider": "local", "fallback": ["backup"], "models": {"local": "llama", "backup": "e"}}}
    router = build_router(doc, env={"LOCAL_URL": "http://localhost:11434/v1"})
    assert "local" in router._providers  # no api_key_env needed for a local server

    with pytest.raises(ConfigError, match="LOCAL_URL is not set"):
        build_router(doc, env={})
    skipped = build_router(doc, env={}, skip_unavailable=True)
    assert run(skipped.complete("t", "hi")).text == "[backup/e] hi"


def test_example_config_with_no_keys_and_skip_unavailable():
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    router = TaskRouter.from_yaml(str(EXAMPLES / "thalamus.yaml"), env={}, skip_unavailable=True)
    assert router._providers == {}
    router = TaskRouter.from_yaml(str(EXAMPLES / "thalamus.yaml"),
                                  env={"ANTHROPIC_API_KEY": "sk-ant-test"}, skip_unavailable=True)
    assert set(router._providers) == {"claude"}


@pytest.mark.parametrize(
    "providers, match",
    [
        ({"x": {"type": "bogus"}}, "type must be one of"),
        ({"x": {"type": "echo", "api_key": "sk-literal"}}, "unknown keys"),
        ({"x": "openai"}, "must be a mapping"),
    ],
)
def test_bad_provider_specs_rejected(providers, match):
    with pytest.raises(ConfigError, match=match):
        build_router({"providers": providers, "defaults": {"provider": "x", "models": {"x": "m"}}}, env={})


def test_tasks_pointing_at_undeclared_providers_rejected():
    doc = {"providers": {"a": {"type": "echo"}}, "defaults": {"models": {"a": "m", "ghost": "g"}},
           "tasks": {"t": {"provider": "ghost"}}}
    with pytest.raises(ConfigError, match="unknown provider 'ghost'"):
        build_router(doc, env={})


# ── CLI ─────────────────────────────────────────────────────────────────────


def test_cli_ask_check_explain(capsys):
    assert cli(["ask", str(EXAMPLES / "demo.yaml"), "chat_reply", "hello"]) == 0
    out, err = capsys.readouterr()
    assert out.strip() == "[claude/claude-opus-5-5] hello"
    assert "claude/claude-opus-5-5 ok (policy" in err

    assert cli(["check", str(EXAMPLES / "thalamus.yaml")]) == 0
    assert cli(["explain", str(EXAMPLES / "thalamus.yaml"), "chat_reply"]) == 0
    assert "fallback    openai  model=gpt-4o-mini" in capsys.readouterr().out
