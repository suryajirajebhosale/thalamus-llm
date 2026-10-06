"""Run the adapters through the real openai / anthropic SDKs with a mocked HTTP layer.

Proves the adapters handle the SDKs' actual response objects, not just fakes.
Skipped when the SDKs aren't installed (pip install -e ".[all]").
"""

import asyncio
import json

import pytest

from thalamus import RoutingPolicy, TaskRouter
from thalamus.providers import anthropic, openai_compatible

try:  # openai>=3 / anthropic>=1 ship on httpx2; older SDK versions use httpx
    import httpx2 as httpx
except ImportError:
    httpx = pytest.importorskip("httpx")

POLICY = RoutingPolicy.from_dict({
    "defaults": {"provider": "p", "models": {"p": "test-model"}},
    "tasks": {"chat": {"tool_calling": True, "temperature": 0}},
})
TOOL = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}


def route_once(handler, request):
    router = TaskRouter(POLICY)
    router.register("p", handler, tool_calling=True)
    return asyncio.run(router.complete("chat", request))


def test_openai_sdk_roundtrip():
    openai = pytest.importorskip("openai")
    seen = {}

    def respond(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "test-model-0001",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "checking",
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "get_weather", "arguments": "{\"city\": \"Pune\"}"}}]}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        })

    client = openai.AsyncOpenAI(api_key="sk-test", base_url="http://llm.test/v1",
                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    result = route_once(openai_compatible(client=client),
                        {"messages": [{"role": "user", "content": "umbrella?"}], "tools": [TOOL]})

    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "test-model" and seen["body"]["temperature"] == 0
    assert seen["body"]["tools"] == [TOOL]
    assert result.text == "checking" and result.model == "test-model-0001"
    assert result.tool_calls[0].arguments == {"city": "Pune"}
    assert result.usage == {"input_tokens": 12, "output_tokens": 4}


def test_anthropic_sdk_roundtrip():
    sdk = pytest.importorskip("anthropic")
    seen = {}

    def respond(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        seen["key"] = request.headers["x-api-key"]
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "test-model",
            "stop_reason": "tool_use", "stop_sequence": None,
            "content": [{"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Pune"}}],
            "usage": {"input_tokens": 20, "output_tokens": 6},
        })

    client = sdk.AsyncAnthropic(api_key="sk-ant-test", base_url="http://claude.test",
                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    result = route_once(anthropic(client=client),
                        {"messages": [{"role": "system", "content": "be brief"},
                                      {"role": "user", "content": "umbrella?"}], "tools": [TOOL]})

    assert seen["url"] == "http://claude.test/v1/messages"
    assert seen["key"] == "sk-ant-test"
    assert seen["body"]["system"] == "be brief" and seen["body"]["max_tokens"] == 16000
    assert "temperature" not in seen["body"]  # policy sets it; current Claude models reject it
    assert seen["body"]["tools"][0]["input_schema"] == {"type": "object"}
    assert result.text == "checking" and result.stop_reason == "tool_use"
    assert result.tool_calls[0].name == "get_weather" and result.tool_calls[0].arguments == {"city": "Pune"}
    assert result.usage == {"input_tokens": 20, "output_tokens": 6}
