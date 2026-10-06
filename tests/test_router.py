import asyncio

import pytest

from thalamus import AllProvidersFailed, CircuitBreaker, RoutingPolicy, TaskRouter

POLICY = RoutingPolicy.from_dict(
    {
        "defaults": {"provider": "vendor", "timeout": 1, "models": {"hosted": "small", "vendor": "big"}},
        "tasks": {
            "classify": {"provider": "hosted", "fallback": ["vendor"], "timeout": 0.05},
            "plan": {"provider": "vendor", "model": "reasoner", "tool_calling": True, "reasoning_effort": "low"},
            "describe": {"provider": "hosted", "tool_calling": True},
        },
    }
)


def run(coro):
    return asyncio.run(coro)


def recording_router(**kwargs):
    events = []
    router = TaskRouter(POLICY, on_event=events.append, **kwargs)
    return router, events


async def echo(request, route):
    return (route.provider, route.model, request)


async def boom(request, route):
    raise ConnectionError("down")


def test_primary_route_carries_policy_settings():
    router, events = recording_router()
    seen = []

    async def vendor(request, route):
        seen.append(route)
        return "ok"

    router.register("vendor", vendor, tool_calling=True)
    assert run(router.invoke("plan", "x")) == "ok"
    route = seen[0]
    assert (route.model, route.reasoning_effort, route.tool_calling, route.reason) == ("reasoner", "low", True, "policy")
    assert [e.outcome for e in events] == ["ok"]


def test_tool_calling_requires_provider_support():
    router, _ = recording_router()
    router.register("hosted", echo, tool_calling=False)
    route = router.plan("describe")[0]
    assert route.tool_calling is False


def test_error_falls_back_with_fallback_providers_own_model():
    router, events = recording_router()
    router.register("hosted", boom)
    router.register("vendor", echo)
    assert run(router.invoke("classify", "x")) == ("vendor", "big", "x")
    assert [(e.provider, e.outcome) for e in events] == [("hosted", "error"), ("vendor", "ok")]
    assert events[1].reason == "fallback:error"


def test_timeout_falls_back():
    router, events = recording_router()

    async def slow(request, route):
        await asyncio.sleep(1)

    router.register("hosted", slow)
    router.register("vendor", echo)
    assert run(router.invoke("classify", "x"))[0] == "vendor"
    assert events[0].outcome == "timeout"
    assert events[1].reason == "fallback:timeout"


def test_open_breaker_skips_provider_without_calling_it():
    router, events = recording_router(
        breaker_factory=lambda name: CircuitBreaker(name, window_size=2, min_calls=2, open_timeout=60)
    )
    calls = []

    async def flaky(request, route):
        calls.append(1)
        raise ConnectionError

    router.register("hosted", flaky)
    router.register("vendor", echo)
    for _ in range(3):
        run(router.invoke("classify", "x"))
    assert len(calls) == 2  # third call short-circuited
    assert events[-2].outcome == "circuit_open"
    assert router.breaker_status()["hosted"]["state"] == "open"


def test_all_failed_raises_with_every_attempt():
    router, _ = recording_router()
    router.register("hosted", boom)
    router.register("vendor", boom)
    with pytest.raises(AllProvidersFailed) as info:
        run(router.invoke("classify", "x"))
    assert [e.provider for e in info.value.events] == ["hosted", "vendor"]


def test_should_fallback_false_reraises_immediately():
    router, events = recording_router(should_fallback=lambda exc: not isinstance(exc, ValueError))

    async def bad_request(request, route):
        raise ValueError("malformed")

    router.register("hosted", bad_request)
    router.register("vendor", echo)
    with pytest.raises(ValueError):
        run(router.invoke("classify", "x"))
    assert len(events) == 1


def test_unregistered_provider_is_skipped():
    router, events = recording_router()
    router.register("vendor", echo)
    assert run(router.invoke("classify", "x"))[0] == "vendor"
    assert events[0].outcome == "skipped"


def test_unknown_task_uses_defaults():
    router, _ = recording_router()
    router.register("vendor", echo)
    assert run(router.invoke("not_in_policy", "x")) == ("vendor", "big", "x")
