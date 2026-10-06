"""TaskRouter: resolve a task id to a provider chain and invoke it.

Providers are plain async callables you register by name. The router owns
everything around the call: policy lookup, model resolution, timeouts,
per-provider circuit breakers, the fallback chain and a telemetry event per
attempt. Your provider adapter owns only "turn this Route + request into an
API call" — so swapping SDKs never touches routing logic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Mapping, Protocol

from .breaker import CircuitBreaker, CircuitOpenError
from .policy import RoutingPolicy, TaskPolicy


@dataclass(frozen=True)
class Route:
    """Everything a provider needs to make one call."""

    task_id: str
    provider: str
    model: str | None
    timeout: float | None
    reasoning_effort: str | None
    tool_calling: bool  # effective: policy asked for it AND provider supports it
    options: Mapping[str, Any]
    tier: str | None
    attempt: int  # 0 = primary, 1.. = fallbacks
    reason: str  # why this provider was chosen: "policy" | "fallback:<cause>"


class ProviderHandler(Protocol):
    def __call__(self, request: Any, route: Route) -> Awaitable[Any]: ...


@dataclass
class RouteEvent:
    """One attempt. Emitted to ``on_event`` for logging / metrics / tracing."""

    task_id: str
    provider: str
    model: str | None
    attempt: int
    reason: str
    outcome: str  # "ok" | "error" | "timeout" | "circuit_open" | "skipped"
    latency_ms: float = 0.0
    error: str | None = None


class AllProvidersFailed(RuntimeError):
    def __init__(self, task_id: str, events: list[RouteEvent]):
        self.task_id = task_id
        self.events = events
        chain = ", ".join(f"{e.provider}={e.outcome}" for e in events)
        super().__init__(f"task {task_id!r}: every provider failed ({chain})")


@dataclass
class _Registered:
    handler: ProviderHandler
    tool_calling: bool
    breaker: CircuitBreaker | None
    default_model: str | None = None


@dataclass
class TaskRouter:
    policy: RoutingPolicy
    on_event: Callable[[RouteEvent], None] | None = None
    default_timeout: float | None = 30.0
    # Decide whether an exception should trigger fallback. Return False for
    # errors another provider would hit too (e.g. a malformed request).
    should_fallback: Callable[[BaseException], bool] = lambda exc: True
    breaker_factory: Callable[[str], CircuitBreaker] = field(default=lambda name: CircuitBreaker(name))
    _providers: dict[str, _Registered] = field(default_factory=dict, init=False)

    @classmethod
    def from_yaml(cls, path: str, **kwargs: Any) -> "TaskRouter":
        """Router with providers and policy from one YAML file. See :mod:`thalamus.config`."""
        from .config import load_router

        return load_router(path, **kwargs)

    # ── registration ───────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        handler: ProviderHandler,
        *,
        tool_calling: bool = False,
        circuit_breaker: bool = True,
        default_model: str | None = None,
    ) -> None:
        """Register a provider.

        ``tool_calling``: the provider can bind native tool schemas. A task
        that asks for tool calling gets it only on providers that support it.
        ``default_model``: used when neither the task nor policy defaults name one.
        """
        self._providers[name] = _Registered(
            handler=handler,
            tool_calling=tool_calling,
            breaker=self.breaker_factory(name) if circuit_breaker else None,
            default_model=default_model,
        )

    def provider(self, name: str, **kwargs: Any) -> Callable[[ProviderHandler], ProviderHandler]:
        """Decorator form of :meth:`register`."""

        def decorate(fn: ProviderHandler) -> ProviderHandler:
            self.register(name, fn, **kwargs)
            return fn

        return decorate

    def reload_policy(self, policy: RoutingPolicy | None = None) -> None:
        """Swap in a new policy (or re-read the current one from disk)."""
        self.policy = policy or self.policy.reload()

    # ── planning ───────────────────────────────────────────────────────────

    def plan(self, task_id: str) -> list[Route]:
        """The ordered provider chain for a task, without calling anything."""
        task = self.policy.resolve(task_id)
        chain = [task.provider, *[p for p in task.fallback if p != task.provider]]
        routes: list[Route] = []
        for attempt, name in enumerate(dict.fromkeys(chain)):
            routes.append(self._route(task, name, attempt, "policy" if attempt == 0 else "fallback"))
        return routes

    def _route(self, task: TaskPolicy, provider: str, attempt: int, reason: str) -> Route:
        registered = self._providers.get(provider)
        if attempt == 0:
            model = task.model
        else:
            # A fallback provider cannot serve the primary's model id; use its own.
            model = self.policy.model_for(provider)
        if model is None and registered is not None:
            model = registered.default_model
        return Route(
            task_id=task.task_id,
            provider=provider,
            model=model,
            timeout=task.timeout if task.timeout is not None else self.default_timeout,
            reasoning_effort=task.reasoning_effort,
            tool_calling=task.tool_calling and bool(registered and registered.tool_calling),
            options=task.options,
            tier=task.tier,
            attempt=attempt,
            reason=reason,
        )

    # ── invocation ─────────────────────────────────────────────────────────

    async def invoke(self, task_id: str, request: Any) -> Any:
        """Call the task's provider chain until one succeeds."""
        events: list[RouteEvent] = []
        cause = ""
        for route in self.plan(task_id):
            if route.attempt > 0:
                route = replace(route, reason=f"fallback:{cause}")
            registered = self._providers.get(route.provider)
            if registered is None:
                cause = "unregistered"
                events.append(self._emit(route, "skipped", error="provider not registered"))
                continue

            breaker = registered.breaker
            if breaker is not None and not breaker.allow():
                cause = "circuit_open"
                events.append(self._emit(route, "circuit_open"))
                continue

            started = time.perf_counter()
            try:
                call = registered.handler(request, route)
                result = await (asyncio.wait_for(call, route.timeout) if route.timeout else call)
            except asyncio.TimeoutError as exc:
                cause = "timeout"
                if breaker:
                    breaker.record_failure()
                events.append(self._emit(route, "timeout", started, exc))
            except Exception as exc:  # noqa: BLE001 - provider errors are arbitrary
                cause = "error"
                if breaker:
                    breaker.record_failure()
                events.append(self._emit(route, "error", started, exc))
                if not self.should_fallback(exc):
                    raise
            else:
                if breaker:
                    breaker.record_success()
                self._emit(route, "ok", started)
                return result

        raise AllProvidersFailed(task_id, events)

    async def complete(self, task_id: str, prompt: Any, *, system: str | None = None) -> Any:
        """Chat-style convenience over :meth:`invoke`.

        ``prompt`` is a string, a messages list, or a ``{"messages", "tools"}``
        dict. ``system`` prepends a system message. With the built-in providers
        the result is a :class:`thalamus.providers.Completion`.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        if system is not None:
            if isinstance(prompt, Mapping):
                prompt = {**prompt, "messages": [{"role": "system", "content": system}, *prompt["messages"]]}
            else:
                prompt = [{"role": "system", "content": system}, *prompt]
        return await self.invoke(task_id, prompt)

    def complete_sync(self, task_id: str, prompt: Any, *, system: str | None = None) -> Any:
        """Blocking :meth:`complete` for scripts and notebooks without an event loop."""
        return asyncio.run(self.complete(task_id, prompt, system=system))

    def breaker_status(self) -> dict[str, dict[str, Any]]:
        return {
            name: reg.breaker.status() for name, reg in self._providers.items() if reg.breaker is not None
        }

    def _emit(
        self,
        route: Route,
        outcome: str,
        started: float | None = None,
        exc: BaseException | None = None,
        error: str | None = None,
    ) -> RouteEvent:
        event = RouteEvent(
            task_id=route.task_id,
            provider=route.provider,
            model=route.model,
            attempt=route.attempt,
            reason=route.reason,
            outcome=outcome,
            latency_ms=(time.perf_counter() - started) * 1000 if started is not None else 0.0,
            error=error or (f"{type(exc).__name__}: {exc}" if exc else None),
        )
        if self.on_event is not None:
            self.on_event(event)
        return event


__all__ = [
    "AllProvidersFailed",
    "CircuitOpenError",
    "ProviderHandler",
    "Route",
    "RouteEvent",
    "TaskRouter",
]
