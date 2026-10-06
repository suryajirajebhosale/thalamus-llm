"""Thalamus: policy-driven LLM routing with fallback chains and circuit breakers."""

from .breaker import BreakerState, CircuitBreaker, CircuitOpenError
from .config import ConfigError, build_router, load_router
from .policy import PolicyError, RoutingPolicy, TaskPolicy
from .providers import Completion, ToolCall
from .router import AllProvidersFailed, Route, RouteEvent, TaskRouter

__all__ = [
    "AllProvidersFailed",
    "BreakerState",
    "CircuitBreaker",
    "CircuitOpenError",
    "Completion",
    "ConfigError",
    "PolicyError",
    "Route",
    "RouteEvent",
    "RoutingPolicy",
    "TaskPolicy",
    "TaskRouter",
    "ToolCall",
    "build_router",
    "load_router",
]
__version__ = "0.1.0"
