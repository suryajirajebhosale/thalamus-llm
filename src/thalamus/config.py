"""Build a ready-to-use router from one YAML file: providers and policy together.

    providers:
      openai:
        type: openai                    # openai | anthropic | echo
        api_key_env: OPENAI_API_KEY     # name of the env var holding the key
      claude:
        type: anthropic
        api_key_env: ANTHROPIC_API_KEY
      local:
        type: openai
        base_url: http://localhost:11434/v1   # Ollama, vLLM, LM Studio...
        tool_calling: false

    defaults: ...                       # the routing policy, see policy.py
    tasks: ...

Keys are never written in the file, only the *name* of the environment
variable that holds them.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from . import providers as builtin
from .policy import PolicyError, RoutingPolicy
from .router import ProviderHandler, TaskRouter

logger = logging.getLogger(__name__)

_SPEC_KEYS = {
    "type",
    "api_key_env",
    "base_url",
    "base_url_env",
    "headers",
    "tool_calling",
    "circuit_breaker",
    "default_model",
    "max_tokens",
}
_DEFAULT_TOOL_CALLING = {"openai": True, "anthropic": True, "echo": False}


class ConfigError(PolicyError):
    """The providers section is malformed or a provider can't be built."""


def _build_openai(spec: Mapping[str, Any], api_key: str | None, base_url: str | None) -> ProviderHandler:
    if api_key is None and base_url and not spec.get("api_key_env"):
        # Local servers (Ollama, vLLM, LM Studio) ignore the key, but the SDK insists on one.
        api_key = "not-needed"
    return builtin.openai_compatible(api_key=api_key, base_url=base_url, default_headers=spec.get("headers"))


def _build_anthropic(spec: Mapping[str, Any], api_key: str | None, base_url: str | None) -> ProviderHandler:
    return builtin.anthropic(api_key=api_key, base_url=base_url, default_max_tokens=int(spec.get("max_tokens", 16000)))


def _build_echo(spec: Mapping[str, Any], api_key: str | None, base_url: str | None) -> ProviderHandler:
    return builtin.echo()


PROVIDER_TYPES: dict[str, Callable[[Mapping[str, Any], str | None, str | None], ProviderHandler]] = {
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "echo": _build_echo,
}


def load_router(
    path: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    skip_unavailable: bool = False,
    **router_kwargs: Any,
) -> TaskRouter:
    """Read a YAML file with ``providers`` + ``defaults`` + ``tasks`` and return a TaskRouter.

    ``skip_unavailable``: if a provider's API key env var is unset, leave it
    unregistered (the router then skips it and uses the fallback) instead of
    raising. Handy when developing with only some keys.
    """
    policy = RoutingPolicy.from_yaml(path)
    import yaml

    document = yaml.safe_load(Path(path).read_text()) or {}
    return build_router(document, policy=policy, env=env, skip_unavailable=skip_unavailable, **router_kwargs)


def build_router(
    document: Mapping[str, Any],
    *,
    policy: RoutingPolicy | None = None,
    env: Mapping[str, str] | None = None,
    skip_unavailable: bool = False,
    **router_kwargs: Any,
) -> TaskRouter:
    env = os.environ if env is None else env
    router = TaskRouter(policy or RoutingPolicy.from_dict(document), **router_kwargs)
    specs = document.get("providers") or {}
    if not isinstance(specs, Mapping):
        raise ConfigError("providers must be a mapping of name -> settings")

    for name, spec in specs.items():
        if not isinstance(spec, Mapping):
            raise ConfigError(f"providers.{name} must be a mapping")
        unknown = set(spec) - _SPEC_KEYS
        if unknown:
            raise ConfigError(f"providers.{name}: unknown keys {sorted(unknown)}")
        kind = spec.get("type")
        if kind not in PROVIDER_TYPES:
            raise ConfigError(f"providers.{name}.type must be one of {sorted(PROVIDER_TYPES)}")

        missing = [spec[k] for k in ("api_key_env", "base_url_env") if spec.get(k) and not env.get(spec[k])]
        handler = None
        if missing:
            problem = f"providers.{name}: environment variable {' and '.join(missing)} is not set"
        else:
            api_key = env.get(spec["api_key_env"]) if spec.get("api_key_env") else None
            base_url = spec.get("base_url") or (env.get(spec["base_url_env"]) if spec.get("base_url_env") else None)
            try:
                handler = PROVIDER_TYPES[kind](spec, api_key, base_url)
            except ImportError:
                raise  # missing SDK: the message already says which extra to install
            except Exception as exc:  # noqa: BLE001 - SDK constructors raise their own error types
                problem = f"providers.{name}: could not create client ({exc})"
        if handler is None:
            if skip_unavailable:
                logger.warning("%s; provider skipped", problem)
                continue
            raise ConfigError(problem)

        router.register(
            name,
            handler,
            tool_calling=bool(spec.get("tool_calling", _DEFAULT_TOOL_CALLING[kind])),
            circuit_breaker=bool(spec.get("circuit_breaker", True)),
            default_model=spec.get("default_model"),
        )

    problems = router.policy.validate(providers=specs.keys()) if specs else []
    if problems:
        raise ConfigError("invalid routing config:\n  " + "\n  ".join(problems))
    return router
