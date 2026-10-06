"""Declarative task -> model routing policy.

A policy is a YAML (or dict) document that says, for every LLM *task* your
application performs, which provider serves it, which model, with what timeout
and options, and where to fall back when that provider is unhealthy::

    defaults:
      provider: primary
      timeout: 30
      fallback: [backup]
      models:
        primary: my-org/small-model
        backup: gpt-4o-mini

    tasks:
      extraction:                 # group name doubles as the default tier
        summarize_thread:
          provider: primary
          timeout: 12
      user_facing:
        chat_reply:
          provider: backup
          model: gpt-4o
          reasoning_effort: low

Code asks for a task by id and never hardcodes a model name. Changing which
model serves a task is a config edit, not a deploy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

# Keys with first-class meaning. Anything else on a task entry is kept in
# ``options`` and handed to the provider untouched (temperature, max_tokens...).
_KNOWN_KEYS = {
    "provider",
    "model",
    "timeout",
    "tier",
    "reasoning_effort",
    "tool_calling",
    "fallback",
}
# Union of provider vocabularies: OpenAI uses none/minimal..high, Claude low..xhigh/max.
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


class PolicyError(ValueError):
    """The policy document is malformed."""


@dataclass(frozen=True)
class TaskPolicy:
    """Routing settings for one task, after defaults have been applied."""

    task_id: str
    provider: str
    model: str | None = None
    timeout: float | None = None
    tier: str | None = None
    reasoning_effort: str | None = None
    tool_calling: bool = False
    fallback: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    configured: bool = True  # False when the task id was not in the policy


@dataclass(frozen=True)
class PolicyDefaults:
    provider: str | None = None
    timeout: float | None = None
    fallback: tuple[str, ...] = ()
    models: Mapping[str, str] = field(default_factory=dict)


class RoutingPolicy:
    """An immutable, validated view over a routing policy document."""

    def __init__(self, document: Mapping[str, Any], *, source: Path | None = None):
        self._source = source
        self.defaults, self._tasks = _parse(document)

    # ── construction ───────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "RoutingPolicy":
        return cls(document)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RoutingPolicy":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError("Install the yaml extra: pip install 'thalamus-llm[yaml]'") from exc
        path = Path(path)
        with path.open() as fh:
            document = yaml.safe_load(fh) or {}
        return cls(document, source=path)

    def reload(self) -> "RoutingPolicy":
        """Re-read the source file. Returns a new policy; this one is unchanged."""
        if self._source is None:
            raise PolicyError("policy was not loaded from a file; nothing to reload")
        return type(self).from_yaml(self._source)

    # ── lookup ─────────────────────────────────────────────────────────────

    @property
    def task_ids(self) -> list[str]:
        return sorted(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def resolve(self, task_id: str) -> TaskPolicy:
        """Settings for ``task_id`` with policy defaults filled in.

        An unknown task does not raise: it resolves to the defaults with
        ``configured=False``, so a new call site works before its policy
        entry exists. Use :meth:`validate` in CI to catch missing entries.
        """
        entry = self._tasks.get(task_id)
        if entry is None:
            if self.defaults.provider is None:
                raise PolicyError(
                    f"task {task_id!r} is not in the policy and no defaults.provider is set"
                )
            entry = TaskPolicy(task_id=task_id, provider=self.defaults.provider, configured=False)

        return replace(
            entry,
            # The model lives in ONE place (defaults.models) unless a task pins
            # its own, so renaming a deployment is a one-line change.
            model=entry.model or self.defaults.models.get(entry.provider),
            timeout=entry.timeout if entry.timeout is not None else self.defaults.timeout,
            fallback=entry.fallback or self.defaults.fallback,
        )

    def model_for(self, provider: str) -> str | None:
        """The policy-wide default model for ``provider``."""
        return self.defaults.models.get(provider)

    # ── checks ─────────────────────────────────────────────────────────────

    def validate(
        self,
        *,
        providers: Iterable[str] | None = None,
        required_tasks: Iterable[str] | None = None,
    ) -> list[str]:
        """Return a list of human-readable problems (empty means OK).

        ``providers``: provider names your app actually registers.
        ``required_tasks``: task ids your code calls (e.g. collected at import).
        """
        problems: list[str] = []
        known = set(providers) if providers is not None else None

        def check_provider(name: str, where: str) -> None:
            if known is not None and name not in known:
                problems.append(f"{where}: unknown provider {name!r}")

        if self.defaults.provider:
            check_provider(self.defaults.provider, "defaults.provider")
        for name in self.defaults.fallback:
            check_provider(name, "defaults.fallback")

        for task_id, task in sorted(self._tasks.items()):
            check_provider(task.provider, f"tasks.{task_id}.provider")
            for name in task.fallback:
                check_provider(name, f"tasks.{task_id}.fallback")
            resolved = self.resolve(task_id)
            if resolved.model is None:
                problems.append(
                    f"tasks.{task_id}: no model and no defaults.models.{task.provider}"
                )
            # Inherited defaults.fallback may include the task's own provider
            # (the router skips it); an explicit self-fallback is a mistake.
            if task.provider in task.fallback:
                problems.append(f"tasks.{task_id}: falls back to its own provider")

        for task_id in required_tasks or ():
            if task_id not in self._tasks:
                problems.append(f"tasks.{task_id}: called by code but missing from policy")
        return problems


# ── parsing ────────────────────────────────────────────────────────────────


def _parse(document: Mapping[str, Any]) -> tuple[PolicyDefaults, dict[str, TaskPolicy]]:
    if not isinstance(document, Mapping):
        raise PolicyError("policy document must be a mapping")

    raw_defaults = document.get("defaults") or {}
    if not isinstance(raw_defaults, Mapping):
        raise PolicyError("defaults must be a mapping")
    defaults = PolicyDefaults(
        provider=raw_defaults.get("provider"),
        timeout=_timeout(raw_defaults.get("timeout"), "defaults.timeout"),
        fallback=_names(raw_defaults.get("fallback"), "defaults.fallback"),
        models=dict(raw_defaults.get("models") or {}),
    )

    raw_tasks = document.get("tasks") or {}
    if not isinstance(raw_tasks, Mapping):
        raise PolicyError("tasks must be a mapping")

    tasks: dict[str, TaskPolicy] = {}
    for key, value in raw_tasks.items():
        if not isinstance(value, Mapping):
            raise PolicyError(f"tasks.{key} must be a mapping")
        # A group is a mapping whose values are all mappings and that has no
        # task-level keys of its own; anything else is a task entry.
        is_group = value and all(isinstance(v, Mapping) for v in value.values()) and not (
            set(value) & _KNOWN_KEYS
        )
        entries = value.items() if is_group else [(key, value)]
        group = key if is_group else None
        for task_id, entry in entries:
            if task_id in tasks:
                raise PolicyError(f"task {task_id!r} is defined more than once")
            tasks[task_id] = _task(task_id, entry, group, defaults)
    return defaults, tasks


def _task(task_id: str, entry: Mapping[str, Any], group: str | None, defaults: PolicyDefaults) -> TaskPolicy:
    where = f"tasks.{task_id}"
    provider = entry.get("provider") or defaults.provider
    if not provider:
        raise PolicyError(f"{where}: no provider and no defaults.provider")

    effort = entry.get("reasoning_effort")
    if effort is not None:
        effort = str(effort).strip().lower()
        if effort not in _REASONING_EFFORTS:
            raise PolicyError(f"{where}.reasoning_effort must be one of {sorted(_REASONING_EFFORTS)}")

    return TaskPolicy(
        task_id=task_id,
        provider=str(provider),
        model=entry.get("model"),
        timeout=_timeout(entry.get("timeout"), f"{where}.timeout"),
        tier=entry.get("tier") or group,
        reasoning_effort=effort,
        tool_calling=bool(entry.get("tool_calling", False)),
        fallback=_names(entry.get("fallback"), f"{where}.fallback"),
        options={k: v for k, v in entry.items() if k not in _KNOWN_KEYS},
    )


def _timeout(value: Any, where: str) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        raise PolicyError(f"{where} must be a number of seconds") from None
    if seconds <= 0:
        raise PolicyError(f"{where} must be positive")
    return seconds


def _names(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise PolicyError(f"{where} must be a provider name or a list of names")
