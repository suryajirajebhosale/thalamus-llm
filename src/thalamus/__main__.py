"""CLI.

    thalamus ask     config.yaml <task_id> "prompt" [--system "..."]   route a real call
    thalamus check   config.yaml [--providers a,b,c]                   validate (CI-friendly)
    thalamus explain config.yaml <task_id>                             show how a task resolves

API keys are read from the environment, or from a .env file in the current
directory when python-dotenv is installed.
"""

from __future__ import annotations

import argparse
import sys

from .policy import PolicyError, RoutingPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thalamus", description="Policy-driven LLM routing.")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="send a prompt through the router for a task")
    ask.add_argument("config", help="YAML file with providers + defaults + tasks")
    ask.add_argument("task_id")
    ask.add_argument("prompt")
    ask.add_argument("--system", help="system message")
    ask.add_argument("--skip-unavailable", action="store_true",
                     help="skip providers whose API key env var is unset instead of failing")

    check = sub.add_parser("check", help="validate a policy/config file")
    check.add_argument("config")
    check.add_argument("--providers", help="comma-separated provider names your app registers "
                                           "(defaults to the file's providers section)")

    explain = sub.add_parser("explain", help="show how a task resolves")
    explain.add_argument("config")
    explain.add_argument("task_id")

    args = parser.parse_args(argv)
    _load_dotenv()
    try:
        if args.command == "ask":
            return _ask(args)
        policy = RoutingPolicy.from_yaml(args.config)
    except PolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.command == "check":
        providers = args.providers.split(",") if args.providers else _declared_providers(args.config)
        problems = policy.validate(providers=providers)
        for problem in problems:
            print(problem)
        print(f"{len(policy.task_ids)} tasks, {len(problems)} problem(s)")
        return 1 if problems else 0

    task = policy.resolve(args.task_id)
    if not task.configured:
        print(f"note: {args.task_id!r} is not in the policy; showing defaults")
    print(f"task        {task.task_id}")
    print(f"tier        {task.tier}")
    print(f"provider    {task.provider}  model={task.model}")
    for name in task.fallback:
        if name == task.provider:
            continue  # the router skips it too
        print(f"fallback    {name}  model={policy.model_for(name)}")
    print(f"timeout     {task.timeout}")
    print(f"reasoning   {task.reasoning_effort}")
    print(f"tools       {task.tool_calling}")
    if task.options:
        print(f"options     {dict(task.options)}")
    return 0


def _ask(args: argparse.Namespace) -> int:
    from .config import load_router
    from .router import AllProvidersFailed

    def show(event):
        print(f"[thalamus] {event.provider}/{event.model} {event.outcome} ({event.reason}, "
              f"{event.latency_ms:.0f} ms){' ' + event.error if event.error else ''}", file=sys.stderr)

    router = load_router(args.config, on_event=show, skip_unavailable=args.skip_unavailable)
    try:
        result = router.complete_sync(args.task_id, args.prompt, system=args.system)
    except AllProvidersFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(getattr(result, "text", result))
    return 0


def _declared_providers(path: str) -> list[str] | None:
    import yaml

    with open(path) as fh:
        providers = (yaml.safe_load(fh) or {}).get("providers")
    return list(providers) if providers else None


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


if __name__ == "__main__":
    raise SystemExit(main())
