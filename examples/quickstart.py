"""No-keys demo: routing, fallback and the circuit breaker in action.

    pip install -e .
    python examples/quickstart.py

Loads examples/demo.yaml (all `echo` providers), then swaps the "hosted"
provider for one that fails 60% of the time, to show traffic falling back and
the breaker opening.
"""

import asyncio
import random
from pathlib import Path

from thalamus import Route, TaskRouter
from thalamus.providers import echo

router = TaskRouter.from_yaml(
    str(Path(__file__).with_name("demo.yaml")),
    on_event=lambda e: print(f"    [{e.outcome:>12}] {e.provider}/{e.model}  ({e.reason})"),
)

real_echo = echo()


async def flaky_hosted(request, route: Route):
    if random.random() < 0.6:
        raise ConnectionError("hosted endpoint unavailable")
    return await real_echo(request, route)


router.register("hosted", flaky_hosted)  # replaces the YAML-declared provider


async def main() -> None:
    for task in ["intent_classifier"] * 8 + ["chat_reply", "summarize_thread"]:
        print(f"{task}:")
        result = await router.complete(task, "hello")
        print(f"    -> {result.text}")
    print("\nbreakers:")
    for name, status in router.breaker_status().items():
        print(f"    {name}: {status['state']} (failure rate {status['failure_rate']})")


if __name__ == "__main__":
    asyncio.run(main())
