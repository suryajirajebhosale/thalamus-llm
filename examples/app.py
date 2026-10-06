"""Using Thalamus inside an application with real providers.

    pip install -e ".[all]"
    cp .env.example .env              # add the keys for the providers you use
    python examples/app.py

Delete the providers (and the tasks pointing at them) from thalamus.yaml that
you don't have keys for, or pass skip_unavailable=True as below so the router
falls back past them.
"""

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from thalamus import AllProvidersFailed, RouteEvent, TaskRouter

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("app")


def record(event: RouteEvent) -> None:
    # Ship these to your logs / metrics / tracing (Langfuse, OpenTelemetry, Prometheus...).
    log.info("route task=%s provider=%s model=%s outcome=%s reason=%s latency_ms=%.0f",
             event.task_id, event.provider, event.model, event.outcome, event.reason, event.latency_ms)


router = TaskRouter.from_yaml(
    str(Path(__file__).with_name("thalamus.yaml")),
    on_event=record,
    skip_unavailable=True,  # providers whose key isn't set are skipped, not fatal
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


async def main() -> None:
    # 1. Plain chat: the policy decides provider + model; code only names the task.
    reply = await router.complete("chat_reply", "Give me one tip for writing clear commit messages.",
                                  system="You are concise.")
    print(f"\nchat_reply via {reply.provider}/{reply.model}:\n{reply.text}\n")

    # 2. Tool calling: tools are sent only to providers that support them.
    plan = await router.complete("plan_steps", {
        "messages": [{"role": "user", "content": "Should I bring an umbrella in Pune today?"}],
        "tools": [WEATHER_TOOL],
    })
    for call in plan.tool_calls:
        print(f"plan_steps wants tool {call.name}({call.arguments})")

    # 3. When every provider in the chain fails you get one error listing each attempt.
    try:
        await router.complete("summarize_thread", "...")
    except AllProvidersFailed as exc:
        for event in exc.events:
            print(f"  {event.provider}: {event.outcome} {event.error or ''}")


if __name__ == "__main__":
    asyncio.run(main())
