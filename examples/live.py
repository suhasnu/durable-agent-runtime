"""One real call, to prove the adapter works. Costs about a tenth of a cent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentrt.agent import build_agent
from agentrt.providers import AnthropicProvider
from agentrt.runtime import Runtime
from agentrt.state import AgentState
from agentrt.tools import ToolRegistry, tool
from agentrt.types import Message


@tool
def city_population(city: str) -> str:
    """Look up the population of a German city."""
    data = {"frankfurt": "773,000", "fulda": "69,000", "berlin": "3,850,000"}
    return data.get(city.lower(), f"no data for {city}")


def main() -> None:
    runtime = Runtime(
        graph=build_agent().compile(),
        provider=AnthropicProvider(),
        tools=ToolRegistry([city_population]),
        system="Use the tools available to you. Answer in one sentence.",
    )
    result = runtime.run(
        AgentState(messages=[
            Message(role="user", content="How many more people live in Frankfurt than in Fulda?")
        ])
    )
    if result.status != "completed":
        print(f"\nrun failed after {result.steps} steps: {result.error}")

    for msg in result.state.messages:
        print(f"\n[{msg.role}] {msg.content or msg.tool_calls or msg.tool_results}")

    print(f"\nnodes: {' -> '.join(result.history)}")
    print(f"tokens: {result.state.usage.input_tokens} in, {result.state.usage.output_tokens} out")


if __name__ == "__main__":
    main()