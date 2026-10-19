"""One real call, to prove the adapter works. Costs about a tenth of a cent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentrt.loop import run_agent
from agentrt.providers import AnthropicProvider
from agentrt.tools import ToolRegistry, tool


@tool
def city_population(city: str) -> str:
    """Look up the population of a German city."""
    data = {"frankfurt": "773,000", "fulda": "69,000", "berlin": "3,850,000"}
    return data.get(city.lower(), f"no data for {city}")


def main() -> None:
    messages, usage = run_agent(
        provider=AnthropicProvider(),
        registry=ToolRegistry([city_population]),
        user_message="How many more people live in Frankfurt than in Fulda?",
        system="Use the tools available to you. Answer in one sentence.",
    )

    for msg in messages:
        print(f"\n[{msg.role}] {msg.content or msg.tool_calls or msg.tool_results}")

    print(f"\ntokens: {usage.input_tokens} in, {usage.output_tokens} out")


if __name__ == "__main__":
    main()