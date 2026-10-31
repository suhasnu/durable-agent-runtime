"""One real call against the Anthropic API, or a conversation of them.

    export ANTHROPIC_API_KEY=...
    python examples/live.py "what is 17 percent of 3400?"
    python examples/live.py --chat

Each question is its own run. A finished run cannot be resumed, because its
last checkpoint points at END, so a follow-up starts a new run seeded with the
previous run's messages. The conversation lives in the state carried forward,
not in the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentrt import AgentState, Budget, Message, Runtime, SqliteCheckpointer, ToolRegistry, tool
from agentrt.agent import build_agent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SYSTEM = """Answer in plain prose, not bullet points. Use the calculator for \
arithmetic rather than doing it in your head, and say so when a question needs \
information you do not have."""


@tool
def calculate(expression: str) -> str:
    """Evaluate a arithmetic expression, such as "17/100*3400"."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "only numbers and + - * / ( ) are allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:
        return f"could not evaluate: {exc}"


def build(db: str) -> Runtime:
    from agentrt.providers import AnthropicProvider

    registry = ToolRegistry([calculate])
    return Runtime(
        graph=build_agent().compile(),
        provider=AnthropicProvider(),
        tools=registry,
        checkpointer=SqliteCheckpointer(db),
        system=SYSTEM,
        budget=Budget(max_total_tokens=40_000, max_model_calls=12),
        max_steps=20,
    )


def ask(runtime: Runtime, history: list[Message], question: str) -> list[Message]:
    """Run one question and return the message list to carry into the next."""
    state = AgentState(messages=[*history, Message(role="user", content=question)])
    result = runtime.run(state)

    if result.status != "completed":
        print(f"\n[{result.status}] {result.error or result.interrupt}")
        return list(result.state.messages)

    print(f"\n{result.last_message.content}")
    print(
        f"\n  run {result.run_id}  steps={result.steps}  "
        f"calls={result.state.usage.calls}  tokens={result.state.usage.total}"
    )
    return list(result.state.messages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--chat", action="store_true", help="keep asking, carrying the history")
    parser.add_argument("--db", default="live.db")
    args = parser.parse_args()

    runtime = build(args.db)
    history: list[Message] = []

    if args.question:
        history = ask(runtime, history, " ".join(args.question))

    if not args.chat:
        if not args.question:
            parser.error("give a question, or --chat")
        return 0

    print("\nblank line or ctrl-c to stop")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not question:
            return 0
        history = ask(runtime, history, question)


if __name__ == "__main__":
    raise SystemExit(main())
