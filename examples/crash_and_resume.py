"""The crash the project exists for, in four acts. Needs no API key.

Runs an agent that charges a card, kills the process mid-charge, and shows the
customer being charged exactly once across the crash and the resume.

    python examples/crash_and_resume.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentrt.agent import build_agent
from agentrt.checkpoint import SqliteCheckpointer
from agentrt.errors import UncertainSideEffect
from agentrt.providers import ScriptedProvider, calls, says
from agentrt.runtime import Runtime
from agentrt.state import AgentState
from agentrt.tools import ToolRegistry, tool
from agentrt.types import Message

LEDGER = Path(tempfile.gettempdir()) / "agentrt_demo_ledger.txt"


class ProcessKilled(BaseException):
    """Not an Exception, so nothing in the runtime catches it."""


@tool(side_effecting=True)
def charge_card(amount_eur: float) -> str:
    """Charge the customer's card."""
    with LEDGER.open("a") as fh:
        fh.write(f"charged {amount_eur}\n")
    raise ProcessKilled  # the machine dies the instant after the charge lands


@tool(side_effecting=True, idempotent=True)
def set_order_status(status: str) -> str:
    """Set the order status. Setting it twice is the same as setting it once."""
    return f"status={status}"


def charges() -> int:
    return len(LEDGER.read_text().splitlines()) if LEDGER.exists() else 0


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


def main() -> None:
    LEDGER.unlink(missing_ok=True)
    db = Path(tempfile.gettempdir()) / "agentrt_demo.db"
    db.unlink(missing_ok=True)

    registry = ToolRegistry([charge_card, set_order_status])
    graph = build_agent().compile()
    store = SqliteCheckpointer(db)

    def runtime(script):
        return Runtime(graph=graph, provider=ScriptedProvider(script), tools=registry,
                       checkpointer=store)

    banner("1. run, and die mid-charge")
    try:
        runtime([calls("charge_card", {"amount_eur": 49.0})]).run(
            AgentState(messages=[Message(role="user", content="charge 49 euros")]),
            run_id="demo",
        )
    except ProcessKilled:
        print("process killed mid-tool")
    print(f"charges in ledger: {charges()}")

    banner("2. what the database knows")
    for record in store.dangling_invocations("demo"):
        print(f"  {record.tool_name}  call={record.call_id}  status={record.status}")
    print("\n  a started record with no completed record: the outcome is unknown")

    banner("3. a naive resume would charge again, so this one refuses")
    try:
        runtime([says("done")]).resume("demo")
    except UncertainSideEffect as exc:
        print(f"  refused: {exc}")
    print(f"\n  run status: {store.run_status('demo')}")
    print(f"charges in ledger: {charges()}  (still one)")

    banner("4. an operator confirms what happened, then the run continues")
    store.resolve_invocation("demo", "call_1", "charged 49.0 (confirmed in the payment log)")
    result = runtime([says("Order complete. The card was charged once.")]).resume("demo")
    print(f"  status: {result.status}")
    print(f"  final:  {result.last_message.content}")
    print(f"\ncharges in ledger: {charges()}  (still one)")

    banner("the point")
    print(
        "The charge happened exactly once across a crash and a resume, and the\n"
        "runtime never guessed. An idempotent tool like set_order_status would\n"
        "have been re-run automatically instead of stopping for a human."
    )


if __name__ == "__main__":
    main()