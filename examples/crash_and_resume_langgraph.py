"""examples/crash_and_resume.py, against the LangGraph port. Needs no API key.

The same agent charges a card and the process dies the instant the charge
lands. Here the death is real: the run happens in a child process that calls
os._exit, so no exception handler, finally block or background flush gets to
run. A second process then resumes from the same SQLite file.

    pip install -e ".[langgraph]"
    python examples/crash_and_resume_langgraph.py

Observed with langgraph 1.2.19, langgraph-checkpoint 4.3.1 and
langgraph-checkpoint-sqlite 3.1.3, on Windows 11 with Python 3.12. The
results are written up in docs/langgraph-crash.md.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agentrt.providers import ScriptedProvider, calls, says  # noqa: E402
from agentrt.tools import ToolRegistry, tool  # noqa: E402
from agentrt.types import Completion, Message  # noqa: E402


def _load_port():
    spec = importlib.util.spec_from_file_location("langgraph_port", HERE / "langgraph_port.py")
    module = importlib.util.module_from_spec(spec)
    # LangGraph resolves the state's string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


port = _load_port()


class ConversationalProvider(ScriptedProvider):
    """Answers from the conversation instead of a fixed script.

    A fixed script has to know where the resumed run picks up, and where it
    picks up is the thing being observed. This one answers the way a model
    would: a tool result gets a final answer, anything else asks for the charge.
    """

    def __init__(self) -> None:
        super().__init__([])

    def complete(self, messages: list[Message], **kwargs: Any) -> Completion:
        self.requests.append({"messages": list(messages), **kwargs})
        if messages[-1].role == "tool":
            return says("Order complete.")
        return calls("charge_card", {"amount_eur": 49.0}, call_id=f"call_{len(self.requests)}")


# Child process ------------------------------------------------------------


def child(action: str, workdir: Path, durability: str) -> None:
    ledger = workdir / "ledger.txt"

    @tool(side_effecting=True)
    def charge_card(amount_eur: float) -> str:
        """Charge the customer's card."""
        with ledger.open("a") as fh:
            fh.write(f"charged {amount_eur}\n")
            fh.flush()
            os.fsync(fh.fileno())
        if action == "run":
            os._exit(137)  # the machine dies the instant after the charge lands
        return f"charged {amount_eur}"

    provider = ConversationalProvider()
    agent = port.build_langgraph_agent(
        provider, ToolRegistry([charge_card]), port.sqlite_saver(workdir / "lg.db")
    )
    config = port.thread("demo")

    if action == "run":
        agent.invoke(port.initial_state("charge 49 euros"), config, durability=durability)
        return

    report: dict[str, Any] = {}
    try:
        final = agent.invoke(None, config, durability=durability)
        report["final"] = final["messages"][-1].content
    except Exception as exc:  # noqa: BLE001 - reporting what happened is the point
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["model_calls"] = len(provider.requests)
    print(json.dumps(report))


# Parent process -----------------------------------------------------------


def spawn(action: str, workdir: Path, durability: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, __file__, "--child", action, str(workdir), durability],
        capture_output=True,
        text=True,
    )


def charges(workdir: Path) -> int:
    ledger = workdir / "ledger.txt"
    return len(ledger.read_text().splitlines()) if ledger.exists() else 0


def crash_then_resume(workdir: Path, durability: str) -> dict[str, Any]:
    workdir.mkdir(parents=True)
    killed = spawn("run", workdir, durability)
    after_crash = charges(workdir)
    resumed = spawn("resume", workdir, durability)
    report = json.loads(resumed.stdout.strip().splitlines()[-1])
    return {
        "exit_code": killed.returncode,
        "after_crash": after_crash,
        "after_resume": charges(workdir),
        **report,
    }


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="lg_crash_"))
    versions = ", ".join(
        f"{name} {importlib.metadata.version(name)}"
        for name in ("langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite")
    )
    print(f"tested against: {versions}")

    banner("1. run in a child process, and kill it mid-charge")
    first = root / "sync"
    first.mkdir()
    killed = spawn("run", first, "sync")
    print(f"child exit code: {killed.returncode}  (os._exit, nothing got to clean up)")
    print(f"charges in ledger: {charges(first)}")

    banner("2. what the database knows")
    agent = port.build_langgraph_agent(
        ConversationalProvider(), ToolRegistry([]), port.sqlite_saver(first / "lg.db")
    )
    snapshot = agent.get_state(port.thread("demo"))
    print(f"  latest checkpoint: step {snapshot.metadata['step']}, next {snapshot.next}")
    print(f"  pending tasks: {[(t.name, t.error) for t in snapshot.tasks]}")
    db = sqlite3.connect(first / "lg.db")
    for task_path, channel in db.execute("SELECT task_path, channel FROM writes"):
        print(f"  write  {task_path:<28} {channel}")
    db.close()
    print("\n  the model's output is saved. nothing records that the tool started.")

    banner("3. resume in a new process")
    resumed = spawn("resume", first, "sync")
    report = json.loads(resumed.stdout.strip().splitlines()[-1])
    print(f"  final: {report.get('final', report.get('error'))}")
    print(f"  model calls on resume: {report['model_calls']}")
    print(f"charges in ledger: {charges(first)}")
    print((first / "ledger.txt").read_text().rstrip())

    banner("4. the same crash under each durability mode, five times each")
    print(
        f"  {'mode':<6} {'trial':<6} {'exit':<5} {'crash':<6} {'resume':<7} "
        f"{'model calls':<12} outcome"
    )
    for durability in ("sync", "async", "exit"):
        for trial in range(1, 6):
            r = crash_then_resume(root / f"{durability}{trial}", durability)
            outcome = r.get("final") or r.get("error")
            print(
                f"  {durability:<6} {trial:<6} {r['exit_code']:<5} {r['after_crash']:<6} "
                f"{r['after_resume']:<7} {r['model_calls']:<12} {outcome}"
            )

    banner("the point")
    print(
        "LangGraph checkpoints what each node returned, never what a node\n"
        "was in the middle of doing. A tool that dies after its effect lands\n"
        "leaves no trace, so the resume runs it again and the card is\n"
        "charged twice. agentrt refuses at this point instead, because its\n"
        "write-ahead log holds a started record with no completed one."
    )


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        child(sys.argv[2], Path(sys.argv[3]), sys.argv[4])
    else:
        main()
