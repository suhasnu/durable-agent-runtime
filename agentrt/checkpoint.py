"""Durable state.

One row per step, inserted and never updated, so step 5's row keeps describing
step 5 forever. That is what makes checkpoint_at() a time machine rather than
just a crash recovery mechanism.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS checkpoints (
    run_id      TEXT NOT NULL,
    step        INTEGER NOT NULL,
    next_node   TEXT NOT NULL,
    state       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (run_id, step),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS tool_invocations (
    run_id       TEXT NOT NULL,
    call_id      TEXT NOT NULL,
    tool_name    TEXT NOT NULL,
    args_hash    TEXT NOT NULL,
    args         TEXT NOT NULL,
    status       TEXT NOT NULL,
    result       TEXT,
    is_error     INTEGER NOT NULL DEFAULT 0,
    started_at   REAL NOT NULL,
    completed_at REAL,
    PRIMARY KEY (run_id, call_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

@dataclass
class InvocationRecord:
    call_id: str
    tool_name: str
    args_hash: str
    status: str
    result: str | None
    is_error: bool


def hash_args(name: str, args: dict[str, Any]) -> str:
    payload = json.dumps({"tool": name, "args": args}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]

@dataclass
class Checkpoint:
    run_id: str
    step: int
    next_node: str
    state: BaseModel


class SqliteCheckpointer:
    def __init__(self, path: str | Path = "agentrt.db"):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL so a reader (a trace viewer) can watch a live run.
        self._conn.execute("PRAGMA journal_mode=WAL")
        # FULL, not the NORMAL default: at commit 13 the safety argument is
        # that a record reaches disk before a side effect happens, and a
        # record sitting in an OS buffer has not reached disk.
        self._conn.execute("PRAGMA synchronous=FULL")
        # Off by default in SQLite, and per connection. Without it a checkpoint
        # can be written for a run that was never created.
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # runs -----------------------------------------------------------------

    def create_run(self, run_id: str, meta: dict[str, Any] | None = None) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT INTO runs (run_id, status, created_at, updated_at, meta) VALUES (?,?,?,?,?)",
            (run_id, "running", now, now, json.dumps(meta or {})),
        )

    def set_run_status(self, run_id: str, status: str) -> None:
        cursor = self._conn.execute(
            "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
            (status, time.time(), run_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"no such run: {run_id!r}")

    def run_status(self, run_id: str) -> str | None:
        row = self._conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return row["status"] if row else None

    def list_runs(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM runs"
        params: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status=?"
            params = (status,)
        sql += " ORDER BY created_at DESC"
        return [dict(r) for r in self._conn.execute(sql, params)]

    # checkpoints ----------------------------------------------------------

    def put_checkpoint(self, cp: Checkpoint) -> None:
        """Insert, never replace.

        A second write at the same step would silently rewrite history, which
        is what checkpoint_at() exists to prevent. The primary key rejects it
        instead, so the bug surfaces where it happens.
        """
        state_step = getattr(cp.state, "step", None)
        if state_step is not None and state_step != cp.step:
            raise ValueError(
                f"checkpoint step {cp.step} disagrees with state.step {state_step}; "
                f"a resume would read two different numbers for the same moment"
            )
        self._conn.execute(
            "INSERT INTO checkpoints (run_id, step, next_node, state, created_at) "
            "VALUES (?,?,?,?,?)",
            (cp.run_id, cp.step, cp.next_node, cp.state.model_dump_json(), time.time()),
        )

    def latest_checkpoint(self, run_id: str, state_type: type[BaseModel]) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE run_id=? ORDER BY step DESC LIMIT 1", (run_id,)
        ).fetchone()
        return self._to_checkpoint(row, state_type) if row else None

    def checkpoint_at(
        self, run_id: str, step: int, state_type: type[BaseModel]
    ) -> Checkpoint | None:
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE run_id=? AND step=?", (run_id, step)
        ).fetchone()
        return self._to_checkpoint(row, state_type) if row else None

    def history(self, run_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._conn.execute(
                "SELECT run_id, step, next_node, created_at FROM checkpoints "
                "WHERE run_id=? ORDER BY step",
                (run_id,),
            )
        ]

    @staticmethod
    def _to_checkpoint(row: sqlite3.Row, state_type: type[BaseModel]) -> Checkpoint:
        return Checkpoint(
            run_id=row["run_id"],
            step=row["step"],
            next_node=row["next_node"],
            state=state_type.model_validate_json(row["state"]),
        )
    
    def begin_invocation(
        self, run_id: str, call_id: str, tool_name: str, args: dict[str, Any]
    ) -> None:
        """Record intent to call a tool. Must commit before the tool executes.

        Writing this after the call would leave no trace of one that died
        mid-flight, which is exactly the call a resume needs to know about.
        """
        self._conn.execute(
            "INSERT INTO tool_invocations "
            "(run_id, call_id, tool_name, args_hash, args, status, started_at) "
            "VALUES (?,?,?,?,?,'started',?)",
            (
                run_id,
                call_id,
                tool_name,
                hash_args(tool_name, args),
                json.dumps(args, default=str),
                time.time(),
            ),
        )

    def complete_invocation(
        self, run_id: str, call_id: str, result: str, is_error: bool = False
    ) -> None:
        self._conn.execute(
            "UPDATE tool_invocations SET status='completed', result=?, is_error=?, "
            "completed_at=? WHERE run_id=? AND call_id=?",
            (result, int(is_error), time.time(), run_id, call_id),
        )

    def get_invocation(self, run_id: str, call_id: str) -> InvocationRecord | None:
        row = self._conn.execute(
            "SELECT * FROM tool_invocations WHERE run_id=? AND call_id=?", (run_id, call_id)
        ).fetchone()
        return _to_invocation(row) if row else None

    def dangling_invocations(self, run_id: str) -> list[InvocationRecord]:
        """Started but never completed. Each one is an unknown side effect."""
        return [
            _to_invocation(r)
            for r in self._conn.execute(
                "SELECT * FROM tool_invocations WHERE run_id=? AND status='started'", (run_id,)
            )
        ]

    def resolve_invocation(
        self, run_id: str, call_id: str, result: str, is_error: bool = False
    ) -> None:
        """Operator override: assert what a dangling invocation actually did."""
        self.complete_invocation(run_id, call_id, result, is_error)
        
def _to_invocation(row: sqlite3.Row) -> InvocationRecord:
    return InvocationRecord(
        call_id=row["call_id"],
        tool_name=row["tool_name"],
        args_hash=row["args_hash"],
        status=row["status"],
        result=row["result"],
        is_error=bool(row["is_error"]),
    )


class InMemoryCheckpointer(SqliteCheckpointer):
    """Same code path as the real one. A test against a different
    implementation tests a different system."""

    def __init__(self) -> None:
        super().__init__(":memory:")