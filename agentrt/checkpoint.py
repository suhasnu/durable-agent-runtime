"""Durable state.

One row per step, inserted and never updated, so step 5's row keeps describing
step 5 forever. That is what makes checkpoint_at() a time machine rather than
just a crash recovery mechanism.
"""

from __future__ import annotations

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
"""


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


class InMemoryCheckpointer(SqliteCheckpointer):
    """Same code path as the real one. A test against a different
    implementation tests a different system."""

    def __init__(self) -> None:
        super().__init__(":memory:")