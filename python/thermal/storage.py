"""Storage repository abstraction.

Local mode uses SQLite; server mode (Phase 12) will add a Postgres-backed
implementation behind the same Repository interface so callers never know
which backend they're talking to. Large telemetry traces go to Parquet
(python/thermal/telemetry.py already writes that format) and are referenced
from the run record by path, not inlined into the relational store.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at_ns INTEGER NOT NULL,
    git_commit TEXT,
    workload_name TEXT NOT NULL,
    workload_version TEXT NOT NULL,
    device TEXT NOT NULL,
    warmup_iterations INTEGER NOT NULL,
    measurement_iterations INTEGER NOT NULL,
    configuration_json TEXT NOT NULL,
    hardware_fingerprint_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    telemetry_path TEXT,
    diagnosis_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_workload ON runs(workload_name);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at_ns);
"""


@dataclass
class RunRecord:
    run_id: str
    created_at_ns: int
    git_commit: Optional[str]
    workload_name: str
    workload_version: str
    device: str
    warmup_iterations: int
    measurement_iterations: int
    configuration: dict[str, Any]
    hardware_fingerprint: dict[str, Any]
    metrics: dict[str, Any]
    telemetry_path: Optional[str] = None
    diagnosis: Optional[dict[str, Any]] = None

    @classmethod
    def new(
        cls,
        workload_name: str,
        workload_version: str,
        device: str,
        warmup_iterations: int,
        measurement_iterations: int,
        configuration: dict[str, Any],
        hardware_fingerprint: dict[str, Any],
        metrics: dict[str, Any],
        git_commit: Optional[str] = None,
        telemetry_path: Optional[str] = None,
        diagnosis: Optional[dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> "RunRecord":
        return cls(
            run_id=run_id or str(uuid.uuid4()),
            created_at_ns=time.time_ns(),
            git_commit=git_commit,
            workload_name=workload_name,
            workload_version=workload_version,
            device=device,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            configuration=configuration,
            hardware_fingerprint=hardware_fingerprint,
            metrics=metrics,
            telemetry_path=telemetry_path,
            diagnosis=diagnosis,
        )


class RunRepository(ABC):
    """Storage-backend-agnostic interface. SQLite (local) and Postgres (server,
    Phase 12) both implement this so callers never branch on backend."""

    @abstractmethod
    def save(self, run: RunRecord) -> None: ...

    @abstractmethod
    def get(self, run_id: str) -> Optional[RunRecord]: ...

    @abstractmethod
    def list(self, workload_name: Optional[str] = None, limit: int = 50) -> list[RunRecord]: ...


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        created_at_ns=row["created_at_ns"],
        git_commit=row["git_commit"],
        workload_name=row["workload_name"],
        workload_version=row["workload_version"],
        device=row["device"],
        warmup_iterations=row["warmup_iterations"],
        measurement_iterations=row["measurement_iterations"],
        configuration=json.loads(row["configuration_json"]),
        hardware_fingerprint=json.loads(row["hardware_fingerprint_json"]),
        metrics=json.loads(row["metrics_json"]),
        telemetry_path=row["telemetry_path"],
        diagnosis=json.loads(row["diagnosis_json"]) if row["diagnosis_json"] else None,
    )


class SQLiteRunRepository(RunRepository):
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Minimal additive migration: add columns introduced after the initial
        schema to any pre-existing database file rather than requiring a wipe."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if "diagnosis_json" not in existing:
            conn.execute("ALTER TABLE runs ADD COLUMN diagnosis_json TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, run: RunRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, created_at_ns, git_commit, workload_name, workload_version,
                    device, warmup_iterations, measurement_iterations,
                    configuration_json, hardware_fingerprint_json, metrics_json, telemetry_path,
                    diagnosis_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.created_at_ns,
                    run.git_commit,
                    run.workload_name,
                    run.workload_version,
                    run.device,
                    run.warmup_iterations,
                    run.measurement_iterations,
                    json.dumps(run.configuration),
                    json.dumps(run.hardware_fingerprint),
                    json.dumps(run.metrics),
                    run.telemetry_path,
                    json.dumps(run.diagnosis) if run.diagnosis is not None else None,
                ),
            )

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return _row_to_record(row) if row is not None else None

    def list(self, workload_name: Optional[str] = None, limit: int = 50) -> list[RunRecord]:
        with self._connect() as conn:
            if workload_name is not None:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE workload_name = ? ORDER BY created_at_ns DESC LIMIT ?",
                    (workload_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY created_at_ns DESC LIMIT ?", (limit,)
                ).fetchall()
            return [_row_to_record(r) for r in rows]


def default_db_path() -> Path:
    return Path.home() / ".thermal" / "thermal.sqlite3"
