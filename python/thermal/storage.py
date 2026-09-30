"""Storage repository abstraction.

Local mode uses SQLite; server mode (Phase 12) will add a Postgres-backed
implementation behind the same Repository interface so callers never know
which backend they're talking to. Large telemetry traces go to Parquet
(python/thermal/telemetry.py already writes that format) and are referenced
from the run record by path, not inlined into the relational store.
"""

from __future__ import annotations

import json
import os
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

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    created_at_ns INTEGER NOT NULL,
    git_commit TEXT,
    workload_name TEXT NOT NULL,
    hypothesis TEXT,
    metric_name TEXT NOT NULL,
    higher_is_better INTEGER NOT NULL,
    repetitions INTEGER NOT NULL,
    baseline_config_json TEXT NOT NULL,
    treatment_config_json TEXT NOT NULL,
    baseline_values_json TEXT NOT NULL,
    treatment_values_json TEXT NOT NULL,
    baseline_device TEXT NOT NULL,
    treatment_device TEXT NOT NULL,
    comparison_json TEXT NOT NULL,
    verdict TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiments_workload ON experiments(workload_name);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at_ns);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL,
    started_at_ns INTEGER,
    finished_at_ns INTEGER,
    request_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at_ns);
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


@dataclass
class ExperimentRecord:
    experiment_id: str
    created_at_ns: int
    git_commit: Optional[str]
    workload_name: str
    hypothesis: str
    metric_name: str
    higher_is_better: bool
    repetitions: int
    baseline_config: dict[str, Any]
    treatment_config: dict[str, Any]
    baseline_values: list[float]
    treatment_values: list[float]
    baseline_device: str
    treatment_device: str
    comparison: dict[str, Any]
    verdict: str

    @classmethod
    def new(
        cls,
        workload_name: str,
        metric_name: str,
        higher_is_better: bool,
        repetitions: int,
        baseline_config: dict[str, Any],
        treatment_config: dict[str, Any],
        baseline_values: list[float],
        treatment_values: list[float],
        baseline_device: str,
        treatment_device: str,
        comparison: dict[str, Any],
        verdict: str,
        hypothesis: str = "",
        git_commit: Optional[str] = None,
    ) -> "ExperimentRecord":
        return cls(
            experiment_id=str(uuid.uuid4()),
            created_at_ns=time.time_ns(),
            git_commit=git_commit,
            workload_name=workload_name,
            hypothesis=hypothesis,
            metric_name=metric_name,
            higher_is_better=higher_is_better,
            repetitions=repetitions,
            baseline_config=baseline_config,
            treatment_config=treatment_config,
            baseline_values=baseline_values,
            treatment_values=treatment_values,
            baseline_device=baseline_device,
            treatment_device=treatment_device,
            comparison=comparison,
            verdict=verdict,
        )


def _row_to_experiment(row: sqlite3.Row) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=row["experiment_id"],
        created_at_ns=row["created_at_ns"],
        git_commit=row["git_commit"],
        workload_name=row["workload_name"],
        hypothesis=row["hypothesis"] or "",
        metric_name=row["metric_name"],
        higher_is_better=bool(row["higher_is_better"]),
        repetitions=row["repetitions"],
        baseline_config=json.loads(row["baseline_config_json"]),
        treatment_config=json.loads(row["treatment_config_json"]),
        baseline_values=json.loads(row["baseline_values_json"]),
        treatment_values=json.loads(row["treatment_values_json"]),
        baseline_device=row["baseline_device"],
        treatment_device=row["treatment_device"],
        comparison=json.loads(row["comparison_json"]),
        verdict=row["verdict"],
    )


class ExperimentRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, experiment: ExperimentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, created_at_ns, git_commit, workload_name, hypothesis,
                    metric_name, higher_is_better, repetitions,
                    baseline_config_json, treatment_config_json,
                    baseline_values_json, treatment_values_json,
                    baseline_device, treatment_device, comparison_json, verdict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.created_at_ns,
                    experiment.git_commit,
                    experiment.workload_name,
                    experiment.hypothesis,
                    experiment.metric_name,
                    int(experiment.higher_is_better),
                    experiment.repetitions,
                    json.dumps(experiment.baseline_config),
                    json.dumps(experiment.treatment_config),
                    json.dumps(experiment.baseline_values),
                    json.dumps(experiment.treatment_values),
                    experiment.baseline_device,
                    experiment.treatment_device,
                    json.dumps(experiment.comparison),
                    experiment.verdict,
                ),
            )

    def get(self, experiment_id: str) -> Optional[ExperimentRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            return _row_to_experiment(row) if row is not None else None

    def list(self, workload_name: Optional[str] = None, limit: int = 50) -> list[ExperimentRecord]:
        with self._connect() as conn:
            if workload_name is not None:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE workload_name = ? ORDER BY created_at_ns DESC LIMIT ?",
                    (workload_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at_ns DESC LIMIT ?", (limit,)
                ).fetchall()
            return [_row_to_experiment(r) for r in rows]


@dataclass
class JobRecord:
    job_id: str
    kind: str  # "workload_run" | "experiment_run"
    status: str  # "pending" | "running" | "completed" | "failed"
    created_at_ns: int
    request: dict[str, Any]
    started_at_ns: Optional[int] = None
    finished_at_ns: Optional[int] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    @classmethod
    def new(cls, kind: str, request: dict[str, Any]) -> "JobRecord":
        return cls(job_id=str(uuid.uuid4()), kind=kind, status="pending", created_at_ns=time.time_ns(), request=request)


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        kind=row["kind"],
        status=row["status"],
        created_at_ns=row["created_at_ns"],
        started_at_ns=row["started_at_ns"],
        finished_at_ns=row["finished_at_ns"],
        request=json.loads(row["request_json"]),
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        error=row["error"],
    )


class JobRepository:
    """Persisted job records -- durable across server restarts, unlike an
    in-memory-only queue, and inspectable the same way runs/experiments are
    (thermal database, the API) rather than being a black box."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, job: JobRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, kind, status, created_at_ns, started_at_ns, finished_at_ns, request_json, result_json, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    started_at_ns = excluded.started_at_ns,
                    finished_at_ns = excluded.finished_at_ns,
                    result_json = excluded.result_json,
                    error = excluded.error
                """,
                (
                    job.job_id,
                    job.kind,
                    job.status,
                    job.created_at_ns,
                    job.started_at_ns,
                    job.finished_at_ns,
                    json.dumps(job.request),
                    json.dumps(job.result) if job.result is not None else None,
                    job.error,
                ),
            )

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return _row_to_job(row) if row is not None else None

    def list(self, status: Optional[str] = None, limit: int = 50) -> list[JobRecord]:
        with self._connect() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at_ns DESC LIMIT ?", (status, limit)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM jobs ORDER BY created_at_ns DESC LIMIT ?", (limit,)).fetchall()
            return [_row_to_job(r) for r in rows]


def default_db_path() -> Path:
    """THERMAL_DB_PATH overrides the default location. Reading it from the
    environment (not a monkeypatched function reference) matters for
    thermal.distributed: worker processes are spawned fresh and don't
    inherit the parent's in-memory state, but they do inherit environment
    variables."""
    override = os.environ.get("THERMAL_DB_PATH")
    if override:
        return Path(override)
    return Path.home() / ".thermal" / "thermal.sqlite3"
