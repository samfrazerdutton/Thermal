"""THERMAL FastAPI service.

Local single-user mode: no auth, SQLite storage. Three ways to start a run,
depending on what the caller needs:

  POST /api/workloads/run, /api/experiments      blocks until done; simplest
  /api/ws/workloads/run, /api/ws/experiments/run  streams live events (Phase 14);
                                                   still tied to one open connection
  POST /api/jobs/workloads/run, /api/jobs/experiments/run   returns a job_id
                                                   immediately (202); the run
                                                   continues in a background
                                                   thread even if the caller
                                                   disconnects -- poll
                                                   GET /api/jobs/{id} for status

The job queue (thermal/jobs.py) is an in-process thread pool, not a separate
worker service or message broker -- true distributed workers are out of scope
for local mode (see services/README.md). Jobs persist to SQLite, so their
status survives a server restart and is inspectable like any other record.

Every endpoint here calls the exact same code the CLI calls
(thermal.runner, thermal.storage, thermal.diagnosis, thermal.causal) so a
run triggered from the API and one triggered from `thermal workload run`
are computed identically -- there is exactly one implementation of "run a
workload" or "run an experiment" in this codebase.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from api.schemas import ExperimentRunRequest, OptimizeGridSearchRequest, WorkloadRunRequest
from thermal.ai import AIExplainer, explain_record
from thermal.causal import build_graph_from_experiments
from thermal.hardware import collect_hardware_report
from thermal.jobs import JobQueue
from thermal.optimization import grid_search
from thermal.report import generate_report
from thermal.runner import run_and_store_experiment, run_and_store_workload
from thermal.storage import ExperimentRepository, SQLiteRunRepository, default_db_path
from thermal.workload import WorkloadRegistry

import workloads  # noqa: F401  (registers built-in workloads)

app = FastAPI(
    title="THERMAL API",
    description=(
        "Local-mode API for THERMAL. Every number this API returns came from "
        "a real measurement on this machine -- see /api/health for what's "
        "actually available."
    ),
    version="0.1.0",
)

job_queue = JobQueue()


def _diagnosis_to_dict(diagnosis) -> dict:
    return {
        "bottleneck": diagnosis.bottleneck.value,
        "confidence": diagnosis.confidence,
        "rationale": diagnosis.rationale,
        "evidence": [asdict(e) for e in diagnosis.evidence],
        "missing_features": diagnosis.missing_features,
    }


def _run_record_to_dict(record) -> dict:
    return {
        "run_id": record.run_id,
        "created_at_ns": record.created_at_ns,
        "git_commit": record.git_commit,
        "workload_name": record.workload_name,
        "workload_version": record.workload_version,
        "device": record.device,
        "warmup_iterations": record.warmup_iterations,
        "measurement_iterations": record.measurement_iterations,
        "configuration": record.configuration,
        "hardware_fingerprint": record.hardware_fingerprint,
        "metrics": record.metrics,
        "telemetry_path": record.telemetry_path,
        "diagnosis": record.diagnosis,
    }


def _experiment_record_to_dict(record) -> dict:
    return {
        "experiment_id": record.experiment_id,
        "created_at_ns": record.created_at_ns,
        "git_commit": record.git_commit,
        "workload_name": record.workload_name,
        "hypothesis": record.hypothesis,
        "metric_name": record.metric_name,
        "higher_is_better": record.higher_is_better,
        "repetitions": record.repetitions,
        "baseline_config": record.baseline_config,
        "treatment_config": record.treatment_config,
        "baseline_values": record.baseline_values,
        "treatment_values": record.treatment_values,
        "baseline_device": record.baseline_device,
        "treatment_device": record.treatment_device,
        "comparison": record.comparison,
        "verdict": record.verdict,
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/hardware")
def hardware() -> dict:
    return collect_hardware_report().to_dict()


@app.get("/api/workloads")
def list_workloads() -> list[dict]:
    return [
        {
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "default_parameters": spec.default_parameters,
            "warmup_iterations": spec.warmup_iterations,
            "measurement_iterations": spec.measurement_iterations,
            "output_metrics": list(spec.output_metrics),
        }
        for spec in WorkloadRegistry.list()
    ]


@app.post("/api/workloads/run")
def run_workload_endpoint(request: WorkloadRunRequest) -> dict:
    try:
        outcome = run_and_store_workload(
            request.workload_name, request.params, samples=request.samples, warmup=request.warmup
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "run": _run_record_to_dict(outcome.record),
        "diagnosis": _diagnosis_to_dict(outcome.diagnosis),
    }


async def _stream_events(websocket: WebSocket, blocking_call, already_accepted: bool = False) -> None:
    """Runs `blocking_call(on_event)` on a worker thread and forwards every
    event it emits to the websocket as JSON, in order, as they happen --
    `blocking_call` is thermal.runner code, which is synchronous and knows
    nothing about asyncio or websockets."""
    if not already_accepted:
        await websocket.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(name: str, payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"event": name, **payload})

    def run() -> None:
        try:
            final = blocking_call(on_event)
            loop.call_soon_threadsafe(queue.put_nowait, {"event": "result", **final})
        except (KeyError, ValueError) as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"event": "error", "detail": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    worker = asyncio.create_task(asyncio.to_thread(run))
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            await websocket.send_json(item)
    except WebSocketDisconnect:
        pass
    finally:
        await worker
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed (client disconnected)


@app.websocket("/api/ws/workloads/run")
async def ws_run_workload(websocket: WebSocket) -> None:
    """Client sends one WorkloadRunRequest as JSON text as the first message
    after the connection is accepted; server streams events, then a final
    {"event": "result", "run": {...}, "diagnosis": {...}} before closing."""
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        request = WorkloadRunRequest.model_validate_json(raw)
    except Exception as exc:
        await websocket.send_json({"event": "error", "detail": f"invalid request: {exc}"})
        await websocket.close()
        return

    def blocking_call(on_event):
        outcome = run_and_store_workload(
            request.workload_name,
            request.params,
            samples=request.samples,
            warmup=request.warmup,
            on_event=on_event,
        )
        return {"run": _run_record_to_dict(outcome.record), "diagnosis": _diagnosis_to_dict(outcome.diagnosis)}

    await _stream_events(websocket, blocking_call, already_accepted=True)


@app.websocket("/api/ws/experiments/run")
async def ws_run_experiment(websocket: WebSocket) -> None:
    """Same protocol as /api/ws/workloads/run, for ExperimentRunRequest."""
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        request = ExperimentRunRequest.model_validate_json(raw)
    except Exception as exc:
        await websocket.send_json({"event": "error", "detail": f"invalid request: {exc}"})
        await websocket.close()
        return

    def blocking_call(on_event):
        outcome = run_and_store_experiment(
            workload_name=request.workload_name,
            baseline_params=request.baseline_params,
            treatment_params=request.treatment_params,
            metric_name=request.metric_name,
            higher_is_better=request.higher_is_better,
            repetitions=request.repetitions,
            warmup_iterations=request.warmup_iterations,
            hypothesis=request.hypothesis,
            on_event=on_event,
        )
        return {"experiment": _experiment_record_to_dict(outcome.record)}

    await _stream_events(websocket, blocking_call, already_accepted=True)


def _job_to_dict(job) -> dict:
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "created_at_ns": job.created_at_ns,
        "started_at_ns": job.started_at_ns,
        "finished_at_ns": job.finished_at_ns,
        "request": job.request,
        "result": job.result,
        "error": job.error,
    }


@app.post("/api/jobs/workloads/run", status_code=202)
def submit_workload_job(request: WorkloadRunRequest) -> dict:
    """Returns immediately with a job_id; the run continues in a background
    thread even if the caller disconnects. Poll GET /api/jobs/{job_id}."""
    job_id = job_queue.submit_workload_run(
        request.workload_name, request.params, samples=request.samples, warmup=request.warmup
    )
    return {"job_id": job_id, "status": "pending"}


@app.post("/api/jobs/experiments/run", status_code=202)
def submit_experiment_job(request: ExperimentRunRequest) -> dict:
    job_id = job_queue.submit_experiment_run(
        workload_name=request.workload_name,
        baseline_params=request.baseline_params,
        treatment_params=request.treatment_params,
        metric_name=request.metric_name,
        higher_is_better=request.higher_is_better,
        repetitions=request.repetitions,
        warmup_iterations=request.warmup_iterations,
        hypothesis=request.hypothesis,
    )
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs")
def list_jobs(status: Optional[str] = None, limit: int = 50) -> list[dict]:
    return [_job_to_dict(j) for j in job_queue.list_jobs(status=status, limit=limit)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = job_queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job found with id {job_id}")
    return _job_to_dict(job)


@app.get("/api/runs")
def list_runs(workload: Optional[str] = None, limit: int = 50) -> list[dict]:
    repo = SQLiteRunRepository(default_db_path())
    return [_run_record_to_dict(r) for r in repo.list(workload_name=workload, limit=limit)]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    repo = SQLiteRunRepository(default_db_path())
    record = repo.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no run found with id {run_id}")
    return _run_record_to_dict(record)


@app.get("/api/diagnoses/{run_id}")
def get_diagnosis(run_id: str) -> dict:
    repo = SQLiteRunRepository(default_db_path())
    record = repo.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no run found with id {run_id}")
    if not record.diagnosis:
        raise HTTPException(status_code=404, detail=f"run {run_id} has no stored diagnosis")
    return record.diagnosis


@app.post("/api/experiments")
def run_experiment_endpoint(request: ExperimentRunRequest) -> dict:
    """Creates AND runs the experiment in one call (local mode has no
    background queue yet -- see docs/roadmap.md Phase 14)."""
    try:
        outcome = run_and_store_experiment(
            workload_name=request.workload_name,
            baseline_params=request.baseline_params,
            treatment_params=request.treatment_params,
            metric_name=request.metric_name,
            higher_is_better=request.higher_is_better,
            repetitions=request.repetitions,
            warmup_iterations=request.warmup_iterations,
            hypothesis=request.hypothesis,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _experiment_record_to_dict(outcome.record)


@app.get("/api/experiments")
def list_experiments(workload: Optional[str] = None, limit: int = 50) -> list[dict]:
    repo = ExperimentRepository(default_db_path())
    return [_experiment_record_to_dict(e) for e in repo.list(workload_name=workload, limit=limit)]


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict:
    repo = ExperimentRepository(default_db_path())
    record = repo.get(experiment_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no experiment found with id {experiment_id}")
    return _experiment_record_to_dict(record)


@app.get("/api/causal-graph")
def get_causal_graph(workload: Optional[str] = None) -> dict:
    """Deviates from a per-run path (as in the original design doc) because
    causal edges in this engine are workload-scoped (parameter -> metric
    relationships derived across that workload's experiments), not
    properties of a single run. Filter with ?workload=<name>."""
    repo = ExperimentRepository(default_db_path())
    experiments = repo.list(workload_name=workload, limit=1000)
    graph = build_graph_from_experiments(experiments)
    return graph.to_dict()


@app.post("/api/optimize/grid-search")
def optimize_grid_search(request: OptimizeGridSearchRequest) -> dict:
    try:
        result = grid_search(
            workload_name=request.workload_name,
            baseline_params=request.baseline_params,
            param_name=request.param_name,
            candidate_values=request.candidate_values,
            metric_name=request.metric_name,
            higher_is_better=request.higher_is_better,
            repetitions=request.repetitions,
            warmup_iterations=request.warmup_iterations,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "baseline_params": result.baseline_params,
        "baseline_metric_value": result.baseline_metric_value,
        "candidates": [asdict(c) for c in result.candidates],
        "best": asdict(result.best) if result.best is not None else None,
    }


@app.get("/api/benchmarks")
def list_benchmarks(limit: int = 50) -> list[dict]:
    """Alias over stored runs -- see docs/roadmap.md Phase 33 (benchmark explorer)."""
    repo = SQLiteRunRepository(default_db_path())
    return [_run_record_to_dict(r) for r in repo.list(limit=limit)]


@app.get("/api/reports/{report_id}", response_class=PlainTextResponse)
def get_report(report_id: str) -> str:
    try:
        return generate_report(report_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/explain/{record_id}")
def get_explanation(record_id: str) -> dict:
    """Optional AI layer (Phase 16). 503 -- not 500 -- when ANTHROPIC_API_KEY
    isn't set: this is an expected, documented mode, not a server error."""
    explainer = AIExplainer.from_env()
    if not explainer.available:
        raise HTTPException(
            status_code=503,
            detail="AI explanation layer is not configured (ANTHROPIC_API_KEY not set). "
            "Diagnosis, experiments, and reports all work without it.",
        )
    try:
        return {"explanation": explain_record(explainer, record_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
