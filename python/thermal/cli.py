"""THERMAL command-line interface."""

from __future__ import annotations

import json as jsonlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer

from analysis.baseline import InsufficientSamplesError, compute_baseline
from thermal.causal import build_graph_from_experiments
from thermal.counterfactual import experiment_command_for, generate_hypotheses
from thermal.diagnosis import BottleneckClass
from thermal.distributed import run_workload_multiprocess
from thermal.doctor import CheckStatus, core_ready, run_doctor
from thermal.hardware import collect_hardware_report
from thermal.ai import AIExplainer, AIUnavailableError, explain_record
from thermal.optimization import grid_search
from thermal.report import generate_report
from thermal.native_bench import (
    KERNEL_NAMES,
    KernelBenchFailedError,
    KernelBenchNotBuiltError,
    achieved_bandwidth_gbps,
    achieved_gflops,
    binary_path,
    is_built,
    run_kernel_bench,
)
from thermal.runner import run_and_store_experiment, run_and_store_workload
from thermal.storage import (
    ExperimentRepository,
    RunRecord,
    SQLiteRunRepository,
    default_db_path,
)
from thermal.telemetry import TelemetryCollector
from thermal.workload import WorkloadRegistry

import workloads  # noqa: F401  (import registers all built-in workloads)


def _parse_scalar(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_params(param: list[str]) -> dict:
    params: dict = {}
    for item in param:
        if "=" not in item:
            typer.echo(f"invalid --param '{item}', expected key=value")
            raise typer.Exit(code=1)
        key, value = item.split("=", 1)
        params[key] = _parse_scalar(value)
    return params


_LOWER_IS_BETTER_HINTS = ("duration", "latency", "time_ms", "time_seconds", "elapsed")
_HIGHER_IS_BETTER_HINTS = ("gflops", "throughput", "bandwidth", "tokens_per_sec", "gbps")


def _infer_primary_metric(metrics: dict) -> tuple[Optional[str], bool]:
    """Pick the metric most useful to optimize and its direction, from naming
    convention -- e.g. duration_seconds is lower-is-better, gflops is higher-
    is-better. Falls back to the first metric, assumed higher-is-better, when
    the name gives no hint (explicit is better than guessing, but a workload
    author who follows the convention gets this right automatically)."""
    for name in metrics:
        lowered = name.lower()
        if any(hint in lowered for hint in _HIGHER_IS_BETTER_HINTS):
            return name, True
    for name in metrics:
        lowered = name.lower()
        if any(hint in lowered for hint in _LOWER_IS_BETTER_HINTS):
            return name, False
    return (next(iter(metrics), None), True)


def _git_commit() -> Optional[str]:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run([git, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None

app = typer.Typer(
    name="thermal",
    help="THERMAL: closed-loop performance engineering for AI and accelerated workloads.",
    no_args_is_help=True,
)

workload_app = typer.Typer(help="Run and inspect THERMAL workloads.", no_args_is_help=True)
app.add_typer(workload_app, name="workload")

kernel_app = typer.Typer(help="Run the native CUDA kernel lab (native/cuda).", no_args_is_help=True)
app.add_typer(kernel_app, name="kernel")


@kernel_app.command("list")
def kernel_list() -> None:
    """List the native CUDA kernels and whether the bench binary is built."""
    typer.echo("THERMAL CUDA KERNEL LAB\n")
    if is_built():
        typer.echo(f"Binary: {binary_path()}\n")
    else:
        typer.echo(f"Binary not built yet: {binary_path()}")
        typer.echo("Build it with: powershell -File scripts/build-native.ps1\n")
    for name in KERNEL_NAMES:
        typer.echo(f"  {name}")


@kernel_app.command("run")
def kernel_run(
    name: str = typer.Argument(..., help="kernel name, e.g. matmul_tiled"),
    n: int = typer.Option(1 << 20, "--n", help="problem size (element count, or N for NxN matmul)"),
    iters: int = typer.Option(20, "--iters"),
    warmup: int = typer.Option(5, "--warmup"),
    block_size: int = typer.Option(256, "--block-size"),
) -> None:
    """Run one native CUDA kernel: correctness check, then a timed baseline."""
    try:
        result = run_kernel_bench(name, n=n, iters=iters, warmup=warmup, block_size=block_size)
    except KernelBenchNotBuiltError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2)
    except (KernelBenchFailedError, ValueError) as exc:
        typer.echo(f"Kernel bench failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"Kernel: {result.kernel}")
    typer.echo(f"Device: {result.device}")
    typer.echo(f"Correct: {result.correct}\n")

    try:
        baseline = compute_baseline(result.iteration_times_ms, metric_name="kernel_time_ms")
        typer.echo(baseline.summary_text())
    except InsufficientSamplesError as exc:
        typer.echo(str(exc))

    mean_ms = sum(result.iteration_times_ms) / len(result.iteration_times_ms)
    gflops = achieved_gflops(result, mean_ms)
    bandwidth = achieved_bandwidth_gbps(result, mean_ms)
    if gflops is not None:
        typer.echo(f"\nachieved GFLOP/s (at mean time): {gflops:.1f}")
    if bandwidth is not None:
        typer.echo(f"achieved bandwidth GB/s (at mean time): {bandwidth:.1f}")


@workload_app.command("list")
def workload_list() -> None:
    """List registered workloads."""
    specs = WorkloadRegistry.list()
    typer.echo("THERMAL WORKLOADS\n")
    for spec in specs:
        typer.echo(f"{spec.name} (v{spec.version})")
        typer.echo(f"  {spec.description}")
        typer.echo(f"  default params: {spec.default_parameters}")
        typer.echo(f"  warmup={spec.warmup_iterations} measured={spec.measurement_iterations}")
        typer.echo()


@workload_app.command("run")
def workload_run(
    name: str = typer.Argument(..., help="workload name, e.g. matmul"),
    param: list[str] = typer.Option([], "--param", "-p", help="key=value, repeatable"),
    samples: Optional[int] = typer.Option(None, "--samples", help="override measured iteration count"),
    warmup: Optional[int] = typer.Option(None, "--warmup", help="override warmup iteration count"),
    output: Optional[Path] = typer.Option(None, "--output", help="write per-iteration metrics as JSON"),
) -> None:
    """Run one workload: warmup iterations, then measured iterations."""
    params = _parse_params(param)

    try:
        outcome = run_and_store_workload(name, params, samples=samples, warmup=warmup)
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    record = outcome.record
    typer.echo(
        f"Running '{name}' - warmup={record.warmup_iterations} measured={record.measurement_iterations}"
    )
    typer.echo(f"\nBASELINE\nDevice: {record.device}\n")
    for baseline in outcome.baselines.values():
        typer.echo(baseline.summary_text())
        typer.echo()

    if output is not None:
        output.write_text(jsonlib.dumps(outcome.per_iteration_metrics, indent=2))
        typer.echo(f"Wrote {len(outcome.per_iteration_metrics)} iteration records -> {output}")

    diagnosis = outcome.diagnosis
    typer.echo("DIAGNOSIS")
    typer.echo(f"  {diagnosis.bottleneck.value}  (confidence: {diagnosis.confidence:.0%})")
    typer.echo(f"  {diagnosis.rationale}")
    for ev in diagnosis.evidence:
        typer.echo(f"    evidence: {ev.feature} = {ev.value:.2f} ({ev.comparison} {ev.threshold})")
    if diagnosis.missing_features:
        typer.echo(f"  not measurable on this run: {', '.join(diagnosis.missing_features)}")
    typer.echo()

    typer.echo(f"Saved run {record.run_id} -> {default_db_path()}")


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO "


@app.command()
def hardware(json: bool = typer.Option(False, "--json", help="emit machine-readable JSON")) -> None:
    """Detect and print real hardware/software capabilities."""
    report = collect_hardware_report()

    if json:
        typer.echo(jsonlib.dumps(report.to_dict(), indent=2, default=str))
        return

    typer.echo("THERMAL HARDWARE\n")
    typer.echo("CPU")
    typer.echo(f"  {report.cpu.name}")
    typer.echo(f"  cores:   {report.cpu.physical_cores}")
    typer.echo(f"  threads: {report.cpu.logical_cores}")
    typer.echo()

    typer.echo("GPU")
    if report.gpu.available:
        typer.echo(f"  {report.gpu.name}")
        typer.echo(f"  VRAM: {report.gpu.memory_total_mb:.0f} MB")
        typer.echo(f"  CUDA driver: {report.gpu.cuda_driver_version}")
        typer.echo(f"  Compute capability: {report.gpu.compute_capability}")
        typer.echo(f"\nDriver:\n  {report.gpu.driver_version}")
    else:
        typer.echo(f"  unavailable ({report.gpu.reason})")
    typer.echo()

    typer.echo("Available telemetry:")
    if report.gpu.available:
        for key, cap in report.gpu.telemetry.items():
            typer.echo(f"  {key:<28} {_yes_no(cap.available)}")
    else:
        typer.echo("  none (no GPU access)")
    typer.echo()

    typer.echo("Toolchain:")
    tc = report.toolchain
    for cap in (tc.cuda_toolkit, tc.msvc, tc.gcc, tc.cmake, tc.docker, tc.git, tc.nsight_systems, tc.nsight_compute):
        status = _yes_no(cap.available)
        detail = cap.reason if not cap.available else ""
        typer.echo(f"  {cap.name:<16} {status}  {detail}")


@app.command()
def doctor() -> None:
    """Run the THERMAL system check."""
    typer.echo("THERMAL SYSTEM CHECK\n")
    checks = run_doctor()
    for check in checks:
        mark = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}[check.status.value]
        req = "" if check.required else " (optional)"
        typer.echo(f"{mark} {check.name}{req}: {check.detail}")

    typer.echo()
    if core_ready(checks):
        typer.echo("THERMAL CORE: READY")
    else:
        typer.echo("THERMAL CORE: NOT READY")
        failed = [c.name for c in checks if c.required and c.status == CheckStatus.FAIL]
        typer.echo(f"Missing required capabilities: {', '.join(failed)}")
        raise typer.Exit(code=1)


@app.command()
def profile(
    duration: float = typer.Option(5.0, "--duration", help="seconds to sample for"),
    interval: float = typer.Option(0.5, "--interval", help="seconds between samples"),
    pid: Optional[int] = typer.Option(None, "--pid", help="also track this process (default: none)"),
    output: Path = typer.Option(Path("thermal_profile.jsonl"), "--output", help="output file (.jsonl or .parquet)"),
) -> None:
    """Sample real CPU/GPU/process telemetry for a fixed duration and write it out.

    This is system-wide hardware sampling, independent of any workload plugin
    (the workload runner is Phase 3 — see docs/roadmap.md). Useful today to
    verify telemetry collection against real hardware.
    """
    collector = TelemetryCollector(pid=pid, interval_seconds=interval)
    typer.echo(f"Sampling every {interval}s for {duration}s ...")
    collector.start()
    try:
        time.sleep(duration)
    finally:
        collector.stop()

    samples = collector.samples
    if output.suffix == ".parquet":
        collector.write_parquet(output)
    else:
        collector.write_jsonl(output)

    typer.echo(f"Collected {len(samples)} samples -> {output}")
    if samples:
        last = samples[-1]
        typer.echo(f"Last sample: CPU {last.cpu.utilization_percent:.1f}%", nl=False)
        if last.gpu is not None:
            typer.echo(
                f"  GPU {last.gpu.utilization_percent:.0f}%  "
                f"mem {last.gpu.memory_used_mb:.0f}MB  temp {last.gpu.temperature_c:.0f}C"
            )
        else:
            typer.echo("  GPU: unavailable")


experiment_app = typer.Typer(help="Run controlled baseline-vs-treatment experiments.", no_args_is_help=True)
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("run")
def experiment_run(
    workload: str = typer.Argument(..., help="workload name, e.g. matmul"),
    baseline_param: list[str] = typer.Option([], "--baseline-param", help="key=value, repeatable"),
    treatment_param: list[str] = typer.Option([], "--treatment-param", help="key=value, repeatable"),
    metric: str = typer.Option(..., "--metric", help="metric name to compare, e.g. gflops"),
    repetitions: int = typer.Option(15, "--repetitions"),
    warmup: int = typer.Option(3, "--warmup"),
    lower_is_better: bool = typer.Option(False, "--lower-is-better", help="set for latency-like metrics"),
    hypothesis: str = typer.Option("", "--hypothesis", help="one-sentence hypothesis being tested"),
) -> None:
    """Run baseline vs. treatment in strict alternation (ABAB), then verify statistically."""
    if hypothesis:
        typer.echo(f"HYPOTHESIS\n{hypothesis}\n")

    try:
        outcome = run_and_store_experiment(
            workload_name=workload,
            baseline_params=_parse_params(baseline_param),
            treatment_params=_parse_params(treatment_param),
            metric_name=metric,
            higher_is_better=not lower_is_better,
            repetitions=repetitions,
            warmup_iterations=warmup,
            hypothesis=hypothesis,
        )
    except (KeyError, ValueError) as exc:
        typer.echo(f"Experiment failed: {exc}")
        raise typer.Exit(code=1)

    result = outcome.result
    typer.echo(f"Baseline device: {result.baseline_device}  Treatment device: {result.treatment_device}\n")
    typer.echo(result.comparison.summary_text())
    typer.echo(f"\nSaved experiment {outcome.record.experiment_id} -> {default_db_path()}")


@experiment_app.command("list")
def experiment_list(
    workload: Optional[str] = typer.Option(None, "--workload"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List stored experiments, most recent first."""
    repo = ExperimentRepository(default_db_path())
    experiments = repo.list(workload_name=workload, limit=limit)
    if not experiments:
        typer.echo(f"No experiments stored yet in {default_db_path()}")
        return
    for exp in experiments:
        typer.echo(
            f"{exp.experiment_id}  {exp.workload_name:<20} metric={exp.metric_name:<15} "
            f"verdict={exp.verdict:<12} n={exp.repetitions}"
        )


@experiment_app.command("compare")
def experiment_compare(experiment_id: str) -> None:
    """Show full detail for one stored experiment."""
    repo = ExperimentRepository(default_db_path())
    exp = repo.get(experiment_id)
    if exp is None:
        typer.echo(f"No experiment found with id {experiment_id}")
        raise typer.Exit(code=1)

    if exp.hypothesis:
        typer.echo(f"HYPOTHESIS\n{exp.hypothesis}\n")
    typer.echo(f"Workload: {exp.workload_name}")
    typer.echo(f"Baseline config: {exp.baseline_config}")
    typer.echo(f"Treatment config: {exp.treatment_config}")
    typer.echo(f"Baseline device: {exp.baseline_device}  Treatment device: {exp.treatment_device}\n")
    comparison = exp.comparison
    typer.echo(f"CONTROL\n{comparison['baseline_mean']:.4g}\n")
    typer.echo(f"TREATMENT\n{comparison['treatment_mean']:.4g}\n")
    typer.echo(f"CHANGE\n{comparison['percent_change']:+.1f}%\n")
    if comparison.get("percent_change_ci_low") is not None:
        typer.echo(
            f"{comparison['confidence_level'] * 100:.0f}% CI\n"
            f"[{comparison['percent_change_ci_low']:+.1f}%, {comparison['percent_change_ci_high']:+.1f}%]\n"
        )
    typer.echo(f"n = {exp.repetitions}\n")
    typer.echo(f"VERDICT:\n{exp.verdict}")


@experiment_app.command("causal-graph")
def experiment_causal_graph(
    workload: Optional[str] = typer.Option(None, "--workload", help="restrict to one workload's experiments"),
    path_from: Optional[str] = typer.Option(None, "--path-from", help="show the causal path from this parameter"),
    path_to: Optional[str] = typer.Option(None, "--path-to", help="...to this metric (requires --path-from)"),
) -> None:
    """Build and print the causal graph derived from stored experiments.

    Every edge came from an experiment whose confidence interval excluded
    zero -- an INCONCLUSIVE experiment asserts no edge at all.
    """
    repo = ExperimentRepository(default_db_path())
    experiments = repo.list(workload_name=workload, limit=1000)
    graph = build_graph_from_experiments(experiments)

    if not graph.edges():
        typer.echo("No causal edges yet -- run experiments with a clear (non-inconclusive) result first.")
        return

    if path_from is not None:
        if path_to is None:
            typer.echo("--path-to is required with --path-from")
            raise typer.Exit(code=1)
        path = graph.path(path_from, path_to)
        if path is None:
            typer.echo(f"No causal path found from '{path_from}' to '{path_to}'.")
            raise typer.Exit(code=1)
        typer.echo(f"{path_from}")
        for edge in path:
            arrow = "-> (+)" if edge.relationship.value == "positive" else "-> (-)"
            typer.echo(f"    {arrow} {edge.target}   [{edge.evidence_kind.value}, {edge.percent_change:+.1f}%]")
        return

    typer.echo("CAUSAL GRAPH (from experimental evidence only)\n")
    for edge in graph.edges():
        typer.echo(
            f"  {edge.source} --[{edge.relationship.value}, {edge.evidence_kind.value}]--> {edge.target}"
            f"  ({edge.percent_change:+.1f}%, experiment {edge.experiment_id})"
        )


@app.command()
def optimize(
    workload: str = typer.Argument(..., help="workload name, e.g. matmul"),
    param: str = typer.Option(..., "--param", help="parameter name to search"),
    values: str = typer.Option(..., "--values", help="comma-separated candidate values, e.g. 128,512,1024"),
    baseline_param: list[str] = typer.Option([], "--baseline-param", help="key=value, repeatable"),
    metric: str = typer.Option(..., "--metric", help="metric to optimize, e.g. gflops"),
    repetitions: int = typer.Option(10, "--repetitions"),
    warmup: int = typer.Option(2, "--warmup"),
    lower_is_better: bool = typer.Option(False, "--lower-is-better"),
) -> None:
    """Grid search one parameter: each candidate is a real controlled experiment
    against the shared baseline (never a single noisy run compared to another)."""
    baseline_params = _parse_params(list(baseline_param))
    candidate_values = [_parse_scalar(v) for v in values.split(",")]

    typer.echo(f"Searching {param} in {values} for workload '{workload}', metric '{metric}'...\n")
    result = grid_search(
        workload_name=workload,
        baseline_params=baseline_params,
        param_name=param,
        candidate_values=candidate_values,
        metric_name=metric,
        higher_is_better=not lower_is_better,
        repetitions=repetitions,
        warmup_iterations=warmup,
    )
    typer.echo(result.summary_text())


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="auto-reload on source changes (development only)"),
) -> None:
    """Start the THERMAL FastAPI service (local mode: no auth, SQLite storage)."""
    import uvicorn

    typer.echo(f"THERMAL API: http://{host}:{port}/docs")
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


database_app = typer.Typer(help="Inspect stored runs (SQLite local mode).", no_args_is_help=True)
app.add_typer(database_app, name="database")


@database_app.command("list")
def database_list(
    workload: Optional[str] = typer.Option(None, "--workload", help="filter by workload name"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List stored runs, most recent first."""
    repo = SQLiteRunRepository(default_db_path())
    runs = repo.list(workload_name=workload, limit=limit)
    if not runs:
        typer.echo(f"No runs stored yet in {default_db_path()}")
        return
    for run in runs:
        typer.echo(f"{run.run_id}  {run.workload_name:<20} device={run.device:<8} n={run.measurement_iterations}  commit={run.git_commit or '-'}")


@database_app.command("show")
def database_show(run_id: str) -> None:
    """Show full detail for one stored run."""
    repo = SQLiteRunRepository(default_db_path())
    run = repo.get(run_id)
    if run is None:
        typer.echo(f"No run found with id {run_id}")
        raise typer.Exit(code=1)
    typer.echo(f"Run: thermal://run/{run.run_id}")
    typer.echo(f"Commit: {run.git_commit or 'unknown'}")
    typer.echo(f"Workload: {run.workload_name} v{run.workload_version}")
    typer.echo(f"Device: {run.device}")
    typer.echo(f"Configuration: {run.configuration}")
    typer.echo(f"Hardware: {run.hardware_fingerprint}")
    typer.echo()
    for metric_name, stats in run.metrics.items():
        typer.echo(f"{metric_name}: mean={stats['mean']:.4g} median={stats['median']:.4g} p95={stats['p95']:.4g} n={stats['n']}")
    if run.diagnosis:
        typer.echo(f"\nDiagnosis: {run.diagnosis['bottleneck']} (confidence {run.diagnosis['confidence']:.0%})")
        typer.echo(f"  {run.diagnosis['rationale']}")


def _resolve_run(run_id: str) -> "RunRecord":
    repo = SQLiteRunRepository(default_db_path())
    if run_id == "latest":
        runs = repo.list(limit=1)
        if not runs:
            typer.echo("No runs stored yet. Run `thermal workload run <name>` first.")
            raise typer.Exit(code=1)
        return runs[0]
    run = repo.get(run_id)
    if run is None:
        typer.echo(f"No run found with id {run_id}")
        raise typer.Exit(code=1)
    return run


@app.command()
def diagnose(run_id: str = typer.Argument("latest", help="run id, or 'latest'")) -> None:
    """Show the stored diagnosis for a run (computed when the run was executed)."""
    run = _resolve_run(run_id)
    if not run.diagnosis:
        typer.echo(f"Run {run.run_id} has no stored diagnosis.")
        raise typer.Exit(code=1)

    typer.echo("ROOT CAUSE\n")
    typer.echo(f"{run.diagnosis['bottleneck']}")
    typer.echo(f"\nConfidence: {run.diagnosis['confidence']:.0%}\n")
    typer.echo("Evidence:")
    for ev in run.diagnosis["evidence"]:
        typer.echo(f"  {ev['feature']:<40} {ev['value']:.2f} ({ev['comparison']} {ev['threshold']})")
    if run.diagnosis["missing_features"]:
        typer.echo(f"\nNot measurable on this run: {', '.join(run.diagnosis['missing_features'])}")
    typer.echo(f"\nWHY?\n{run.diagnosis['rationale']}")
    if run.telemetry_path:
        typer.echo(f"\nRaw telemetry: {run.telemetry_path}")

    try:
        workload_cls = WorkloadRegistry.get(run.workload_name)
        known_params = set(workload_cls.spec.default_parameters.keys())
    except KeyError:
        known_params = set()

    metric_name, higher_is_better = _infer_primary_metric(run.metrics)
    if metric_name is not None and known_params:
        bottleneck = BottleneckClass(run.diagnosis["bottleneck"])
        hypotheses = generate_hypotheses(bottleneck, run.configuration, known_params)
        actionable = [h for h in hypotheses if h.actionable]
        if actionable:
            typer.echo("\nTESTABLE HYPOTHESIS\n")
            for h in actionable:
                typer.echo(h.description)
                cmd = experiment_command_for(
                    h, run.workload_name, run.configuration, metric_name, higher_is_better=higher_is_better
                )
                typer.echo(f"\n[RUN EXPERIMENT]\n{cmd}\n")
        else:
            not_actionable = [h.reason_not_actionable for h in hypotheses if h.reason_not_actionable]
            if not_actionable:
                typer.echo(f"\nNo actionable hypothesis for this workload: {'; '.join(not_actionable)}")


@app.command()
def distribute(
    workload: str = typer.Argument(..., help="workload name, e.g. vector_ops"),
    param: list[str] = typer.Option([], "--param", "-p", help="key=value, repeatable"),
    workers: int = typer.Option(2, "--workers", help="number of parallel worker processes"),
    samples: int = typer.Option(10, "--samples"),
    warmup: int = typer.Option(2, "--warmup"),
) -> None:
    """Run a workload across multiple worker processes in parallel.

    Multi-GPU and multi-node dispatch are not implemented -- this machine
    has one GPU, so there is nothing to verify them against (see
    docs/roadmap.md Phase 18). This dispatches real OS processes and reports
    which device each one actually landed on; run `thermal database list
    --workload <name>` afterward to see each worker's persisted run.
    """
    params = _parse_params(param)
    typer.echo(f"Dispatching '{workload}' across {workers} worker processes...")
    result = run_workload_multiprocess(workload, params, num_workers=workers, samples=samples, warmup=warmup)

    typer.echo(f"\nWall clock: {result.wall_clock_seconds:.2f}s for {workers} workers")
    typer.echo(f"Distinct devices: {', '.join(result.distinct_devices)}")
    if not result.actually_multi_gpu:
        typer.echo("(all workers landed on the same device -- this is multi-process, not multi-GPU)")
    typer.echo()
    for w in result.workers:
        typer.echo(f"  pid {w.worker_pid}  run {w.run_id}  device {w.device}")


@app.command()
def report(
    record_id: str = typer.Argument("latest", help="run id, experiment id, or 'latest' (most recent run)"),
    output: Optional[Path] = typer.Option(None, "--output", help="write to a file instead of stdout"),
) -> None:
    """Generate a Markdown report for a run or experiment."""
    if record_id == "latest":
        run = _resolve_run("latest")
        record_id = run.run_id

    try:
        markdown = generate_report(record_id)
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    if output is not None:
        output.write_text(markdown, encoding="utf-8")
        typer.echo(f"Wrote report -> {output}")
    else:
        typer.echo(markdown)


@app.command()
def explain(record_id: str = typer.Argument("latest", help="run id, experiment id, or 'latest' (most recent run)")) -> None:
    """Optional AI explanation of a run's diagnosis or an experiment's result.

    Never invents a number: the model only sees the evidence already stored
    for this record and is instructed to say so if something isn't in it.
    Requires ANTHROPIC_API_KEY -- everything else in THERMAL works without
    this command.
    """
    if record_id == "latest":
        run = _resolve_run("latest")
        record_id = run.run_id

    explainer = AIExplainer.from_env()
    if not explainer.available:
        typer.echo(
            "AI explanation layer is not configured -- set ANTHROPIC_API_KEY to enable it.\n"
            "Everything else (diagnosis, experiments, reports) works without it."
        )
        raise typer.Exit(code=2)

    try:
        typer.echo(explain_record(explainer, record_id))
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)
    except AIUnavailableError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2)


_NOT_IMPLEMENTED = [
    "similar",
    "compare",
    "research",
]

for _name in _NOT_IMPLEMENTED:

    def _make_stub(name: str):
        def _stub() -> None:
            typer.echo(f"thermal {name}: NOT IMPLEMENTED (see docs/roadmap.md)")
            raise typer.Exit(code=2)

        return _stub

    app.command(name=_name)(_make_stub(_name))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
