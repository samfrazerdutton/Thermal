"""THERMAL command-line interface."""

from __future__ import annotations

import json as jsonlib
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from analysis.baseline import InsufficientSamplesError, compute_baseline
from thermal.diagnosis import classify, features_from_telemetry
from thermal.doctor import CheckStatus, core_ready, run_doctor
from thermal.experiment import ExperimentSpec, run_experiment
from thermal.hardware import collect_hardware_report
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
from thermal.storage import (
    ExperimentRecord,
    ExperimentRepository,
    RunRecord,
    SQLiteRunRepository,
    default_db_path,
)
from thermal.telemetry import TelemetryCollector
from thermal.workload import WorkloadRegistry, run_workload


def _parse_params(param: list[str]) -> dict:
    params: dict = {}
    for item in param:
        if "=" not in item:
            typer.echo(f"invalid --param '{item}', expected key=value")
            raise typer.Exit(code=1)
        key, value = item.split("=", 1)
        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value
    return params


def _git_commit() -> Optional[str]:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run([git, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None

import workloads  # noqa: F401  (import registers all built-in workloads)

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
        workload_cls = WorkloadRegistry.get(name)
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    instance = workload_cls(params)
    if samples is not None or warmup is not None:
        instance.spec = instance.spec.__class__(
            **{
                **instance.spec.__dict__,
                "warmup_iterations": warmup if warmup is not None else instance.spec.warmup_iterations,
                "measurement_iterations": samples if samples is not None else instance.spec.measurement_iterations,
            }
        )

    typer.echo(f"Running '{name}' - warmup={instance.spec.warmup_iterations} measured={instance.spec.measurement_iterations}")

    telemetry = TelemetryCollector(interval_seconds=0.2)
    telemetry.start()
    try:
        result = run_workload(instance)
    finally:
        telemetry.stop()

    typer.echo(f"\nBASELINE\nDevice: {result.device}\n")
    numeric_keys = [
        k
        for k in result.per_iteration_metrics[0]
        if k != "device" and isinstance(result.per_iteration_metrics[0].get(k), (int, float))
    ]
    baseline_metrics: dict = {}
    for key in numeric_keys:
        values = [m[key] for m in result.per_iteration_metrics if isinstance(m.get(key), (int, float))]
        try:
            baseline = compute_baseline(values, metric_name=key)
        except InsufficientSamplesError as exc:
            typer.echo(f"{key}: {exc}\n")
            continue
        baseline_metrics[key] = asdict(baseline)
        typer.echo(baseline.summary_text())
        typer.echo()

    if output is not None:
        output.write_text(jsonlib.dumps([m for m in result.per_iteration_metrics], indent=2))
        typer.echo(f"Wrote {len(result.per_iteration_metrics)} iteration records -> {output}")

    features = features_from_telemetry(telemetry.samples)
    diagnosis = classify(features)
    typer.echo("DIAGNOSIS")
    typer.echo(f"  {diagnosis.bottleneck.value}  (confidence: {diagnosis.confidence:.0%})")
    typer.echo(f"  {diagnosis.rationale}")
    for ev in diagnosis.evidence:
        typer.echo(f"    evidence: {ev.feature} = {ev.value:.2f} ({ev.comparison} {ev.threshold})")
    if diagnosis.missing_features:
        typer.echo(f"  not measurable on this run: {', '.join(diagnosis.missing_features)}")
    typer.echo()

    telemetry_dir = default_db_path().parent / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    report = collect_hardware_report()
    run_record = RunRecord.new(
        workload_name=instance.spec.name,
        workload_version=instance.spec.version,
        device=result.device,
        warmup_iterations=result.warmup_iterations,
        measurement_iterations=result.measurement_iterations,
        configuration=instance.params,
        hardware_fingerprint={
            "cpu_name": report.cpu.name,
            "gpu_name": report.gpu.name,
            "gpu_driver_version": report.gpu.driver_version,
            "cuda_driver_version": report.gpu.cuda_driver_version,
            "os": report.software.os_name,
        },
        metrics=baseline_metrics,
        git_commit=_git_commit(),
        diagnosis={
            "bottleneck": diagnosis.bottleneck.value,
            "confidence": diagnosis.confidence,
            "rationale": diagnosis.rationale,
            "evidence": [asdict(e) for e in diagnosis.evidence],
            "missing_features": diagnosis.missing_features,
        },
    )
    telemetry_path = telemetry_dir / f"{run_record.run_id}.jsonl"
    telemetry.write_jsonl(telemetry_path)
    run_record.telemetry_path = str(telemetry_path)

    repo = SQLiteRunRepository(default_db_path())
    repo.save(run_record)
    typer.echo(f"Saved run {run_record.run_id} -> {default_db_path()}")


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO "


@app.command()
def hardware(json: bool = typer.Option(False, "--json", help="emit machine-readable JSON")) -> None:
    """Detect and print real hardware/software capabilities."""
    report = collect_hardware_report()

    if json:
        typer.echo(
            jsonlib.dumps(
                {
                    "cpu": vars(report.cpu),
                    "gpu": {
                        k: v.to_dict() if hasattr(v, "to_dict") else v
                        for k, v in vars(report.gpu).items()
                        if k != "telemetry"
                    }
                    | {"telemetry": {k: v.to_dict() for k, v in report.gpu.telemetry.items()}},
                    "toolchain": {k: v.to_dict() for k, v in vars(report.toolchain).items()},
                    "software": {
                        "os_name": report.software.os_name,
                        "os_version": report.software.os_version,
                        "python_version": report.software.python_version,
                        "torch_installed": report.software.torch_installed.to_dict(),
                        "torch_cuda": report.software.torch_cuda.to_dict(),
                    },
                },
                indent=2,
                default=str,
            )
        )
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
    spec = ExperimentSpec(
        workload_name=workload,
        baseline_params=_parse_params(baseline_param),
        treatment_params=_parse_params(treatment_param),
        metric_name=metric,
        higher_is_better=not lower_is_better,
        repetitions=repetitions,
        warmup_iterations=warmup,
        hypothesis=hypothesis,
    )

    if hypothesis:
        typer.echo(f"HYPOTHESIS\n{hypothesis}\n")

    try:
        result = run_experiment(spec)
    except (KeyError, ValueError) as exc:
        typer.echo(f"Experiment failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo(f"Baseline device: {result.baseline_device}  Treatment device: {result.treatment_device}\n")
    typer.echo(result.comparison.summary_text())

    from dataclasses import asdict as _asdict

    record = ExperimentRecord.new(
        workload_name=workload,
        metric_name=metric,
        higher_is_better=not lower_is_better,
        repetitions=repetitions,
        baseline_config=spec.baseline_params,
        treatment_config=spec.treatment_params,
        baseline_values=result.baseline_values,
        treatment_values=result.treatment_values,
        baseline_device=result.baseline_device,
        treatment_device=result.treatment_device,
        comparison=_asdict(result.comparison),
        verdict=result.comparison.verdict.value,
        hypothesis=hypothesis,
        git_commit=_git_commit(),
    )
    ExperimentRepository(default_db_path()).save(record)
    typer.echo(f"\nSaved experiment {record.experiment_id} -> {default_db_path()}")


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


_NOT_IMPLEMENTED = [
    "optimize",
    "report",
    "serve",
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
