"""THERMAL command-line interface."""

from __future__ import annotations

import json as jsonlib
import time
from pathlib import Path
from typing import Optional

import typer

from thermal.doctor import CheckStatus, core_ready, run_doctor
from thermal.hardware import collect_hardware_report
from thermal.telemetry import TelemetryCollector
from thermal.workload import WorkloadRegistry, run_workload

import workloads  # noqa: F401  (import registers all built-in workloads)

app = typer.Typer(
    name="thermal",
    help="THERMAL: closed-loop performance engineering for AI and accelerated workloads.",
    no_args_is_help=True,
)

workload_app = typer.Typer(help="Run and inspect THERMAL workloads.", no_args_is_help=True)
app.add_typer(workload_app, name="workload")


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
    output: Optional[Path] = typer.Option(None, "--output", help="write per-iteration metrics as JSON"),
) -> None:
    """Run one workload: warmup iterations, then measured iterations."""
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

    try:
        workload_cls = WorkloadRegistry.get(name)
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)

    instance = workload_cls(params)
    typer.echo(f"Running '{name}' - warmup={instance.spec.warmup_iterations} measured={instance.spec.measurement_iterations}")
    result = run_workload(instance)

    typer.echo(f"Device: {result.device}")
    keys = [k for k in result.per_iteration_metrics[0] if k != "device"]
    for key in keys:
        values = [m[key] for m in result.per_iteration_metrics if isinstance(m.get(key), (int, float))]
        if values:
            typer.echo(f"  {key}: mean={sum(values) / len(values):.4f}  min={min(values):.4f}  max={max(values):.4f}")

    if output is not None:
        output.write_text(jsonlib.dumps([m for m in result.per_iteration_metrics], indent=2))
        typer.echo(f"Wrote {len(result.per_iteration_metrics)} iteration records -> {output}")


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


_NOT_IMPLEMENTED = [
    "diagnose",
    "experiment",
    "optimize",
    "report",
    "serve",
    "database",
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
