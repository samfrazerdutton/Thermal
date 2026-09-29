"""THERMAL command-line interface."""

from __future__ import annotations

import json as jsonlib

import typer

from thermal.doctor import CheckStatus, core_ready, run_doctor
from thermal.hardware import collect_hardware_report

app = typer.Typer(
    name="thermal",
    help="THERMAL: closed-loop performance engineering for AI and accelerated workloads.",
    no_args_is_help=True,
)


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


_NOT_IMPLEMENTED = [
    "workload",
    "profile",
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
