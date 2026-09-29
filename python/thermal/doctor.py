"""thermal doctor: verify the local machine can actually run THERMAL."""

from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from thermal.hardware import collect_hardware_report


class CheckStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class DoctorCheck:
    name: str
    status: CheckStatus
    required: bool
    detail: Optional[str] = None


def _check_sqlite() -> DoctorCheck:
    try:
        path = Path(tempfile.gettempdir()) / "thermal_doctor_check.sqlite3"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        conn.close()
        path.unlink(missing_ok=True)
        return DoctorCheck("sqlite", CheckStatus.OK, required=True, detail=f"sqlite3 {sqlite3.sqlite_version}")
    except Exception as exc:
        return DoctorCheck("sqlite", CheckStatus.FAIL, required=True, detail=str(exc))


def _check_parquet() -> DoctorCheck:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401

        return DoctorCheck("parquet", CheckStatus.OK, required=True, detail=f"pyarrow {pyarrow.__version__}")
    except ImportError as exc:
        return DoctorCheck("parquet", CheckStatus.FAIL, required=True, detail=str(exc))


def run_doctor() -> list[DoctorCheck]:
    report = collect_hardware_report()
    checks: list[DoctorCheck] = []

    checks.append(
        DoctorCheck(
            "cuda_toolkit",
            CheckStatus.OK if report.toolchain.cuda_toolkit.available else CheckStatus.WARN,
            required=False,
            detail=report.toolchain.cuda_toolkit.value or report.toolchain.cuda_toolkit.reason,
        )
    )
    checks.append(
        DoctorCheck(
            "nvidia_gpu_nvml",
            CheckStatus.OK if report.gpu.available else CheckStatus.FAIL,
            required=True,
            detail=report.gpu.name or report.gpu.reason,
        )
    )

    have_compiler = report.toolchain.msvc.available or report.toolchain.gcc.available
    checks.append(
        DoctorCheck(
            "cpp_compiler",
            CheckStatus.OK if have_compiler else CheckStatus.WARN,
            required=False,
            detail=(
                "MSVC build tools found (run from Developer Command Prompt to build native/)"
                if report.toolchain.msvc.available
                else (report.toolchain.gcc.value if report.toolchain.gcc.available else "no C++ compiler found")
            ),
        )
    )
    checks.append(
        DoctorCheck(
            "cmake",
            CheckStatus.OK if report.toolchain.cmake.available else CheckStatus.FAIL,
            required=True,
            detail=report.toolchain.cmake.value or report.toolchain.cmake.reason,
        )
    )
    checks.append(
        DoctorCheck(
            "python",
            CheckStatus.OK,
            required=True,
            detail=report.software.python_version,
        )
    )
    checks.append(
        DoctorCheck(
            "torch",
            CheckStatus.OK if report.software.torch_installed.available else CheckStatus.FAIL,
            required=True,
            detail=report.software.torch_installed.value or report.software.torch_installed.reason,
        )
    )
    checks.append(
        DoctorCheck(
            "torch_cuda",
            CheckStatus.OK if report.software.torch_cuda.available else CheckStatus.WARN,
            required=False,
            detail=report.software.torch_cuda.value or report.software.torch_cuda.reason,
        )
    )
    checks.append(_check_sqlite())
    checks.append(_check_parquet())
    checks.append(
        DoctorCheck(
            "nsight_systems",
            CheckStatus.OK if report.toolchain.nsight_systems.available else CheckStatus.WARN,
            required=False,
            detail=report.toolchain.nsight_systems.value or report.toolchain.nsight_systems.reason,
        )
    )
    checks.append(
        DoctorCheck(
            "nsight_compute",
            CheckStatus.OK if report.toolchain.nsight_compute.available else CheckStatus.WARN,
            required=False,
            detail=report.toolchain.nsight_compute.value or report.toolchain.nsight_compute.reason,
        )
    )

    return checks


def core_ready(checks: list[DoctorCheck]) -> bool:
    return all(c.status != CheckStatus.FAIL for c in checks if c.required)
