"""Real hardware and software capability detection.

Every field is either a genuine measurement or explicitly marked unavailable
with a reason. Nothing here is fabricated or estimated.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import psutil


@dataclass
class Capability:
    """A single detected fact: either a real value, or an honest gap."""

    name: str
    available: bool
    value: Any = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "available": self.available,
            "value": self.value,
            "reason": self.reason,
        }


def _run(cmd: list[str], timeout: float = 5.0) -> Optional[str]:
    """Run a subprocess and return stdout, or None if it can't be run."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@dataclass
class CPUInfo:
    name: str
    physical_cores: Optional[int]
    logical_cores: Optional[int]
    max_frequency_mhz: Optional[float]
    current_utilization_percent: Optional[float]


def detect_cpu() -> CPUInfo:
    name = platform.processor() or platform.uname().processor or "unknown"
    freq = psutil.cpu_freq()
    return CPUInfo(
        name=name,
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        max_frequency_mhz=freq.max if freq else None,
        current_utilization_percent=psutil.cpu_percent(interval=0.1),
    )


#: Real, published PCI-SIG per-lane, per-direction bandwidth in MB/s for each
#: PCIe generation (accounts for line coding overhead, e.g. Gen3's 128b/130b).
#: Used to turn NVML's raw KB/s throughput reading into a genuine percentage
#: of this GPU's actual negotiated link capacity, rather than leaving
#: pcie_utilization_percent permanently unmeasurable.
PCIE_LANE_MBPS = {1: 250, 2: 500, 3: 985, 4: 1969, 5: 3938}


def pcie_ceiling_kbps(generation: Optional[int], width: Optional[int]) -> Optional[float]:
    if generation is None or width is None or generation not in PCIE_LANE_MBPS:
        return None
    return PCIE_LANE_MBPS[generation] * width * 1000.0


@dataclass
class GPUInfo:
    available: bool
    name: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_driver_version: Optional[str] = None
    memory_total_mb: Optional[float] = None
    memory_used_mb: Optional[float] = None
    compute_capability: Optional[str] = None
    pcie_link_generation: Optional[int] = None
    pcie_link_width: Optional[int] = None
    telemetry: dict[str, Capability] = field(default_factory=dict)
    reason: Optional[str] = None


def detect_gpu() -> GPUInfo:
    try:
        import pynvml
    except ImportError:
        return GPUInfo(available=False, reason="nvidia-ml-py not installed")

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        return GPUInfo(available=False, reason=f"NVML init failed: {exc}")

    try:
        if pynvml.nvmlDeviceGetCount() == 0:
            return GPUInfo(available=False, reason="no NVIDIA GPU detected")

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()

        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode()

        cuda_version_int = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda_version = f"{cuda_version_int // 1000}.{(cuda_version_int % 1000) // 10}"

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = f"{major}.{minor}"
        except pynvml.NVMLError:
            compute_capability = None

        # Current negotiated link, not the card's max capability -- a laptop
        # or riser can (and this reference machine does) run at a narrower
        # width than the GPU supports, and that's the real constraint on
        # this system right now.
        try:
            pcie_link_generation = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
            pcie_link_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
        except pynvml.NVMLError:
            pcie_link_generation = None
            pcie_link_width = None

        telemetry: dict[str, Capability] = {}

        def _try(name_: str, fn):
            try:
                telemetry[name_] = Capability(name_, True, fn())
            except pynvml.NVMLError as exc:
                telemetry[name_] = Capability(name_, False, reason=str(exc))

        _try("utilization_percent", lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        _try("memory_utilization_percent", lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).memory)
        _try("temperature_c", lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        _try("power_watts", lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        _try("sm_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        _try("memory_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        _try("fan_speed_percent", lambda: pynvml.nvmlDeviceGetFanSpeed(handle))
        _try("performance_state", lambda: pynvml.nvmlDeviceGetPowerState(handle))
        _try(
            "pcie_throughput_tx_kbps",
            lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES),
        )
        _try(
            "pcie_throughput_rx_kbps",
            lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES),
        )
        _try("nvlink_state", lambda: pynvml.nvmlDeviceGetNvLinkState(handle, 0))

        return GPUInfo(
            available=True,
            name=name,
            driver_version=driver_version,
            cuda_driver_version=cuda_version,
            memory_total_mb=mem.total / (1024 * 1024),
            memory_used_mb=mem.used / (1024 * 1024),
            compute_capability=compute_capability,
            pcie_link_generation=pcie_link_generation,
            pcie_link_width=pcie_link_width,
            telemetry=telemetry,
        )
    finally:
        pynvml.nvmlShutdown()


@dataclass
class ToolchainInfo:
    cuda_toolkit: Capability
    msvc: Capability
    gcc: Capability
    cmake: Capability
    docker: Capability
    git: Capability
    nsight_systems: Capability
    nsight_compute: Capability


def _detect_msvc() -> Capability:
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not shutil.which(vswhere) and not __import__("os").path.exists(vswhere):
        return Capability("msvc", False, reason="vswhere.exe not found")
    output = _run(
        [
            vswhere,
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ]
    )
    if not output:
        return Capability("msvc", False, reason="no VC++ build tools component found")
    return Capability(
        "msvc",
        True,
        {"installation_path": output},
        reason="cl.exe requires running inside a Developer Command Prompt (vcvarsall.bat)",
    )


def detect_toolchain() -> ToolchainInfo:
    nvcc_out = _run(["nvcc", "--version"])
    cmake_out = _run(["cmake", "--version"])
    docker_out = _run(["docker", "--version"])
    git_out = _run(["git", "--version"])
    gxx_out = _run(["g++", "--version"])
    nsys_out = shutil.which("nsys")
    ncu_out = shutil.which("ncu") or shutil.which("ncu.bat")

    return ToolchainInfo(
        cuda_toolkit=Capability("cuda_toolkit", nvcc_out is not None, nvcc_out.splitlines()[-1] if nvcc_out else None,
                                 None if nvcc_out else "nvcc not found on PATH"),
        msvc=_detect_msvc(),
        gcc=Capability("gcc", gxx_out is not None, gxx_out.splitlines()[0] if gxx_out else None,
                        None if gxx_out else "g++ not found on PATH"),
        cmake=Capability("cmake", cmake_out is not None, cmake_out.splitlines()[0] if cmake_out else None,
                          None if cmake_out else "cmake not found on PATH"),
        docker=Capability("docker", docker_out is not None, docker_out, None if docker_out else "docker not found on PATH"),
        git=Capability("git", git_out is not None, git_out, None if git_out else "git not found on PATH"),
        nsight_systems=Capability("nsight_systems", nsys_out is not None, nsys_out,
                                   None if nsys_out else "nsys not found on PATH (optional profiling tool)"),
        nsight_compute=Capability("nsight_compute", ncu_out is not None, ncu_out,
                                   None if ncu_out else "ncu not found on PATH (optional profiling tool)"),
    )


@dataclass
class SoftwareInfo:
    os_name: str
    os_version: str
    python_version: str
    torch_installed: Capability
    torch_cuda: Capability


def detect_software() -> SoftwareInfo:
    try:
        import torch  # type: ignore

        torch_installed = Capability("torch", True, torch.__version__)
        try:
            cuda_ok = torch.cuda.is_available()
            torch_cuda = Capability(
                "torch_cuda",
                cuda_ok,
                torch.version.cuda if cuda_ok else None,
                None if cuda_ok else "installed torch build has no CUDA support (CPU-only wheel)",
            )
        except Exception as exc:  # pragma: no cover - defensive
            torch_cuda = Capability("torch_cuda", False, reason=str(exc))
    except ImportError:
        torch_installed = Capability("torch", False, reason="torch not installed")
        torch_cuda = Capability("torch_cuda", False, reason="torch not installed")

    return SoftwareInfo(
        os_name=platform.system(),
        os_version=platform.version(),
        python_version=sys.version.split()[0],
        torch_installed=torch_installed,
        torch_cuda=torch_cuda,
    )


@dataclass
class HardwareReport:
    cpu: CPUInfo
    gpu: GPUInfo
    toolchain: ToolchainInfo
    software: SoftwareInfo

    def to_dict(self) -> dict:
        return {
            "cpu": vars(self.cpu),
            "gpu": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in vars(self.gpu).items()
                if k != "telemetry"
            }
            | {"telemetry": {k: v.to_dict() for k, v in self.gpu.telemetry.items()}},
            "toolchain": {k: v.to_dict() for k, v in vars(self.toolchain).items()},
            "software": {
                "os_name": self.software.os_name,
                "os_version": self.software.os_version,
                "python_version": self.software.python_version,
                "torch_installed": self.software.torch_installed.to_dict(),
                "torch_cuda": self.software.torch_cuda.to_dict(),
            },
        }


def collect_hardware_report() -> HardwareReport:
    return HardwareReport(
        cpu=detect_cpu(),
        gpu=detect_gpu(),
        toolchain=detect_toolchain(),
        software=detect_software(),
    )
