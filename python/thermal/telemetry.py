"""Continuous hardware/process telemetry sampling.

Produces records that conform to schemas/telemetry_v1.json. Every counter this
process cannot actually read is reported under `unavailable_counters` with a
reason -- never silently omitted, never substituted with an estimate.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import psutil

SCHEMA_VERSION = 1
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "telemetry_v1.json"


@dataclass
class UnavailableCounter:
    name: str
    reason: str


@dataclass
class CPUSample:
    utilization_percent: Optional[float] = None
    per_core_utilization_percent: Optional[list[float]] = None
    load_average_1m: Optional[float] = None
    context_switches_per_sec: Optional[float] = None
    page_faults_per_sec: Optional[float] = None
    memory_used_mb: Optional[float] = None
    frequency_mhz: Optional[float] = None
    unavailable_counters: list[UnavailableCounter] = field(default_factory=list)


@dataclass
class GPUSample:
    device_index: int = 0
    utilization_percent: Optional[float] = None
    memory_utilization_percent: Optional[float] = None
    memory_used_mb: Optional[float] = None
    temperature_c: Optional[float] = None
    power_watts: Optional[float] = None
    sm_clock_mhz: Optional[float] = None
    memory_clock_mhz: Optional[float] = None
    pcie_tx_kbps: Optional[float] = None
    pcie_rx_kbps: Optional[float] = None
    fan_speed_percent: Optional[float] = None
    performance_state: Optional[int] = None
    unavailable_counters: list[UnavailableCounter] = field(default_factory=list)


@dataclass
class ProcessSample:
    pid: int
    cpu_time_seconds: Optional[float] = None
    rss_mb: Optional[float] = None
    num_threads: Optional[int] = None
    io_read_bytes: Optional[int] = None
    io_write_bytes: Optional[int] = None
    num_ctx_switches: Optional[int] = None


@dataclass
class ApplicationSample:
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    batch_size: Optional[int] = None
    queue_time_ms: Optional[float] = None


@dataclass
class TelemetrySample:
    run_id: str
    timestamp_ns: int
    monotonic_ns: int
    cpu: CPUSample
    gpu: Optional[GPUSample]
    process: Optional[ProcessSample]
    application: Optional[ApplicationSample] = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "timestamp_ns": self.timestamp_ns,
            "monotonic_ns": self.monotonic_ns,
            "run_id": self.run_id,
            "cpu": asdict(self.cpu),
            "gpu": asdict(self.gpu) if self.gpu is not None else None,
            "process": asdict(self.process) if self.process is not None else None,
            "application": asdict(self.application) if self.application is not None else None,
        }


def validate_sample(sample: dict) -> None:
    """Raise jsonschema.ValidationError if `sample` doesn't match telemetry_v1."""
    import jsonschema

    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.validate(instance=sample, schema=schema)


class _NvmlGpuSource:
    """Wraps pynvml so GPU sampling degrades to `None` cleanly when unavailable."""

    def __init__(self) -> None:
        self._pynvml = None
        self._handle = None
        self._init_error: Optional[str] = None
        try:
            import pynvml

            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() == 0:
                self._init_error = "no NVIDIA GPU detected"
            else:
                self._pynvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:  # ImportError or NVMLError
            self._init_error = str(exc)

    @property
    def available(self) -> bool:
        return self._handle is not None

    def sample(self) -> GPUSample:
        pynvml = self._pynvml
        handle = self._handle
        result = GPUSample(device_index=0)
        unavailable: list[UnavailableCounter] = []

        def _try(field_name: str, fn):
            try:
                setattr(result, field_name, fn())
            except pynvml.NVMLError as exc:
                unavailable.append(UnavailableCounter(field_name, str(exc)))

        _try("utilization_percent", lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
        _try("memory_utilization_percent", lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).memory))
        _try("memory_used_mb", lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024))
        _try("temperature_c", lambda: float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)))
        _try("power_watts", lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        _try("sm_clock_mhz", lambda: float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)))
        _try("memory_clock_mhz", lambda: float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)))
        _try(
            "pcie_tx_kbps",
            lambda: float(pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)),
        )
        _try(
            "pcie_rx_kbps",
            lambda: float(pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)),
        )
        _try("fan_speed_percent", lambda: float(pynvml.nvmlDeviceGetFanSpeed(handle)))
        _try("performance_state", lambda: int(pynvml.nvmlDeviceGetPowerState(handle)))

        result.unavailable_counters = unavailable
        return result

    def shutdown(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


class TelemetryCollector:
    """Samples system + optional process telemetry on a background thread.

    Usage:
        collector = TelemetryCollector(run_id, pid=os.getpid(), interval_seconds=0.5)
        collector.start()
        ... run workload ...
        collector.stop()
        collector.write_jsonl(Path("run.telemetry.jsonl"))
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        pid: Optional[int] = None,
        interval_seconds: float = 0.5,
        on_sample: Optional[Callable[[TelemetrySample], None]] = None,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.on_sample = on_sample

        self._samples: list[TelemetrySample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._gpu_source = _NvmlGpuSource()
        self._process = psutil.Process(pid) if pid is not None else None

        # priming reads so the first real sample has a non-None delta baseline
        psutil.cpu_percent(percpu=True)
        self._prev_cpu_stats = psutil.cpu_stats()
        self._prev_stats_time = time.monotonic()

    def _sample_cpu(self) -> CPUSample:
        now = time.monotonic()
        elapsed = max(now - self._prev_stats_time, 1e-9)
        stats = psutil.cpu_stats()
        ctx_per_sec = (stats.ctx_switches - self._prev_cpu_stats.ctx_switches) / elapsed
        unavailable: list[UnavailableCounter] = []
        faults_per_sec = None
        if hasattr(stats, "page_faults") and hasattr(self._prev_cpu_stats, "page_faults"):
            faults_per_sec = (stats.page_faults - self._prev_cpu_stats.page_faults) / elapsed
        else:
            unavailable.append(
                UnavailableCounter("page_faults_per_sec", "psutil.cpu_stats() exposes no page_faults counter on this platform (Windows)")
            )
        self._prev_cpu_stats = stats
        self._prev_stats_time = now

        try:
            load1 = psutil.getloadavg()[0]
        except (OSError, AttributeError):
            load1 = None
            unavailable.append(UnavailableCounter("load_average_1m", "not available on this platform (Windows)"))

        freq = psutil.cpu_freq()
        vmem = psutil.virtual_memory()

        return CPUSample(
            utilization_percent=psutil.cpu_percent(percpu=False),
            per_core_utilization_percent=psutil.cpu_percent(percpu=True),
            load_average_1m=load1,
            context_switches_per_sec=ctx_per_sec,
            page_faults_per_sec=faults_per_sec,
            memory_used_mb=vmem.used / (1024 * 1024),
            frequency_mhz=freq.current if freq else None,
            unavailable_counters=unavailable,
        )

    def _sample_process(self) -> Optional[ProcessSample]:
        if self._process is None:
            return None
        try:
            with self._process.oneshot():
                cpu_times = self._process.cpu_times()
                mem = self._process.memory_info()
                try:
                    io = self._process.io_counters()
                    io_read, io_write = io.read_bytes, io.write_bytes
                except (psutil.AccessDenied, NotImplementedError):
                    io_read, io_write = None, None
                try:
                    ctx = self._process.num_ctx_switches()
                    num_ctx = ctx.voluntary + ctx.involuntary
                except (psutil.AccessDenied, NotImplementedError):
                    num_ctx = None

                return ProcessSample(
                    pid=self._process.pid,
                    cpu_time_seconds=cpu_times.user + cpu_times.system,
                    rss_mb=mem.rss / (1024 * 1024),
                    num_threads=self._process.num_threads(),
                    io_read_bytes=io_read,
                    io_write_bytes=io_write,
                    num_ctx_switches=num_ctx,
                )
        except psutil.NoSuchProcess:
            return None

    def sample_once(self) -> TelemetrySample:
        sample = TelemetrySample(
            run_id=self.run_id,
            timestamp_ns=time.time_ns(),
            monotonic_ns=time.monotonic_ns(),
            cpu=self._sample_cpu(),
            gpu=self._gpu_source.sample() if self._gpu_source.available else None,
            process=self._sample_process(),
        )
        with self._lock:
            self._samples.append(sample)
        if self.on_sample is not None:
            self.on_sample(sample)
        return sample

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.sample_once()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("TelemetryCollector already started")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds * 5 + 1)
        self._thread = None
        self._gpu_source.shutdown()

    @property
    def samples(self) -> list[TelemetrySample]:
        with self._lock:
            return list(self._samples)

    def write_jsonl(self, path: Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for sample in self.samples:
                fh.write(json.dumps(sample.to_dict()) + "\n")

    def write_parquet(self, path: Path) -> None:
        import pandas as pd

        rows = []
        for sample in self.samples:
            d = sample.to_dict()
            flat = {"schema_version": d["schema_version"], "timestamp_ns": d["timestamp_ns"], "monotonic_ns": d["monotonic_ns"], "run_id": d["run_id"]}
            for section in ("cpu", "gpu", "process", "application"):
                if d[section] is not None:
                    for k, v in d[section].items():
                        if k == "unavailable_counters":
                            continue
                        if isinstance(v, list):
                            continue
                        flat[f"{section}.{k}"] = v
            rows.append(flat)
        pd.DataFrame(rows).to_parquet(Path(path))
