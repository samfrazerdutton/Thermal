"""Unit tests for thermal.device (device override -- lets a user force CPU
or GPU execution instead of always silently preferring whatever's available,
which matters for a direct CPU-vs-GPU comparison experiment)."""

import pytest

from thermal.device import gpu_available, select_device
from thermal.workload import UnsupportedHardwareError


def test_auto_prefers_gpu_when_available():
    expected = "cuda:0" if gpu_available() else "cpu"
    assert select_device(prefer_gpu=True, override="auto") == expected


def test_override_cpu_forces_cpu_even_if_gpu_available():
    assert select_device(prefer_gpu=True, override="cpu") == "cpu"


def test_override_cuda_returns_cuda_when_available():
    if not gpu_available():
        pytest.skip("no CUDA device on this machine to request")
    assert select_device(override="cuda:0") == "cuda:0"


def test_override_cuda_raises_when_unavailable(monkeypatch):
    import thermal.device as device_mod

    monkeypatch.setattr(device_mod, "gpu_available", lambda: False)
    with pytest.raises(UnsupportedHardwareError):
        select_device(override="cuda:0")


def test_invalid_override_raises_value_error():
    with pytest.raises(ValueError):
        select_device(override="not_a_real_device")


def test_default_override_is_auto():
    expected = "cuda:0" if gpu_available() else "cpu"
    assert select_device(prefer_gpu=True) == expected
