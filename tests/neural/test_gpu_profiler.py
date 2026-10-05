"""GPU profiler: measured timings on the resolved device, honest baselines."""

import logging
import os

import pytest

torch = pytest.importorskip("torch")

from mtc.core import device as device_module  # noqa: E402
from mtc.core.config import Settings  # noqa: E402
from mtc.neural import gpu_profiler as profiler_module  # noqa: E402
from mtc.neural.gpu_profiler import (  # noqa: E402
    GPUProfiler,
    global_profiler,
    profile_gpu_function,
)
from mtc.neural.mps_utils import MPSManager  # noqa: E402


@pytest.fixture
def cpu_manager(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    device_module._LOGGED.clear()
    return MPSManager(settings=Settings())


class Ticks:
    """A perf_counter stand-in that returns the scripted instants in order."""

    def __init__(self, *instants: float) -> None:
        self.instants = list(instants)

    def __call__(self) -> float:
        return self.instants.pop(0)


def test_profile_context_records_duration(cpu_manager, monkeypatch):
    profiler = GPUProfiler(manager=cpu_manager)
    monkeypatch.setattr(profiler_module.time, "perf_counter", Ticks(10.0, 10.25))
    with profiler.profile_operation("matmul", metadata={"rows": 4}):
        pass
    (record,) = profiler.operations
    assert record.operation_name == "matmul"
    assert record.duration_ms == pytest.approx(250.0)
    assert record.metadata == {"rows": 4}
    assert record.memory_before_mb is None and record.memory_delta_mb is None
    assert profiler.stats["total_operations"] == 1
    assert profiler.stats["total_time_ms"] == pytest.approx(250.0)

    monkeypatch.setattr(profiler_module.time, "perf_counter", Ticks(0.0, 0.001))
    with pytest.raises(RuntimeError):
        with profiler.profile_operation("failing"):
            raise RuntimeError("boom")
    assert profiler.operations[-1].operation_name == "failing"  # still recorded
    stats = profiler.get_operation_stats("matmul")
    assert stats["count"] == 1 and stats["avg_time_ms"] == pytest.approx(250.0)
    assert profiler.get_operation_stats("unknown") == {"count": 0}


def test_baseline_percentile_one_does_not_raise(cpu_manager, monkeypatch):
    profiler = GPUProfiler(manager=cpu_manager)
    monkeypatch.setattr(
        profiler_module.time,
        "perf_counter",
        Ticks(0.0, 0.001, 1.0, 1.003, 2.0, 2.002),
    )
    for _ in range(3):
        with profiler.profile_operation("op"):
            pass
    assert profiler.auto_baseline_from_recent("op", 1.0) == pytest.approx(3.0)
    assert profiler.baselines["op"] == pytest.approx(3.0)
    assert profiler.auto_baseline_from_recent("op", 0.0) == pytest.approx(1.0)
    assert profiler.auto_baseline_from_recent("op", 0.5) == pytest.approx(2.0)
    assert profiler.auto_baseline_from_recent("absent") is None
    with pytest.raises(ValueError):
        profiler.auto_baseline_from_recent("op", 1.5)


def test_regression_uses_the_factor(cpu_manager, monkeypatch, caplog):
    profiler = GPUProfiler(manager=cpu_manager, regression_factor=3.0)
    profiler.set_baseline("op", 10.0)
    monkeypatch.setattr(profiler_module.time, "perf_counter", Ticks(0.0, 0.025))
    with profiler.profile_operation("op"):
        pass  # 25 ms: slower than the baseline but within the factor
    assert profiler.stats["regressions_detected"] == 0
    caplog.set_level(logging.WARNING, logger="mtc.neural.gpu_profiler")
    monkeypatch.setattr(profiler_module.time, "perf_counter", Ticks(0.0, 0.031))
    with profiler.profile_operation("op"):
        pass
    assert profiler.stats["regressions_detected"] == 1
    assert any("regression" in r.getMessage() for r in caplog.records)


def test_history_is_bounded(cpu_manager):
    profiler = GPUProfiler(manager=cpu_manager, max_history=2)
    for name in ("a", "b", "c"):
        with profiler.profile_operation(name):
            pass
    assert [op.operation_name for op in profiler.operations] == ["b", "c"]
    assert profiler.stats["total_operations"] == 3


def test_disabled_profiler_records_nothing(cpu_manager):
    profiler = GPUProfiler(manager=cpu_manager, enabled=False)
    with profiler.profile_operation("op"):
        pass
    assert profiler.operations == [] and profiler.stats["total_operations"] == 0


def test_profiler_reports_resolved_device(cpu_manager):
    profiler = GPUProfiler(manager=cpu_manager)

    @profile_gpu_function(profiler=profiler, operation_name="decorated")
    def work(x):
        return x * 2

    assert work(torch.ones(2)).tolist() == [2.0, 2.0]
    assert profiler.operations[-1].device == "cpu"
    summary = profiler.get_summary()
    assert summary["device"] == "cpu"
    assert summary["current_memory_mb"] is None  # the CPU path measures none
    assert summary["memory"]["device"] == "cpu"
    assert "decorated" in summary["operation_stats"]
    assert isinstance(global_profiler, GPUProfiler)


def test_profiler_status(cpu_manager, monkeypatch):
    report = GPUProfiler(manager=cpu_manager).status()
    assert report["backend"] == "torch" and report["available"] is True
    assert report["device"] == "cpu" and report["last_error"] is None
    assert report["operations"] == 0
    monkeypatch.setenv("MTC_DEVICE", "cuda")
    monkeypatch.setattr(device_module, "_cuda_available", lambda t: False)
    missing = MPSManager(settings=Settings())
    report = GPUProfiler(manager=missing).status()  # never raises
    assert report["available"] is False and "MTC_DEVICE" in report["last_error"]
