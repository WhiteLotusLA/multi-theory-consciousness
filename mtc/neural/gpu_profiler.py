"""
GPU Profiler
============

Measures operations on the device the device manager resolves
(``mtc.neural.mps_utils``, which follows ``MTC_DEVICE``): wall time from
``time.perf_counter`` around a ``profile_operation`` block (the accelerator is
synchronized before the end time is taken), and the allocated device memory
before and after where the backend reports it. On the CPU no allocator figure
exists, so the memory fields are None rather than zero.

A baseline per operation turns later timings into regression warnings: an
operation slower than ``regression_factor`` times its baseline is logged and
counted. ``auto_baseline_from_recent`` takes a percentile (0 to 1, inclusive)
of the recorded durations. A memory change larger than
``transfer_threshold_mb`` is logged as a large transfer. The last
``max_history`` operations are kept. Every figure here is measured at runtime;
nothing states a speedup. No config keys beyond the device switch and no
persisted state.
"""

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from mtc.neural.mps_utils import device_status, mps_manager

logger = logging.getLogger(__name__)


@dataclass
class OperationProfile:
    """One measured operation; memory fields are None when unmeasured."""

    operation_name: str
    start_time: float
    duration_ms: float
    memory_before_mb: Optional[float]
    memory_after_mb: Optional[float]
    memory_delta_mb: Optional[float]
    device: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class GPUProfiler:
    """Times operations and tracks device memory on the resolved device."""

    def __init__(
        self,
        transfer_threshold_mb: float = 10.0,
        max_history: int = 1000,
        regression_factor: float = 2.0,
        enabled: bool = True,
        manager: Any = None,
    ) -> None:
        if max_history < 1:
            raise ValueError("max_history must be at least 1")
        if regression_factor <= 1.0:
            raise ValueError("regression_factor must be greater than 1")
        self.transfer_threshold_mb = float(transfer_threshold_mb)
        self.max_history = int(max_history)
        self.regression_factor = float(regression_factor)
        self.enabled = bool(enabled)
        self._manager = manager if manager is not None else mps_manager
        self.operations: List[OperationProfile] = []
        self.baselines: Dict[str, float] = {}
        self.stats: Dict[str, float] = {
            "total_operations": 0,
            "total_time_ms": 0.0,
            "total_memory_allocated_mb": 0.0,
            "total_memory_freed_mb": 0.0,
            "large_transfers_detected": 0,
            "regressions_detected": 0,
            "warnings_issued": 0,
        }

    def get_current_memory_mb(self) -> Optional[float]:
        """Return the allocated device memory in MB, or None when unmeasured."""
        value = self._manager.get_memory_stats().get("allocated_mb")
        return float(value) if value is not None else None

    @contextmanager
    def profile_operation(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Iterator[None]:
        """Time the enclosed block; it is recorded even when it raises."""
        if not self.enabled:
            yield
            return
        wall_start = time.time()
        memory_before = self.get_current_memory_mb()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._manager.synchronize()
            duration_ms = (time.perf_counter() - started) * 1000.0
            memory_after = self.get_current_memory_mb()
            delta = (
                memory_after - memory_before
                if memory_before is not None and memory_after is not None
                else None
            )
            self._record(
                OperationProfile(
                    operation_name=operation_name,
                    start_time=wall_start,
                    duration_ms=duration_ms,
                    memory_before_mb=memory_before,
                    memory_after_mb=memory_after,
                    memory_delta_mb=delta,
                    device=str(self._manager.device),
                    metadata=dict(metadata or {}),
                )
            )

    def _record(self, profile: OperationProfile) -> None:
        self.operations.append(profile)
        if len(self.operations) > self.max_history:
            del self.operations[: len(self.operations) - self.max_history]
        self.stats["total_operations"] += 1
        self.stats["total_time_ms"] += profile.duration_ms
        delta = profile.memory_delta_mb
        if delta is not None:
            if delta > 0:
                self.stats["total_memory_allocated_mb"] += delta
            else:
                self.stats["total_memory_freed_mb"] += abs(delta)
            if abs(delta) > self.transfer_threshold_mb:
                self.stats["large_transfers_detected"] += 1
                self.stats["warnings_issued"] += 1
                logger.warning(
                    "large memory change in %s: %+.1f MB",
                    profile.operation_name,
                    delta,
                )
        baseline = self.baselines.get(profile.operation_name)
        if baseline is not None and profile.duration_ms > (
            baseline * self.regression_factor
        ):
            self.stats["regressions_detected"] += 1
            self.stats["warnings_issued"] += 1
            logger.warning(
                "performance regression in %s: %.2f ms against a %.2f ms baseline",
                profile.operation_name,
                profile.duration_ms,
                baseline,
            )

    def set_baseline(self, operation_name: str, duration_ms: float) -> None:
        """Set the baseline duration of an operation."""
        self.baselines[operation_name] = float(duration_ms)

    def auto_baseline_from_recent(
        self, operation_name: str, percentile: float = 0.5
    ) -> Optional[float]:
        """Set the baseline from a percentile (0 to 1) of recorded durations.

        Returns the baseline, or None when the operation was never recorded.
        """
        if not 0.0 <= percentile <= 1.0:
            raise ValueError("percentile must be within [0, 1]")
        durations = sorted(
            op.duration_ms
            for op in self.operations
            if op.operation_name == operation_name
        )
        if not durations:
            logger.warning("no recorded operations named %s", operation_name)
            return None
        index = min(int(len(durations) * percentile), len(durations) - 1)
        self.set_baseline(operation_name, durations[index])
        return durations[index]

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Return count, time and memory figures of one operation."""
        matching = [op for op in self.operations if op.operation_name == operation_name]
        if not matching:
            return {"count": 0}
        durations = [op.duration_ms for op in matching]
        deltas = [
            op.memory_delta_mb for op in matching if op.memory_delta_mb is not None
        ]
        return {
            "count": len(matching),
            "total_time_ms": sum(durations),
            "avg_time_ms": sum(durations) / len(durations),
            "min_time_ms": min(durations),
            "max_time_ms": max(durations),
            "total_memory_delta_mb": sum(deltas) if deltas else None,
            "avg_memory_delta_mb": sum(deltas) / len(deltas) if deltas else None,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return the totals, per-operation figures, memory and baselines."""
        names = sorted({op.operation_name for op in self.operations})
        return {
            "enabled": self.enabled,
            "device": str(self._manager.device),
            "current_memory_mb": self.get_current_memory_mb(),
            "memory": self._manager.get_memory_stats(),
            "total_stats": dict(self.stats),
            "operation_stats": {name: self.get_operation_stats(name) for name in names},
            "num_operations_tracked": len(self.operations),
            "num_unique_operations": len(names),
            "baselines": dict(self.baselines),
        }

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the device.

        Never raises: a device the manager cannot provide is reported here.
        """
        return device_status(
            lambda: self._manager.device,
            enabled=self.enabled,
            operations=len(self.operations),
        )

    def clear_history(self) -> None:
        """Forget the recorded operations (baselines and totals stay)."""
        self.operations.clear()

    def __repr__(self) -> str:
        return f"GPUProfiler(enabled={self.enabled}, operations={len(self.operations)})"


def profile_gpu_function(
    profiler: Optional[GPUProfiler] = None, operation_name: Optional[str] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a function so each call is profiled (global profiler by default)."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            active = profiler or global_profiler
            name = operation_name or f"{func.__module__}.{func.__name__}"
            with active.profile_operation(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# The shared profiler; it resolves the device on first use.
global_profiler = GPUProfiler()
