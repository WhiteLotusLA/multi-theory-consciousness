"""
Stage Profiler
==============

Wall-clock timing of named pipeline stages, in milliseconds. ``stage(name)``
is a context manager that works in both ``with`` and ``async with`` blocks and
records the elapsed time on exit, including when the block raises.
``summary()`` reports count, mean, median, 95th percentile and maximum per
stage; a stage with no samples reports None for every statistic rather than a
zero, so an unmeasured stage is never read as an instant one.

Samples are bounded per stage (``max_samples``), so a long-running process
keeps a recent window. No Settings are read, nothing is persisted and the
stage names are the only text held.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Dict, Iterable, List, Optional


def _percentile(sorted_values: List[float], fraction: float) -> float:
    """Nearest-rank percentile of an ascending list (non-empty)."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(1, int(round(fraction * len(sorted_values))))
    return sorted_values[min(rank, len(sorted_values)) - 1]


class _Stage:
    """The context manager ``StageProfiler.stage`` returns."""

    def __init__(self, profiler: "StageProfiler", name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._start: Optional[float] = None

    def _enter(self) -> "_Stage":
        self._start = self._profiler._clock()
        return self

    def _exit(self) -> None:
        if self._start is None:
            return
        elapsed_ms = (self._profiler._clock() - self._start) * 1000.0
        self._profiler.record(self._name, elapsed_ms)
        self._start = None

    def __enter__(self) -> "_Stage":
        return self._enter()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._exit()
        return False

    async def __aenter__(self) -> "_Stage":
        return self._enter()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._exit()
        return False


class StageProfiler:
    """Bounded per-stage timing with percentile summaries.

    Args:
        max_samples: Samples kept per stage (at least 1).
        clock: Monotonic seconds source; defaults to ``time.perf_counter``.
    """

    def __init__(
        self, max_samples: int = 200, clock: Optional[Callable[[], float]] = None
    ) -> None:
        if int(max_samples) < 1:
            raise ValueError(f"max_samples must be at least 1, got {max_samples}")
        self.max_samples = int(max_samples)
        self._clock = clock or time.perf_counter
        self._samples: Dict[str, Deque[float]] = {}

    def stage(self, name: str) -> _Stage:
        """A context manager (sync or async) timing ``name``."""
        self._samples.setdefault(name, deque(maxlen=self.max_samples))
        return _Stage(self, str(name))

    def record(self, name: str, elapsed_ms: float) -> None:
        """Record one sample for ``name`` in milliseconds."""
        bucket = self._samples.setdefault(name, deque(maxlen=self.max_samples))
        bucket.append(float(elapsed_ms))

    def stages(self) -> List[str]:
        """Stage names seen since construction (or the last ``reset``)."""
        return list(self._samples)

    def summary(
        self, names: Optional[Iterable[str]] = None
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Per-stage ``count``, ``mean``, ``p50``, ``p95`` and ``max`` in
        milliseconds; every statistic is None for a stage without samples."""
        selected = list(names) if names is not None else list(self._samples)
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for name in selected:
            values = sorted(self._samples.get(name, ()))
            if not values:
                out[name] = {
                    "count": 0,
                    "mean": None,
                    "p50": None,
                    "p95": None,
                    "max": None,
                }
                continue
            out[name] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "max": values[-1],
            }
        return out

    def reset(self) -> None:
        """Drop every sample; the stage names stay known and report None."""
        for bucket in self._samples.values():
            bucket.clear()
