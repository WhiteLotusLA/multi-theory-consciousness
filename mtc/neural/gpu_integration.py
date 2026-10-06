"""
GPU Integration Layer
=====================

Chooses between the accelerated and the CPU variants of the neural components
from one configuration, on the device the device manager resolves
(``mtc.neural.mps_utils``, which follows ``MTC_DEVICE``):

- vector coordinator: ``MPSVectorCoordinator`` (batch embedding and top-k on
  the device) or ``VectorCoordinator``;
- liquid state machine: ``MPSLiquidStateMachine`` (torch reservoir) or the
  numpy ``LiquidStateMachine``;
- thought processor: the continuous thought machine's novelty filter,
  ``AcceleratedNoveltyFilter`` (its cache on the device) or the numpy
  ``NoveltyFilter`` (``mtc.consciousness.continuous.novelty``), imported behind
  a guard so the neural package imports without the continuous package.

``GPUIntegrationConfig`` records what was requested; whether a component runs
accelerated is decided when it is built (requested and the manager reports an
accelerated device), so building the configuration resolves no device. The
vector factory passes no host, port or collection of its own: the coordinator
reads ``QDRANT_HOST``, ``QDRANT_PORT`` and ``QDRANT_COLLECTION`` from Settings
unless the caller passes them. ``GPUPerformanceMonitor`` sums the operation
counts and the time savings its callers measured and reports the memory pool
and profiler figures; it states no speedup of its own. No persisted state.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from mtc.neural.gpu_profiler import GPUProfiler, global_profiler
from mtc.neural.liquid.lsm_core import LiquidStateMachine
from mtc.neural.liquid.mps_lsm import MPSLiquidStateMachine
from mtc.neural.memory_pool import global_memory_pool
from mtc.neural.mps_utils import mps_manager
from mtc.neural.vectors.coordinator import VectorCoordinator
from mtc.neural.vectors.mps_coordinator import MPSVectorCoordinator

# Optional thought processor: the continuous thought machine's novelty filter.
try:
    from mtc.consciousness.continuous.novelty import (
        AcceleratedNoveltyFilter,
        NoveltyFilter,
    )

    _HAS_THOUGHT_PROCESSOR = True
except ImportError:
    AcceleratedNoveltyFilter = None
    NoveltyFilter = None
    _HAS_THOUGHT_PROCESSOR = False

logger = logging.getLogger(__name__)

__all__ = [
    "GPUComponentFactory",
    "GPUIntegrationConfig",
    "GPUPerformanceMonitor",
    "GPUProfiler",
    "create_gpu_thought_processor",
    "create_gpu_vector_coordinator",
    "get_gpu_performance_summary",
    "gpu_config",
    "gpu_factory",
    "gpu_monitor",
    "print_gpu_performance",
]


class GPUIntegrationConfig:
    """Which components should run accelerated (decided against the device
    when each is built)."""

    def __init__(
        self,
        enable_gpu: bool = True,
        gpu_vectors: bool = True,
        gpu_thoughts: bool = True,
        gpu_lsm_batch: bool = True,
        gpu_snn: bool = False,
        gpu_htm: bool = False,
        profiling_enabled: bool = True,
        memory_pool_enabled: bool = True,
        manager: Any = None,
    ) -> None:
        self.requested = {
            "enable_gpu": bool(enable_gpu),
            "gpu_vectors": bool(gpu_vectors),
            "gpu_thoughts": bool(gpu_thoughts),
            "gpu_lsm_batch": bool(gpu_lsm_batch),
            "gpu_snn": bool(gpu_snn),
            "gpu_htm": bool(gpu_htm),
        }
        self.profiling_enabled = bool(profiling_enabled)
        self.memory_pool_enabled = bool(memory_pool_enabled)
        self.manager = manager if manager is not None else mps_manager

    def _active(self, name: str) -> bool:
        if not self.requested["enable_gpu"] or not self.requested[name]:
            return False
        return bool(self.manager.enabled)

    @property
    def enable_gpu(self) -> bool:
        return self.requested["enable_gpu"] and bool(self.manager.enabled)

    @property
    def gpu_vectors(self) -> bool:
        return self._active("gpu_vectors")

    @property
    def gpu_thoughts(self) -> bool:
        return self._active("gpu_thoughts")

    @property
    def gpu_lsm_batch(self) -> bool:
        return self._active("gpu_lsm_batch")

    @property
    def gpu_snn(self) -> bool:
        return self._active("gpu_snn")

    @property
    def gpu_htm(self) -> bool:
        return self._active("gpu_htm")

    def describe(self) -> Dict[str, Any]:
        """Return the resolved device and the per-component choice."""
        return {
            "device": str(self.manager.device),
            "vectors": "gpu" if self.gpu_vectors else "cpu",
            "thoughts": "gpu" if self.gpu_thoughts else "cpu",
            "lsm": "gpu" if self.gpu_lsm_batch else "cpu",
            "snn": "gpu" if self.gpu_snn else "cpu",
            "htm": "gpu" if self.gpu_htm else "cpu",
            "profiling": self.profiling_enabled,
            "memory_pool": self.memory_pool_enabled,
        }

    @classmethod
    def optimal_config(cls) -> "GPUIntegrationConfig":
        """Vectors, thoughts and LSM batches accelerated; SNN and HTM on the CPU."""
        return cls(
            enable_gpu=True,
            gpu_vectors=True,
            gpu_thoughts=True,
            gpu_lsm_batch=True,
            gpu_snn=False,
            gpu_htm=False,
        )

    @classmethod
    def cpu_only_config(cls) -> "GPUIntegrationConfig":
        """Every component on the CPU."""
        return cls(enable_gpu=False)


class GPUComponentFactory:
    """Builds the accelerated or the CPU variant of each component."""

    def __init__(self, config: Optional[GPUIntegrationConfig] = None) -> None:
        self.config = config or GPUIntegrationConfig.optimal_config()

    def create_vector_coordinator(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> VectorCoordinator:
        """Return the accelerated coordinator when configured, else the CPU one.

        Host, port and collection default to the coordinator's Settings reads.
        """
        if self.config.gpu_vectors:
            logger.info("building the accelerated vector coordinator")
            return MPSVectorCoordinator(
                host=host,
                port=port,
                collection_name=collection_name,
                use_gpu=True,
                manager=self.config.manager,
                **kwargs,
            )
        logger.info("building the CPU vector coordinator")
        return VectorCoordinator(
            host=host, port=port, collection_name=collection_name, **kwargs
        )

    def create_thought_processor(
        self,
        vector_coordinator: Optional[VectorCoordinator] = None,
        embedder: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Return the thought novelty filter, or None while its module is absent.

        The accelerated filter when thoughts are configured accelerated and the
        manager reports an accelerated device, else the numpy filter. The
        embedder is the argument, else the vector coordinator's embedder;
        ``kwargs`` go to the filter (cache size, thresholds,
        ``allow_non_semantic``).
        """
        if not _HAS_THOUGHT_PROCESSOR:
            logger.warning("thought novelty filter module not available")
            return None
        if embedder is None and vector_coordinator is not None:
            embedder = getattr(vector_coordinator, "embedder", None)
        if self.config.gpu_thoughts:
            logger.info("building the accelerated thought novelty filter")
            return AcceleratedNoveltyFilter(
                embedder, device=str(self.config.manager.device), **kwargs
            )
        logger.info("building the CPU thought novelty filter")
        return NoveltyFilter(embedder, **kwargs)

    def create_lsm(
        self, n_input: int = 10, n_reservoir: int = 1000, n_output: int = 10, **kwargs
    ) -> Any:
        """Return the torch LSM when LSM batching is accelerated, else the CPU LSM."""
        if self.config.gpu_lsm_batch:
            logger.info("building the torch liquid state machine")
            return MPSLiquidStateMachine(
                n_input=n_input,
                n_reservoir=n_reservoir,
                n_output=n_output,
                use_gpu=True,
                manager=self.config.manager,
                **kwargs,
            )
        logger.info("building the CPU liquid state machine")
        return LiquidStateMachine(
            n_input=n_input, n_reservoir=n_reservoir, n_output=n_output, **kwargs
        )


class GPUPerformanceMonitor:
    """Sums the accelerated and CPU operation counts its callers record."""

    def __init__(self, profiler: Optional[GPUProfiler] = None, manager: Any = None):
        self.profiler = profiler or global_profiler
        self.manager = manager if manager is not None else mps_manager
        self.start_time = datetime.now(timezone.utc)
        self.metrics: Dict[str, Dict[str, float]] = {
            "vector_operations": {"gpu": 0, "cpu": 0, "time_saved_ms": 0.0},
            "thought_operations": {"gpu": 0, "cpu": 0, "time_saved_ms": 0.0},
            "lsm_operations": {"gpu": 0, "cpu": 0, "time_saved_ms": 0.0},
        }

    def _record(self, kind: str, used_gpu: bool, time_saved_ms: float) -> None:
        bucket = self.metrics[kind]
        if used_gpu:
            bucket["gpu"] += 1
            bucket["time_saved_ms"] += float(time_saved_ms)
        else:
            bucket["cpu"] += 1

    def record_vector_operation(self, used_gpu: bool, time_saved_ms: float = 0.0):
        """Count a vector operation; ``time_saved_ms`` is the caller's measurement."""
        self._record("vector_operations", used_gpu, time_saved_ms)

    def record_thought_operation(self, used_gpu: bool, time_saved_ms: float = 0.0):
        """Count a thought operation; ``time_saved_ms`` is the caller's measurement."""
        self._record("thought_operations", used_gpu, time_saved_ms)

    def record_lsm_operation(self, used_gpu: bool, time_saved_ms: float = 0.0):
        """Count an LSM operation; ``time_saved_ms`` is the caller's measurement."""
        self._record("lsm_operations", used_gpu, time_saved_ms)

    def get_summary(self) -> Dict[str, Any]:
        """Return the counts, the measured savings, the pool and the profiler."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        profiler_stats: Dict[str, Any] = {}
        if self.profiler is not None and getattr(self.profiler, "enabled", False):
            profiler_stats = self.profiler.get_summary()
        gpu_ops = sum(m["gpu"] for m in self.metrics.values())
        cpu_ops = sum(m["cpu"] for m in self.metrics.values())
        saved_ms = sum(m["time_saved_ms"] for m in self.metrics.values())
        total = gpu_ops + cpu_ops
        return {
            "uptime_seconds": uptime,
            "gpu_enabled": bool(self.manager.enabled),
            "device": str(self.manager.device),
            "component_metrics": {k: dict(v) for k, v in self.metrics.items()},
            "totals": {
                "gpu_operations": gpu_ops,
                "cpu_operations": cpu_ops,
                "time_saved_ms": saved_ms,
                "time_saved_seconds": saved_ms / 1000.0,
                "gpu_percentage": gpu_ops / total * 100 if total else None,
            },
            "memory_pool": global_memory_pool.get_stats(),
            "profiler": profiler_stats,
        }

    def print_summary(self) -> None:
        """Print the summary."""
        summary = self.get_summary()
        print(f"device: {summary['device']}")
        print(f"uptime: {summary['uptime_seconds']:.1f} s")
        for name, metrics in summary["component_metrics"].items():
            print(f"{name}: gpu {metrics['gpu']}, cpu {metrics['cpu']}")
        totals = summary["totals"]
        print(
            f"totals: gpu {totals['gpu_operations']}, cpu {totals['cpu_operations']}, "
            f"measured saving {totals['time_saved_seconds']:.2f} s"
        )
        pool = summary["memory_pool"]
        print(
            f"memory pool: {pool['total_entries']} entries, {pool['size_mb']:.1f} MB, "
            f"hit rate {pool['hit_rate_percent']}"
        )


# Shared instances; none resolves the device until it is used.
gpu_config = GPUIntegrationConfig.optimal_config()
gpu_factory = GPUComponentFactory(gpu_config)
gpu_monitor = GPUPerformanceMonitor()


def create_gpu_vector_coordinator(**kwargs: Any) -> VectorCoordinator:
    """Build a vector coordinator with the shared configuration."""
    return gpu_factory.create_vector_coordinator(**kwargs)


def create_gpu_thought_processor(**kwargs: Any) -> Any:
    """Build the thought processor with the shared configuration (or None)."""
    return gpu_factory.create_thought_processor(**kwargs)


def get_gpu_performance_summary() -> Dict[str, Any]:
    """Return the shared monitor's summary."""
    return gpu_monitor.get_summary()


def print_gpu_performance() -> None:
    """Print the shared monitor's summary."""
    gpu_monitor.print_summary()
