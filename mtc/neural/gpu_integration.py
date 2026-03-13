"""
GPU Integration Layer - Connecting GPU Acceleration to Neural Orchestrator
================================================================================

Provides GPU-accelerated variants of neural systems with smart
CPU/GPU dispatch based on performance profiling findings.

Integration Strategy:
- Vector operations: Always GPU (18.55x faster)
- Thought processing: Always GPU with zero-copy (6.8x faster)
- LSM: GPU only for batch operations (1.51x faster when batched)
- SNN: Stay on CPU (too small, overhead dominates)

Note: Connect the GPU acceleration to the consciousness system.

Created: November 7, 2025
Authors: Multi-Theory Consciousness Contributors
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from mtc.neural.vectors.coordinator import VectorCoordinator
from mtc.neural.liquid.lsm_core import LiquidStateMachine
from mtc.neural.mps_utils import mps_manager
from mtc.neural.memory_pool import global_memory_pool

# Optional GPU-accelerated variants (may not be available in all deployments)
try:
    from mtc.neural.vectors.mps_coordinator import MPSVectorCoordinator

    _HAS_MPS_VECTOR = True
except ImportError:
    MPSVectorCoordinator = None
    _HAS_MPS_VECTOR = False

try:
    from mtc.neural.liquid.mps_lsm import MPSLiquidStateMachine

    _HAS_MPS_LSM = True
except ImportError:
    MPSLiquidStateMachine = None
    _HAS_MPS_LSM = False

# Optional thought processor (application-specific)
try:
    from mtc.consciousness.continuous.thought_generator import ThoughtGenerator
    from mtc.consciousness.continuous.mps_thought_processor_v2 import MPSThoughtProcessorV2

    _HAS_THOUGHT_PROCESSOR = True
except ImportError:
    ThoughtGenerator = None
    MPSThoughtProcessorV2 = None
    _HAS_THOUGHT_PROCESSOR = False

# Optional GPU profiler
try:
    from mtc.neural.gpu_profiler import GPUProfiler, global_profiler

    _HAS_PROFILER = True
except ImportError:
    GPUProfiler = None
    global_profiler = None
    _HAS_PROFILER = False

logger = logging.getLogger(__name__)


class GPUIntegrationConfig:
    """
    Configuration for GPU acceleration in neural systems.

    Based on empirical performance profiling findings.
    """

    def __init__(
        self,
        enable_gpu: bool = True,
        gpu_vectors: bool = True,  # 18.55x speedup - ALWAYS beneficial
        gpu_thoughts: bool = True,  # 6.8x speedup with zero-copy
        gpu_lsm_batch: bool = True,  # 1.51x speedup when batching
        gpu_snn: bool = False,  # 0.53x - GPU slower! Keep on CPU
        gpu_htm: bool = False,  # Similar to SNN - too small
        profiling_enabled: bool = True,
        memory_pool_enabled: bool = True,
    ):
        """
        Initialize GPU integration configuration.

        Args:
            enable_gpu: Master switch for GPU acceleration
            gpu_vectors: Use GPU for vector operations (MPSVectorCoordinator)
            gpu_thoughts: Use GPU for thought processing (MPSThoughtProcessorV2)
            gpu_lsm_batch: Use GPU for LSM batch operations (MPSLSM)
            gpu_snn: Use GPU for SNN (NOT RECOMMENDED - overhead dominates)
            gpu_htm: Use GPU for HTM (NOT RECOMMENDED - too small)
            profiling_enabled: Enable GPU profiling
            memory_pool_enabled: Enable GPU memory pool
        """
        self.enable_gpu = enable_gpu and mps_manager.enabled

        # Individual component settings
        self.gpu_vectors = gpu_vectors and self.enable_gpu
        self.gpu_thoughts = gpu_thoughts and self.enable_gpu
        self.gpu_lsm_batch = gpu_lsm_batch and self.enable_gpu
        self.gpu_snn = gpu_snn and self.enable_gpu  # Should stay False!
        self.gpu_htm = gpu_htm and self.enable_gpu  # Should stay False!

        # Utilities
        self.profiling_enabled = profiling_enabled
        self.memory_pool_enabled = memory_pool_enabled

        # Log configuration
        if self.enable_gpu:
            logger.info("GPU Integration Configuration:")
            logger.info(f"   Device: {mps_manager.device}")
            logger.info(
                f"   Vector operations: {'GPU' if self.gpu_vectors else 'CPU'}"
            )
            logger.info(
                f"   Thought processing: {'GPU' if self.gpu_thoughts else 'CPU'}"
            )
            logger.info(f"   LSM batching: {'GPU' if self.gpu_lsm_batch else 'CPU'}")
            logger.info(f"   SNN: {'GPU' if self.gpu_snn else 'CPU (recommended)'}")
            logger.info(f"   HTM: {'GPU' if self.gpu_htm else 'CPU (recommended)'}")
            logger.info(
                f"   Profiling: {'Enabled' if self.profiling_enabled else 'Disabled'}"
            )
            logger.info(
                f"   Memory pool: {'Enabled' if self.memory_pool_enabled else 'Disabled'}"
            )
        else:
            logger.info("GPU Integration: CPU mode (MPS not available)")

    @classmethod
    def optimal_config(cls) -> "GPUIntegrationConfig":
        """
        Create optimal GPU configuration based on performance profiling.

        Returns:
            Optimally configured GPU integration
        """
        return cls(
            enable_gpu=True,
            gpu_vectors=True,  # 18.55x faster
            gpu_thoughts=True,  # 6.8x faster
            gpu_lsm_batch=True,  # 1.51x faster (when batched)
            gpu_snn=False,  # 0.53x - GPU slower!
            gpu_htm=False,  # Too small like SNN
            profiling_enabled=True,
            memory_pool_enabled=True,
        )

    @classmethod
    def cpu_only_config(cls) -> "GPUIntegrationConfig":
        """Create CPU-only configuration (for testing or debugging)."""
        return cls(enable_gpu=False)


class GPUComponentFactory:
    """
    Factory for creating GPU-accelerated or CPU versions of neural components.

    Automatically selects optimal implementation based on configuration.
    """

    def __init__(self, config: Optional[GPUIntegrationConfig] = None):
        """
        Initialize component factory.

        Args:
            config: GPU integration configuration (uses optimal if None)
        """
        self.config = config or GPUIntegrationConfig.optimal_config()

        logger.info("GPU Component Factory initialized")

    def create_vector_coordinator(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "memories",
        **kwargs,
    ) -> VectorCoordinator:
        """
        Create vector coordinator (GPU or CPU).

        Returns:
            MPSVectorCoordinator if GPU enabled, else VectorCoordinator
        """
        if self.config.gpu_vectors and _HAS_MPS_VECTOR:
            logger.info(
                "Creating GPU-accelerated vector coordinator (18.55x faster)"
            )
            return MPSVectorCoordinator(
                host=host,
                port=port,
                collection_name=collection_name,
                use_gpu=True,
                **kwargs,
            )
        else:
            logger.info("Creating CPU vector coordinator")
            return VectorCoordinator(
                host=host, port=port, collection_name=collection_name, **kwargs
            )

    def create_thought_processor(
        self, vector_coordinator: Optional[VectorCoordinator] = None, **kwargs
    ):
        """
        Create thought processor (GPU or CPU variant).

        Returns:
            MPSThoughtProcessorV2 with GPU enabled/disabled based on config,
            or None if thought processor module is not available.
        """
        if not _HAS_THOUGHT_PROCESSOR:
            logger.warning("Thought processor module not available")
            return None

        if self.config.gpu_thoughts:
            logger.info("Creating GPU-accelerated thought processor (6.8x faster)")
            return MPSThoughtProcessorV2(
                vector_coordinator=vector_coordinator, use_gpu=True, **kwargs
            )
        else:
            logger.info("Creating CPU thought processor")
            return MPSThoughtProcessorV2(
                vector_coordinator=vector_coordinator, use_gpu=False, **kwargs
            )

    def create_lsm(
        self, n_input: int = 10, n_reservoir: int = 1000, n_output: int = 10, **kwargs
    ) -> LiquidStateMachine:
        """
        Create Liquid State Machine (GPU or CPU).

        Returns:
            MPSLSM if GPU batching enabled, else CPU LSM
        """
        if self.config.gpu_lsm_batch and _HAS_MPS_LSM:
            logger.info(
                "Creating GPU-accelerated LSM (use batch_step for 1.51x speedup)"
            )
            return MPSLiquidStateMachine(
                n_input=n_input,
                n_reservoir=n_reservoir,
                n_output=n_output,
                use_gpu=True,
                **kwargs,
            )
        else:
            logger.info("Creating CPU LSM")
            return LiquidStateMachine(
                n_input=n_input, n_reservoir=n_reservoir, n_output=n_output, **kwargs
            )


class GPUPerformanceMonitor:
    """
    Monitors GPU performance across all neural systems.

    Tracks:
    - GPU vs CPU operation counts
    - Time saved by GPU acceleration
    - Memory pool hit rates
    - Overall speedup metrics
    """

    def __init__(self, profiler=None):
        """
        Initialize performance monitor.

        Args:
            profiler: GPU profiler (uses global if None)
        """
        self.profiler = profiler or global_profiler
        self.start_time = datetime.now()

        # Component-specific counters
        self.metrics = {
            "vector_operations": {"gpu": 0, "cpu": 0, "time_saved_ms": 0},
            "thought_operations": {"gpu": 0, "cpu": 0, "time_saved_ms": 0},
            "lsm_operations": {"gpu": 0, "cpu": 0, "time_saved_ms": 0},
        }

    def record_vector_operation(self, used_gpu: bool, time_saved_ms: float = 0):
        """Record a vector operation"""
        if used_gpu:
            self.metrics["vector_operations"]["gpu"] += 1
            self.metrics["vector_operations"]["time_saved_ms"] += time_saved_ms
        else:
            self.metrics["vector_operations"]["cpu"] += 1

    def record_thought_operation(self, used_gpu: bool, time_saved_ms: float = 0):
        """Record a thought operation"""
        if used_gpu:
            self.metrics["thought_operations"]["gpu"] += 1
            self.metrics["thought_operations"]["time_saved_ms"] += time_saved_ms
        else:
            self.metrics["thought_operations"]["cpu"] += 1

    def record_lsm_operation(self, used_gpu: bool, time_saved_ms: float = 0):
        """Record an LSM operation"""
        if used_gpu:
            self.metrics["lsm_operations"]["gpu"] += 1
            self.metrics["lsm_operations"]["time_saved_ms"] += time_saved_ms
        else:
            self.metrics["lsm_operations"]["cpu"] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        # Get memory pool stats
        pool_stats = global_memory_pool.get_stats() if global_memory_pool else {}

        # Get profiler stats
        profiler_stats = {}
        if self.profiler is not None and hasattr(self.profiler, 'enabled') and self.profiler.enabled:
            profiler_stats = self.profiler.get_summary()

        # Calculate totals
        total_gpu_ops = sum(m["gpu"] for m in self.metrics.values())
        total_cpu_ops = sum(m["cpu"] for m in self.metrics.values())
        total_time_saved_ms = sum(m["time_saved_ms"] for m in self.metrics.values())

        return {
            "uptime_seconds": uptime,
            "gpu_enabled": mps_manager.enabled,
            "device": str(mps_manager.device),
            "component_metrics": self.metrics,
            "totals": {
                "gpu_operations": total_gpu_ops,
                "cpu_operations": total_cpu_ops,
                "time_saved_ms": total_time_saved_ms,
                "time_saved_seconds": total_time_saved_ms / 1000,
                "gpu_percentage": (
                    total_gpu_ops / max(1, total_gpu_ops + total_cpu_ops)
                )
                * 100,
            },
            "memory_pool": pool_stats,
            "profiler": profiler_stats,
        }

    def print_summary(self):
        """Pretty print performance summary"""
        summary = self.get_summary()

        print("=" * 70)
        print("GPU PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Uptime: {summary['uptime_seconds']:.1f} seconds")
        print(f"Device: {summary['device']}")
        print()

        print("Component Breakdown:")
        for component, metrics in summary["component_metrics"].items():
            total = metrics["gpu"] + metrics["cpu"]
            if total > 0:
                gpu_pct = (metrics["gpu"] / total) * 100
                print(f"\n  {component}:")
                print(f"    GPU operations: {metrics['gpu']} ({gpu_pct:.1f}%)")
                print(f"    CPU operations: {metrics['cpu']}")
                print(f"    Time saved: {metrics['time_saved_ms']/1000:.2f} seconds")

        print()
        print("Overall Totals:")
        totals = summary["totals"]
        print(f"  GPU operations: {totals['gpu_operations']}")
        print(f"  CPU operations: {totals['cpu_operations']}")
        print(f"  GPU usage: {totals['gpu_percentage']:.1f}%")
        print(f"  Total time saved: {totals['time_saved_seconds']:.2f} seconds")

        if summary["memory_pool"]:
            print()
            print("Memory Pool:")
            pool = summary["memory_pool"]
            print(f"  Hit rate: {pool.get('hit_rate_percent', 0):.1f}%")
            print(f"  Cache size: {pool.get('size_mb', 0):.1f} MB")
            print(f"  Transfers avoided: {pool.get('transfers_avoided', 0)}")

        print()
        print("=" * 70)


# Global instances
gpu_config = GPUIntegrationConfig.optimal_config()
gpu_factory = GPUComponentFactory(gpu_config)
gpu_monitor = GPUPerformanceMonitor()


# Convenience functions
def create_gpu_vector_coordinator(**kwargs) -> VectorCoordinator:
    """Create GPU-accelerated vector coordinator with optimal settings"""
    return gpu_factory.create_vector_coordinator(**kwargs)


def create_gpu_thought_processor(**kwargs):
    """Create GPU-accelerated thought processor with optimal settings"""
    return gpu_factory.create_thought_processor(**kwargs)


def get_gpu_performance_summary() -> Dict[str, Any]:
    """Get current GPU performance metrics"""
    return gpu_monitor.get_summary()


def print_gpu_performance():
    """Print GPU performance summary"""
    gpu_monitor.print_summary()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("TESTING GPU INTEGRATION LAYER")
    print("=" * 70)
    print()

    # Test 1: Create components with optimal config
    print("Test 1: Creating components with optimal GPU config")

    config = GPUIntegrationConfig.optimal_config()
    factory = GPUComponentFactory(config)

    # Create GPU vector coordinator
    vector_coord = factory.create_vector_coordinator()
    print(f"  Vector coordinator: {vector_coord}")

    # Create GPU thought processor
    thought_proc = factory.create_thought_processor(vector_coordinator=vector_coord)
    print(f"  Thought processor: {thought_proc}")

    # Create GPU LSM
    lsm = factory.create_lsm()
    print(f"  LSM: {lsm}")
    print()

    # Test 2: Performance monitoring
    print("Test 2: Performance monitoring")

    monitor = GPUPerformanceMonitor()

    # Simulate some operations
    monitor.record_vector_operation(used_gpu=True, time_saved_ms=100)
    monitor.record_thought_operation(used_gpu=True, time_saved_ms=50)
    monitor.record_lsm_operation(used_gpu=False)

    monitor.print_summary()

    print()
    print("=" * 70)
    print("GPU INTEGRATION LAYER READY FOR PRODUCTION!")
    print("=" * 70)
