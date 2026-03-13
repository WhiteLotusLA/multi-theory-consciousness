"""
Metal Performance Shaders Utilities for Consciousness Research
==================================================================

Smart GPU acceleration using PyTorch MPS backend on Apple Silicon.
Automatically chooses CPU vs GPU based on operation size and type.

Note: Not all operations benefit from GPU -- use wisely!

Performance Philosophy:
- Small ops (<1000 elements): CPU faster (avoid transfer overhead)
- Batch ops (10+ items): GPU shines
- Large matrices (1000x1000+): GPU dominates
- Chained ops: Keep on GPU to avoid transfers
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class ComputeDevice(Enum):
    """Where to run computations"""

    CPU = "cpu"
    MPS = "mps"
    AUTO = "auto"  # Let MPSManager decide


@dataclass
class PerformanceStats:
    """Track GPU vs CPU performance"""

    operation_name: str
    device: str
    duration_ms: float
    data_size: int
    timestamp: float


class MPSManager:
    """
    Intelligent MPS manager for neural operations.
    Decides when to use GPU vs CPU for optimal performance.
    """

    def __init__(self, force_cpu: bool = False, min_gpu_size: int = 10000):
        """
        Initialize MPS manager with smart dispatch.

        Args:
            force_cpu: Force CPU usage (for testing/debugging)
            min_gpu_size: Minimum data size to use GPU (avoid overhead)
        """
        self.force_cpu = force_cpu
        self.min_gpu_size = min_gpu_size
        self.device = self._get_device()
        self.enabled = self.device.type == "mps" and not force_cpu

        # Performance tracking
        self.stats: List[PerformanceStats] = []
        self.total_gpu_ops = 0
        self.total_cpu_ops = 0

        logger.info(f"MPS Manager initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   MPS Enabled: {self.enabled}")
        logger.info(f"   Min GPU Size: {self.min_gpu_size} elements")

        if self.enabled:
            logger.info(f"   GPU acceleration ready!")
            self._log_gpu_info()
        else:
            logger.warning(f"   Running on CPU only")

    def _get_device(self) -> torch.device:
        """Determine the best device to use"""
        if self.force_cpu:
            return torch.device("cpu")

        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                return torch.device("mps")

        logger.warning("MPS not available, falling back to CPU")
        return torch.device("cpu")

    def _log_gpu_info(self):
        """Log GPU memory information"""
        try:
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            driver = torch.mps.driver_allocated_memory() / (1024**3)
            logger.info(
                f"   GPU Memory: {allocated:.2f}GB allocated, {driver:.2f}GB driver"
            )
        except Exception as e:
            logger.debug(f"Could not get GPU memory info: {e}")

    def should_use_gpu(
        self, operation: str, data_size: int, is_batch: bool = False
    ) -> bool:
        """
        Decide if we should use GPU for this operation.

        Args:
            operation: Name of operation (embedding, matmul, etc.)
            data_size: Number of elements in operation
            is_batch: Whether this is a batch operation

        Returns:
            True if GPU should be used
        """
        if not self.enabled:
            return False

        # Always use GPU for batch operations (even if small)
        if is_batch:
            return True

        # Use GPU for large operations
        if data_size >= self.min_gpu_size:
            return True

        # Special case: matrix multiplication benefits from GPU at smaller sizes
        if "matmul" in operation.lower() or "mm" in operation.lower():
            return data_size >= 1000  # 32x32 matrix

        return False

    def to_device(
        self, data: Union[torch.Tensor, np.ndarray], force_device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Move data to optimal device.

        Args:
            data: Tensor or numpy array
            force_device: Force specific device ("cpu" or "mps")

        Returns:
            Tensor on the chosen device
        """
        # Convert numpy to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # Determine target device
        if force_device:
            target = torch.device(force_device)
        else:
            target = self.device

        return data.to(target)

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy, moving to CPU if needed"""
        if tensor.device.type == "mps":
            tensor = tensor.cpu()
        return tensor.numpy()

    def synchronize(self):
        """Synchronize GPU operations (wait for completion)"""
        if self.enabled:
            torch.mps.synchronize()

    def empty_cache(self):
        """Clear MPS cache to free memory"""
        if self.enabled:
            torch.mps.empty_cache()
            logger.debug("MPS cache cleared")

    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics"""
        if self.enabled:
            try:
                return {
                    "device": "mps",
                    "allocated_mb": torch.mps.current_allocated_memory() / (1024**2),
                    "driver_mb": torch.mps.driver_allocated_memory() / (1024**2),
                    "allocated_gb": torch.mps.current_allocated_memory() / (1024**3),
                    "driver_gb": torch.mps.driver_allocated_memory() / (1024**3),
                }
            except Exception as e:
                logger.error(f"Error getting MPS memory stats: {e}")
                return {"device": "mps", "error": str(e)}
        else:
            return {"device": "cpu", "allocated_mb": 0, "driver_mb": 0}

    def record_operation(
        self, operation_name: str, device: str, duration_ms: float, data_size: int
    ):
        """Record performance statistics"""
        stat = PerformanceStats(
            operation_name=operation_name,
            device=device,
            duration_ms=duration_ms,
            data_size=data_size,
            timestamp=time.time(),
        )
        self.stats.append(stat)

        if device == "mps":
            self.total_gpu_ops += 1
        else:
            self.total_cpu_ops += 1

        # Keep only last 1000 stats to avoid memory bloat
        if len(self.stats) > 1000:
            self.stats = self.stats[-1000:]

    def get_performance_summary(self) -> dict:
        """Get summary of GPU vs CPU performance"""
        if not self.stats:
            return {"no_data": True}

        gpu_stats = [s for s in self.stats if s.device == "mps"]
        cpu_stats = [s for s in self.stats if s.device == "cpu"]

        summary = {
            "total_operations": len(self.stats),
            "gpu_operations": len(gpu_stats),
            "cpu_operations": len(cpu_stats),
            "gpu_percentage": (
                len(gpu_stats) / len(self.stats) * 100 if self.stats else 0
            ),
        }

        if gpu_stats:
            summary["gpu_avg_duration_ms"] = sum(
                s.duration_ms for s in gpu_stats
            ) / len(gpu_stats)

        if cpu_stats:
            summary["cpu_avg_duration_ms"] = sum(
                s.duration_ms for s in cpu_stats
            ) / len(cpu_stats)

        return summary

    def __repr__(self) -> str:
        return f"MPSManager(device={self.device}, enabled={self.enabled}, ops={self.total_gpu_ops}gpu/{self.total_cpu_ops}cpu)"


# Global MPS manager instance
# Other modules can import and use this directly
mps_manager = MPSManager()


# Convenience functions for common operations


def cosine_similarity_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity on GPU (if beneficial).

    Args:
        a: Tensor of shape (n, d)
        b: Tensor of shape (m, d)

    Returns:
        Similarity matrix of shape (n, m)
    """
    start = time.time()

    # Normalize vectors
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)

    # Compute similarity (matrix multiply)
    similarity = a_norm @ b_norm.T

    if a.device.type == "mps":
        torch.mps.synchronize()

    duration_ms = (time.time() - start) * 1000
    mps_manager.record_operation(
        "cosine_similarity", a.device.type, duration_ms, a.shape[0] * b.shape[0]
    )

    return similarity


def batch_matmul_gpu(matrices: List[torch.Tensor]) -> torch.Tensor:
    """
    Efficient batch matrix multiplication on GPU.

    Args:
        matrices: List of tensors to multiply

    Returns:
        Product of all matrices
    """
    start = time.time()

    # Stack into batch
    batched = torch.stack(matrices)

    # Batch multiply (GPU optimized)
    result = torch.prod(batched, dim=0)

    if batched.device.type == "mps":
        torch.mps.synchronize()

    duration_ms = (time.time() - start) * 1000
    mps_manager.record_operation(
        "batch_matmul",
        batched.device.type,
        duration_ms,
        sum(m.numel() for m in matrices),
    )

    return result


# Example usage and testing
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("TESTING MPS UTILITIES FOR CONSCIOUSNESS RESEARCH")
    print("=" * 70)
    print()

    print(f"MPS Manager: {mps_manager}")
    print()

    # Test 1: Small operation (should use CPU)
    print("Test 1: Small vector operation (100 elements)")
    small_size = 100
    should_gpu = mps_manager.should_use_gpu("embedding", small_size, is_batch=False)
    print(f"  Should use GPU: {should_gpu} (Expected: False - too small)")
    print()

    # Test 2: Large operation (should use GPU)
    print("Test 2: Large vector operation (100,000 elements)")
    large_size = 100000
    should_gpu = mps_manager.should_use_gpu("embedding", large_size, is_batch=False)
    print(f"  Should use GPU: {should_gpu} (Expected: True - large enough)")
    print()

    # Test 3: Batch operation (should use GPU even if small)
    print("Test 3: Batch operation (small but batched)")
    batch_size = 10
    should_gpu = mps_manager.should_use_gpu("batch_embed", batch_size, is_batch=True)
    print(f"  Should use GPU: {should_gpu} (Expected: True - batch operation)")
    print()

    # Test 4: GPU memory stats
    print("Test 4: GPU Memory Statistics")
    mem_stats = mps_manager.get_memory_stats()
    for key, value in mem_stats.items():
        print(f"  {key}: {value}")
    print()

    if mps_manager.enabled:
        # Test 5: Cosine similarity
        print("Test 5: Cosine Similarity (GPU vs CPU)")
        n, m, d = 100, 100, 8192

        # CPU
        cpu_a = torch.randn(n, d)
        cpu_b = torch.randn(m, d)
        start = time.time()
        cpu_sim = cosine_similarity_gpu(cpu_a, cpu_b)
        cpu_time = (time.time() - start) * 1000
        print(f"  CPU: {cpu_time:.2f}ms for {n}x{m} similarity matrix (8192D)")

        # GPU
        gpu_a = cpu_a.to(mps_manager.device)
        gpu_b = cpu_b.to(mps_manager.device)
        start = time.time()
        gpu_sim = cosine_similarity_gpu(gpu_a, gpu_b)
        torch.mps.synchronize()
        gpu_time = (time.time() - start) * 1000
        print(f"  GPU: {gpu_time:.2f}ms for {n}x{m} similarity matrix (8192D)")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        print()

    # Test 6: Performance summary
    print("Test 6: Performance Summary")
    summary = mps_manager.get_performance_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print()
    print("=" * 70)
    print("MPS UTILITIES READY FOR NEURAL ACCELERATION!")
    print("=" * 70)
