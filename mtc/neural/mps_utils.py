"""
Device Manager for Tensor Work
==============================

``MPSManager`` is the device manager the GPU helpers share (the vector
coordinator's GPU variant, the GPU liquid state machine, the memory pool and
the profiler). Its name is historical: it handles MPS, CUDA and the CPU. The
device comes from ``mtc.core.device`` and so from ``MTC_DEVICE`` (``auto``
keeps the MPS, CUDA, CPU order; an explicit device that is missing raises
``DeviceUnavailableError`` naming the setting instead of falling back). A
``device`` argument overrides the setting and ``force_cpu`` pins the CPU.

Resolution is lazy: building a manager (including the module instance
``mps_manager``) resolves nothing, and the first use of ``device`` or
``enabled`` does, so importing this module never fails on a device setting.
``enabled`` means an accelerated device (MPS or CUDA) is active.

Dispatch: ``should_use_gpu`` sends batch operations, operations of at least
``min_gpu_size`` elements, and matrix products of at least 1000 elements to the
accelerator; everything else stays on the CPU. ``synchronize``,
``empty_cache`` and ``get_memory_stats`` call the backend of the resolved
device (MPS or CUDA) and do nothing on the CPU; memory figures that the CPU
path cannot measure are None. Timings recorded with ``record_operation`` are
measurements taken at runtime; nothing here states a speedup. Config keys:
``DEVICE`` (``MTC_DEVICE``) through ``mtc.core.device``. No persisted state.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from mtc.core.config import Settings, get_settings
from mtc.core.device import DeviceUnavailableError, resolve_device

logger = logging.getLogger(__name__)

MATMUL_MIN_SIZE = 1000
MAX_STATS = 1000
DEVICE_BACKEND = "torch"
ACCELERATED_KINDS = ("mps", "cuda")
_MB = 1024**2


class ComputeDevice(Enum):
    """Where to run computations."""

    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    AUTO = "auto"  # let the device manager decide


@dataclass
class PerformanceStats:
    """One recorded operation timing."""

    operation_name: str
    device: str
    duration_ms: float
    data_size: int
    timestamp: float


def synchronize_device(device: Any) -> None:
    """Wait for queued work on an accelerator device; no-op on the CPU."""
    kind = getattr(device, "type", str(device))
    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize()


def device_status(
    resolve: Callable[[], Any],
    last_error: Optional[str] = None,
    **details: Any,
) -> Dict[str, Any]:
    """Return the ``status()`` mapping of a device-backed object.

    ``resolve`` returns the object's device; a ``DeviceUnavailableError`` from
    it is reported as ``available`` False with the error text instead of being
    raised. ``last_error`` is the object's own most recent failure, reported
    when the device itself resolves.
    """
    try:
        device = resolve()
    except DeviceUnavailableError as exc:
        return {
            "backend": DEVICE_BACKEND,
            "available": False,
            "device": None,
            "accelerated": False,
            "last_error": str(exc),
            **details,
        }
    kind = getattr(device, "type", str(device))
    return {
        "backend": DEVICE_BACKEND,
        "available": True,
        "device": kind,
        "accelerated": kind in ACCELERATED_KINDS,
        "last_error": last_error,
        **details,
    }


class MPSManager:
    """Device manager on the ``MTC_DEVICE`` switch (the name is historical:
    it serves MPS, CUDA and the CPU)."""

    def __init__(
        self,
        force_cpu: bool = False,
        min_gpu_size: int = 10000,
        device: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.force_cpu = bool(force_cpu)
        self.min_gpu_size = int(min_gpu_size)
        self._preference = "cpu" if self.force_cpu else device
        self._settings = settings
        self._device: Optional[torch.device] = None
        self._last_error: Optional[str] = None
        self._warned = False
        self.stats: List[PerformanceStats] = []
        self.total_gpu_ops = 0
        self.total_cpu_ops = 0

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The resolved ``torch.device`` (resolved on first use)."""
        if self._device is None:
            active = self._settings if self._settings is not None else get_settings()
            try:
                self._device = resolve_device(self._preference, settings=active)
            except DeviceUnavailableError as exc:
                self._last_error = str(exc)
                if not self._warned:
                    self._warned = True
                    logger.warning("device manager has no device: %s", exc)
                raise
            self._last_error = None
            logger.info(
                "device manager on %s (accelerated: %s)",
                self._device,
                self._device.type != "cpu",
            )
            if self._device.type != "cpu":
                self._log_memory()
        return self._device

    @property
    def enabled(self) -> bool:
        """True when an accelerated device (MPS or CUDA) is active."""
        return self.device.type in ACCELERATED_KINDS

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the device.

        Never raises: an explicit device that is not available shows up as
        ``available`` False with the error naming the setting.
        """
        requested = self._preference
        if requested is None:
            active = self._settings if self._settings is not None else get_settings()
            requested = getattr(active, "DEVICE", None)
        return device_status(
            lambda: self.device,
            requested=requested,
            force_cpu=self.force_cpu,
            gpu_operations=self.total_gpu_ops,
            cpu_operations=self.total_cpu_ops,
        )

    def refresh(self) -> torch.device:
        """Resolve the device again (after a settings change)."""
        self._device = None
        return self.device

    def _log_memory(self) -> None:
        stats = self.get_memory_stats()
        if stats.get("allocated_mb") is not None:
            logger.info(
                "accelerator memory: %.1f MB allocated, %s MB held by the driver",
                stats["allocated_mb"],
                stats.get("driver_mb"),
            )

    # ------------------------------------------------------------------
    # Dispatch and transfers
    # ------------------------------------------------------------------

    def should_use_gpu(
        self, operation: str, data_size: int, is_batch: bool = False
    ) -> bool:
        """Return True when this operation should run on the accelerator."""
        if not self.enabled:
            return False
        if is_batch or data_size >= self.min_gpu_size:
            return True
        name = operation.lower()
        if "matmul" in name or "mm" in name:
            return data_size >= MATMUL_MIN_SIZE
        return False

    def to_device(
        self,
        data: Union[torch.Tensor, np.ndarray],
        force_device: Optional[str] = None,
    ) -> torch.Tensor:
        """Move data (a tensor or an array) to ``force_device`` or the device."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        target = torch.device(force_device) if force_device else self.device
        return data.to(target)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Return a tensor as a numpy array, moving it off any accelerator."""
        detached = tensor.detach() if hasattr(tensor, "detach") else tensor
        if getattr(getattr(detached, "device", None), "type", "cpu") != "cpu":
            detached = detached.cpu()
        return detached.numpy()

    def synchronize(self) -> None:
        """Wait for queued accelerator work; no-op on the CPU."""
        synchronize_device(self.device)

    def empty_cache(self) -> None:
        """Release cached accelerator memory; no-op on the CPU."""
        kind = self.device.type
        if kind == "mps":
            torch.mps.empty_cache()
        elif kind == "cuda":
            torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return allocated and driver memory in MB (None when unmeasured)."""
        kind = self.device.type
        try:
            if kind == "mps":
                allocated = torch.mps.current_allocated_memory() / _MB
                driver = torch.mps.driver_allocated_memory() / _MB
            elif kind == "cuda":
                allocated = torch.cuda.memory_allocated(self.device) / _MB
                driver = torch.cuda.memory_reserved(self.device) / _MB
            else:
                return {"device": "cpu", "allocated_mb": None, "driver_mb": None}
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.warning("could not read %s memory: %s", kind, exc)
            return {
                "device": kind,
                "allocated_mb": None,
                "driver_mb": None,
                "error": str(exc),
            }
        return {
            "device": kind,
            "allocated_mb": allocated,
            "driver_mb": driver,
            "allocated_gb": allocated / 1024,
            "driver_gb": driver / 1024,
        }

    # ------------------------------------------------------------------
    # Measured timings
    # ------------------------------------------------------------------

    def record_operation(
        self, operation_name: str, device: str, duration_ms: float, data_size: int
    ) -> None:
        """Record one measured operation (the last ``MAX_STATS`` are kept)."""
        self.stats.append(
            PerformanceStats(
                operation_name=operation_name,
                device=device,
                duration_ms=duration_ms,
                data_size=data_size,
                timestamp=time.time(),
            )
        )
        if device == "cpu":
            self.total_cpu_ops += 1
        else:
            self.total_gpu_ops += 1
        if len(self.stats) > MAX_STATS:
            self.stats = self.stats[-MAX_STATS:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Summarize the recorded timings by device kind."""
        if not self.stats:
            return {"no_data": True}
        gpu = [s for s in self.stats if s.device != "cpu"]
        cpu = [s for s in self.stats if s.device == "cpu"]
        summary: Dict[str, Any] = {
            "total_operations": len(self.stats),
            "gpu_operations": len(gpu),
            "cpu_operations": len(cpu),
            "gpu_percentage": len(gpu) / len(self.stats) * 100,
        }
        if gpu:
            summary["gpu_avg_duration_ms"] = sum(s.duration_ms for s in gpu) / len(gpu)
        if cpu:
            summary["cpu_avg_duration_ms"] = sum(s.duration_ms for s in cpu) / len(cpu)
        return summary

    def __repr__(self) -> str:
        where = str(self._device) if self._device is not None else "unresolved"
        return (
            f"MPSManager(device={where}, "
            f"ops={self.total_gpu_ops}gpu/{self.total_cpu_ops}cpu)"
        )


# The shared manager; it resolves MTC_DEVICE on first use.
mps_manager = MPSManager()


def cosine_similarity_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the (n, m) cosine similarity of row sets ``a`` and ``b``."""
    start = time.perf_counter()
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    similarity = a_norm @ b_norm.T
    synchronize_device(a.device)
    mps_manager.record_operation(
        "cosine_similarity",
        a.device.type,
        (time.perf_counter() - start) * 1000,
        a.shape[0] * b.shape[0],
    )
    return similarity


def batch_matmul_gpu(matrices: List[torch.Tensor]) -> torch.Tensor:
    """Return the elementwise product of a list of equal-shape tensors."""
    start = time.perf_counter()
    batched = torch.stack(matrices)
    result = torch.prod(batched, dim=0)
    synchronize_device(batched.device)
    mps_manager.record_operation(
        "batch_matmul",
        batched.device.type,
        (time.perf_counter() - start) * 1000,
        sum(m.numel() for m in matrices),
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"device manager: {mps_manager} on {mps_manager.device}")
    print("dispatch for 100 elements:", mps_manager.should_use_gpu("embed", 100))
    print("dispatch for a batch:", mps_manager.should_use_gpu("embed", 10, True))
    print("memory:", mps_manager.get_memory_stats())
    n, m, d = 100, 100, 384
    cpu_a, cpu_b = torch.randn(n, d), torch.randn(m, d)
    began = time.perf_counter()
    cosine_similarity_gpu(cpu_a, cpu_b)
    print(f"cpu: {(time.perf_counter() - began) * 1000:.2f} ms for {n}x{m} at {d}D")
    if mps_manager.enabled:
        dev_a, dev_b = cpu_a.to(mps_manager.device), cpu_b.to(mps_manager.device)
        began = time.perf_counter()
        cosine_similarity_gpu(dev_a, dev_b)
        mps_manager.synchronize()
        elapsed = (time.perf_counter() - began) * 1000
        print(f"{mps_manager.device}: {elapsed:.2f} ms for {n}x{m} at {d}D")
    print("summary:", mps_manager.get_performance_summary())
