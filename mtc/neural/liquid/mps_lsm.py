"""
Torch Liquid State Machine
==========================

A liquid state machine whose reservoir runs as torch tensors on the device the
device manager resolves (``mtc.neural.mps_utils``, which follows
``MTC_DEVICE``): MPS, CUDA or the CPU. The reservoir is a sparse random
recurrent matrix scaled to ``spectral_radius``, leaky-integrator neurons with
per-neuron time constants drawn from 20 to 40 ms, a tanh activation and a
linear readout trained by ridge regression.

Reproducibility: every random draw comes from ``np.random.default_rng(seed)``;
the global numpy generator is never seeded or used, so two machines built with
the same seed are identical and an unseeded one does not disturb anyone
else's draws. The spectral normalization computes the eigenvalues once.

``step`` advances one timestep; ``batch_step`` runs a sequence of inputs in
order (the reservoir carries state from step to step, so a batch is the same
computation as calling ``step`` on each row). Until ``train_readout`` succeeds
the readout weights are the small random initial ones and ``get_metrics``
reports ``readout_trained`` False; callers that read a meaning off the readout
check it. Step timings are measured at runtime. No config keys beyond the
device switch; the reservoir state is not persisted.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from mtc.neural.mps_utils import device_status, mps_manager

logger = logging.getLogger(__name__)

TAU_RANGE_MS = (20.0, 40.0)
READOUT_INIT_SCALE = 0.01


class MPSLiquidStateMachine:
    """Liquid state machine on the resolved torch device."""

    def __init__(
        self,
        n_input: int,
        n_reservoir: int = 1000,
        n_output: int = 10,
        dt: float = 1.0,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        connectivity: float = 0.1,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        manager: Any = None,
    ) -> None:
        if min(n_input, n_reservoir, n_output) < 1:
            raise ValueError("n_input, n_reservoir and n_output must be positive")
        self.n_input = int(n_input)
        self.n_reservoir = int(n_reservoir)
        self.n_output = int(n_output)
        self.dt = float(dt)
        self.spectral_radius = float(spectral_radius)
        self.input_scaling = float(input_scaling)
        self.connectivity = float(connectivity)
        self.seed = seed
        self._manager = manager if manager is not None else mps_manager
        self.use_gpu = bool(use_gpu) and bool(self._manager.enabled)
        self.device = self._manager.device if self.use_gpu else torch.device("cpu")
        self.current_time = 0.0
        self.readout_trained = False
        self._last_error: Optional[str] = None

        rng = np.random.default_rng(seed)
        self.tau = self._tensor(rng.uniform(*TAU_RANGE_MS, size=self.n_reservoir))
        self.reservoir_state = torch.zeros(self.n_reservoir, device=self.device)
        self._init_weights(rng)

        self.metrics: Dict[str, Any] = {
            "total_steps": 0,
            "gpu_steps": 0,
            "cpu_steps": 0,
            "avg_step_time_ms": None,
        }
        logger.info(
            "liquid state machine with %d neurons on %s", self.n_reservoir, self.device
        )

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), dtype=torch.float32).to(self.device)

    def _init_weights(self, rng: np.random.Generator) -> None:
        w_in = (
            rng.standard_normal((self.n_reservoir, self.n_input)) * self.input_scaling
        )
        w_res = rng.standard_normal((self.n_reservoir, self.n_reservoir))
        w_res *= rng.random((self.n_reservoir, self.n_reservoir)) < self.connectivity
        radius = float(np.max(np.abs(np.linalg.eigvals(w_res))))
        if radius > 0.0:
            w_res *= self.spectral_radius / radius
            self.achieved_spectral_radius = self.spectral_radius
        else:
            self.achieved_spectral_radius = 0.0
        w_out = rng.standard_normal((self.n_output, self.n_reservoir))
        self.W_in = self._tensor(w_in)
        self.W_res = self._tensor(w_res)
        self.W_out = self._tensor(w_out * READOUT_INIT_SCALE)

    def _input(self, signal: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(signal, torch.Tensor):
            tensor = signal.to(device=self.device, dtype=torch.float32)
        else:
            tensor = self._tensor(signal)
        if tuple(tensor.shape) != (self.n_input,):
            raise ValueError(
                f"input has shape {tuple(tensor.shape)}; expected ({self.n_input},)"
            )
        return tensor

    @torch.no_grad()
    def step(
        self, input_signal: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance one timestep; return the reservoir state and the readout."""
        started = time.perf_counter()
        drive = (
            self.W_in @ self._input(input_signal) + self.W_res @ self.reservoir_state
        )
        state = self.reservoir_state + (self.dt / self.tau) * (
            -self.reservoir_state + drive
        )
        self.reservoir_state = torch.clamp(torch.tanh(state), -1.0, 1.0)
        readout = self.W_out @ self.reservoir_state
        if self.use_gpu:
            self._manager.synchronize()
        self.current_time += self.dt
        self._count_step((time.perf_counter() - started) * 1000.0)
        return (
            self._manager.to_numpy(self.reservoir_state),
            self._manager.to_numpy(readout),
        )

    def _count_step(self, elapsed_ms: float) -> None:
        metrics = self.metrics
        metrics["total_steps"] += 1
        metrics["gpu_steps" if self.use_gpu else "cpu_steps"] += 1
        n = metrics["total_steps"]
        previous = metrics["avg_step_time_ms"] or 0.0
        metrics["avg_step_time_ms"] = previous + (elapsed_ms - previous) / n

    def batch_step(self, input_signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the rows of ``input_signals`` in order; return stacked results."""
        inputs = self._tensor(input_signals)
        if inputs.ndim != 2:
            raise ValueError("input_signals must have shape (steps, n_input)")
        results = [self.step(row) for row in inputs]
        states = np.stack([state for state, _ in results])
        readouts = np.stack([readout for _, readout in results])
        return states, readouts

    def reset_state(self) -> None:
        """Zero the reservoir and the clock."""
        self.reservoir_state = torch.zeros(self.n_reservoir, device=self.device)
        self.current_time = 0.0

    def get_state(self) -> np.ndarray:
        """Return the reservoir state."""
        return self._manager.to_numpy(self.reservoir_state)

    def set_state(self, state: np.ndarray) -> None:
        """Replace the reservoir state."""
        tensor = self._tensor(state)
        if tuple(tensor.shape) != (self.n_reservoir,):
            raise ValueError(f"state must have shape ({self.n_reservoir},)")
        self.reservoir_state = tensor

    @torch.no_grad()
    def train_readout(
        self, states: np.ndarray, targets: np.ndarray, reg: float = 0.01
    ) -> bool:
        """Fit the readout by ridge regression; return True on success.

        A failed solve is logged and leaves the previous weights in place.
        """
        s = self._tensor(states)
        t = self._tensor(targets)
        gram = s.T @ s + reg * torch.eye(self.n_reservoir, device=self.device)
        try:
            self.W_out = torch.linalg.solve(gram, s.T @ t).T.contiguous()
        except Exception as exc:  # noqa: BLE001 - reported, weights kept
            logger.error("readout training failed: %s", exc)
            self._last_error = f"readout training failed: {exc}"
            return False
        self._last_error = None
        self.readout_trained = True
        logger.info("readout trained on %d samples", int(s.shape[0]))
        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Return measured step counts and timings and the readout state."""
        total = self.metrics["total_steps"]
        return {
            **self.metrics,
            "gpu_enabled": self.use_gpu,
            "device": str(self.device),
            "gpu_percentage": (
                self.metrics["gpu_steps"] / total * 100 if total else None
            ),
            "readout_trained": self.readout_trained,
            "spectral_radius": self.achieved_spectral_radius,
        }

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the device.

        ``last_error`` is the most recent failed readout solve, if any.
        """
        return device_status(
            lambda: self.device,
            last_error=self._last_error,
            readout_trained=self.readout_trained,
            n_reservoir=self.n_reservoir,
        )

    def __repr__(self) -> str:
        return (
            f"MPSLiquidStateMachine(reservoir={self.n_reservoir}, "
            f"device={self.device}, steps={self.metrics['total_steps']})"
        )
