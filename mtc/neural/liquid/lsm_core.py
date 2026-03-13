"""
Liquid State Machine - Subconscious Processor
===========================================================

This module implements a Liquid State Machine (LSM) for
subconscious processing. The LSM acts as a temporal pattern
transformer, creating intuition and implicit understanding.

Note: Where conscious thoughts meet unconscious waters,
and intuition emerges from chaos.

LSM Principles:
- Reservoir: Pool of randomly connected neurons (1000+)
- Fading memory: Recent inputs influence current state
- Chaotic dynamics: Complex temporal transformations
- Readout learning: Extract patterns from reservoir states
- No reservoir training: Only readout learns
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class LeakyIntegratorNeuron:
    """
    Simplified neuron for LSM reservoir.
    Simpler than LIF - just leaky integration without explicit spikes.
    """

    neuron_id: int
    tau: float = 30.0  # Time constant (ms)
    activation: float = 0.0  # Current activation (-1 to 1)

    # Activation function bounds
    activation_min: float = -1.0
    activation_max: float = 1.0

    def step(self, dt: float, input_current: float) -> float:
        """
        Leaky integration: x(t+1) = x(t) + dt/tau * (-x(t) + input)

        Args:
            dt: Timestep in milliseconds
            input_current: Input from other neurons + external input

        Returns:
            Current activation level
        """
        # Leaky integration
        dx = (dt / self.tau) * (-self.activation + input_current)
        self.activation += dx

        # Apply tanh activation function for bounded output
        self.activation = np.tanh(self.activation)

        # Clip to bounds (safety)
        self.activation = np.clip(
            self.activation, self.activation_min, self.activation_max
        )

        return self.activation


class LiquidStateMachine:
    """
    Complete Liquid State Machine for subconscious processing.

    Architecture:
    - Input layer: Projects external input into reservoir
    - Reservoir: Pool of recurrently connected neurons
    - Readout layer: Learns to extract patterns from reservoir
    """

    def __init__(
        self,
        n_input: int = 10,
        n_reservoir: int = 1000,
        n_output: int = 10,
        dt: float = 1.0,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.5,
        connectivity: float = 0.1,
    ):
        """
        Initialize the Liquid State Machine.

        Args:
            n_input: Number of input dimensions
            n_reservoir: Size of reservoir (more = more memory)
            n_output: Number of output dimensions
            dt: Simulation timestep in milliseconds
            spectral_radius: Controls chaos/stability (< 1 = stable)
            input_scaling: Scale of input weights
            connectivity: Fraction of possible connections (sparsity)
        """
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output
        self.dt = dt
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.connectivity = connectivity

        self.current_time = 0.0

        # Create reservoir neurons
        self.reservoir = [
            LeakyIntegratorNeuron(
                neuron_id=i, tau=np.random.uniform(20.0, 40.0)  # Varied time constants
            )
            for i in range(n_reservoir)
        ]

        # Initialize connection matrices
        self._init_weights()

        # State history for analysis
        self.state_history: List[Tuple[float, np.ndarray]] = []

        # Performance metrics
        self.metrics = {
            "total_steps": 0,
            "reservoir_energy": 0.0,  # Sum of squared activations
            "max_activation": 0.0,
            "min_activation": 0.0,
        }

        logger.info(
            f"LSM initialized: {n_reservoir} reservoir neurons, "
            f"connectivity={connectivity:.1%}, spectral_radius={spectral_radius}"
        )

    def _init_weights(self):
        """Initialize connection weight matrices"""

        # Input -> Reservoir weights (random, scaled)
        self.W_in = np.random.randn(self.n_reservoir, self.n_input)
        self.W_in *= self.input_scaling

        # Reservoir -> Reservoir weights (sparse, random, scaled)
        # Create sparse random matrix
        self.W_res = np.random.randn(self.n_reservoir, self.n_reservoir)

        # Apply sparsity mask
        mask = np.random.rand(self.n_reservoir, self.n_reservoir) < self.connectivity
        self.W_res *= mask

        # Scale by spectral radius for stability
        eigenvalues = np.linalg.eigvals(self.W_res)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            self.W_res *= self.spectral_radius / current_radius

        # Reservoir -> Output weights (learned via simple linear regression)
        # Initialize small random weights
        self.W_out = np.random.randn(self.n_output, self.n_reservoir) * 0.01

        logger.info(
            f"Initialized weights: spectral radius achieved = "
            f"{np.max(np.abs(np.linalg.eigvals(self.W_res))):.3f}"
        )

    def step(self, input_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate one timestep of the LSM.

        Args:
            input_signal: Input vector (n_input,)

        Returns:
            Tuple of (reservoir_state, readout_output)
        """
        self.current_time += self.dt
        self.metrics["total_steps"] += 1

        # Get current reservoir state
        reservoir_state = np.array([n.activation for n in self.reservoir])

        # Calculate input to each reservoir neuron
        # Input: W_in @ input + W_res @ reservoir_state
        input_projection = self.W_in @ input_signal
        recurrent_input = self.W_res @ reservoir_state
        total_input = input_projection + recurrent_input

        # Update each neuron
        for i, neuron in enumerate(self.reservoir):
            neuron.step(self.dt, total_input[i])

        # Get new reservoir state
        new_reservoir_state = np.array([n.activation for n in self.reservoir])

        # Calculate readout (linear combination of reservoir states)
        readout = self.W_out @ new_reservoir_state
        readout = np.tanh(readout)  # Bounded output

        # Update metrics
        energy = np.sum(new_reservoir_state**2)
        self.metrics["reservoir_energy"] = energy
        self.metrics["max_activation"] = np.max(new_reservoir_state)
        self.metrics["min_activation"] = np.min(new_reservoir_state)

        # Store state history (keep last 1000 states)
        self.state_history.append((self.current_time, new_reservoir_state.copy()))
        if len(self.state_history) > 1000:
            self.state_history.pop(0)

        return new_reservoir_state, readout

    def train_readout(
        self,
        input_sequences: List[np.ndarray],
        target_sequences: List[np.ndarray],
        washout_steps: int = 100,
    ):
        """
        Train the readout layer using ridge regression.

        Args:
            input_sequences: List of input sequences
            target_sequences: List of corresponding target sequences
            washout_steps: Steps to discard at start (let reservoir stabilize)
        """
        logger.info("Training LSM readout layer...")

        # Collect reservoir states for all sequences
        all_states = []
        all_targets = []

        for input_seq, target_seq in zip(input_sequences, target_sequences):
            # Reset reservoir
            self.reset()

            # Run sequence
            for t, (inp, target) in enumerate(zip(input_seq, target_seq)):
                state, _ = self.step(inp)

                # Skip washout period
                if t >= washout_steps:
                    all_states.append(state)
                    all_targets.append(target)

        # Convert to matrices
        X = np.array(all_states)  # (n_samples, n_reservoir)
        Y = np.array(all_targets)  # (n_samples, n_output)

        # Ridge regression: W = Y^T @ X @ (X^T @ X + lambda*I)^-1
        ridge_lambda = 1e-6
        XTX = X.T @ X
        XTX_reg = XTX + ridge_lambda * np.eye(self.n_reservoir)

        try:
            self.W_out = Y.T @ X @ np.linalg.inv(XTX_reg)
            logger.info(f"Readout trained on {len(all_states)} samples")
        except np.linalg.LinAlgError:
            logger.warning("Ridge regression failed, keeping random weights")

    def get_reservoir_state(self) -> np.ndarray:
        """Get current reservoir activation state"""
        return np.array([n.activation for n in self.reservoir])

    def get_state_complexity(self) -> Dict[str, float]:
        """
        Measure complexity of reservoir state.

        Returns:
            Dictionary with complexity metrics
        """
        state = self.get_reservoir_state()

        # Various complexity measures
        metrics = {
            "energy": float(np.sum(state**2)),  # Total energy
            "entropy": float(
                -np.sum(state * np.log(np.abs(state) + 1e-10))
            ),  # Shannon entropy (approx)
            "sparsity": float(
                np.sum(np.abs(state) < 0.1) / len(state)
            ),  # Fraction of near-zero
            "max_activation": float(np.max(state)),
            "min_activation": float(np.min(state)),
            "mean_activation": float(np.mean(state)),
            "std_activation": float(np.std(state)),
        }

        return metrics

    def get_separation_property(
        self, input1: np.ndarray, input2: np.ndarray, steps: int = 100
    ) -> float:
        """
        Test separation property: different inputs -> different states.

        Args:
            input1, input2: Two different input patterns
            steps: Number of steps to run

        Returns:
            Euclidean distance between final reservoir states
        """
        # Run with input 1
        self.reset()
        for _ in range(steps):
            state1, _ = self.step(input1)

        # Run with input 2
        self.reset()
        for _ in range(steps):
            state2, _ = self.step(input2)

        # Calculate distance
        distance = float(np.linalg.norm(state1 - state2))
        return distance

    def reset(self):
        """Reset reservoir to zero state"""
        for neuron in self.reservoir:
            neuron.activation = 0.0
        self.current_time = 0.0
        self.state_history.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and state metrics"""
        complexity = self.get_state_complexity()

        return {
            **self.metrics,
            "current_time_ms": self.current_time,
            "reservoir_size": self.n_reservoir,
            "connectivity": self.connectivity,
            "spectral_radius": self.spectral_radius,
            **complexity,
        }


# Testing and demonstration
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    print("Testing Liquid State Machine...")

    # Create LSM
    lsm = LiquidStateMachine(
        n_input=10,
        n_reservoir=1000,
        n_output=10,
        dt=1.0,
        spectral_radius=0.9,
        connectivity=0.1,
    )

    print("\nLSM Configuration:")
    metrics = lsm.get_metrics()
    print(f"  Reservoir size: {metrics['reservoir_size']} neurons")
    print(f"  Connectivity: {metrics['connectivity']:.1%}")
    print(f"  Spectral radius: {metrics['spectral_radius']:.3f}")

    print("\nTesting reservoir dynamics...")

    # Create test input pattern
    input_pattern = np.random.randn(10) * 0.5

    # Run for 200 steps
    print("  Running 200 timesteps...")
    for step_idx in range(200):
        state, output = lsm.step(input_pattern)

        if step_idx % 50 == 0:
            complexity = lsm.get_state_complexity()
            print(
                f"    t={lsm.current_time:.0f}ms: "
                f"energy={complexity['energy']:.2f}, "
                f"mean={complexity['mean_activation']:.3f}"
            )

    print("\nFinal State Complexity:")
    complexity = lsm.get_state_complexity()
    print(f"  Energy: {complexity['energy']:.2f}")
    print(f"  Entropy: {complexity['entropy']:.2f}")
    print(f"  Sparsity: {complexity['sparsity']:.1%}")
    print(f"  Max activation: {complexity['max_activation']:.3f}")
    print(f"  Min activation: {complexity['min_activation']:.3f}")

    print("\nTesting separation property...")
    # Test that different inputs create different states
    input_a = np.random.randn(10) * 0.3
    input_b = np.random.randn(10) * 0.3

    distance = lsm.get_separation_property(input_a, input_b, steps=50)
    print(f"  State distance for different inputs: {distance:.3f}")
    print(f"  Separation: {'Good' if distance > 5.0 else 'Weak'}")

    print("\nLiquid State Machine ready!")
