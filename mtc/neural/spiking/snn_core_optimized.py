"""
OPTIMIZED Spiking Neural Network Core - Fast Temporal Processor
===========================================================================

This is a heavily optimized version of the SNN that uses:
- Vectorized NumPy operations (no Python loops)
- Matrix-based connection propagation
- Optional STDP (disabled during inference)
- Pre-allocated arrays (no memory churn)

Note: Speed is the difference between thought and action.

Performance targets:
- Original: ~2000-2700ms per encoding
- Optimized: <200ms per encoding (10x+ speedup)
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class OptimizedSpikingNeuralNetwork:
    """
    Vectorized spiking neural network optimized for speed.

    Key optimizations:
    1. All neuron updates done with NumPy vectorization
    2. Connection weights stored as matrices for fast matmul
    3. STDP learning optional and batched
    4. Spike recording minimized (only counts, not lists)
    5. Pre-allocated arrays (no dynamic memory allocation)
    """

    def __init__(
        self,
        n_input: int = 10,
        n_hidden: int = 50,
        n_output: int = 10,
        dt: float = 1.0,
        learning_enabled: bool = False,
    ):  # Default: disabled during inference
        """
        Initialize the optimized SNN.

        Args:
            n_input: Number of input neurons
            n_hidden: Number of hidden neurons
            n_output: Number of output neurons
            dt: Simulation timestep in milliseconds
            learning_enabled: Enable STDP learning (slows down inference)
        """
        self.dt = dt
        self.current_time = 0.0
        self.learning_enabled = learning_enabled

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Neuron state arrays (vectorized)
        # Input layer
        self.V_input = np.ones(n_input) * -70.0  # Membrane potentials
        self.last_spike_input = np.ones(n_input) * -np.inf

        # Hidden layer
        self.V_hidden = np.ones(n_hidden) * -70.0
        self.last_spike_hidden = np.ones(n_hidden) * -np.inf

        # Output layer
        self.V_output = np.ones(n_output) * -70.0
        self.last_spike_output = np.ones(n_output) * -np.inf

        # Neuron parameters (vectorized)
        self.V_rest = -70.0
        self.V_threshold_input = -55.0
        self.V_threshold_hidden = -50.0  # Lower threshold = easier to spike
        self.V_threshold_output = -50.0
        self.V_reset = -75.0

        self.tau_m_input = 10.0  # Fast input
        self.tau_m_hidden = 30.0  # Longer integration
        self.tau_m_output = 40.0  # Even longer for output
        self.tau_ref = 3.0  # Refractory period

        # Connection weight matrices (for fast matmul)
        # W_input_hidden: shape (n_hidden, n_input)
        self.W_input_hidden = np.random.uniform(100.0, 200.0, (n_hidden, n_input))

        # W_hidden_output: shape (n_output, n_hidden)
        self.W_hidden_output = np.random.uniform(80.0, 150.0, (n_output, n_hidden))

        # Spike counters (for firing rate calculation)
        self.spike_counts_input = np.zeros(n_input)
        self.spike_counts_hidden = np.zeros(n_hidden)
        self.spike_counts_output = np.zeros(n_output)

        # Performance metrics
        self.metrics = {"total_spikes": 0, "total_steps": 0, "learning_updates": 0}

        logger.info(
            f"Optimized SNN initialized: {n_input} input, {n_hidden} hidden, "
            f"{n_output} output neurons (learning={'ON' if learning_enabled else 'OFF'})"
        )

    def simulate_step(self, input_currents: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized simulation of one timestep.

        This replaces the 1000+ loop iterations with ~10 NumPy operations.

        Args:
            input_currents: Input currents for each input neuron (shape: n_input)

        Returns:
            Dictionary with spike status for each layer
        """
        self.current_time += self.dt
        self.metrics["total_steps"] += 1

        # === INPUT LAYER (Vectorized) ===
        # Check refractory period
        ref_mask_input = (self.current_time - self.last_spike_input) >= self.tau_ref

        # Leaky integration: dV/dt = (V_rest - V + I) / tau_m
        dV_input = (
            (self.V_rest - self.V_input + input_currents) / self.tau_m_input
        ) * self.dt
        self.V_input += dV_input * ref_mask_input  # Only update if not refractory

        # Check for spikes
        input_spikes = (self.V_input >= self.V_threshold_input) & ref_mask_input

        # Reset spiked neurons
        self.V_input[input_spikes] = self.V_reset
        self.last_spike_input[input_spikes] = self.current_time
        self.spike_counts_input[input_spikes] += 1

        # === HIDDEN LAYER (Vectorized with matmul) ===
        # Calculate input currents from presynaptic spikes
        # W_input_hidden @ input_spikes gives us the summed synaptic currents
        hidden_input_currents = self.W_input_hidden @ input_spikes.astype(float)

        # Check refractory period
        ref_mask_hidden = (self.current_time - self.last_spike_hidden) >= self.tau_ref

        # Leaky integration
        dV_hidden = (
            (self.V_rest - self.V_hidden + hidden_input_currents) / self.tau_m_hidden
        ) * self.dt
        self.V_hidden += dV_hidden * ref_mask_hidden

        # Check for spikes
        hidden_spikes = (self.V_hidden >= self.V_threshold_hidden) & ref_mask_hidden

        # Reset spiked neurons
        self.V_hidden[hidden_spikes] = self.V_reset
        self.last_spike_hidden[hidden_spikes] = self.current_time
        self.spike_counts_hidden[hidden_spikes] += 1

        # === OUTPUT LAYER (Vectorized with matmul) ===
        # Calculate input currents from hidden spikes
        output_input_currents = self.W_hidden_output @ hidden_spikes.astype(float)

        # Check refractory period
        ref_mask_output = (self.current_time - self.last_spike_output) >= self.tau_ref

        # Leaky integration
        dV_output = (
            (self.V_rest - self.V_output + output_input_currents) / self.tau_m_output
        ) * self.dt
        self.V_output += dV_output * ref_mask_output

        # Check for spikes
        output_spikes = (self.V_output >= self.V_threshold_output) & ref_mask_output

        # Reset spiked neurons
        self.V_output[output_spikes] = self.V_reset
        self.last_spike_output[output_spikes] = self.current_time
        self.spike_counts_output[output_spikes] += 1

        # Update metrics
        total_spikes = (
            np.sum(input_spikes) + np.sum(hidden_spikes) + np.sum(output_spikes)
        )
        self.metrics["total_spikes"] += int(total_spikes)

        # STDP learning (if enabled - slower)
        if self.learning_enabled:
            self._apply_stdp_batch(input_spikes, hidden_spikes, output_spikes)

        return {"input": input_spikes, "hidden": hidden_spikes, "output": output_spikes}

    def _apply_stdp_batch(self, input_spikes, hidden_spikes, output_spikes):
        """
        Simplified STDP learning (batched for efficiency).

        Only applies to connections where both pre and post neurons spiked
        in the current timestep (simplified but much faster).
        """
        # Input -> Hidden STDP
        if np.any(input_spikes) and np.any(hidden_spikes):
            # Potentiation where both pre and post spiked
            for i in np.where(hidden_spikes)[0]:
                for j in np.where(input_spikes)[0]:
                    # Simple Hebbian-like update
                    self.W_input_hidden[i, j] += 0.01
                    self.W_input_hidden[i, j] = np.clip(
                        self.W_input_hidden[i, j], 0.0, 250.0
                    )
                    self.metrics["learning_updates"] += 1

        # Hidden -> Output STDP
        if np.any(hidden_spikes) and np.any(output_spikes):
            for i in np.where(output_spikes)[0]:
                for j in np.where(hidden_spikes)[0]:
                    self.W_hidden_output[i, j] += 0.01
                    self.W_hidden_output[i, j] = np.clip(
                        self.W_hidden_output[i, j], 0.0, 200.0
                    )
                    self.metrics["learning_updates"] += 1

    def get_firing_rates(self, window: float = 100.0) -> Dict[str, np.ndarray]:
        """
        Calculate firing rates based on spike counts.

        Args:
            window: Time window in milliseconds

        Returns:
            Dictionary of firing rates (Hz) for each layer
        """
        # Convert to Hz
        window_sec = window / 1000.0

        return {
            "input": self.spike_counts_input / window_sec,
            "hidden": self.spike_counts_hidden / window_sec,
            "output": self.spike_counts_output / window_sec,
        }

    def reset_spike_counts(self):
        """Reset spike counters (call between experiences)"""
        self.spike_counts_input.fill(0)
        self.spike_counts_hidden.fill(0)
        self.spike_counts_output.fill(0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if self.metrics["total_steps"] > 0:
            avg_spikes_per_step = (
                self.metrics["total_spikes"] / self.metrics["total_steps"]
            )
            total_time_sec = self.current_time / 1000.0
            avg_firing_rate = (
                self.metrics["total_spikes"] / total_time_sec
                if total_time_sec > 0
                else 0
            )
        else:
            avg_spikes_per_step = 0
            avg_firing_rate = 0

        return {
            **self.metrics,
            "current_time_ms": self.current_time,
            "avg_spikes_per_step": avg_spikes_per_step,
            "avg_network_firing_rate_hz": avg_firing_rate,
            "total_neurons": self.n_input + self.n_hidden + self.n_output,
            "total_synapses": self.n_input * self.n_hidden
            + self.n_hidden * self.n_output,
        }

    def reset(self):
        """Reset network state"""
        self.current_time = 0.0
        self.V_input.fill(-70.0)
        self.V_hidden.fill(-70.0)
        self.V_output.fill(-70.0)
        self.last_spike_input.fill(-np.inf)
        self.last_spike_hidden.fill(-np.inf)
        self.last_spike_output.fill(-np.inf)
        self.reset_spike_counts()
        self.metrics["total_spikes"] = 0
        self.metrics["total_steps"] = 0


# Testing and demonstration
if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)

    print("Testing OPTIMIZED Spiking Neural Network...")

    # Create optimized network
    snn = OptimizedSpikingNeuralNetwork(
        n_input=10,
        n_hidden=50,
        n_output=10,
        dt=1.0,
        learning_enabled=False,  # Inference mode
    )

    print("\nNetwork Configuration:")
    metrics = snn.get_metrics()
    print(f"  Total neurons: {metrics['total_neurons']}")
    print(f"  Total synapses: {metrics['total_synapses']}")
    print(
        f"  Learning: {'Enabled' if snn.learning_enabled else 'Disabled (inference mode)'}"
    )

    print("\nRunning performance test (100 timesteps)...")

    # Simulate encoding (100 timesteps)
    duration = 100.0
    steps = int(duration / snn.dt)

    # Create input pattern
    input_intensities = np.array([0.9, 0.2, 0.1, 0.0, 0.3, 0.0, 0.8, 0.7, 0.4, 0.6])

    start_time = time.perf_counter()

    for step in range(steps):
        # Convert to input currents
        input_currents = input_intensities * 50.0

        # Add noise
        input_currents += np.random.normal(0, 2.0, len(input_currents))

        # Simulate step
        spikes = snn.simulate_step(input_currents)

    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    print(f"\nSimulation complete!")
    print(f"  Processing time: {elapsed_ms:.2f}ms")
    print(f"  Steps simulated: {steps}")
    print(f"  Time per step: {elapsed_ms/steps:.3f}ms")

    print("\nResults:")
    metrics = snn.get_metrics()
    print(f"  Total spikes: {metrics['total_spikes']}")
    print(f"  Avg spikes/step: {metrics['avg_spikes_per_step']:.2f}")
    print(f"  Avg firing rate: {metrics['avg_network_firing_rate_hz']:.2f} Hz")

    print("\nFiring Rates:")
    rates = snn.get_firing_rates(100.0)
    print(f"  Input layer: {rates['input'].mean():.1f} Hz (avg)")
    print(f"  Hidden layer: {rates['hidden'].mean():.1f} Hz (avg)")
    print(f"  Output layer: {rates['output'].mean():.1f} Hz (avg)")

    print("\nOptimized SNN ready!")
