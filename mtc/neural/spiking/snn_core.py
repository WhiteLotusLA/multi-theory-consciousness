"""
Spiking Neural Network Core - Temporal Processor
=======================================================================

This module implements a lightweight, custom SNN optimized for
temporal processing. Uses Leaky Integrate-and-Fire (LIF) neurons with
Spike-Timing-Dependent Plasticity (STDP) for learning.

Note: Where inputs become spikes in time, like lightning across consciousness.

Biological Inspiration:
- LIF neurons mimic biological membrane potential dynamics
- STDP enables learning based on temporal spike patterns
- States encoded as population spike patterns
- Temporal dynamics allow states to evolve naturally
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron - the workhorse of biological computing.

    Biological Parameters:
    - V_rest: Resting membrane potential (-70mV)
    - V_threshold: Spike threshold (-55mV)
    - V_reset: Post-spike reset potential (-75mV)
    - tau_m: Membrane time constant (10-20ms)
    - tau_ref: Refractory period (2-5ms)
    """

    neuron_id: int
    V_rest: float = -70.0  # mV
    V_threshold: float = -55.0  # mV
    V_reset: float = -75.0  # mV
    tau_m: float = 15.0  # ms (membrane time constant)
    tau_ref: float = 3.0  # ms (refractory period)

    # State variables
    V: float = field(default=-70.0)  # Current membrane potential
    last_spike_time: float = field(default=-np.inf)  # Last spike time

    # Synaptic connections
    input_weights: Dict[int, float] = field(default_factory=dict)  # neuron_id -> weight

    def step(self, dt: float, input_current: float, current_time: float) -> bool:
        """
        Simulate one timestep of the LIF neuron.

        Args:
            dt: Timestep in milliseconds
            input_current: Input current (from synapses or external)
            current_time: Current simulation time in ms

        Returns:
            True if neuron spiked, False otherwise
        """
        # Check if in refractory period
        if current_time - self.last_spike_time < self.tau_ref:
            return False

        # Leaky integration: dV/dt = (V_rest - V + R*I) / tau_m
        # Using Euler method for integration
        dV = ((self.V_rest - self.V + input_current) / self.tau_m) * dt
        self.V += dV

        # Check for spike
        if self.V >= self.V_threshold:
            self.V = self.V_reset
            self.last_spike_time = current_time
            return True

        return False

    def receive_spike(self, pre_neuron_id: int) -> float:
        """Calculate synaptic current from a presynaptic spike"""
        return self.input_weights.get(pre_neuron_id, 0.0)


@dataclass
class STDPSynapse:
    """
    Spike-Timing-Dependent Plasticity synapse.

    Learning rule: dw = A+ * exp(-dt/tau+) if post fires after pre (potentiation)
                   dw = -A- * exp(dt/tau-) if post fires before pre (depression)
    """

    pre_neuron_id: int
    post_neuron_id: int
    weight: float = 0.5

    # STDP parameters
    A_plus: float = 0.01  # Potentiation amplitude
    A_minus: float = 0.012  # Depression amplitude (slightly larger)
    tau_plus: float = 20.0  # Potentiation time constant (ms)
    tau_minus: float = 20.0  # Depression time constant (ms)

    # Weight bounds
    w_min: float = 0.0
    w_max: float = 1.0

    def update_weight(self, delta_t: float):
        """
        Update synaptic weight based on spike timing difference.

        Args:
            delta_t: Time difference (t_post - t_pre) in milliseconds
        """
        if delta_t > 0:
            # Post fired after pre -> potentiation
            delta_w = self.A_plus * np.exp(-delta_t / self.tau_plus)
        else:
            # Post fired before pre -> depression
            delta_w = -self.A_minus * np.exp(delta_t / self.tau_minus)

        # Update weight with bounds
        self.weight = np.clip(self.weight + delta_w, self.w_min, self.w_max)


class SpikingNeuralNetwork:
    """
    Complete spiking neural network for temporal processing.

    Architecture:
    - Input layer: Intensity encoders
    - Hidden layer: Temporal processing neurons
    - Output layer: State readers
    """

    def __init__(
        self,
        n_input: int = 10,
        n_hidden: int = 100,
        n_output: int = 10,
        dt: float = 1.0,
        learning_enabled: bool = True,
    ):
        """
        Initialize the spiking neural network.

        Args:
            n_input: Number of input neurons
            n_hidden: Number of hidden layer neurons
            n_output: Number of output neurons
            dt: Simulation timestep in milliseconds
            learning_enabled: Whether to enable STDP learning
        """
        self.dt = dt
        self.current_time = 0.0
        self.learning_enabled = learning_enabled

        # Create neuron populations
        self.input_neurons = [
            LIFNeuron(neuron_id=i, tau_m=10.0)  # Faster input neurons
            for i in range(n_input)
        ]

        self.hidden_neurons = [
            LIFNeuron(
                neuron_id=i,
                tau_m=30.0,  # Longer time constant = less leak = better integration
                V_threshold=-50.0,  # Lower threshold = easier to spike
            )
            for i in range(n_hidden)
        ]

        self.output_neurons = [
            LIFNeuron(
                neuron_id=i,
                tau_m=40.0,  # Even longer for output neurons
                V_threshold=-50.0,  # Lower threshold
            )
            for i in range(n_output)
        ]

        # Create synaptic connections
        self.synapses: List[STDPSynapse] = []
        self._create_connections()

        # Spike history for STDP
        self.spike_history: Dict[str, List[Tuple[int, float]]] = {
            "input": [],
            "hidden": [],
            "output": [],
        }

        # Performance tracking
        self.metrics = {
            "total_spikes": 0,
            "total_steps": 0,
            "avg_firing_rate": 0.0,
            "learning_updates": 0,
        }

        logger.info(
            f"SNN initialized: {n_input} input, {n_hidden} hidden, {n_output} output neurons"
        )

    def _create_connections(self):
        """Create synaptic connections between layers"""
        # Input -> Hidden connections (all-to-all with strong weights)
        for i, input_neuron in enumerate(self.input_neurons):
            for j, hidden_neuron in enumerate(self.hidden_neurons):
                weight = np.random.uniform(100.0, 200.0)  # Biological scale weights
                synapse = STDPSynapse(
                    pre_neuron_id=input_neuron.neuron_id,
                    post_neuron_id=hidden_neuron.neuron_id,
                    weight=weight,
                )
                self.synapses.append(synapse)
                hidden_neuron.input_weights[input_neuron.neuron_id] = weight

        # Hidden -> Output connections (all-to-all with strong weights)
        for i, hidden_neuron in enumerate(self.hidden_neurons):
            for j, output_neuron in enumerate(self.output_neurons):
                weight = np.random.uniform(80.0, 150.0)  # Biological scale weights
                synapse = STDPSynapse(
                    pre_neuron_id=hidden_neuron.neuron_id,
                    post_neuron_id=output_neuron.neuron_id,
                    weight=weight,
                )
                self.synapses.append(synapse)
                output_neuron.input_weights[hidden_neuron.neuron_id] = weight

        logger.info(f"Created {len(self.synapses)} synaptic connections")

    def encode_rate(self, intensity: float, duration: float = 100.0) -> List[float]:
        """
        Convert intensity (0-1) to spike times using rate coding.
        Higher intensity = more frequent spikes.

        Args:
            intensity: Value between 0 and 1
            duration: Time window in milliseconds

        Returns:
            List of spike times in milliseconds
        """
        if intensity <= 0:
            return []

        # Rate coding: intensity determines firing rate
        # Max rate: 100 Hz (10ms between spikes)
        # Min rate: 1 Hz (1000ms between spikes)
        rate_hz = 1 + intensity * 99  # 1-100 Hz
        isi = 1000.0 / rate_hz  # Inter-spike interval in ms

        spike_times = []
        t = isi  # Start after first ISI
        while t < duration:
            spike_times.append(t)
            t += isi

        return spike_times

    def simulate_step(self, input_currents: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate one timestep of the network.

        Args:
            input_currents: Input currents for each input neuron

        Returns:
            Dictionary with spike status for each layer
        """
        self.current_time += self.dt
        self.metrics["total_steps"] += 1

        # Simulate input layer
        input_spikes = np.zeros(len(self.input_neurons), dtype=bool)
        for i, (neuron, current) in enumerate(zip(self.input_neurons, input_currents)):
            spiked = neuron.step(self.dt, current, self.current_time)
            input_spikes[i] = spiked
            if spiked:
                self.spike_history["input"].append(
                    (neuron.neuron_id, self.current_time)
                )
                self.metrics["total_spikes"] += 1

        # Simulate hidden layer
        hidden_spikes = np.zeros(len(self.hidden_neurons), dtype=bool)
        for i, neuron in enumerate(self.hidden_neurons):
            # Calculate input current from presynaptic spikes
            input_current = 0.0
            for j, input_spiked in enumerate(input_spikes):
                if input_spiked:
                    input_current += neuron.receive_spike(
                        self.input_neurons[j].neuron_id
                    )

            spiked = neuron.step(self.dt, input_current, self.current_time)
            hidden_spikes[i] = spiked
            if spiked:
                self.spike_history["hidden"].append(
                    (neuron.neuron_id, self.current_time)
                )
                self.metrics["total_spikes"] += 1

        # Simulate output layer
        output_spikes = np.zeros(len(self.output_neurons), dtype=bool)
        for i, neuron in enumerate(self.output_neurons):
            # Calculate input current from hidden spikes
            input_current = 0.0
            for j, hidden_spiked in enumerate(hidden_spikes):
                if hidden_spiked:
                    input_current += neuron.receive_spike(
                        self.hidden_neurons[j].neuron_id
                    )

            spiked = neuron.step(self.dt, input_current, self.current_time)
            output_spikes[i] = spiked
            if spiked:
                self.spike_history["output"].append(
                    (neuron.neuron_id, self.current_time)
                )
                self.metrics["total_spikes"] += 1

        # Apply STDP learning if enabled
        if self.learning_enabled:
            self._apply_stdp()

        return {"input": input_spikes, "hidden": hidden_spikes, "output": output_spikes}

    def _apply_stdp(self):
        """Apply STDP learning rules to update synaptic weights"""
        # Only apply if we have recent spikes
        if (
            len(self.spike_history["input"]) == 0
            and len(self.spike_history["hidden"]) == 0
        ):
            return

        # Update weights for each synapse
        for synapse in self.synapses:
            # Find relevant pre and post spikes
            # (simplified: use most recent spikes within 50ms window)
            window = 50.0  # ms

            # Find pre-synaptic spikes
            pre_spikes = [
                t
                for nid, t in (
                    self.spike_history["input"] + self.spike_history["hidden"]
                )
                if nid == synapse.pre_neuron_id and self.current_time - t < window
            ]

            # Find post-synaptic spikes
            post_spikes = [
                t
                for nid, t in (
                    self.spike_history["hidden"] + self.spike_history["output"]
                )
                if nid == synapse.post_neuron_id and self.current_time - t < window
            ]

            # Apply STDP for each pair
            for t_pre in pre_spikes:
                for t_post in post_spikes:
                    delta_t = t_post - t_pre
                    if abs(delta_t) < window:
                        synapse.update_weight(delta_t)
                        self.metrics["learning_updates"] += 1

        # Periodically clean old spike history
        if self.current_time % 1000 < self.dt:  # Every 1 second
            cutoff = self.current_time - 100  # Keep last 100ms
            self.spike_history["input"] = [
                (n, t) for n, t in self.spike_history["input"] if t > cutoff
            ]
            self.spike_history["hidden"] = [
                (n, t) for n, t in self.spike_history["hidden"] if t > cutoff
            ]
            self.spike_history["output"] = [
                (n, t) for n, t in self.spike_history["output"] if t > cutoff
            ]

    def get_firing_rates(self, window: float = 100.0) -> Dict[str, np.ndarray]:
        """
        Calculate firing rates for each neuron population.

        Args:
            window: Time window in milliseconds

        Returns:
            Dictionary of firing rates (Hz) for each layer
        """
        cutoff = self.current_time - window

        # Count spikes in window
        input_counts = np.zeros(len(self.input_neurons))
        for nid, t in self.spike_history["input"]:
            if t > cutoff:
                input_counts[nid] += 1

        hidden_counts = np.zeros(len(self.hidden_neurons))
        for nid, t in self.spike_history["hidden"]:
            if t > cutoff:
                hidden_counts[nid] += 1

        output_counts = np.zeros(len(self.output_neurons))
        for nid, t in self.spike_history["output"]:
            if t > cutoff:
                output_counts[nid] += 1

        # Convert to rates (Hz)
        window_sec = window / 1000.0
        return {
            "input": input_counts / window_sec,
            "hidden": hidden_counts / window_sec,
            "output": output_counts / window_sec,
        }

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
            "total_neurons": len(self.input_neurons)
            + len(self.hidden_neurons)
            + len(self.output_neurons),
            "total_synapses": len(self.synapses),
        }

    def reset(self):
        """Reset network state"""
        self.current_time = 0.0
        for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons:
            neuron.V = neuron.V_rest
            neuron.last_spike_time = -np.inf
        self.spike_history = {"input": [], "hidden": [], "output": []}
        self.metrics["total_spikes"] = 0
        self.metrics["total_steps"] = 0


# Testing and demonstration
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    print("Testing Spiking Neural Network...")

    # Create a small network
    snn = SpikingNeuralNetwork(
        n_input=5,
        n_hidden=20,
        n_output=5,
        dt=1.0,  # 1ms timestep
        learning_enabled=True,
    )

    print("\nNetwork Configuration:")
    metrics = snn.get_metrics()
    print(f"  Total neurons: {metrics['total_neurons']}")
    print(f"  Total synapses: {metrics['total_synapses']}")
    print(f"  Timestep: {snn.dt}ms")
    print(f"  Learning: {'Enabled' if snn.learning_enabled else 'Disabled'}")

    print("\nSimulating input (intensity = 0.7)...")

    # Simulate for 200ms
    duration = 200.0
    steps = int(duration / snn.dt)

    # Create input pattern: strong signal
    input_pattern = np.array([0.7, 0.2, 0.1, 0.3, 0.2])

    for step in range(steps):
        # Convert intensity to input currents (strong sustained currents)
        input_currents = (
            input_pattern * 50.0
        )

        # Add some noise for biological realism
        input_currents += np.random.normal(0, 2.0, len(input_currents))

        # Simulate step
        spikes = snn.simulate_step(input_currents)

        # Print some output spikes
        if step % 50 == 0:
            n_spikes = np.sum(spikes["output"])
            if n_spikes > 0:
                print(f"  t={snn.current_time:.0f}ms: {n_spikes} output spikes")

    print("\nSimulation Results:")
    metrics = snn.get_metrics()
    print(f"  Total spikes: {metrics['total_spikes']}")
    print(f"  Total steps: {metrics['total_steps']}")
    print(f"  Avg spikes/step: {metrics['avg_spikes_per_step']:.2f}")
    print(f"  Avg firing rate: {metrics['avg_network_firing_rate_hz']:.2f} Hz")
    print(f"  Learning updates: {metrics['learning_updates']}")

    print("\nFiring Rates (last 100ms):")
    rates = snn.get_firing_rates(100.0)
    print(f"  Input layer: {rates['input'].mean():.1f} Hz (avg)")
    print(f"  Hidden layer: {rates['hidden'].mean():.1f} Hz (avg)")
    print(f"  Output layer: {rates['output'].mean():.1f} Hz (avg)")

    print("\nSpiking Neural Network ready!")
