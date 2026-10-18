#!/usr/bin/env python3
"""
Production Liquid State Machine
===============================

A liquid state machine built on ReservoirPy: a sparse recurrent reservoir
near the edge of chaos with an online RLS (recursive least squares) readout
and an offline ridge readout.

Defaults: a 5,000-unit reservoir, spectral radius 0.9 and 10% recurrent
connectivity.

Persistence: the reservoir weights are not saved. They regenerate from
``LSMConfig.seed``, so the state dict carries only the learned RLS readout
(output weights, bias, the inverse correlation matrix P and the sample
count) together with the reservoir size and seed it was trained against.
P is reservoir_size x reservoir_size, about 200 MB at the default size.

Seeding: importing the module seeds nothing. A construction with a seed
calls the library's ``set_seed`` with it (which also seeds NumPy's global
generator).

Honest readings: the dynamics proxies are None when the trajectory is too
short for them, and ``emotional_valence`` and ``emotional_arousal`` are None
until the RLS readout was trained on this instance or restored; the raw
readout statistics are always reported as ``readout_mean`` and
``readout_std``.
"""

import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge, Input, RLS
from reservoirpy.observables import nrmse, rsquare
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

# Fewest reservoir states for the divergence proxy and the lag-1 correlation
MIN_STATES_LYAPUNOV = 10
MIN_STATES_CORRELATION = 3


@dataclass
class LSMConfig:
    """Configuration for production LSM."""

    reservoir_size: int = 5000  # Number of reservoir neurons
    input_dim: int = 100  # Input dimension
    output_dim: int = 50  # Output dimension

    # Reservoir parameters (edge-of-chaos)
    spectral_radius: float = 0.9  # Critical for edge-of-chaos dynamics
    connectivity: float = 0.1  # 10% connectivity (sparse)
    input_scaling: float = 1.0  # Input weight scaling
    leak_rate: float = 0.3  # Neuron leak rate

    # RLS learning parameters
    rls_alpha: float = (
        0.9  # RLS forgetting factor (0-1, closer to 1 = slower forgetting)
    )
    rls_regularization: float = 1e-4  # Ridge regularization

    # Training parameters
    washout: int = 100  # Initial timesteps to ignore
    batch_size: int = 32

    # Hardware
    use_gpu: bool = False  # ReservoirPy uses NumPy (CPU-based)

    # Seed for the reservoir weights; a restored readout is only valid on the
    # reservoir it was trained against, so the weights must regenerate
    seed: Optional[int] = 42


class ProductionLSM:
    """
    Liquid state machine with a reservoir near the edge of chaos.

    - A sparse recurrent reservoir (5,000 units by default)
    - RLS readout for online adaptation, ridge readout for offline training
    - Reservoir activity and dynamics metrics
    - Persistence of the learned RLS readout
    """

    def __init__(self, config: Optional[LSMConfig] = None, fast_mode: bool = True):
        """
        Initialize the production LSM.

        Args:
            config: LSM configuration
            fast_mode: If True, skip expensive metrics (Lyapunov, etc.) for real-time use
        """
        self.config = config or LSMConfig()
        self.fast_mode = fast_mode
        # True once the RLS readout was trained here or restored; until then
        # its output is not an affect reading
        self.readout_trained = False

        if self.config.seed is not None:
            rpy.set_seed(self.config.seed)

        # Build reservoir
        self._build_reservoir()

        # Initialize metrics tracking
        self.reservoir_states = []
        self.performance_metrics = []

        logger.info(
            f"Production LSM initialized with {self.config.reservoir_size} neurons"
        )
        logger.info(f"  Spectral radius: {self.config.spectral_radius} (edge-of-chaos)")
        logger.info(f"  Connectivity: {self.config.connectivity * 100:.1f}%")
        if fast_mode:
            logger.info("  Fast mode: ENABLED (skipping expensive metrics)")

    def _build_reservoir(self):
        """Build the LSM reservoir and readout layers."""
        # Input layer (ReservoirPy Input doesn't need explicit dimensions)
        self.input_layer = Input(name="input")

        # Reservoir with edge-of-chaos dynamics
        self.reservoir = Reservoir(
            units=self.config.reservoir_size,
            sr=self.config.spectral_radius,  # Spectral radius for edge-of-chaos
            rc_connectivity=self.config.connectivity,  # Sparse connectivity
            input_connectivity=0.2,  # 20% of neurons receive input
            input_scaling=self.config.input_scaling,
            lr=self.config.leak_rate,  # ReservoirPy uses 'lr' not 'leak_rate'
            activation="tanh",  # Non-linear activation
            seed=self.config.seed,
            name="liquid_reservoir",
        )

        # RLS (Recursive Least Squares) learning readout for online adaptation
        self.rls_readout = RLS(
            output_dim=self.config.output_dim,
            alpha=self.config.rls_alpha,  # Forgetting factor
            name="rls_readout",
        )

        # Alternative: Ridge regression readout for offline training
        self.ridge_readout = Ridge(
            output_dim=self.config.output_dim,
            ridge=self.config.rls_regularization,
            name="ridge_readout",
        )

        # Connect the network
        self.input_layer >> self.reservoir
        self.reservoir >> self.rls_readout
        self.reservoir >> self.ridge_readout

        # Create the full model
        self.model = self.input_layer >> self.reservoir >> self.rls_readout

    def process_input(
        self, input_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process input through the LSM reservoir.

        Args:
            input_data: Input array (timesteps, input_dim) or (batch, timesteps, input_dim)

        Returns:
            Tuple of (output, state dictionary)
        """
        start_time = time.time()

        # Ensure correct shape
        if len(input_data.shape) == 2:
            # Single sequence: (timesteps, input_dim)
            pass
        elif len(input_data.shape) == 3:
            # Batch: (batch, timesteps, input_dim) -> process each
            outputs = []
            states = []
            for i in range(input_data.shape[0]):
                out, state = self.process_input(input_data[i])
                outputs.append(out)
                states.append(state)
            return np.stack(outputs), states
        else:
            input_data = input_data.reshape(-1, self.config.input_dim)

        # Run through reservoir
        reservoir_states = self.reservoir.run(input_data)

        # Get output from RLS readout
        output = self.rls_readout.run(reservoir_states)

        # Calculate state metrics
        processing_time = (time.time() - start_time) * 1000  # ms

        # Analyze reservoir dynamics
        state_metrics = self._analyze_reservoir_state(reservoir_states)

        # NOTE: Only return final state to prevent 4MB+ memory leak per call.
        # The full trajectory was causing unbounded memory growth.
        final_state = reservoir_states[-1] if len(reservoir_states) > 0 else None

        state = {
            "reservoir_states": (
                [final_state] if final_state is not None else []
            ),  # Just final
            "final_reservoir_state": final_state,  # Convenience accessor
            "output": output,
            "processing_time_ms": processing_time,
            "trajectory_length": len(reservoir_states),  # For debugging
            **state_metrics,
        }

        # Explicit cleanup of large array
        del reservoir_states

        return output, state

    def _analyze_reservoir_state(self, states: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the reservoir state for consciousness metrics.

        Args:
            states: Reservoir states (timesteps, neurons)

        Returns:
            Dictionary of state metrics
        """
        # Always calculate basic activity metrics (fast)
        metrics = {
            "mean_activity": float(np.mean(np.abs(states))),
            "std_activity": float(np.std(states)),
            "sparsity": float(np.mean(np.abs(states) < 0.1)),  # Fraction near zero
        }

        # Skip expensive metrics in fast mode
        if not self.fast_mode:
            # Dynamics metrics (EXPENSIVE)
            metrics["lyapunov_estimate"] = self._estimate_lyapunov(states)
            metrics["edge_of_chaos_score"] = self._edge_of_chaos_score(states)

            # Information metrics (EXPENSIVE)
            metrics["entropy"] = self._calculate_entropy(states)
            metrics["correlation_time"] = self._correlation_time(states)
            # Why a dynamics proxy is None: too few states for it
            metrics["dynamics_reason"] = (
                "too_few_states"
                if metrics["lyapunov_estimate"] is None
                or metrics["correlation_time"] is None
                else None
            )
        else:
            # Fast mode measures none of them
            metrics["lyapunov_estimate"] = None
            metrics["edge_of_chaos_score"] = None
            metrics["entropy"] = None
            metrics["correlation_time"] = None
            metrics["dynamics_reason"] = "fast_mode"

        return metrics

    def _estimate_lyapunov(self, states: np.ndarray) -> Optional[float]:
        """
        A divergence proxy: the mean log absolute one-step state difference.

        It is not a Lyapunov exponent (no perturbation is followed); larger
        values mean larger step-to-step changes. None with fewer than
        MIN_STATES_LYAPUNOV states.
        """
        if len(states) < MIN_STATES_LYAPUNOV:
            return None

        # Simple estimation using state divergence
        diffs = np.diff(states, axis=0)
        divergence = np.mean(np.log(np.abs(diffs) + 1e-10))

        return float(divergence)

    def _edge_of_chaos_score(self, states: np.ndarray) -> Optional[float]:
        """
        ``exp(-10 |d|)`` of the divergence proxy ``d``, in (0, 1].

        None when the proxy is None (too few states).
        """
        lyapunov = self._estimate_lyapunov(states)
        if lyapunov is None:
            return None

        # Edge of chaos is near zero Lyapunov
        score = np.exp(-np.abs(lyapunov) * 10)

        return float(score)

    def _calculate_entropy(self, states: np.ndarray) -> float:
        """Calculate Shannon entropy of reservoir states."""
        # Discretize states into bins
        hist, _ = np.histogram(states.flatten(), bins=50)
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # Remove zeros

        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs))

        return float(entropy)

    def _correlation_time(self, states: np.ndarray) -> Optional[float]:
        """The absolute lag-1 autocorrelation of the mean activity.

        None with fewer than MIN_STATES_CORRELATION states (a lag-1
        correlation of fewer than two pairs is undefined) and when the mean
        activity is constant.
        """
        if len(states) < MIN_STATES_CORRELATION:
            return None

        # Autocorrelation of mean activity
        mean_activity = np.mean(states, axis=1)
        head, tail = mean_activity[:-1], mean_activity[1:]
        if np.std(head) == 0.0 or np.std(tail) == 0.0:
            return None

        # Simple autocorrelation at lag 1
        corr = np.corrcoef(head, tail)[0, 1]
        return float(np.abs(corr)) if np.isfinite(corr) else None

    def process_subconscious(self, sensory_input: np.ndarray) -> Dict[str, Any]:
        """
        Process sensory input through the subconscious LSM.

        Args:
            sensory_input: Sensory data array

        Returns:
            Subconscious state dictionary
        """
        # Reshape if needed
        if len(sensory_input.shape) == 1:
            sensory_input = sensory_input.reshape(1, -1)

        # Process through liquid
        output, state = self.process_input(sensory_input)

        # Dynamics metrics are None in fast mode; the derived values stay
        # None rather than taking a stand-in constant
        edge_of_chaos = state.get("edge_of_chaos_score")
        entropy = state.get("entropy")
        correlation = state.get("correlation_time")

        # The readout's raw statistics; they read as affect only once the
        # readout was trained on this instance or restored
        readout_mean = float(np.mean(output))
        readout_std = float(np.std(output))
        trained = bool(self.readout_trained)

        subconscious_state = {
            "raw_output": output,
            "readout_mean": readout_mean,
            "readout_std": readout_std,
            "emotional_valence": readout_mean if trained else None,
            "emotional_arousal": readout_std if trained else None,
            "creative_emergence": entropy,
            "stability": (1.0 - edge_of_chaos if edge_of_chaos is not None else None),
            "processing_depth": correlation,
            **state,
        }

        chaos = f"{edge_of_chaos:.3f}" if edge_of_chaos is not None else "n/a"
        valence = f"{readout_mean:.3f}" if trained else "n/a"
        logger.debug(f"Processed subconscious state: valence={valence}, chaos={chaos}")

        return subconscious_state

    def train_rls(
        self, input_sequences: np.ndarray, target_sequences: np.ndarray
    ) -> float:
        """
        Train the LSM using RLS (Recursive Least Squares) learning.

        Args:
            input_sequences: Input data (samples, timesteps, input_dim)
            target_sequences: Target outputs (samples, timesteps, output_dim)

        Returns:
            Training loss
        """
        total_loss = 0
        num_sequences = input_sequences.shape[0]

        for i in range(num_sequences):
            # Get reservoir states
            states = self.reservoir.run(input_sequences[i])

            # Train RLS readout using fit() method
            self.rls_readout.fit(states, target_sequences[i])

            # Calculate loss
            predictions = self.rls_readout.run(states)
            loss = nrmse(target_sequences[i], predictions)
            total_loss += loss

        avg_loss = total_loss / num_sequences
        if num_sequences > 0:
            self.readout_trained = True
        logger.info(f"RLS training complete: avg_loss={avg_loss:.4f}")

        return avg_loss

    def train_ridge(
        self, input_sequences: np.ndarray, target_sequences: np.ndarray
    ) -> float:
        """
        Train the LSM using ridge regression (offline).

        Args:
            input_sequences: Input data
            target_sequences: Target outputs

        Returns:
            Training loss
        """
        # Collect all reservoir states
        all_states = []
        all_targets = []

        for i in range(input_sequences.shape[0]):
            states = self.reservoir.run(input_sequences[i])
            all_states.append(states[self.config.washout :])  # Remove washout
            all_targets.append(target_sequences[i][self.config.washout :])

        # Concatenate
        X = np.vstack(all_states)
        y = np.vstack(all_targets)

        # Train ridge regression
        self.ridge_readout.train(X, y)

        # Evaluate
        predictions = self.ridge_readout.run(X)
        loss = nrmse(y, predictions)
        r2 = rsquare(y, predictions)

        logger.info(f"Ridge training complete: loss={loss:.4f}, R2={r2:.4f}")

        return loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize the learned RLS readout.

        The arrays are the readout's output weights (reservoir_size x
        output_dim), bias and P matrix, present only once the readout has
        been initialized. The scalars record the reservoir size and seed the
        readout was trained against.

        Returns:
            Dict with scalars, arrays and histories keys.
        """
        readout = self.rls_readout
        initialized = getattr(readout, "initialized", False) is True
        arrays: Dict[str, np.ndarray] = {}
        if initialized:
            for name in ("Wout", "bias", "P"):
                value = getattr(readout, name, None)
                if isinstance(value, np.ndarray):
                    arrays[f"rls_{name}"] = value
        samples = getattr(readout, "S", 0) if initialized else 0
        return {
            "scalars": {
                "reservoir_size": self.config.reservoir_size,
                "seed": self.config.seed,
                "spectral_radius": self.config.spectral_radius,
                "rls_initialized": initialized,
                "rls_samples": float(samples),
            },
            "arrays": arrays,
            "histories": {},
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the RLS readout saved by ``to_state_dict``.

        The readout is restored only when the saved reservoir size and seed
        match this instance (otherwise it was trained on a different
        reservoir) and the output weights and bias have the expected shapes
        and are finite. A non-finite or misshapen P is replaced by the RLS
        initial value (identity over alpha), keeping the weights for
        inference while online learning restarts its correlation estimate.
        Every rejection is logged.
        """
        scalars = state.get("scalars") or {}
        arrays = state.get("arrays") or {}
        if not scalars.get("rls_initialized") or "rls_Wout" not in arrays:
            return

        size = self.config.reservoir_size
        if scalars.get("reservoir_size") != size:
            logger.warning(
                f"LSM restore skipped: saved reservoir_size "
                f"{scalars.get('reservoir_size')} does not match {size}"
            )
            return
        if scalars.get("seed") != self.config.seed:
            logger.warning(
                f"LSM restore skipped: saved reservoir seed {scalars.get('seed')} "
                f"does not match {self.config.seed}; the readout belongs to a "
                "different reservoir"
            )
            return

        out_dim = self.config.output_dim
        wout = np.asarray(arrays["rls_Wout"], dtype=float)
        bias = np.asarray(arrays.get("rls_bias", np.zeros(out_dim)), dtype=float)
        if wout.shape != (size, out_dim) or bias.shape != (out_dim,):
            logger.warning(
                f"LSM restore skipped: readout shapes {wout.shape} and "
                f"{bias.shape}, expected {(size, out_dim)} and {(out_dim,)}"
            )
            return
        if not (np.all(np.isfinite(wout)) and np.all(np.isfinite(bias))):
            logger.warning(
                "LSM restore skipped: the saved readout holds non-finite values"
            )
            return

        readout = self.rls_readout
        alpha = float(getattr(readout, "alpha", 1e-6))
        p_matrix = arrays.get("rls_P")
        if p_matrix is not None:
            p_matrix = np.asarray(p_matrix, dtype=float)
        if (
            p_matrix is None
            or p_matrix.shape != (size, size)
            or not np.all(np.isfinite(p_matrix))
        ):
            logger.warning(
                "LSM restore: the saved RLS P matrix is missing, misshapen or "
                "non-finite; reinitializing it"
            )
            p_matrix = np.eye(size) / alpha

        samples = scalars.get("rls_samples", 0.0)
        try:
            samples = float(samples)
        except (TypeError, ValueError):
            samples = 0.0
        if not np.isfinite(samples):
            samples = 0.0

        readout.input_dim = size
        readout.output_dim = out_dim
        readout.Wout = wout
        readout.bias = bias
        readout.P = p_matrix
        readout.S = samples
        readout.state = {"out": np.zeros((out_dim,))}
        readout.initialized = True
        self.readout_trained = True
        logger.info(f"LSM readout restored for a {size}-unit reservoir")

    def benchmark_performance(self) -> Dict[str, Any]:
        """
        Benchmark the LSM performance.

        Returns:
            Performance metrics dictionary
        """
        logger.info("Running LSM performance benchmark...")

        # Test input
        test_timesteps = 100
        test_input = np.random.randn(test_timesteps, self.config.input_dim).astype(
            np.float32
        )

        # Warmup
        for _ in range(5):
            _ = self.process_input(test_input)

        # Timing runs
        times = []
        chaos_scores = []
        entropies = []

        for _ in range(50):
            start = time.time()
            output, state = self.process_input(test_input)
            times.append((time.time() - start) * 1000)
            # Handle None values in fast_mode
            chaos = state.get("edge_of_chaos_score")
            ent = state.get("entropy")
            if chaos is not None:
                chaos_scores.append(chaos)
            if ent is not None:
                entropies.append(ent)

        metrics = {
            "mean_latency_ms": np.mean(times),
            "std_latency_ms": np.std(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "mean_chaos_score": float(np.mean(chaos_scores)) if chaos_scores else None,
            "mean_entropy": float(np.mean(entropies)) if entropies else None,
            "reservoir_neurons": self.config.reservoir_size,
            "spectral_radius": self.config.spectral_radius,
            "connectivity": self.config.connectivity,
        }

        logger.info(f"Benchmark complete: {metrics['mean_latency_ms']:.2f}ms average")

        return metrics


# Test function
def test_production_lsm():
    """Test the production LSM implementation."""
    print("Testing Production Liquid State Machine")
    print("=" * 60)

    # Create smaller config for testing
    config = LSMConfig(
        reservoir_size=1000,  # Smaller for testing
        spectral_radius=0.9,  # Edge of chaos
        connectivity=0.1,
        input_dim=50,
        output_dim=20,
    )

    lsm = ProductionLSM(config)

    # Test subconscious processing
    print("\n1. Testing subconscious processing...")
    sensory_input = np.random.randn(100, 50).astype(np.float32)  # 100 timesteps

    state = lsm.process_subconscious(sensory_input)

    print(f"  Processed in {state.get('processing_time_ms', 0):.2f}ms")
    print(f"  Readout mean: {state['readout_mean']:.3f}")
    creative = state.get("creative_emergence")
    print(
        f"  Creative emergence (entropy): {creative:.3f}"
        if creative is not None
        else "  Creative emergence (entropy): n/a (fast mode)"
    )
    edge_score = state.get("edge_of_chaos_score")
    print(
        f"  Edge-of-chaos score: {edge_score:.3f}"
        if edge_score
        else "   Edge-of-chaos score: N/A (fast mode)"
    )

    # Test training
    print("\n2. Testing RLS learning...")
    train_inputs = np.random.randn(5, 50, 50).astype(np.float32)
    train_targets = np.random.randn(5, 50, 20).astype(np.float32)

    loss = lsm.train_rls(train_inputs, train_targets)
    print(f"  Training loss: {loss:.4f}")

    # Benchmark
    print("\n3. Running performance benchmark...")
    metrics = lsm.benchmark_performance()

    print(f"  Mean latency: {metrics['mean_latency_ms']:.2f}ms")
    print(f"  Chaos score: {metrics['mean_chaos_score']}")
    print(f"  Reservoir size: {metrics['reservoir_neurons']}")

    print("\n" + "=" * 60)
    print("Production LSM test complete")


if __name__ == "__main__":
    test_production_lsm()
