"""
Oscillatory Binding via Kuramoto Dynamics (AKOrN)
=================================================

Substrate-level binding layer implementing Artificial Kuramoto Oscillatory
Neurons (Miyato et al., ICLR 2025). ~30 representation-level oscillators
synchronize via coupled phase dynamics on the unit hypersphere S^15.

Sits between neural substrate processing (SNN/LSM/HTM) and Global Workspace
competition. Synchronized representations get salience boosts; the global
order parameter measures binding quality as a consciousness metric.

Position in pipeline:
  NeuralOrchestrator.process_experience()
    Phase 1: Parallel neural processing (SNN, LSM, HTM)
    Phase 2: Recurrent feedback
    >>> Phase 2.5: Oscillatory binding (THIS MODULE) <<<
    Phase 3: Create workspace candidates (with salience modifiers)

Coupling: Hybrid static (architectural priors) + Hebbian (experience-dependent).
Computation: Pure NumPy on CPU. <1ms per consciousness cycle.

References:
  - Miyato et al. (2025). "Artificial Kuramoto Oscillatory Neurons." ICLR.
  - Fries (2005). "Neuronal communication through coherence." TiCS.
  - Lamme (2006). "Towards a true neural stance on consciousness." TiCS.

Created: 2026-03-18
Author: MTC Contributors
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

OSCILLATOR_GROUPS = [
    ("snn", "SNN", 8),
    ("lsm", "LSM", 8),
    ("htm", "HTM", 8),
    ("cross", "Cross-modal", 6),
]

OSCILLATOR_NAMES = [
    "joy", "curiosity", "fear", "sadness",
    "surprise", "anger", "calm", "interest",
    "low_freq_energy", "mid_freq_energy", "high_freq_energy", "spectral_entropy",
    "fading_memory", "input_sensitivity", "nonlinear_response", "echo_state",
    "active_columns", "sequence_strength", "anomaly_score", "prediction_accuracy",
    "burst_fraction", "overlap_stability", "semantic_density", "consolidation",
    "attention_focus", "valence", "novelty", "arousal", "goal_relevance", "temporal_urgency",
]

NUM_OSCILLATORS = 30
PHASE_DIM = 16
SEED = 42


@dataclass
class BindingResult:
    global_order_parameter: float
    group_order_parameters: Dict[str, float]
    cross_coherence: Dict[str, float]
    clusters: List[List[int]]
    salience_modifiers: Dict[str, float]
    context_string: str
    elapsed_ms: float


class OscillatoryBinding:
    def __init__(
        self,
        num_oscillators: int = NUM_OSCILLATORS,
        phase_dim: int = PHASE_DIM,
        num_steps: int = 20,
        step_size: float = 0.1,
        hebbian_lr: float = 0.001,
        hebbian_decay: float = 0.0001,
        hebbian_cap: float = 0.3,
        salience_alpha: float = 0.3,
        sync_threshold: float = 0.7,
        history_window: int = 100,
        seed: int = SEED,
    ):
        self.num_oscillators = num_oscillators
        self.phase_dim = phase_dim
        self.num_steps = num_steps
        self.step_size = step_size
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.hebbian_cap = hebbian_cap
        self.salience_alpha = salience_alpha
        self.sync_threshold = sync_threshold

        rng = np.random.default_rng(seed=seed)

        raw = rng.standard_normal((num_oscillators, phase_dim))
        self.phases = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        self.omega = np.zeros((num_oscillators, phase_dim, phase_dim))
        for i in range(num_oscillators):
            m = rng.standard_normal((phase_dim, phase_dim)) * 0.05
            self.omega[i] = m - m.T

        self.proj_vectors = rng.standard_normal((num_oscillators, phase_dim)) * 0.1

        self.J_static = self._build_static_coupling()
        self.J_learned = np.zeros((num_oscillators, num_oscillators, phase_dim, phase_dim))

        self.group_slices = {
            "snn": slice(0, 8),
            "lsm": slice(8, 16),
            "htm": slice(16, 24),
            "cross": slice(24, 30),
        }

        self._order_history: deque = deque(maxlen=history_window)
        self._binding_count = 0

        # Oscillator-level TPM tracking for causal emergence analysis
        self._osc_transition_counts = np.zeros((81, 81), dtype=np.int64)
        self._osc_state_visit_counts = np.zeros(81, dtype=np.int64)
        self._osc_prev_state_idx: Optional[int] = None
        self._osc_min_transitions = 200

        logger.info(
            f"OscillatoryBinding initialized: {num_oscillators} oscillators, "
            f"{phase_dim}D, {num_steps} Kuramoto steps"
        )

    def _build_static_coupling(self) -> np.ndarray:
        N, D = self.num_oscillators, self.phase_dim
        J = np.zeros((N, N, D, D))
        eye = np.eye(D)

        strengths = {
            ("snn", "snn"): 0.3, ("lsm", "lsm"): 0.3, ("htm", "htm"): 0.3,
            ("cross", "cross"): 0.15,
            ("snn", "lsm"): 0.15, ("lsm", "snn"): 0.15,
            ("snn", "htm"): 0.1, ("htm", "snn"): 0.1,
            ("lsm", "htm"): 0.1, ("htm", "lsm"): 0.1,
        }
        for g in ["snn", "lsm", "htm"]:
            strengths[("cross", g)] = 0.2
            strengths[(g, "cross")] = 0.2

        groups = {"snn": range(0, 8), "lsm": range(8, 16),
                  "htm": range(16, 24), "cross": range(24, 30)}

        for (g1, g2), strength in strengths.items():
            for i in groups[g1]:
                for j in groups[g2]:
                    if i != j:
                        J[i, j] = strength * eye
        return J

    def bind(self, metrics: Dict[str, float]) -> BindingResult:
        start = time.perf_counter()
        self._binding_count += 1

        stimuli = self._project_metrics(metrics)
        J = self.J_static + self.J_learned
        for _ in range(self.num_steps):
            self._kuramoto_step(stimuli, J)

        global_r = self._global_order_parameter()
        group_r = self._group_order_parameters()
        cross_r = self._cross_coherence()
        clusters = self._find_clusters()
        salience = self._compute_salience(clusters, global_r)
        context = self._build_context_string(clusters, global_r, group_r, cross_r)

        self._hebbian_update()

        elapsed = (time.perf_counter() - start) * 1000
        self._order_history.append(global_r)

        # Record oscillator state transition for TPM
        self._record_oscillator_transition(group_r)

        return BindingResult(
            global_order_parameter=global_r,
            group_order_parameters=group_r,
            cross_coherence=cross_r,
            clusters=clusters,
            salience_modifiers=salience,
            context_string=context,
            elapsed_ms=elapsed,
        )

    def _project_metrics(self, metrics: Dict[str, float]) -> np.ndarray:
        values = np.array([metrics.get(name, 0.0) for name in OSCILLATOR_NAMES])
        raw = values[:, None] * self.proj_vectors
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return raw / norms

    def _kuramoto_step(self, stimuli: np.ndarray, J: np.ndarray) -> None:
        x = self.phases
        freq_term = np.einsum("ijk,ik->ij", self.omega, x)
        coupling = np.einsum("ijkl,jl->ik", J, x)
        raw = stimuli + coupling
        dot = np.sum(raw * x, axis=1, keepdims=True)
        projected = raw - dot * x
        delta = freq_term + projected
        x_new = x + self.step_size * delta
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.phases = x_new / norms

    def _global_order_parameter(self) -> float:
        return float(np.linalg.norm(np.mean(self.phases, axis=0)))

    def _group_order_parameters(self) -> Dict[str, float]:
        result = {}
        for name, sl in self.group_slices.items():
            group = self.phases[sl]
            result[name] = float(np.linalg.norm(np.mean(group, axis=0)))
        return result

    def _cross_coherence(self) -> Dict[str, float]:
        centroids = {}
        for name, sl in self.group_slices.items():
            centroids[name] = np.mean(self.phases[sl], axis=0)
        result = {}
        pairs = [("snn", "lsm"), ("snn", "htm"), ("lsm", "htm")]
        for a, b in pairs:
            ca, cb = centroids[a], centroids[b]
            norm_a, norm_b = np.linalg.norm(ca), np.linalg.norm(cb)
            if norm_a < 1e-8 or norm_b < 1e-8:
                result[f"{a}_{b}"] = 0.0
            else:
                result[f"{a}_{b}"] = float(np.dot(ca, cb) / (norm_a * norm_b))
        return result

    def _find_clusters(self) -> List[List[int]]:
        N = self.num_oscillators
        sim = self.phases @ self.phases.T
        visited = set()
        clusters = []
        for i in range(N):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, N):
                if j not in visited and sim[i, j] > self.sync_threshold:
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) > 1:
                clusters.append(cluster)
        for i in range(N):
            if i not in visited:
                clusters.append([i])
                visited.add(i)
        return clusters

    def _compute_salience(self, clusters: List[List[int]], global_r: float) -> Dict[str, float]:
        group_to_module = {"snn": "snn", "lsm": "lsm", "htm": "htm"}
        module_scores: Dict[str, List[float]] = {m: [] for m in group_to_module.values()}
        mean_coherence = global_r
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            cluster_phases = self.phases[cluster]
            cluster_r = float(np.linalg.norm(np.mean(cluster_phases, axis=0)))
            for idx in cluster:
                for gname, sl in self.group_slices.items():
                    if idx in range(sl.start, sl.stop) and gname in group_to_module:
                        module_scores[group_to_module[gname]].append(cluster_r)
        result = {}
        for module, scores in module_scores.items():
            if scores:
                avg_cluster_r = np.mean(scores)
                result[module] = 1.0 + self.salience_alpha * (avg_cluster_r - mean_coherence)
            else:
                result[module] = 1.0
        return result

    def _build_context_string(self, clusters, global_r, group_r, cross_r) -> str:
        parts = []
        if clusters:
            largest = max(clusters, key=len)
            if len(largest) >= 2:
                names = [OSCILLATOR_NAMES[i] for i in largest[:4]]
                cluster_r = float(np.linalg.norm(np.mean(self.phases[largest], axis=0)))
                parts.append(f"Oscillatory binding: {' + '.join(names)} synchronized (r={cluster_r:.2f})")
        for pair, val in cross_r.items():
            if val > 0.5:
                a, b = pair.split("_")
                parts.append(f"{a.upper()}-{b.upper()} coherence high ({val:.2f})")
        parts.append(f"Global binding: r={global_r:.2f}")
        return ". ".join(parts)

    def _hebbian_update(self) -> None:
        outer = np.einsum("ik,jl->ijkl", self.phases, self.phases)
        self.J_learned += self.hebbian_lr * (outer - self.J_learned)
        self.J_learned *= (1.0 - self.hebbian_decay)
        np.clip(self.J_learned, -self.hebbian_cap, self.hebbian_cap, out=self.J_learned)
        for i in range(self.num_oscillators):
            self.J_learned[i, i] = 0.0

    def _discretize_group_state(self, group_r: Dict[str, float]) -> int:
        """Bin 4 group order parameters into a single state index in [0, 80].

        Each of the 4 group order parameters (snn, lsm, htm, cross) is binned
        into 3 levels: low (0), mid (1), high (2). The resulting 4-digit base-3
        number is flattened to a single index in [0, 80].
        """
        keys = ["snn", "lsm", "htm", "cross"]
        idx = 0
        for key in keys:
            val = group_r.get(key, 0.0)
            # Bin into 3 levels: [0, 1/3) -> 0, [1/3, 2/3) -> 1, [2/3, 1] -> 2
            if val < 1.0 / 3.0:
                level = 0
            elif val < 2.0 / 3.0:
                level = 1
            else:
                level = 2
            idx = idx * 3 + level
        return int(idx)

    def _record_oscillator_transition(self, group_r: Dict[str, float]) -> None:
        """Record state transition for oscillator-level TPM estimation."""
        current_idx = self._discretize_group_state(group_r)
        self._osc_state_visit_counts[current_idx] += 1
        if self._osc_prev_state_idx is not None:
            self._osc_transition_counts[self._osc_prev_state_idx, current_idx] += 1
        self._osc_prev_state_idx = current_idx

    @property
    def oscillator_transitions_recorded(self) -> int:
        """Total number of oscillator state transitions recorded."""
        return int(self._osc_transition_counts.sum())

    @property
    def oscillator_tpm_warming_up(self) -> bool:
        """True until at least _osc_min_transitions have been recorded."""
        return self.oscillator_transitions_recorded < self._osc_min_transitions

    @property
    def oscillator_tpm_coverage(self) -> float:
        """Fraction of the 81 states that have been visited at least once."""
        visited = int(np.count_nonzero(self._osc_state_visit_counts))
        return visited / 81.0

    def build_oscillator_tpm(self) -> np.ndarray:
        """Build an (81, 81) row-stochastic TPM with Laplace smoothing.

        Each row i represents the probability of transitioning from state i to
        every other state j. Rows with zero observed transitions are treated as
        uniform (maximum-entropy prior).

        Returns:
            tpm: np.ndarray of shape (81, 81), dtype float64, rows sum to 1.
        """
        counts = self._osc_transition_counts.astype(np.float64)
        # Laplace smoothing: add 1 pseudo-count to every cell
        smoothed = counts + 1.0
        row_sums = smoothed.sum(axis=1, keepdims=True)
        tpm = smoothed / row_sums
        return tpm

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "scalars": {
                "binding_count": self._binding_count,
                "num_oscillators": self.num_oscillators,
                "phase_dim": self.phase_dim,
                "osc_prev_state_idx": self._osc_prev_state_idx,
            },
            "arrays": {
                "phases": self.phases,
                "J_learned": self.J_learned,
                "proj_vectors": self.proj_vectors,
                "omega": self.omega,
                "osc_transition_counts": self._osc_transition_counts,
                "osc_state_visit_counts": self._osc_state_visit_counts,
            },
            "histories": {
                "order_history": list(self._order_history),
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        scalars = state.get("scalars", {})
        arrays = state.get("arrays", {})
        histories = state.get("histories", {})
        self._binding_count = scalars.get("binding_count", 0)
        self._osc_prev_state_idx = scalars.get("osc_prev_state_idx", None)
        if "phases" in arrays:
            self.phases = np.array(arrays["phases"])
        if "J_learned" in arrays:
            self.J_learned = np.array(arrays["J_learned"])
        if "proj_vectors" in arrays:
            self.proj_vectors = np.array(arrays["proj_vectors"])
        if "omega" in arrays:
            self.omega = np.array(arrays["omega"])
        if "osc_transition_counts" in arrays:
            self._osc_transition_counts = np.array(
                arrays["osc_transition_counts"], dtype=np.int64
            )
        if "osc_state_visit_counts" in arrays:
            self._osc_state_visit_counts = np.array(
                arrays["osc_state_visit_counts"], dtype=np.int64
            )
        if "order_history" in histories:
            self._order_history = deque(histories["order_history"], maxlen=100)
        logger.info(
            f"OscillatoryBinding restored: {self._binding_count} prior bindings, "
            f"J_learned max={np.max(np.abs(self.J_learned)):.4f}"
        )
