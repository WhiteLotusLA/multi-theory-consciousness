"""
PhiTracker -- Architectural Phi measurement for MTC
====================================================

Tracks binary activation states across 5 consciousness modules (GWT, AST,
HOT, FEP, BLT), builds an empirical Transition Probability Matrix (TPM),
and provides a lightweight Phi approximation.

Uses hash-based probabilistic binarization: each module's continuous signal
becomes its probability of being ON per cycle. This ensures state diversity
even when underlying signals are slowly varying.

Based on the reference implementation (April 2026).
"""

import hashlib
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PhiResult:
    """Result of a Phi computation cycle."""

    phi: Optional[float]
    phi_normalized: Optional[float]
    tpm_coverage: float
    transitions_recorded: int
    warming_up: bool
    error: Optional[str]

    @classmethod
    def warming_up_result(
        cls,
        transitions_recorded: int = 0,
        tpm_coverage: float = 0.0,
    ) -> "PhiResult":
        return cls(
            phi=None,
            phi_normalized=None,
            tpm_coverage=tpm_coverage,
            transitions_recorded=transitions_recorded,
            warming_up=True,
            error=None,
        )


class PhiTracker:
    """
    Tracks consciousness module states and builds empirical TPMs.

    Observes 5 binary nodes (one per consciousness module), records state
    transitions, and constructs a TPM suitable for IIT analysis.
    """

    NODE_LABELS = ["GWT", "AST", "HOT", "FEP", "BLT"]
    N_NODES = 5
    N_STATES = 32  # 2^5

    def __init__(
        self,
        min_transitions: int = 128,
        history_size: int = 1024,
        laplace_alpha: float = 0.1,
    ):
        self.min_transitions = min_transitions
        self.history_size = history_size
        self.laplace_alpha = laplace_alpha

        self.state_history: deque = deque(maxlen=history_size)

        # Transition counts: [N_STATES, N_NODES]
        self.transition_counts = np.zeros(
            (self.N_STATES, self.N_NODES), dtype=np.float64
        )
        self.state_visit_counts = np.zeros(self.N_STATES, dtype=np.float64)

        # Connectivity matrix: cm[i][j] = 1 means node j influences node i
        self.connectivity = np.array(
            [
                [1, 1, 1, 1, 1],  # GWT: receives from all
                [1, 1, 0, 0, 0],  # AST: reads GWT + self
                [1, 1, 1, 0, 0],  # HOT: reads GWT + AST + self
                [1, 0, 1, 1, 0],  # FEP: reads GWT + HOT + self
                [0, 1, 1, 1, 1],  # BLT: reads AST + HOT + FEP + self
            ],
            dtype=np.int32,
        )

    def extract_module_states(self, consciousness_state) -> Tuple[int, ...]:
        """
        Extract binary activation state for each module using hash-based
        probabilistic binarization.

        Higher signals -> higher probability of being ON.
        Signal 0.8 -> ON ~80% of cycles. Signal 0.2 -> ON ~20%.
        """
        cs = consciousness_state

        # Extract continuous signals (0-1) from each module
        signals = {}

        # GWT
        integration = float(getattr(cs, "integration_level", 0.0))
        broadcast = float(getattr(cs, "broadcast_coverage", 0.0))
        signals["GWT"] = max(integration, broadcast)

        # AST
        protoself = getattr(cs, "protoself_stability", None)
        if protoself is not None:
            signals["AST"] = float(protoself)
        else:
            attn = getattr(cs, "attention_schema", None)
            signals["AST"] = 0.7 if attn and getattr(attn, "current_focus", None) else 0.0

        # HOT
        fok = getattr(cs, "feeling_of_knowing", None)
        if fok is not None:
            signals["HOT"] = float(fok)
        else:
            hots = getattr(cs, "higher_order_thoughts", [])
            signals["HOT"] = min(1.0, float(len(hots)) / 3.0) if hots else 0.0

        # FEP
        urgency = getattr(cs, "homeostatic_urgency", None)
        if urgency is not None and isinstance(urgency, (tuple, list)) and len(urgency) >= 2:
            signals["FEP"] = float(urgency[1])
        else:
            signals["FEP"] = min(1.0, float(getattr(cs, "prediction_error", 0.0)))

        # BLT
        binding = getattr(cs, "binding_quality", None)
        if binding is not None:
            signals["BLT"] = float(binding)
        else:
            signals["BLT"] = float(getattr(cs, "beautiful_loop_quality", 0.0))

        # Probabilistic binarization via deterministic hash
        cycle = len(self.state_history)
        result = []

        for label in self.NODE_LABELS:
            sig = np.clip(signals.get(label, 0.0), 0.0, 1.0)
            h = hashlib.md5(f"{cycle}:{label}".encode()).hexdigest()
            threshold = int(h[:8], 16) / 0xFFFFFFFF
            bit = 1 if sig > threshold else 0
            result.append(bit)

        return tuple(result)

    def _state_to_index(self, state: Tuple[int, ...]) -> int:
        """Convert binary state tuple to integer index (big-endian)."""
        index = 0
        for bit in state:
            index = (index << 1) | bit
        return index

    def record_module_states(self, consciousness_state) -> None:
        """Extract and record current module states, updating transition counts."""
        current = self.extract_module_states(consciousness_state)

        if self.state_history:
            prev = self.state_history[-1]
            prev_idx = self._state_to_index(prev)
            for node_idx in range(self.N_NODES):
                self.transition_counts[prev_idx, node_idx] += current[node_idx]
            self.state_visit_counts[prev_idx] += 1

        self.state_history.append(current)

    @property
    def transitions_recorded(self) -> int:
        return int(self.state_visit_counts.sum())

    @property
    def warming_up(self) -> bool:
        return self.transitions_recorded < self.min_transitions

    def build_tpm(self) -> np.ndarray:
        """
        Build node-factored TPM from observed transitions.

        Returns: (32, 5) array where tpm[s, i] = P(node_i=1 | source_state=s).
        Uses Laplace smoothing for unobserved states.
        """
        tpm = np.full((self.N_STATES, self.N_NODES), 0.5)

        for s in range(self.N_STATES):
            visits = self.state_visit_counts[s]
            if visits > 0:
                for i in range(self.N_NODES):
                    tpm[s, i] = (
                        (self.transition_counts[s, i] + self.laplace_alpha)
                        / (visits + 2 * self.laplace_alpha)
                    )

        return tpm

    @property
    def tpm_coverage(self) -> float:
        """Fraction of states observed at least once."""
        return float(np.sum(self.state_visit_counts > 0) / self.N_STATES)

    @property
    def latest_phi(self) -> Optional[float]:
        """
        Lightweight synchronous Phi approximation from the empirical TPM.

        Measures integration as average KL divergence between joint
        transition probabilities and marginal (independent) predictions.
        Returns None while still warming up.
        """
        if self.transitions_recorded < self.min_transitions:
            return None
        try:
            tpm = self.build_tpm()
            visits = self.state_visit_counts.copy()
            total_visits = visits.sum()
            if total_visits == 0:
                return 0.0

            weights = visits / total_visits
            marginals = np.dot(weights, tpm)
            marginals = np.clip(marginals, 0.01, 0.99)

            observed = visits > 0
            tpm_obs = np.clip(tpm[observed], 0.01, 0.99)
            w_obs = weights[observed]

            kl_per_state = np.zeros(len(w_obs))
            for idx in range(len(w_obs)):
                p = tpm_obs[idx]
                q = marginals
                kl = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
                kl_per_state[idx] = np.sum(kl)

            phi_approx = float(np.dot(w_obs, kl_per_state))
            return max(0.0, phi_approx)
        except Exception as e:
            logger.warning(f"Phi approximation failed: {e}")
            return 0.0
