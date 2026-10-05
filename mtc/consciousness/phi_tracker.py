"""
PhiTracker -- Architectural Phi measurement for MTC
====================================================

Tracks binary activation states across the consciousness modules (by default
GWT, AST, HOT, FEP, BLT), builds an empirical state-by-node Transition
Probability Matrix (TPM), and measures integration two ways:

- ``latest_phi``: a cheap per-cycle integration proxy in bits (the weighted
  KL divergence between each source state's transition row and the marginal
  prediction). It is labelled a proxy; it is not IIT Phi.
- ``compute_architectural_phi``: exact IIT 4.0 Phi (``new_big_phi.sia``) on
  the module graph, computed through ``mtc.consciousness.pyphi_bridge`` so
  the TPM rows and connectivity reach PyPhi in PyPhi's conventions.

The default 5-node connectivity reflects how information flows between the
modules (row = target, column = source):

    GWT: receives from all (global workspace)
    AST: reads GWT (attention monitors the workspace)
    HOT: reads GWT + AST (metacognition observes attention and workspace)
    FEP: reads GWT + HOT (prediction uses workspace and metacognition)
    BLT: reads AST + HOT + FEP (the beautiful loop integrates the rest)

Each module's continuous signal is binarized probabilistically with a
deterministic per-cycle hash threshold, so a module with signal 0.8 is ON in
about 80 percent of cycles. The TPM uses Laplace smoothing; rows are indexed
big-endian (the first label is the most significant bit).
"""

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from mtc.consciousness import pyphi_bridge

logger = logging.getLogger(__name__)


# ============================================================================
# PhiResult dataclass
# ============================================================================


@dataclass
class PhiResult:
    """
    Result of an architectural Phi computation.

    Holds the computed Phi (or None while warming up or on error) plus
    metadata about the computation and the TPM it was computed from.
    """

    phi: Optional[float]
    phi_normalized: Optional[float]
    distinctions_count: Optional[int]
    relations_count: Optional[int]
    partition_info: Optional[str]
    is_exact: bool
    computation_time_ms: float
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
        """A result saying the tracker has not yet seen enough transitions."""
        return cls(
            phi=None,
            phi_normalized=None,
            distinctions_count=None,
            relations_count=None,
            partition_info=None,
            is_exact=False,
            computation_time_ms=0.0,
            tpm_coverage=tpm_coverage,
            transitions_recorded=transitions_recorded,
            warming_up=True,
            error=None,
        )

    @classmethod
    def error_result(
        cls,
        error: str,
        transitions_recorded: int = 0,
        tpm_coverage: float = 0.0,
        computation_time_ms: float = 0.0,
    ) -> "PhiResult":
        """A structured failure: no Phi, the reason in ``error``."""
        return cls(
            phi=None,
            phi_normalized=None,
            distinctions_count=None,
            relations_count=None,
            partition_info=None,
            is_exact=False,
            computation_time_ms=computation_time_ms,
            tpm_coverage=tpm_coverage,
            transitions_recorded=transitions_recorded,
            warming_up=False,
            error=error,
        )


# ============================================================================
# PhiTracker
# ============================================================================


class PhiTracker:
    """
    Tracks consciousness module states and builds empirical TPMs.

    Observes one binary node per module, records state transitions, and
    constructs a TPM suitable for IIT analysis.
    """

    DEFAULT_NODE_LABELS: Tuple[str, ...] = ("GWT", "AST", "HOT", "FEP", "BLT")

    # Target-by-source connectivity among the default labels:
    # DEFAULT_CONNECTIVITY[i][j] == 1 means module j drives module i.
    DEFAULT_CONNECTIVITY = np.array(
        [
            [1, 1, 1, 1, 1],  # GWT: self-loop + receives from all
            [1, 1, 0, 0, 0],  # AST: reads GWT + self
            [1, 1, 1, 0, 0],  # HOT: reads GWT + AST + self
            [1, 0, 1, 1, 0],  # FEP: reads GWT + HOT + self
            [0, 1, 1, 1, 1],  # BLT: reads AST + HOT + FEP + self
        ],
        dtype=np.int32,
    )

    # Rolling phi history used by normalize_phi.
    _PHI_HISTORY_SIZE = 200
    _PHI_NORM_PERCENTILE = 95
    _PHI_NORM_MIN_HISTORY = 20

    # Largest module count the tracker accepts (2**n TPM rows).
    _MAX_MODULES = 16

    def __init__(
        self,
        num_modules: int = 5,
        history_size: int = 1024,
        phi_scale: float = 1.0,
        phi_clamp: Optional[float] = None,
        module_labels: Optional[Sequence[str]] = None,
        min_transitions: int = 128,
        laplace_alpha: float = 0.1,
        phi_warmup_reference: float = 5.0,
        connectivity: Optional[np.ndarray] = None,
    ):
        """
        Initialize the PhiTracker.

        Args:
            num_modules: Number of tracked modules. Without ``module_labels``
                the first ``num_modules`` default labels are used.
            history_size: Maximum number of states kept in the history deque.
            phi_scale: Multiplier applied to ``latest_phi`` (raw bits at 1.0).
            phi_clamp: Optional upper bound on ``latest_phi`` after scaling.
            module_labels: Explicit node labels; their count sets the module
                count. A non-default ``num_modules`` that disagrees raises.
            min_transitions: Transitions required before any Phi is reported.
            laplace_alpha: Laplace smoothing for the TPM. Higher values pull
                sparsely observed rows toward 0.5.
            phi_warmup_reference: Display calibration for ``normalize_phi``:
                the denominator used before enough phi values have been
                recorded to take a percentile. It shapes the normalized
                readout during warm-up only, never the raw measurement.
            connectivity: Target-by-source connectivity, shape (n, n). The
                default is the architectural matrix restricted to the chosen
                default labels, or full connectivity for custom labels.
        """
        if module_labels is not None:
            labels = [str(label) for label in module_labels]
            if num_modules != 5 and num_modules != len(labels):
                raise ValueError(
                    f"num_modules={num_modules} disagrees with "
                    f"{len(labels)} module_labels"
                )
        else:
            if not 1 <= num_modules <= len(self.DEFAULT_NODE_LABELS):
                raise ValueError(
                    f"num_modules={num_modules} needs explicit module_labels "
                    f"beyond the {len(self.DEFAULT_NODE_LABELS)} defaults"
                )
            labels = list(self.DEFAULT_NODE_LABELS[:num_modules])
        if not 1 <= len(labels) <= self._MAX_MODULES:
            raise ValueError(f"between 1 and {self._MAX_MODULES} modules supported")
        if len(set(labels)) != len(labels):
            raise ValueError(f"module labels must be unique, got {labels}")

        self.NODE_LABELS = labels
        self.N_NODES = len(labels)
        self.N_STATES = 2**self.N_NODES

        self.min_transitions = min_transitions
        self.history_size = history_size
        self.laplace_alpha = laplace_alpha
        self.phi_scale = float(phi_scale)
        self.phi_clamp = None if phi_clamp is None else float(phi_clamp)
        self.phi_warmup_reference = float(phi_warmup_reference)
        if self.phi_warmup_reference <= 0.0:
            raise ValueError("phi_warmup_reference must be positive")

        self.connectivity = self._resolve_connectivity(connectivity)

        # Bounded history of binary state tuples.
        self.state_history: deque = deque(maxlen=history_size)

        # Rolling history of observed phi values for normalize_phi.
        self._phi_history: deque = deque(maxlen=self._PHI_HISTORY_SIZE)

        # Monotonic cycle counter for the per-cycle hash thresholds. It must
        # not depend on len(state_history), which stops growing at
        # history_size and would freeze every threshold from then on.
        self._cycle_counter: int = 0

        # transition_counts[s, i]: how often node i was ON in the successor
        # of source state s. state_visit_counts[s]: how often s was a source.
        self.transition_counts = np.zeros(
            (self.N_STATES, self.N_NODES), dtype=np.float64
        )
        self.state_visit_counts = np.zeros(self.N_STATES, dtype=np.float64)

    def _resolve_connectivity(self, connectivity: Optional[np.ndarray]) -> np.ndarray:
        """Pick or validate the target-by-source connectivity matrix."""
        n = self.N_NODES
        if connectivity is not None:
            cm = np.asarray(connectivity, dtype=np.int32)
            if cm.shape != (n, n):
                raise ValueError(
                    f"connectivity shape {cm.shape} does not match {n} modules"
                )
            return cm.copy()
        defaults = list(self.DEFAULT_NODE_LABELS)
        if all(label in defaults for label in self.NODE_LABELS):
            idx = [defaults.index(label) for label in self.NODE_LABELS]
            return self.DEFAULT_CONNECTIVITY[np.ix_(idx, idx)].copy()
        return np.ones((n, n), dtype=np.int32)

    # ========================================================================
    # State extraction from ConsciousnessState
    # ========================================================================

    @staticmethod
    def _module_signals(cs: Any) -> Dict[str, float]:
        """Continuous 0 to 1 signal per default module, read via getattr."""
        signals: Dict[str, float] = {}

        # GWT: integration_level when the neural signals are wired. When it
        # stays at 0.0, fall back to a composite of is_conscious and the
        # ignition count, both of which vary per cycle. broadcast_coverage is
        # deliberately not used: it is a success rate that can read 1.0 while
        # the workspace is idle, which would pin the GWT bit on.
        integration = float(getattr(cs, "integration_level", 0.0) or 0.0)
        if integration > 0.0:
            signals["GWT"] = min(integration, 1.0)
        else:
            is_conscious = bool(getattr(cs, "is_conscious", False))
            ignitions = float(getattr(cs, "ignition_events", 0) or 0)
            if not is_conscious and ignitions == 0:
                signals["GWT"] = 0.0
            else:
                # 0.3 baseline (workspace active) + 0.4 ignited + up to 0.3
                # for an ignition burst.
                signals["GWT"] = (
                    0.3
                    + (0.4 if is_conscious else 0.0)
                    + 0.3 * min(1.0, ignitions / 3.0)
                )

        # AST: protoself stability, else presence of an attention focus.
        protoself = getattr(cs, "protoself_stability", None)
        if protoself is not None:
            signals["AST"] = float(protoself)
        else:
            attn = getattr(cs, "attention_schema", None)
            signals["AST"] = (
                0.7 if attn and getattr(attn, "current_focus", None) else 0.0
            )

        # HOT: feeling of knowing, else the higher-order thought count.
        fok = getattr(cs, "feeling_of_knowing", None)
        if fok is not None:
            signals["HOT"] = float(fok)
        else:
            hots = getattr(cs, "higher_order_thoughts", None) or []
            signals["HOT"] = min(1.0, float(len(hots)) / 3.0)

        # FEP: homeostatic urgency, else prediction error.
        urgency = getattr(cs, "homeostatic_urgency", None)
        if isinstance(urgency, (tuple, list)) and len(urgency) >= 2:
            signals["FEP"] = float(urgency[1])
        else:
            signals["FEP"] = min(1.0, float(getattr(cs, "prediction_error", 0.0)))

        # BLT: binding quality, else the loop quality.
        binding = getattr(cs, "binding_quality", None)
        if binding is not None:
            signals["BLT"] = float(binding)
        else:
            signals["BLT"] = float(getattr(cs, "beautiful_loop_quality", 0.0))

        return signals

    def extract_module_states(self, consciousness_state: Any) -> Tuple[int, ...]:
        """
        Binarize each module's signal with a deterministic per-cycle threshold.

        The threshold for (cycle, label) comes from a hash, so a module with
        signal 0.8 is ON in about 80 percent of cycles, signal 0.0 is always
        OFF and signal 1.0 always ON. This creates the state diversity a
        meaningful TPM needs while keeping signal strength causally tied to
        the ON probability. Labels outside the default set read signal 0.0.

        Extraction does not advance the cycle counter, so repeated calls
        between records return the same tuple.

        Returns:
            Tuple of N_NODES ints (0 or 1), in NODE_LABELS order.
        """
        signals = self._module_signals(consciousness_state)
        cycle = self._cycle_counter
        result = []
        for label in self.NODE_LABELS:
            sig = float(np.clip(signals.get(label, 0.0), 0.0, 1.0))
            digest = hashlib.md5(
                f"{cycle}:{label}".encode(), usedforsecurity=False
            ).hexdigest()
            threshold = int(digest[:8], 16) / 0xFFFFFFFF
            result.append(1 if sig > threshold else 0)
        return tuple(result)

    def _state_to_index(self, state: Sequence[int]) -> int:
        """
        Convert a binary state tuple to its big-endian row index.

        The first label is the most significant bit: for the default labels
        (1,0,0,0,0) -> 16, (0,0,0,0,1) -> 1, (1,1,1,1,1) -> 31.
        """
        index = 0
        for bit in state:
            index = (index << 1) | int(bit)
        return index

    def record_module_states(self, consciousness_state: Any) -> None:
        """
        Extract and record the current module states.

        Updates the state history and, when a previous state exists, the
        transition counts from that state into this one. Advances the cycle
        counter afterwards.
        """
        current = self.extract_module_states(consciousness_state)

        if self.state_history:
            prev_idx = self._state_to_index(self.state_history[-1])
            self.transition_counts[prev_idx] += np.asarray(current, dtype=np.float64)
            self.state_visit_counts[prev_idx] += 1

        self.state_history.append(current)
        self._cycle_counter += 1

    @property
    def transitions_recorded(self) -> int:
        """Total transitions recorded (sum of state visit counts)."""
        return int(self.state_visit_counts.sum())

    @property
    def warming_up(self) -> bool:
        """Whether the tracker is still collecting its initial transitions."""
        return self.transitions_recorded < self.min_transitions

    # ========================================================================
    # Integration proxy and phi normalization
    # ========================================================================

    @property
    def latest_phi(self) -> Optional[float]:
        """
        Per-cycle integration proxy in bits (not IIT Phi).

        For each observed source state s and node i, the KL divergence (base
        2) between the transition probability TPM[s, i] and the node's
        visit-weighted marginal, summed over nodes and weighted by P(s). It
        measures how much knowing the whole system state matters for
        predicting each node's next state. The value is multiplied by
        ``phi_scale`` and capped at ``phi_clamp`` when one is set.

        Returns None while warming up, when no transition has been seen, or
        when the computation fails (the failure is logged as a warning).
        Use ``compute_architectural_phi`` for exact IIT 4.0 Phi.
        """
        if self.warming_up:
            return None
        try:
            visits = self.state_visit_counts
            total_visits = float(visits.sum())
            if total_visits <= 0.0:
                return None
            tpm = self.build_tpm()
            weights = visits / total_visits
            marginals = np.clip(weights @ tpm, 0.01, 0.99)

            observed = visits > 0
            tpm_obs = np.clip(tpm[observed], 0.01, 0.99)
            w_obs = weights[observed]
            kl = tpm_obs * np.log2(tpm_obs / marginals) + (1.0 - tpm_obs) * np.log2(
                (1.0 - tpm_obs) / (1.0 - marginals)
            )
            value = float(w_obs @ kl.sum(axis=1)) * self.phi_scale
            value = max(0.0, value)
            if self.phi_clamp is not None:
                value = min(value, self.phi_clamp)
            return value
        except Exception as exc:
            logger.warning("PhiTracker integration proxy failed: %s", exc)
            return None

    def record_phi(self, value: Optional[float]) -> None:
        """Record an observed phi value for ``normalize_phi``.

        None, NaN, negative and non-numeric values are ignored.
        """
        if value is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(v) or v < 0.0:
            return
        self._phi_history.append(v)

    def normalize_phi(self, value: Optional[float]) -> Optional[float]:
        """Normalize a phi value to [0, 1].

        The reference is the rolling 95th percentile of recorded phi values.
        Until enough values are recorded, ``phi_warmup_reference`` stands in.
        A missing or non-numeric value stays missing (None).
        """
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(v):
            return None
        v = max(0.0, v)
        if len(self._phi_history) < self._PHI_NORM_MIN_HISTORY:
            return min(1.0, v / self.phi_warmup_reference)
        ref = float(np.percentile(list(self._phi_history), self._PHI_NORM_PERCENTILE))
        if ref < 1e-9:
            return 0.0
        return max(0.0, min(1.0, v / ref))

    # ========================================================================
    # TPM construction with Laplace smoothing
    # ========================================================================

    def get_tpm_coverage(self) -> float:
        """Fraction of TPM rows (source states) observed at least once."""
        observed = np.sum(self.state_visit_counts > 0)
        return float(observed / self.N_STATES)

    @property
    def tpm_coverage(self) -> float:
        """Alias of ``get_tpm_coverage()``."""
        return self.get_tpm_coverage()

    def build_tpm(self) -> np.ndarray:
        """
        Build the empirical state-by-node TPM with Laplace smoothing.

            P(node_i = 1 | state = s) = (count_i + alpha) / (total + 2 * alpha)

        Unvisited rows reduce to 0.5. With alpha = 0 an unvisited row would
        divide by zero, so it falls back to 0.5 explicitly.

        Returns:
            Array of shape (N_STATES, N_NODES), big-endian rows, values in
            [0, 1].
        """
        alpha = self.laplace_alpha
        denominator = self.state_visit_counts + 2.0 * alpha
        tpm = np.full((self.N_STATES, self.N_NODES), 0.5, dtype=np.float64)
        valid = denominator > 0.0
        tpm[valid] = (self.transition_counts[valid] + alpha) / denominator[
            valid, np.newaxis
        ]
        return tpm

    # ========================================================================
    # Exact architectural Phi (IIT 4.0) through the PyPhi bridge
    # ========================================================================

    async def compute_architectural_phi(
        self, current_state: Optional[Sequence[int]] = None
    ) -> PhiResult:
        """
        Compute exact IIT 4.0 Phi of the module graph.

        Builds a PyPhi network from the empirical TPM and the connectivity
        matrix through the bridge (which converts both to PyPhi's
        conventions), takes the subsystem at ``current_state`` and runs the
        system irreducibility analysis. The analysis runs inline: PyPhi's
        configuration override and warning filters are process-global, so
        it is not moved to a worker thread.

        Args:
            current_state: Binary tuple of length N_NODES; defaults to the
                most recently recorded state.

        Returns:
            A warming-up result before ``min_transitions``; a structured
            error (``error`` set, ``phi`` None) when PyPhi is missing, the
            state is invalid or the analysis raises; otherwise the clamped
            non-negative Phi with partition info and timing.
        """
        coverage = self.get_tpm_coverage()
        transitions = self.transitions_recorded

        if self.warming_up:
            return PhiResult.warming_up_result(
                transitions_recorded=transitions, tpm_coverage=coverage
            )

        if not pyphi_bridge.pyphi_available():
            return PhiResult.error_result(
                "PyPhi is not installed",
                transitions_recorded=transitions,
                tpm_coverage=coverage,
            )

        if current_state is None:
            if not self.state_history:
                return PhiResult.error_result(
                    "no recorded state to evaluate",
                    transitions_recorded=transitions,
                    tpm_coverage=coverage,
                )
            current_state = self.state_history[-1]
        state = tuple(int(bit) for bit in current_state)
        if len(state) != self.N_NODES or any(bit not in (0, 1) for bit in state):
            return PhiResult.error_result(
                f"state must be {self.N_NODES} binary values, got {state}",
                transitions_recorded=transitions,
                tpm_coverage=coverage,
            )

        t0 = time.perf_counter()
        try:
            tpm = self.build_tpm()
            network = pyphi_bridge.build_network(
                tpm, self.connectivity, self.NODE_LABELS
            )
            subsystem = pyphi_bridge.build_subsystem(network, state)
            sia_result = pyphi_bridge.system_irreducibility(subsystem)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Phi is non-negative by definition; PyPhi can return small
            # negatives from floating-point noise on near-disconnected systems.
            phi_val = max(0.0, float(sia_result.phi))
            normalized = getattr(sia_result, "normalized_phi", None)
            norm_phi = None if normalized is None else max(0.0, float(normalized))
            partition = getattr(sia_result, "partition", None)
            partition_str = str(partition) if partition else None
            distinctions, relations = self._phi_structure_counts(sia_result)

            return PhiResult(
                phi=phi_val,
                phi_normalized=norm_phi,
                distinctions_count=distinctions,
                relations_count=relations,
                partition_info=partition_str,
                is_exact=True,
                computation_time_ms=elapsed_ms,
                tpm_coverage=coverage,
                transitions_recorded=transitions,
                warming_up=False,
                error=None,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            error_name = type(exc).__name__
            logger.warning("PyPhi SIA failed (%s): %s", error_name, exc)
            return PhiResult.error_result(
                f"{error_name}: {exc}",
                transitions_recorded=transitions,
                tpm_coverage=coverage,
                computation_time_ms=elapsed_ms,
            )

    @staticmethod
    def _phi_structure_counts(sia_result: Any) -> Tuple[Optional[int], Optional[int]]:
        """Distinction and relation counts, or (None, None) without them.

        A system irreducibility result does not carry a phi structure; only
        a result type that exposes both ``distinctions`` and ``relations``
        (the installed PyPhi's ``PhiStructure``) yields counts.
        """
        distinctions = getattr(sia_result, "distinctions", None)
        relations = getattr(sia_result, "relations", None)
        if distinctions is None or relations is None:
            return None, None
        try:
            return len(distinctions), len(relations)
        except TypeError:  # exposed but not countable
            return None, None

    # ========================================================================
    # Persistence
    # ========================================================================

    def to_state_dict(self) -> dict:
        """Serialize tracker state as {"scalars", "arrays", "histories"}."""
        return {
            "scalars": {
                "transitions_recorded": self.transitions_recorded,
                "min_transitions": self.min_transitions,
                "history_size": self.history_size,
                "laplace_alpha": self.laplace_alpha,
                "cycle_counter": self._cycle_counter,
                "n_nodes": self.N_NODES,
            },
            "arrays": {
                "transition_counts": self.transition_counts.copy(),
                "state_visit_counts": self.state_visit_counts.copy(),
            },
            "histories": {
                "state_history": [list(s) for s in self.state_history],
                "phi_history": list(self._phi_history),
            },
        }

    def from_state_dict(self, state: dict) -> None:
        """
        Restore tracker state produced by ``to_state_dict``.

        Arrays and history entries whose shape does not match this tracker's
        module count are skipped with a warning. The output calibration
        (``phi_scale``, ``phi_clamp``, ``phi_warmup_reference``) stays as
        constructed.
        """
        s = state.get("scalars", {})
        a = state.get("arrays", {})
        h = state.get("histories", {})

        self.min_transitions = int(s.get("min_transitions", self.min_transitions))
        self.history_size = int(s.get("history_size", self.history_size))
        self.laplace_alpha = float(s.get("laplace_alpha", self.laplace_alpha))
        self._cycle_counter = int(s.get("cycle_counter", 0))

        if "transition_counts" in a:
            arr = np.asarray(a["transition_counts"], dtype=np.float64)
            if arr.shape == (self.N_STATES, self.N_NODES):
                self.transition_counts = arr.copy()
            else:
                logger.warning(
                    "PhiTracker: transition_counts shape %s does not match %s, "
                    "skipping",
                    arr.shape,
                    (self.N_STATES, self.N_NODES),
                )

        if "state_visit_counts" in a:
            arr = np.asarray(a["state_visit_counts"], dtype=np.float64)
            if arr.shape == (self.N_STATES,):
                self.state_visit_counts = arr.copy()
            else:
                logger.warning(
                    "PhiTracker: state_visit_counts shape %s does not match %s, "
                    "skipping",
                    arr.shape,
                    (self.N_STATES,),
                )

        self.state_history = deque(maxlen=self.history_size)
        for entry in h.get("state_history", []):
            bits = tuple(int(b) for b in entry)
            if len(bits) == self.N_NODES and all(b in (0, 1) for b in bits):
                self.state_history.append(bits)
            else:
                logger.warning(
                    "PhiTracker: skipping state history entry of length %d",
                    len(bits),
                )

        self._phi_history = deque(maxlen=self._PHI_HISTORY_SIZE)
        for value in h.get("phi_history", []):
            self.record_phi(value)

    # ========================================================================
    # Neural integration (spectral approximation)
    # ========================================================================

    def get_neural_integration(
        self,
        snn_state: np.ndarray,
        lsm_state: np.ndarray,
        htm_state: np.ndarray,
    ) -> float:
        """
        Estimate neural integration from the spectrum of cross-correlations.

        Coarse-grains three substrate state vectors into three macro-nodes,
        computes their 3x3 correlation matrix and returns ``log(3)`` minus the
        spectral entropy of its eigenvalues. Correlated substrates concentrate
        variance in fewer eigenvalues (lower entropy), so a higher value means
        more integrated; independent substrates spread variance evenly and
        score near 0.

        Args:
            snn_state: 1-D SNN activations.
            lsm_state: 1-D LSM reservoir activations.
            htm_state: 1-D HTM column activations.

        Returns:
            Non-negative float in [0, log(3)]. 0.0 for fewer than two common
            samples, constant inputs, or any numerical failure.
        """
        try:
            max_samples = 50
            snn = np.asarray(snn_state, dtype=np.float64).ravel()[:max_samples]
            lsm = np.asarray(lsm_state, dtype=np.float64).ravel()[:max_samples]
            htm = np.asarray(htm_state, dtype=np.float64).ravel()[:max_samples]

            # Truncate to the common length.
            n = min(len(snn), len(lsm), len(htm))
            if n < 2:
                return 0.0
            stacked = np.vstack([snn[:n], lsm[:n], htm[:n]])

            # Constant signals have zero variance and no defined correlation.
            if np.any(np.std(stacked, axis=1) == 0.0):
                return 0.0
            corr = np.corrcoef(stacked)
            if np.any(np.isnan(corr)):
                return 0.0

            eigenvalues = np.abs(np.linalg.eigvalsh(corr))
            total = eigenvalues.sum()
            if total == 0.0:
                return 0.0
            probs = eigenvalues / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))

            max_entropy = np.log(len(eigenvalues))
            return max(0.0, float(max_entropy - entropy))
        except Exception as exc:
            logger.warning("Neural integration computation failed: %s", exc)
            return 0.0
