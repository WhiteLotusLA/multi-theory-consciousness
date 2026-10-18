"""
Causal Emergence Analyzer - Effective Information (Hoel 2013) & CE 2.0 (2025)
==============================================================================

Computes Effective Information across consciousness modules to test
whether macro-level descriptions carry genuine causal power. Exhaustively
searches all 52 Bell partitions of 5 modules (GWT, AST, HOT, FEP, BLT).

Primary analysis (module-level):
  - Reads PhiTracker's empirical TPM (32x5 node-factored)
  - Reorders its rows to little-endian when they arrive big-endian (the
    PhiTracker's convention; see ``tpm_row_order``)
  - Converts to state-to-state TPM (32x32)
  - Computes EI at micro (32 states) and all 52 macro scales
  - Finds optimal macro partition where EI peaks

Secondary analysis (oscillator-level):
  - Uses OscillatoryBinding's transition counts (81x81)
  - Computes EI at per-group (81 states) and global (3 states) scales

Row order: the functions below read bit i of a state index as node i
(little-endian). The PhiTracker builds its TPM with big-endian rows (the first
label is the most significant bit), so the analyzer reorders them first by
default. Micro EI does not depend on row order; the macro coarse-graining
does, because it groups source states by node membership.

References:
  - Hoel, Albantakis & Tononi (2013). PNAS.
  - Hoel (2025). "Causal Emergence 2.0." arXiv:2503.13395.

Persistence: ``to_state_dict`` writes the analysis count and the last
module-level result (its scalars, the optimal partition as sorted integer
lists and the computation time). ``from_state_dict`` rebuilds that result
when every numeric field is a finite real, without the partition table, and
the research report marks it ``restored``; a record with a missing or
non-finite field restores no result and logs one WARNING.
"""

import logging
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mtc.consciousness import pyphi_bridge

logger = logging.getLogger(__name__)

TPM_ROW_ORDERS = ("big_endian", "little_endian")

# Module labels matching PhiTracker.NODE_LABELS
MODULE_LABELS = ["GWT", "AST", "HOT", "FEP", "BLT"]
N_MODULES = 5
N_MICRO_STATES = 32  # 2^5


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) in bits. Handles zeros gracefully."""
    p = np.asarray(p, dtype=np.float64)
    # Use masked log to avoid log(0); terms with p=0 contribute 0 to entropy
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log2(p[mask])
    return float(-np.sum(result))


def compute_ei(tpm: np.ndarray) -> float:
    """Compute Effective Information for a state-to-state TPM.

    EI = determinism + specificity (in bits).

    Args:
        tpm: (n, n) transition probability matrix, rows sum to 1.

    Returns:
        EI value in bits. Always >= 0.
    """
    n = tpm.shape[0]
    if n <= 1:
        return 0.0

    log_n = np.log2(n)

    # Determinism: log2(n) - average row entropy
    row_entropies = np.array([shannon_entropy(tpm[s]) for s in range(n)])
    determinism = log_n - np.mean(row_entropies)

    # Specificity: log2(n) - entropy of marginal effect distribution
    marginal = np.mean(tpm, axis=0)  # uniform interventions
    specificity = log_n - shannon_entropy(marginal)

    return max(0.0, determinism + specificity)


def node_tpm_to_state_tpm(node_tpm: np.ndarray) -> np.ndarray:
    """Convert node-factored TPM (N_states, N_nodes) to state-to-state (N_states, N_states).

    Assumes conditional independence of nodes given source state:
      P(s' | s) = prod_i P(node_i = s'_i | s)

    Args:
        node_tpm: (n_states, n_nodes) where entry [s, i] = P(node_i = ON | source = s)

    Returns:
        (n_states, n_states) state-to-state TPM, rows sum to 1.
    """
    n_states, n_nodes = node_tpm.shape
    state_tpm = np.ones((n_states, n_states), dtype=np.float64)

    for i in range(n_nodes):
        p_on = node_tpm[:, i]  # (n_states,) - P(node_i ON | each source)
        p_off = 1.0 - p_on

        # For each target state, check if node i is ON (bit i set)
        for s_prime in range(n_states):
            bit = (s_prime >> i) & 1
            if bit:
                state_tpm[:, s_prime] *= p_on
            else:
                state_tpm[:, s_prime] *= p_off

    return state_tpm


def generate_bell_partitions(n: int) -> List[List[set]]:
    """Generate all Bell partitions of {0, 1, ..., n-1}.

    Uses recursive algorithm. Bell(5) = 52.

    Returns:
        List of partitions, where each partition is a list of sets.
    """
    if n == 0:
        return [[]]

    elements = list(range(n))
    result = []

    def _partition(remaining, current_partition):
        if not remaining:
            result.append([set(group) for group in current_partition])
            return

        elem = remaining[0]
        rest = remaining[1:]

        # Add to existing group
        for i, group in enumerate(current_partition):
            group.append(elem)
            _partition(rest, current_partition)
            group.pop()

        # Start new group
        current_partition.append([elem])
        _partition(rest, current_partition)
        current_partition.pop()

    _partition(elements, [])
    return result


def build_macro_tpm(
    micro_tpm: np.ndarray,
    partition: List[set],
    n_micro_nodes: int,
) -> np.ndarray:
    """Build macro TPM from micro state-to-state TPM and a node partition.

    Macro binarization rule: a macro node (group) is ON if ANY member is ON (OR).

    Args:
        micro_tpm: (n_micro_states, n_micro_states) state-to-state TPM
        partition: List of sets, each set contains node indices in that macro group
        n_micro_nodes: Number of micro nodes (for bit extraction)

    Returns:
        (n_macro_states, n_macro_states) macro TPM, rows sum to 1.
    """
    n_micro = micro_tpm.shape[0]
    k = len(partition)  # number of macro nodes
    n_macro = 2**k

    # Build micro-to-macro state mapping
    def micro_to_macro(micro_state: int) -> int:
        macro_state = 0
        for g, group in enumerate(partition):
            # Macro node g is ON if any member node is ON
            for node in group:
                if (micro_state >> node) & 1:
                    macro_state |= 1 << g
                    break
        return macro_state

    # Group micro states by their macro state
    macro_members = [[] for _ in range(n_macro)]
    for s in range(n_micro):
        macro_members[micro_to_macro(s)].append(s)

    # Build macro TPM
    macro_tpm = np.zeros((n_macro, n_macro), dtype=np.float64)

    for mi in range(n_macro):
        members = macro_members[mi]
        if not members:
            # Unreachable macro state - uniform (Laplace)
            macro_tpm[mi, :] = 1.0 / n_macro
            continue

        for mj in range(n_macro):
            target_members = macro_members[mj]
            if not target_members:
                continue
            # Average over source microstates, sum over target microstates
            prob = 0.0
            for s in members:
                for s_prime in target_members:
                    prob += micro_tpm[s, s_prime]
            macro_tpm[mi, mj] = prob / len(members)

    return macro_tpm


def partition_to_label(partition: List[set]) -> str:
    """Human-readable label for a partition of module indices."""
    labels = MODULE_LABELS
    parts = []
    for group in sorted(partition, key=lambda g: min(g)):
        names = [labels[i] for i in sorted(group)]
        parts.append("+".join(names))
    return " | ".join(parts)


def compute_causal_primitives(tpm: np.ndarray) -> float:
    """Compute Causal Primitives (CE 2.0, Hoel 2025).

    CP = determinism_cp + specificity_cp - 1

    Where:
        determinism_cp = 1 - H(effects | cause) / log2(n)
        specificity_cp = 1 - H(causes | effect) / log2(n)
    """
    n = tpm.shape[0]
    if n <= 1:
        return 0.0

    log_n = np.log2(n)

    # Determinism: 1 - avg conditional entropy / log2(n)
    row_entropies = np.array([shannon_entropy(tpm[s]) for s in range(n)])
    determinism_cp = 1.0 - np.mean(row_entropies) / log_n

    # Specificity: need P(cause | effect)
    # Under uniform interventions P(cause) = 1/n:
    # P(cause=s | effect=s') = P(s'|s) * P(s) / P(s') = tpm[s, s'] / (n * marginal[s'])
    marginal = np.mean(tpm, axis=0)  # P(effect) under uniform cause
    col_entropies = []
    for s_prime in range(n):
        if marginal[s_prime] < 1e-10:
            col_entropies.append(0.0)
            continue
        # P(cause | effect = s') for each cause
        p_cause_given_effect = tpm[:, s_prime] / (n * marginal[s_prime])
        p_cause_given_effect = np.clip(p_cause_given_effect, 1e-10, 1.0)
        # Normalize (should sum to 1 but clip may have shifted)
        p_cause_given_effect /= p_cause_given_effect.sum()
        col_entropies.append(shannon_entropy(p_cause_given_effect))

    specificity_cp = 1.0 - np.mean(col_entropies) / log_n

    return determinism_cp + specificity_cp - 1.0


@dataclass
class CausalEmergenceResult:
    """Result of a causal emergence analysis."""

    micro_ei: float
    optimal_macro_ei: float
    optimal_partition: List[set]
    optimal_partition_label: str
    causal_emergence: float  # CE = optimal_macro_ei - micro_ei
    ce2_delta_cp: float  # CE 2.0 causal primitives delta
    all_partitions: List[Dict[str, Any]]  # sorted by EI
    phi_comparison: Optional[float]
    computation_time_ms: Optional[float]
    warming_up: bool = False
    restored: bool = False


# Fields of a persisted result that must be finite reals for a restore.
_RESTORED_REAL_FIELDS = (
    "micro_ei",
    "optimal_macro_ei",
    "causal_emergence",
    "ce2_delta_cp",
)


def _finite_real(value: Any) -> Optional[float]:
    """A finite real number as float, else None (bools are not numbers)."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _partition_lists(partition: Optional[List[set]]) -> List[List[int]]:
    """A partition as sorted lists of plain integers, in a stable order."""
    groups = [sorted(int(node) for node in group) for group in (partition or [])]
    return sorted(groups)


def _partition_from_lists(value: Any) -> Optional[List[set]]:
    """The saved partition lists as sets, None when they are malformed."""
    if not isinstance(value, list):
        return None
    partition = []
    for group in value:
        if not isinstance(group, list) or not group:
            return None
        if not all(
            isinstance(node, int) and not isinstance(node, bool) and node >= 0
            for node in group
        ):
            return None
        partition.append(set(group))
    return partition


class CausalEmergenceAnalyzer:
    """Analyzes causal emergence across consciousness modules.

    Exhaustively searches all 52 Bell partitions of 5 modules to find
    the macro scale where Effective Information peaks.
    """

    def __init__(self, tpm_row_order: str = "big_endian"):
        """
        Args:
            tpm_row_order: Row order of the node-factored TPM the tracker
                returns. ``big_endian`` (default, the PhiTracker's convention)
                reorders the rows to little-endian before the analysis;
                ``little_endian`` uses the rows as given, the v0.4.0 path.
        """
        if tpm_row_order not in TPM_ROW_ORDERS:
            raise ValueError(
                f"tpm_row_order must be one of {TPM_ROW_ORDERS}, "
                f"got {tpm_row_order!r}"
            )
        self.tpm_row_order = tpm_row_order
        self._partitions = generate_bell_partitions(N_MODULES)
        self._last_result: Optional[CausalEmergenceResult] = None
        self._last_oscillator_result: Optional[Dict[str, Any]] = None
        self._analysis_count = 0

        logger.info(
            f"CausalEmergenceAnalyzer initialized: "
            f"{len(self._partitions)} partitions to search"
        )

    def analyze_modules(self, phi_tracker) -> Optional[CausalEmergenceResult]:
        """Run full causal emergence analysis on module-level TPM.

        Args:
            phi_tracker: PhiTracker instance (reads its TPM)

        Returns:
            CausalEmergenceResult, or None if still warming up.
        """
        if phi_tracker.warming_up:
            return None

        start = time.perf_counter()

        # Step 1: Get and convert TPM
        node_tpm = phi_tracker.build_tpm()  # (32, 5)
        if self.tpm_row_order == "big_endian":
            # Pure numpy reorder; bit reversal maps big-endian to little-endian
            node_tpm = pyphi_bridge.to_little_endian_rows(node_tpm)
        state_tpm = node_tpm_to_state_tpm(node_tpm)  # (32, 32)

        # Step 2: Compute micro EI
        micro_ei = compute_ei(state_tpm)
        micro_cp = compute_causal_primitives(state_tpm)

        # Step 3: Search all 52 partitions
        partition_results = []
        for partition in self._partitions:
            n_groups = len(partition)
            if n_groups == N_MODULES:
                # Identity partition - same as micro
                ei = micro_ei
                cp = micro_cp
            elif n_groups == 1:
                # Fully merged - compute on 2-state TPM
                macro_tpm = build_macro_tpm(state_tpm, partition, N_MODULES)
                ei = compute_ei(macro_tpm)
                cp = compute_causal_primitives(macro_tpm)
            else:
                macro_tpm = build_macro_tpm(state_tpm, partition, N_MODULES)
                ei = compute_ei(macro_tpm)
                cp = compute_causal_primitives(macro_tpm)

            partition_results.append(
                {
                    "partition": partition,
                    "label": partition_to_label(partition),
                    "n_groups": n_groups,
                    "n_macro_states": 2**n_groups,
                    "ei": ei,
                    "cp": cp,
                }
            )

        # Step 4: Find optimal
        partition_results.sort(key=lambda r: r["ei"], reverse=True)
        best = partition_results[0]

        ce = best["ei"] - micro_ei
        delta_cp = best["cp"] - micro_cp

        elapsed = (time.perf_counter() - start) * 1000
        self._analysis_count += 1

        result = CausalEmergenceResult(
            micro_ei=micro_ei,
            optimal_macro_ei=best["ei"],
            optimal_partition=best["partition"],
            optimal_partition_label=best["label"],
            causal_emergence=ce,
            ce2_delta_cp=delta_cp,
            all_partitions=partition_results,
            phi_comparison=phi_tracker.latest_phi,
            computation_time_ms=elapsed,
        )

        self._last_result = result
        logger.info(
            f"Causal emergence analysis #{self._analysis_count}: "
            f"micro_EI={micro_ei:.3f}, optimal_macro_EI={best['ei']:.3f}, "
            f"CE={ce:.3f}, optimal={best['label']}, {elapsed:.1f}ms"
        )

        return result

    def analyze_oscillators(self, oscillatory_binding) -> Optional[Dict[str, Any]]:
        """Run causal emergence analysis on oscillator-level TPM.

        Note: This analysis is at a DIFFERENT granularity than module-level.
        The "micro" here is per-substrate-group order parameters (already
        coarse-grained from 30 oscillators). The two CE values are not
        directly comparable.

        Args:
            oscillatory_binding: OscillatoryBinding instance

        Returns:
            Dict with oscillator-level EI/CE, or None if warming up.
        """
        if oscillatory_binding.oscillator_tpm_warming_up:
            return None

        osc_tpm = oscillatory_binding.build_oscillator_tpm()  # (81, 81)
        micro_ei = compute_ei(osc_tpm)
        micro_cp = compute_causal_primitives(osc_tpm)

        # Macro: 4 groups -> 1 global. Coarse-grain 81 states to 3.
        # State index = snn*27 + lsm*9 + htm*3 + cross
        # Global bin = bin(mean(snn_bin, lsm_bin, htm_bin, cross_bin))
        macro_tpm = np.zeros((3, 3), dtype=np.float64)
        macro_counts = np.zeros(3, dtype=np.float64)

        for s in range(81):
            # Extract per-group bins from state index
            bins = [
                (s // 27) % 3,
                (s // 9) % 3,
                (s // 3) % 3,
                s % 3,
            ]
            macro_s = min(2, int(np.mean(bins) + 0.5))  # round to nearest bin

            for s_prime in range(81):
                bins_prime = [
                    (s_prime // 27) % 3,
                    (s_prime // 9) % 3,
                    (s_prime // 3) % 3,
                    s_prime % 3,
                ]
                macro_s_prime = min(2, int(np.mean(bins_prime) + 0.5))
                macro_tpm[macro_s, macro_s_prime] += osc_tpm[s, s_prime]
                macro_counts[macro_s] += osc_tpm[s, s_prime]

        # Normalize rows
        for i in range(3):
            row_sum = macro_tpm[i].sum()
            if row_sum > 0:
                macro_tpm[i] /= row_sum
            else:
                macro_tpm[i] = 1.0 / 3.0

        macro_ei = compute_ei(macro_tpm)
        macro_cp = compute_causal_primitives(macro_tpm)

        self._last_oscillator_result = {
            "micro_ei": micro_ei,
            "macro_ei": macro_ei,
            "causal_emergence": macro_ei - micro_ei,
            "ce2_delta_cp": macro_cp - micro_cp,
            "warming_up": False,
            "transitions_recorded": oscillatory_binding.oscillator_transitions_recorded,
        }
        return self._last_oscillator_result

    def get_latest_ce(self) -> Optional[float]:
        """Return latest causal emergence value (for indicator)."""
        if self._last_result is None:
            return None
        return self._last_result.causal_emergence

    def get_research_report(self) -> Dict[str, Any]:
        """Return full research report for API/export."""
        if self._last_result is None:
            return {"warming_up": True, "analysis_count": 0}

        r = self._last_result
        osc = getattr(self, "_last_oscillator_result", None)
        return {
            "module_level": {
                "micro_ei": r.micro_ei,
                "optimal_macro_ei": r.optimal_macro_ei,
                "optimal_partition": r.optimal_partition_label,
                "optimal_interpretation": r.optimal_partition_label,
                "causal_emergence": r.causal_emergence,
                "ce2_delta_cp": r.ce2_delta_cp,
                "all_partitions": [
                    {
                        "label": p["label"],
                        "ei": p["ei"],
                        "cp": p["cp"],
                        "n_groups": p["n_groups"],
                    }
                    for p in r.all_partitions
                ],
                "phi_comparison": r.phi_comparison,
            },
            "oscillator_level": (
                osc
                if osc
                else {
                    "warming_up": True,
                    "transitions_recorded": 0,
                }
            ),
            "warming_up": False,
            "restored": r.restored,
            "analysis_count": self._analysis_count,
            "computation_time_ms": r.computation_time_ms,
        }

    def to_state_dict(self) -> Dict[str, Any]:
        """Export state for persistence; every value is JSON-safe (a
        non-finite number is written as null)."""
        scalars = {"analysis_count": int(self._analysis_count)}
        histories = {}
        if self._last_result:
            result = self._last_result
            histories["last_result"] = {
                "micro_ei": _finite_real(result.micro_ei),
                "optimal_macro_ei": _finite_real(result.optimal_macro_ei),
                "optimal_partition_label": str(result.optimal_partition_label),
                "optimal_partition": _partition_lists(result.optimal_partition),
                "causal_emergence": _finite_real(result.causal_emergence),
                "ce2_delta_cp": _finite_real(result.ce2_delta_cp),
                "phi_comparison": _finite_real(result.phi_comparison),
                "computation_time_ms": _finite_real(result.computation_time_ms),
            }
        return {"scalars": scalars, "arrays": {}, "histories": histories}

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the analysis count and the last module-level result.

        The count must be a non-negative integer (otherwise it restores as
        0 with a WARNING). The result is rebuilt when ``micro_ei``,
        ``optimal_macro_ei``, ``causal_emergence`` and ``ce2_delta_cp`` are
        finite reals; it carries the saved partition (empty when an older
        state has none), no partition table and the ``restored`` mark. A
        record with a missing or non-finite field restores no result and
        logs one WARNING.
        """
        state = state if isinstance(state, dict) else {}
        scalars = state.get("scalars") or {}
        count = scalars.get("analysis_count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            logger.warning(
                "CausalEmergenceAnalyzer: ignoring invalid analysis_count %r",
                count,
            )
            count = 0
        self._analysis_count = count
        self._last_result = None

        record = (state.get("histories") or {}).get("last_result")
        if isinstance(record, dict):
            self._last_result = self._result_from_record(record)
        logger.info(
            "CausalEmergenceAnalyzer restored: %d prior analyses, last result %s",
            self._analysis_count,
            "restored" if self._last_result is not None else "absent",
        )

    @staticmethod
    def _result_from_record(
        record: Dict[str, Any],
    ) -> Optional[CausalEmergenceResult]:
        values = {
            name: _finite_real(record.get(name)) for name in _RESTORED_REAL_FIELDS
        }
        unusable = sorted(name for name, value in values.items() if value is None)
        partition = (
            _partition_from_lists(record["optimal_partition"])
            if "optimal_partition" in record
            else []
        )
        if partition is None:
            unusable.append("optimal_partition")
        if unusable:
            logger.warning(
                "CausalEmergenceAnalyzer: persisted last result not restored; "
                "missing or non-finite: %s",
                ", ".join(unusable),
            )
            return None
        return CausalEmergenceResult(
            micro_ei=values["micro_ei"],
            optimal_macro_ei=values["optimal_macro_ei"],
            optimal_partition=partition,
            optimal_partition_label=str(record.get("optimal_partition_label", "")),
            causal_emergence=values["causal_emergence"],
            ce2_delta_cp=values["ce2_delta_cp"],
            all_partitions=[],
            phi_comparison=_finite_real(record.get("phi_comparison")),
            computation_time_ms=_finite_real(record.get("computation_time_ms")),
            restored=True,
        )
