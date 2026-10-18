#!/usr/bin/env python3
"""
Consciousness Assessment: 30 Positive and 3 Negative Indicators
===============================================================

Measures 30 positive and 3 negative consciousness indicators across 8
theories (GWT, IIT, AST, HOT, FEP, RPT, BLT, AE), following the indicator
method of Butlin et al. (2023, 2025): "Consciousness in Artificial
Intelligence: Insights from the Science of Consciousness". Indicators carry
their Butlin ID where one applies; the others are framework extensions.

Measurement vocabulary (one meaning for every indicator):

- ``measured``: the module produced the evidence.
- ``absent``: the caller did not supply the module or input, so the assessed
  architecture lacks it. It scores 0.0 and stays in the averages, which is
  what an ablation compares.
- ``insufficient_data``: the module is present but has not produced the
  evidence yet, or a statistic it should emit is missing. It gets no score,
  is left out of the overall and theory averages and of the passing count,
  and is reported by name.

An assessment in which nothing is measured has no overall score (None) and
no theory scores: absent indicators enter an average only beside a measured
one.

No constant stands in for a missing input. ``STATS_CONTRACT`` lists every
statistic the v05 formulas read from each producer, with its expected type.

Phi has two paths:

- the PhiTracker (``phi_tracker``): exact IIT 4.0 Phi (system irreducibility
  analysis) of the module graph, computed through
  ``mtc.consciousness.pyphi_bridge``. It is a module-graph integration proxy,
  not a statement about the system as a whole.
- the PhiCalculator: exact IIT 4.0 Phi through the same bridge when a TPM is
  supplied, otherwise a labelled spectral integration proxy over the supplied
  connectivity.

Importing this module never imports PyPhi.

Behaviour switches (constructor fields; the new behaviour is the default and
the legacy_v04 preset sets the v0.4.0 value): ``indicator_formulas``,
``missing_signal_policy``, ``irreducibility_method``,
``causal_emergence_threshold``, ``phi_source``, ``phi_normalization``,
``phi_fixed_reference``, ``saturated_baseline_min_range`` and
``noise_baseline_matching``.

An indicator body that raises is counted under ``indicator:<name>`` in the
assessment's ``failures`` (copied onto the report), its evidence says
``failed``, and the exception is re-raised when ``Settings.DEBUG`` is set.

Features:
- Noise-baseline normalization of every measured indicator
- Report status, coverage and the insufficient indicators by name
- Negative indicators with their penalties on the report
- Ablation and longitudinal study support
- JSON-ready report export
"""

import asyncio
import json
import logging
import math
import random
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mtc.consciousness import pyphi_bridge
from mtc.consciousness.pad_affect import PADState
from mtc.core.config import get_settings
from mtc.core.paths import ensure_dir, resolve_path

logger = logging.getLogger(__name__)


# ============================================================================
# MEASUREMENT VOCABULARY AND SWITCH CHOICES
# ============================================================================

MEASURED = "measured"
ABSENT = "absent"
INSUFFICIENT_DATA = "insufficient_data"
INDICATOR_STATUSES = (MEASURED, ABSENT, INSUFFICIENT_DATA)
REPORT_STATUSES = (MEASURED, "partial", INSUFFICIENT_DATA)

INDICATOR_FORMULAS = ("v05", "v04")
MISSING_SIGNAL_POLICIES = ("insufficient_data", "v04_constants")
IRREDUCIBILITY_METHODS = ("weighted_fiedler", "spectral_entropy")
PHI_SOURCES = ("tracker_preferred", "calculator")
PHI_NORMALIZATIONS = ("fixed", "percentile")
PHI_METHODS = ("iit4_sia", "spectral_proxy")
# producer: a Phi value is normalised only against a noise baseline made by
# the same producer; v04_any: against whatever baseline the stub pass made.
NOISE_BASELINE_MATCHINGS = ("producer", "v04_any")

PHI_INTERPRETATIONS = {
    "phi_tracker_iit4": "module-graph integration proxy",
    "calculator_iit4_sia": "IIT 4.0 Phi of the supplied TPM",
    "calculator_spectral_proxy": "spectral integration proxy over the supplied "
    "connectivity (not IIT Phi)",
}

# Workspace sources that count as perceptual (the workspace's level 0).
PERCEPTUAL_SOURCES = frozenset({"snn", "lsm", "sensory"})

# The missing-statistic name the recurrence indicators report while the RPT
# measurement has had no SNN or LSM state to measure local recurrence from.
_NO_LOCAL_RECURRENCE = "rpt_measurement.average_local (no SNN or LSM state supplied)"

_COUNT = (int, np.integer)
_REAL = (int, float, np.integer, np.floating)
# A statistic the producer reports as None until it has measured it.
_OPTIONAL_REAL = _REAL + (type(None),)

# Every statistic path the v05 formulas read, per module keyword, with its
# expected type. Keys are the run_full_assessment keyword of the producer;
# "rpt_measurement.oscillatory_binding" is the RPT measurement's
# get_oscillatory_binding_stats() output. A dotted path walks nested dicts.
# Some paths exist only once the producer has activity (for example
# ignition.sustain_rate after the first ignition, and the binding's
# zero_stimulus_order_parameter once a null history exists); the formulas
# read them only then. An optional real is None while the producer has not
# measured it; the v05 formula then reports insufficient_data naming the path.
STATS_CONTRACT: Dict[str, Dict[str, Any]] = {
    "global_workspace": {
        "cycle_count": _COUNT,
        "broadcast": dict,
        "broadcast.total_broadcasts": _COUNT,
        "broadcast.coverage_ratio": _OPTIONAL_REAL,
        "ignition": dict,
        "ignition.total_ignitions": _COUNT,
        "ignition.sustain_rate": _OPTIONAL_REAL,
        "bottleneck": dict,
        "bottleneck.total_competitions": _COUNT,
        "bottleneck.avg_candidates": _REAL,
        "bottleneck.source_distribution": dict,
    },
    "attention_schema": {
        "total_updates": _COUNT,
        "history_length": _COUNT,
        "prediction_accuracy": _OPTIONAL_REAL,
        "capacity_used": _REAL,
        "voluntary_shifts": _COUNT,
        "voluntary_ratio": _OPTIONAL_REAL,
    },
    "metacognition": {
        "total_hots_generated": _COUNT,
        "introspection_count": _COUNT,
        "first_order_buffer_size": _COUNT,
        "overall_confidence": _OPTIONAL_REAL,
    },
    "active_inference": {
        "total_inferences": _COUNT,
        "total_model_updates": _COUNT,
        "avg_prediction_error": _REAL,
        "prediction_error_trend": str,
        "hierarchical_prediction_error": _REAL,
        "variational_free_energy": _REAL,
    },
    "beautiful_loop": {
        "cycle_count": _COUNT,
        "average_loop_quality": _REAL,
        "average_binding_quality": _OPTIONAL_REAL,
        "field_evidencing_count": _COUNT,
        "epistemic_depth": dict,
        "epistemic_depth.average_depth": _REAL,
        "epistemic_depth.peak_depth": _COUNT,
    },
    "rpt_measurement": {
        "measurement_count": _COUNT,
        "average_local": _OPTIONAL_REAL,
        "average_global": _REAL,
    },
    "rpt_measurement.oscillatory_binding": {
        "global_order_parameter": _REAL,
        "cross_coherence": dict,
        "binding_count": _COUNT,
        "zero_stimulus_order_parameter": _OPTIONAL_REAL,
    },
}

# The method that produces each STATS_CONTRACT entry.
STATS_METHODS: Dict[str, Tuple[str, str]] = {
    "global_workspace": ("global_workspace", "get_statistics"),
    "attention_schema": ("attention_schema", "get_statistics"),
    "metacognition": ("metacognition", "get_statistics"),
    "active_inference": ("active_inference", "get_statistics"),
    "beautiful_loop": ("beautiful_loop", "get_loop_statistics"),
    "rpt_measurement": ("rpt_measurement", "get_statistics"),
    "rpt_measurement.oscillatory_binding": (
        "rpt_measurement",
        "get_oscillatory_binding_stats",
    ),
}


# ============================================================================
# SMALL HELPERS
# ============================================================================


def _real(value: Any) -> Optional[float]:
    """A finite real number as float, else None (missing or non-numeric)."""
    if isinstance(value, _REAL):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _count(value: Any) -> int:
    """A non-negative integer count; 0 for missing or non-numeric values."""
    number = _real(value)
    if number is None or number < 0:
        return 0
    return int(number)


def _none_as(value: Any, replacement: Any) -> Any:
    """``value`` unchanged, or ``replacement`` when the producer reported None.

    The v0.4.0 formula bodies read a statistic their producers now leave
    None until it is measured; they keep the number v0.4.0 saw there.
    """
    return replacement if value is None else value


def _mapping(value: Any) -> Dict[str, Any]:
    """The value when it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def _sized_len(value: Any) -> Optional[int]:
    """len() of a real container; None for anything else (including mocks)."""
    if isinstance(value, (list, tuple, deque, dict, np.ndarray)):
        return len(value)
    return None


def _statistics(module: Any, method: str = "get_statistics") -> Dict[str, Any]:
    """A module's statistics dict; {} when the method is missing or odd."""
    getter = getattr(module, method, None)
    if not callable(getter):
        return {}
    stats = getter()
    return stats if isinstance(stats, dict) else {}


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fmt(value: Optional[float], spec: str = ".3f") -> str:
    """Format a score that may be insufficient (None)."""
    return INSUFFICIENT_DATA if value is None else format(value, spec)


def weighted_fiedler_irreducibility(
    connectivity: Any,
) -> Tuple[float, Dict[str, Any]]:
    """Irreducibility as the normalized algebraic connectivity of a weighted graph.

    W is the symmetrized absolute weight matrix with a zero diagonal. The
    score is the Fiedler value (second smallest eigenvalue) of the normalized
    Laplacian I - D^(-1/2) W D^(-1/2), times (n - 1) / n so a uniform complete
    graph scores 1.0, clamped to [0, 1]. It is 0.0 for fewer than two nodes,
    an isolated node or a disconnected graph. No binarization: a dense random
    matrix does not turn into a complete graph.

    Raises:
        ValueError: the matrix is not square or holds non-finite values.
    """
    matrix = np.abs(np.asarray(connectivity, dtype=np.float64))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"connectivity must be square, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("connectivity holds non-finite values")
    weights = (matrix + matrix.T) / 2.0
    np.fill_diagonal(weights, 0.0)
    n_nodes = weights.shape[0]
    details: Dict[str, Any] = {"graph_nodes": int(n_nodes)}
    if n_nodes < 2:
        return 0.0, {**details, "fiedler_value": 0.0, "reason": "fewer than 2 nodes"}
    degrees = weights.sum(axis=1)
    if np.any(degrees <= 0.0):
        return 0.0, {**details, "fiedler_value": 0.0, "reason": "isolated node"}
    inv_sqrt = 1.0 / np.sqrt(degrees)
    laplacian = np.eye(n_nodes) - inv_sqrt[:, None] * weights * inv_sqrt[None, :]
    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
    fiedler = float(eigenvalues[1])
    if fiedler < 1e-10:
        return 0.0, {**details, "fiedler_value": 0.0, "reason": "disconnected"}
    score = _clamp_unit(fiedler * (n_nodes - 1) / n_nodes)
    return score, {**details, "fiedler_value": fiedler}


def _error_trend(errors: Sequence[float]) -> Tuple[float, float, float]:
    """(slope, relative trend, reduction magnitude) over a window of errors.

    The relative trend is the least-squares slope times the window length
    divided by the mean error (the fractional change across the window); the
    magnitude is min(1, max(0, -relative)), so only a falling error counts.
    """
    values = np.asarray(errors, dtype=np.float64)
    n = values.size
    slope = float(np.polyfit(np.arange(n, dtype=np.float64), values, 1)[0])
    mean = float(np.mean(values))
    relative = slope * n / mean if mean > 1e-12 else 0.0
    magnitude = min(1.0, max(0.0, -relative))
    return slope, relative, magnitude


# ============================================================================
# CONSCIOUSNESS INDICATOR DEFINITIONS
# ============================================================================


class ConsciousnessTheory(Enum):
    """The consciousness theories the framework assesses.

    Matches the Butlin et al. 2025 theory groupings (GWT, HOT, AST, RPT, PP,
    AE) plus framework extensions (IIT, BLT). FEP stands for Butlin's PP
    (predictive processing).
    """

    GWT = "Global Workspace Theory"  # Baars, 1988
    IIT = "Integrated Information Theory"  # Tononi, 2004
    AST = "Attention Schema Theory"  # Graziano, 2013
    HOT = "Higher-Order Thought Theory"  # Rosenthal, 2005
    FEP = "Free Energy Principle"  # Friston, 2010 (Butlin: PP)
    RPT = "Recurrent Processing Theory"  # Lamme, 2006
    BLT = "Beautiful Loop Theory"  # Laukkonen, Friston & Chandaria, 2025
    AE = "Agency & Embodiment"  # Butlin et al., 2025 (AE-1, AE-2)


@dataclass
class IndicatorResult:
    """Result from measuring a single consciousness indicator.

    ``status`` is measured, absent or insufficient_data. An insufficient
    indicator keeps a 0.0 score in memory for compatibility; it is never
    averaged or counted as passing, and ``to_dict`` exports its score as null.
    """

    name: str
    theory: ConsciousnessTheory
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    threshold: float  # Threshold for "passing"
    passes_threshold: bool
    evidence: Dict[str, Any]
    measurement_time_ms: float
    status: str = MEASURED


@dataclass
class PhiMeasurement:
    """Result of a Phi calculation.

    ``method`` is ``iit4_sia`` (exact IIT 4.0 system irreducibility analysis
    through the PyPhi bridge) or ``spectral_proxy`` (the labelled spectral
    integration proxy).
    """

    phi: float  # The integrated information value
    phi_normalized: float  # Phi normalized to 0-1 scale
    subsystem_size: int
    computation_time_ms: float
    is_approximate: bool  # True for the spectral proxy
    cause_repertoire: Optional[Dict] = None
    effect_repertoire: Optional[Dict] = None
    method: str = "spectral_proxy"


@dataclass
class NormalizedAssessmentResult:
    """Result from a noise-normalized consciousness assessment.

    Insufficient indicators are left out of the score dicts.
    """

    raw_scores: Dict[str, float]
    noise_baselines: Dict[str, float]
    normalized_scores: Dict[str, float]
    flagged_indicators: List[str]  # Indicators where noise baseline > 0.3
    raw_report: "ConsciousnessReport"
    normalized_overall_score: Optional[float]
    noise_iterations: int


@dataclass
class ConsciousnessReport:
    """Complete consciousness assessment report.

    Under the ``insufficient_data`` policy ``overall_score`` is None when no
    indicator is measured (absent indicators alone give no number) and
    ``theory_scores`` omits a theory with no measured indicator; under
    ``v04_constants`` both always carry a number, as in v0.4.0.
    ``passing_count`` is always an integer. ``status`` is measured (every
    indicator measured), partial or insufficient_data (none measured);
    ``coverage`` is the measured share of all indicators.
    """

    session_id: str
    timestamp: datetime

    # Overall metrics
    overall_score: Optional[float]  # 0-1 weighted mean; None with nothing measured
    architecture_functional: bool  # NOTE: Indicates all modules operating as designed, NOT phenomenal consciousness
    confidence: float

    # Indicator results
    indicator_results: Dict[str, IndicatorResult]
    passing_count: int
    total_indicators: int

    # Theory-level scores (a theory with no measured indicator is omitted)
    theory_scores: Dict[str, float]

    # IIT-specific
    phi_measurement: Optional[PhiMeasurement]

    # Ablation info (if applicable)
    ablation_config: Optional[Dict[str, bool]] = None

    # Metadata
    processing_time_ms: float = 0.0
    system_state_snapshot: Optional[Dict[str, Any]] = None

    # Honest-measurement summary
    status: str = MEASURED
    coverage: float = 1.0
    insufficient_indicators: List[str] = field(default_factory=list)
    negative_indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Indicator bodies that raised during this assessment, by step
    # ("indicator:<name>" -> count); empty on a clean run
    failures: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        indicator_dict = {}
        if self.indicator_results:
            for k, v in self.indicator_results.items():
                indicator_dict[k] = {
                    "name": v.name,
                    "theory": (
                        v.theory.value if hasattr(v.theory, "value") else str(v.theory)
                    ),
                    "status": v.status,
                    "score": None if v.status == INSUFFICIENT_DATA else v.score,
                    "confidence": v.confidence,
                    "threshold": v.threshold,
                    "passes_threshold": v.passes_threshold,
                    "evidence": v.evidence,
                    "measurement_time_ms": v.measurement_time_ms,
                }

        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "status": self.status,
            "coverage": self.coverage,
            "insufficient_indicators": list(self.insufficient_indicators),
            "architecture_functional": self.architecture_functional,
            "confidence": self.confidence,
            "passing_count": self.passing_count,
            "total_indicators": self.total_indicators,
            "theory_scores": self.theory_scores,
            "negative_indicators": self.negative_indicators,
            "phi_measurement": (
                asdict(self.phi_measurement) if self.phi_measurement else None
            ),
            "indicator_results": indicator_dict,
            "ablation_config": self.ablation_config,
            "processing_time_ms": self.processing_time_ms,
            "failures": dict(self.failures),
        }


# ============================================================================
# PHI CALCULATION (exact IIT 4.0 through the PyPhi bridge, or a spectral proxy)
# ============================================================================


class PhiCalculator:
    """
    Calculates integrated information for a supplied subsystem.

    Two methods, always labelled in ``PhiMeasurement.method``:

    - ``iit4_sia``: exact IIT 4.0 Phi (the system irreducibility analysis
      only, not the full phi structure) through
      ``mtc.consciousness.pyphi_bridge``. It runs only when a state-by-node
      TPM of shape (2^n, n) with big-endian rows is supplied, n is at most
      ``max_exact_nodes``, the state is binary and PyPhi is installed. A
      connectivity matrix alone does not define a TPM.
    - ``spectral_proxy``: an IIT 3.0-inspired spectral approximation over the
      connectivity matrix. It is a proxy, not Phi.

    Normalization: ``fixed`` divides by ``fixed_reference`` (5.0 by default,
    the v0.4.0 value); ``percentile`` divides by the rolling 95th percentile
    of this calculator's own history once 20 measurements exist (before that
    the fixed reference applies), which makes the score relative: a steady
    Phi reads near 1.0.
    """

    NORMALIZATION_PERCENTILE = 95
    NORMALIZATION_MIN_HISTORY = 20

    def __init__(
        self,
        max_exact_nodes: int = 8,
        normalization: str = "fixed",
        fixed_reference: float = 5.0,
        history_size: int = 100,
    ):
        """
        Initialize Phi calculator.

        Args:
            max_exact_nodes: Largest node count for the exact computation.
            normalization: ``fixed`` or ``percentile``.
            fixed_reference: Denominator of the fixed normalization.
            history_size: Measurements kept in ``phi_history``.
        """
        if normalization not in PHI_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {PHI_NORMALIZATIONS}, "
                f"got {normalization!r}"
            )
        reference = _real(fixed_reference)
        if reference is None or reference <= 0.0:
            raise ValueError(
                f"fixed_reference must be a positive number, got {fixed_reference!r}"
            )
        if int(history_size) < 1:
            raise ValueError(f"history_size must be at least 1, got {history_size}")
        self.max_exact_nodes = max_exact_nodes
        self.normalization = normalization
        self.fixed_reference = reference
        self.history_size = int(history_size)
        self.phi_history: List[PhiMeasurement] = []
        self.pyphi_available = pyphi_bridge.pyphi_available()

        if not self.pyphi_available:
            logger.info("PyPhi not available: exact Phi needs the 'iit' extra")
        else:
            logger.info("PyPhi available for exact IIT 4.0 Phi")

    def _normalize(self, phi: float) -> float:
        """Normalize phi to [0, 1] with the configured method."""
        phi = max(0.0, float(phi))
        if (
            self.normalization == "percentile"
            and len(self.phi_history) >= self.NORMALIZATION_MIN_HISTORY
        ):
            recent = [m.phi for m in self.phi_history]
            reference = float(np.percentile(recent, self.NORMALIZATION_PERCENTILE))
            if reference < 1e-9:
                return 0.0
            return _clamp_unit(phi / reference)
        return min(1.0, phi / self.fixed_reference)

    def _exact_inputs(
        self,
        state_matrix: Any,
        tpm: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, Tuple[int, ...]]]:
        """The (tpm, binary state) pair when the exact path can run, else None."""
        if tpm is None or not self.pyphi_available:
            return None
        tpm_array = np.asarray(tpm, dtype=np.float64)
        n_nodes = len(state_matrix)
        if n_nodes > self.max_exact_nodes or tpm_array.shape != (2**n_nodes, n_nodes):
            return None
        state = np.asarray(state_matrix, dtype=np.float64)
        if not np.all(np.isin(state, (0.0, 1.0))):
            logger.warning("Exact Phi needs a binary state; using the spectral proxy")
            return None
        return tpm_array, tuple(int(bit) for bit in state)

    async def calculate_phi(
        self,
        state_matrix: np.ndarray,
        connectivity_matrix: np.ndarray,
        tpm: Optional[np.ndarray] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> PhiMeasurement:
        """
        Calculate Phi for a subsystem.

        Args:
            state_matrix: Current node states (binary for the exact path).
            connectivity_matrix: (n, n) weights, row = target, column =
                source; binarized (|w| > 0) for the exact path.
            tpm: Optional state-by-node TPM, shape (2^n, n), big-endian rows.
                Without it the spectral proxy runs.
            labels: Optional node labels for the exact path.

        Returns:
            PhiMeasurement with Phi, its normalization and the method used.
        """
        start_time = time.time()
        n_nodes = len(state_matrix)
        exact = self._exact_inputs(state_matrix, tpm)

        measurement: Optional[PhiMeasurement] = None
        if exact is not None:
            try:
                phi, cause_rep, effect_rep = self._calculate_phi_exact(
                    exact[1], exact[0], connectivity_matrix, labels
                )
                measurement = PhiMeasurement(
                    phi=phi,
                    phi_normalized=self._normalize(phi),
                    subsystem_size=n_nodes,
                    computation_time_ms=(time.time() - start_time) * 1000,
                    is_approximate=False,
                    cause_repertoire=cause_rep,
                    effect_repertoire=effect_rep,
                    method="iit4_sia",
                )
            except Exception as e:
                logger.warning(f"Exact Phi failed, using the spectral proxy: {e}")

        if measurement is None:
            phi = await self._calculate_phi_approximation(
                state_matrix, connectivity_matrix
            )
            measurement = PhiMeasurement(
                phi=phi,
                phi_normalized=self._normalize(phi),
                subsystem_size=n_nodes,
                computation_time_ms=(time.time() - start_time) * 1000,
                is_approximate=True,
                method="spectral_proxy",
            )

        self.phi_history.append(measurement)
        if len(self.phi_history) > self.history_size:
            self.phi_history = self.phi_history[-self.history_size :]
        return measurement

    def _calculate_phi_exact(
        self,
        state: Tuple[int, ...],
        tpm_big_endian: np.ndarray,
        connectivity: Optional[np.ndarray],
        labels: Optional[Sequence[str]],
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        """Exact IIT 4.0 Phi (SIA only) through the PyPhi bridge.

        The bridge reorders the rows to little-endian and transposes the
        connectivity; only the SIA runs (about 0.1 s for 5 nodes; the full
        phi structure with relations takes minutes).
        """
        n_nodes = len(state)
        cm = None
        if connectivity is not None:
            cm = (np.abs(np.asarray(connectivity, dtype=np.float64)) > 0).astype(int)
        node_labels = (
            [str(label) for label in labels]
            if labels is not None
            else [f"n{i}" for i in range(n_nodes)]
        )
        network = pyphi_bridge.build_network(tpm_big_endian, cm, node_labels)
        subsystem = pyphi_bridge.build_subsystem(network, state)
        sia = pyphi_bridge.system_irreducibility(subsystem)
        phi = max(0.0, float(sia.phi))
        partition = getattr(sia, "partition", None)
        return (
            phi,
            {"iit_version": "4.0"},
            {"partition": str(partition) if partition is not None else None},
        )

    async def _calculate_phi_approximation(
        self, state: np.ndarray, connectivity: np.ndarray
    ) -> float:
        """
        Spectral integration proxy for systems without a TPM.

        IIT 3.0-inspired approximation. Based on methodology from:
        - Oizumi, Albantakis, Tononi (2014) - PLOS Comp Bio
        - Balduzzi & Tononi (2008) - Integrated Information in Discrete Dynamical Systems

        Components:
        1. Integration (I) - Total correlation / mutual information
        2. Intrinsic information (i) - Self-referential information
        3. Minimum Information Partition (MIP) - Irreducibility across all cuts

        This approximation avoids the NP-hard exact computation; it is a
        proxy and is labelled ``spectral_proxy``.

        Raises:
            ValueError: the connectivity is not a square matrix matching the
                state, or holds non-finite values. A linear algebra error
                propagates; nothing stands in for a failed term.
        """
        n = len(state)
        connectivity = np.asarray(connectivity, dtype=np.float64)
        if connectivity.ndim != 2 or connectivity.shape != (n, n):
            raise ValueError(
                f"connectivity must be a square ({n}, {n}) matrix matching the "
                f"state, got shape {connectivity.shape}"
            )
        if not np.all(np.isfinite(connectivity)):
            raise ValueError("connectivity holds non-finite values")

        if n < 2:
            return 0.0

        # 1. Calculate INTEGRATION via spectral analysis (Tononi-style)
        # The eigenvalue distribution of the effective connectivity matrix
        # reveals how much information the system integrates as a whole
        C_normalized = connectivity / (np.max(np.abs(connectivity)) + 1e-10)

        # Add self-connections (autoregressive component)
        C_effective = C_normalized + np.eye(n) * 0.5

        # Eigenvalue analysis - integrated systems have concentrated spectrum
        eigenvalues = np.linalg.eigvalsh(C_effective)
        eigenvalues = np.abs(eigenvalues)
        eigenvalues = eigenvalues / (np.sum(eigenvalues) + 1e-10)

        # Integration = entropy of eigenvalue distribution
        # Low entropy = concentrated spectrum = high integration
        spectral_entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
        max_entropy = np.log(n)
        integration = 1.0 - (spectral_entropy / (max_entropy + 1e-10))

        # 2. Calculate INTRINSIC INFORMATION via state complexity
        # How much information does the current state carry about itself?
        if np.sum(state) > 0:
            # Normalize state to probability distribution
            state_prob = np.abs(state) / (np.sum(np.abs(state)) + 1e-10)
            state_prob = np.clip(state_prob, 1e-10, 1.0)

            # State entropy (information content)
            state_entropy = -np.sum(state_prob * np.log(state_prob))
            max_state_entropy = np.log(n)
            intrinsic_info = state_entropy / (max_state_entropy + 1e-10)
        else:
            intrinsic_info = 0.0

        # 3. Estimate MINIMUM INFORMATION PARTITION (MIP)
        # True MIP requires checking all 2^(n-1) partitions - NP-hard!
        # We approximate by:
        # a) Random partition sampling
        # b) Connectivity-based heuristic for decomposability

        # Connectivity density = how interconnected the system is
        connectivity_density = np.sum(np.abs(connectivity) > 0.1) / (n * n)

        # Cluster coefficient approximation (resistance to cuts): check for
        # modules/clusters that would make the system reducible, through
        # the singular values. If they are concentrated, the system is more
        # reducible.
        U, S, Vt = np.linalg.svd(connectivity)
        sv_normalized = S / (np.sum(S) + 1e-10)
        sv_entropy = -np.sum(sv_normalized * np.log(sv_normalized + 1e-10))
        irreducibility = sv_entropy / (np.log(len(S)) + 1e-10)

        # Ensure irreducibility is boosted by high connectivity
        irreducibility = min(1.0, irreducibility + connectivity_density * 0.5)

        # Combine components following IIT formula:
        # Phi = integration * intrinsic_info * irreducibility
        # Scale to produce values in reasonable range (0-20 typical for small systems)
        phi_raw = integration * intrinsic_info * irreducibility

        # Scale factor based on system size (larger systems can have higher Phi)
        size_factor = np.log(n + 1) / np.log(8)  # Normalize to 8-element reference
        phi_approx = phi_raw * size_factor * 10.0

        return float(max(0.0, phi_approx))


# ============================================================================
# 30-INDICATOR CONSCIOUSNESS ASSESSMENT
# ============================================================================


class ConsciousnessAssessment:
    """
    30 positive and 3 negative consciousness indicators across 8 theories.

    Based on Butlin et al. (2023, 2025) plus framework extensions from IIT,
    BLT (Laukkonen et al. 2025) and RPT (Lamme 2006).

    Integrates the consciousness modules (each optional; a missing one makes
    its indicators ``absent``):
    - EnhancedGlobalWorkspace (GWT)
    - AttentionSchemaModule (AST)
    - MetacognitionModule (HOT)
    - ActiveInferenceModule (FEP)
    - BeautifulLoop (BLT)
    - RPTMeasurement (RPT)
    - PhiTracker (IIT 4.0) and the PhiCalculator
    - PADAffectModel, CausalEmergenceAnalyzer
    """

    def __init__(
        self,
        saturated_baseline_min_range: float = 0.1,
        indicator_formulas: str = "v05",
        missing_signal_policy: str = "insufficient_data",
        irreducibility_method: str = "weighted_fiedler",
        causal_emergence_threshold: float = 0.5,
        phi_source: str = "tracker_preferred",
        phi_normalization: str = "fixed",
        phi_fixed_reference: float = 5.0,
        noise_baseline_matching: str = "producer",
    ):
        """Initialize the assessment.

        Args:
            saturated_baseline_min_range: minimum headroom (1.0 minus the noise
                baseline) an indicator must leave before its normalized score
                counts. Below it the indicator is treated as uninformative and
                scores 0.0, because noise alone nearly maxes it out. 0.0 disables
                the guard (the v0.4.0 behaviour).
            indicator_formulas: ``v05`` reads the statistics the producers emit;
                ``v04`` runs the v0.4.0 formulas, dead statistics keys and
                presence floors included.
            missing_signal_policy: ``insufficient_data`` (absent modules score
                0.0, present modules without data are excluded and named) or
                ``v04_constants`` (the v0.4.0 constants, aggregation and
                confidence).
            irreducibility_method: ``weighted_fiedler`` or ``spectral_entropy``
                (the v0.4.0 method).
            causal_emergence_threshold: passing threshold of the causal
                emergence indicator. Under the v05 formulas the score must be
                strictly above it (sigmoid(0) is exactly 0.5); v04 compares
                inclusively.
            phi_source: ``tracker_preferred`` uses a supplied PhiTracker's
                exact IIT 4.0 Phi when it has one; ``calculator`` keeps the
                v0.4.0 path through the PhiCalculator.
            phi_normalization: ``fixed`` or ``percentile`` (see PhiCalculator;
                for the tracker, percentile means the tracker's normalize_phi).
            phi_fixed_reference: denominator of the fixed normalization.
            noise_baseline_matching: ``producer`` leaves a Phi value
                un-normalised (raw score kept, evidence says so) when the
                internal noise pass made its baseline with a different
                producer, as it does for a PhiTracker value against the
                spectral proxy's baseline; ``v04_any`` normalises against
                whatever baseline the stub pass made (v0.4.0). Baselines a
                caller supplies always apply.

        Raises:
            ValueError: an invalid choice; the message names the field.
        """
        _check_choice("indicator_formulas", indicator_formulas, INDICATOR_FORMULAS)
        _check_choice(
            "missing_signal_policy", missing_signal_policy, MISSING_SIGNAL_POLICIES
        )
        _check_choice(
            "irreducibility_method", irreducibility_method, IRREDUCIBILITY_METHODS
        )
        _check_choice("phi_source", phi_source, PHI_SOURCES)
        _check_choice("phi_normalization", phi_normalization, PHI_NORMALIZATIONS)
        _check_choice(
            "noise_baseline_matching", noise_baseline_matching, NOISE_BASELINE_MATCHINGS
        )
        threshold = _real(causal_emergence_threshold)
        if threshold is None or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "causal_emergence_threshold must be a number in [0, 1], "
                f"got {causal_emergence_threshold!r}"
            )
        reference = _real(phi_fixed_reference)
        if reference is None or reference <= 0.0:
            raise ValueError(
                "phi_fixed_reference must be a positive number, "
                f"got {phi_fixed_reference!r}"
            )

        self.saturated_baseline_min_range = float(saturated_baseline_min_range)
        self.indicator_formulas = indicator_formulas
        self.missing_signal_policy = missing_signal_policy
        self.irreducibility_method = irreducibility_method
        self.causal_emergence_threshold = threshold
        self.phi_source = phi_source
        self.phi_normalization = phi_normalization
        self.phi_fixed_reference = reference
        self.noise_baseline_matching = noise_baseline_matching
        self._honest = missing_signal_policy == "insufficient_data"

        self.phi_calculator = PhiCalculator(
            normalization=phi_normalization, fixed_reference=reference
        )
        self.assessment_history: List[ConsciousnessReport] = []  # last 50
        # Indicator bodies that raised, by step ("indicator:<name>"), over
        # every assessment this instance ran
        self.failures: Counter = Counter()

        # Indicators with thresholds, weights and Butlin 2025 metadata.
        #
        # Metadata fields (after the Butlin et al. 2025 methodology):
        #   butlin_id:    canonical Butlin indicator ID, None for framework
        #                 extensions
        #   polarity:     "positive" (raises credence in consciousness) or
        #                 "negative" (lowers it); the negative indicators are
        #                 configured separately below
        #   specificity:  how rarely the property occurs without
        #                 consciousness (0 to 1, higher is more specific)
        #   sensitivity:  how often conscious systems have the property
        #                 (0 to 1, higher is more sensitive)
        #   entails:      indicators this one logically entails
        #   supported_by: indicators that provide supporting evidence
        #
        # Specificity and sensitivity are expert estimates informed by the
        # 2025 paper's discussion of indicator quality. They are not
        # measurements: there is no ground truth for AI consciousness.
        self.indicator_configs = {
            # Global Workspace Theory (GWT-2, GWT-3)
            "global_broadcast": {
                "theory": ConsciousnessTheory.GWT,
                "threshold": 0.5,
                "weight": 1.2,
                "description": "Global broadcast: information available to all modules",
                "butlin_id": "GWT-3",
                "polarity": "positive",
                "specificity": 0.7,
                "sensitivity": 0.9,
                "entails": ["recurrent_processing"],
                "supported_by": [],
            },
            "ignition_dynamics": {
                "theory": ConsciousnessTheory.GWT,
                "threshold": 0.4,
                "weight": 1.0,
                "description": "Non-linear amplification when workspace threshold crossed",
                "butlin_id": "GWT-2",
                "polarity": "positive",
                "specificity": 0.6,
                "sensitivity": 0.8,
                "entails": [],
                "supported_by": ["global_broadcast"],
            },
            "global_ignition_nuanced": {
                "theory": ConsciousnessTheory.GWT,
                "threshold": 0.4,
                "weight": 1.0,
                "description": "Refined ignition with sustain and decay dynamics",
                "butlin_id": "GWT-3",  # Extension of GWT-3
                "polarity": "positive",
                "specificity": 0.65,
                "sensitivity": 0.75,
                "entails": ["recurrent_processing"],
                "supported_by": ["ignition_dynamics"],
            },
            # Attention Schema Theory (AST-1)
            "attention_schema": {
                "theory": ConsciousnessTheory.AST,
                "threshold": 0.5,
                "weight": 1.1,
                "description": "Predictive model representing and controlling attention state",
                "butlin_id": "AST-1",
                "polarity": "positive",
                "specificity": 0.7,
                "sensitivity": 0.7,
                "entails": [],
                "supported_by": [],
            },
            "attention_control": {
                "theory": ConsciousnessTheory.AST,
                "threshold": 0.4,
                "weight": 1.0,
                "description": "Voluntary attention shifting and self-monitoring",
                "butlin_id": "AST-1",  # Sub-aspect of AST-1
                "polarity": "positive",
                "specificity": 0.6,
                "sensitivity": 0.7,
                "entails": [],
                "supported_by": ["attention_schema"],
            },
            # Higher-Order Theories (HOT-1, HOT-2, HOT-4)
            "higher_order_representations": {
                "theory": ConsciousnessTheory.HOT,
                "threshold": 0.5,
                "weight": 1.2,
                "description": "Generative, top-down perception with higher-order states",
                "butlin_id": "HOT-1",
                "polarity": "positive",
                "specificity": 0.65,
                "sensitivity": 0.8,
                "entails": [],
                "supported_by": [],
            },
            "metacognition": {
                "theory": ConsciousnessTheory.HOT,
                "threshold": 0.4,
                "weight": 1.1,
                "description": "Metacognitive monitoring distinguishing signal from noise",
                "butlin_id": "HOT-2",
                "polarity": "positive",
                "specificity": 0.7,
                "sensitivity": 0.75,
                "entails": [],
                "supported_by": ["higher_order_representations"],
            },
            "sparse_smooth_coding": {
                "theory": ConsciousnessTheory.HOT,
                "threshold": 0.3,
                "weight": 0.8,
                "description": "Sparse and smooth coding generating a quality space",
                "butlin_id": "HOT-4",
                "polarity": "positive",
                "specificity": 0.5,
                "sensitivity": 0.6,
                "entails": [],
                "supported_by": [],
            },
            # Predictive Processing / FEP (PP-1)
            "prediction_error_minimization": {
                "theory": ConsciousnessTheory.FEP,
                "threshold": 0.4,
                "weight": 1.0,
                "description": "Input modules using predictive coding",
                "butlin_id": "PP-1",
                "polarity": "positive",
                "specificity": 0.5,
                "sensitivity": 0.85,
                "entails": ["recurrent_processing", "higher_order_representations"],
                "supported_by": [],
            },
            "hierarchical_prediction": {
                "theory": ConsciousnessTheory.FEP,
                "threshold": 0.4,
                "weight": 1.0,
                "description": "Multi-level hierarchical prediction and error minimization",
                "butlin_id": "PP-1",  # Deeper aspect of PP-1
                "polarity": "positive",
                "specificity": 0.55,
                "sensitivity": 0.8,
                "entails": [],
                "supported_by": ["prediction_error_minimization"],
            },
            "temporal_prediction_depth": {
                "theory": ConsciousnessTheory.FEP,
                "threshold": 0.3,
                "weight": 0.9,
                "description": "Temporal depth of predictive model (planning horizon)",
                "butlin_id": None,  # Framework extension of PP-1
                "polarity": "positive",
                "specificity": 0.5,
                "sensitivity": 0.65,
                "entails": [],
                "supported_by": ["hierarchical_prediction"],
            },
            "agent_environment_coupling": {
                "theory": ConsciousnessTheory.FEP,
                "threshold": 0.3,
                "weight": 0.9,
                "description": "Bidirectional coupling: agent acts on and is shaped by environment",
                "butlin_id": None,  # Bridges PP-1 and AE-2
                "polarity": "positive",
                "specificity": 0.55,
                "sensitivity": 0.7,
                "entails": [],
                "supported_by": ["embodiment", "prediction_error_minimization"],
            },
            "affect_coherence": {
                "theory": ConsciousnessTheory.FEP,
                "threshold": 0.3,
                "weight": 0.9,
                "description": "PAD affect dimensions form consistent emotional states",
                "butlin_id": None,  # Framework extension (valenced experience)
                "polarity": "positive",
                "specificity": 0.6,
                "sensitivity": 0.6,
                "entails": [],
                "supported_by": [],
            },
            # Integrated Information Theory. Butlin et al. 2025 leave IIT out
            # because it is incompatible with computational functionalism;
            # the framework keeps it as an independent integration measure.
            "integrated_information": {
                "theory": ConsciousnessTheory.IIT,
                "threshold": 0.3,  # Phi > 0 is significant
                "weight": 1.3,
                "description": "System generates integrated information (Phi)",
                "butlin_id": None,  # IIT is outside the Butlin indicators
                "polarity": "positive",
                "specificity": 0.8,
                "sensitivity": 0.5,  # Lower: many conscious systems may have low Phi
                "entails": ["irreducibility"],
                "supported_by": [],
            },
            "irreducibility": {
                "theory": ConsciousnessTheory.IIT,
                "threshold": 0.4,
                "weight": 1.0,
                "description": "System cannot be decomposed without information loss",
                "butlin_id": None,
                "polarity": "positive",
                "specificity": 0.75,
                "sensitivity": 0.5,
                "entails": [],
                "supported_by": ["integrated_information"],
            },
            "causal_emergence": {
                "theory": ConsciousnessTheory.IIT,
                "threshold": threshold,  # causal_emergence_threshold
                "weight": 1.2,
                "description": "Macro-level carries more causal power than micro (Hoel EI)",
                "butlin_id": None,
                "polarity": "positive",
                "specificity": 0.7,
                "sensitivity": 0.45,
                "entails": [],
                "supported_by": ["integrated_information"],
            },
            # Recurrent Processing Theory (RPT-1, RPT-2)
            "recurrent_processing": {
                "theory": ConsciousnessTheory.RPT,
                "threshold": 0.4,
                "weight": 0.9,
                "description": "Feedback loops between perceptual and higher-order processing",
                "butlin_id": "RPT-1",
                "polarity": "positive",
                "specificity": 0.55,
                "sensitivity": 0.85,
                "entails": [],
                "supported_by": ["global_broadcast"],
            },
            "algorithmic_recurrence": {
                "theory": ConsciousnessTheory.RPT,
                "threshold": 0.3,
                "weight": 0.9,
                "description": "Algorithmic recurrence in neural substrates (SNN/LSM)",
                "butlin_id": "RPT-1",  # Implementation detail of RPT-1
                "polarity": "positive",
                "specificity": 0.5,
                "sensitivity": 0.7,
                "entails": [],
                "supported_by": ["recurrent_processing"],
            },
            "local_recurrence": {
                "theory": ConsciousnessTheory.RPT,
                "threshold": 0.3,
                "weight": 0.8,
                "description": "Local recurrent circuits amplify representations before global access",
                "butlin_id": "RPT-2",
                "polarity": "positive",
                "specificity": 0.5,
                "sensitivity": 0.7,
                "entails": [],
                "supported_by": [],
            },
            "oscillatory_binding_coherence": {
                "theory": ConsciousnessTheory.RPT,
                "threshold": 0.3,
                "weight": 1.0,
                "description": "Phase synchronization across neural substrates (Kuramoto)",
                "butlin_id": None,  # Framework extension (oscillatory binding)
                "polarity": "positive",
                "specificity": 0.6,
                "sensitivity": 0.6,
                "entails": [],
                "supported_by": ["recurrent_processing", "local_recurrence"],
            },
            # Agency and Embodiment (AE-1, AE-2)
            "agency": {
                "theory": ConsciousnessTheory.AE,
                "threshold": 0.5,
                "weight": 1.0,
                "description": "Goal-directed behavior with flexible responsiveness to competing goals",
                "butlin_id": "AE-1",
                "polarity": "positive",
                "specificity": 0.5,
                "sensitivity": 0.8,
                "entails": [],
                "supported_by": ["prediction_error_minimization"],
            },
            "embodiment": {
                "theory": ConsciousnessTheory.AE,
                "threshold": 0.3,
                "weight": 0.8,
                "description": "Modeling output-input contingencies for perception and control",
                "butlin_id": "AE-2",
                "polarity": "positive",
                "specificity": 0.45,
                "sensitivity": 0.65,
                "entails": [],
                "supported_by": [],
            },
            # Beautiful Loop Theory (Laukkonen et al. 2025). BLT post-dates
            # Butlin et al. 2025, so these are framework extensions.
            "bayesian_binding_quality": {
                "theory": ConsciousnessTheory.BLT,
                "threshold": 0.3,
                "weight": 1.0,
                "description": "Quality of Bayesian inference binding into unified percept",
                "butlin_id": None,
                "polarity": "positive",
                "specificity": 0.65,
                "sensitivity": 0.6,
                "entails": [],
                "supported_by": [],
            },
            "epistemic_depth": {
                "theory": ConsciousnessTheory.BLT,
                "threshold": 0.3,
                "weight": 1.1,
                "description": "Recursive self-reference depth (predict-the-prediction)",
                "butlin_id": None,
                "polarity": "positive",
                "specificity": 0.75,
                "sensitivity": 0.55,
                "entails": [],
                "supported_by": ["bayesian_binding_quality"],
            },
            "genuine_implementation": {
                "theory": ConsciousnessTheory.BLT,
                "threshold": 0.4,
                "weight": 1.2,
                "description": "Anti-mimicry: genuine vs superficial implementation",
                "butlin_id": None,  # Addresses Butlin's gaming problem (Box 3)
                "polarity": "positive",
                "specificity": 0.8,
                "sensitivity": 0.5,
                "entails": [],
                "supported_by": ["epistemic_depth", "bayesian_binding_quality"],
            },
            # Butlin 2025 expanded: sequential querying (GWT-4), organized
            # percepts (RPT-2), flexible responsiveness, online learning and
            # complex agency (AE-1 expansions)
            "sequential_module_querying": {
                "theory": ConsciousnessTheory.GWT,
                "threshold": 0.3,
                "weight": 1.0,
                "description": "Workspace queries modules in succession for complex tasks",
                "butlin_id": "GWT-4",
                "polarity": "positive",
                "specificity": 0.7,
                "sensitivity": 0.75,
                "entails": ["recurrent_processing"],
                "supported_by": ["global_broadcast", "ignition_dynamics"],
            },
            "organized_percepts": {
                "theory": ConsciousnessTheory.RPT,
                "threshold": 0.3,
                "weight": 0.9,
                "description": "Input modules generate organized, integrated perceptual representations",
                "butlin_id": "RPT-2",
                "polarity": "positive",
                "specificity": 0.55,
                "sensitivity": 0.75,
                "entails": [],
                "supported_by": ["local_recurrence"],
            },
            "flexible_responsiveness": {
                "theory": ConsciousnessTheory.GWT,
                "threshold": 0.3,
                "weight": 1.0,
                "description": "Flexible adaptation when conditions change mid-task (Butlin AE-1)",
                "butlin_id": "AE-1",
                "polarity": "positive",
                "specificity": 0.6,
                "sensitivity": 0.75,
                "entails": ["agency"],
                "supported_by": ["global_broadcast", "attention_control"],
            },
            "online_reportable_learning": {
                "theory": ConsciousnessTheory.HOT,
                "threshold": 0.3,
                "weight": 1.0,
                "description": "Learning during operation that can be introspected and reported (Butlin 2025)",
                "butlin_id": None,
                "polarity": "positive",
                "specificity": 0.65,
                "sensitivity": 0.7,
                "entails": [],
                "supported_by": ["metacognition", "higher_order_representations"],
            },
            "complex_agency": {
                "theory": ConsciousnessTheory.FEP,
                "threshold": 0.3,
                "weight": 0.9,
                "description": "Navigation of conflicting goals, ambiguity, and judgment calls (Butlin AE-1 expanded)",
                "butlin_id": "AE-1",
                "polarity": "positive",
                "specificity": 0.6,
                "sensitivity": 0.7,
                "entails": ["agency", "flexible_responsiveness"],
                "supported_by": [
                    "prediction_error_minimization",
                    "agent_environment_coupling",
                ],
            },
        }

        # Negative indicators (Butlin 2025 section 3): properties whose
        # presence lowers credence in consciousness. Each one above its
        # threshold subtracts score x weight x 0.05 from the overall score.
        self.negative_indicator_configs = {
            "purely_reactive": {
                "theory": ConsciousnessTheory.GWT,
                "threshold": 0.3,
                "weight": 0.8,
                "description": "System is purely stimulus-driven with no internal workspace dynamics",
                "butlin_id": None,
                "polarity": "negative",
                "specificity": 0.7,
                "sensitivity": 0.6,
            },
            "fixed_repertoire": {
                "theory": ConsciousnessTheory.AE,
                "threshold": 0.3,
                "weight": 0.7,
                "description": "Behavior is drawn from a fixed repertoire without flexible adaptation",
                "butlin_id": None,
                "polarity": "negative",
                "specificity": 0.6,
                "sensitivity": 0.65,
            },
            "no_metacognitive_access": {
                "theory": ConsciousnessTheory.HOT,
                "threshold": 0.3,
                "weight": 0.9,
                "description": "System cannot represent or report on its own internal states",
                "butlin_id": None,
                "polarity": "negative",
                "specificity": 0.75,
                "sensitivity": 0.7,
            },
        }

        # Theory-level empirical weights. Provenance: support counts in the
        # ConTraSt database (contrastdb.tau.ac.il) and the 2025-2026
        # adversarial collaboration results. They are judgement calls
        # informed by that evidence, not fitted values. A theory missing from
        # this table weighs 0.5.
        self.theory_empirical_weights = {
            "GWT": 1.0,  # 239 supporting studies in ConTraSt
            "FEP": 0.85,  # Strong theoretical support, growing empirical support
            "IIT": 0.75,  # Mixed: the COGITATE collaboration challenged key predictions
            "BLT": 0.70,  # Formally published in 2025
            "AE": 0.70,  # Supported by GWT, PP and PRM (Butlin et al. 2025 notes)
            "AST": 0.65,  # Moderate empirical support
            "RPT": 0.60,  # Strong empirical support (Lamme), few ConTraSt entries
            "HOT": 0.50,  # 14 supporting studies in ConTraSt
        }

        self._noise_baselines = None
        # Indicator -> the producer of the Phi the internal noise pass
        # normalised against (None under the v04 formulas, which computed
        # no Phi on noise)
        self._noise_baseline_sources: Dict[str, Optional[str]] = {}

        logger.info(
            "Consciousness Assessment initialized with %d indicators "
            "(formulas=%s, missing signals=%s)",
            len(self.indicator_configs),
            self.indicator_formulas,
            self.missing_signal_policy,
        )

    # ========================================================================
    # FULL ASSESSMENT
    # ========================================================================

    async def run_full_assessment(
        self,
        global_workspace=None,
        attention_schema=None,
        metacognition=None,
        active_inference=None,
        neural_states: Optional[Dict[str, np.ndarray]] = None,
        conversation_data: Optional[Dict[str, Any]] = None,
        ablation_config: Optional[Dict[str, bool]] = None,
        beautiful_loop=None,
        rpt_measurement=None,
        pad_affect_model=None,
        causal_emergence_analyzer=None,
        warmup_cycles: int = 25,
        normalize: bool = True,
        noise_baselines: Optional[Dict[str, float]] = None,
        phi_tracker=None,
    ) -> ConsciousnessReport:
        """
        Run the full consciousness assessment.

        Args:
            global_workspace: EnhancedGlobalWorkspace instance (GWT)
            attention_schema: AttentionSchemaModule instance (AST)
            metacognition: MetacognitionModule instance (HOT)
            active_inference: ActiveInferenceModule instance (FEP)
            neural_states: Raw neural data. ``combined_state`` and
                ``connectivity`` feed the PhiCalculator and irreducibility;
                an optional ``tpm`` (and ``labels``) enables exact Phi.
            conversation_data: Recent conversation for context
            ablation_config: Which components are disabled (for ablation study)
            beautiful_loop: BeautifulLoop instance
            rpt_measurement: RPTMeasurement instance
            pad_affect_model: PADAffectModel instance
            causal_emergence_analyzer: CausalEmergenceAnalyzer instance
            warmup_cycles: Number of warmup cycles actually run (for quality-based scaling)
            normalize: Whether to noise-normalize indicator scores (default True)
            noise_baselines: Pre-computed noise baselines (indicator name -> noise score).
                If provided, used instead of running stub-based noise baseline.
                Typically computed by running real modules with different random seeds.
                Caller baselines are taken as made by the same producer as
                the result and always apply; only the internal stub pass is
                subject to ``noise_baseline_matching``.
            phi_tracker: PhiTracker whose exact IIT 4.0 Phi feeds the
                integrated_information indicator under ``phi_source =
                "tracker_preferred"``.

        Returns:
            Complete ConsciousnessReport
        """
        start_time = time.time()
        self._warmup_cycles = warmup_cycles
        session_id = str(uuid.uuid4())[:8]

        logger.info(f"Starting consciousness assessment (session: {session_id})")

        # Resolve Phi BEFORE the indicator loop so integrated_information
        # reads the real value.
        phi_measurement, phi_info = await self._resolve_phi(neural_states, phi_tracker)
        phi_value = float(phi_info.get("value") or 0.0)

        indicator_results: Dict[str, IndicatorResult] = {}
        run_failures: Counter = Counter()

        # Measure each indicator
        for name, config in self.indicator_configs.items():
            indicator_start = time.time()

            try:
                score, confidence, evidence = await self._measure_indicator(
                    name,
                    config,
                    global_workspace=global_workspace,
                    attention_schema=attention_schema,
                    metacognition=metacognition,
                    active_inference=active_inference,
                    neural_states=neural_states,
                    conversation_data=conversation_data,
                    ablation_config=ablation_config,
                    beautiful_loop=beautiful_loop,
                    rpt_measurement=rpt_measurement,
                    pad_affect_model=pad_affect_model,
                    causal_emergence_analyzer=causal_emergence_analyzer,
                    phi_value=phi_value,
                    phi_info=phi_info,
                    warmup_cycles=warmup_cycles,
                )
            except Exception as e:
                # Counted by step; raised under DEBUG. The result keeps its
                # shape: insufficient_data (v04_constants: measured 0.0).
                step = f"indicator:{name}"
                self.failures[step] += 1
                run_failures[step] += 1
                if bool(get_settings().DEBUG):
                    raise
                logger.warning(f"Failed to measure {name}: {e}")
                status = INSUFFICIENT_DATA if self._honest else MEASURED
                score, confidence, evidence = 0.0, 0.0, {"error": str(e)}
                evidence["failed"] = True
                evidence["status"] = status

            indicator_results[name] = self._build_result(
                name,
                config,
                score,
                confidence,
                evidence,
                (time.time() - indicator_start) * 1000,
            )

        # Noise-normalize scores so random inputs score near zero. Caller
        # baselines always apply; the internal stub pass carries the source
        # of its Phi so a value from another producer is not normalised
        # against it.
        if normalize:
            if noise_baselines is not None:
                baselines = noise_baselines
                baseline_sources = None
            else:
                if (
                    not hasattr(self, "_noise_baselines")
                    or self._noise_baselines is None
                ):
                    self._noise_baselines = await self.run_noise_baseline(iterations=5)
                baselines = self._noise_baselines
                baseline_sources = self._noise_baseline_sources
            self._normalize_results(indicator_results, baselines, baseline_sources)

        # Theory-level and overall scores
        theory_scores = self._calculate_theory_scores(indicator_results)
        overall_score = self._overall_score(indicator_results)

        # Negative indicator penalty (Butlin 2025 section 3)
        negative_indicators = self._negative_indicator_results(
            global_workspace=global_workspace,
            attention_schema=attention_schema,
            metacognition=metacognition,
            active_inference=active_inference,
        )
        negative_penalty = sum(v["penalty"] for v in negative_indicators.values())
        if overall_score is not None:
            overall_score = max(0.0, overall_score - negative_penalty)

        # Count passing indicators (only measured ones can pass)
        passing_count = sum(1 for r in indicator_results.values() if r.passes_threshold)

        # Determine if architecture is fully functional
        # NOTE: This indicates all modules are operating as designed, NOT phenomenal consciousness
        half_indicators = len(indicator_results) // 2
        architecture_functional = bool(
            overall_score is not None
            and passing_count >= half_indicators  # At least half indicators pass
            and overall_score >= 0.5  # Overall score above threshold
            and theory_scores.get("GWT", 0) >= 0.4  # GWT must show activity
            and theory_scores.get("HOT", 0) >= 0.3  # HOT must show meta-cognition
        )

        confidence = self._calculate_confidence(indicator_results)
        measured_count = sum(
            1 for r in indicator_results.values() if r.status == MEASURED
        )
        total = len(indicator_results)
        coverage = measured_count / total if total else 0.0
        if total and measured_count == total:
            status = MEASURED
        elif measured_count == 0:
            status = INSUFFICIENT_DATA
        else:
            status = "partial"

        report = ConsciousnessReport(
            session_id=session_id,
            timestamp=datetime.now(),
            overall_score=overall_score,
            architecture_functional=architecture_functional,
            confidence=confidence,
            indicator_results=indicator_results,
            passing_count=passing_count,
            total_indicators=total,
            theory_scores=theory_scores,
            phi_measurement=phi_measurement,
            ablation_config=ablation_config,
            processing_time_ms=(time.time() - start_time) * 1000,
            status=status,
            coverage=coverage,
            insufficient_indicators=[
                name
                for name, r in indicator_results.items()
                if r.status == INSUFFICIENT_DATA
            ],
            negative_indicators=negative_indicators,
            failures=dict(run_failures),
        )

        self.assessment_history.append(report)
        if len(self.assessment_history) > 50:
            self.assessment_history = self.assessment_history[-50:]

        # Log summary
        self._log_assessment_summary(report)

        return report

    # ------------------------------------------------------------------
    # Result building, pass rule, normalization and aggregation
    # ------------------------------------------------------------------

    def _passes(self, name: str, score: float, threshold: float) -> bool:
        """The pass rule: inclusive, except causal emergence under v05.

        sigmoid(0) is exactly 0.5, so an inclusive comparison against 0.5
        would pass a system whose macro scale carries no extra causal power.
        """
        if name == "causal_emergence" and self.indicator_formulas == "v05":
            return bool(score > threshold)
        return bool(score >= threshold)

    def _result_passes(self, name: str, status: str, score: float, threshold) -> bool:
        if self._honest and status != MEASURED:
            return False
        return self._passes(name, score, threshold)

    def _build_result(
        self,
        name: str,
        config: Dict[str, Any],
        score: Any,
        confidence: Any,
        evidence: Dict[str, Any],
        elapsed_ms: float,
    ) -> IndicatorResult:
        """Clamp the score to [0, 1] and attach status and the pass decision."""
        evidence = dict(evidence)
        status = evidence.get("status", MEASURED)
        if status not in INDICATOR_STATUSES:
            status = MEASURED
        number = _real(score)
        if number is None:
            # A formula that produced no finite number measured nothing.
            evidence["non_finite_score"] = True
            if self._honest and status == MEASURED:
                status = INSUFFICIENT_DATA
            number = 0.0
        if status == INSUFFICIENT_DATA:
            number = 0.0
        number = _clamp_unit(number)
        evidence["status"] = status
        threshold = float(config["threshold"])
        return IndicatorResult(
            name=name,
            theory=config["theory"],
            score=number,
            confidence=_clamp_unit(_real(confidence) or 0.0),
            threshold=threshold,
            passes_threshold=self._result_passes(name, status, number, threshold),
            evidence=evidence,
            measurement_time_ms=elapsed_ms,
            status=status,
        )

    def _normalize_results(
        self,
        indicator_results: Dict[str, IndicatorResult],
        baselines: Dict[str, float],
        baseline_sources: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        """Noise-normalize every result that is not insufficient_data.

        An indicator with no baseline keeps its raw score and records
        ``noise_baseline: None``. Under ``noise_baseline_matching =
        "producer"`` a baseline that ``baseline_sources`` attributes to a
        different producer than the result's evidence ``source`` is treated
        the same way, with ``noise_baseline_skipped`` naming both sources; no
        baseline is synthesised for the result's producer.
        """
        for name, result in list(indicator_results.items()):
            if result.status == INSUFFICIENT_DATA:
                continue
            baseline = baselines.get(name)
            skipped: Optional[Dict[str, Any]] = None
            if baseline is not None and baseline_sources is not None:
                skipped = self._mismatched_baseline(name, result, baseline_sources)
                if skipped is not None:
                    baseline = None
            if baseline is None:
                normalized = result.score
                saturated = False
            else:
                normalized = self.get_normalized_score(
                    result.score, baseline, self.saturated_baseline_min_range
                )
                saturated = bool(
                    baseline >= 1.0
                    or (1.0 - baseline) < self.saturated_baseline_min_range
                )
            indicator_results[name] = IndicatorResult(
                name=result.name,
                theory=result.theory,
                score=float(normalized),
                confidence=result.confidence,
                threshold=result.threshold,
                passes_threshold=self._result_passes(
                    name, result.status, float(normalized), result.threshold
                ),
                evidence={
                    **result.evidence,
                    "raw_score": result.score,
                    "noise_baseline": baseline,
                    "noise_saturated": saturated,
                    **({"noise_baseline_skipped": skipped} if skipped else {}),
                },
                measurement_time_ms=result.measurement_time_ms,
                status=result.status,
            )

    def _mismatched_baseline(
        self,
        name: str,
        result: IndicatorResult,
        baseline_sources: Dict[str, Optional[str]],
    ) -> Optional[Dict[str, Any]]:
        """Both sources when the baseline's producer differs from the result's.

        None when the baselines are taken as matching: under ``v04_any``,
        when the result names no source, or when the baseline's source is
        unknown or the same.
        """
        if self.noise_baseline_matching != "producer":
            return None
        result_source = result.evidence.get("source")
        baseline_source = baseline_sources.get(name)
        if result_source is None or baseline_source is None:
            return None
        if baseline_source == result_source:
            return None
        return {"baseline_source": baseline_source, "result_source": result_source}

    def _included(self, result: IndicatorResult) -> bool:
        """Whether a result enters the averages (measured and absent do)."""
        return not self._honest or result.status != INSUFFICIENT_DATA

    def _overall_score(
        self, indicator_results: Dict[str, IndicatorResult]
    ) -> Optional[float]:
        """Weighted mean with the ConTraSt theory weighting over included results.

        Under the ``insufficient_data`` policy the overall is None until at
        least one included indicator is measured: absent indicators enter the
        mean (an ablation compares them) but on their own measure nothing.
        ``v04_constants`` averages every result, as v0.4.0 did.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        measured = 0
        for name, result in indicator_results.items():
            if not self._included(result):
                continue
            if result.status == MEASURED:
                measured += 1
            ind_weight = self.indicator_configs[name]["weight"]
            theory_name = self.indicator_configs[name]["theory"].name
            theory_weight = self.theory_empirical_weights.get(theory_name, 0.5)
            combined_weight = ind_weight * theory_weight
            weighted_sum += result.score * combined_weight
            total_weight += combined_weight
        if self._honest and measured == 0:
            return None
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_theory_scores(
        self, indicator_results: Dict[str, IndicatorResult]
    ) -> Dict[str, float]:
        """Average score per theory over included results.

        Under the ``insufficient_data`` policy a theory with no measured
        member is omitted (its absent members alone measure nothing); a
        theory with one enters with the mean over its measured and absent
        members. Under ``v04_constants`` every result is included, as in
        v0.4.0.
        """
        theory_totals: Dict[str, List[float]] = {}
        theories_measured = set()

        for result in indicator_results.values():
            if not self._included(result):
                continue
            theory_name = result.theory.name
            if theory_name not in theory_totals:
                theory_totals[theory_name] = []
            theory_totals[theory_name].append(result.score)
            if result.status == MEASURED:
                theories_measured.add(theory_name)

        if not self._honest:
            return {
                theory: np.mean(scores) if scores else 0.0
                for theory, scores in theory_totals.items()
            }
        return {
            theory: float(np.mean(scores))
            for theory, scores in theory_totals.items()
            if scores and theory in theories_measured
        }

    def _calculate_confidence(
        self, indicator_results: Dict[str, IndicatorResult]
    ) -> float:
        """Report confidence.

        insufficient_data policy: mean measured confidence times coverage.
        v04_constants: the v0.4.0 mean over every indicator.
        """
        if not indicator_results:
            return 0.0

        if not self._honest:
            confidences = [r.confidence for r in indicator_results.values()]
            return float(np.mean(confidences))

        measured = [
            r.confidence for r in indicator_results.values() if r.status == MEASURED
        ]
        if not measured:
            return 0.0
        coverage = len(measured) / len(indicator_results)
        return float(np.mean(measured)) * coverage

    # ------------------------------------------------------------------
    # Phi resolution (tracker, calculator, or neither)
    # ------------------------------------------------------------------

    async def _resolve_phi(
        self,
        neural_states: Optional[Dict[str, Any]],
        phi_tracker: Any,
    ) -> Tuple[Optional[PhiMeasurement], Dict[str, Any]]:
        """Phi for the integrated_information indicator.

        Returns the PhiMeasurement for the report and a dict with the status,
        normalized value, source and evidence the indicator reads.
        """
        tracker_note: Dict[str, Any] = {}
        use_tracker = self.phi_source == "tracker_preferred" and phi_tracker is not None
        if use_tracker:
            try:
                result = await phi_tracker.compute_architectural_phi()
            except Exception as e:
                logger.warning(f"PhiTracker computation failed: {e}")
                result = None
                tracker_note = {"tracker_error": str(e)}
            if result is not None and getattr(result, "phi", None) is not None:
                return self._tracker_measurement(phi_tracker, result)
            if result is not None:
                tracker_note = {
                    "tracker_warming_up": bool(getattr(result, "warming_up", False)),
                    "tracker_error": getattr(result, "error", None),
                    "transitions_recorded": int(
                        getattr(result, "transitions_recorded", 0) or 0
                    ),
                }

        measurement, info = await self._calculator_phi(neural_states)
        if use_tracker:
            if info["status"] == ABSENT:
                # The tracker was supplied (so Phi is not absent) but has no
                # result, and there is nothing for the calculator to use.
                info = {
                    "status": INSUFFICIENT_DATA,
                    "value": 0.0,
                    "missing": "phi_tracker result (warming up or failed) and "
                    "neural_states for the calculator",
                    "evidence": dict(tracker_note),
                }
            else:
                info["evidence"] = {
                    **info.get("evidence", {}),
                    "tracker_fallback": dict(tracker_note),
                }
        return measurement, info

    def _tracker_measurement(
        self, tracker: Any, result: Any
    ) -> Tuple[PhiMeasurement, Dict[str, Any]]:
        """PhiMeasurement and indicator input from a PhiTracker PhiResult."""
        raw = max(0.0, float(result.phi))
        normalized: Optional[float] = None
        if self.phi_normalization == "percentile":
            normalize = getattr(tracker, "normalize_phi", None)
            if callable(normalize):
                normalized = _real(normalize(raw))
        if normalized is None:
            normalized = min(1.0, raw / self.phi_fixed_reference)
        normalized = _clamp_unit(normalized)
        tpm_coverage = float(getattr(result, "tpm_coverage", 0.0) or 0.0)
        transitions = int(getattr(result, "transitions_recorded", 0) or 0)
        partition = getattr(result, "partition_info", None)
        measurement = PhiMeasurement(
            phi=raw,
            phi_normalized=normalized,
            subsystem_size=int(getattr(tracker, "N_NODES", 0) or 0),
            computation_time_ms=float(getattr(result, "computation_time_ms", 0.0)),
            is_approximate=not bool(getattr(result, "is_exact", False)),
            cause_repertoire={
                "tpm_coverage": tpm_coverage,
                "transitions_recorded": transitions,
                "iit_version": "4.0",
                "distinctions": getattr(result, "distinctions_count", None),
                "relations": getattr(result, "relations_count", None),
            },
            effect_repertoire={"partition": partition},
            method="iit4_sia",
        )
        source = "phi_tracker_iit4"
        info = {
            "status": MEASURED,
            "value": normalized,
            "phi": raw,
            "method": "iit4_sia",
            "exact": not measurement.is_approximate,
            "source": source,
            "interpretation": PHI_INTERPRETATIONS[source],
            "evidence": {
                "tpm_coverage": tpm_coverage,
                "transitions_recorded": transitions,
                "partition": partition,
            },
        }
        return measurement, info

    async def _calculator_phi(
        self, neural_states: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[PhiMeasurement], Dict[str, Any]]:
        """The PhiCalculator path (the v0.4.0 path under phi_source=calculator)."""
        if not neural_states:
            return None, {"status": ABSENT, "value": 0.0}
        if "combined_state" not in neural_states:
            return None, {
                "status": INSUFFICIENT_DATA,
                "value": 0.0,
                "missing": "neural_states.combined_state",
            }
        try:
            state = neural_states["combined_state"]
            connectivity = neural_states.get("connectivity", np.eye(len(state)))
            measurement = await self.phi_calculator.calculate_phi(
                state,
                connectivity,
                tpm=neural_states.get("tpm"),
                labels=neural_states.get("labels"),
            )
        except Exception as e:
            logger.warning(f"Phi calculation failed: {e}")
            return None, {
                "status": INSUFFICIENT_DATA,
                "value": 0.0,
                "missing": "phi calculation",
                "evidence": {"phi_error": str(e)},
            }
        value = (
            measurement.phi_normalized
            if measurement.phi_normalized is not None
            else 0.0
        )
        source = f"calculator_{measurement.method}"
        return measurement, {
            "status": MEASURED,
            "value": float(value),
            "phi": float(measurement.phi),
            "method": measurement.method,
            "exact": not measurement.is_approximate,
            "source": source,
            "interpretation": PHI_INTERPRETATIONS.get(source, ""),
            "evidence": {},
        }

    # ------------------------------------------------------------------
    # Negative indicators
    # ------------------------------------------------------------------

    def _negative_indicator_results(self, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Score, penalty and status of each negative indicator."""
        results: Dict[str, Dict[str, Any]] = {}
        for neg_name, neg_cfg in self.negative_indicator_configs.items():
            neg_score, status = self._measure_negative_indicator(neg_name, **kwargs)
            penalty = 0.0
            if neg_score is not None and neg_score > neg_cfg["threshold"]:
                penalty = neg_score * neg_cfg["weight"] * 0.05
            results[neg_name] = {
                "score": neg_score,
                "penalty": penalty,
                "status": status,
            }
        return results

    def _measure_negative_indicator(
        self, name: str, **kwargs
    ) -> Tuple[Optional[float], str]:
        """Measure a negative indicator (Butlin 2025 section 3).

        Returns (score, status): 0 to 1 where higher means MORE of the
        negative property. A missing module keeps its v0.4.0 rule level
        (absent); a present module without data is insufficient_data and
        applies no penalty. The v04 formulas read the never-emitted
        ``total_generated`` key; v05 reads ``total_hots_generated``.
        """
        if name == "purely_reactive":
            gw = kwargs.get("global_workspace")
            if gw is None:
                return 0.8, ABSENT
            stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
            ignitions = stats.get("ignition", {}).get("total_ignitions", 0)
            cycles = stats.get("cycle_count", 0)
            if self._honest and _count(cycles) == 0:
                return None, INSUFFICIENT_DATA
            if cycles > 0 and ignitions > 0:
                return 0.0, MEASURED
            return 0.5, MEASURED
        elif name == "fixed_repertoire":
            ai = kwargs.get("active_inference")
            if ai is None:
                return 0.7, ABSENT
            stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
            updates = stats.get("total_model_updates", 0)
            inferences = stats.get("total_inferences", 0)
            if self._honest and _count(inferences) == 0:
                return None, INSUFFICIENT_DATA
            if inferences > 0 and updates > 0:
                return 0.0, MEASURED
            return 0.4, MEASURED
        elif name == "no_metacognitive_access":
            meta = kwargs.get("metacognition")
            if meta is None:
                return 0.9, ABSENT
            stats = meta.get_statistics() if hasattr(meta, "get_statistics") else {}
            if self._honest and not _metacognition_ran(stats):
                return None, INSUFFICIENT_DATA
            key = (
                "total_hots_generated"
                if self.indicator_formulas == "v05"
                else ("total_generated")
            )
            hots = stats.get(key, 0)
            if hots > 0:
                return 0.0, MEASURED
            return 0.3, MEASURED
        return 0.0, MEASURED

    # ========================================================================
    # ACTIVITY-NORMALIZED SCORING
    # ========================================================================

    async def run_noise_baseline(self, iterations: int = 10) -> Dict[str, float]:
        """
        Run assessment with random/null inputs to establish floor scores.

        Any indicator scoring above 0 with noise is measuring activity, not
        consciousness. The returned baselines represent what you'd get with
        NO consciousness -- just noise from the scoring logic itself.

        For each indicator we generate synthetic "module" objects that return
        random statistics in the producers' schema, then measure the indicator
        against that noise. Only measured iterations are averaged; an
        indicator that never measured is left out of the result.

        Args:
            iterations: Number of noise runs to average over (default 10).

        Returns:
            Dict mapping indicator name to its mean noise-baseline score.
        """
        logger.info(
            f"Running noise baseline with {iterations} iterations "
            f"across {len(self.indicator_configs)} indicators"
        )

        # Accumulate scores across iterations: {indicator_name: [scores]}
        accumulated: Dict[str, List[float]] = {
            name: [] for name in self.indicator_configs
        }
        noise_phi = PhiCalculator(
            normalization=self.phi_normalization,
            fixed_reference=self.phi_fixed_reference,
        )
        phi_source: Optional[str] = None

        for i in range(iterations):
            # Build randomised mock modules for each theory group; the v04
            # formulas run on the v0.4.0 stubs.
            noise = _noise_modules(self.indicator_formulas == "v04")

            # Random neural states (small system, random values)
            noise_neural = {
                "combined_state": np.random.rand(8),
                "connectivity": np.random.rand(8, 8),
            }
            phi_info = await self._noise_phi_info(noise_phi, noise_neural)
            phi_source = phi_info.get("source")

            for name, config in self.indicator_configs.items():
                try:
                    score, _conf, evidence = await self._measure_indicator(
                        name,
                        config,
                        **noise,
                        neural_states=noise_neural,
                        conversation_data=None,
                        ablation_config=None,
                        phi_value=float(phi_info.get("value") or 0.0),
                        phi_info=phi_info,
                        warmup_cycles=25,
                    )
                except Exception:
                    if self._honest:
                        continue
                    score, evidence = 0.0, {"status": MEASURED}
                if evidence.get("status", MEASURED) != MEASURED:
                    continue
                number = _real(score)
                accumulated[name].append(0.0 if number is None else _clamp_unit(number))

        baselines = {
            name: float(np.mean(scores))
            for name, scores in accumulated.items()
            if scores
        }
        # The producer of the Phi these baselines were made with: the
        # spectral proxy under the v05 formulas, none under v04.
        self._noise_baseline_sources = {"integrated_information": phi_source}

        logger.info(
            "Noise baselines computed: %s",
            {k: f"{v:.3f}" for k, v in baselines.items()},
        )
        return baselines

    async def _noise_phi_info(
        self, calculator: "PhiCalculator", neural_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Phi input for one noise iteration.

        The v04 formulas keep the v0.4.0 noise pass, which never computed Phi
        (integrated_information read 0.0). The v05 formulas run the spectral
        proxy on the random neural states, the same path a real assessment
        with random substrates takes.
        """
        if self.indicator_formulas == "v04":
            return {"status": MEASURED, "value": 0.0}
        measurement = await calculator.calculate_phi(
            neural_states["combined_state"], neural_states["connectivity"]
        )
        source = f"calculator_{measurement.method}"
        return {
            "status": MEASURED,
            "value": float(measurement.phi_normalized),
            "phi": float(measurement.phi),
            "method": measurement.method,
            "exact": not measurement.is_approximate,
            "source": source,
            "interpretation": PHI_INTERPRETATIONS.get(source, ""),
            "evidence": {},
        }

    @staticmethod
    def get_normalized_score(
        raw_score: float, baseline: float, min_range: float = 0.1
    ) -> float:
        """
        Normalize a raw indicator score by subtracting the noise baseline.

        The formula rescales the score into the range [baseline, 1.0] so that
        only the portion *above* the noise floor counts. If the baseline itself
        leaves less than ``min_range`` of headroom the indicator is uninformative
        (noise alone nearly maxes it out), so the score is 0.0.

        Args:
            raw_score: The indicator's raw score (0.0 - 1.0).
            baseline:  The mean score the indicator produces on pure noise.

        Returns:
            Normalized score in [0.0, 1.0].
        """
        headroom = 1.0 - baseline
        if baseline >= 1.0 or headroom < min_range:
            # Noise alone (nearly) maxes the indicator out: it carries no signal.
            return 0.0
        return max(0.0, (raw_score - baseline) / headroom)

    async def run_normalized_assessment(
        self,
        noise_iterations: int = 10,
        global_workspace=None,
        attention_schema=None,
        metacognition=None,
        active_inference=None,
        neural_states: Optional[Dict[str, np.ndarray]] = None,
        conversation_data: Optional[Dict[str, Any]] = None,
        ablation_config: Optional[Dict[str, bool]] = None,
        beautiful_loop=None,
        rpt_measurement=None,
        pad_affect_model=None,
        causal_emergence_analyzer=None,
        warmup_cycles: int = 25,
        phi_tracker=None,
    ) -> NormalizedAssessmentResult:
        """
        Run a full assessment with noise-baseline normalization.

        1. Runs the stub noise baseline once (``noise_iterations`` runs).
        2. Runs the real assessment with every module keyword
           ``run_full_assessment`` takes, without its own normalization, so
           the raw report stays raw.
        3. Normalizes a copy of its indicator results exactly as
           ``run_full_assessment(normalize=True)`` would with these
           baselines: the configured ``saturated_baseline_min_range``, the
           missing-baseline branch (an indicator without a baseline keeps
           its raw score) and ``noise_baseline_matching`` all apply.
        4. Computes the normalized overall with the same weighted mean and
           the same negative-indicator penalty, floored at 0.0; it is None
           when nothing is measured.
        5. Flags every indicator whose applied noise baseline exceeds 0.3 as
           potentially measuring activity rather than consciousness.

        Insufficient indicators are left out of the scores and the overall.

        Args:
            noise_iterations: How many noise runs for baseline (default 10).
            The remaining arguments are those of ``run_full_assessment``.

        Returns:
            NormalizedAssessmentResult with raw, baseline, and normalized scores.
        """
        logger.info("Starting normalized consciousness assessment")

        # Step 1: Establish noise baselines (and the producer of their Phi)
        baselines = await self.run_noise_baseline(iterations=noise_iterations)
        baseline_sources = dict(self._noise_baseline_sources)

        # Step 2: Run real assessment (raw scores; normalization happens below)
        report = await self.run_full_assessment(
            global_workspace=global_workspace,
            attention_schema=attention_schema,
            metacognition=metacognition,
            active_inference=active_inference,
            neural_states=neural_states,
            conversation_data=conversation_data,
            ablation_config=ablation_config,
            beautiful_loop=beautiful_loop,
            rpt_measurement=rpt_measurement,
            pad_affect_model=pad_affect_model,
            causal_emergence_analyzer=causal_emergence_analyzer,
            warmup_cycles=warmup_cycles,
            phi_tracker=phi_tracker,
            normalize=False,
        )

        # Step 3: Normalize a copy of the results the way a normalizing
        # assessment does (the raw report keeps its own result objects)
        normalized_results = dict(report.indicator_results)
        self._normalize_results(normalized_results, baselines, baseline_sources)

        raw_scores: Dict[str, float] = {}
        normalized_scores: Dict[str, float] = {}
        flagged: List[str] = []
        for name, normalized in normalized_results.items():
            if not self._included(normalized):
                continue
            raw_scores[name] = report.indicator_results[name].score
            normalized_scores[name] = normalized.score

            # Step 5: Flag suspicious indicators (an applied baseline only)
            baseline = normalized.evidence.get("noise_baseline")
            if baseline is not None and baseline > 0.3:
                flagged.append(name)
                logger.warning(
                    "Indicator '%s' has noise baseline %.3f (> 0.3) -- "
                    "potentially measuring activity, not consciousness",
                    name,
                    baseline,
                )

        # Step 4: the normalized overall, with the same weights and penalty
        normalized_overall = self._overall_score(normalized_results)
        if normalized_overall is not None:
            penalty = sum(v["penalty"] for v in report.negative_indicators.values())
            normalized_overall = max(0.0, normalized_overall - penalty)

        result = NormalizedAssessmentResult(
            raw_scores=raw_scores,
            noise_baselines=baselines,
            normalized_scores=normalized_scores,
            flagged_indicators=flagged,
            raw_report=report,
            normalized_overall_score=normalized_overall,
            noise_iterations=noise_iterations,
        )

        logger.info(
            "Normalized assessment complete: raw_overall=%s, "
            "normalized_overall=%s, flagged=%d indicators",
            _fmt(report.overall_score),
            _fmt(normalized_overall),
            len(flagged),
        )

        return result

    def _log_assessment_summary(self, report: ConsciousnessReport):
        """Log a summary of the assessment."""
        logger.info("=" * 70)
        logger.info(f"CONSCIOUSNESS ASSESSMENT COMPLETE (Session: {report.session_id})")
        logger.info("=" * 70)
        logger.info(f"Overall Score: {_fmt(report.overall_score)}")
        logger.info(
            f"Status: {report.status} (coverage {report.coverage:.0%}, "
            f"{len(report.insufficient_indicators)} insufficient)"
        )
        logger.info(
            f"Architecture Functional: {'YES' if report.architecture_functional else 'NO'}"
        )
        logger.info(f"Confidence: {report.confidence:.1%}")
        logger.info(
            f"Passing Indicators: {report.passing_count}/{report.total_indicators}"
        )

        logger.info("\nTheory Scores:")
        for theory, score in report.theory_scores.items():
            logger.info(f" {theory}: {score:.3f}")

        if report.phi_measurement:
            logger.info(
                f"\nPhi ({report.phi_measurement.method}): "
                f"{report.phi_measurement.phi:.4f}"
            )
            logger.info(f" Normalized: {report.phi_measurement.phi_normalized:.3f}")
            logger.info(
                f" Approximate: {'Yes' if report.phi_measurement.is_approximate else 'No'}"
            )

        logger.info(f"\nProcessing Time: {report.processing_time_ms:.2f}ms")
        logger.info("=" * 70)

    # ========================================================================
    # INDICATOR MEASUREMENT
    # ========================================================================

    async def _measure_indicator(
        self, name: str, config: Dict, **kwargs
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Measure a single consciousness indicator.

        Returns (score, confidence, evidence) with ``evidence["status"]``
        always set. Under ``indicator_formulas="v05"`` each indicator reads
        what its producer emits (STATS_CONTRACT); under ``v04`` the v0.4.0
        bodies run, behind the missing-signal gate when the policy is
        ``insufficient_data``.
        """
        if self.indicator_formulas == "v04":
            if self._honest:
                gated = self._legacy_v04_gate(name, kwargs)
                if gated is not None:
                    return gated
            score, confidence, evidence = await self._legacy_v04_measure_indicator(
                name, config, **kwargs
            )
            evidence = dict(evidence)
            evidence.setdefault("status", _legacy_status(evidence))
            return score, confidence, evidence

        method_name = _V05_MEASURES.get(name)
        if method_name is None:
            raise KeyError(f"no measurement defined for indicator {name!r}")
        return getattr(self, method_name)(kwargs)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _absent_result(
        self, reason: str, legacy: Tuple[float, float] = (0.1, 0.3)
    ) -> Tuple[float, float, Dict[str, Any]]:
        """The caller did not supply the module: 0.0 (v0.4.0: a constant)."""
        evidence = {"ablated": True, "reason": reason, "status": ABSENT}
        if self._honest:
            return 0.0, 0.0, evidence
        return legacy[0], legacy[1], evidence

    @staticmethod
    def _insufficient_result(missing: str, **evidence) -> Tuple[float, float, Dict]:
        """The module is present but has not produced this evidence yet."""
        return (
            0.0,
            0.0,
            {**evidence, "missing": missing, "status": INSUFFICIENT_DATA},
        )

    @staticmethod
    def _measured(
        score: float, confidence: float, evidence: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        return (
            float(score),
            float(confidence),
            {**evidence, "module_active": True, "status": MEASURED},
        )

    def _stat_or_default(
        self, value: Any, default: float
    ) -> Tuple[Optional[float], bool]:
        """(number, missing) for a real statistic.

        Missing under v04_constants falls back to the v0.4.0 default; under
        insufficient_data it stays missing for the caller to report.
        """
        number = _real(value)
        if number is not None:
            return number, False
        if self._honest:
            return None, True
        return default, False

    # ------------------------------------------------------------------
    # GWT
    # ------------------------------------------------------------------

    def _workspace(self, kw) -> Tuple[Any, Dict[str, Any], int]:
        gw = kw.get("global_workspace")
        stats = _statistics(gw) if gw is not None else {}
        return gw, stats, _count(stats.get("cycle_count"))

    def _v05_global_broadcast(self, kw):
        gw, stats, cycles = self._workspace(kw)
        if gw is None:
            return self._absent_result("global_workspace not provided")
        if self._honest and cycles == 0:
            return self._insufficient_result("global_workspace.cycle_count")
        broadcast = _mapping(stats.get("broadcast"))
        broadcasts = _count(broadcast.get("total_broadcasts"))
        coverage, missing = self._stat_or_default(broadcast.get("coverage_ratio"), 0.0)
        if missing:
            return self._insufficient_result(
                "global_workspace.broadcast.coverage_ratio"
            )
        score = 0.0 if broadcasts == 0 else min(1.0, coverage)
        confidence = 0.7 if broadcasts > 10 else 0.4
        return self._measured(
            score, confidence, {"broadcasts": broadcasts, "coverage": coverage}
        )

    def _ignition(self, stats: Dict[str, Any]) -> Tuple[int, Optional[float]]:
        """(ignitions, sustain rate); the rate exists only after an ignition."""
        ignition = _mapping(stats.get("ignition"))
        ignitions = _count(ignition.get("total_ignitions"))
        if ignitions == 0:
            return 0, 0.0
        sustain, _missing = self._stat_or_default(ignition.get("sustain_rate"), 0.0)
        return ignitions, sustain

    def _v05_ignition_dynamics(self, kw):
        gw, stats, cycles = self._workspace(kw)
        if gw is None:
            return self._absent_result("global_workspace not provided")
        if self._honest and cycles == 0:
            return self._insufficient_result("global_workspace.cycle_count")
        ignitions, sustain_rate = self._ignition(stats)
        if sustain_rate is None:
            return self._insufficient_result("global_workspace.ignition.sustain_rate")
        if ignitions == 0:
            score = 0.0
        else:
            ignition_rate = min(1.0, ignitions / max(1, cycles))
            score = 0.6 * ignition_rate + 0.4 * sustain_rate
        confidence = 0.7 if ignitions > 5 else 0.4
        return self._measured(
            score,
            confidence,
            {
                "ignitions": ignitions,
                "ignition_rate": ignitions / max(1, cycles) if cycles > 0 else 0.0,
                "sustain_rate": sustain_rate,
            },
        )

    def _v05_global_ignition_nuanced(self, kw):
        gw, stats, cycles = self._workspace(kw)
        if gw is None:
            return self._absent_result("global_workspace not provided")
        if self._honest and cycles == 0:
            return self._insufficient_result("global_workspace.cycle_count")
        ignitions, sustain_rate = self._ignition(stats)
        if sustain_rate is None:
            return self._insufficient_result("global_workspace.ignition.sustain_rate")
        if ignitions == 0:
            score = 0.0
        else:
            ignition_rate = min(1.0, ignitions / max(1, cycles))
            # The ignition detector emits no decay statistic, so the dynamics
            # term is the sustain rate alone.
            score = 0.5 * ignition_rate + 0.5 * sustain_rate
        confidence = 0.6 if ignitions > 3 else 0.4
        return self._measured(
            score,
            confidence,
            {
                "ignitions": ignitions,
                "sustain_rate": sustain_rate,
                "decay_measure": None,
            },
        )

    def _bottleneck(self, stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The workspace's bottleneck statistics, None when not emitted."""
        if "bottleneck" not in stats:
            return None
        return _mapping(stats.get("bottleneck"))

    def _v05_sequential_module_querying(self, kw):
        gw, stats, cycles = self._workspace(kw)
        if gw is None:
            return self._absent_result("global_workspace not provided")
        if self._honest and cycles == 0:
            return self._insufficient_result("global_workspace.cycle_count")
        bottleneck = self._bottleneck(stats)
        if bottleneck is None:
            if self._honest:
                return self._insufficient_result("global_workspace.bottleneck")
            bottleneck = {}
        competitions = _count(bottleneck.get("total_competitions"))
        avg_candidates = 0.0
        if competitions > 0:
            avg_candidates, missing = self._stat_or_default(
                bottleneck.get("avg_candidates"), 0.0
            )
            if missing:
                return self._insufficient_result(
                    "global_workspace.bottleneck.avg_candidates"
                )
        # Successive competitions query the modules; the candidates offered
        # per competition show how many are queried each time.
        competition_score = min(1.0, competitions / 5)
        candidate_score = min(1.0, avg_candidates / 10)
        score = 0.5 * competition_score + 0.5 * candidate_score
        confidence = 0.5 + 0.3 * min(1.0, competitions / 5)
        return self._measured(
            score,
            confidence,
            {
                "cycles": cycles,
                "total_competitions": competitions,
                "avg_candidates": avg_candidates,
            },
        )

    def _source_distribution(self, bottleneck: Dict[str, Any]) -> Dict[str, int]:
        if _count(bottleneck.get("total_competitions")) == 0:
            return {}
        distribution = _mapping(bottleneck.get("source_distribution"))
        return {str(k): _count(v) for k, v in distribution.items()}

    def _v05_flexible_responsiveness(self, kw):
        gw, stats, cycles = self._workspace(kw)
        if gw is None:
            return self._absent_result("global_workspace not provided")
        if self._honest and cycles == 0:
            return self._insufficient_result("global_workspace.cycle_count")
        bottleneck = self._bottleneck(stats)
        if bottleneck is None:
            if self._honest:
                return self._insufficient_result("global_workspace.bottleneck")
            bottleneck = {}
        distribution = self._source_distribution(bottleneck)
        distinct = sum(1 for count in distribution.values() if count > 0)
        diversity_ratio = min(1.0, distinct / max(3, len(distribution)))

        # Attention shifts: consecutive competitions whose winner sets differ
        history = getattr(getattr(gw, "bottleneck", None), "competition_history", None)
        typed = []
        if isinstance(history, (list, tuple, deque)):
            for event in list(history)[-20:]:
                sources = getattr(event, "winning_sources", None)
                if isinstance(sources, (list, tuple, set, frozenset)):
                    typed.append(frozenset(str(s) for s in sources))
        pairs = len(typed) - 1
        shifts = sum(1 for a, b in zip(typed, typed[1:]) if a != b)
        attention_shifts = shifts / pairs if pairs > 0 else 0.0

        # Belief updates from active inference
        belief_updates = 0.0
        ai = kw.get("active_inference")
        if ai is not None:
            model_updates = _count(_statistics(ai).get("total_model_updates"))
            belief_updates = min(1.0, model_updates / 10)

        score = 0.4 * diversity_ratio + 0.3 * attention_shifts + 0.3 * belief_updates
        confidence = 0.5 + 0.2 * min(1.0, len(typed) / 10)
        return self._measured(
            score,
            confidence,
            {
                "source_diversity": distinct,
                "diversity_ratio": round(diversity_ratio, 3),
                "attention_shifts": round(attention_shifts, 3),
                "competition_events": len(typed),
                "belief_updates": round(belief_updates, 3),
            },
        )

    # ------------------------------------------------------------------
    # AST
    # ------------------------------------------------------------------

    def _attention(self, kw) -> Tuple[Any, Dict[str, Any], bool]:
        ast = kw.get("attention_schema")
        stats = _statistics(ast) if ast is not None else {}
        ran = (
            _count(stats.get("total_updates")) > 0
            or _count(stats.get("history_length")) > 0
        )
        return ast, stats, ran

    def _v05_attention_schema(self, kw):
        ast, stats, ran = self._attention(kw)
        if ast is None:
            return self._absent_result("attention_schema not provided")
        if self._honest and not ran:
            return self._insufficient_result("attention_schema.total_updates")
        warmup_cycles = kw.get("warmup_cycles", 25)
        schema = getattr(ast, "schema", None)
        has_focus = (
            schema is not None and getattr(schema, "current_focus", None) is not None
        )
        total_updates = _count(stats.get("total_updates"))
        history_len = _count(stats.get("history_length"))
        pred_accuracy, missing = self._stat_or_default(
            stats.get("prediction_accuracy"), 0.0
        )
        if missing:
            return self._insufficient_result("attention_schema.prediction_accuracy")
        schema_active = min(1.0, total_updates / max(1, warmup_cycles * 0.5))
        focus_score = 1.0 if has_focus else 0.0
        history_score = min(1.0, history_len / max(1, warmup_cycles * 0.3))
        score = (
            0.15 * schema_active
            + 0.15 * focus_score
            + 0.20 * history_score
            + 0.50 * pred_accuracy
        )
        confidence = 0.5 + 0.3 * min(1.0, total_updates / 5)
        return self._measured(
            score,
            confidence,
            {
                "has_focus": has_focus,
                "total_updates": total_updates,
                "history_length": history_len,
                "prediction_accuracy": round(pred_accuracy, 3),
            },
        )

    def _v05_attention_control(self, kw):
        ast, stats, ran = self._attention(kw)
        if ast is None:
            return self._absent_result("attention_schema not provided")
        if self._honest and not ran:
            return self._insufficient_result("attention_schema.total_updates")
        voluntary_shifts = _count(stats.get("voluntary_shifts"))
        voluntary_ratio = 0.0
        if voluntary_shifts > 0:
            voluntary_ratio, missing = self._stat_or_default(
                stats.get("voluntary_ratio"), 0.0
            )
            if missing:
                return self._insufficient_result("attention_schema.voluntary_ratio")
        score = 0.0 if voluntary_shifts == 0 else min(1.0, voluntary_ratio)
        confidence = 0.6 if voluntary_shifts > 3 else 0.3
        return self._measured(
            score,
            confidence,
            {"voluntary_shifts": voluntary_shifts, "voluntary_ratio": voluntary_ratio},
        )

    def _v05_embodiment(self, kw):
        """AE-2 from the attention schema's perspective statistics, no floor."""
        ast, stats, ran = self._attention(kw)
        if ast is None:
            return self._absent_result("attention_schema not provided")
        if self._honest and not ran:
            return self._insufficient_result("attention_schema.total_updates")
        warmup_cycles = kw.get("warmup_cycles", 25)
        capacity, missing = self._stat_or_default(stats.get("capacity_used"), 0.0)
        if missing:
            return self._insufficient_result("attention_schema.capacity_used")
        history_len = _count(stats.get("history_length"))
        # The schema reported 0.5 before it could classify a shift; that is
        # the v0.4.0 default, not 0.0.
        vol_ratio, missing = self._stat_or_default(stats.get("voluntary_ratio"), 0.5)
        if missing:
            return self._insufficient_result("attention_schema.voluntary_ratio")
        capacity_score = min(1.0, capacity) if capacity > 0 else 0.0
        perspective_score = min(1.0, history_len / max(1, warmup_cycles * 0.5))
        agency_score = vol_ratio
        score = 0.4 * capacity_score + 0.3 * perspective_score + 0.3 * agency_score
        confidence = 0.5 + 0.2 * min(1.0, history_len / 5)
        return self._measured(
            score,
            confidence,
            {
                "capacity_used": capacity,
                "history_length": history_len,
                "voluntary_ratio": vol_ratio,
            },
        )

    # ------------------------------------------------------------------
    # HOT
    # ------------------------------------------------------------------

    def _v05_higher_order_representations(self, kw):
        meta = kw.get("metacognition")
        if meta is None:
            return self._absent_result("metacognition not provided")
        stats = _statistics(meta)
        if self._honest and not _metacognition_ran(stats):
            return self._insufficient_result("metacognition.total_hots_generated")
        hot_count = _count(stats.get("total_hots_generated"))
        gw, _gw_stats, cycle_count = self._workspace(kw)
        if self._honest and gw is None:
            # Coverage is HOTs per workspace cycle. With no workspace the
            # architecture lacks the broadcast these HOTs would cover.
            return self._absent_result("global_workspace not provided")
        if self._honest and cycle_count == 0:
            # Coverage is HOTs per workspace cycle; without cycles there is
            # no denominator.
            return self._insufficient_result(
                "global_workspace.cycle_count", hots_generated=hot_count
            )
        if cycle_count == 0 or hot_count == 0:
            score = 0.0
        else:
            score = min(1.0, hot_count / cycle_count)
        confidence = 0.7 if hot_count > 5 else 0.4
        return self._measured(
            score,
            confidence,
            {
                "hots_generated": hot_count,
                "cycle_count": cycle_count,
                "coverage_ratio": hot_count / max(1, cycle_count),
            },
        )

    def _v05_metacognition(self, kw):
        meta = kw.get("metacognition")
        if meta is None:
            return self._absent_result("metacognition not provided")
        stats = _statistics(meta)
        if self._honest and not _metacognition_ran(stats):
            return self._insufficient_result("metacognition.total_hots_generated")
        introspections = _count(stats.get("introspection_count"))
        overall_confidence = 0.0
        if introspections > 0:
            overall_confidence, missing = self._stat_or_default(
                stats.get("overall_confidence"), 0.0
            )
            if missing:
                return self._insufficient_result("metacognition.overall_confidence")
        gw, _gw_stats, cycle_count = self._workspace(kw)
        activity_status = MEASURED
        if introspections == 0:
            score = 0.0
            activity_rate = 0.0
        else:
            if self._honest and gw is None:
                # The activity term is introspections per workspace cycle.
                # No workspace: the term is absent and scores 0.0.
                activity_rate = 0.0
                activity_status = ABSENT
            elif self._honest and cycle_count == 0:
                return self._insufficient_result(
                    "global_workspace.cycle_count", introspections=introspections
                )
            else:
                activity_rate = min(1.0, introspections / max(1, cycle_count))
            score = 0.4 * activity_rate + 0.6 * overall_confidence
        confidence = 0.6 if introspections > 2 else 0.3
        return self._measured(
            score,
            confidence,
            {
                "introspections": introspections,
                "overall_confidence": overall_confidence,
                "activity_rate": activity_rate,
                "activity_status": activity_status,
                "cycle_count": cycle_count,
            },
        )

    def _v05_online_reportable_learning(self, kw):
        meta = kw.get("metacognition")
        ai = kw.get("active_inference")
        if meta is None and ai is None:
            return self._absent_result(
                "metacognition and active_inference not provided"
            )
        ai_stats = _statistics(ai) if ai is not None else {}
        meta_stats = _statistics(meta) if meta is not None else {}
        ai_ran = _count(ai_stats.get("total_inferences")) > 0
        meta_ran = meta is not None and _metacognition_ran(meta_stats)
        if self._honest and not (ai_ran or meta_ran):
            return self._insufficient_result(
                "active_inference.total_inferences or metacognition activity"
            )
        active_beliefs = _count(ai_stats.get("total_model_updates"))
        introspections = _count(meta_stats.get("introspection_count"))
        has_recent = min(1.0, introspections / 5) if introspections > 0 else 0.0
        # No credit for merely having a learning rate: only updates and
        # introspected experience count.
        belief_score = min(1.0, active_beliefs / 10) * 0.5
        experience_score = has_recent * 0.3
        score = belief_score + experience_score
        confidence = 0.4 + 0.2 * min(1.0, active_beliefs / 5)
        return self._measured(
            score,
            confidence,
            {
                "active_beliefs": active_beliefs,
                "has_recent_experiences": has_recent > 0,
            },
        )

    # ------------------------------------------------------------------
    # FEP
    # ------------------------------------------------------------------

    def _inference(self, kw) -> Tuple[Any, Dict[str, Any], int]:
        ai = kw.get("active_inference")
        stats = _statistics(ai) if ai is not None else {}
        return ai, stats, _count(stats.get("total_inferences"))

    def _avg_error(self, stats: Dict[str, Any]) -> Tuple[Optional[float], bool]:
        return self._stat_or_default(stats.get("avg_prediction_error"), 0.5)

    def _v05_sparse_smooth_coding(self, kw):
        ai, stats, inferences = self._inference(kw)
        if ai is None:
            return self._absent_result("active_inference not provided")
        if self._honest and inferences == 0:
            return self._insufficient_result("active_inference.total_inferences")
        avg_error, missing = self._avg_error(stats)
        if missing:
            return self._insufficient_result("active_inference.avg_prediction_error")
        score = 0.0 if inferences == 0 else max(0.0, 1.0 - avg_error * 2.0)
        return self._measured(
            score, 0.5, {"avg_error": avg_error, "sparsity_proxy": score}
        )

    def _v05_prediction_error_minimization(self, kw):
        ai, stats, inferences = self._inference(kw)
        if ai is None:
            return self._absent_result("active_inference not provided")
        if self._honest and inferences == 0:
            return self._insufficient_result("active_inference.total_inferences")
        avg_error, missing = self._avg_error(stats)
        if missing:
            return self._insufficient_result("active_inference.avg_prediction_error")
        history = getattr(ai, "inference_history", None)
        errors: List[float] = []
        if isinstance(history, (list, tuple, deque)):
            for record in list(history)[-20:]:
                value = _real(getattr(record, "prediction_error", None))
                if value is not None:
                    errors.append(value)
        if len(errors) < 10:
            if self._honest:
                return self._insufficient_result(
                    "active_inference.inference_history (10 prediction errors)",
                    errors_available=len(errors),
                )
            slope, relative, magnitude = 0.0, 0.0, 0.0
        else:
            slope, relative, magnitude = _error_trend(errors)
        level_score = max(0.0, 1.0 - avg_error)
        score = 0.7 * magnitude + 0.3 * level_score
        confidence = 0.6 if inferences > 5 else 0.3
        return self._measured(
            score,
            confidence,
            {
                "avg_error": avg_error,
                "trend_slope": slope,
                "relative_trend": relative,
                "trend_magnitude": magnitude,
                "trend_window": len(errors),
                "trend_label": stats.get("prediction_error_trend"),
                "total_inferences": inferences,
            },
        )

    def _v05_hierarchical_prediction(self, kw):
        ai, stats, inferences = self._inference(kw)
        if ai is None:
            return self._absent_result("active_inference not provided")
        if self._honest and inferences == 0:
            return self._insufficient_result("active_inference.total_inferences")
        warmup_cycles = kw.get("warmup_cycles", 25)
        error, missing = self._stat_or_default(
            stats.get("hierarchical_prediction_error"), 0.0
        )
        if missing:
            return self._insufficient_result(
                "active_inference.hierarchical_prediction_error"
            )
        if inferences == 0:
            score = 0.0
        else:
            error_quality = max(0.0, 1.0 - error)
            activity_rate = min(1.0, inferences / max(1, warmup_cycles))
            score = 0.7 * error_quality + 0.3 * activity_rate
        confidence = 0.6 if inferences > 5 else 0.3
        return self._measured(
            score, confidence, {"hierarchical_error": error, "inferences": inferences}
        )

    def _v05_temporal_prediction_depth(self, kw):
        ai, stats, inferences = self._inference(kw)
        if ai is None:
            return self._absent_result("active_inference not provided")
        if self._honest and inferences == 0:
            return self._insufficient_result("active_inference.total_inferences")
        horizon = _real(getattr(getattr(ai, "config", None), "planning_horizon", None))
        if horizon is None:
            if self._honest:
                return self._insufficient_result(
                    "active_inference.config.planning_horizon"
                )
            horizon = 3.0
        avg_error, missing = self._avg_error(stats)
        if missing:
            return self._insufficient_result("active_inference.avg_prediction_error")
        belief_depth = _sized_len(getattr(ai, "belief_history", None)) or 0
        fe_depth = _sized_len(getattr(ai, "free_energy_history", None)) or 0
        if inferences == 0:
            score = 0.0
        else:
            horizon_quality = min(1.0, horizon / 10.0) * max(0.0, 1.0 - avg_error)
            score = (
                0.6 * horizon_quality
                + 0.2 * min(1.0, belief_depth / 20.0)
                + 0.2 * min(1.0, fe_depth / 20.0)
            )
        confidence = 0.5 + 0.2 * min(1.0, inferences / 10)
        return self._measured(
            score,
            confidence,
            {
                "planning_horizon": horizon,
                "avg_error": avg_error,
                "belief_history_depth": belief_depth,
                "free_energy_history_depth": fe_depth,
                "total_inferences": inferences,
            },
        )

    def _v05_agent_environment_coupling(self, kw):
        ai, stats, inferences = self._inference(kw)
        if ai is None:
            return self._absent_result("active_inference not provided")
        updates = _count(stats.get("total_model_updates"))
        if self._honest and inferences == 0 and updates == 0:
            return self._insufficient_result("active_inference.total_inferences")
        warmup_cycles = kw.get("warmup_cycles", 25)
        if inferences == 0 and updates == 0:
            score = 0.0
        else:
            act_rate = min(1.0, inferences / max(1, warmup_cycles))
            obs_rate = min(1.0, updates / max(1, warmup_cycles))
            score = (act_rate * obs_rate) ** 0.5
        confidence = 0.5 + 0.3 * min(1.0, (inferences + updates) / 20)
        return self._measured(
            score, confidence, {"actions": inferences, "observations": updates}
        )

    def _v05_affect_coherence(self, kw):
        """Affect coherence reads the PAD model alone."""
        pad = kw.get("pad_affect_model")
        if pad is None:
            return self._absent_result("pad_affect_model not provided")
        coherence = pad.compute_coherence()
        history_length = _sized_len(getattr(pad, "_history", None))
        explicit = _real(getattr(pad, "history_length", None))
        if explicit is not None:
            history_length = int(explicit)
        history_length = history_length or 0
        value = _real(coherence)
        if value is None:
            if self._honest:
                return self._insufficient_result(
                    "pad_affect_model coherence history",
                    history_length=history_length,
                )
            value = 0.3  # the v0.4.0 warm-up value
        latest = (
            pad.get_latest() if callable(getattr(pad, "get_latest", None)) else None
        )
        confidence = 0.5 + 0.3 * min(1.0, history_length / 20)
        return self._measured(
            value,
            confidence,
            {
                "coherence_score": value,
                "coherence_method": getattr(pad, "coherence_method", None),
                "history_length": history_length,
                "latest_pad": {
                    "pleasure": getattr(latest, "pleasure", None),
                    "arousal": getattr(latest, "arousal", None),
                    "dominance": getattr(latest, "dominance", None),
                    "label": getattr(latest, "label", None),
                },
            },
        )

    def _v05_complex_agency(self, kw):
        ai, stats, inferences = self._inference(kw)
        if ai is None:
            return self._absent_result("active_inference not provided")
        if self._honest and inferences == 0:
            return self._insufficient_result("active_inference.total_inferences")

        # Policy diversity: the agent's real policy set
        policies = getattr(getattr(ai, "agent", None), "policies", None)
        num_policies = _sized_len(policies)
        if num_policies is None:
            if self._honest:
                return self._insufficient_result("active_inference.agent.policies")
            num_policies = 1
        policy_diversity = min(1.0, num_policies / 5)

        # Drive conflict: drives with urgency above 0.2
        urgencies: List[float] = []
        drives = getattr(getattr(ai, "homeostatic_drives", None), "drives", None)
        if isinstance(drives, dict):
            for drive in drives.values():
                get_urgency = getattr(drive, "get_urgency", None)
                if callable(get_urgency):
                    value = _real(get_urgency())
                    if value is not None:
                        urgencies.append(value)
        urgent = sum(1 for u in urgencies if u > 0.2)
        drive_conflict = min(1.0, urgent / 3)

        # Prediction variance: variance of the recent free energy
        fe_history = getattr(ai, "free_energy_history", None)
        fe_values: List[float] = []
        if isinstance(fe_history, (list, tuple, deque)):
            for value in list(fe_history)[-20:]:
                number = _real(value)
                if number is not None:
                    fe_values.append(number)
        variance = float(np.var(fe_values)) if len(fe_values) >= 2 else 0.0
        prediction_variance = min(1.0, variance / 0.05)

        score = (
            0.4 * policy_diversity + 0.3 * drive_conflict + 0.3 * prediction_variance
        )
        confidence = 0.5 + 0.2 * min(1.0, inferences / 10)
        return self._measured(
            score,
            confidence,
            {
                "policy_count": num_policies,
                "policy_diversity": round(policy_diversity, 3),
                "urgent_drives": urgent,
                "drive_conflict": round(drive_conflict, 3),
                "free_energy_variance": variance,
                "prediction_variance": round(prediction_variance, 3),
            },
        )

    def _v05_agency(self, kw):
        """AE-1 from inference activity, model updates and free energy, no floor."""
        ai, stats, inferences = self._inference(kw)
        if ai is None:
            return self._absent_result("active_inference not provided")
        if self._honest and inferences == 0:
            return self._insufficient_result("active_inference.total_inferences")
        warmup_cycles = kw.get("warmup_cycles", 25)
        model_updates = _count(stats.get("total_model_updates"))
        free_energy, missing = self._stat_or_default(
            stats.get("variational_free_energy"), 0.0
        )
        if missing:
            return self._insufficient_result("active_inference.variational_free_energy")
        if inferences == 0:
            score = 0.0
        else:
            inference_rate = min(1.0, inferences / max(1, warmup_cycles))
            update_rate = min(1.0, model_updates / max(1, inferences))
            drive_score = (
                min(0.3, 0.3 * (1.0 - 1.0 / (1.0 + free_energy)))
                if free_energy > 0
                else 0.0
            )
            score = min(
                1.0, 0.4 * inference_rate + 0.3 * update_rate + 0.3 * drive_score
            )
        confidence = 0.6 if inferences > 5 else 0.3
        return self._measured(
            score,
            confidence,
            {
                "inferences": inferences,
                "model_updates": model_updates,
                "free_energy": free_energy,
            },
        )

    # ------------------------------------------------------------------
    # IIT
    # ------------------------------------------------------------------

    def _v05_integrated_information(self, kw):
        info = kw.get("phi_info") or {"status": ABSENT, "value": 0.0}
        status = info.get("status", ABSENT)
        detail = dict(info.get("evidence") or {})
        if status == ABSENT:
            return self._absent_result(
                "no Phi input: neither neural_states nor a phi_tracker",
                legacy=(0.0, 0.4),
            )
        if status == INSUFFICIENT_DATA:
            if self._honest:
                return self._insufficient_result(info.get("missing", "phi"), **detail)
            return 0.0, 0.4, {**detail, "phi_zero": True, "status": MEASURED}
        value = _real(info.get("value")) or 0.0
        exact = bool(info.get("exact"))
        source = info.get("source")
        return self._measured(
            min(1.0, max(0.0, value)),
            0.8 if exact else 0.6,
            {
                **detail,
                "phi": info.get("phi"),
                "phi_normalized": value,
                "exact": exact,
                "method": info.get("method"),
                "source": source,
                "phi_source_used": source,
                "interpretation": info.get("interpretation"),
            },
        )

    def _irreducibility_score(self, connectivity: Any) -> Tuple[float, Dict[str, Any]]:
        """Irreducibility of the supplied connectivity (method per the switch)."""
        if self.irreducibility_method == "weighted_fiedler":
            try:
                score, details = weighted_fiedler_irreducibility(connectivity)
            except (ValueError, np.linalg.LinAlgError) as e:
                return 0.0, {"computation_error": str(e), "method": "weighted_fiedler"}
            return score, {**details, "method": "weighted_fiedler"}

        # spectral_entropy: the v0.4.0 method. legacy_v04 compatibility,
        # remove in v0.6.0.
        conn = np.array(connectivity)
        try:
            eigenvalues = np.linalg.eigvalsh(conn)
            eigenvalues = np.abs(eigenvalues)
            total = eigenvalues.sum()
            if total > 0:
                eigenvalues_norm = eigenvalues / total
                spectral_entropy = -np.sum(
                    eigenvalues_norm * np.log(eigenvalues_norm + 1e-10)
                )
                max_entropy = np.log(len(eigenvalues))
                score = spectral_entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                score = 0.0
                spectral_entropy = 0.0
                max_entropy = 0.0
            evidence = {
                "spectral_entropy": float(spectral_entropy),
                "max_entropy": float(max_entropy),
                "n_eigenvalues": len(eigenvalues),
            }
        except Exception as e:
            score = 0.0
            evidence = {"computation_error": str(e)}
        return score, evidence

    def _v05_irreducibility(self, kw):
        neural = kw.get("neural_states")
        if not neural:
            return self._absent_result("neural_states not provided", legacy=(0.0, 0.5))
        if "connectivity" not in neural:
            if self._honest:
                return self._insufficient_result("neural_states.connectivity")
            return 0.0, 0.5, {"connectivity_missing": True, "status": MEASURED}
        score, evidence = self._irreducibility_score(neural["connectivity"])
        if "computation_error" in evidence and self._honest:
            return self._insufficient_result(
                "valid neural_states.connectivity", **evidence
            )
        return self._measured(score, 0.5, evidence)

    def _v05_causal_emergence(self, kw):
        analyzer = kw.get("causal_emergence_analyzer")
        if analyzer is None:
            return self._absent_result("analyzer not provided", legacy=(0.0, 0.3))
        ce_val = _real(analyzer.get_latest_ce())
        if ce_val is None:
            if self._honest:
                return self._insufficient_result("causal_emergence_analyzer result")
            return 0.0, 0.2, {"warming_up": True, "status": MEASURED}
        # Sigmoid: CE = 0 maps to exactly 0.5, CE > 0 above it
        score = 1.0 / (1.0 + np.exp(-ce_val * 2.0))
        return self._measured(
            score,
            0.7,
            {
                "causal_emergence": ce_val,
                "macro_beats_micro": ce_val > 0,
                "threshold": self.causal_emergence_threshold,
                "strict_threshold": True,
            },
        )

    # ------------------------------------------------------------------
    # RPT
    # ------------------------------------------------------------------

    def _rpt(self, kw) -> Tuple[Any, Dict[str, Any], int]:
        rpt = kw.get("rpt_measurement")
        stats = _statistics(rpt) if rpt is not None else {}
        return rpt, stats, _count(stats.get("measurement_count"))

    def _v05_recurrent_processing(self, kw):
        gw, stats, cycles = self._workspace(kw)
        if gw is None:
            return self._absent_result("global_workspace not provided")
        if self._honest and cycles == 0:
            return self._insufficient_result("global_workspace.cycle_count")
        broadcast = _mapping(stats.get("broadcast"))
        coverage, missing = self._stat_or_default(broadcast.get("coverage_ratio"), 0.0)
        if missing:
            return self._insufficient_result(
                "global_workspace.broadcast.coverage_ratio"
            )
        broadcasts = _count(broadcast.get("total_broadcasts"))
        if cycles == 0:
            gwt_term = 0.0
        else:
            broadcast_rate = min(1.0, broadcasts / cycles)
            gwt_term = 0.5 * coverage + 0.5 * broadcast_rate
        rpt, rpt_stats, measurements = self._rpt(kw)
        avg_global = None
        if rpt is not None and measurements > 0:
            avg_global = _real(rpt_stats.get("average_global"))
        if avg_global is not None:
            score = 0.5 * avg_global + 0.5 * gwt_term
        else:
            score = gwt_term
        confidence = 0.5 + 0.2 * min(1.0, cycles / 10)
        return self._measured(
            score,
            confidence,
            {
                "coverage": coverage,
                "cycles": cycles,
                "broadcasts": broadcasts,
                "gwt_term": gwt_term,
                "rpt_average_global": avg_global,
                "rpt_measurements": measurements,
            },
        )

    def _v05_algorithmic_recurrence(self, kw):
        rpt, stats, measurements = self._rpt(kw)
        if rpt is None:
            return self._absent_result("rpt_measurement not provided")
        if self._honest and measurements == 0:
            return self._insufficient_result("rpt_measurement.measurement_count")
        avg_local, missing = self._stat_or_default(stats.get("average_local"), 0.0)
        if missing:
            return self._insufficient_result(
                _NO_LOCAL_RECURRENCE, measurements=measurements
            )
        avg_global, missing = self._stat_or_default(stats.get("average_global"), 0.0)
        if missing:
            return self._insufficient_result(
                "rpt_measurement.average_global", measurements=measurements
            )
        score = 0.0 if measurements == 0 else 0.6 * avg_local + 0.4 * avg_global
        confidence = 0.5 + 0.3 * min(1.0, measurements / 10)
        return self._measured(
            score,
            confidence,
            {
                "avg_local": avg_local,
                "avg_global": avg_global,
                "measurements": measurements,
            },
        )

    def _v05_local_recurrence(self, kw):
        rpt, stats, measurements = self._rpt(kw)
        if rpt is None:
            return self._absent_result("rpt_measurement not provided")
        if self._honest and measurements == 0:
            return self._insufficient_result(
                "rpt_measurement.measurement_count",
                avg_local_recurrence=None,
                measurements=0,
            )
        avg_local, missing = self._stat_or_default(stats.get("average_local"), 0.0)
        if missing:
            return self._insufficient_result(
                _NO_LOCAL_RECURRENCE,
                avg_local_recurrence=None,
                measurements=measurements,
            )
        score = avg_local if measurements > 0 else 0.0
        confidence = 0.5 + 0.2 * min(1.0, measurements / 10)
        return self._measured(
            score,
            confidence,
            {"avg_local_recurrence": avg_local, "measurements": measurements},
        )

    def _v05_oscillatory_binding_coherence(self, kw):
        rpt = kw.get("rpt_measurement")
        if rpt is None:
            return self._absent_result("rpt_measurement not provided")
        osc_stats = _statistics(rpt, "get_oscillatory_binding_stats")
        binding_count = _count(osc_stats.get("binding_count")) if osc_stats else 0
        if self._honest and binding_count == 0:
            return self._insufficient_result("oscillatory binding statistics")
        global_r, missing = self._stat_or_default(
            osc_stats.get("global_order_parameter"), 0.0
        )
        if missing:
            return self._insufficient_result(
                "rpt_measurement.oscillatory_binding.global_order_parameter"
            )
        cross_vals = _mapping(osc_stats.get("cross_coherence"))
        numbers = [v for v in (_real(x) for x in cross_vals.values()) if v is not None]
        mean_cross = max(0.0, float(np.mean(numbers))) if numbers else 0.0
        score = 0.0 if binding_count == 0 else 0.6 * global_r + 0.4 * mean_cross
        confidence = 0.5 + 0.3 * min(1.0, binding_count / 10)
        # The binding layer's own zero-stimulus floor: the order parameter
        # the same dynamics reach with no input. When it leaves less than
        # saturated_baseline_min_range of headroom the order parameter cannot
        # register an input and the indicator is uninformative, so it scores
        # 0.0 (the same guard a noise baseline gets). None when the producer
        # did not supply the floor; 0.0 for the minimum range disables it.
        floor = None
        if "zero_stimulus_order_parameter" in osc_stats:
            floor = _real(osc_stats.get("zero_stimulus_order_parameter"))
        floor_saturated = bool(
            floor is not None and (1.0 - floor) < self.saturated_baseline_min_range
        )
        if floor_saturated:
            score = 0.0
        return self._measured(
            score,
            confidence,
            {
                "global_order_parameter": global_r,
                "cross_coherence": dict(cross_vals),
                "mean_cross_coherence": mean_cross,
                "binding_count": binding_count,
                "zero_stimulus_order_parameter": floor,
                "floor_saturated": floor_saturated,
            },
        )

    def _v05_organized_percepts(self, kw):
        gw, stats, cycles = self._workspace(kw)
        rpt, rpt_stats, measurements = self._rpt(kw)
        if gw is None and rpt is None:
            return self._absent_result(
                "global_workspace and rpt_measurement not provided"
            )
        gw_ran = gw is not None and cycles > 0
        rpt_ran = rpt is not None and measurements > 0
        if self._honest and not (gw_ran or rpt_ran):
            return self._insufficient_result(
                "global_workspace.cycle_count or rpt_measurement.measurement_count"
            )
        perceptual_count = 0
        if gw_ran:
            bottleneck = self._bottleneck(stats)
            if bottleneck is None:
                if self._honest:
                    return self._insufficient_result("global_workspace.bottleneck")
                bottleneck = {}
            distribution = self._source_distribution(bottleneck)
            perceptual_count = sum(
                count
                for source, count in distribution.items()
                if source in PERCEPTUAL_SOURCES
            )
        perceptual_score = min(1.0, perceptual_count / 5) if perceptual_count else 0.0
        # The RPT term is optional: without an RPT, or with an RPT that has
        # no local value yet (no substrate state arrived), it contributes
        # nothing and the evidence records None (the v04_constants policy
        # records the constant 0.0); it is insufficient only when the
        # workspace did not run as well.
        avg_local = None if self._honest else 0.0
        if rpt_ran:
            avg_local, missing = self._stat_or_default(
                rpt_stats.get("average_local"), 0.0
            )
            if missing:
                if not gw_ran:
                    return self._insufficient_result(_NO_LOCAL_RECURRENCE)
                avg_local = None
        local_term = 0.0 if avg_local is None else avg_local
        score = 0.5 * perceptual_score + 0.5 * local_term
        confidence = 0.4 + 0.2 * min(1.0, perceptual_count / 3)
        return self._measured(
            score,
            confidence,
            {
                "perceptual_candidates": perceptual_count,
                "avg_local_recurrence": avg_local,
            },
        )

    # ------------------------------------------------------------------
    # BLT
    # ------------------------------------------------------------------

    def _loop(self, kw) -> Tuple[Any, Dict[str, Any], int]:
        bl = kw.get("beautiful_loop")
        stats = _statistics(bl, "get_loop_statistics") if bl is not None else {}
        return bl, stats, _count(stats.get("cycle_count"))

    def _v05_bayesian_binding_quality(self, kw):
        bl, stats, cycles = self._loop(kw)
        if bl is None:
            return self._absent_result("beautiful_loop not provided")
        if self._honest and cycles == 0:
            return self._insufficient_result("beautiful_loop.cycle_count")
        avg_binding, missing = self._stat_or_default(
            stats.get("average_binding_quality"), 0.0
        )
        if missing:
            return self._insufficient_result("beautiful_loop.average_binding_quality")
        score = avg_binding if cycles > 0 else 0.0
        confidence = 0.5 + 0.3 * min(1.0, cycles / 10)
        return self._measured(
            score, confidence, {"avg_binding": avg_binding, "cycles": cycles}
        )

    def _depth(self, stats: Dict[str, Any]) -> Tuple[Optional[float], bool, int]:
        """(average depth, whether it is missing, peak depth) of the loop."""
        depth_stats = _mapping(stats.get("epistemic_depth"))
        avg_depth, missing = self._stat_or_default(
            depth_stats.get("average_depth"), 0.0
        )
        return avg_depth, missing, _count(depth_stats.get("peak_depth"))

    def _v05_epistemic_depth(self, kw):
        bl, stats, cycles = self._loop(kw)
        if bl is None:
            return self._absent_result("beautiful_loop not provided")
        if self._honest and cycles == 0:
            return self._insufficient_result("beautiful_loop.cycle_count")
        avg_depth, missing, peak_depth = self._depth(stats)
        if missing:
            return self._insufficient_result(
                "beautiful_loop.epistemic_depth.average_depth"
            )
        score = min(1.0, avg_depth / 4.0) if avg_depth > 0 else 0.0
        confidence = 0.6 if peak_depth >= 2 else 0.3
        return self._measured(
            score, confidence, {"avg_depth": avg_depth, "peak_depth": peak_depth}
        )

    def _v05_genuine_implementation(self, kw):
        bl, stats, cycles = self._loop(kw)
        if bl is None:
            return self._absent_result("beautiful_loop not provided")
        if self._honest and cycles == 0:
            return self._insufficient_result("beautiful_loop.cycle_count")
        avg_quality, missing = self._stat_or_default(
            stats.get("average_loop_quality"), 0.0
        )
        if missing:
            return self._insufficient_result("beautiful_loop.average_loop_quality")
        fe_count = _count(stats.get("field_evidencing_count"))
        _avg_depth, _depth_missing, peak_depth = self._depth(stats)
        has_field_evidencing = fe_count > 0
        has_loop_quality = avg_quality >= 0.4
        has_strange_loop = peak_depth >= 2
        conditions_met = sum([has_field_evidencing, has_loop_quality, has_strange_loop])
        score = 0.0 if cycles == 0 else conditions_met / 3.0
        confidence = 0.5 + 0.3 * min(1.0, cycles / 20)
        return self._measured(
            score,
            confidence,
            {
                "avg_loop_quality": avg_quality,
                "field_evidencing_count": fe_count,
                "has_field_evidencing": has_field_evidencing,
                "has_loop_quality": has_loop_quality,
                "has_strange_loop": has_strange_loop,
                "peak_depth": peak_depth,
                "conditions_met": conditions_met,
            },
        )

    # ------------------------------------------------------------------
    # The v0.4.0 path behind the missing-signal gate
    # ------------------------------------------------------------------

    def _legacy_v04_gate(self, name: str, kw: Dict[str, Any]):
        """Absent or insufficient_data for the v04 formulas, else None.

        legacy_v04 compatibility (the v04 formulas under the
        insufficient_data policy), remove in v0.6.0.
        """
        if name == "integrated_information":
            info = kw.get("phi_info") or {"status": ABSENT}
            status = info.get("status", ABSENT)
            if status == ABSENT:
                return self._absent_result(
                    "no Phi input: neither neural_states nor a phi_tracker"
                )
            if status == INSUFFICIENT_DATA:
                return self._insufficient_result(
                    info.get("missing", "phi"), **dict(info.get("evidence") or {})
                )
            return None
        if name == "irreducibility":
            neural = kw.get("neural_states")
            if not neural:
                return self._absent_result("neural_states not provided")
            if "connectivity" not in neural:
                return self._insufficient_result("neural_states.connectivity")
            return None
        if name == "causal_emergence":
            analyzer = kw.get("causal_emergence_analyzer")
            if analyzer is None:
                return self._absent_result("analyzer not provided")
            if _real(analyzer.get_latest_ce()) is None:
                return self._insufficient_result("causal_emergence_analyzer result")
            return None
        if name == "affect_coherence":
            if kw.get("active_inference") is None:
                return self._absent_result("active_inference not provided")
            pad = kw.get("pad_affect_model")
            if pad is None:
                return self._absent_result("pad_affect_model not provided")
            if _real(pad.compute_coherence()) is None:
                return self._insufficient_result("pad_affect_model coherence history")
            return None
        if name == "organized_percepts":
            gw = kw.get("global_workspace")
            rpt = kw.get("rpt_measurement")
            if gw is None and rpt is None:
                return self._absent_result("global_workspace and rpt_measurement")
            if not (
                (gw is not None and _module_ran("global_workspace", gw))
                or (rpt is not None and _module_ran("rpt_measurement", rpt))
            ):
                return self._insufficient_result("module activity")
            return None
        if name == "online_reportable_learning":
            meta = kw.get("metacognition")
            ai = kw.get("active_inference")
            if meta is None and ai is None:
                return self._absent_result("metacognition and active_inference")
            if not (
                (meta is not None and _module_ran("metacognition", meta))
                or (ai is not None and _module_ran("active_inference", ai))
            ):
                return self._insufficient_result("module activity")
            return None
        module_key = _GATE_MODULES.get(name)
        if module_key is None:
            return None
        module = kw.get(module_key)
        if module is None:
            return self._absent_result(f"{module_key} not provided")
        if name == "oscillatory_binding_coherence":
            osc_stats = _statistics(module, "get_oscillatory_binding_stats")
            if _count(osc_stats.get("binding_count")) == 0:
                return self._insufficient_result("oscillatory binding statistics")
            return None
        if name == "agent_environment_coupling":
            stats = _statistics(module)
            if (
                _count(stats.get("total_inferences")) == 0
                and _count(stats.get("total_model_updates")) == 0
            ):
                return self._insufficient_result("active_inference.total_inferences")
            return None
        if not _module_ran(module_key, module):
            return self._insufficient_result(f"{module_key} activity")
        return None

    async def _legacy_v04_measure_indicator(
        self, name: str, config: Dict, **kwargs
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        The v0.4.0 indicator bodies, unchanged arithmetic (dead statistics
        keys, presence floors and 0.1 constants for absent modules included).

        Two adaptations keep the other switches independent: irreducibility
        dispatches on irreducibility_method (spectral_entropy is the
        v0.4.0 code), and a PAD coherence of None (the temporal method before
        its minimum history) reads as the v0.4.0 warm-up value 0.3.

        legacy_v04 compatibility, remove in v0.6.0.

        Returns: (score, confidence, evidence_dict)
        """
        theory = config["theory"]
        evidence = {}
        score = 0.0
        confidence = 0.5  # Default confidence

        # Check if component is ablated
        kwargs.get("ablation_config", {})

        # GWT Indicators (from EnhancedGlobalWorkspace)
        if theory == ConsciousnessTheory.GWT:
            gw = kwargs.get("global_workspace")
            if gw is not None:
                if name == "global_broadcast":
                    stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                    broadcast_stats = stats.get("broadcast", {})
                    broadcast_count = broadcast_stats.get("total_broadcasts", 0)
                    coverage = _none_as(broadcast_stats.get("coverage_ratio", 0.0), 0.0)
                    if broadcast_count == 0:
                        score = 0.0
                    else:
                        score = min(1.0, coverage)
                    confidence = 0.7 if broadcast_count > 10 else 0.4
                    evidence = {
                        "broadcasts": broadcast_count,
                        "coverage": coverage,
                        "module_active": True,
                    }

                elif name == "ignition_dynamics":
                    stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                    ignition_stats = stats.get("ignition", {})
                    ignitions = ignition_stats.get("total_ignitions", 0)
                    sustain_rate = _none_as(
                        ignition_stats.get("sustain_rate", 0.0), 0.0
                    )
                    cycle_count = stats.get("cycle_count", 0)
                    if ignitions == 0:
                        score = 0.0
                    else:
                        ignition_rate = min(1.0, ignitions / max(1, cycle_count))
                        sustain_quality = sustain_rate
                        score = 0.6 * ignition_rate + 0.4 * sustain_quality
                    confidence = 0.7 if ignitions > 5 else 0.4
                    evidence = {
                        "ignitions": ignitions,
                        "ignition_rate": (
                            ignitions / max(1, cycle_count) if cycle_count > 0 else 0.0
                        ),
                        "sustain_rate": sustain_rate,
                        "module_active": True,
                    }

            else:
                # GWT component missing/ablated
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True, "reason": "global_workspace not provided"}

        # AST Indicators (from AttentionSchemaModule)
        elif theory == ConsciousnessTheory.AST:
            ast = kwargs.get("attention_schema")
            if ast is not None:
                if name == "attention_schema":
                    try:
                        stats = (
                            ast.get_statistics()
                            if hasattr(ast, "get_statistics")
                            else {}
                        )
                        schema = getattr(ast, "schema", None)
                        warmup_cycles = kwargs.get("warmup_cycles", 25)

                        has_focus = (
                            schema is not None
                            and getattr(schema, "current_focus", None) is not None
                        )
                        total_updates = stats.get("total_updates", 0)
                        history_len = stats.get("history_length", 0)
                        pred_accuracy = _none_as(
                            stats.get("prediction_accuracy", 0.0), 0.0
                        )

                        schema_active = min(
                            1.0, total_updates / max(1, warmup_cycles * 0.5)
                        )
                        focus_score = 1.0 if has_focus else 0.0
                        history_score = min(
                            1.0, history_len / max(1, warmup_cycles * 0.3)
                        )
                        prediction_score = pred_accuracy

                        score = (
                            0.15 * schema_active
                            + 0.15 * focus_score
                            + 0.20 * history_score
                            + 0.50 * prediction_score
                        )
                        confidence = 0.5 + 0.3 * min(1.0, total_updates / 5)
                        evidence = {
                            "has_focus": has_focus,
                            "total_updates": total_updates,
                            "history_length": history_len,
                            "prediction_accuracy": round(pred_accuracy, 3),
                            "module_active": True,
                        }
                    except Exception as e:
                        score = 0.0
                        confidence = 0.3
                        evidence = {"module_present": True, "access_error": str(e)}

                elif name == "attention_control":
                    stats = (
                        ast.get_statistics() if hasattr(ast, "get_statistics") else {}
                    )
                    voluntary_shifts = stats.get("voluntary_shifts", 0)
                    # The schema reported 0.5 before it could classify a shift
                    voluntary_ratio = _none_as(stats.get("voluntary_ratio", 0.0), 0.5)
                    if voluntary_shifts == 0:
                        score = 0.0
                    else:
                        score = min(1.0, voluntary_ratio)
                    confidence = 0.6 if voluntary_shifts > 3 else 0.3
                    evidence = {
                        "voluntary_shifts": voluntary_shifts,
                        "voluntary_ratio": voluntary_ratio,
                        "module_active": True,
                    }

                elif name == "embodiment":
                    stats = (
                        ast.get_statistics() if hasattr(ast, "get_statistics") else {}
                    )
                    warmup_cycles = kwargs.get("warmup_cycles", 25)
                    capacity = stats.get("capacity_used", 0.0)
                    history_len = stats.get("history_length", 0)
                    vol_ratio = _none_as(stats.get("voluntary_ratio", 0.0), 0.5)
                    capacity_score = min(1.0, capacity) if capacity > 0 else 0.0
                    perspective_score = min(
                        1.0, history_len / max(1, warmup_cycles * 0.5)
                    )
                    agency_score = vol_ratio
                    score = (
                        0.4 * capacity_score
                        + 0.3 * perspective_score
                        + 0.3 * agency_score
                    )
                    confidence = 0.5 + 0.2 * min(1.0, history_len / 5)
                    evidence = {
                        "capacity_used": capacity,
                        "history_length": history_len,
                        "voluntary_ratio": vol_ratio,
                        "module_active": True,
                    }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True, "reason": "attention_schema not provided"}

        # HOT Indicators (from MetacognitionModule)
        elif theory == ConsciousnessTheory.HOT:
            meta = kwargs.get("metacognition")
            if meta is not None:
                if name == "higher_order_representations":
                    stats = (
                        meta.get_statistics() if hasattr(meta, "get_statistics") else {}
                    )
                    hot_count = stats.get("total_hots_generated", 0)
                    gw = kwargs.get("global_workspace")
                    gw_stats = (
                        gw.get_statistics()
                        if gw is not None and hasattr(gw, "get_statistics")
                        else {}
                    )
                    cycle_count = gw_stats.get("cycle_count", 0)
                    if cycle_count == 0 or hot_count == 0:
                        score = 0.0
                    else:
                        coverage_ratio = min(1.0, hot_count / cycle_count)
                        score = coverage_ratio
                    confidence = 0.7 if hot_count > 5 else 0.4
                    evidence = {
                        "hots_generated": hot_count,
                        "cycle_count": cycle_count,
                        "coverage_ratio": hot_count / max(1, cycle_count),
                        "module_active": True,
                    }

                elif name == "metacognition":
                    stats = (
                        meta.get_statistics() if hasattr(meta, "get_statistics") else {}
                    )
                    introspections = stats.get("introspection_count", 0)
                    belief_evaluations = stats.get("belief_evaluations", 0)
                    # The module reported a constant 0.5 before it measured it
                    overall_confidence = _none_as(
                        stats.get("overall_confidence", 0.0), 0.5
                    )
                    gw = kwargs.get("global_workspace")
                    gw_stats = (
                        gw.get_statistics()
                        if gw is not None and hasattr(gw, "get_statistics")
                        else {}
                    )
                    cycle_count = gw_stats.get("cycle_count", 0)
                    if introspections + belief_evaluations == 0:
                        score = 0.0
                    else:
                        activity_rate = min(
                            1.0,
                            (introspections + belief_evaluations) / max(1, cycle_count),
                        )
                        quality = overall_confidence
                        score = 0.4 * activity_rate + 0.6 * quality
                    confidence = 0.6 if introspections > 2 else 0.3
                    evidence = {
                        "introspections": introspections,
                        "evaluations": belief_evaluations,
                        "overall_confidence": overall_confidence,
                        "module_active": True,
                    }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True, "reason": "metacognition not provided"}

        # FEP Indicators (from ActiveInferenceModule)
        elif theory == ConsciousnessTheory.FEP:
            ai = kwargs.get("active_inference")
            if ai is not None:
                if name == "prediction_error_minimization":
                    stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                    avg_error = stats.get("avg_prediction_error", 0.5)
                    error_trend = stats.get("prediction_error_trend", 0)
                    total_inferences = stats.get("total_inferences", 0)
                    if isinstance(avg_error, str):
                        avg_error = 0.5
                    if total_inferences < 2:
                        score = 0.0
                    elif isinstance(error_trend, (int, float)) and error_trend < 0:
                        trend_magnitude = min(1.0, abs(float(error_trend)) * 5)
                        level_score = max(0.0, 1.0 - float(avg_error))
                        score = 0.7 * trend_magnitude + 0.3 * level_score
                    else:
                        level_score = max(0.0, 1.0 - float(avg_error)) * 0.3
                        score = level_score
                    confidence = 0.6 if total_inferences > 5 else 0.3
                    evidence = {
                        "avg_error": avg_error,
                        "trend": error_trend,
                        "total_inferences": total_inferences,
                        "module_active": True,
                    }

                elif name == "hierarchical_prediction":
                    stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                    hierarchical_error = stats.get("hierarchical_prediction_error", 0.5)
                    total_inferences = stats.get("total_inferences", 0)
                    warmup_cycles = kwargs.get("warmup_cycles", 25)
                    if total_inferences == 0:
                        score = 0.0
                    else:
                        if isinstance(hierarchical_error, (int, float)):
                            error_quality = max(0.0, 1.0 - float(hierarchical_error))
                        else:
                            error_quality = 0.0
                        activity_rate = min(
                            1.0, total_inferences / max(1, warmup_cycles)
                        )
                        score = 0.7 * error_quality + 0.3 * activity_rate
                    confidence = 0.6 if total_inferences > 5 else 0.3
                    evidence = {
                        "hierarchical_error": hierarchical_error,
                        "inferences": total_inferences,
                        "module_active": True,
                    }

                elif name == "temporal_prediction_depth":
                    fep = kwargs.get("active_inference")
                    if fep is not None:
                        horizon = (
                            fep.config.planning_horizon if hasattr(fep, "config") else 3
                        )
                        stats = (
                            fep.get_statistics()
                            if hasattr(fep, "get_statistics")
                            else {}
                        )
                        total_inferences = stats.get("total_inferences", 0)
                        avg_error = stats.get("avg_prediction_error", 0.5)
                        if isinstance(avg_error, str):
                            avg_error = 0.5
                        if total_inferences == 0:
                            score = 0.0
                        else:
                            horizon_factor = min(1.0, horizon / 10.0)
                            quality = max(0.0, 1.0 - float(avg_error))
                            score = horizon_factor * quality
                        confidence = 0.5 + 0.2 * min(1.0, total_inferences / 10)
                        evidence = {
                            "planning_horizon": horizon,
                            "total_inferences": total_inferences,
                            "avg_error": avg_error,
                        }
                    else:
                        score, confidence = 0.0, 0.3
                        evidence = {"module_absent": True}

                elif name == "agent_environment_coupling":
                    fep = kwargs.get("active_inference")
                    if fep is not None:
                        stats = (
                            fep.get_statistics()
                            if hasattr(fep, "get_statistics")
                            else {}
                        )
                        warmup_cycles = kwargs.get("warmup_cycles", 25)
                        action_count = stats.get("total_inferences", 0)
                        observation_count = stats.get("total_model_updates", 0)
                        if action_count == 0 and observation_count == 0:
                            score = 0.0
                        else:
                            act_rate = min(1.0, action_count / max(1, warmup_cycles))
                            obs_rate = min(
                                1.0, observation_count / max(1, warmup_cycles)
                            )
                            score = (act_rate * obs_rate) ** 0.5
                        confidence = 0.5 + 0.3 * min(
                            1.0, (action_count + observation_count) / 20
                        )
                        evidence = {
                            "actions": action_count,
                            "observations": observation_count,
                        }
                    else:
                        score, confidence = 0.0, 0.3
                        evidence = {"module_absent": True}

                elif name == "agency":
                    stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                    warmup_cycles = kwargs.get("warmup_cycles", 25)
                    inferences = stats.get("total_inferences", 0)
                    model_updates = stats.get("total_model_updates", 0)
                    free_energy = stats.get("variational_free_energy", 0.0)
                    if inferences == 0:
                        score = 0.0
                    else:
                        inference_rate = min(1.0, inferences / max(1, warmup_cycles))
                        update_rate = min(1.0, model_updates / max(1, inferences))
                        drive_score = (
                            min(0.3, 0.3 * (1.0 - 1.0 / (1.0 + float(free_energy))))
                            if isinstance(free_energy, (int, float)) and free_energy > 0
                            else 0.0
                        )
                        score = min(
                            1.0,
                            0.4 * inference_rate
                            + 0.3 * update_rate
                            + 0.3 * drive_score,
                        )
                    confidence = 0.6 if inferences > 5 else 0.3
                    evidence = {
                        "inferences": inferences,
                        "model_updates": model_updates,
                        "free_energy": free_energy,
                        "module_active": True,
                    }

                elif name == "affect_coherence":
                    pad = kwargs.get("pad_affect_model")
                    if pad is not None:
                        score = pad.compute_coherence()
                        if score is None:
                            score = 0.3  # v0.4.0 warm-up value
                        confidence = 0.5 + 0.3 * min(1.0, len(pad._history) / 20)
                        latest = pad.get_latest()
                        evidence = {
                            "coherence_score": score,
                            "history_length": len(pad._history),
                            "latest_pad": {
                                "pleasure": latest.pleasure if latest else None,
                                "arousal": latest.arousal if latest else None,
                                "dominance": latest.dominance if latest else None,
                                "label": latest.label if latest else None,
                            },
                            "module_active": True,
                        }
                    else:
                        score = 0.1
                        confidence = 0.3
                        evidence = {
                            "ablated": True,
                            "reason": "pad_affect_model not provided",
                        }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True, "reason": "active_inference not provided"}

        # IIT Indicators (from PhiCalculator)
        elif theory == ConsciousnessTheory.IIT:
            neural = kwargs.get("neural_states", {})
            if name == "integrated_information":
                phi_val = kwargs.get("phi_value", 0.0)
                if phi_val > 0:
                    score = min(1.0, phi_val)
                    confidence = 0.7
                    evidence = {
                        "phi_normalized": phi_val,
                        "source": "pre_computed_phi",
                    }
                else:
                    score = 0.0
                    confidence = 0.4
                    evidence = {"phi_zero": True, "neural_present": bool(neural)}

            elif name == "irreducibility":
                if neural and "connectivity" in neural:
                    score, evidence = self._irreducibility_score(neural["connectivity"])
                else:
                    score = 0.0
                    evidence = {"connectivity_missing": True}
                confidence = 0.5

            elif name == "causal_emergence":
                ce_analyzer = kwargs.get("causal_emergence_analyzer")
                if ce_analyzer is not None:
                    ce_val = ce_analyzer.get_latest_ce()
                    if ce_val is not None:
                        score = 1.0 / (1.0 + np.exp(-ce_val * 2.0))
                        confidence = 0.7
                        evidence = {
                            "causal_emergence": ce_val,
                            "macro_beats_micro": ce_val > 0,
                            "module_active": True,
                        }
                    else:
                        score = 0.0
                        confidence = 0.2
                        evidence = {"warming_up": True}
                else:
                    score = 0.0
                    confidence = 0.3
                    evidence = {"ablated": True, "reason": "analyzer not provided"}

        # RPT Indicators (from RPTMeasurement -- Lamme 2006)
        elif theory == ConsciousnessTheory.RPT:
            rpt = kwargs.get("rpt_measurement")
            gw = kwargs.get("global_workspace")
            if name == "recurrent_processing":
                if gw is not None:
                    stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                    broadcast_stats = stats.get("broadcast", {})
                    coverage = _none_as(broadcast_stats.get("coverage_ratio", 0.0), 0.0)
                    cycle_count = stats.get("cycle_count", 0)
                    broadcasts = broadcast_stats.get("total_broadcasts", 0)
                    if cycle_count == 0:
                        score = 0.0
                    else:
                        broadcast_rate = min(1.0, broadcasts / cycle_count)
                        score = 0.5 * coverage + 0.5 * broadcast_rate
                    confidence = 0.5 + 0.2 * min(1.0, cycle_count / 10)
                    evidence = {
                        "coverage": coverage,
                        "cycles": cycle_count,
                        "broadcasts": broadcasts,
                        "module_active": True,
                    }
                else:
                    score = 0.1
                    confidence = 0.3
                    evidence = {
                        "ablated": True,
                        "reason": "global_workspace not provided",
                    }
            elif rpt is not None:
                if name == "algorithmic_recurrence":
                    stats = (
                        rpt.get_statistics() if hasattr(rpt, "get_statistics") else {}
                    )
                    avg_local = _none_as(stats.get("average_local", 0.0), 0.0)
                    avg_global = _none_as(stats.get("average_global", 0.0), 0.0)
                    measurement_count = stats.get("measurement_count", 0)
                    if measurement_count == 0:
                        score = 0.0
                    else:
                        score = 0.6 * avg_local + 0.4 * avg_global
                    confidence = 0.5 + 0.3 * min(1.0, measurement_count / 10)
                    evidence = {
                        "avg_local": avg_local,
                        "avg_global": avg_global,
                        "measurements": measurement_count,
                        "module_active": True,
                    }

                elif name == "local_recurrence":
                    stats = (
                        rpt.get_statistics() if hasattr(rpt, "get_statistics") else {}
                    )
                    avg_local = _none_as(stats.get("average_local", 0.0), 0.0)
                    measurement_count = stats.get("measurement_count", 0)
                    score = avg_local if measurement_count > 0 else 0.0
                    confidence = 0.5 + 0.2 * min(1.0, measurement_count / 10)
                    evidence = {
                        "avg_local_recurrence": avg_local,
                        "measurements": measurement_count,
                        "module_active": True,
                    }

                elif name == "oscillatory_binding_coherence":
                    osc_stats = (
                        rpt.get_oscillatory_binding_stats()
                        if hasattr(rpt, "get_oscillatory_binding_stats")
                        else {}
                    )
                    global_r = osc_stats.get("global_order_parameter", 0.0)
                    cross_vals = osc_stats.get("cross_coherence", {})
                    binding_count = osc_stats.get("binding_count", 0)
                    if cross_vals:
                        mean_cross = max(0.0, float(np.mean(list(cross_vals.values()))))
                    else:
                        mean_cross = 0.0
                    if binding_count == 0:
                        score = 0.0
                    else:
                        score = 0.6 * global_r + 0.4 * mean_cross
                    confidence = 0.5 + 0.3 * min(1.0, binding_count / 10)
                    evidence = {
                        "global_order_parameter": global_r,
                        "cross_coherence": cross_vals,
                        "mean_cross_coherence": mean_cross,
                        "binding_count": binding_count,
                        "module_active": bool(osc_stats),
                    }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True, "reason": "rpt_measurement not provided"}

        # Beautiful Loop Theory Indicators (from BeautifulLoop -- Laukkonen et al. 2025)
        elif theory == ConsciousnessTheory.BLT:
            bl = kwargs.get("beautiful_loop")
            if bl is not None:
                stats = (
                    bl.get_loop_statistics()
                    if hasattr(bl, "get_loop_statistics")
                    else {}
                )

                if name == "bayesian_binding_quality":
                    avg_binding = _none_as(
                        stats.get("average_binding_quality", 0.0), 0.0
                    )
                    cycle_count = stats.get("cycle_count", 0)
                    score = avg_binding if cycle_count > 0 else 0.0
                    confidence = 0.5 + 0.3 * min(1.0, cycle_count / 10)
                    evidence = {
                        "avg_binding": avg_binding,
                        "cycles": cycle_count,
                        "module_active": True,
                    }

                elif name == "epistemic_depth":
                    depth_stats = stats.get("epistemic_depth", {})
                    avg_depth = depth_stats.get("average_depth", 0)
                    max_depth = depth_stats.get("max_depth_reached", 0)
                    score = min(1.0, avg_depth / 4.0) if avg_depth > 0 else 0.0
                    confidence = 0.6 if max_depth >= 2 else 0.3
                    evidence = {
                        "avg_depth": avg_depth,
                        "max_depth": max_depth,
                        "module_active": True,
                    }

                elif name == "genuine_implementation":
                    avg_quality = _none_as(stats.get("average_loop_quality", 0.0), 0.0)
                    fe_count = stats.get("field_evidencing_count", 0)
                    cycle_count = stats.get("cycle_count", 0)
                    depth_stats = stats.get("epistemic_depth", {})

                    if cycle_count == 0:
                        score = 0.0
                    else:
                        has_field_evidencing = fe_count > 0
                        has_loop_quality = avg_quality >= 0.4
                        has_strange_loop = depth_stats.get("max_depth_reached", 0) >= 2
                        conditions_met = sum(
                            [has_field_evidencing, has_loop_quality, has_strange_loop]
                        )
                        score = conditions_met / 3.0
                    confidence = 0.5 + 0.3 * min(1.0, cycle_count / 20)
                    evidence = {
                        "avg_loop_quality": avg_quality,
                        "field_evidencing_count": fe_count,
                        "has_field_evidencing": fe_count > 0,
                        "has_loop_quality": avg_quality >= 0.4,
                        "has_strange_loop": depth_stats.get("max_depth_reached", 0)
                        >= 2,
                        "conditions_met": sum(
                            [
                                fe_count > 0,
                                avg_quality >= 0.4,
                                depth_stats.get("max_depth_reached", 0) >= 2,
                            ]
                        ),
                        "module_active": True,
                    }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True, "reason": "beautiful_loop not provided"}

        # AE Indicators (Agency & Embodiment - Butlin 2025)
        elif theory == ConsciousnessTheory.AE:
            ai = kwargs.get("active_inference")
            ast = kwargs.get("attention_schema")
            if name == "agency":
                if ai is not None:
                    stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                    inferences = stats.get("total_inferences", 0)
                    model_updates = stats.get("total_model_updates", 0)
                    urgency = stats.get("urgency_level", 0.0)
                    free_energy = stats.get("variational_free_energy", 0.0)
                    base_score = 0.4
                    inference_score = (
                        min(0.6, inferences / 10) if inferences > 0 else 0.0
                    )
                    drive_score = (
                        min(0.3, 0.3 * (1.0 - 1.0 / (1.0 + float(free_energy))))
                        if isinstance(free_energy, (int, float)) and free_energy > 0
                        else 0.0
                    )
                    urgency_bonus = (
                        min(0.1, float(urgency) * 0.2)
                        if isinstance(urgency, (int, float))
                        else 0.0
                    )
                    score = min(
                        1.0,
                        max(base_score, inference_score + drive_score + urgency_bonus),
                    )
                    confidence = 0.6 if inferences > 5 else 0.4
                    evidence = {
                        "inferences": inferences,
                        "model_updates": model_updates,
                        "urgency": urgency,
                        "free_energy": free_energy,
                        "module_active": True,
                    }
                else:
                    score = 0.1
                    confidence = 0.3
                    evidence = {
                        "ablated": True,
                        "reason": "active_inference not provided",
                    }
            elif name == "embodiment":
                if ast is not None:
                    stats = (
                        ast.get_statistics() if hasattr(ast, "get_statistics") else {}
                    )
                    base_score = 0.4
                    capacity = stats.get("capacity_used", 0.0)
                    history_len = stats.get("history_length", 0)
                    vol_ratio = _none_as(stats.get("voluntary_ratio", 0.0), 0.5)
                    capacity_score = min(1.0, capacity * 2) if capacity > 0 else 0.3
                    perspective_score = min(1.0, history_len / 5)
                    agency_score = vol_ratio
                    score = max(
                        base_score,
                        0.3 * capacity_score
                        + 0.4 * perspective_score
                        + 0.3 * agency_score,
                    )
                    confidence = 0.5 + 0.2 * min(1.0, history_len / 5)
                    evidence = {
                        "capacity_used": capacity,
                        "history_length": history_len,
                        "voluntary_ratio": vol_ratio,
                        "module_active": True,
                    }
                else:
                    score = 0.1
                    confidence = 0.3
                    evidence = {
                        "ablated": True,
                        "reason": "attention_schema not provided",
                    }

        # New indicators (refined measurements / Butlin 2025 additions)
        if name == "sparse_smooth_coding":
            ai = kwargs.get("active_inference")
            if ai is not None:
                stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                avg_error = stats.get("avg_prediction_error", 0.5)
                total_inferences = stats.get("total_inferences", 0)
                if isinstance(avg_error, str):
                    avg_error = 0.5
                if total_inferences == 0:
                    score = 0.0
                else:
                    sparsity_score = max(0.0, 1.0 - float(avg_error) * 2.0)
                    score = sparsity_score
                confidence = 0.5
                evidence = {
                    "avg_error": avg_error,
                    "sparsity_proxy": score,
                    "module_active": True,
                }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True}

        elif name == "global_ignition_nuanced":
            gw = kwargs.get("global_workspace")
            if gw is not None:
                stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                ignition_stats = stats.get("ignition", {})
                ignitions = ignition_stats.get("total_ignitions", 0)
                sustain_rate = _none_as(ignition_stats.get("sustain_rate", 0.0), 0.0)
                decay_rate = ignition_stats.get("decay_rate", 0.0)
                cycle_count = stats.get("cycle_count", 0)
                if ignitions == 0:
                    score = 0.0
                else:
                    ignition_rate = min(1.0, ignitions / max(1, cycle_count))
                    dynamics_quality = sustain_rate * 0.7 + (1.0 - decay_rate) * 0.3
                    score = 0.5 * ignition_rate + 0.5 * dynamics_quality
                confidence = 0.6 if ignitions > 3 else 0.4
                evidence = {
                    "ignitions": ignitions,
                    "sustain_rate": sustain_rate,
                    "decay_rate": decay_rate,
                    "module_active": True,
                }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True}

        elif name == "sequential_module_querying":
            gw = kwargs.get("global_workspace")
            if gw is not None:
                stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                cycle_count = stats.get("cycle_count", 0)
                competition_stats = stats.get("competition", {})
                total_candidates = competition_stats.get("total_candidates", 0)
                base_score = 0.35
                cycle_score = min(1.0, cycle_count / 5)
                diversity_score = min(1.0, total_candidates / 10)
                score = max(base_score, 0.5 * cycle_score + 0.5 * diversity_score)
                confidence = 0.5 + 0.3 * min(1.0, cycle_count / 5)
                evidence = {
                    "cycles": cycle_count,
                    "total_candidates": total_candidates,
                    "module_active": True,
                }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True}

        elif name == "organized_percepts":
            gw = kwargs.get("global_workspace")
            rpt = kwargs.get("rpt_measurement")
            if gw is not None or rpt is not None:
                base_score = 0.3
                gw_stats = (
                    gw.get_statistics() if gw and hasattr(gw, "get_statistics") else {}
                )
                comp_stats = gw_stats.get("competition", {})
                source_counts = comp_stats.get("source_counts", {})
                perceptual_count = sum(
                    source_counts.get(s, 0) for s in ["snn", "lsm", "sensory", "vision"]
                )
                perceptual_score = (
                    min(1.0, perceptual_count / 5) if perceptual_count > 0 else 0.0
                )
                rpt_stats = (
                    rpt.get_statistics()
                    if rpt and hasattr(rpt, "get_statistics")
                    else {}
                )
                avg_local = _none_as(rpt_stats.get("average_local", 0.0), 0.0)
                score = max(base_score, 0.5 * perceptual_score + 0.5 * avg_local)
                confidence = 0.4 + 0.2 * min(1.0, perceptual_count / 3)
                evidence = {
                    "perceptual_candidates": perceptual_count,
                    "avg_local_recurrence": avg_local,
                    "module_active": True,
                }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True}

        elif name == "flexible_responsiveness":
            gw = kwargs.get("global_workspace")
            ai = kwargs.get("active_inference")
            if gw is not None:
                try:
                    gw_stats = (
                        gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                    )
                    comp_stats = gw_stats.get("competition", {})
                    source_counts = comp_stats.get("source_counts", {})
                    num_sources = len([s for s, c in source_counts.items() if c > 0])
                    total_sources = max(1, len(source_counts)) if source_counts else 1
                    diversity_ratio = min(1.0, num_sources / max(3, total_sources))
                    belief_updates = 0.0
                    if ai is not None:
                        try:
                            ai_stats = (
                                ai.get_statistics()
                                if hasattr(ai, "get_statistics")
                                else {}
                            )
                            model_updates = ai_stats.get("total_model_updates", 0)
                            belief_updates = (
                                min(1.0, model_updates / 10)
                                if model_updates > 0
                                else 0.0
                            )
                        except Exception:
                            pass
                    base_score = 0.3
                    computed = diversity_ratio * 0.5 + belief_updates * 0.5
                    score = max(base_score, computed)
                    confidence = 0.5
                    evidence = {
                        "source_diversity": num_sources,
                        "belief_updates": round(belief_updates, 3),
                        "module_active": True,
                    }
                except Exception:
                    score = 0.3
                    confidence = 0.4
                    evidence = {"module_present": True}
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True}

        elif name == "online_reportable_learning":
            meta = kwargs.get("metacognition")
            ai = kwargs.get("active_inference")
            if meta is None and ai is None:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True}
            else:
                active_beliefs = 0
                has_recent = 0.0
                lr_nonzero = 0.0
                try:
                    if ai is not None:
                        ai_stats = (
                            ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                        )
                        model_updates = ai_stats.get("total_model_updates", 0)
                        if model_updates > 0:
                            active_beliefs = model_updates
                        lr = getattr(ai, "learning_rate", 0.0)
                        if (
                            lr is not None
                            and isinstance(lr, (int, float))
                            and float(lr) > 0
                        ):
                            lr_nonzero = 1.0
                    if meta is not None:
                        meta_stats = (
                            meta.get_statistics()
                            if hasattr(meta, "get_statistics")
                            else {}
                        )
                        introspections = meta_stats.get("introspection_count", 0)
                        if introspections > 0:
                            has_recent = min(1.0, introspections / 5)
                    base_score = 0.25
                    belief_score = min(1.0, active_beliefs / 10) * 0.5
                    experience_score = has_recent * 0.3
                    lr_score = lr_nonzero * 0.2
                    score = max(base_score, belief_score + experience_score + lr_score)
                    confidence = 0.4 + 0.2 * min(1.0, active_beliefs / 5)
                    evidence = {
                        "active_beliefs": active_beliefs,
                        "has_recent_experiences": has_recent > 0,
                        "module_active": True,
                    }
                except Exception:
                    score = 0.2
                    confidence = 0.3
                    evidence = {"error": "measurement failed"}

        elif name == "complex_agency":
            ai = kwargs.get("active_inference")
            if ai is not None:
                try:
                    ai_stats = (
                        ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                    )
                    num_policies = ai_stats.get("active_policies", 1)
                    policy_diversity = (
                        min(1.0, num_policies / 5) if num_policies > 1 else 0.2
                    )
                    avg_error = ai_stats.get("avg_prediction_error", 0.0)
                    prediction_variance = (
                        0.3
                        if isinstance(avg_error, (int, float)) and float(avg_error) > 0
                        else 0.0
                    )
                    base_score = 0.25
                    computed = policy_diversity * 0.5 + prediction_variance * 0.5
                    score = max(base_score, computed)
                    confidence = 0.5
                    evidence = {
                        "policy_diversity": round(policy_diversity, 3),
                        "prediction_variance": round(prediction_variance, 3),
                        "module_active": True,
                    }
                except Exception:
                    score = 0.25
                    confidence = 0.4
                    evidence = {"module_present": True}
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True}

        return score, confidence, evidence


# ============================================================================
# MODULE-LEVEL HELPERS FOR THE ASSESSMENT
# ============================================================================


def _check_choice(field_name: str, value: Any, choices: Tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {choices}, got {value!r}")


def _metacognition_ran(stats: Dict[str, Any]) -> bool:
    """A metacognition module ran once it saw a first-order state or a HOT."""
    return (
        _count(stats.get("total_hots_generated")) > 0
        or _count(stats.get("introspection_count")) > 0
        or _count(stats.get("first_order_buffer_size")) > 0
    )


def _module_ran(module_key: str, module: Any) -> bool:
    """Whether a supplied module has produced any evidence yet."""
    if module_key == "global_workspace":
        return _count(_statistics(module).get("cycle_count")) > 0
    if module_key == "attention_schema":
        stats = _statistics(module)
        return (
            _count(stats.get("total_updates")) > 0
            or _count(stats.get("history_length")) > 0
        )
    if module_key == "metacognition":
        return _metacognition_ran(_statistics(module))
    if module_key == "active_inference":
        return _count(_statistics(module).get("total_inferences")) > 0
    if module_key == "rpt_measurement":
        return _count(_statistics(module).get("measurement_count")) > 0
    if module_key == "beautiful_loop":
        return _count(_statistics(module, "get_loop_statistics").get("cycle_count")) > 0
    return True


def _legacy_status(evidence: Dict[str, Any]) -> str:
    """Status of a v0.4.0 result: absent when its module was not supplied."""
    if evidence.get("ablated") or evidence.get("module_absent"):
        return ABSENT
    return MEASURED


# Indicator name -> the module whose presence and activity gate it (v04 path)
_GATE_MODULES = {
    "global_broadcast": "global_workspace",
    "ignition_dynamics": "global_workspace",
    "global_ignition_nuanced": "global_workspace",
    "sequential_module_querying": "global_workspace",
    "flexible_responsiveness": "global_workspace",
    "recurrent_processing": "global_workspace",
    "attention_schema": "attention_schema",
    "attention_control": "attention_schema",
    "embodiment": "attention_schema",
    "higher_order_representations": "metacognition",
    "metacognition": "metacognition",
    "sparse_smooth_coding": "active_inference",
    "prediction_error_minimization": "active_inference",
    "hierarchical_prediction": "active_inference",
    "temporal_prediction_depth": "active_inference",
    "agent_environment_coupling": "active_inference",
    "complex_agency": "active_inference",
    "agency": "active_inference",
    "algorithmic_recurrence": "rpt_measurement",
    "local_recurrence": "rpt_measurement",
    "oscillatory_binding_coherence": "rpt_measurement",
    "bayesian_binding_quality": "beautiful_loop",
    "epistemic_depth": "beautiful_loop",
    "genuine_implementation": "beautiful_loop",
}

# Indicator name -> the v05 measurement method
_V05_MEASURES = {
    "global_broadcast": "_v05_global_broadcast",
    "ignition_dynamics": "_v05_ignition_dynamics",
    "global_ignition_nuanced": "_v05_global_ignition_nuanced",
    "attention_schema": "_v05_attention_schema",
    "attention_control": "_v05_attention_control",
    "higher_order_representations": "_v05_higher_order_representations",
    "metacognition": "_v05_metacognition",
    "sparse_smooth_coding": "_v05_sparse_smooth_coding",
    "prediction_error_minimization": "_v05_prediction_error_minimization",
    "hierarchical_prediction": "_v05_hierarchical_prediction",
    "temporal_prediction_depth": "_v05_temporal_prediction_depth",
    "agent_environment_coupling": "_v05_agent_environment_coupling",
    "affect_coherence": "_v05_affect_coherence",
    "integrated_information": "_v05_integrated_information",
    "irreducibility": "_v05_irreducibility",
    "causal_emergence": "_v05_causal_emergence",
    "recurrent_processing": "_v05_recurrent_processing",
    "algorithmic_recurrence": "_v05_algorithmic_recurrence",
    "local_recurrence": "_v05_local_recurrence",
    "oscillatory_binding_coherence": "_v05_oscillatory_binding_coherence",
    "agency": "_v05_agency",
    "embodiment": "_v05_embodiment",
    "bayesian_binding_quality": "_v05_bayesian_binding_quality",
    "epistemic_depth": "_v05_epistemic_depth",
    "genuine_implementation": "_v05_genuine_implementation",
    "sequential_module_querying": "_v05_sequential_module_querying",
    "organized_percepts": "_v05_organized_percepts",
    "flexible_responsiveness": "_v05_flexible_responsiveness",
    "online_reportable_learning": "_v05_online_reportable_learning",
    "complex_agency": "_v05_complex_agency",
}


# ============================================================================
# LONGITUDINAL STUDY FRAMEWORK
# ============================================================================


# The modules a system-state dict may carry, forwarded by name to
# run_full_assessment.
STUDY_MODULE_KEYS = (
    "global_workspace",
    "attention_schema",
    "metacognition",
    "active_inference",
    "neural_states",
    "beautiful_loop",
    "rpt_measurement",
    "pad_affect_model",
    "causal_emergence_analyzer",
    "phi_tracker",
)


def _study_modules(system_state: Dict[str, Any]) -> Dict[str, Any]:
    """The assessment keywords present in a system-state dict."""
    return {key: system_state.get(key) for key in STUDY_MODULE_KEYS}


def _trend(values: Sequence[float]) -> float:
    if len(values) > 1:
        return float(np.polyfit(np.arange(len(values)), values, 1)[0])
    return 0.0


@dataclass
class StudyMeasurement:
    """Single measurement in a longitudinal study.

    consciousness_score is None when the assessment had nothing to
    measure; phi is None when no Phi was computed.
    """

    timestamp: datetime
    study_day: float
    consciousness_score: Optional[float]
    phi: Optional[float]
    passing_indicators: int
    full_assessment: ConsciousnessReport


@dataclass
class StudyResults:
    """Results from a longitudinal consciousness study."""

    study_id: str
    start_time: datetime
    end_time: datetime
    duration_days: float
    total_measurements: int

    # Score statistics (None when no measurement produced an overall score)
    consciousness_mean: Optional[float]
    consciousness_std: Optional[float]
    consciousness_trend: Optional[float]  # Positive = improving

    # Phi statistics (None when no measurement produced Phi)
    phi_mean: Optional[float]
    phi_std: Optional[float]
    phi_trend: Optional[float]

    # Emergence detection
    emergence_detected: bool
    emergence_day: Optional[float]

    # All measurements
    measurements: List[StudyMeasurement]


class LongitudinalStudy:
    """
    Framework for running longitudinal consciousness studies.

    Tracks consciousness metrics over time to:
    1. Establish baselines
    2. Detect consciousness emergence
    3. Validate theories
    4. Generate publishable data

    Results go to ``output_dir`` when one is given, else to
    ``Settings.LONGITUDINAL_OUTPUT_DIR`` under ``PROJECT_ROOT``, resolved at
    write time; no directory is created before the first write.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize longitudinal study framework."""
        self._output_dir = Path(output_dir) if output_dir is not None else None

        self.assessment = ConsciousnessAssessment()
        self.study_id = str(uuid.uuid4())[:8]
        self.measurements: List[StudyMeasurement] = []
        self.start_time: Optional[datetime] = None
        self.is_running = False

        logger.info(f"Longitudinal Study Framework initialized (ID: {self.study_id})")

    @property
    def output_dir(self) -> Path:
        """Where results are written (resolved now, created at the first write)."""
        if self._output_dir is not None:
            return self._output_dir
        settings = get_settings()
        return resolve_path(settings.LONGITUDINAL_OUTPUT_DIR, settings=settings)

    async def run_study(
        self,
        duration_days: int = 7,
        measurement_interval_hours: float = 4.0,
        get_system_state: Optional[callable] = None,
    ) -> StudyResults:
        """
        Run a longitudinal consciousness study.

        Args:
            duration_days: How long to run the study
            measurement_interval_hours: How often to measure
            get_system_state: Callable that returns current system state

        Returns:
            Complete study results
        """
        self.start_time = datetime.now()
        self.is_running = True
        end_time = time.time() + (duration_days * 24 * 60 * 60)
        interval_seconds = measurement_interval_hours * 60 * 60

        logger.info(f"Starting longitudinal study {self.study_id}")
        logger.info(f"Duration: {duration_days} days")
        logger.info(f"Measurement interval: {measurement_interval_hours} hours")

        measurement_count = 0

        while time.time() < end_time and self.is_running:
            try:
                # Get current system state
                system_state = {}
                if get_system_state:
                    system_state = await get_system_state()

                # Take measurement
                measurement = await self._take_measurement(system_state)
                self.measurements.append(measurement)
                measurement_count += 1

                # Log progress
                elapsed_days = (time.time() - self.start_time.timestamp()) / (
                    24 * 60 * 60
                )
                logger.info(
                    f"Study day {elapsed_days:.1f}: "
                    f"Consciousness score = {_fmt(measurement.consciousness_score)}, "
                    f"Phi = {_fmt(measurement.phi)}"
                )

                # Save intermediate results
                if measurement_count % 6 == 0:  # Every ~24 hours
                    await self._save_intermediate_results()

            except Exception as e:
                logger.error(f"Measurement failed: {e}")

            # Wait for next measurement
            await asyncio.sleep(interval_seconds)

        # Analyze final results
        results = self._analyze_results()

        # Save final results
        await self._save_final_results(results)

        self.is_running = False
        return results

    def stop_study(self):
        """Stop a running study."""
        self.is_running = False
        logger.info(f"Study {self.study_id} stopped")

    async def _take_measurement(self, system_state: Dict[str, Any]) -> StudyMeasurement:
        """Take a single measurement."""
        # Run full assessment with every module the system state carries
        assessment = await self.assessment.run_full_assessment(
            **_study_modules(system_state)
        )

        phi = assessment.phi_measurement.phi if assessment.phi_measurement else None

        return StudyMeasurement(
            timestamp=datetime.now(),
            study_day=(time.time() - self.start_time.timestamp()) / (24 * 60 * 60),
            consciousness_score=assessment.overall_score,
            phi=phi,
            passing_indicators=assessment.passing_count,
            full_assessment=assessment,
        )

    def _analyze_results(self) -> StudyResults:
        """Analyze longitudinal study results."""
        if not self.measurements:
            raise ValueError("No measurements to analyze!")

        # A None overall or Phi is a missing value, not a zero
        scored = [
            (m.study_day, m.consciousness_score)
            for m in self.measurements
            if m.consciousness_score is not None
        ]
        scores = [score for _day, score in scored]
        phis = [m.phi for m in self.measurements if m.phi is not None]

        # Detect emergence (sustained increase in consciousness indicators)
        emergence_detected, emergence_day = self._detect_emergence(scored)

        return StudyResults(
            study_id=self.study_id,
            start_time=self.start_time,
            end_time=datetime.now(),
            duration_days=(time.time() - self.start_time.timestamp()) / (24 * 60 * 60),
            total_measurements=len(self.measurements),
            consciousness_mean=float(np.mean(scores)) if scores else None,
            consciousness_std=float(np.std(scores)) if scores else None,
            consciousness_trend=_trend(scores) if scores else None,
            phi_mean=float(np.mean(phis)) if phis else None,
            phi_std=float(np.std(phis)) if phis else None,
            phi_trend=_trend(phis) if phis else None,
            emergence_detected=emergence_detected,
            emergence_day=emergence_day,
            measurements=self.measurements,
        )

    def _detect_emergence(
        self, scored: List[Tuple[float, float]]
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect if consciousness emergence occurred.

        Emergence = sustained shift from low to higher consciousness scores.
        scored holds (study_day, score) pairs of the measurements that
        produced an overall score.
        """
        if len(scored) < 5:
            return False, None

        # Use change point detection
        threshold = 0.5
        scores = [score for _day, score in scored]

        # Find first point where score crosses threshold and stays above
        for i in range(len(scores) - 3):
            if scores[i] < threshold:
                # Check if next 3 measurements are above threshold
                if all(s >= threshold for s in scores[i + 1 : i + 4]):
                    emergence_day = scored[i + 1][0]
                    return True, emergence_day

        return False, None

    async def _save_intermediate_results(self):
        """Save intermediate results for safety."""
        directory = ensure_dir(self.output_dir)
        filepath = directory / f"study_{self.study_id}_intermediate.json"

        data = {
            "study_id": self.study_id,
            "start_time": self.start_time.isoformat(),
            "measurements_count": len(self.measurements),
            "latest_score": (
                self.measurements[-1].consciousness_score if self.measurements else None
            ),
            "measurements": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "study_day": m.study_day,
                    "score": m.consciousness_score,
                    "phi": m.phi,
                    "passing": m.passing_indicators,
                }
                for m in self.measurements
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Intermediate results saved to {filepath}")

    async def _save_final_results(self, results: StudyResults):
        """Save final study results."""
        directory = ensure_dir(self.output_dir)
        filepath = directory / f"study_{self.study_id}_final.json"

        data = {
            "study_id": results.study_id,
            "start_time": results.start_time.isoformat(),
            "end_time": results.end_time.isoformat(),
            "duration_days": results.duration_days,
            "total_measurements": results.total_measurements,
            "consciousness": {
                "mean": results.consciousness_mean,
                "std": results.consciousness_std,
                "trend": results.consciousness_trend,
            },
            "phi": {
                "mean": results.phi_mean,
                "std": results.phi_std,
                "trend": results.phi_trend,
            },
            "emergence": {
                "detected": results.emergence_detected,
                "day": results.emergence_day,
            },
            "measurements": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "study_day": m.study_day,
                    "score": m.consciousness_score,
                    "phi": m.phi,
                    "passing": m.passing_indicators,
                }
                for m in results.measurements
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Final results saved to {filepath}")


# ============================================================================
# ABLATION STUDY INTEGRATION
# ============================================================================


class AblationStudy:
    """
    Run ablation studies using the full consciousness assessment.

    Tests the contribution of each consciousness module:
    - GWT (EnhancedGlobalWorkspace)
    - AST (AttentionSchemaModule)
    - HOT (MetacognitionModule)
    - FEP (ActiveInferenceModule)

    Every other module the system state carries (beautiful loop, RPT, PAD,
    causal emergence analyzer, PhiTracker, neural states) is forwarded too.

    Results go to ``output_dir`` when one is given, else to
    ``Settings.ABLATION_OUTPUT_DIR`` under ``PROJECT_ROOT``, resolved at write
    time; no directory is created before the first write.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize ablation study."""
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self.assessment = ConsciousnessAssessment()

        logger.info("Ablation Study Framework initialized")

    @property
    def output_dir(self) -> Path:
        """Where results are written (resolved now, created at the first write)."""
        if self._output_dir is not None:
            return self._output_dir
        settings = get_settings()
        return resolve_path(settings.ABLATION_OUTPUT_DIR, settings=settings)

    async def run_full_ablation(self, get_system_state: callable) -> Dict[str, Any]:
        """
        Run complete ablation study disabling each consciousness module.

        Args:
            get_system_state: Callable returning dict with all modules

        Returns:
            Complete ablation results with comparison
        """
        logger.info("Starting Full Ablation Study")
        logger.info("=" * 70)

        configurations = [
            ("Baseline", {}),  # All enabled
            ("No_GWT", {"global_workspace": False}),
            ("No_AST", {"attention_schema": False}),
            ("No_HOT", {"metacognition": False}),
            ("No_FEP", {"active_inference": False}),
            ("No_GWT_AST", {"global_workspace": False, "attention_schema": False}),
            ("No_HOT_FEP", {"metacognition": False, "active_inference": False}),
        ]

        results = {"timestamp": datetime.now().isoformat(), "configurations": []}

        for config_name, ablation_config in configurations:
            logger.info(f"\nRunning configuration: {config_name}")

            # Get system state
            state = await get_system_state()

            # Apply ablation
            for module, disabled in ablation_config.items():
                if disabled:
                    state[module] = None

            # Run assessment with every module the system state carries
            report = await self.assessment.run_full_assessment(
                **_study_modules(state),
                ablation_config=ablation_config,
            )

            results["configurations"].append(
                {
                    "name": config_name,
                    "ablation_config": ablation_config,
                    "overall_score": report.overall_score,
                    "status": report.status,
                    "coverage": report.coverage,
                    "architecture_functional": report.architecture_functional,
                    "passing_indicators": report.passing_count,
                    "theory_scores": {
                        k: float(v) for k, v in report.theory_scores.items()
                    },
                }
            )

        # Calculate impacts (a missing overall leaves the impact missing)
        baseline_score = results["configurations"][0]["overall_score"]
        results["impacts"] = []

        for config in results["configurations"][1:]:
            if baseline_score is None or config["overall_score"] is None:
                results["impacts"].append(
                    {
                        "configuration": config["name"],
                        "score_drop": None,
                        "impact_percentage": None,
                        "interpretation": INSUFFICIENT_DATA,
                    }
                )
                continue
            impact = baseline_score - config["overall_score"]
            impact_pct = (impact / baseline_score * 100) if baseline_score > 0 else 0

            results["impacts"].append(
                {
                    "configuration": config["name"],
                    "score_drop": impact,
                    "impact_percentage": impact_pct,
                    "interpretation": self._interpret_impact(impact_pct),
                }
            )

        # Save results
        self._save_results(results)

        return results

    def _interpret_impact(self, impact_pct: float) -> str:
        """Interpret ablation impact."""
        if impact_pct < 5:
            return "Minimal - Component not critical for consciousness"
        elif impact_pct < 15:
            return "Moderate - Component contributes to consciousness"
        elif impact_pct < 30:
            return "Significant - Component important for consciousness"
        else:
            return "Critical - Component essential for consciousness"

    def _save_results(self, results: Dict[str, Any]):
        """Save ablation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = ensure_dir(self.output_dir)
        filepath = directory / f"ablation_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Ablation results saved to {filepath}")


# ============================================================================
# NOISE-BASELINE MOCK MODULES
# ============================================================================
#
# These lightweight stubs return randomised statistics in the producers'
# schema (STATS_CONTRACT), plus the v0.4.0 keys the legacy formulas read, so
# the noise-baseline pass exercises every code path in _measure_indicator
# without real consciousness modules. Each stub reports activity (its
# counters start at one), so every indicator is measured on noise. Each call
# to get_statistics() returns freshly randomised values so successive noise
# iterations are not identical.


_NOISE_SOURCES = ["sensory", "memory", "concept", "snn", "lsm", "gist"]


class _NoiseCompetitionEvent:
    """Noise stand-in for a workspace CompetitionEvent."""

    def __init__(self):
        self.winning_sources = random.sample(_NOISE_SOURCES, random.randint(1, 3))


class _NoiseBottleneck:
    """Noise stand-in for the workspace's attention bottleneck."""

    def __init__(self):
        self.competition_history = [
            _NoiseCompetitionEvent() for _ in range(random.randint(2, 10))
        ]


class _NoiseGlobalWorkspace:
    """Noise stub for EnhancedGlobalWorkspace."""

    def __init__(self):
        self.bottleneck = _NoiseBottleneck()

    def get_statistics(self) -> Dict[str, Any]:
        sources = random.sample(_NOISE_SOURCES, random.randint(1, 3))
        return {
            "broadcast": {
                "total_broadcasts": random.randint(0, 5),
                "coverage_ratio": random.random() * 0.3,
            },
            "ignition": {
                "total_ignitions": random.randint(0, 3),
                "sustain_rate": random.random() * 0.2,
                "decay_rate": random.random() * 0.3,  # v0.4.0 key
            },
            "cycle_count": random.randint(1, 5),
            "bottleneck": {
                "total_competitions": random.randint(1, 5),
                "avg_candidates": random.uniform(1.0, 7.0),
                "source_distribution": {s: random.randint(1, 3) for s in sources},
            },
            "competition": {  # v0.4.0 key (never emitted by the workspace)
                "total_candidates": random.randint(0, 5),
                "source_counts": {},
            },
        }


class _NoiseAttentionSchema:
    """Noise stub for AttentionSchemaModule."""

    def __init__(self):
        self.schema = _NoiseSchema()

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "voluntary_shifts": random.randint(0, 2),
            "voluntary_ratio": random.random() * 0.3,
            "capacity_used": random.random() * 0.2,
            "history_length": random.randint(1, 3),
            "total_updates": random.randint(1, 3),
            "prediction_accuracy": random.random() * 0.3,
        }


class _NoiseSchema:
    """Noise stub for the inner attention schema object."""

    def __init__(self):
        self.current_focus = None  # No focus -- pure noise
        self.attention_capacity_used = random.random() * 0.2


class _NoiseMetacognition:
    """Noise stub for MetacognitionModule."""

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_hots_generated": random.randint(0, 3),
            "overall_confidence": random.random() * 0.3,
            "introspection_count": random.randint(0, 2),
            "first_order_buffer_size": random.randint(1, 3),
            "belief_evaluations": random.randint(0, 2),  # v0.4.0 key
        }


class _NoiseActiveInferenceConfig:
    """Noise stub for ActiveInferenceConfig."""

    def __init__(self):
        self.planning_horizon = random.randint(1, 2)


class _NoiseInferenceRecord:
    """Noise stand-in for an InferenceResult (its prediction error only)."""

    def __init__(self):
        self.prediction_error = 0.3 + random.random() * 0.5


class _NoiseAgent:
    """Noise stand-in for the pymdp agent (its policy list only)."""

    def __init__(self):
        self.policies = [None] * random.randint(1, 4)


class _NoiseDrive:
    def __init__(self):
        self._urgency = random.random() * 0.3

    def get_urgency(self) -> float:
        return self._urgency


class _NoiseDrives:
    def __init__(self):
        self.drives = {f"drive_{i}": _NoiseDrive() for i in range(3)}


class _NoiseActiveInference:
    """Noise stub for ActiveInferenceModule."""

    def __init__(self):
        self.config = _NoiseActiveInferenceConfig()
        self.agent = _NoiseAgent()
        self.homeostatic_drives = _NoiseDrives()
        # Short random histories for temporal depth and variance noise
        self.belief_history = [None] * random.randint(0, 3)
        self.free_energy_history = [
            random.random() * 0.5 for _ in range(random.randint(0, 3))
        ]
        self.inference_history = [
            _NoiseInferenceRecord() for _ in range(random.randint(10, 20))
        ]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "avg_prediction_error": 0.3 + random.random() * 0.5,  # 0.3-0.8
            "prediction_error_trend": random.choice(
                ["decreasing", "stable", "increasing"]
            ),
            "hierarchical_prediction_error": 0.3 + random.random() * 0.5,
            "total_inferences": random.randint(1, 3),
            "total_model_updates": random.randint(0, 2),
            "urgency_level": random.random() * 0.2,
            "variational_free_energy": random.random() * 0.5,
        }


class _NoiseBeautifulLoop:
    """Noise stub for BeautifulLoop."""

    def get_loop_statistics(self) -> Dict[str, Any]:
        peak = random.randint(0, 1)
        return {
            "cycle_count": random.randint(1, 3),
            "average_loop_quality": random.random() * 0.3,
            "average_binding_quality": random.random() * 0.3,
            "epistemic_depth": {
                "current_depth": peak,
                "average_depth": random.random() * 1.0,
                "peak_depth": peak,
                "max_depth_reached": peak,  # v0.4.0 key
                "measurement_count": random.randint(1, 3),
            },
            "field_evidencing_count": random.randint(0, 1),
            "field_evidencing_rate": random.random() * 0.1,
        }


class _NoiseRPTMeasurement:
    """Noise stub for RPTMeasurement."""

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "average_local": random.random() * 0.3,
            "average_global": random.random() * 0.2,
            "measurement_count": random.randint(1, 3),
        }

    def get_oscillatory_binding_stats(self) -> Dict[str, Any]:
        return {
            "global_order_parameter": random.random() * 0.5,
            "cross_coherence": {
                "snn_lsm": random.random(),
                "snn_htm": random.random(),
                "lsm_htm": random.random(),
            },
            "binding_count": random.randint(1, 5),
        }


class _NoisePADAffectModel:
    """Noise stub for PADAffectModel."""

    def __init__(self):
        self._history = [
            PADState(
                pleasure=random.random() * 2 - 1,
                arousal=random.random(),
                dominance=random.random(),
                label="Neutral",
            )
            for _ in range(random.randint(10, 20))
        ]

    @property
    def history_length(self) -> int:
        return len(self._history)

    def compute_coherence(self) -> float:
        return random.random() * 0.3

    def get_latest(self) -> PADState:
        return self._history[-1]


class _NoiseCausalEmergenceAnalyzer:
    """Noise stub for CausalEmergenceAnalyzer."""

    def get_latest_ce(self):
        return random.uniform(-0.5, 0.5)


# legacy_v04 compatibility, remove in v0.6.0.
#
# The v0.4.0 noise stubs, unchanged: counters drawn from zero, a numeric
# prediction_error_trend, the v0.4.0 key set only and the same random draws
# in the same order. Under indicator_formulas="v04" the noise baseline runs on
# these, so stub-normalized legacy numbers keep v0.4.0's inputs.


class _LegacyV04NoiseGlobalWorkspace:
    """v0.4.0 noise stub for EnhancedGlobalWorkspace."""

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "broadcast": {
                "total_broadcasts": random.randint(0, 5),
                "coverage_ratio": random.random() * 0.3,
            },
            "ignition": {
                "total_ignitions": random.randint(0, 3),
                "sustain_rate": random.random() * 0.2,
                "decay_rate": random.random() * 0.3,
            },
            "cycle_count": random.randint(0, 5),
            "competition": {
                "total_candidates": random.randint(0, 5),
                "source_counts": {},
            },
        }


class _LegacyV04NoiseAttentionSchema:
    """v0.4.0 noise stub for AttentionSchemaModule."""

    def __init__(self):
        self.schema = _NoiseSchema()

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "voluntary_shifts": random.randint(0, 2),
            "voluntary_ratio": random.random() * 0.3,
            "capacity_used": random.random() * 0.2,
            "history_length": random.randint(0, 3),
            "total_updates": random.randint(0, 3),
            "prediction_accuracy": random.random() * 0.3,
        }


class _LegacyV04NoiseMetacognition:
    """v0.4.0 noise stub for MetacognitionModule."""

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_hots_generated": random.randint(0, 3),
            "overall_confidence": random.random() * 0.3,
            "introspection_count": random.randint(0, 2),
            "belief_evaluations": random.randint(0, 2),
        }


class _LegacyV04NoiseActiveInference:
    """v0.4.0 noise stub for ActiveInferenceModule."""

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "avg_prediction_error": 0.3 + random.random() * 0.5,  # 0.3-0.8
            "prediction_error_trend": random.uniform(-0.1, 0.1),
            "hierarchical_prediction_error": 0.3 + random.random() * 0.5,
            "total_inferences": random.randint(0, 3),
            "total_model_updates": random.randint(0, 2),
            "urgency_level": random.random() * 0.2,
            "variational_free_energy": random.random() * 0.5,
        }


class _LegacyV04NoiseBeautifulLoop:
    """v0.4.0 noise stub for BeautifulLoop."""

    def get_loop_statistics(self) -> Dict[str, Any]:
        return {
            "cycle_count": random.randint(0, 3),
            "average_loop_quality": random.random() * 0.3,
            "average_binding_quality": random.random() * 0.3,
            "epistemic_depth": {
                "average_depth": random.random() * 1.0,
                "max_depth_reached": random.randint(0, 1),
            },
            "field_evidencing_count": random.randint(0, 1),
            "field_evidencing_rate": random.random() * 0.1,
        }


class _LegacyV04NoiseRPTMeasurement:
    """v0.4.0 noise stub for RPTMeasurement (no oscillatory stats method)."""

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "average_local": random.random() * 0.3,
            "average_global": random.random() * 0.2,
            "measurement_count": random.randint(0, 3),
        }


class _LegacyV04NoisePADAffectModel:
    """v0.4.0 noise stub for PADAffectModel."""

    def compute_coherence(self) -> float:
        return random.random() * 0.3

    def get_latest(self):
        return None

    @property
    def _history(self):
        return []


def _noise_modules(legacy_v04: bool) -> Dict[str, Any]:
    """One iteration's noise modules, keyed by run_full_assessment keyword.

    ``legacy_v04`` selects the v0.4.0 stubs (legacy_v04 compatibility,
    remove in v0.6.0); otherwise the stubs emit the producers' schema.
    """
    if legacy_v04:
        return {
            "global_workspace": _LegacyV04NoiseGlobalWorkspace(),
            "attention_schema": _LegacyV04NoiseAttentionSchema(),
            "metacognition": _LegacyV04NoiseMetacognition(),
            "active_inference": _LegacyV04NoiseActiveInference(),
            "beautiful_loop": _LegacyV04NoiseBeautifulLoop(),
            "rpt_measurement": _LegacyV04NoiseRPTMeasurement(),
            "pad_affect_model": _LegacyV04NoisePADAffectModel(),
            "causal_emergence_analyzer": _NoiseCausalEmergenceAnalyzer(),
        }
    return {
        "global_workspace": _NoiseGlobalWorkspace(),
        "attention_schema": _NoiseAttentionSchema(),
        "metacognition": _NoiseMetacognition(),
        "active_inference": _NoiseActiveInference(),
        "beautiful_loop": _NoiseBeautifulLoop(),
        "rpt_measurement": _NoiseRPTMeasurement(),
        "pad_affect_model": _NoisePADAffectModel(),
        "causal_emergence_analyzer": _NoiseCausalEmergenceAnalyzer(),
    }


# ============================================================================
# MAIN TEST
# ============================================================================


async def test_assessment():
    """Test the consciousness assessment framework."""
    print("Testing Consciousness Assessment")
    print("=" * 70)

    # Initialize assessment
    assessment = ConsciousnessAssessment()

    # Create mock neural states
    neural_states = {
        "combined_state": np.random.rand(8),
        "connectivity": np.random.rand(8, 8) * 0.5 + np.eye(8) * 0.5,
    }

    # Run assessment without consciousness modules (baseline test)
    print("\n1. Testing assessment without consciousness modules...")
    report = await assessment.run_full_assessment(neural_states=neural_states)

    print(f"\n  Overall Score: {_fmt(report.overall_score)}")
    print(f"  Status: {report.status} (coverage {report.coverage:.0%})")
    print(f"  Architecture Functional: {report.architecture_functional}")
    print(f"  Passing Indicators: {report.passing_count}/{report.total_indicators}")

    # Test Phi calculation
    print("\n2. Testing the Phi calculator...")
    phi_calc = PhiCalculator()
    state = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Binary state
    connectivity = np.random.rand(8, 8)
    phi_result = await phi_calc.calculate_phi(state, connectivity)

    print(f"  Phi: {phi_result.phi:.4f} ({phi_result.method})")
    print(f"  Normalized: {phi_result.phi_normalized:.3f}")
    print(f"  Approximate: {phi_result.is_approximate}")
    print(f"  Computation time: {phi_result.computation_time_ms:.2f}ms")

    print("\n" + "=" * 70)
    print("Consciousness Assessment Framework operational")
    print(
        f"  {len(assessment.indicator_configs)} positive and "
        f"{len(assessment.negative_indicator_configs)} negative indicators configured"
    )
    print(
        "  PyPhi: "
        + ("available" if phi_calc.pyphi_available else "not installed (proxy only)")
    )

    return True


if __name__ == "__main__":
    asyncio.run(test_assessment())
