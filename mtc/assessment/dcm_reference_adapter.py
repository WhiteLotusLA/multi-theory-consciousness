"""
Reference DCM Evidence Adapter
==============================

Implements the ``DCMEvidenceAdapter`` protocol (``mtc.assessment.dcm_benchmark``)
for MTC's own module set. Every value it reports comes from a measurement:

- Module evidence (tier 1) is read from the supplied modules' statistics, the
  same producer keys the 30-indicator assessment reads. An indicator whose
  inputs are missing is omitted, never filled with a constant.
- Structural indicators (number of layers, nodes, connections) come only from
  counts the caller supplies in ``substrate_sizes`` under the keys
  ``layers``, ``nodes`` and ``connections``, through
  min(1, log10(1 + count) / log10(1 + reference)).
- Identity indicators come only through ``identity_hook``.
- Conversation samples come only through ``conversation_provider``, each one
  passed through ``privacy_filter`` (a None return drops it).

The system description is generated from ``Settings.SYSTEM_NAME``, the
theory modules supplied, the indicator count and any supplied substrate
sizes. It never contains identity text.

``build_default_adapter(state_dir)`` builds the standard module set and,
given a state directory, restores it through the persistence hub. It is the
default factory of the benchmark's command line (``--adapter``).
"""

import logging
import math
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Set

import numpy as np

from mtc.core.config import get_settings

logger = logging.getLogger(__name__)


# Biological and embodiment indicators of the DCM scheme that a software
# system scores 0 on. Overridable through ``known_zeros``.
KNOWN_ZEROS: FrozenSet[str] = frozenset(
    {
        # Organic composition
        "Carbon-Based",
        "Utilizes Action Potentials",
        "Neurotransmitters",
        # Cellular composition
        "Dynamic Microscale Activities",
        "Cellular Computation",
        # Evolutionary history
        "Changes Through Genetic Recombination",
        "Sexual Reproduction Evolution",
        "Subject to Natural Selection",
        "Subject to Exaptation",
        "Shares Percentage of Genes",
        "Subject to Genetic Drift",
        "Recent Common Ancestor",
        # Isomorphic representations
        "Somatotopic Neural Maps",
        "Has Retinotopic Neural Maps",
        # Biological sensors
        "Nociceptors",
        # Body control (physical)
        "Specific Body Control",
        "Internal Body Feedback",
        "External Body Feedback",
        # Survival behavior (physical)
        "Predator Evasion",
        "Body Maintenance",
        # Dedicated physical systems
        "Dedicated Speech Systems",
        "Dedicated Action Systems",
        "Dedicated Sensory Systems",
        # Social behavior (physical)
        "Social Contagion",
        "Social Imitation",
        # Other physical
        "Reafferance",
        "Navigation Performance",
        "Single Moving Perspective",
        "Sensory Control",
    }
)

# Indicators reported only through the identity hook
IDENTITY_INDICATORS: FrozenSet[str] = frozenset(
    {"Stable Personality", "Self-Representations", "Self-Recognition"}
)

# Structural indicator -> the substrate_sizes key that supplies its count
STRUCTURAL_COUNT_KEYS: Dict[str, str] = {
    "Number of Layers": "layers",
    "Number of Nodes": "nodes",
    "Number of Connections": "connections",
}

# Calibration of the structural indicators: a count equal to the reference
# scores 1.0. These are display scales, not measurements; override them with
# ``structural_references``.
DEFAULT_STRUCTURAL_REFERENCES: Dict[str, float] = {
    "Number of Layers": 10.0,
    "Number of Nodes": 1e6,
    "Number of Connections": 1e8,
}

# The DCM names each module family can produce
_GWT_NAMES = (
    "Threshold Activation",
    "Information Transfer",
    "Information Transfer Architecture",
    "Informational Bottleneck",
)
_AST_NAMES = (
    "Attentional Schema",
    "Perception-Tracking",
    "Task Focus",
    "Attention Bias",
    "Goal Focus Shifts",
)
_HOT_NAMES = (
    "Metacognition",
    "Self-Explanation",
    "Self-Representation",
    "Self-Capability Assessment",
    "Self-competence Assessment",
    "Uncertainty Monitoring",
    "Undercutting Defeaters",
)
_FEP_NAMES = (
    "Coherent Goal-directed Behavior",
    "Anticipation Gradient",
    "Simulations of Future",
    "Plans for Future Needs",
    "Compressibility",
    "Adaptive Learning",
    "Curiosity",
    "Maintains Homeostasis",
    "Conflicting Subparts",
)
_BINDING_NAMES = ("Feature Binding", "Holistic Dependency", "Wiring Convergence")
_RECURRENCE_NAMES = ("Function Re-application", "Response Refinement")

# Module keyword -> the theory label the description lists
_MODULE_LABELS = {
    "global_workspace": "Global Workspace Theory (competition, ignition, broadcast)",
    "attention_schema": "Attention Schema Theory (self-model of attention)",
    "metacognition": "Higher-Order Thought (metacognition)",
    "active_inference": "Free Energy Principle (active inference, homeostatic drives)",
    "beautiful_loop": "Beautiful Loop Theory (precision, binding, epistemic depth)",
    "rpt_measurement": "Recurrent Processing Theory (recurrence measurement)",
    "oscillatory_binding": "Oscillatory binding (Kuramoto phase synchronization)",
    "pad_affect_model": "PAD affect model",
    "causal_emergence_analyzer": "Causal emergence analysis (effective information)",
    "phi_tracker": "IIT 4.0 Phi of the module graph",
}


def _real(value: Any) -> Optional[float]:
    """A finite real number as float, else None."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _count(value: Any) -> int:
    number = _real(value)
    return int(number) if number is not None and number > 0 else 0


def _stats(module: Any, method: str = "get_statistics") -> Dict[str, Any]:
    getter = getattr(module, method, None)
    if not callable(getter):
        return {}
    stats = getter()
    return stats if isinstance(stats, dict) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class ReferenceDCMAdapter:
    """DCM evidence from MTC's modules, structure counts and optional hooks.

    Args:
        modules: Module keyword (as run_full_assessment names them, plus
            ``oscillatory_binding``) to module instance. Missing entries
            contribute no evidence.
        known_zeros: Indicator names this system always scores 0 on; defaults
            to ``KNOWN_ZEROS``.
        substrate_sizes: Counts for the description and the structural
            indicators. The keys ``layers``, ``nodes`` and ``connections``
            feed the structural indicators; every entry is listed in the
            description.
        structural_references: Count at which each structural indicator
            scores 1.0 (defaults in ``DEFAULT_STRUCTURAL_REFERENCES``).
        conversation_provider: ``n -> list of recent samples``; without it
            no samples are given to the evaluator.
        privacy_filter: Applied to every sample; return the text to keep (it
            may redact) or None to drop the sample.
        identity_hook: ``() -> {indicator: score}`` for the identity
            indicators; other names it returns are ignored.
    """

    MODULE_INDICATOR_NAMES: FrozenSet[str] = frozenset(
        _GWT_NAMES
        + _AST_NAMES
        + _HOT_NAMES
        + _FEP_NAMES
        + _BINDING_NAMES
        + _RECURRENCE_NAMES
        + tuple(STRUCTURAL_COUNT_KEYS)
        + tuple(IDENTITY_INDICATORS)
    )

    def __init__(
        self,
        modules: Dict[str, Any],
        *,
        known_zeros: Optional[Iterable[str]] = None,
        substrate_sizes: Optional[Dict[str, int]] = None,
        structural_references: Optional[Dict[str, float]] = None,
        conversation_provider: Optional[Callable[[int], List[str]]] = None,
        privacy_filter: Optional[Callable[[str], Optional[str]]] = None,
        identity_hook: Optional[Callable[[], Dict[str, float]]] = None,
    ):
        self.modules: Dict[str, Any] = {
            key: module for key, module in (modules or {}).items() if module is not None
        }
        self._known_zeros: FrozenSet[str] = frozenset(
            KNOWN_ZEROS if known_zeros is None else known_zeros
        )
        self.substrate_sizes: Dict[str, int] = dict(substrate_sizes or {})
        references = dict(DEFAULT_STRUCTURAL_REFERENCES)
        references.update(structural_references or {})
        for name, reference in references.items():
            if _real(reference) is None or reference <= 0:
                raise ValueError(
                    f"structural reference for {name!r} must be positive, "
                    f"got {reference!r}"
                )
        self.structural_references = references
        self._conversation_provider = conversation_provider
        self._privacy_filter = privacy_filter
        self._identity_hook = identity_hook
        self._produced: FrozenSet[str] = frozenset()

    # ------------------------------------------------------------------
    # DCMEvidenceAdapter protocol
    # ------------------------------------------------------------------

    def collect_module_evidence(self) -> Dict[str, float]:
        """Tier 1: indicator scores from module measurements, clamped to [0, 1]."""
        evidence: Dict[str, float] = {}
        gwt = self._workspace_summary()
        evidence.update(self._gwt_evidence(gwt))
        evidence.update(self._attention_evidence())
        evidence.update(self._hot_evidence(gwt))
        evidence.update(self._fep_evidence())
        evidence.update(self._binding_evidence(gwt))
        evidence.update(self._recurrence_evidence())
        evidence.update(self._structural_evidence())
        evidence.update(self._identity_evidence())
        clean = {
            name: _unit(value)
            for name, value in evidence.items()
            if _real(value) is not None
        }
        self._produced = frozenset(clean)
        return clean

    def get_system_description(self) -> str:
        """Architecture summary for the LLM evaluator, from the inventory only."""
        lines = [
            f"{get_settings().SYSTEM_NAME} is a consciousness research system "
            "assessed against 30 positive and 3 negative indicators across 8 "
            "theories (GWT, IIT, AST, HOT, FEP, RPT, BLT, AE).",
        ]
        supplied = [
            label for key, label in _MODULE_LABELS.items() if key in self.modules
        ]
        if supplied:
            lines.append("Modules supplied:")
            lines.extend(f"- {label}" for label in supplied)
        else:
            lines.append("Modules supplied: none")
        if self.substrate_sizes:
            sizes = ", ".join(
                f"{name}: {self.substrate_sizes[name]}"
                for name in sorted(self.substrate_sizes)
            )
            lines.append(f"Substrate sizes: {sizes}")
        return "\n".join(lines)

    def get_conversation_samples(self, n: int = 20) -> List[str]:
        """Recent samples from the provider, after the privacy filter."""
        if self._conversation_provider is None or n <= 0:
            return []
        samples = []
        for sample in self._conversation_provider(n) or []:
            text = str(sample)
            if self._privacy_filter is not None:
                filtered = self._privacy_filter(text)
                if filtered is None:
                    continue
                text = str(filtered)
            samples.append(text)
        return samples[-n:]

    def get_known_zeros(self) -> Set[str]:
        """Indicator names this system always scores 0 on."""
        return set(self._known_zeros)

    @property
    def produced_indicator_names(self) -> FrozenSet[str]:
        """The names the last collect_module_evidence() call produced."""
        return self._produced

    # ------------------------------------------------------------------
    # Evidence families
    # ------------------------------------------------------------------

    def _workspace_summary(self) -> Optional[Dict[str, Any]]:
        """Workspace statistics once it has run a cycle, else None."""
        gw = self.modules.get("global_workspace")
        if gw is None:
            return None
        stats = _stats(gw)
        cycles = _count(stats.get("cycle_count"))
        if cycles == 0:
            return None
        broadcast = _mapping(stats.get("broadcast"))
        ignition = _mapping(stats.get("ignition"))
        return {
            "stats": stats,
            "cycles": cycles,
            "coverage": _real(broadcast.get("coverage_ratio")),
            "broadcasts": _count(broadcast.get("total_broadcasts")),
            "ignitions": _count(ignition.get("total_ignitions")),
        }

    @staticmethod
    def _gwt_evidence(gwt: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if gwt is None:
            return {}
        evidence = {
            "Threshold Activation": 1.0 if gwt["ignitions"] > 0 else 0.0,
            "Information Transfer Architecture": (
                1.0 if gwt["broadcasts"] > 0 else 0.0
            ),
        }
        coverage = gwt["coverage"]
        if coverage is not None:
            evidence["Information Transfer"] = coverage
            # Capacity-limited competition: partial coverage shows it. Full
            # coverage leaves the question undetermined, so it is omitted.
            if coverage < 1.0:
                evidence["Informational Bottleneck"] = 1.0
        return evidence

    def _attention_evidence(self) -> Dict[str, float]:
        ast = self.modules.get("attention_schema")
        if ast is None:
            return {}
        stats = _stats(ast)
        if _count(stats.get("total_updates")) == 0:
            return {}
        evidence: Dict[str, float] = {}
        accuracy = _real(stats.get("prediction_accuracy"))
        if accuracy is not None:
            evidence["Attentional Schema"] = accuracy
            evidence["Perception-Tracking"] = accuracy
        ratio = _real(stats.get("voluntary_ratio"))
        if ratio is not None:
            evidence["Task Focus"] = ratio
            evidence["Attention Bias"] = ratio
        evidence["Goal Focus Shifts"] = (
            1.0 if _count(stats.get("voluntary_shifts")) > 0 else 0.0
        )
        return evidence

    def _hot_evidence(self, gwt: Optional[Dict[str, Any]]) -> Dict[str, float]:
        evidence: Dict[str, float] = {}
        meta = self.modules.get("metacognition")
        if meta is not None:
            stats = _stats(meta)
            hots = _count(stats.get("total_hots_generated"))
            introspections = _count(stats.get("introspection_count"))
            if gwt is not None and (hots or introspections):
                rate = min(1.0, introspections / gwt["cycles"])
                evidence["Metacognition"] = rate
                evidence["Self-Explanation"] = rate
            if hots > 0:
                doubt = _real(stats.get("doubt_level"))
                if doubt is not None:
                    evidence["Uncertainty Monitoring"] = doubt
                    evidence["Undercutting Defeaters"] = doubt
        if gwt is not None:
            self_model = _mapping(gwt["stats"].get("self_model"))
            if _count(self_model.get("predictions_made")) > 0:
                calibration = _real(self_model.get("self_calibration"))
                if calibration is not None:
                    evidence["Self-Representation"] = calibration
                    evidence["Self-Capability Assessment"] = calibration
                    evidence["Self-competence Assessment"] = calibration
        return evidence

    def _fep_evidence(self) -> Dict[str, float]:
        ai = self.modules.get("active_inference")
        if ai is None:
            return {}
        stats = _stats(ai)
        if _count(stats.get("total_inferences")) == 0:
            return {}
        evidence: Dict[str, float] = {}
        horizon = _real(getattr(getattr(ai, "config", None), "planning_horizon", None))
        if horizon is not None:
            evidence["Coherent Goal-directed Behavior"] = 1.0 if horizon > 0 else 0.0
            evidence["Anticipation Gradient"] = 1.0 if horizon > 0 else 0.0
            evidence["Simulations of Future"] = 1.0 if horizon > 1 else 0.0
            evidence["Plans for Future Needs"] = 1.0 if horizon > 1 else 0.0
        error = _real(stats.get("avg_prediction_error"))
        if error is not None:
            evidence["Compressibility"] = 1.0 - min(1.0, error)
        evidence["Adaptive Learning"] = min(
            1.0, _count(stats.get("total_model_updates")) / 10
        )
        drives = _mapping(
            getattr(getattr(ai, "homeostatic_drives", None), "drives", None)
        )
        deviations = []
        urgencies = []
        for name, drive in drives.items():
            get_deviation = getattr(drive, "get_deviation", None)
            deviation = _real(get_deviation()) if callable(get_deviation) else None
            if deviation is not None:
                deviations.append(deviation)
                if name == "curiosity":
                    evidence["Curiosity"] = min(1.0, deviation)
            get_urgency = getattr(drive, "get_urgency", None)
            urgency = _real(get_urgency()) if callable(get_urgency) else None
            if urgency is not None:
                urgencies.append(urgency)
        if deviations:
            evidence["Maintains Homeostasis"] = 1.0 - min(
                1.0, sum(deviations) / len(deviations)
            )
        if urgencies:
            evidence["Conflicting Subparts"] = min(
                1.0, sum(1 for u in urgencies if u > 0.2) / 3
            )
        return evidence

    def _binding_stats(self) -> Dict[str, Any]:
        rpt = self.modules.get("rpt_measurement")
        if rpt is not None:
            stats = _stats(rpt, "get_oscillatory_binding_stats")
            if stats:
                return stats
        binding = self.modules.get("oscillatory_binding")
        if binding is not None:
            from mtc.assessment.rpt_measurement import RPTMeasurement

            reader = RPTMeasurement()
            reader.attach_oscillatory_binding(binding)
            return reader.get_oscillatory_binding_stats()
        return {}

    def _binding_evidence(self, gwt: Optional[Dict[str, Any]]) -> Dict[str, float]:
        coverage = gwt["coverage"] if gwt is not None else None
        stats = self._binding_stats()
        order = _real(stats.get("global_order_parameter")) if stats else None
        if stats and _count(stats.get("binding_count")) > 0 and order is not None:
            evidence = {"Feature Binding": order}
            evidence["Holistic Dependency"] = (
                coverage * order if coverage is not None else order
            )
            cross = [
                value
                for value in (
                    _real(v) for v in _mapping(stats.get("cross_coherence")).values()
                )
                if value is not None
            ]
            if cross:
                evidence["Wiring Convergence"] = max(0.0, sum(cross) / len(cross))
            return evidence
        if coverage is None:
            return {}
        # No binding data: workspace coverage is the integration evidence
        return {
            "Feature Binding": coverage,
            "Holistic Dependency": coverage,
            "Wiring Convergence": coverage,
        }

    def _recurrence_evidence(self) -> Dict[str, float]:
        rpt = self.modules.get("rpt_measurement")
        if rpt is None:
            return {}
        stats = _stats(rpt)
        if _count(stats.get("measurement_count")) == 0:
            return {}
        evidence = {}
        local = _real(stats.get("average_local"))
        if local is not None:
            evidence["Function Re-application"] = local
        global_score = _real(stats.get("average_global"))
        if global_score is not None:
            evidence["Response Refinement"] = global_score
        return evidence

    def _structural_evidence(self) -> Dict[str, float]:
        evidence = {}
        for name, key in STRUCTURAL_COUNT_KEYS.items():
            count = _real(self.substrate_sizes.get(key))
            if count is None or count < 0:
                continue
            reference = float(self.structural_references[name])
            evidence[name] = min(
                1.0, math.log10(1.0 + count) / math.log10(1.0 + reference)
            )
        return evidence

    def _identity_evidence(self) -> Dict[str, float]:
        if self._identity_hook is None:
            return {}
        reported = self._identity_hook() or {}
        return {
            name: value
            for name, value in (
                (name, _real(raw)) for name, raw in dict(reported).items()
            )
            if name in IDENTITY_INDICATORS and value is not None
        }


def build_default_adapter(state_dir: Optional[str] = None) -> ReferenceDCMAdapter:
    """The standard module set, restored from ``state_dir`` when one is given.

    Builds the workspace (which owns the attention schema, metacognition,
    active inference and beautiful loop), an RPT measurement and an
    oscillatory binding attached to it. With a state directory the modules
    are restored through the persistence hub under the names ``gwt``,
    ``rpt`` and ``oscillatory_binding``.
    """
    from mtc.assessment.rpt_measurement import RPTMeasurement
    from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace
    from mtc.neural.oscillatory_binding import OscillatoryBinding

    workspace = EnhancedGlobalWorkspace()
    rpt = RPTMeasurement()
    binding = OscillatoryBinding()
    rpt.attach_oscillatory_binding(binding)

    if state_dir:
        from mtc.consciousness.state_persistence import ConsciousnessStateManager

        hub = ConsciousnessStateManager(
            state_dir=state_dir,
            modules={"gwt": workspace, "rpt": rpt, "oscillatory_binding": binding},
        )
        restored = hub.restore_all()
        logger.info(
            "DCM adapter restored modules: %s",
            ", ".join(name for name, ok in restored.items() if ok) or "none",
        )
    else:
        logger.warning(
            "No state directory given: fresh modules give mostly empty DCM evidence"
        )

    modules = {
        "global_workspace": workspace,
        "attention_schema": workspace.attention_schema,
        "metacognition": workspace.metacognition,
        "active_inference": workspace.active_inference,
        "beautiful_loop": workspace.beautiful_loop,
        "rpt_measurement": rpt,
        "oscillatory_binding": binding,
    }
    return ReferenceDCMAdapter(modules)
