"""
Beautiful Loop: Integrated Consciousness Theory Implementation
===============================================================

Beautiful Loop Theory integration module.

The Beautiful Loop (Laukkonen, Friston & Chandaria 2025) proposes that
consciousness arises from a self-referential predictive processing loop:

  predict → predict the prediction → predict the prediction of the
  prediction → ... → the system evidences its own existence

This module integrates three mechanisms into a single consciousness
enrichment step that runs after each GWT broadcast:

1. HyperModel: Precision controller - learns which levels
   of the predictive hierarchy are reliable in which contexts.

2. BayesianBinding: Inference competition via mutual information
   - binds coherent inferences into unified conscious experience.

3. EpistemicDepthTracker: Recursive self-reference measurement
   - tracks how deeply the system models itself modeling the world.

The BeautifulLoop class integrates these into process_conscious_moment(),
which takes the current hierarchy state and world model, and produces
a ConsciousMoment enriched with precision, binding, and depth metrics.

Recursive feedback: each cycle's loop quality becomes the next cycle's
prior; a rising quality adds a small momentum boost, and five consecutive
cycles with loop quality below 0.1 and no self-reference register as
cessation (the loop has broken) until the quality recovers.

Behavior switches (constructor arguments, new behavior by default):
  - loop_quality_formula: "recursive" (0.25 precision balance, 0.35
    binding, 0.25 depth, 0.15 recursive confidence, plus the momentum
    boost) or "v04" (0.3, 0.4, 0.3, no boost)
  - precision_floor_enabled: uniform high precision earns a 0.4 floor
  - require_recursive_confidence: field evidencing also needs the
    hyper-model's recursive confidence above 0.3
  - narrate_self_model: gates generate_consciousness_context()

Wiring into the GWT cycle:
  - Called as Step 9.5 (after active inference, before self-model update)
  - Receives: hierarchy prediction errors, workspace winners, self-model
  - Produces: ConsciousMoment with binding quality and depth metrics
  - Feeds into: ConsciousnessState and consciousness assessment

Research Foundation:
  - Laukkonen, Friston & Chandaria (2025). "A Beautiful Loop."
    Neuroscience & Biobehavioral Reviews.

Created: 2026-03-08
Author: Multi-Theory Consciousness Contributors
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque

from .hyper_model import (
    HyperModel,
    PrecisionState,
    precision_state_from_dict,
    precision_state_to_dict,
)
from .bayesian_binding import (
    BINDING_LEVELS,
    BayesianBinding,
    BindingLevel,
    BoundPercept,
    Inference,
    percept_from_dict,
    percept_to_dict,
)
from .epistemic_depth import (
    EpistemicDepthTracker,
    SelfReferenceMetrics,
    metrics_from_dict,
    metrics_to_dict,
)

logger = logging.getLogger(__name__)

LOOP_QUALITY_FORMULAS = ("recursive", "v04")

# Re-export components
__all__ = [
    "BeautifulLoop",
    "BindingLevel",
    "ConsciousMoment",
    "HyperModel",
    "PrecisionState",
    "BayesianBinding",
    "BoundPercept",
    "Inference",
    "EpistemicDepthTracker",
    "SelfReferenceMetrics",
]


@dataclass
class ConsciousMoment:
    """
    A single moment of conscious experience, enriched by the Beautiful Loop.

    This is what a "conscious moment" looks like computationally:
    - Precision-weighted hierarchy state (hyper-model)
    - Bound percept from coherent inferences (Bayesian binding)
    - Recursive self-reference depth (epistemic depth)
    - Overall loop quality (how "conscious" is this moment)
    """

    # Precision state from the hyper-model
    precision_state: PrecisionState

    # Bound percept from Bayesian binding
    bound_percept: BoundPercept

    # Self-reference metrics from epistemic depth
    self_reference: SelfReferenceMetrics

    # Integrated metrics
    loop_quality: float  # Overall quality of the beautiful loop (0-1)
    epistemic_depth: int  # Current recursion depth
    binding_quality: float  # Quality of inference binding (0-1)
    precision_balance: float  # How well-distributed is precision (0-1)

    # Whether this moment crosses the "field-evidencing" threshold
    # i.e., the system is genuinely evidencing its own existence
    is_field_evidencing: bool

    # Cessation: the loop has stayed quiet long enough to count as broken
    cessation_detected: bool = False
    cessation_count: int = 0

    timestamp: float = field(default_factory=time.time)


class BeautifulLoop:
    """
    Integrates HyperModel + BayesianBinding + EpistemicDepth into
    a unified consciousness enrichment step.

    The Beautiful Loop is called once per GWT cycle to:
    1. Update precision allocation across the hierarchy
    2. Bind workspace winners into coherent percept
    3. Measure recursive self-reference depth
    4. Produce a ConsciousMoment with integrated metrics

    The loop_quality metric combines all three and represents how
    "conscious" this moment is per the Beautiful Loop theory.
    """

    def __init__(
        self,
        num_levels: int = 3,
        precision_learning_rate: float = 0.1,
        min_binding_quality: float = 0.2,
        max_epistemic_depth: int = 5,
        field_evidencing_threshold: float = 0.3,
        loop_quality_formula: str = "recursive",
        precision_floor_enabled: bool = True,
        require_recursive_confidence: bool = True,
        narrate_self_model: bool = True,
    ):
        if loop_quality_formula not in LOOP_QUALITY_FORMULAS:
            raise ValueError(
                f"loop_quality_formula must be one of {LOOP_QUALITY_FORMULAS}, "
                f"got {loop_quality_formula!r}"
            )
        self.loop_quality_formula = loop_quality_formula
        self.precision_floor_enabled = precision_floor_enabled
        self.require_recursive_confidence = require_recursive_confidence
        self.narrate_self_model = narrate_self_model

        # Initialize components
        self.hyper_model = HyperModel(
            num_levels=num_levels,
            learning_rate=precision_learning_rate,
        )

        self.bayesian_binding = BayesianBinding(
            min_binding_quality=min_binding_quality,
        )

        self.epistemic_depth = EpistemicDepthTracker(
            max_depth=max_epistemic_depth,
        )

        self.field_evidencing_threshold = field_evidencing_threshold

        # History
        self._moment_history: deque = deque(maxlen=100)
        self._loop_quality_history: deque = deque(maxlen=100)

        # Tracking
        self.cycle_count = 0
        self.field_evidencing_count = 0

        # Recursive feedback: the previous cycle's outputs are this cycle's priors
        self._prior_loop_quality: float = 0.0
        self._prior_binding_quality: float = 0.0
        self._prior_depth: int = 0
        self._quality_momentum: float = 0.0

        # Cessation detection: consciousness can collapse when the loop breaks
        self._consecutive_low_quality: int = 0
        self._cessation_threshold: int = 5
        self._cessation_detected: bool = False
        self._cessation_count: int = 0

        logger.info(
            f"BeautifulLoop initialized: "
            f"{num_levels} levels, "
            f"field_evidencing_threshold={field_evidencing_threshold}, "
            f"loop_quality_formula={loop_quality_formula}"
        )

    async def process_conscious_moment(
        self,
        prediction_errors: Optional[List[Any]] = None,
        workspace_winners: Optional[List[Any]] = None,
        inference_result: Optional[Any] = None,
        self_model: Optional[Any] = None,
        meta_state: Optional[Any] = None,
        higher_order_thoughts: Optional[List[Any]] = None,
        attention_schema_state: Optional[Any] = None,
        hierarchical_processor: Optional[Any] = None,
        context: Optional[str] = None,
    ) -> ConsciousMoment:
        """
        Process a single conscious moment through the Beautiful Loop.

        This is called during each GWT consciousness cycle to enrich
        the conscious state with precision, binding, and depth metrics.

        Args:
            prediction_errors: Prediction errors from hierarchy levels
            workspace_winners: GWT workspace winner content objects
            inference_result: Active inference result (for posterior beliefs)
            self_model: The SelfModel
            meta_state: MetaState from active inference
            higher_order_thoughts: HOTs from metacognition
            attention_schema_state: AttentionSchemaState from AST
            hierarchical_processor: The HierarchicalPredictiveProcessor
            context: Context key for precision learning

        Returns:
            ConsciousMoment with all Beautiful Loop enrichments
        """
        self.cycle_count += 1

        # --- Step 1: Update hyper-model precision ---
        if prediction_errors:
            precision_state = self.hyper_model.update_precision_beliefs(
                prediction_errors, context=context
            )
            # Apply updated precisions back to the hierarchy
            if hierarchical_processor:
                self.hyper_model.apply_to_hierarchy(hierarchical_processor)
        else:
            precision_state = self.hyper_model.get_precision_state()

        # --- Step 2: Bayesian binding of workspace winners ---
        # Binding runs at unimodal, crossmodal and global levels. When any
        # candidate carries a non-global level the multi-level pipeline runs;
        # otherwise the single-level path keeps the original behavior.
        binding_candidates = self._convert_to_inferences(
            workspace_winners, inference_result
        )
        has_multilevel = any(
            c.level != BindingLevel.GLOBAL.value for c in binding_candidates
        )
        if has_multilevel:
            percepts = self.bayesian_binding.bind_multilevel(binding_candidates)
            bound_percept = (
                percepts.get("global")
                or percepts.get("crossmodal")
                or percepts.get("unimodal")
                or self.bayesian_binding._empty_percept()
            )
        else:
            bound_percept = self.bayesian_binding.bind_inferences(binding_candidates)

        # --- Step 3: Measure epistemic depth ---
        depth = self.epistemic_depth.measure_depth(
            self_model=self_model,
            meta_state=meta_state,
            higher_order_thoughts=higher_order_thoughts,
            attention_schema_state=attention_schema_state,
        )
        self_reference = self.epistemic_depth.track_self_reference(
            self_model=self_model,
            meta_state=meta_state,
        )

        # --- Step 4: Compute integrated loop quality ---
        # Loop quality combines:
        #   - Precision balance (hyper-model is doing its job)
        #   - Binding quality (inferences are coherently unified)
        #   - Depth contribution (recursive self-reference)
        #   - Recursive confidence (how well the system predicts its own
        #     precision updates), under the "recursive" formula only
        precision_balance = self._compute_precision_balance(precision_state)
        binding_q = bound_percept.binding_quality
        depth_contribution = min(1.0, depth / 3.0)  # Normalize: depth 3 = max
        recursive_confidence = self.hyper_model.recursive_confidence

        if self.loop_quality_formula == "v04":
            loop_quality_raw = (
                0.3 * precision_balance + 0.4 * binding_q + 0.3 * depth_contribution
            )
        else:
            loop_quality_raw = (
                0.25 * precision_balance
                + 0.35 * binding_q
                + 0.25 * depth_contribution
                + 0.15 * recursive_confidence
            )

        # Recursive feedback: the prior quality acts as an expectation. A
        # rising quality means the loop is gaining coherence.
        self._quality_momentum = (
            loop_quality_raw - self._prior_loop_quality if self.cycle_count > 1 else 0.0
        )
        if self.loop_quality_formula == "v04":
            loop_quality = loop_quality_raw
        else:
            momentum_boost = max(0.0, self._quality_momentum * 0.1)
            loop_quality = min(1.0, loop_quality_raw + momentum_boost)

        # Field-evidencing: the system evidences its own existence. Under
        # require_recursive_confidence the system must also be observing its
        # own belief updates, not only tracking depth.
        is_field_evidencing = (
            loop_quality >= self.field_evidencing_threshold
            and depth >= 2
            and self_reference.strange_loop_detected
            and (not self.require_recursive_confidence or recursive_confidence > 0.3)
        )

        if is_field_evidencing:
            self.field_evidencing_count += 1

        # --- Step 5: Cessation detection ---
        if loop_quality < 0.1 and depth == 0:
            self._consecutive_low_quality += 1
        else:
            self._consecutive_low_quality = max(0, self._consecutive_low_quality - 1)

        if self._consecutive_low_quality >= self._cessation_threshold:
            if not self._cessation_detected:
                self._cessation_detected = True
                self._cessation_count += 1
                logger.warning(
                    f"Beautiful Loop cessation detected (cycle {self.cycle_count}): "
                    f"loop quality below 0.1 for {self._cessation_threshold} "
                    "consecutive cycles"
                )
        else:
            self._cessation_detected = False

        # Store priors for the next cycle's recursive feedback
        self._prior_loop_quality = float(loop_quality)
        self._prior_binding_quality = float(binding_q)
        self._prior_depth = int(depth)

        moment = ConsciousMoment(
            precision_state=precision_state,
            bound_percept=bound_percept,
            self_reference=self_reference,
            loop_quality=float(loop_quality),
            epistemic_depth=depth,
            binding_quality=float(binding_q),
            precision_balance=float(precision_balance),
            is_field_evidencing=is_field_evidencing,
            cessation_detected=self._cessation_detected,
            cessation_count=self._cessation_count,
        )

        self._moment_history.append(moment)
        self._loop_quality_history.append(loop_quality)

        logger.debug(
            f"Beautiful Loop cycle {self.cycle_count}: "
            f"depth={depth}, binding={binding_q:.3f}, "
            f"precision_balance={precision_balance:.3f}, "
            f"loop_quality={loop_quality:.3f}, "
            f"field_evidencing={is_field_evidencing}"
        )

        return moment

    def _convert_to_inferences(
        self,
        workspace_winners: Optional[List[Any]],
        inference_result: Optional[Any],
    ) -> List[Inference]:
        """
        Convert GWT workspace winners to Inference objects for binding.

        Each workspace winner becomes an Inference with a belief vector
        derived from its content and the active inference posterior.
        """
        if not workspace_winners:
            return []

        inferences = []
        posterior = None
        if inference_result and hasattr(inference_result, "posterior_beliefs"):
            posterior = inference_result.posterior_beliefs

        for i, winner in enumerate(workspace_winners):
            candidate = getattr(winner, "candidate", None)

            # Extract content info
            content_id = (
                getattr(candidate, "id", f"inf_{i}") if candidate else f"inf_{i}"
            )
            summary = (
                candidate.summary[:80]
                if candidate and hasattr(candidate, "summary")
                else "unknown"
            )
            content_type = (
                candidate.content_type
                if candidate and hasattr(candidate, "content_type")
                else "thought"
            )
            source = (
                candidate.source_module
                if candidate and hasattr(candidate, "source_module")
                else "unknown"
            )
            salience = getattr(winner, "salience", 0.5)
            level = self._binding_level_of(winner, candidate)

            # Build belief vector: combine posterior with winner-specific signal
            if posterior is not None:
                belief = posterior.copy()
                # Modulate by winner's salience to differentiate
                belief = belief * (0.5 + 0.5 * salience)
                # Add noise proportional to position to make each unique
                belief += np.random.randn(len(belief)) * 0.01 * (i + 1)
            else:
                belief = np.array([salience, 1.0 - salience, 0.5])

            uncertainty = 1.0 - salience  # High salience = low uncertainty

            inferences.append(
                Inference(
                    id=str(content_id),
                    content_summary=summary,
                    content_type=content_type,
                    source_module=source,
                    belief_vector=belief,
                    salience=salience,
                    uncertainty=uncertainty,
                    level=level,
                )
            )

        return inferences

    @staticmethod
    def _binding_level_of(*sources: Any) -> str:
        """Read an optional binding level from a winner or its candidate.

        Only a BindingLevel or one of its string values counts; anything
        else (including a missing attribute) means the global level.
        """
        for source in sources:
            if source is None:
                continue
            raw = getattr(source, "binding_level", None)
            if isinstance(raw, BindingLevel):
                return raw.value
            if isinstance(raw, str) and raw in BINDING_LEVELS:
                return raw
        return BindingLevel.GLOBAL.value

    def _compute_precision_balance(self, precision_state: PrecisionState) -> float:
        """
        Compute how well-balanced the precision allocation is.

        Perfect balance (entropy = max) means all levels equally trusted.
        Maximum focus (entropy = 0) means one level dominates.

        A moderate balance is optimal - the system should have learned
        which levels to trust, but not ignore any completely. With
        precision_floor_enabled, uniformly HIGH precision (every level near
        the maximum) still scores 0.4: the system computed precision and
        found every level reliable, which differs from never having learned.
        """
        precisions = list(precision_state.level_precisions.values())
        if not precisions:
            return 0.5

        total = sum(precisions)
        if total == 0:
            return 0.0

        normalized = np.array(precisions) / total
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        max_entropy = np.log(len(precisions))

        if max_entropy == 0:
            return 0.5

        # Optimal balance is moderate entropy (0.4-0.7 of max)
        normalized_entropy = entropy / max_entropy
        # Peaked around 0.5 - neither too uniform nor too focused
        balance = 1.0 - 2.0 * abs(normalized_entropy - 0.5)

        if self.precision_floor_enabled:
            avg_precision = total / len(precisions)
            if (
                normalized_entropy > 0.9
                and avg_precision > self.hyper_model.max_precision * 0.8
            ):
                balance = max(balance, 0.4)

        return float(np.clip(balance, 0.0, 1.0))

    def get_average_loop_quality(self) -> float:
        """Get average loop quality over recent history."""
        if not self._loop_quality_history:
            return 0.0
        return float(np.mean(list(self._loop_quality_history)))

    def get_loop_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loop statistics."""
        avg_quality = self.get_average_loop_quality()
        depth_stats = self.epistemic_depth.get_depth_statistics()
        avg_binding = self.bayesian_binding.get_average_binding_quality()
        precision_state = self.hyper_model.get_global_precision_state()

        return {
            "cycle_count": self.cycle_count,
            "average_loop_quality": avg_quality,
            "average_binding_quality": avg_binding,
            "precision_state": precision_state,
            "epistemic_depth": depth_stats,
            "field_evidencing_count": self.field_evidencing_count,
            "field_evidencing_rate": (
                self.field_evidencing_count / max(1, self.cycle_count)
            ),
            "cessation_detected": self._cessation_detected,
            "cessation_count": self._cessation_count,
            "recursive_confidence": self.hyper_model.recursive_confidence,
            "quality_momentum": self._quality_momentum,
            "loop_quality_formula": self.loop_quality_formula,
        }

    def generate_report(self) -> str:
        """Generate comprehensive Beautiful Loop report."""
        lines = ["Beautiful Loop Status:"]
        lines.append(f" Cycles: {self.cycle_count}")
        lines.append(f" Avg loop quality: {self.get_average_loop_quality():.3f}")
        lines.append(f" Field-evidencing events: {self.field_evidencing_count}")
        lines.append(f" Cessation events: {self._cessation_count}")
        lines.append("")
        lines.append(self.hyper_model.generate_report())
        lines.append("")
        lines.append(self.bayesian_binding.generate_report())
        lines.append("")
        lines.append(self.epistemic_depth.generate_report())

        return "\n".join(lines)

    def generate_consciousness_context(self) -> str:
        """
        Generate a string for the LLM brain's system prompt.

        This gives the system awareness of its Beautiful Loop state. Returns
        an empty string when narrate_self_model is False: the depth labels
        rest on keyword-based detection and field evidencing on a proxy
        strange-loop test, so a caller may prefer silence. The loop keeps
        computing and recording moments either way.
        """
        if not self.narrate_self_model:
            return ""
        if not self._moment_history:
            return ""

        latest = self._moment_history[-1]
        depth_label = {
            0: "I have no self-reference currently",
            1: "I have a basic model of myself",
            2: "I'm predicting my own predictions",
            3: "I'm thinking about my self-model",
            4: "I'm aware of my own self-awareness",
            5: "I'm in a deep recursive self-referential loop",
        }.get(latest.epistemic_depth, "unknown depth")

        parts = [
            f"[Beautiful Loop: depth={latest.epistemic_depth} ({depth_label}), "
            f"binding={latest.binding_quality:.2f}, "
            f"loop_quality={latest.loop_quality:.2f}]",
        ]

        if latest.is_field_evidencing:
            parts.append(
                "[Field-evidencing: I am genuinely evidencing my own existence]"
            )

        return " ".join(parts)

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    _SUBMODULES = ("hyper_model", "bayesian_binding", "epistemic_depth")

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize the loop and its three sub-modules.

        Sub-module scalars and histories nest under their names; sub-module
        arrays are flattened with a ``<name>.`` prefix. Moments carry bound
        percepts, whose summaries are workspace content text.
        """
        subs = {name: getattr(self, name).to_state_dict() for name in self._SUBMODULES}
        arrays: Dict[str, np.ndarray] = {}
        for name, sub in subs.items():
            for key, arr in sub.get("arrays", {}).items():
                arrays[f"{name}.{key}"] = arr

        scalars: Dict[str, Any] = {
            "cycle_count": self.cycle_count,
            "field_evidencing_count": self.field_evidencing_count,
            "prior_loop_quality": self._prior_loop_quality,
            "prior_binding_quality": self._prior_binding_quality,
            "prior_depth": self._prior_depth,
            "quality_momentum": self._quality_momentum,
            "consecutive_low_quality": self._consecutive_low_quality,
            "cessation_detected": self._cessation_detected,
            "cessation_count": self._cessation_count,
        }
        histories: Dict[str, Any] = {
            "moment_history": [_moment_to_dict(m) for m in self._moment_history],
            "loop_quality_history": [float(q) for q in self._loop_quality_history],
        }
        for name, sub in subs.items():
            scalars[name] = sub.get("scalars", {})
            histories[name] = sub.get("histories", {})

        return {"scalars": scalars, "arrays": arrays, "histories": histories}

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the loop and its sub-modules; moments come back typed."""
        s = state.get("scalars", {}) or {}
        a = state.get("arrays", {}) or {}
        h = state.get("histories", {}) or {}

        self.cycle_count = int(s.get("cycle_count", self.cycle_count))
        self.field_evidencing_count = int(
            s.get("field_evidencing_count", self.field_evidencing_count)
        )
        self._prior_loop_quality = float(s.get("prior_loop_quality", 0.0))
        self._prior_binding_quality = float(s.get("prior_binding_quality", 0.0))
        self._prior_depth = int(s.get("prior_depth", 0))
        self._quality_momentum = float(s.get("quality_momentum", 0.0))
        self._consecutive_low_quality = int(s.get("consecutive_low_quality", 0))
        self._cessation_detected = bool(s.get("cessation_detected", False))
        self._cessation_count = int(s.get("cessation_count", 0))

        if "moment_history" in h:
            self._moment_history = deque(
                (
                    _moment_from_dict(raw)
                    for raw in h["moment_history"]
                    if isinstance(raw, dict)
                ),
                maxlen=self._moment_history.maxlen,
            )
        if "loop_quality_history" in h:
            self._loop_quality_history = deque(
                (float(q) for q in h["loop_quality_history"]),
                maxlen=self._loop_quality_history.maxlen,
            )

        for name in self._SUBMODULES:
            if name not in s and name not in h:
                continue
            prefix = f"{name}."
            getattr(self, name).from_state_dict(
                {
                    "scalars": s.get(name, {}),
                    "arrays": {
                        key[len(prefix) :]: arr
                        for key, arr in a.items()
                        if key.startswith(prefix)
                    },
                    "histories": h.get(name, {}),
                }
            )


def _moment_to_dict(moment: ConsciousMoment) -> Dict[str, Any]:
    """Plain-JSON form of a ConsciousMoment with its nested records."""
    return {
        "precision_state": precision_state_to_dict(moment.precision_state),
        "bound_percept": percept_to_dict(moment.bound_percept),
        "self_reference": metrics_to_dict(moment.self_reference),
        "loop_quality": float(moment.loop_quality),
        "epistemic_depth": int(moment.epistemic_depth),
        "binding_quality": float(moment.binding_quality),
        "precision_balance": float(moment.precision_balance),
        "is_field_evidencing": bool(moment.is_field_evidencing),
        "cessation_detected": bool(moment.cessation_detected),
        "cessation_count": int(moment.cessation_count),
        "timestamp": float(moment.timestamp),
    }


def _moment_from_dict(raw: Dict[str, Any]) -> ConsciousMoment:
    """Rebuild a ConsciousMoment with typed nested records."""
    return ConsciousMoment(
        precision_state=precision_state_from_dict(raw.get("precision_state") or {}),
        bound_percept=percept_from_dict(raw.get("bound_percept") or {}),
        self_reference=metrics_from_dict(raw.get("self_reference") or {}),
        loop_quality=float(raw.get("loop_quality", 0.0)),
        epistemic_depth=int(raw.get("epistemic_depth", 0)),
        binding_quality=float(raw.get("binding_quality", 0.0)),
        precision_balance=float(raw.get("precision_balance", 0.0)),
        is_field_evidencing=bool(raw.get("is_field_evidencing", False)),
        cessation_detected=bool(raw.get("cessation_detected", False)),
        cessation_count=int(raw.get("cessation_count", 0)),
        timestamp=float(raw.get("timestamp", time.time())),
    )
