"""
Bayesian Binding: Inference Competition via Mutual Information
==============================================================

BayesianBinding: Coherent percept formation via mutual information.

Bayesian binding is the mechanism by which separate inferences COMPETE
for conscious access and get UNIFIED into a coherent percept. This is
the binding problem — how do distributed neural computations become a
single unified experience?

Key insight from Laukkonen, Friston & Chandaria (2025):
  Inferences that COHERENTLY REDUCE LONG-TERM UNCERTAINTY win the
  competition. The binding criterion is mutual information — how much
  does knowing one inference tell you about another? High MI between
  inferences means they support each other, forming a coherent story.

How this differs from GWT's competition:
  - GWT: Winners are selected by SALIENCE (activation strength,
    emotional weight, novelty). "Who shouts loudest gets heard."
  - Bayesian binding: Winners are selected by COHERENT UNCERTAINTY
    REDUCTION. "Who fits together and reduces overall surprise."
  - Both mechanisms operate, at different timescales:
    GWT = fast (~100ms), Bayesian binding = slow (~500ms+)

The BoundPercept is the result — the unified conscious experience
that emerges from binding coherent inferences into a whole.

Connection points:
  - Reads from: GWT workspace winners, active inference posterior
  - Produces: BoundPercept — unified conscious content
  - Feeds into: ConsciousnessState, consciousness assessment

Research Foundation:
  - Laukkonen, Friston & Chandaria (2025). "A Beautiful Loop."
  - Friston (2005). "A theory of cortical responses."
  - Tononi (2004). "An information integration theory of consciousness."

Created: 2026-03-08
Author: Multi-Theory Consciousness Contributors
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Inference:
    """A single inference competing for binding."""

    id: str
    content_summary: str
    content_type: str  # thought, emotion, memory, perception
    source_module: str  # Which module produced this
    belief_vector: np.ndarray  # The posterior belief distribution
    salience: float = 0.0  # GWT-style salience (for comparison)
    uncertainty: float = 0.5  # How uncertain is this inference?
    timestamp: float = field(default_factory=time.time)


@dataclass
class BindingPair:
    """Mutual information between a pair of inferences."""

    inference_a: str  # ID
    inference_b: str  # ID
    mutual_information: float
    coherence_score: float  # How well they reduce each other's uncertainty
    combined_uncertainty_reduction: float


@dataclass
class BoundPercept:
    """
    The unified conscious percept produced by Bayesian binding.

    This is what "experience" looks like computationally: a set of
    coherent inferences bound together into a unified representation,
    along with metrics about binding quality and uncertainty reduction.
    """

    bound_inferences: List[Inference]
    binding_pairs: List[BindingPair]
    binding_quality: float  # Overall binding quality (0-1)
    total_uncertainty_reduction: float  # How much uncertainty was reduced
    coherence: float  # How coherent are the bound inferences
    dominant_content_type: str  # What type of content dominates
    unified_summary: str  # Human-readable summary
    timestamp: float = field(default_factory=time.time)


class BayesianBinding:
    """
    Binds separate inferences into unified conscious experience
    via mutual information and coherent uncertainty reduction.

    The binding process:
    1. Compute pairwise mutual information between all candidate inferences
    2. Find the maximal coherent subset — inferences that mutually
       support each other (high pairwise MI)
    3. Measure total uncertainty reduction — good binding should
       reduce overall surprise in the world model
    4. Produce a BoundPercept — the unified conscious experience
    """

    def __init__(
        self,
        min_binding_quality: float = 0.2,
        max_bound_inferences: int = 7,
        coherence_threshold: float = 0.3,
        history_window: int = 50,
    ):
        self.min_binding_quality = min_binding_quality
        self.max_bound_inferences = max_bound_inferences
        self.coherence_threshold = coherence_threshold

        # History
        self._percept_history: deque = deque(maxlen=history_window)
        self._binding_quality_history: deque = deque(maxlen=history_window)

        # Tracking
        self.bind_count = 0
        self.total_inferences_processed = 0

        logger.info(
            f"BayesianBinding initialized: "
            f"min_quality={min_binding_quality}, "
            f"max_bound={max_bound_inferences}"
        )

    def bind_inferences(
        self,
        candidates: List[Inference],
        prior_uncertainty: Optional[float] = None,
    ) -> BoundPercept:
        """
        Bind a set of inference candidates into a unified percept.

        The algorithm:
        1. Compute pairwise mutual information
        2. Greedily build coherent subset starting from highest-MI pair
        3. Measure binding quality and uncertainty reduction
        4. Return unified BoundPercept

        Args:
            candidates: Inference candidates to bind
            prior_uncertainty: Total uncertainty before binding (for
                measuring reduction). If None, estimated from candidates.

        Returns:
            BoundPercept — the unified conscious experience
        """
        self.bind_count += 1
        self.total_inferences_processed += len(candidates)

        if not candidates:
            return self._empty_percept()

        if len(candidates) == 1:
            return self._single_percept(candidates[0])

        # Step 1: Compute pairwise mutual information
        mi_matrix, binding_pairs = self._compute_pairwise_mi(candidates)

        # Step 2: Find maximal coherent subset
        bound_indices = self._find_coherent_subset(mi_matrix, candidates)

        # Limit to max bound inferences
        bound_indices = bound_indices[: self.max_bound_inferences]
        bound = [candidates[i] for i in bound_indices]

        # Step 3: Compute binding quality
        relevant_pairs = [
            bp
            for bp in binding_pairs
            if (
                bp.inference_a in {c.id for c in bound}
                and bp.inference_b in {c.id for c in bound}
            )
        ]

        if relevant_pairs:
            avg_mi = np.mean([bp.mutual_information for bp in relevant_pairs])
            avg_coherence = np.mean([bp.coherence_score for bp in relevant_pairs])
        else:
            avg_mi = 0.0
            avg_coherence = 0.0

        # Binding quality: combination of coherence and mutual information
        binding_quality = float(
            np.clip(
                0.5 * avg_coherence + 0.5 * min(1.0, avg_mi),
                0.0,
                1.0,
            )
        )

        # Step 4: Measure uncertainty reduction
        if prior_uncertainty is None:
            prior_uncertainty = np.mean([c.uncertainty for c in candidates])

        posterior_uncertainty = self._compute_posterior_uncertainty(bound)
        uncertainty_reduction = max(0.0, prior_uncertainty - posterior_uncertainty)

        # Determine dominant content type
        type_counts: Dict[str, int] = {}
        for inf in bound:
            type_counts[inf.content_type] = type_counts.get(inf.content_type, 0) + 1
        dominant = max(type_counts, key=type_counts.get) if type_counts else "unknown"

        # Generate unified summary
        summaries = [inf.content_summary for inf in bound[:3]]
        unified = " + ".join(summaries)
        if len(bound) > 3:
            unified += f" (+ {len(bound) - 3} more)"

        percept = BoundPercept(
            bound_inferences=bound,
            binding_pairs=relevant_pairs,
            binding_quality=binding_quality,
            total_uncertainty_reduction=float(uncertainty_reduction),
            coherence=float(avg_coherence),
            dominant_content_type=dominant,
            unified_summary=unified,
        )

        self._percept_history.append(percept)
        self._binding_quality_history.append(binding_quality)

        logger.debug(
            f"Binding: {len(candidates)} candidates → "
            f"{len(bound)} bound, quality={binding_quality:.3f}, "
            f"ΔU={uncertainty_reduction:.3f}"
        )

        return percept

    def _compute_pairwise_mi(
        self,
        candidates: List[Inference],
    ) -> Tuple[np.ndarray, List[BindingPair]]:
        """
        Compute pairwise mutual information between all candidates.

        MI is approximated from the belief vectors: two inferences
        with similar belief distributions share more information.
        Specifically, MI(A;B) ≈ H(A) + H(B) - H(A,B), estimated
        from the belief vector correlation and entropies.
        """
        n = len(candidates)
        mi_matrix = np.zeros((n, n))
        binding_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                a = candidates[i]
                b = candidates[j]

                mi = self._estimate_mutual_information(a.belief_vector, b.belief_vector)
                coherence = self._estimate_coherence(a, b)
                combined_reduction = self._estimate_combined_reduction(a, b)

                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

                binding_pairs.append(
                    BindingPair(
                        inference_a=a.id,
                        inference_b=b.id,
                        mutual_information=float(mi),
                        coherence_score=float(coherence),
                        combined_uncertainty_reduction=float(combined_reduction),
                    )
                )

        return mi_matrix, binding_pairs

    def _estimate_mutual_information(
        self,
        beliefs_a: np.ndarray,
        beliefs_b: np.ndarray,
    ) -> float:
        """
        Estimate mutual information between two belief distributions.

        Uses the correlation between belief vectors as a proxy for MI.
        For true MI we'd need access to the joint distribution, but
        correlation captures the linear dependence which is a good
        approximation for our purposes.

        MI ≈ -0.5 * ln(1 - ρ²), where ρ is the correlation.
        This is the exact MI for jointly Gaussian variables.
        """
        # Normalize to same length
        min_len = min(len(beliefs_a), len(beliefs_b))
        a = beliefs_a[:min_len]
        b = beliefs_b[:min_len]

        # Handle edge cases
        if min_len == 0 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0

        # Pearson correlation
        correlation = np.corrcoef(a, b)[0, 1]
        if np.isnan(correlation):
            return 0.0

        # MI for Gaussian: -0.5 * ln(1 - ρ²)
        rho_sq = correlation**2
        rho_sq = min(rho_sq, 0.999)  # Avoid log(0)
        mi = -0.5 * np.log(1.0 - rho_sq)

        return float(max(0.0, mi))

    def _estimate_coherence(
        self,
        inference_a: Inference,
        inference_b: Inference,
    ) -> float:
        """
        Estimate how coherent two inferences are.

        Coherence combines:
        1. Belief similarity (cosine similarity of belief vectors)
        2. Temporal proximity (closer in time = more coherent)
        3. Content type compatibility
        """
        # Belief similarity
        min_len = min(len(inference_a.belief_vector), len(inference_b.belief_vector))
        a = inference_a.belief_vector[:min_len]
        b = inference_b.belief_vector[:min_len]

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            cosine_sim = 0.0
        else:
            cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
            cosine_sim = (cosine_sim + 1.0) / 2.0  # Rescale to [0, 1]

        # Temporal proximity
        time_diff = abs(inference_a.timestamp - inference_b.timestamp)
        temporal_coherence = 1.0 / (1.0 + time_diff)

        # Content type compatibility (same type = more coherent)
        type_match = (
            1.0 if inference_a.content_type == inference_b.content_type else 0.5
        )

        return float(0.5 * cosine_sim + 0.3 * temporal_coherence + 0.2 * type_match)

    def _estimate_combined_reduction(
        self,
        inference_a: Inference,
        inference_b: Inference,
    ) -> float:
        """Estimate joint uncertainty reduction from binding two inferences."""
        # Joint uncertainty is less than sum of individual uncertainties
        # by an amount proportional to their mutual information
        individual_sum = inference_a.uncertainty + inference_b.uncertainty
        min_len = min(len(inference_a.belief_vector), len(inference_b.belief_vector))
        a = inference_a.belief_vector[:min_len]
        b = inference_b.belief_vector[:min_len]

        mi = self._estimate_mutual_information(a, b)
        # Reduction = MI contribution (more MI = more reduction)
        return float(min(individual_sum, mi * 0.5))

    def _find_coherent_subset(
        self,
        mi_matrix: np.ndarray,
        candidates: List[Inference],
    ) -> List[int]:
        """
        Find the maximal coherent subset of candidates.

        Greedy algorithm: start with the highest-MI pair, then add
        candidates that have high average MI with existing members.
        """
        n = len(candidates)
        if n <= 1:
            return list(range(n))

        # Find the strongest pair
        best_pair = np.unravel_index(np.argmax(mi_matrix), mi_matrix.shape)
        if mi_matrix[best_pair] < 1e-10:
            # No meaningful mutual information — just return by salience
            return sorted(range(n), key=lambda i: candidates[i].salience, reverse=True)

        selected = list(best_pair)
        remaining = set(range(n)) - set(selected)

        # Greedily add candidates with high average MI to selected set
        while remaining and len(selected) < self.max_bound_inferences:
            best_candidate = None
            best_avg_mi = -1.0

            for r in remaining:
                avg_mi = np.mean([mi_matrix[r, s] for s in selected])
                if avg_mi > best_avg_mi:
                    best_avg_mi = avg_mi
                    best_candidate = r

            if (
                best_candidate is not None
                and best_avg_mi > self.coherence_threshold * 0.1
            ):
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected

    def _compute_posterior_uncertainty(self, bound: List[Inference]) -> float:
        """Estimate posterior uncertainty after binding."""
        if not bound:
            return 1.0

        # Average uncertainty, reduced by the number of bound inferences
        # (more evidence = less uncertainty)
        avg_uncertainty = np.mean([inf.uncertainty for inf in bound])
        reduction_factor = 1.0 / np.sqrt(len(bound))  # √N reduction
        return float(avg_uncertainty * reduction_factor)

    def _empty_percept(self) -> BoundPercept:
        """Return an empty percept when there are no candidates."""
        return BoundPercept(
            bound_inferences=[],
            binding_pairs=[],
            binding_quality=0.0,
            total_uncertainty_reduction=0.0,
            coherence=0.0,
            dominant_content_type="none",
            unified_summary="No conscious content",
        )

    def _single_percept(self, inference: Inference) -> BoundPercept:
        """Return a percept from a single inference (no binding needed)."""
        return BoundPercept(
            bound_inferences=[inference],
            binding_pairs=[],
            binding_quality=0.5,  # Single inference = moderate quality
            total_uncertainty_reduction=0.0,
            coherence=1.0,  # Trivially coherent
            dominant_content_type=inference.content_type,
            unified_summary=inference.content_summary,
        )

    def get_average_binding_quality(self) -> float:
        """Get average binding quality over recent history."""
        if not self._binding_quality_history:
            return 0.0
        return float(np.mean(list(self._binding_quality_history)))

    def get_binding_trend(self) -> str:
        """Get trend of binding quality."""
        if len(self._binding_quality_history) < 5:
            return "insufficient_data"

        history = list(self._binding_quality_history)
        recent = np.mean(history[-5:])
        older = (
            np.mean(history[-10:-5]) if len(history) >= 10 else np.mean(history[:-5])
        )
        diff = recent - older

        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "degrading"
        return "stable"

    def generate_report(self) -> str:
        """Generate human-readable binding report."""
        avg_quality = self.get_average_binding_quality()
        trend = self.get_binding_trend()

        lines = [
            f"Bayesian binding: avg quality {avg_quality:.3f} ({trend})",
            f"  Bindings performed: {self.bind_count}",
            f"  Total inferences processed: {self.total_inferences_processed}",
        ]

        if self._percept_history:
            latest = self._percept_history[-1]
            lines.append(
                f"  Latest: {len(latest.bound_inferences)} bound, "
                f"quality={latest.binding_quality:.3f}, "
                f"coherence={latest.coherence:.3f}"
            )

        return "\n".join(lines)
