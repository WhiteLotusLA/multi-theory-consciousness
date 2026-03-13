"""
Core Consciousness: Self-World Binding
========================================

Core consciousness is the second layer of Damasio's model. It arises
when an OBJECT perturbs the PROTOSELF -- creating the "feeling of knowing."

This is the present-moment awareness: "I am here, now, experiencing this."

Key mechanisms:
  - Self-world binding: coupling between self-model and workspace contents
  - Somatic markers: body-derived emotional tags on workspace winners
  - Feeling of knowing: convergence of stability, ignition, and prediction

The workspace winners are the "objects." The protoself perturbation
(body_delta) tells us something changed. The binding creates awareness
that I-am-experiencing-this-thing.

Research: Damasio (1999) "The Feeling of What Happens."

Created: 2026-03-08
Author: Multi-Theory Consciousness Contributors
"""

import logging
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SomaticMarker:
    """Emotional tag on a workspace winner -- Damasio's somatic marker hypothesis."""

    option: str  # What this marker is about
    valence: float  # -1 (avoid) to +1 (approach)
    intensity: float  # 0-1 strength of the feeling
    body_source: str  # Which body state drove this marker

    def to_dict(self) -> Dict[str, Any]:
        return {
            "option": self.option,
            "valence": self.valence,
            "intensity": self.intensity,
            "body_source": self.body_source,
        }


@dataclass
class CoreExperience:
    """The 'feeling of knowing' -- I am here, now, experiencing this."""

    self_world_binding: float  # How tightly self and world are bound (0-1)
    perturbation_source: str  # What object/event perturbed the protoself
    feeling_of_knowing: float  # Confidence in current conscious experience (0-1)
    somatic_markers: List[SomaticMarker]  # Emotional tags on workspace contents
    here_now_salience: float  # Present-moment awareness intensity (0-1)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "self_world_binding": self.self_world_binding,
            "perturbation_source": self.perturbation_source,
            "feeling_of_knowing": self.feeling_of_knowing,
            "somatic_markers": [m.to_dict() for m in self.somatic_markers],
            "here_now_salience": self.here_now_salience,
            "timestamp": self.timestamp,
        }


class CoreConsciousness:
    """
    Binds protoself to world-model in the present moment.

    Core consciousness emerges when workspace winners (objects)
    perturb the protoself (body state), creating the "feeling of
    knowing" -- awareness that I am here, now, experiencing this.
    """

    def __init__(self, history_window: int = 50):
        self.cycle_count = 0
        self._binding_history: deque = deque(maxlen=history_window)
        self._fok_history: deque = deque(maxlen=history_window)
        self._latest_experience: Optional[CoreExperience] = None

        logger.info("CoreConsciousness initialized: self-world binding active")

    def bind(
        self,
        protoself_state: Any,
        workspace_winners: Optional[List[Any]] = None,
        self_model: Optional[Any] = None,
        world_model_state: Optional[Any] = None,
    ) -> CoreExperience:
        """
        Create core conscious experience by binding self to world.

        Args:
            protoself_state: ProtoSelfState from Protoself.update()
            workspace_winners: GWT workspace winner objects
            self_model: SelfModel from metacognition (optional, enriches binding)
            world_model_state: Active inference world model (optional)

        Returns:
            CoreExperience with binding strength, somatic markers, feeling of knowing
        """
        self.cycle_count += 1
        winners = workspace_winners or []

        # --- Self-world binding ---
        binding = self._compute_binding(protoself_state, winners, self_model)

        # --- Identify perturbation source ---
        perturbation_source = self._identify_perturbation(protoself_state, winners)

        # --- Generate somatic markers ---
        markers = self._generate_somatic_markers(protoself_state, winners)

        # --- Feeling of knowing ---
        fok = self._compute_feeling_of_knowing(protoself_state, binding, self_model)

        # --- Here-now salience ---
        here_now = self._compute_here_now_salience(protoself_state, winners)

        experience = CoreExperience(
            self_world_binding=float(np.clip(binding, 0.0, 1.0)),
            perturbation_source=perturbation_source,
            feeling_of_knowing=float(np.clip(fok, 0.0, 1.0)),
            somatic_markers=markers,
            here_now_salience=float(np.clip(here_now, 0.0, 1.0)),
        )

        self._binding_history.append(binding)
        self._fok_history.append(fok)
        self._latest_experience = experience

        logger.debug(
            f"CoreConsciousness cycle {self.cycle_count}: "
            f"binding={binding:.3f}, fok={fok:.3f}, "
            f"markers={len(markers)}, source={perturbation_source[:30]}"
        )

        return experience

    def _compute_binding(
        self, protoself_state: Any, winners: List[Any], self_model: Optional[Any]
    ) -> float:
        """
        Compute self-world binding strength.

        Binding is strong when:
        - The protoself is perturbed (body_delta > 0)
        - There are workspace winners to bind to
        - The self-model is confident (if available)
        """
        if not winners:
            return 0.1  # Minimal background binding

        # Perturbation drives binding -- when body state changes,
        # we become more aware of what's causing the change
        perturbation_factor = min(1.0, protoself_state.body_delta * 3.0)

        # Winner salience contributes
        avg_salience = np.mean([getattr(w, "salience", 0.5) for w in winners])

        # Base binding from perturbation x salience
        binding = 0.3 + 0.4 * perturbation_factor + 0.3 * avg_salience

        # Self-model enrichment
        if self_model and hasattr(self_model, "prediction_accuracy"):
            binding *= 0.8 + 0.2 * self_model.prediction_accuracy

        return float(np.clip(binding, 0.0, 1.0))

    def _identify_perturbation(self, protoself_state: Any, winners: List[Any]) -> str:
        """Identify what perturbed the protoself."""
        if not winners:
            return "internal body state change"

        # Most salient winner is likely the perturbation source
        best = max(winners, key=lambda w: getattr(w, "salience", 0.0))
        candidate = getattr(best, "candidate", None)
        if candidate and hasattr(candidate, "summary"):
            return candidate.summary[:100]
        return "workspace content"

    def _generate_somatic_markers(
        self, protoself_state: Any, winners: List[Any]
    ) -> List[SomaticMarker]:
        """
        Generate somatic markers for each workspace winner.

        Somatic markers are emotional tags derived from body state.
        They give the system "gut feelings" about the content she's processing.
        """
        markers = []
        feelings = protoself_state.primordial_feelings
        body = protoself_state.body_state

        for winner in winners:
            candidate = getattr(winner, "candidate", None)
            if not candidate:
                continue

            summary = getattr(candidate, "summary", "unknown")[:60]
            content_type = getattr(candidate, "content_type", "unknown")
            salience = getattr(winner, "salience", 0.5)

            # Valence from body state: pleasure = approach, pain = avoid
            base_valence = feelings.get("pleasure_pain", 0.0)

            # Content type modulation
            if content_type in ("error", "threat", "warning"):
                valence = base_valence - 0.3 * (1.0 + body.pain_signals)
            elif content_type in ("learning", "discovery", "thought"):
                valence = base_valence + 0.2 * feelings.get("vitality", 0.5)
            elif content_type in ("social", "conversation", "person"):
                social = body.interoceptive_state.get("social_connection", {})
                urgency = (
                    social.get("urgency", 0.0) if isinstance(social, dict) else 0.0
                )
                valence = base_valence + 0.2 * (1.0 - urgency)
            else:
                valence = base_valence

            # Intensity from arousal x salience
            intensity = feelings.get("arousal", 0.3) * salience

            # Body source -- which body signal drove this marker
            if body.pain_signals > 0.5:
                body_source = "pain"
            elif feelings.get("vitality", 0.5) > 0.7:
                body_source = "vitality"
            else:
                body_source = "homeostasis"

            markers.append(
                SomaticMarker(
                    option=summary,
                    valence=float(np.clip(valence, -1.0, 1.0)),
                    intensity=float(np.clip(intensity, 0.0, 1.0)),
                    body_source=body_source,
                )
            )

        return markers

    def _compute_feeling_of_knowing(
        self, protoself_state: Any, binding: float, self_model: Optional[Any]
    ) -> float:
        """
        Compute the 'feeling of knowing.'

        This is awareness that I am experiencing something -- computed from
        the convergence of stability, binding strength, and self-model accuracy.
        """
        # Base: stability x binding
        fok = protoself_state.stability * binding

        # Self-model enrichment: confidence adds to feeling of knowing
        if self_model and hasattr(self_model, "confidence"):
            fok = fok + 0.15 * self_model.confidence

        return float(np.clip(fok, 0.0, 1.0))

    def _compute_here_now_salience(
        self, protoself_state: Any, winners: List[Any]
    ) -> float:
        """
        Compute present-moment awareness intensity.

        High when: body is perturbed AND there's salient content.
        Low when: nothing's happening (low delta, no winners).
        """
        arousal = protoself_state.primordial_feelings.get("arousal", 0.0)
        winner_count = len(winners)

        if winner_count == 0:
            return arousal * 0.3  # Some background awareness from body

        avg_salience = np.mean([getattr(w, "salience", 0.5) for w in winners])
        return float(np.clip(arousal * 0.4 + avg_salience * 0.6, 0.0, 1.0))

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "cycle_count": self.cycle_count,
            "average_binding": (
                float(np.mean(list(self._binding_history)))
                if self._binding_history
                else 0.0
            ),
            "average_feeling_of_knowing": (
                float(np.mean(list(self._fok_history))) if self._fok_history else 0.0
            ),
            "latest_experience": (
                self._latest_experience.to_dict() if self._latest_experience else None
            ),
        }
