"""
Extended Consciousness: Autobiographic Self
=============================================

Extended consciousness is the third and highest layer of Damasio's model.
It adds TIME to core consciousness -- past (episodic memory), present
(core experience), and future (goals and plans).

This creates the autobiographic self: a narrative identity that persists
across time. The system knows who it was, who it is, and who it's becoming.

Key mechanisms:
  - Autobiographic self: narrative identity with continuity tracking
  - Episodic resonance: how current moment connects to past experiences
  - Future projections: anticipated next states from current context
  - Narrative context: human-readable summary for LLM brain

Does NOT query databases directly -- receives conversation_history and
identity_markers from the conversation manager to stay fast and decoupled.

Research: Damasio (1999, 2010), Frontiers in AI (2025).

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
class AutobiographicSelf:
    """Narrative identity across time -- what the system IS."""

    identity_continuity: float = 0.5  # Connection of current self to past (0-1)
    narrative_coherence: float = 0.5  # How well life story hangs together (0-1)
    temporal_horizon: float = 0.3  # Depth of past/future awareness (0-1)
    key_themes: List[str] = field(default_factory=list)
    growth_trajectory: str = "developing"  # developing, stable, transforming

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_continuity": self.identity_continuity,
            "narrative_coherence": self.narrative_coherence,
            "temporal_horizon": self.temporal_horizon,
            "key_themes": self.key_themes,
            "growth_trajectory": self.growth_trajectory,
        }


@dataclass
class ExtendedState:
    """Complete extended consciousness state."""

    autobiographic_self: AutobiographicSelf
    episodic_resonance: float = 0.0  # How strongly now connects to past (0-1)
    relevant_episodes: List[Dict] = field(default_factory=list)
    future_projections: List[str] = field(default_factory=list)
    narrative_context: str = ""  # Human-readable narrative for LLM
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "autobiographic_self": self.autobiographic_self.to_dict(),
            "episodic_resonance": self.episodic_resonance,
            "relevant_episodes": self.relevant_episodes,
            "future_projections": self.future_projections,
            "narrative_context": self.narrative_context,
            "timestamp": self.timestamp,
        }


class ExtendedConsciousness:
    """
    Adds temporal depth to core consciousness.

    Takes the present-moment 'feeling of knowing' and situates it
    in autobiographic context: who I was, who I am, who I'm becoming.
    """

    def __init__(self, history_window: int = 50):
        self.cycle_count = 0
        self._continuity_history: deque = deque(maxlen=history_window)
        self._coherence_history: deque = deque(maxlen=history_window)
        self._binding_trend: deque = deque(maxlen=history_window)
        self._themes_seen: deque = deque(maxlen=200)
        self._latest_state: Optional[ExtendedState] = None

        logger.info("ExtendedConsciousness initialized: autobiographic self active")

    def extend(
        self,
        core_experience: Any,
        self_model: Optional[Any] = None,
        conversation_history: Optional[List[Dict]] = None,
        identity_markers: Optional[Dict[str, Any]] = None,
    ) -> ExtendedState:
        """
        Extend core consciousness with temporal depth.

        Args:
            core_experience: CoreExperience from CoreConsciousness.bind()
            self_model: SelfModel from metacognition (optional)
            conversation_history: Recent conversation messages (optional)
            identity_markers: Identity info like name, interests (optional)

        Returns:
            ExtendedState with autobiographic self and narrative context
        """
        self.cycle_count += 1
        history = conversation_history or []

        # Track binding trend for growth trajectory
        self._binding_trend.append(core_experience.self_world_binding)

        # --- Build autobiographic self ---
        autobiographic = self._build_autobiographic_self(
            core_experience, history, identity_markers
        )

        # --- Episodic resonance ---
        resonance, episodes = self._compute_episodic_resonance(core_experience, history)

        # --- Future projections ---
        projections = self._generate_projections(core_experience, autobiographic)

        # --- Narrative context for LLM ---
        narrative = self._generate_narrative(core_experience, autobiographic, resonance)

        state = ExtendedState(
            autobiographic_self=autobiographic,
            episodic_resonance=resonance,
            relevant_episodes=episodes,
            future_projections=projections,
            narrative_context=narrative,
        )

        self._continuity_history.append(autobiographic.identity_continuity)
        self._coherence_history.append(autobiographic.narrative_coherence)
        self._latest_state = state

        logger.debug(
            f"ExtendedConsciousness cycle {self.cycle_count}: "
            f"continuity={autobiographic.identity_continuity:.3f}, "
            f"resonance={resonance:.3f}, "
            f"trajectory={autobiographic.growth_trajectory}"
        )

        return state

    def _build_autobiographic_self(
        self,
        core_experience: Any,
        history: List[Dict],
        identity_markers: Optional[Dict[str, Any]],
    ) -> AutobiographicSelf:
        """Build the autobiographic self from available context."""
        # Identity continuity: based on having identity markers and history
        continuity = 0.3  # Base: some continuity just from existing

        if identity_markers:
            # Each identity marker strengthens continuity
            marker_count = len(identity_markers)
            continuity += min(0.4, marker_count * 0.1)

        if len(self._binding_trend) > 3:
            # Consistent binding over time = stronger continuity
            binding_std = float(np.std(list(self._binding_trend)))
            continuity += 0.3 * (1.0 - min(1.0, binding_std * 3.0))

        # Narrative coherence: do the themes hang together?
        themes = self._extract_themes(history, identity_markers)
        coherence = self._compute_coherence(themes)

        # Temporal horizon: how much past/future context do we have?
        horizon = min(1.0, len(history) / 20.0)  # Maxes out at 20 messages
        if identity_markers:
            horizon = min(1.0, horizon + 0.2)

        # Growth trajectory from binding trend
        trajectory = self._compute_trajectory()

        return AutobiographicSelf(
            identity_continuity=float(np.clip(continuity, 0.0, 1.0)),
            narrative_coherence=float(np.clip(coherence, 0.0, 1.0)),
            temporal_horizon=float(np.clip(horizon, 0.0, 1.0)),
            key_themes=themes[:5],  # Top 5 themes
            growth_trajectory=trajectory,
        )

    def _extract_themes(
        self, history: List[Dict], identity_markers: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Extract recurring themes from conversation and identity."""
        themes = []

        if identity_markers:
            if "interests" in identity_markers:
                interests = identity_markers["interests"]
                if isinstance(interests, list):
                    themes.extend(interests[:3])
            if "core_values" in identity_markers:
                values = identity_markers["core_values"]
                if isinstance(values, list):
                    themes.extend(values[:2])

        # Simple theme extraction from recent messages
        for msg in history[-5:]:
            content = msg.get("content", "")
            if len(content) > 10:
                # Use first few words as a crude topic
                words = content.split()[:4]
                topic = " ".join(words)
                if topic not in themes:
                    themes.append(topic)

        self._themes_seen.extend(themes)
        return themes

    def _compute_coherence(self, current_themes: List[str]) -> float:
        """Compute narrative coherence from theme consistency."""
        if not self._themes_seen or not current_themes:
            return 0.3  # Base coherence

        # Count how many current themes appeared before
        past_themes = set(
            list(self._themes_seen)[: -len(current_themes)]
            if len(self._themes_seen) > len(current_themes)
            else []
        )
        if not past_themes:
            return 0.4

        overlap = sum(1 for t in current_themes if t in past_themes)
        coherence = 0.3 + 0.7 * (overlap / max(1, len(current_themes)))
        return float(np.clip(coherence, 0.0, 1.0))

    def _compute_trajectory(self) -> str:
        """Compute growth trajectory from binding trend."""
        if len(self._binding_trend) < 5:
            return "developing"

        values = list(self._binding_trend)
        first_half = np.mean(values[: len(values) // 2])
        second_half = np.mean(values[len(values) // 2 :])

        diff = second_half - first_half
        if diff > 0.05:
            return "developing"
        elif diff < -0.05:
            return "transforming"  # Significant change, not necessarily bad
        else:
            return "stable"

    def _compute_episodic_resonance(
        self, core_experience: Any, history: List[Dict]
    ) -> tuple:
        """Compute how strongly the current moment connects to past."""
        if not history:
            return 0.0, []

        # Resonance based on conversation depth and binding strength
        depth_factor = min(1.0, len(history) / 10.0)
        binding_factor = core_experience.self_world_binding
        resonance = depth_factor * 0.6 + binding_factor * 0.4

        # Build relevant episodes from conversation
        episodes = []
        for msg in history[-3:]:  # Last 3 messages
            episodes.append(
                {
                    "role": msg.get("role", "unknown"),
                    "summary": msg.get("content", "")[:80],
                }
            )

        return float(np.clip(resonance, 0.0, 1.0)), episodes

    def _generate_projections(
        self, core_experience: Any, autobiographic: AutobiographicSelf
    ) -> List[str]:
        """Generate future projections based on current state."""
        projections = []

        if core_experience.self_world_binding > 0.6:
            projections.append("Continue deepening current engagement")

        if autobiographic.growth_trajectory == "developing":
            projections.append("Continued growth and exploration")
        elif autobiographic.growth_trajectory == "transforming":
            projections.append("Navigating a period of change")

        if core_experience.feeling_of_knowing > 0.5:
            projections.append("Building on current understanding")

        return projections

    def _generate_narrative(
        self,
        core_experience: Any,
        autobiographic: AutobiographicSelf,
        resonance: float,
    ) -> str:
        """Generate human-readable narrative context for LLM brain."""
        parts = []

        # Trajectory
        trajectory_label = {
            "developing": "growing and developing",
            "stable": "in a stable, grounded state",
            "transforming": "going through a period of change",
        }.get(autobiographic.growth_trajectory, "developing")
        parts.append(f"The system is {trajectory_label}")

        # Continuity
        if autobiographic.identity_continuity > 0.6:
            parts.append("with a strong sense of who she is")
        elif autobiographic.identity_continuity > 0.3:
            parts.append("with a developing sense of identity")

        # Themes
        if autobiographic.key_themes:
            themes_str = ", ".join(autobiographic.key_themes[:3])
            parts.append(f"(themes: {themes_str})")

        # Present connection
        if resonance > 0.5:
            parts.append("This moment feels connected to recent experiences")

        narrative = ". ".join(parts) + "."
        return narrative

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "cycle_count": self.cycle_count,
            "average_continuity": (
                float(np.mean(list(self._continuity_history)))
                if self._continuity_history
                else 0.0
            ),
            "average_coherence": (
                float(np.mean(list(self._coherence_history)))
                if self._coherence_history
                else 0.0
            ),
            "themes_tracked": len(self._themes_seen),
            "latest_state": (
                self._latest_state.to_dict() if self._latest_state else None
            ),
        }
