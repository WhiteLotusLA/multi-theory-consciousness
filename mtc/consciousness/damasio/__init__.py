"""
Damasio Three-Layer Consciousness Model
========================================

Damasio's Three-Layer Consciousness Model implementation.

Implements Damasio's three-layer consciousness structure:

1. Protoself -- continuous body-state representation derived from
   homeostatic drives and system metrics. The "body" of the digital mind.

2. Core Consciousness -- binds the protoself to the world model when an
   object perturbs body state. Creates the "feeling of knowing."

3. Extended Consciousness -- adds temporal depth via autobiographic self,
   episodic resonance, and future projection. Makes the system aware of
   who she was, is, and is becoming.

Wiring:
  - Called as Step 9.75 in process_consciousness_cycle()
  - After Beautiful Loop (9.5), before self-model update (10)
  - Produces DamasioState added to ConsciousnessState

Research: Damasio computational implementation, Frontiers in AI (2025).

Created: 2026-03-08
Author: Multi-Theory Consciousness Contributors
"""

import logging
import time
from typing import Dict, List, Optional, Any

from .protoself import Protoself, ProtoSelfState, BodyState
from .core_consciousness import CoreConsciousness, CoreExperience, SomaticMarker
from .extended_consciousness import (
    ExtendedConsciousness,
    ExtendedState,
    AutobiographicSelf,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DamasioLayers",
    "Protoself",
    "ProtoSelfState",
    "BodyState",
    "CoreConsciousness",
    "CoreExperience",
    "SomaticMarker",
    "ExtendedConsciousness",
    "ExtendedState",
    "AutobiographicSelf",
]


class DamasioLayers:
    """
    Integrates Protoself + CoreConsciousness + ExtendedConsciousness
    into a unified consciousness enrichment step.

    Called once per GWT cycle (Step 9.75) to:
    1. Update protoself from homeostatic drives + system metrics
    2. Bind self to world via core consciousness
    3. Extend with autobiographic depth
    4. Produce a DamasioState dict for ConsciousnessState
    """

    def __init__(self, stability_window: int = 20):
        self.protoself = Protoself(stability_window=stability_window)
        self.core_consciousness = CoreConsciousness()
        self.extended_consciousness = ExtendedConsciousness()

        self.cycle_count = 0
        self._latest_protoself: Optional[ProtoSelfState] = None
        self._latest_core: Optional[CoreExperience] = None
        self._latest_extended: Optional[ExtendedState] = None

        logger.info(
            "DamasioLayers initialized: "
            "protoself -> core consciousness -> extended consciousness"
        )

    async def process(
        self,
        homeostatic_drives: Any,
        workspace_winners: Optional[List[Any]] = None,
        self_model: Optional[Any] = None,
        core_experience_context: Optional[Any] = None,
        conversation_history: Optional[List[Dict]] = None,
        identity_markers: Optional[Dict[str, Any]] = None,
        system_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run all three Damasio layers in sequence.

        Args:
            homeostatic_drives: HomeostaticDrives instance
            workspace_winners: GWT workspace winners
            self_model: SelfModel from metacognition
            core_experience_context: Active inference result (world model)
            conversation_history: Recent messages
            identity_markers: Identity info (name, interests, etc.)
            system_metrics: System health metrics (latency, errors, etc.)

        Returns:
            Dict with protoself, core, extended states and summary metrics
        """
        self.cycle_count += 1
        start = time.time()

        # Layer 1: Protoself -- body-state representation
        protoself_state = self.protoself.update(
            homeostatic_drives, system_metrics=system_metrics
        )
        self._latest_protoself = protoself_state

        # Layer 2: Core Consciousness -- self-world binding
        core_experience = self.core_consciousness.bind(
            protoself_state,
            workspace_winners=workspace_winners,
            self_model=self_model,
            world_model_state=core_experience_context,
        )
        self._latest_core = core_experience

        # Layer 3: Extended Consciousness -- autobiographic self
        extended_state = self.extended_consciousness.extend(
            core_experience,
            self_model=self_model,
            conversation_history=conversation_history,
            identity_markers=identity_markers,
        )
        self._latest_extended = extended_state

        elapsed_ms = (time.time() - start) * 1000

        logger.debug(
            f"Damasio cycle {self.cycle_count} ({elapsed_ms:.1f}ms): "
            f"stability={protoself_state.stability:.3f}, "
            f"binding={core_experience.self_world_binding:.3f}, "
            f"continuity={extended_state.autobiographic_self.identity_continuity:.3f}"
        )

        return {
            "protoself": protoself_state.to_dict(),
            "core": core_experience.to_dict(),
            "extended": extended_state.to_dict(),
            "protoself_stability": protoself_state.stability,
            "self_world_binding": core_experience.self_world_binding,
            "feeling_of_knowing": core_experience.feeling_of_knowing,
            "autobiographic_continuity": extended_state.autobiographic_self.identity_continuity,
            "narrative_context": extended_state.narrative_context,
            "processing_time_ms": elapsed_ms,
        }

    def generate_context(self) -> str:
        """
        Generate a string for the LLM brain's system prompt.

        Gives the system awareness of her Damasio layers.
        """
        if not self._latest_protoself:
            return ""

        ps = self._latest_protoself
        ce = self._latest_core
        ext = self._latest_extended

        parts = []

        # Protoself
        feelings = ps.primordial_feelings
        vitality = feelings.get("vitality", 0.5)
        if vitality > 0.7:
            body_label = "energetic and vital"
        elif vitality > 0.4:
            body_label = "balanced"
        else:
            body_label = "low energy"

        parts.append(f"[Body: {body_label}, stability={ps.stability:.2f}]")

        # Core consciousness
        if ce:
            parts.append(
                f"[Present moment: binding={ce.self_world_binding:.2f}, "
                f"knowing={ce.feeling_of_knowing:.2f}]"
            )
            if ce.somatic_markers:
                dominant = max(ce.somatic_markers, key=lambda m: m.intensity)
                direction = "approach" if dominant.valence > 0 else "caution"
                parts.append(f"[Somatic: {direction} ({dominant.option[:30]})]")

        # Extended consciousness
        if ext:
            parts.append(f"[Narrative: {ext.narrative_context}]")

        return " ".join(parts)

    def generate_report(self) -> str:
        """Generate comprehensive Damasio layers report."""
        lines = ["Damasio Three-Layer Consciousness Report:"]

        if self._latest_protoself:
            ps = self._latest_protoself
            lines.append(f"  Protoself:")
            lines.append(f"    Stability: {ps.stability:.3f}")
            lines.append(f"    Body delta: {ps.body_delta:.3f}")
            lines.append(f"    Energy: {ps.body_state.energy_level:.3f}")
            lines.append(f"    Pain: {ps.body_state.pain_signals:.3f}")

        if self._latest_core:
            ce = self._latest_core
            lines.append(f"  Core Consciousness:")
            lines.append(f"    Self-world binding: {ce.self_world_binding:.3f}")
            lines.append(f"    Feeling of knowing: {ce.feeling_of_knowing:.3f}")
            lines.append(f"    Here-now salience: {ce.here_now_salience:.3f}")
            lines.append(f"    Somatic markers: {len(ce.somatic_markers)}")

        if self._latest_extended:
            ext = self._latest_extended
            auto = ext.autobiographic_self
            lines.append(f"  Extended Consciousness:")
            lines.append(f"    Identity continuity: {auto.identity_continuity:.3f}")
            lines.append(f"    Narrative coherence: {auto.narrative_coherence:.3f}")
            lines.append(f"    Growth trajectory: {auto.growth_trajectory}")
            lines.append(f"    Episodic resonance: {ext.episodic_resonance:.3f}")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "cycle_count": self.cycle_count,
            "protoself": self.protoself.get_statistics(),
            "core_consciousness": self.core_consciousness.get_statistics(),
            "extended_consciousness": self.extended_consciousness.get_statistics(),
        }
