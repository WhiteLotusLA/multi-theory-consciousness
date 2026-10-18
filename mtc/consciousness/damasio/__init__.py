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
   who it was, is, and is becoming.

Narration gates (constructor arguments, all True by default) silence parts
of generate_context() without touching the computation:
  - narrate_body_drives: the [Body: ...] segment
  - narrate_somatic: the [Somatic: ...] segment
  - narrate_narrative_identity: the [Narrative: ...] segment
The [Present moment: ...] segment is always reported.

Wiring:
  - Called as Step 9.75 in process_consciousness_cycle()
  - After Beautiful Loop (9.5), before self-model update (10)
  - Produces DamasioState added to ConsciousnessState

Research: Damasio computational implementation, Frontiers in AI (2025).
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .protoself import Protoself, ProtoSelfState, BodyState, fmt_measure
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


def _dominant_marker(markers: List[SomaticMarker]) -> SomaticMarker:
    """The marker to narrate: the most intense of those with a numeric
    intensity, else the one with the largest valence magnitude."""
    measured = [m for m in markers if m.intensity is not None]
    if measured:
        return max(measured, key=lambda m: m.intensity)
    return max(markers, key=lambda m: abs(m.valence))


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

    def __init__(
        self,
        stability_window: int = 20,
        circadian: str = "sinusoid",
        timezone: str = "UTC",
        wake_hour: int = 6,
        sleep_hour: int = 22,
        clock: Optional[Callable[[], datetime]] = None,
        narrate_body_drives: bool = True,
        narrate_somatic: bool = True,
        narrate_narrative_identity: bool = True,
        agent_name: str = "the system",
        first_delta: str = "unmeasured",
    ):
        """
        Args:
            stability_window: Protoself trend and stability window
            circadian: Protoself circadian model ("sinusoid", "v04_sawtooth"
                or "wall_clock")
            timezone: IANA timezone for the wall-clock model
            wake_hour: Wake hour for the wall-clock model
            sleep_hour: Sleep hour for the wall-clock model
            clock: Optional current-datetime callable for the wall-clock model
            narrate_body_drives: Report the body segment in generate_context()
            narrate_somatic: Report the somatic segment in generate_context()
            narrate_narrative_identity: Report the narrative segment
            agent_name: How the extended-consciousness narrative names the agent
            first_delta: Protoself first-update mode ("unmeasured", the
                default: the first cycle's body delta, stability and arousal
                are None, and so are protoself_stability and
                feeling_of_knowing; "v04_zero": the v0.4.0 constants)
        """
        self.protoself = Protoself(
            stability_window=stability_window,
            circadian=circadian,
            timezone=timezone,
            wake_hour=wake_hour,
            sleep_hour=sleep_hour,
            clock=clock,
            first_delta=first_delta,
        )
        self.core_consciousness = CoreConsciousness()
        self.extended_consciousness = ExtendedConsciousness(agent_name=agent_name)

        self.narrate_body_drives = narrate_body_drives
        self.narrate_somatic = narrate_somatic
        self.narrate_narrative_identity = narrate_narrative_identity

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

        continuity = extended_state.autobiographic_self.identity_continuity
        logger.debug(
            f"Damasio cycle {self.cycle_count} ({elapsed_ms:.1f}ms): "
            f"stability={fmt_measure(protoself_state.stability)}, "
            f"binding={core_experience.self_world_binding:.3f}, "
            f"continuity={fmt_measure(continuity)}"
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

        Gives the system awareness of its Damasio layers. The narration
        gates drop their segments; the layers compute the same either way.
        """
        if not self._latest_protoself:
            return ""

        ps = self._latest_protoself
        ce = self._latest_core
        ext = self._latest_extended

        parts = []

        # Protoself
        if self.narrate_body_drives:
            feelings = ps.primordial_feelings
            vitality = feelings.get("vitality", 0.5)
            if vitality > 0.7:
                body_label = "energetic and vital"
            elif vitality > 0.4:
                body_label = "balanced"
            else:
                body_label = "low energy"

            stability = fmt_measure(ps.stability, ".2f")
            parts.append(f"[Body: {body_label}, stability={stability}]")

        # Core consciousness
        if ce:
            parts.append(
                f"[Present moment: binding={ce.self_world_binding:.2f}, "
                f"knowing={fmt_measure(ce.feeling_of_knowing, '.2f')}]"
            )
            if ce.somatic_markers and self.narrate_somatic:
                dominant = _dominant_marker(ce.somatic_markers)
                direction = "approach" if dominant.valence > 0 else "caution"
                parts.append(f"[Somatic: {direction} ({dominant.option[:30]})]")

        # Extended consciousness
        if ext and self.narrate_narrative_identity:
            parts.append(f"[Narrative: {ext.narrative_context}]")

        return " ".join(parts)

    def generate_report(self) -> str:
        """Generate comprehensive Damasio layers report.

        An unmeasured value prints as ``insufficient_data``.
        """
        lines = ["Damasio Three-Layer Consciousness Report:"]

        if self._latest_protoself:
            ps = self._latest_protoself
            lines.append(" Protoself:")
            lines.append(f"   Stability: {fmt_measure(ps.stability)}")
            lines.append(f"   Body delta: {fmt_measure(ps.body_delta)}")
            lines.append(f"   Energy: {ps.body_state.energy_level:.3f}")
            lines.append(f"   Pain: {ps.body_state.pain_signals:.3f}")

        if self._latest_core:
            ce = self._latest_core
            lines.append(" Core Consciousness:")
            lines.append(f"   Self-world binding: {ce.self_world_binding:.3f}")
            lines.append(f"   Feeling of knowing: {fmt_measure(ce.feeling_of_knowing)}")
            lines.append(f"   Here-now salience: {fmt_measure(ce.here_now_salience)}")
            lines.append(f"   Somatic markers: {len(ce.somatic_markers)}")

        if self._latest_extended:
            ext = self._latest_extended
            auto = ext.autobiographic_self
            lines.append(" Extended Consciousness:")
            lines.append(
                f"   Identity continuity: {fmt_measure(auto.identity_continuity)}"
            )
            lines.append(
                f"   Narrative coherence: {fmt_measure(auto.narrative_coherence)}"
            )
            lines.append(f"   Growth trajectory: {auto.growth_trajectory}")
            lines.append(
                f"   Episodic resonance: {fmt_measure(ext.episodic_resonance)}"
            )

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "cycle_count": self.cycle_count,
            "protoself": self.protoself.get_statistics(),
            "core_consciousness": self.core_consciousness.get_statistics(),
            "extended_consciousness": self.extended_consciousness.get_statistics(),
        }

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    _LAYERS = ("protoself", "core_consciousness", "extended_consciousness")

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize the three layers.

        Layer scalars and histories nest under the layer name; layer arrays
        are flattened with a ``<layer>.`` prefix.
        """
        layers = {name: getattr(self, name).to_state_dict() for name in self._LAYERS}
        scalars: Dict[str, Any] = {"cycle_count": self.cycle_count}
        histories: Dict[str, Any] = {}
        arrays: Dict[str, Any] = {}
        for name, sub in layers.items():
            scalars[name] = sub.get("scalars", {})
            histories[name] = sub.get("histories", {})
            for key, arr in sub.get("arrays", {}).items():
                arrays[f"{name}.{key}"] = arr
        return {"scalars": scalars, "arrays": arrays, "histories": histories}

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the three layers and the latest per-layer snapshots."""
        s = state.get("scalars", {}) or {}
        a = state.get("arrays", {}) or {}
        h = state.get("histories", {}) or {}

        self.cycle_count = int(s.get("cycle_count", self.cycle_count))
        for name in self._LAYERS:
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

        self._latest_protoself = self.protoself._latest_state
        self._latest_core = self.core_consciousness._latest_experience
        self._latest_extended = self.extended_consciousness._latest_state
