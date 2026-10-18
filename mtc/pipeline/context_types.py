"""
Pipeline Context Types
======================

The data that crosses the seams of one conversation turn: what the pipeline
gathers before the consciousness cycle (``RawConversationContext``), what the
orchestrator hands the brain after its four modulation layers
(``ModulatedContext`` with its ``TheorySections``), and what a turn returns to
the caller (``TurnResult``).

Honesty rules carried by these types: every Optional consciousness signal the
orchestrator could not fill is named in ``ModulatedContext.absent_signals``
with a reason from ``ABSENT_REASONS``; ``pad_sources_measured`` lists the PAD
inputs that were measured this cycle; a failed cycle yields
``ModulatedContext.unavailable`` with an empty narrative, never a composed
one; ``cycle_winner`` names a cycle's winner only when that cycle ignited.
``TurnResult.to_payload`` carries the speaker's display label only, never the
identity's trust, role, authentication state or any token; it also carries the
state of the safety gate for the turn (``SAFETY_GATE_STATES``: ``unchecked``
when the check raised or did not run, ``applied`` only when it ran) and, for a
blocked turn, the safety monitor's decision.

The message, history, memories, thoughts and the speaker's name are private
conversation data at runtime. Nothing here reads Settings or persists state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from mtc.pipeline.speaker_trust import SpeakerIdentity

ABSENT_REASONS: Tuple[str, ...] = (
    "warming_up",
    "cadence_not_reached",
    "no_producer",
    "no_input",
    "disabled",
    "failed",
)

# The safety gate of a turn: no safety stack configured, skipped for a trusted
# identity, a check that ran, or a check that raised or did not run.
SAFETY_GATE_STATES: Tuple[str, ...] = ("absent", "bypassed", "applied", "unchecked")


def cycle_winner(state: Any) -> Optional[Any]:
    """The candidate that won this cycle, or None.

    The workspace keeps its primary content across cycles in which nothing
    ignites, so the primary content counts as this cycle's winner only when
    the cycle's ``ignition_events`` is a positive count.
    """
    ignition = getattr(state, "ignition_events", None)
    if isinstance(ignition, bool) or not isinstance(ignition, int) or ignition <= 0:
        return None
    primary = getattr(state, "primary_content", None)
    return getattr(primary, "candidate", None) if primary is not None else None


@dataclass
class RawConversationContext:
    """Everything the pipeline gathers before the consciousness cycle."""

    message: str
    identity: SpeakerIdentity
    retrieved_memories: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_concepts: List[Dict[str, Any]] = field(default_factory=list)
    ctm_thoughts: List[Dict[str, Any]] = field(default_factory=list)
    thought_gists: List[Any] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    core_memory: Dict[str, str] = field(default_factory=dict)
    coherence_context: Dict[str, Any] = field(default_factory=dict)
    tom_context: str = ""
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    now: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SpeakerIdentity):
            raise TypeError(
                "RawConversationContext.identity must be a SpeakerIdentity "
                "from SpeakerTrust.verify"
            )


@dataclass
class TheorySections:
    """The five theory slots of the prompt, filled from the module reports.

    A slot is a string or a mapping the prompt assembler renders, or None
    when the producer gave nothing this cycle.
    """

    neural_state: Optional[Any] = None
    attention_guidance: Optional[Any] = None
    metacognitive_insights: Optional[Any] = None
    active_inference_context: Optional[Any] = None
    global_workspace_context: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        """The filled slots only."""
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if getattr(self, item.name) is not None
        }


@dataclass
class ModulatedContext:
    """What the brain receives after the consciousness cycle shaped the turn."""

    prompt_context: str
    theory_sections: TheorySections
    generation_params: Dict[str, float]
    behavioral_directives: List[str]
    consciousness_self_awareness: str
    consciousness_state: Optional[Any]
    workspace_winner: str
    winner_is_message: bool
    cycle_ok: bool
    failure: Optional[str] = None
    absent_signals: Dict[str, str] = field(default_factory=dict)
    pad_sources_measured: Tuple[str, ...] = ()
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unavailable(cls, reason: str, temperature: float) -> "ModulatedContext":
        """The context for a turn whose cycle failed: nothing narrated or
        directed, the baseline temperature, and the failure named."""
        return cls(
            prompt_context="",
            theory_sections=TheorySections(),
            generation_params={"temperature": float(temperature)},
            behavioral_directives=[],
            consciousness_self_awareness="",
            consciousness_state=None,
            workspace_winner="",
            winner_is_message=True,
            cycle_ok=False,
            failure=str(reason),
        )

    def name_absent(self, signal: str, reason: str) -> None:
        """Record why a signal is absent; the reason must be in ``ABSENT_REASONS``."""
        if reason not in ABSENT_REASONS:
            raise ValueError(
                f"absent reason {reason!r} is not in ABSENT_REASONS {ABSENT_REASONS}"
            )
        self.absent_signals[signal] = reason


@dataclass
class TurnResult:
    """The outcome of one turn as the caller sees it.

    ``safety_gate`` is one of ``SAFETY_GATE_STATES`` for a turn the pipeline
    ran (None when the result was built elsewhere and nobody said);
    ``safety_decision`` is the safety monitor's decision string for a blocked
    turn (``emergency_shutdown`` when the conversations were suspended) and
    None otherwise.
    """

    success: bool
    response: Optional[str]
    thinking: Optional[str]
    conversation_id: Optional[str]
    message_count: int
    timestamp: datetime
    error: Optional[str] = None
    error_code: Optional[str] = None
    cycle_ok: Optional[bool] = None
    speaker_label: str = ""
    provider_id: Optional[str] = None
    blocked: bool = False
    safety_gate: Optional[str] = None
    safety_decision: Optional[str] = None

    def __post_init__(self) -> None:
        if self.safety_gate is not None and self.safety_gate not in (
            SAFETY_GATE_STATES
        ):
            raise ValueError(
                f"safety_gate {self.safety_gate!r} is not in SAFETY_GATE_STATES "
                f"{SAFETY_GATE_STATES}"
            )

    def to_payload(self) -> Dict[str, Any]:
        """A JSON-ready view: the display label only, never identity internals
        or tokens."""
        stamp = self.timestamp
        return {
            "success": bool(self.success),
            "response": self.response,
            "thinking": self.thinking,
            "conversation_id": self.conversation_id,
            "message_count": int(self.message_count),
            "timestamp": stamp.isoformat() if isinstance(stamp, datetime) else stamp,
            "error": self.error,
            "error_code": self.error_code,
            "cycle_ok": self.cycle_ok,
            "speaker_label": self.speaker_label,
            "provider_id": self.provider_id,
            "blocked": bool(self.blocked),
            "safety_gate": self.safety_gate,
            "safety_decision": self.safety_decision,
        }
