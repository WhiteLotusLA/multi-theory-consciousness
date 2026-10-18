"""The hooks that run after the system has answered a turn.

``PostTurnIntegrator.run`` takes a ``TurnRecord`` (the verified speaker
identity, the message, the spoken reply, the cycle's state and modulated
context) and runs one hook per step, in ``STEP_ORDER``:

1. ``verification``: the verifier's coherence and emotional-authenticity
   tests on the state the assembler builds from the integration's real
   substrate blocks and the PAD coordinate, every turn (``verify_every_turn``).
   A suite with no measured test is reported ``insufficient_data`` and no
   score is recorded for it.
2. ``temporal_verification``: the turn's assembled state is appended to the
   integration's temporal samples every turn; every ``temporal_every_n``
   turns the temporal-consistency test runs over them.
3. ``neural_experience``: one experience through the neural orchestrator.
   Its emotions are the ``emotions`` of this cycle's winner
   (``cycle_winner``: the state's primary content, only when the cycle
   ignited; the workspace's standing content is never read) when the winner
   carries them; otherwise they come from the PAD coordinate by
   ``PAD_EMOTION_MAPPING`` (joy is the positive part of pleasure, sadness the
   negative part). The ``trust`` emotion always comes from
   ``SpeakerTrust.emotion_trust`` of the verified identity, never from a name
   or a candidate. With neither winner emotions nor a PAD coordinate the step
   is skipped: no affect is invented.
4. ``emotion``: the emotion extractor's label and confidence for the state,
   run through the emotion dynamics with the PAD coordinate when both exist,
   and one emotional-history row (the label, its confidence and the PAD
   coordinate) through the persistence layer.
5. ``learning``: the learning system records the turn with the state's
   metrics and, when this cycle's PAD coordinate is complete, the turn's
   affect as ``emotion_after`` (the emotion step's label, the pleasure as
   valence and the arousal); its outcomes classify the shift of that affect
   between consecutive turns of a conversation.
6. ``theory_of_mind``: the speaker's model is updated inside
   ``SpeakerTrust.use_identity``, so its trust is the verified identity's. A
   speaker who is not authenticated is modelled under the bracketed label, so
   a claimed name never updates an authenticated speaker's model.
7. ``episodic_memory``: one episodic memory ``<speaker> said: <message>``
   and ``<SYSTEM_NAME> responded: <spoken>`` at ``EPISODIC_IMPORTANCE``.
8. ``concepts``: concepts of the message and of the spoken reply through the
   graph seam (reasoning is never extracted; the speaker's name, bare or
   bracketed, and the system's name never become concepts).
9. ``homeostasis``: the drives see a ``conversation`` activity, or
   ``exploring`` when the message asks a question.
10. ``safety_log``: the safety integration's post-message log of the reply
    with the PAD coordinate.
11. ``affect_shadow``: the affect shadow tap (``affect_shadow``), shadow only.
12. ``identity_marker``: every ``identity_marker_every_n`` turns a marker
    with the values this turn supplies only (a conversation anchor, the PAD
    model's coordinate for this cycle as the emotional fingerprint, the
    provider id): no trait, value or growth stage is invented.
13. ``baseline``: the orchestrator's baseline assessment is refreshed when
    it is due, with the orchestrator's due reason as the snapshot trigger (an
    assessment the orchestrator could not run reads ``failed: assessment``;
    the orchestrator counts it).

The PAD coordinate every hook uses is this cycle's: the PAD recorded on the
cycle's state, else a PAD model reading taken during this cycle (its
timestamp not older than the state's), else None on every axis; an earlier
reading is never used.

Every hook is isolated: a failure is counted by step name
(``status()['failures']``), the first and every hundredth logged at WARNING
with the step and error type only, the later hooks still run and nothing
reaches the caller, unless ``DEBUG`` is true, which raises instead. Absent
components report ``absent``; every step reports its outcome in
``PostTurnReport.steps``. The reply passes through ``strip_reasoning_blocks``
before anything is stored or extracted.

Config keys: ``SYSTEM_NAME``, ``EPISODIC_IMPORTANCE`` and ``DEBUG``, read at
call time; trust comes from ``SpeakerTrust`` only. The message, the reply,
speaker names and the derived memories are private runtime data; log lines
carry step names and counts only.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from mtc.brain.protocol import strip_reasoning_blocks
from mtc.core.config import get_settings
from mtc.pipeline.context_types import cycle_winner
from mtc.pipeline.memory_retrieval import extract_and_store_concepts
from mtc.pipeline.message_persistence import pad_coordinate
from mtc.pipeline.speaker_trust import SpeakerIdentity, SpeakerTrust

logger = logging.getLogger(__name__)

STEP_ORDER = (
    "verification",
    "temporal_verification",
    "neural_experience",
    "emotion",
    "learning",
    "theory_of_mind",
    "episodic_memory",
    "concepts",
    "homeostasis",
    "safety_log",
    "affect_shadow",
    "identity_marker",
    "baseline",
)

# How the PAD coordinate becomes the neural experience's emotions when the
# winner carries none: joy is the positive part of pleasure and sadness the
# negative part. Trust is added from the speaker trust source.
PAD_EMOTION_MAPPING = {
    "joy": "max(0, pleasure)",
    "sadness": "max(0, -pleasure)",
}

EXPLORING_ACTIVITY = "exploring"
CONVERSATION_ACTIVITY = "conversation"
CONTEXT_CHARS = 200


@dataclass(frozen=True)
class PostTurnConfig:
    """Cadences and switches of the post-turn hooks."""

    verify_every_turn: bool = True
    temporal_every_n: int = 10
    identity_marker_every_n: int = 20
    affect_shadow: bool = True


@dataclass
class TurnRecord:
    """One answered turn (private content).

    ``history`` is the window the brain saw (for the coherence test's
    continuity) and ``provider_id`` the provider that answered.
    """

    identity: SpeakerIdentity
    message: str
    spoken: str
    thinking: Optional[str]
    state: Any
    modulated: Any
    conversation_id: str
    turn_number: int
    topics: Sequence[str] = ()
    history: Sequence[Mapping[str, Any]] = ()
    provider_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SpeakerIdentity):
            raise TypeError(
                "TurnRecord.identity must be a SpeakerIdentity from SpeakerTrust.verify"
            )


@dataclass
class PostTurnReport:
    """Each step's outcome (``ok``, ``absent``, ``not_due``, ``skipped_*``,
    ``insufficient_data`` or ``failed: <error type>``) and its details."""

    steps: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class _Skip(Exception):
    """A step that has nothing honest to do this turn."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


def _pad_state(pad_model: Any, state: Any) -> Dict[str, Optional[float]]:
    """This cycle's PAD coordinate as the PAD model computed it from the
    inputs listed in ``pad_sources_measured``, never an earlier one.

    The stored copies of the coordinate (the emotional-history row, the
    episodic emotional context, the identity marker) do not carry that list;
    the turn's provenance snapshot does. The PAD recorded on the cycle's
    state comes first. When the state carries none, the PAD model's latest
    reading counts only when it was taken during this cycle (its timestamp is
    not older than the state's); otherwise every axis is None.
    """
    coordinate = pad_coordinate(state)
    if any(value is not None for value in coordinate.values()):
        return coordinate
    reader = getattr(pad_model, "get_latest", None) if pad_model is not None else None
    latest = reader() if callable(reader) else None
    taken = _number(getattr(latest, "timestamp", None))
    formed = _number(getattr(state, "timestamp", None))
    if latest is not None and taken is not None and formed is not None:
        if taken >= formed:
            return pad_coordinate(latest)
    return {"valence": None, "arousal": None, "dominance": None}


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pad_complete(coordinate: Mapping[str, Optional[float]]) -> bool:
    return all(
        coordinate.get(axis) is not None for axis in ("valence", "arousal", "dominance")
    )


class PostTurnIntegrator:
    """Runs the post-turn hooks, each isolated and counted."""

    def __init__(
        self,
        *,
        speaker_trust: SpeakerTrust,
        neural_orchestrator: Any = None,
        verifier: Any = None,
        state_assembler: Any = None,
        integration: Any = None,
        learning: Any = None,
        theory_of_mind: Any = None,
        emotion_extractor: Any = None,
        emotion_dynamics: Any = None,
        persistence: Any = None,
        memory_manager: Any = None,
        graph: Any = None,
        workspace: Any = None,
        pad_model: Any = None,
        safety: Any = None,
        identity_continuity: Any = None,
        orchestrator: Any = None,
        config: Optional[PostTurnConfig] = None,
        settings: Any = None,
        affect_shadow_tap: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not isinstance(speaker_trust, SpeakerTrust):
            raise TypeError("PostTurnIntegrator needs the SpeakerTrust source")
        self.speaker_trust = speaker_trust
        self.neural_orchestrator = neural_orchestrator
        self.verifier = verifier
        self.integration = integration
        if state_assembler is None and verifier is not None:
            from mtc.consciousness.verification_state_assembler import (
                ConsciousnessStateAssembler,
            )

            state_assembler = ConsciousnessStateAssembler(integration)
        self.state_assembler = state_assembler
        self.learning = learning
        self.theory_of_mind = theory_of_mind
        self.emotion_extractor = emotion_extractor
        self.emotion_dynamics = emotion_dynamics
        self.persistence = persistence
        self.memory_manager = memory_manager
        self.graph = graph
        self.workspace = workspace
        self.pad_model = pad_model
        self.safety = safety
        self.identity_continuity = identity_continuity
        self.orchestrator = orchestrator
        self.config = config or PostTurnConfig()
        self._settings = settings
        self._shadow_tap = affect_shadow_tap
        self._failures: Dict[str, int] = {}
        self._runs = 0
        self._last_steps: Dict[str, str] = {}

    def _cfg(self) -> Any:
        return self._settings if self._settings is not None else get_settings()

    def _label(self, identity: SpeakerIdentity) -> str:
        """The speaker as the stored lines and models name them: the name
        when authenticated or anonymous, the bracketed label otherwise."""
        if identity.authenticated or identity.is_default:
            return identity.name
        return self.speaker_trust.label_for(identity) or f"[{identity.name}]"

    def _count(self, step: str, exc: BaseException) -> None:
        count = self._failures.get(step, 0) + 1
        self._failures[step] = count
        if count == 1 or count % 100 == 0:
            logger.warning(
                "post-turn step %s failed (%d so far): %s",
                step,
                count,
                type(exc).__name__,
            )

    async def run(self, record: TurnRecord) -> PostTurnReport:
        """Run every hook in order; never raises unless ``DEBUG`` is true."""
        if not isinstance(record, TurnRecord):
            raise TypeError("run takes a TurnRecord")
        self._runs += 1
        report = PostTurnReport()
        spoken = strip_reasoning_blocks(record.spoken or "")
        pad = _pad_state(self.pad_model, record.state)
        context = {"spoken": spoken, "pad": pad, "verification_state": None}
        for step in STEP_ORDER:
            hook = getattr(self, f"_step_{step}")
            try:
                outcome = hook(record, context, report)
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                report.steps[step] = outcome or "ok"
            except _Skip as skip:
                report.steps[step] = skip.status
            except Exception as exc:  # noqa: BLE001 - counted, isolated
                self._count(step, exc)
                report.steps[step] = f"failed: {type(exc).__name__}"
                if bool(self._cfg().DEBUG):
                    raise
        self._last_steps = dict(report.steps)
        return report

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _assembled(self, record: TurnRecord, context: Dict[str, Any]) -> Any:
        if context["verification_state"] is None:
            pad = context["pad"]
            pad_state = (
                {
                    "pleasure": pad["valence"],
                    "arousal": pad["arousal"],
                    "dominance": pad["dominance"],
                }
                if any(value is not None for value in pad.values())
                else None
            )
            context["verification_state"] = self.state_assembler.assemble(
                conversation_history=[dict(entry) for entry in record.history],
                pad_state=pad_state,
            )
        return context["verification_state"]

    def _step_verification(self, record, context, report) -> str:
        if self.verifier is None:
            return "absent"
        if not self.config.verify_every_turn:
            return "disabled"
        from mtc.consciousness.verification_framework import TestType

        state = self._assembled(record, context)
        suite = self.verifier.run_verification_suite(
            state, test_types=[TestType.COHERENCE, TestType.EMOTIONAL_AUTHENTICITY]
        )
        report.details["verification"] = {
            "status": suite.status,
            "overall_score": suite.overall_score,
            "measured": list(suite.measured),
            "insufficient": list(suite.insufficient),
        }
        return "insufficient_data" if suite.overall_score is None else "ok"

    def _step_temporal_verification(self, record, context, report) -> str:
        if self.verifier is None or self.integration is None:
            return "absent"
        from mtc.consciousness.verification_framework import TestType

        state = self._assembled(record, context)
        appender = getattr(self.integration, "append_temporal_sample", None)
        if callable(appender):
            appender(state)
        every = max(1, int(self.config.temporal_every_n))
        if record.turn_number <= 0 or record.turn_number % every != 0:
            return "not_due"
        temporal = self.integration.get_temporal_data()
        suite = self.verifier.run_verification_suite(
            {**state, "temporal_data": temporal},
            test_types=[TestType.TEMPORAL_CONSISTENCY],
        )
        report.details["temporal_verification"] = {
            "status": suite.status,
            "overall_score": suite.overall_score,
            "samples": len(temporal.get("states", [])),
        }
        return "insufficient_data" if suite.overall_score is None else "ok"

    @staticmethod
    def _winner_emotions(state: Any) -> Optional[Dict[str, float]]:
        """The emotions of this cycle's winner (``cycle_winner``): None when
        nothing ignited this cycle, whatever the workspace still holds."""
        candidate = cycle_winner(state)
        metadata = getattr(candidate, "metadata", None) or {}
        emotions = metadata.get("emotions") if isinstance(metadata, Mapping) else None
        if not isinstance(emotions, Mapping):
            return None
        cleaned = {}
        for name, value in emotions.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value == value:
                    cleaned[str(name)] = float(value)
        return cleaned or None

    def _step_neural_experience(self, record, context, report) -> str:
        if self.neural_orchestrator is None:
            return "absent"
        pad = context["pad"]
        emotions = self._winner_emotions(record.state)
        source = "winner"
        if emotions is None:
            if pad["valence"] is None:
                raise _Skip("skipped_no_affect")
            pleasure = float(pad["valence"])
            emotions = {"joy": max(0.0, pleasure), "sadness": max(0.0, -pleasure)}
            source = "pad"
        emotions["trust"] = float(self.speaker_trust.emotion_trust(record.identity))
        label = self._label(record.identity)
        self.neural_orchestrator.process_experience(
            experience_type=CONVERSATION_ACTIVITY,
            emotions=emotions,
            context=f"{label}: {record.message[:CONTEXT_CHARS]}",
            valence=pad["valence"],
            arousal=pad["arousal"],
        )
        report.details["neural_experience"] = {
            "emotions_from": source,
            "emotions": dict(emotions),
        }
        return "ok"

    async def _step_emotion(self, record, context, report) -> str:
        if self.emotion_extractor is None and self.emotion_dynamics is None:
            return "absent"
        label: Optional[str] = None
        secondary: Optional[str] = None
        confidence: Optional[float] = None
        if self.emotion_extractor is not None:
            extracted = await self.emotion_extractor.extract(
                record.state, session_id=record.conversation_id or "default"
            )
            label = extracted.label
            secondary = extracted.secondary
            confidence = float(extracted.confidence) if label else None
        pad = context["pad"]
        if self.emotion_dynamics is not None and _pad_complete(pad):
            from mtc.consciousness.pad_affect import PADState

            dynamic = self.emotion_dynamics.update(
                PADState(
                    pleasure=float(pad["valence"]),
                    arousal=float(pad["arousal"]),
                    dominance=float(pad["dominance"]),
                ),
                dominant=label,
                secondary=secondary,
                confidence=confidence,
            )
            label = getattr(dynamic, "dominant", None) or label
        report.details["emotion"] = {"label": label, "confidence": confidence}
        if self.persistence is not None:
            row_id = await self.persistence.store_emotional_state(
                {
                    "valence": pad["valence"],
                    "arousal": pad["arousal"],
                    "dominance": pad["dominance"],
                },
                label,
                confidence,
                record.message[:CONTEXT_CHARS],
            )
            report.details["emotion"]["row_id"] = row_id
        return "ok"

    async def _step_learning(self, record, context, report) -> str:
        if self.learning is None:
            return "absent"
        state = record.state
        urgency = getattr(state, "homeostatic_urgency", None)
        # The turn's affect for the learning system's outcome: this cycle's
        # PAD (the state's, or the PAD model's reading taken this cycle) with
        # the emotion step's label, only when the coordinate is complete
        pad = context["pad"]
        emotion_after: Optional[Dict[str, Any]] = None
        if _pad_complete(pad):
            emotion = report.details.get("emotion")
            emotion_after = {
                "label": emotion.get("label") if isinstance(emotion, Mapping) else None,
                "valence": pad["valence"],
                "arousal": pad["arousal"],
            }
        await self.learning.record_turn(
            speaker=self._label(record.identity),
            message=record.message,
            response=context["spoken"],
            emotion_after=emotion_after,
            state=state,
            conversation_depth=max(1, int(record.turn_number)),
            conversation_id=record.conversation_id,
            topics_discussed=list(record.topics) or None,
            homeostatic_drive=urgency[0] if urgency else None,
            attention_focus=getattr(state, "attention_focus", None),
        )
        return "ok"

    async def _step_theory_of_mind(self, record, context, report) -> str:
        if self.theory_of_mind is None:
            return "absent"
        name = self._label(record.identity)
        with self.speaker_trust.use_identity(record.identity):
            model = self.theory_of_mind.update_from_turn(
                name,
                record.message,
                emotion=None,
                topics=list(record.topics) or None,
                conversation_depth=max(1, int(record.turn_number)),
            )
            persist = getattr(self.theory_of_mind, "persist_model", None)
            if persist is not None:
                await persist(name)
        report.details["theory_of_mind"] = {
            "trust": getattr(getattr(model, "relationship", None), "trust", None)
        }
        return "ok"

    async def _step_episodic_memory(self, record, context, report) -> str:
        if self.memory_manager is None:
            return "absent"
        cfg = self._cfg()
        content = (
            f"{self._label(record.identity)} said: {record.message}\n"
            f"{cfg.SYSTEM_NAME} responded: {context['spoken']}"
        )
        pad = context["pad"]
        emotional = {
            key: value
            for key, value in (
                ("pleasure", pad["valence"]),
                ("arousal", pad["arousal"]),
                ("dominance", pad["dominance"]),
            )
            if value is not None
        }
        result = await self.memory_manager.store_memory(
            content,
            memory_type="episodic",
            importance=float(cfg.EPISODIC_IMPORTANCE),
            emotional_context=emotional or None,
        )
        errors = dict(getattr(result, "errors", {}) or {})
        report.details["episodic_memory"] = {
            "postgres_id": getattr(result, "postgres_id", None),
            "buffered": getattr(result, "buffered", None),
            "errors": sorted(errors),
        }
        if "episodic" in errors:
            raise RuntimeError("the episodic store refused the memory")
        return "ok"

    async def _step_concepts(self, record, context, report) -> str:
        if self.graph is None:
            return "absent"
        label = self._label(record.identity)
        system = str(self._cfg().SYSTEM_NAME)
        # The speaker's bare name never becomes a concept, from either side
        name = [record.identity.name]
        stored = await extract_and_store_concepts(
            self.graph, record.message, record.conversation_id, label, exclude=name
        )
        stored += await extract_and_store_concepts(
            self.graph,
            context["spoken"],
            record.conversation_id,
            system,
            exclude=name,
        )
        report.details["concepts"] = {"stored": stored}
        return "ok"

    async def _step_homeostasis(self, record, context, report) -> str:
        inference = getattr(self.workspace, "active_inference", None)
        if inference is None:
            return "absent"
        activity = (
            EXPLORING_ACTIVITY
            if "?" in (record.message or "")
            else CONVERSATION_ACTIVITY
        )
        await inference.update_homeostasis({"type": activity})
        report.details["homeostasis"] = {"activity": activity}
        return "ok"

    async def _step_safety_log(self, record, context, report) -> str:
        if self.safety is None:
            return "absent"
        pad = context["pad"]
        emotional = (
            {
                "pleasure": pad["valence"],
                "arousal": pad["arousal"],
                "dominance": pad["dominance"],
            }
            if all(value is not None for value in pad.values())
            else None
        )
        await self.safety.post_message_log(context["spoken"], emotional_state=emotional)
        return "ok"

    def _step_affect_shadow(self, record, context, report) -> str:
        if not self.config.affect_shadow:
            return "disabled"
        tap = self._shadow_tap
        if tap is None:
            from mtc.consciousness.affect_shadow import log_turn_affect_shadow

            tap = log_turn_affect_shadow
        pad = context["pad"]
        pad_value = (
            {
                "pleasure": pad["valence"],
                "arousal": pad["arousal"],
                "dominance": pad["dominance"],
            }
            if any(value is not None for value in pad.values())
            else None
        )
        tap(context["spoken"], record.message, pad_value)
        return "ok"

    def _step_identity_marker(self, record, context, report) -> str:
        if self.identity_continuity is None:
            return "absent"
        every = max(1, int(self.config.identity_marker_every_n))
        if record.turn_number <= 0 or record.turn_number % every != 0:
            return "not_due"
        pad = context["pad"]
        fingerprint = {
            key: value
            for key, value in (
                ("pleasure", pad["valence"]),
                ("arousal", pad["arousal"]),
                ("dominance", pad["dominance"]),
            )
            if value is not None
        }
        anchors: List[Dict[str, Any]] = [
            {
                "type": "conversation",
                "turn": int(record.turn_number),
                "conversation_id": record.conversation_id,
            }
        ]
        marker = self.identity_continuity.create_marker(
            core_values=None,
            personality_traits=None,
            memory_anchors=anchors,
            emotional_fingerprint=fingerprint or None,
            relationship_bonds=None,
            growth_stage=None,
            model_context=(
                {"provider_id": str(record.provider_id)} if record.provider_id else None
            ),
        )
        self.identity_continuity.store_identity_marker(marker)
        report.details["identity_marker"] = {"marker_id": marker.marker_id}
        return "ok"

    async def _step_baseline(self, record, context, report) -> str:
        if self.orchestrator is None:
            return "absent"
        due = getattr(self.orchestrator, "baseline_due", False)
        if callable(due):
            due = due()
        if not due:
            return "not_due"
        reason = getattr(self.orchestrator, "baseline_due_reason", None)
        trigger = str(reason) if reason else "heartbeat"
        result = await self.orchestrator.refresh_baseline(trigger=trigger)
        report.details["baseline"] = {
            "trigger": trigger,
            "refreshed": result is not None,
            "overall_score": getattr(result, "overall_score", None),
        }
        return "ok" if result is not None else "failed: assessment"

    def status(self) -> Dict[str, Any]:
        """Runs, failures by step and the last run's outcomes (no content)."""
        return {
            "runs": self._runs,
            "failures": dict(self._failures),
            "last_steps": dict(self._last_steps),
            "config": {
                "verify_every_turn": self.config.verify_every_turn,
                "temporal_every_n": self.config.temporal_every_n,
                "identity_marker_every_n": self.config.identity_marker_every_n,
                "affect_shadow": self.config.affect_shadow,
            },
        }


__all__ = (
    "PAD_EMOTION_MAPPING",
    "STEP_ORDER",
    "PostTurnConfig",
    "PostTurnIntegrator",
    "PostTurnReport",
    "TurnRecord",
)
