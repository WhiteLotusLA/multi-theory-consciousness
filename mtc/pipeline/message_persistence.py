"""Writing a conversation's turns to its three stores.

``MessagePersistence`` writes each turn three times, each write in its own
try so one failing store never stops the others:

- the conversation repository (``ConversationRepository``: the message row
  with its stored role and ``to_speaker`` addressee),
- the conversation turn store (``ConversationTurnStore``: the turn with its
  emotional context, topics and the turn's provenance snapshot in
  ``metadata.consciousness_snapshot``),
- working memory (``WorkingMemory``: the short buffer keyed by the
  conversation id).

A None store is the in-memory implementation, never a network client.
Each call returns a ``PersistResult`` naming what was written, what a
resilient wrapper buffered for replay and what failed (the error type and
message); failures are counted by target in ``status()``, the first and
every hundredth logged at WARNING with the target and error type only, and
raised instead when ``DEBUG`` is true.

``store_conversation_start`` opens the conversation with the participants
``SYSTEM_NAME`` and the given participant. The participant is the key the
history is later rehydrated by (``SpeakerTrust.session_key`` of the
speaker), so an unauthenticated name never shares an authenticated
speaker's stored history. ``store_conversation_end`` closes it with the
caller's summary or none at all (no summary text is composed here) and
clears the working-memory buffer. ``store_emotional_state`` writes one
emotional-history row: the emotion extractor's label in ``primary_emotion``
and its confidence in ``intensity`` (None when it has none) and the PAD
coordinate (pleasure as valence, arousal, dominance) read from a
``PADState``, the PAD fields recorded on a ``ConsciousnessState`` or a
mapping. The arousal is always the PAD arousal, never a count from the
attention distribution. ``build_turn_snapshot`` builds the provenance record
of one turn: what the system was told about itself (narrative, theory
sections, directives, self-awareness), the generation parameters, the
computed state with its PAD coordinate and the integration proxy labelled as
such, the measured PAD sources, the absent signals and the provider id.

Config keys: ``SYSTEM_NAME``, ``DEBUG`` and, for the in-memory working
memory, ``WORKING_MEMORY_CAPACITY``. Timestamps come from the
injected clock as aware UTC. Message text, speaker names, snapshots and
emotional context are private runtime data; nothing here logs them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from mtc.core.config import get_settings
from mtc.memory.inmemory import (
    InMemoryConversationRepository,
    InMemoryConversationTurnStore,
    InMemoryWorkingMemory,
)
from mtc.memory.resilient_wrappers import BUFFERED

logger = logging.getLogger(__name__)

TARGETS = ("repository", "turn_store", "working_memory")
INTEGRATION_PROXY_LABEL = "integration proxy"

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PersistResult:
    """What one persistence call did, by target name."""

    written: Tuple[str, ...]
    failed: Dict[str, str] = field(default_factory=dict)
    buffered: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failed


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def pad_coordinate(source: Any) -> Dict[str, Optional[float]]:
    """``valence``, ``arousal`` and ``dominance`` of a PAD source, each None
    when the source does not carry it.

    Reads a ``PADState`` (``pleasure``, ``arousal``, ``dominance``), the
    ``pad_pleasure``, ``pad_arousal`` and ``pad_dominance`` fields of a
    ``ConsciousnessState``, or a mapping with those keys (``valence`` is
    accepted for pleasure).
    """
    if source is None:
        return {"valence": None, "arousal": None, "dominance": None}
    if isinstance(source, Mapping):

        def read(*names: str) -> Optional[float]:
            for name in names:
                if name in source:
                    return _number(source[name])
            return None

        return {
            "valence": read("pleasure", "valence", "pad_pleasure"),
            "arousal": read("arousal", "pad_arousal"),
            "dominance": read("dominance", "pad_dominance"),
        }
    if hasattr(source, "pad_arousal") or hasattr(source, "pad_pleasure"):
        return {
            "valence": _number(getattr(source, "pad_pleasure", None)),
            "arousal": _number(getattr(source, "pad_arousal", None)),
            "dominance": _number(getattr(source, "pad_dominance", None)),
        }
    return {
        "valence": _number(getattr(source, "pleasure", None)),
        "arousal": _number(getattr(source, "arousal", None)),
        "dominance": _number(getattr(source, "dominance", None)),
    }


def _plain(value: Any) -> Any:
    """A JSON-ready copy: mappings and sequences recursively, numbers and
    strings as they are, anything else as its string form."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return str(value)


def _computed_state(state: Any) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    pad = pad_coordinate(state)
    return {
        "is_conscious": getattr(state, "is_conscious", None),
        "attention_focus": getattr(state, "attention_focus", None),
        "ignition_events": getattr(state, "ignition_events", None),
        "integration_level": _number(getattr(state, "integration_level", None)),
        "prediction_error": _number(getattr(state, "prediction_error", None)),
        "epistemic_depth": getattr(state, "epistemic_depth", None),
        "pad": {
            "pleasure": pad["valence"],
            "arousal": pad["arousal"],
            "dominance": pad["dominance"],
            "label": getattr(state, "pad_label", None),
        },
        "integration_proxy": {
            "label": INTEGRATION_PROXY_LABEL,
            "value": _number(getattr(state, "architectural_phi", None)),
            "warming_up": getattr(state, "phi_warming_up", None),
        },
    }


def build_turn_snapshot(
    modulated: Any, provider_id: Optional[str] = None
) -> Dict[str, Any]:
    """The provenance record of one turn from its ``ModulatedContext``."""
    sections = getattr(modulated, "theory_sections", None)
    filled = sections.as_dict() if hasattr(sections, "as_dict") else {}
    return {
        "narrative": getattr(modulated, "prompt_context", ""),
        "theory_sections": _plain(filled),
        "directives": _plain(list(getattr(modulated, "behavioral_directives", ()))),
        "self_awareness": getattr(modulated, "consciousness_self_awareness", ""),
        "generation_params": _plain(dict(getattr(modulated, "generation_params", {}))),
        "computed_state": _plain(
            _computed_state(getattr(modulated, "consciousness_state", None))
        ),
        "pad_sources_measured": list(getattr(modulated, "pad_sources_measured", ())),
        "absent_signals": dict(getattr(modulated, "absent_signals", {})),
        "workspace_winner": getattr(modulated, "workspace_winner", ""),
        "winner_is_message": getattr(modulated, "winner_is_message", None),
        "cycle_ok": getattr(modulated, "cycle_ok", None),
        "failure": getattr(modulated, "failure", None),
        "provider_id": provider_id,
    }


class MessagePersistence:
    """Three isolated writes per turn, each counted."""

    def __init__(
        self,
        repository: Any = None,
        turn_store: Any = None,
        working_memory: Any = None,
        settings: Any = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self._clock: Clock = clock or _utc_now
        self._settings = settings
        self.repository = (
            repository
            if repository is not None
            else InMemoryConversationRepository(clock=self._clock)
        )
        self.turn_store = (
            turn_store
            if turn_store is not None
            else InMemoryConversationTurnStore(clock=self._clock)
        )
        self.working_memory = (
            working_memory
            if working_memory is not None
            else InMemoryWorkingMemory(
                capacity=int(self._cfg().WORKING_MEMORY_CAPACITY), clock=self._clock
            )
        )
        self._written = {target: 0 for target in TARGETS}
        self._buffered = {target: 0 for target in TARGETS}
        self._failures: Dict[str, int] = {}
        self._emotional = {"written": 0, "failed": 0}

    def _cfg(self) -> Any:
        return self._settings if self._settings is not None else get_settings()

    def _fail(self, target: str, exc: BaseException) -> str:
        count = self._failures.get(target, 0) + 1
        self._failures[target] = count
        if count == 1 or count % 100 == 0:
            logger.warning(
                "persistence write to %s failed (%d so far): %s",
                target,
                count,
                type(exc).__name__,
            )
        if bool(self._cfg().DEBUG):
            raise exc
        return f"{type(exc).__name__}: {exc}"

    async def _run(
        self,
        target: str,
        operation: Any,
        written: List[str],
        buffered: List[str],
        failed: Dict[str, str],
        *,
        is_buffered: Callable[[Any], bool] = lambda value: value is BUFFERED,
    ) -> Any:
        try:
            value = await operation
        except Exception as exc:  # noqa: BLE001 - counted, reported, isolated
            failed[target] = self._fail(target, exc)
            return None
        if is_buffered(value):
            buffered.append(target)
            self._buffered[target] += 1
        else:
            written.append(target)
            self._written[target] += 1
        return value

    async def store_conversation_start(
        self, conversation_id: str, participant: str
    ) -> PersistResult:
        """Open the conversation in the repository and the turn store."""
        system = str(self._cfg().SYSTEM_NAME)
        participants = [system]
        if participant and participant != system:
            participants.append(str(participant))
        written: List[str] = []
        buffered: List[str] = []
        failed: Dict[str, str] = {}
        await self._run(
            "repository",
            self.repository.add_conversation(
                conversation_id, participants, self._clock()
            ),
            written,
            buffered,
            failed,
        )
        await self._run(
            "turn_store",
            self.turn_store.start_conversation_session(
                conversation_id, str(participant)
            ),
            written,
            buffered,
            failed,
        )
        return PersistResult(tuple(written), failed, tuple(buffered))

    async def store_turn(
        self,
        conversation_id: str,
        speaker: str,
        role: str,
        content: str,
        turn_number: int,
        emotional_context: Optional[Dict[str, Any]] = None,
        snapshot: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        *,
        to_speaker: Optional[str] = None,
    ) -> PersistResult:
        """Write one turn to the three stores, each write isolated."""
        moment = self._clock()
        written: List[str] = []
        buffered: List[str] = []
        failed: Dict[str, str] = {}
        await self._run(
            "repository",
            self.repository.add_message(
                conversation_id, speaker, role, content, moment, to_speaker
            ),
            written,
            buffered,
            failed,
            is_buffered=lambda value: not _is_id(value),
        )
        metadata: Dict[str, Any] = {}
        if snapshot is not None:
            metadata["consciousness_snapshot"] = snapshot
        if to_speaker is not None:
            metadata["to_speaker"] = to_speaker
        await self._run(
            "turn_store",
            self.turn_store.store_conversation_turn(
                conversation_id=conversation_id,
                speaker=speaker,
                role=role,
                content=content,
                turn_number=int(turn_number),
                emotional_context=emotional_context,
                topics=list(topics) if topics is not None else None,
                metadata=metadata or None,
            ),
            written,
            buffered,
            failed,
        )
        await self._run(
            "working_memory",
            self.working_memory.add_to_working_memory(
                conversation_id,
                {
                    "role": role,
                    "content": content,
                    "speaker": speaker,
                    "timestamp": moment.isoformat(),
                },
            ),
            written,
            buffered,
            failed,
        )
        return PersistResult(tuple(written), failed, tuple(buffered))

    async def store_conversation_end(
        self, conversation_id: str, summary: Optional[str] = None
    ) -> PersistResult:
        """Close the conversation with the caller's summary (or none) and
        clear the working-memory buffer."""
        written: List[str] = []
        buffered: List[str] = []
        failed: Dict[str, str] = {}
        ended = await self._run(
            "repository",
            self.repository.end_conversation(conversation_id, self._clock(), summary),
            written,
            buffered,
            failed,
        )
        if ended is False and "repository" in written:
            # nothing was confirmed: no such conversation, or a buffered update
            written.remove("repository")
            self._written["repository"] -= 1
            failed["repository"] = "conversation end not confirmed"
            self._failures["repository"] = self._failures.get("repository", 0) + 1
        extra = {"summary": summary} if summary is not None else {}
        await self._run(
            "turn_store",
            self.turn_store.end_conversation_session(conversation_id, **extra),
            written,
            buffered,
            failed,
        )
        await self._run(
            "working_memory",
            self.working_memory.clear_working_memory(conversation_id),
            written,
            buffered,
            failed,
        )
        return PersistResult(tuple(written), failed, tuple(buffered))

    async def store_emotional_state(
        self,
        pad: Any,
        label: Optional[str],
        confidence: Optional[float],
        trigger_context: Optional[str],
    ) -> Optional[int]:
        """Write one emotional-history row and return its id (None when the
        write failed or was buffered)."""
        coordinate = pad_coordinate(pad)
        record = {
            "timestamp": self._clock(),
            "primary_emotion": label if label else None,
            "intensity": _number(confidence) if label else None,
            "valence": coordinate["valence"],
            "arousal": coordinate["arousal"],
            "dominance": coordinate["dominance"],
            "trigger_context": trigger_context,
        }
        try:
            row_id = await self.repository.add_emotional_state(record)
        except Exception as exc:  # noqa: BLE001 - counted, reported
            self._emotional["failed"] += 1
            self._fail("emotional_history", exc)
            return None
        if not _is_id(row_id):
            return None
        self._emotional["written"] += 1
        return int(row_id)

    def status(self) -> Dict[str, Any]:
        """Counts by target; no text, names or snapshots."""
        return {
            "written": dict(self._written),
            "buffered": dict(self._buffered),
            "failures": dict(self._failures),
            "emotional_history": dict(self._emotional),
        }


def _is_id(value: Any) -> bool:
    return isinstance(value, (int, str)) and not isinstance(value, bool)


__all__ = (
    "INTEGRATION_PROXY_LABEL",
    "TARGETS",
    "MessagePersistence",
    "PersistResult",
    "build_turn_snapshot",
    "pad_coordinate",
)
