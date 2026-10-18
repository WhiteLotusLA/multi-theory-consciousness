"""What a turn gathers before the cycle and the window the brain receives.

``gather_context`` runs the turn's sources at the same time, each in its own
task, and isolates their failures: the core memory blocks (``identity``,
``persona`` and ``human`` through the memory manager), memory recall
(``mtc.pipeline.memory_retrieval.retrieve_relevant_memories`` with the recalled
memories that only echo the history window removed by
``suppress_rag_echoes``), the surfaced continuous thoughts and the thought
gists (``mtc.pipeline.consciousness_helpers``) and the conversation-coherence
package of ``mtc.consciousness.conversation_coherence``. A source that fails
contributes its empty value and is named in ``GatheredContext.errors`` with the
error type and message; nothing is invented in its place, and the recall
method is reported as the manager labelled it (``vector``, ``text`` or
``none``).

``build_brain_window`` turns a session's history into the ``BrainWindow`` the
brain's ``think_with_context`` receives: a copy of the entries with the current
message split off as ``current_prompt``, every user turn labelled by the trust
source (``SpeakerTrust.label_for`` of the verified identity: the bracketed name
for a speaker no token backs, nothing for the authenticated creator and the
anonymous default), tool turns (role ``tool``, speaker the tool's name) turned
into user turns labelled ``tool <name>``, repeated assistant sentences removed
(``deduplicate_context``), the night clock note inserted where a night fell
when ``NIGHT_SEAM_MARKER_ENABLED`` is true (``apply_night_seam``), the
``window_notes`` that describe what was changed, and ``last_turn_at``, the
newest timestamp before the current message (the clock's gap lines read the
time before this turn): with ``now`` given only entries stamped strictly
earlier than ``now`` count, so a tool turn stamped with this turn's time is
not the last exchange, and the session's ``last_turn_at`` is the fallback. The
window never trims tokens: the brain runs the one budget pass. The session's
own history is never modified here.

Config keys: ``PROMPT_MAX_THOUGHTS`` and ``PROMPT_MAX_GISTS`` (the caps the
thought and gist reads take), ``RAG_TOP_K``, ``RAG_MIN_SCORE`` and
``SYSTEM_NAME`` through the retrieval helpers, and the night seam keys through
``apply_night_seam``, all read at call time. Messages, memories, thoughts and
the speaker's name are private conversation data; nothing here logs them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mtc.core.config import Settings, get_settings
from mtc.memory.core_blocks import CORE_BLOCK_NAMES
from mtc.pipeline.consciousness_helpers import (
    get_recent_ctm_thoughts,
    get_thought_gists,
)
from mtc.pipeline.context_budget import apply_night_seam, infer_last_turn_at
from mtc.pipeline.context_dedup import deduplicate_context
from mtc.pipeline.memory_retrieval import (
    retrieve_relevant_memories,
    suppress_rag_echoes,
)
from mtc.pipeline.session_state import Session
from mtc.pipeline.speaker_trust import SpeakerIdentity, SpeakerTrust

logger = logging.getLogger(__name__)

TOOL_ROLE = "tool"
SOURCES: Tuple[str, ...] = ("core_memory", "recall", "thoughts", "gists", "coherence")


@dataclass
class GatheredContext:
    """Everything the sources returned for one turn, failures named."""

    core_memory: Dict[str, str] = field(default_factory=dict)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    recall_method: str = "none"
    ctm_thoughts: List[Dict[str, Any]] = field(default_factory=list)
    gists: List[Any] = field(default_factory=list)
    coherence: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class BrainWindow:
    """The history the brain sees, the current message and the window notes."""

    history: List[Dict[str, Any]]
    current_prompt: str
    window_notes: List[str] = field(default_factory=list)
    last_turn_at: Optional[datetime] = None


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _cfg(settings: Optional[Settings]) -> Settings:
    return settings if settings is not None else get_settings()


async def _core_blocks(manager: Any) -> Dict[str, str]:
    reads = await asyncio.gather(
        *(manager.get_core_memory(name) for name in CORE_BLOCK_NAMES)
    )
    return {
        name: str(text)
        for name, text in zip(CORE_BLOCK_NAMES, reads)
        if text is not None and str(text).strip()
    }


async def _recall(
    manager: Any, message: str, history: Sequence[Dict[str, Any]], cfg: Settings
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, str]]:
    result = await retrieve_relevant_memories(manager, message, settings=cfg)
    memories = suppress_rag_echoes(result.memories, history)
    return memories, result.concepts, result.method, dict(result.errors)


async def _coherence(
    module: Any, history: Sequence[Dict[str, Any]], message: str
) -> Dict[str, Any]:
    spoken = [
        {"role": str(e.get("role", "")), "content": str(e.get("content") or "")}
        for e in history
        if not e.get("is_clock_note")
    ]
    package = module.get_conversation_context_for_brain(spoken, message)
    return dict(package or {})


async def gather_context(
    message: str,
    session: Session,
    *,
    memory_manager: Any,
    thought_store: Any = None,
    gist_store: Any = None,
    coherence: Any = None,
    settings: Optional[Settings] = None,
) -> GatheredContext:
    """Run every source at once; a failing source is named, never replaced."""
    cfg = _cfg(settings)
    history = list(session.history)
    gathered = GatheredContext()
    tasks: Dict[str, Any] = {}
    if memory_manager is None:
        gathered.errors["memory_manager"] = "memory manager absent"
    else:
        tasks["core_memory"] = _core_blocks(memory_manager)
        tasks["recall"] = _recall(memory_manager, message, history, cfg)
    if thought_store is not None:
        tasks["thoughts"] = get_recent_ctm_thoughts(
            thought_store, int(cfg.PROMPT_MAX_THOUGHTS)
        )
    if gist_store is not None:
        tasks["gists"] = get_thought_gists(gist_store, int(cfg.PROMPT_MAX_GISTS))
    if coherence is not None:
        tasks["coherence"] = _coherence(coherence, history, message)
    names = list(tasks)
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for name, outcome in zip(names, results):
        if isinstance(outcome, BaseException):
            if isinstance(outcome, (asyncio.CancelledError, KeyboardInterrupt)):
                raise outcome
            gathered.errors[name] = _describe(outcome)
            continue
        if name == "core_memory":
            gathered.core_memory = outcome
        elif name == "recall":
            memories, concepts, method, errors = outcome
            gathered.memories = memories
            gathered.concepts = concepts
            gathered.recall_method = method
            for part, error in errors.items():
                gathered.errors[f"recall_{part}"] = error
        elif name == "thoughts":
            gathered.ctm_thoughts = list(outcome)
        elif name == "gists":
            gathered.gists = list(outcome)
        elif name == "coherence":
            gathered.coherence = outcome
    if gathered.errors:
        logger.info(
            "context gathered with %d failing source(s): %s",
            len(gathered.errors),
            ", ".join(sorted(gathered.errors)),
        )
    return gathered


def _is_speech(entry: Dict[str, Any]) -> bool:
    """A user's own words: not a clock note and not a converted tool turn (a
    tool result appended after the message must never become the prompt)."""
    return (
        entry.get("role") == "user"
        and not entry.get("is_clock_note")
        and not entry.get("is_tool_turn")
    )


def _split_current(
    entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    for index in range(len(entries) - 1, -1, -1):
        if _is_speech(entries[index]):
            current = entries[index]
            return entries[:index] + entries[index + 1 :], str(
                current.get("content") or ""
            )
    return entries, ""


def _last_turn_before(
    history: Sequence[Dict[str, Any]], now: Optional[datetime]
) -> Optional[datetime]:
    """The newest readable timestamp of ``history``; with ``now`` given, the
    newest one strictly earlier than ``now`` (a naive ``now`` counts as
    UTC)."""
    if now is None:
        return infer_last_turn_at(history)
    limit = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    earlier: List[datetime] = []
    for entry in history:
        moment = infer_last_turn_at([entry])
        if moment is not None and moment < limit:
            earlier.append(moment)
    return max(earlier) if earlier else None


def build_brain_window(
    session: Session,
    identity: SpeakerIdentity,
    speaker_trust: SpeakerTrust,
    *,
    settings: Optional[Settings] = None,
    now: Optional[datetime] = None,
) -> BrainWindow:
    """The window for this turn; see the module docstring for each step."""
    if not isinstance(identity, SpeakerIdentity):
        raise TypeError("build_brain_window takes the verified SpeakerIdentity")
    cfg = _cfg(settings)
    notes: List[str] = []
    entries = [dict(entry) for entry in session.history]

    tool_turns = 0
    for entry in entries:
        if entry.get("role") == TOOL_ROLE:
            name = str(entry.get("speaker") or "unnamed").strip() or "unnamed"
            entry["role"] = "user"
            entry["speaker"] = f"tool {name}"
            entry["is_tool_turn"] = True
            tool_turns += 1
    if tool_turns:
        notes.append(f"{tool_turns} tool result(s) appear as labelled user turns.")

    history, current_prompt = _split_current(entries)

    label = speaker_trust.label_for(identity)
    for entry in history:
        if _is_speech(entry):
            if label:
                entry["speaker"] = label
            else:
                entry.pop("speaker", None)

    before = sum(len(str(e.get("content") or "")) for e in history)
    history = deduplicate_context(history)
    after = sum(len(str(e.get("content") or "")) for e in history)
    if after < before:
        notes.append(
            "Earlier assistant turns had repeated sentences removed from this "
            "window; the stored conversation is unchanged."
        )

    last_turn_at = _last_turn_before(history, now) or session.last_turn_at

    seamed = apply_night_seam(history, now=now, settings=cfg)
    if len(seamed) > len(history):
        notes.append("A clock note marks where a night passed in this window.")
    history = seamed

    return BrainWindow(
        history=history,
        current_prompt=current_prompt,
        window_notes=notes,
        last_turn_at=last_turn_at,
    )


__all__ = (
    "SOURCES",
    "TOOL_ROLE",
    "BrainWindow",
    "GatheredContext",
    "build_brain_window",
    "gather_context",
)
