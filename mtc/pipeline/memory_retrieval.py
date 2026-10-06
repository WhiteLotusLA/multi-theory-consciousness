"""Memory recall, history rehydration and concept capture for a turn.

- ``retrieve_relevant_memories`` asks the memory manager's ``semantic_recall``
  for up to ``RAG_TOP_K`` memories at ``RAG_MIN_SCORE`` or above and puts the
  recent important memories (``get_recent_important_memories``: the last
  ``RECENCY_HOURS`` hours at importance ``RECENCY_MIN_IMPORTANCE`` or more,
  ``RECENCY_LIMIT`` of them) first, each memory once. The result names the
  recall method the manager used (``vector``, ``text`` or ``none``) and the
  error of every part that failed; nothing is invented when recall fails.
- ``suppress_rag_echoes`` drops a recalled memory whose words are already in
  a line of the history window (the memory only echoes the conversation).
- ``rehydrate_recent_history`` refills a new session from the conversation
  repository: the participant's messages of the last ``hours`` hours, oldest
  first, with each row's stored ``role`` (never a role guessed from a name)
  and only messages addressed to the system (``to_speaker`` empty or
  ``SYSTEM_NAME``). The participant argument must be the key the session is
  stored under (``SpeakerTrust.session_key``), so an unauthenticated name
  never reads an authenticated speaker's history.
- ``latest_turn_at`` returns the newest message time addressed to the system
  (aware UTC) or None.
- ``extract_and_store_concepts`` picks up to ``max_concepts`` content words of
  a text and records each through the graph seam: ``merge_concept`` with a
  neutral description that names no speaker, then
  ``link_concept_to_conversation``. The speaker and every excluded name never
  become concepts, word by word, so a bracketed label excludes the bare name
  too. This module holds no query language; the graph backend owns it.
  ``CONCEPT_STOP_WORDS`` is a generic English list.

Store failures are not hidden: the repository helpers propagate
``StoreUnavailable``, the recall reports each failing part, and the concept
helper counts nothing it did not store. Config keys: ``RAG_TOP_K``,
``RAG_MIN_SCORE`` and ``SYSTEM_NAME``, read at call time. Memories, messages,
speaker names and concepts are private runtime data; log lines carry counts
only. Recalled memories show their stored date (the prompt renders it in
``TIMEZONE``); no relative-age label is added.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from mtc.core.config import get_settings

logger = logging.getLogger(__name__)

RECENCY_HOURS = 24.0
RECENCY_MIN_IMPORTANCE = 5.0
RECENCY_LIMIT = 3
REHYDRATION_WINDOW_HOURS = 24.0
REHYDRATION_LIMIT = 40
ECHO_THRESHOLD = 0.6
CONCEPT_MIN_CHARS = 4
CONCEPT_CATEGORY = "conversation"
CONCEPT_DESCRIPTION = "Concept mentioned in a conversation"

CONCEPT_STOP_WORDS = frozenset("""
    about above after again against also always another anyone anything around
    away back because been before being below between both came cannot could
    come day days does doing done down during each else even ever every from
    getting give going gone good got great have having here hers herself him
    himself into itself just know last later like little look made make many
    maybe more most much must myself need needs never next nothing now often once
    only other ours ourselves over own people really right said same seem
    seems should since some something sometimes soon still such sure take
    than thank thanks that thats their theirs them themselves then there these
    they thing things think this those though thought through time today
    tomorrow told too under until very want wants was way well went were what
    whatever when where which while who whom whose why will with without
    would yeah year years yes yesterday yet your yours yourself yourselves
    responded okay
    """.split())

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_ECHO_WORD = re.compile(r"[a-z0-9']+")
_ECHO_IGNORED = frozenset({"said", "responded"})


@dataclass
class RecallResult:
    """Memories and concepts recalled for one query.

    ``method`` is the manager's recall method (``vector``, ``text`` or
    ``none``); ``errors`` maps each failing part to its error.
    """

    memories: List[Dict[str, Any]] = field(default_factory=list)
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "none"
    errors: Dict[str, str] = field(default_factory=dict)


def _cfg(settings: Any) -> Any:
    return settings if settings is not None else get_settings()


def _memory_key(memory: Mapping[str, Any]) -> Any:
    memory_id = memory.get("id")
    if memory_id is not None:
        return ("id", memory_id)
    return ("content", str(memory.get("content") or "").strip().lower())


async def retrieve_relevant_memories(
    manager: Any,
    query: str,
    *,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    settings: Any = None,
) -> RecallResult:
    """Recall memories for ``query``: recent important ones first, then the
    manager's semantic recall, each memory once."""
    if manager is None:
        return RecallResult(errors={"manager": "memory manager absent"})
    cfg = _cfg(settings)
    limit = int(top_k) if top_k is not None else int(cfg.RAG_TOP_K)
    floor = float(min_score) if min_score is not None else float(cfg.RAG_MIN_SCORE)
    result = RecallResult()
    try:
        recalled = await manager.semantic_recall(query, top_k=limit, min_score=floor)
    except Exception as exc:  # noqa: BLE001 - named in the result
        result.errors["recall"] = f"{type(exc).__name__}: {exc}"
        recalled = {}
    stats = dict(recalled.get("search_stats") or {})
    result.method = str(stats.get("method") or "none")
    for key, value in stats.items():
        if key.endswith("_error") and value:
            result.errors[key] = str(value)
    if isinstance(stats.get("concepts"), str):
        result.errors["concepts"] = stats["concepts"]
    rag = [dict(m) for m in recalled.get("memories") or ()]
    result.concepts = [dict(c) for c in recalled.get("concepts") or ()]

    anchored: List[Dict[str, Any]] = []
    fetch = getattr(manager, "get_recent_important_memories", None)
    if fetch is not None:
        try:
            anchored = [
                dict(m)
                for m in await fetch(
                    hours=RECENCY_HOURS,
                    min_importance=RECENCY_MIN_IMPORTANCE,
                    limit=RECENCY_LIMIT,
                )
            ]
        except Exception as exc:  # noqa: BLE001 - named in the result
            result.errors["recency"] = f"{type(exc).__name__}: {exc}"
    seen: Set[Any] = set()
    for memory in anchored + rag:
        key = _memory_key(memory)
        if key in seen:
            continue
        seen.add(key)
        result.memories.append(memory)
    if result.errors:
        logger.info(
            "memory recall method %s with %d failing part(s)",
            result.method,
            len(result.errors),
        )
    return result


def _echo_words(text: str) -> Set[str]:
    return {
        word
        for word in _ECHO_WORD.findall(str(text or "").lower())
        if len(word) > 2 and word not in _ECHO_IGNORED
    }


def suppress_rag_echoes(
    memories: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    threshold: float = ECHO_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Return the memories that are not echoes of the history window.

    A memory is an echo when at least ``threshold`` of its words (three
    letters or more, ignoring the episodic line's ``said`` and ``responded``)
    appear in one history line. Memories without words are kept.
    """
    lines = [_echo_words(entry.get("content", "")) for entry in history]
    lines = [words for words in lines if words]
    kept: List[Dict[str, Any]] = []
    for memory in memories:
        words = _echo_words(memory.get("content", ""))
        if words and any(
            len(words & line) / len(words) >= float(threshold) for line in lines
        ):
            continue
        kept.append(dict(memory))
    return kept


async def rehydrate_recent_history(
    repository: Any,
    participant: str,
    hours: float = REHYDRATION_WINDOW_HOURS,
    limit: int = REHYDRATION_LIMIT,
    *,
    settings: Any = None,
) -> List[Dict[str, Any]]:
    """History entries for a new session, oldest first, from the stored rows.

    Each entry has ``role`` (the stored role), ``speaker``, ``content``,
    ``timestamp`` and ``conversation_id``. A repository failure propagates.
    """
    if repository is None:
        return []
    system = str(_cfg(settings).SYSTEM_NAME)
    rows = await repository.recent_messages(participant, hours, limit, addressee=system)
    return [
        {
            "role": row["role"],
            "speaker": row["speaker"],
            "content": row["content"],
            "timestamp": row["timestamp"],
            "conversation_id": row["conversation_id"],
        }
        for row in rows
    ]


async def latest_turn_at(
    repository: Any, participant: Optional[str] = None, *, settings: Any = None
) -> Optional[datetime]:
    """The newest message time addressed to the system, or None."""
    if repository is None:
        return None
    system = str(_cfg(settings).SYSTEM_NAME)
    return await repository.latest_message_time(participant, addressee=system)


def concept_candidates(
    text: str, max_concepts: int = 5, exclude: Sequence[str] = ()
) -> List[str]:
    """Content words of ``text`` (lower case, at least ``CONCEPT_MIN_CHARS``
    letters, not a stop word or an excluded name), most frequent first, ties
    in order of first appearance.

    An excluded name excludes every word in it, so a bracketed label such as
    the one an unauthenticated speaker is stored under excludes the bare name
    as well, and a name of several words excludes each of them.
    """
    excluded: Set[str] = set()
    for name in exclude:
        label = str(name).strip().lower()
        if not label:
            continue
        excluded.add(label)
        for match in _WORD.finditer(label):
            excluded.add(match.group(0).strip("'-"))
    counts: Dict[str, int] = {}
    order: Dict[str, int] = {}
    for match in _WORD.finditer(text or ""):
        word = match.group(0).strip("'-").lower()
        if len(word) < CONCEPT_MIN_CHARS or "'" in word:
            continue
        if word in CONCEPT_STOP_WORDS or word in excluded:
            continue
        counts[word] = counts.get(word, 0) + 1
        order.setdefault(word, len(order))
    ranked = sorted(counts, key=lambda w: (-counts[w], order[w]))
    return ranked[: max(0, int(max_concepts))]


async def extract_and_store_concepts(
    graph: Any,
    text: str,
    conversation_id: str,
    speaker: str,
    max_concepts: int = 5,
    *,
    exclude: Sequence[str] = (),
) -> int:
    """Record up to ``max_concepts`` concepts of ``text`` in the graph and
    return how many were stored. The speaker is recorded on the mention only
    (private data), never in the concept's description."""
    if graph is None or not text:
        return 0
    names = concept_candidates(text, max_concepts, exclude=[speaker, *exclude])
    stored = 0
    for name in names:
        await asyncio.to_thread(
            graph.merge_concept, name, CONCEPT_DESCRIPTION, CONCEPT_CATEGORY
        )
        if conversation_id:
            await asyncio.to_thread(
                graph.link_concept_to_conversation, name, conversation_id, speaker
            )
        stored += 1
    return stored


__all__ = (
    "CONCEPT_STOP_WORDS",
    "RecallResult",
    "concept_candidates",
    "extract_and_store_concepts",
    "latest_turn_at",
    "rehydrate_recent_history",
    "retrieve_relevant_memories",
    "suppress_rag_echoes",
)
