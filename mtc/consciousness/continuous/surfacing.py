"""Which stored thoughts and gists the conversation prompt may see.

A thought surfaces when it was marked expressible by the novelty gate and its
type does not start with an excluded prefix (``dream`` by default: dream
thoughts never surface, by type). Repeated texts surface once. A gist surfaces
when its retention is at or above a floor, most important first.

This module returns data; the prompt assembler is the one renderer of the
thought and gist sections. ``thought_items`` and ``gist_items`` return the
dicts it reads: a thought item has ``content``, ``timestamp``, ``thought_type``,
``emotional_tone`` (the strongest emotion's name, or None), ``form``
(``template``, ``model`` or ``injected``, None when the record names none) and
``salience_basis`` (what the stored salience is: a measurement or a design
constant), a gist item has
``summary``, ``source_count`` and ``dominant_emotion`` (None when no thought in
the gist carried a measured emotion). The assembler stamps a mood only when
``NARRATE_THOUGHT_MOOD`` is on and the value is not None, so the tone is
carried as data and never narrated here. Phi appears in no item. Thought text and gist
summaries are private runtime data and are not logged. No config keys are read
directly; the caller passes the limits (``PROMPT_MAX_THOUGHTS`` and
``PROMPT_MAX_GISTS``) and the excluded prefixes (``CTMConfig.surfacing``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from mtc.consciousness.continuous.schemas import ThoughtRecord
from mtc.consciousness.continuous.store import (
    DEFAULT_EXCLUDED_PREFIXES,
    ThoughtStore,
)
from mtc.memory.protocols import ThoughtGistStore
from mtc.memory.thought_gist import ThoughtGist


@dataclass(frozen=True)
class SurfacedThought:
    """A thought as the prompt sees it."""

    thought_id: str
    content: str
    timestamp: datetime
    thought_type: str
    emotional_tone: Optional[str]
    stream_id: Optional[str]
    form: Optional[str] = None
    salience_basis: Optional[str] = None


def _tone(emotions: Dict[str, float]) -> Optional[str]:
    """The name of the strongest emotion, or None when there is none."""
    scored = [(name, score) for name, score in emotions.items() if _number(score)]
    if not scored:
        return None
    return max(scored, key=lambda item: (item[1], item[0]))[0]


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _surfaced(record: ThoughtRecord) -> SurfacedThought:
    return SurfacedThought(
        thought_id=record.thought_id,
        content=record.content,
        timestamp=record.timestamp,
        thought_type=record.thought_type.value,
        emotional_tone=_tone(record.emotional_context),
        stream_id=record.stream_id,
        form=_text((record.processing_context or {}).get("form")),
        salience_basis=_text((record.processing_context or {}).get("salience_basis")),
    )


def _text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


async def get_surfaced_thoughts(
    store: ThoughtStore,
    limit: int = 5,
    exclude_type_prefixes: Sequence[str] = DEFAULT_EXCLUDED_PREFIXES,
) -> List[SurfacedThought]:
    """Return up to ``limit`` expressible thoughts, newest first.

    Only thoughts the novelty gate marked expressible surface; a type starting
    with an excluded prefix and a repeated text are left out.
    """
    if limit <= 0:
        return []
    records = await store.expressible(limit, tuple(exclude_type_prefixes))
    return [_surfaced(record) for record in records[:limit]]


async def get_gists(
    gist_store: ThoughtGistStore, limit: int = 5, min_retention: float = 0.5
) -> List[ThoughtGist]:
    """Return up to ``limit`` gists with retention at or above the floor.

    Most important first (the order ``active_gists`` defines).
    """
    if limit <= 0:
        return []
    gists = await gist_store.active_gists(min_retention=min_retention, limit=limit)
    return list(gists)[:limit]


def thought_items(thoughts: Sequence[SurfacedThought]) -> List[Dict[str, Any]]:
    """Return the dicts the prompt assembler reads for thoughts."""
    return [
        {
            "thought_id": t.thought_id,
            "content": t.content,
            "timestamp": t.timestamp,
            "thought_type": t.thought_type,
            "emotional_tone": t.emotional_tone,
            "stream_id": t.stream_id,
            "form": t.form,
            "salience_basis": t.salience_basis,
        }
        for t in thoughts
    ]


def gist_items(gists: Sequence[ThoughtGist]) -> List[Dict[str, Any]]:
    """Return the dicts the prompt assembler reads for gists.

    The count of source thoughts is ``source_count``, the key the assembler
    reads and the key ``ThoughtGist`` stores.
    """
    return [
        {
            "gist_id": g.gist_id,
            "summary": g.summary,
            "source_count": g.source_count,
            "dominant_emotion": g.dominant_emotion,
            "timestamp": g.created_at,
            "importance_score": g.importance_score,
            "retention_strength": g.retention_strength,
        }
        for g in gists
    ]


__all__ = (
    "SurfacedThought",
    "gist_items",
    "get_gists",
    "get_surfaced_thoughts",
    "thought_items",
)
