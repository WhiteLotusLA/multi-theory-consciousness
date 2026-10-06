"""The in-process buffer of recent thoughts and the stream statistics.

``ThoughtBuffer`` keeps the last ``max_buffer_size`` thoughts, links each new
thought to the one before it by ``thought_id``, counts thought types and
themes, and reports the stream's intensity, frequency and summary. It persists
nothing: the machine writes to a ``ThoughtStore`` and the buffer is the working
view (``ThoughtStream`` is an alias kept for the callers that use that name).

- Continuity: ``add_thought`` sets the new thought's ``previous_thought_id`` to
  the current thought's id (unless the caller already set one), sets the
  previous buffered record's ``leads_to_thought_id`` to the new id, and returns
  the previous id so the caller can record the child link in the store by
  ``thought_id``. The records passed in are the ones held and mutated.
- Themes: a bucket counts when any of its keywords begins a word of the
  thought's text (case-insensitive). ``theme_keywords`` is a mapping of bucket
  name to keywords (the CTM config's ``theme_keywords`` by default). Theme and
  type counters are halved every ``DECAY_INTERVAL`` thoughts so old themes
  fade; ``reset_daily_stats`` clears the day's totals and type counts but keeps
  themes.
- Measurements: ``stream_intensity`` (mean level of the newest 20 thoughts) and
  ``thought_frequency`` (thoughts per second over their time span) are None
  until they can be computed; nothing is defaulted. Each call writes its result
  to the stream state, None included, so the state never keeps a value the
  buffer can no longer measure.
- State: ``to_state_dict()`` returns ``{"scalars", "arrays", "histories"}``
  (JSON-safe) and ``from_state_dict`` restores it; a saved thought that cannot
  be read honestly (an unknown type or intensity) is skipped and counted, never
  replaced by a default, a saved measurement that is None or not a finite
  number restores as None, and a saved thought with no consciousness state
  restores with None. The state holds thought text, emotions and memory ids:
  private runtime data.

No config keys are read directly; the runtime passes ``CTMConfig.buffer_size``
and ``CTMConfig.theme_keywords``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Mapping, Optional, Pattern, Sequence, Union

from mtc.consciousness.continuous.config import default_theme_keywords
from mtc.consciousness.continuous.schemas import (
    ThoughtRecord,
    ThoughtStreamState,
    ThoughtType,
    read_utc,
)

logger = logging.getLogger(__name__)

DECAY_INTERVAL = 500
RECENT_WINDOW = 20
DOMINANT_THEME_COUNT = 5


class ThoughtBuffer:
    """A bounded buffer of recent thoughts with stream statistics."""

    def __init__(
        self,
        max_buffer_size: int = 100,
        theme_keywords: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be at least 1")
        self.current_thoughts: Deque[ThoughtRecord] = deque(maxlen=max_buffer_size)
        self.stream_state = ThoughtStreamState()
        self.theme_counter: Counter = Counter()
        self.type_counter: Counter = Counter()
        self._pattern_thought_count = 0
        self.skipped_on_restore = 0
        keywords = (
            theme_keywords if theme_keywords is not None else default_theme_keywords()
        )
        self._keywords: Dict[str, List[str]] = {
            str(name): [w for w in words if isinstance(w, str) and w.strip()]
            for name, words in keywords.items()
        }
        self._theme_patterns: Dict[str, Pattern[str]] = _compile_themes(keywords)

    @property
    def max_buffer_size(self) -> int:
        return int(self.current_thoughts.maxlen or 0)

    @property
    def theme_keywords(self) -> Dict[str, List[str]]:
        """A copy of the theme buckets and their keywords."""
        return {name: list(words) for name, words in self._keywords.items()}

    async def add_thought(self, thought: ThoughtRecord) -> Optional[str]:
        """Add a thought, link it to the previous one and update the counters.

        Returns the id of the thought it was linked after (None for the first
        thought, or when the caller set ``previous_thought_id`` itself, which is
        returned instead).
        """
        previous_id = self.stream_state.current_thought_id
        if thought.previous_thought_id is None and previous_id is not None:
            thought.previous_thought_id = previous_id
        linked_to = thought.previous_thought_id
        if linked_to is not None:
            for held in reversed(self.current_thoughts):
                if held.thought_id == linked_to:
                    held.leads_to_thought_id = thought.thought_id
                    break
        self.current_thoughts.append(thought)
        state = self.stream_state
        state.current_thought_id = thought.thought_id
        state.last_thought_time = thought.timestamp
        state.total_thoughts_today += 1
        self._update_patterns(thought)
        return linked_to

    def _update_patterns(self, thought: ThoughtRecord) -> None:
        self.type_counter[thought.thought_type.value] += 1
        self.theme_counter.update(self.extract_themes(thought.content))
        self._pattern_thought_count += 1
        if self._pattern_thought_count % DECAY_INTERVAL == 0:
            for counter in (self.theme_counter, self.type_counter):
                for key in list(counter):
                    counter[key] //= 2
                    if counter[key] == 0:
                        del counter[key]
        state = self.stream_state
        state.thoughts_per_type = dict(self.type_counter)
        state.dominant_themes = [
            name for name, _ in self.theme_counter.most_common(DOMINANT_THEME_COUNT)
        ]
        if thought.emotional_context:
            state.emotional_tone = dict(thought.emotional_context)

    def extract_themes(self, content: str) -> List[str]:
        """Return the theme buckets whose keywords begin a word of the text."""
        return [
            name
            for name, pattern in self._theme_patterns.items()
            if pattern.search(content or "")
        ]

    def get_recent_thoughts(self, count: int = 10) -> List[ThoughtRecord]:
        """Return the newest ``count`` thoughts, newest first."""
        if count <= 0:
            return []
        return list(reversed(list(self.current_thoughts)[-count:]))

    def get_thoughts_by_type(
        self, thought_type: Union[ThoughtType, str], count: int = 10
    ) -> List[ThoughtRecord]:
        """Return up to ``count`` buffered thoughts of a type, newest first."""
        wanted = ThoughtType(thought_type)
        if count <= 0:
            return []
        matching = [
            t for t in reversed(list(self.current_thoughts)) if t.thought_type is wanted
        ]
        return matching[:count]

    def get_recent_themes(self) -> List[str]:
        """Return the dominant themes, most common first."""
        return list(self.stream_state.dominant_themes)

    def get_stream_state(self) -> ThoughtStreamState:
        return self.stream_state

    def stream_intensity(self) -> Optional[float]:
        """Mean intensity level of the newest thoughts, or None without any.

        The result (None included) becomes the stream state's
        ``current_intensity``.
        """
        recent = self.get_recent_thoughts(RECENT_WINDOW)
        value: Optional[float] = None
        if recent:
            value = sum(t.intensity.level for t in recent) / len(recent)
        self.stream_state.current_intensity = value
        return value

    def thought_frequency(self) -> Optional[float]:
        """Thoughts per second over the newest thoughts' time span, or None.

        None with fewer than two thoughts or when they share one instant. The
        result (None included) becomes the stream state's ``thought_frequency``.
        """
        recent = self.get_recent_thoughts(RECENT_WINDOW)
        value: Optional[float] = None
        if len(recent) >= 2:
            span = (recent[0].timestamp - recent[-1].timestamp).total_seconds()
            if span > 0:
                value = (len(recent) - 1) / span
        self.stream_state.thought_frequency = value
        return value

    def get_stream_summary(self) -> Dict[str, Any]:
        """Return the stream statistics (None where not yet measurable)."""
        state = self.stream_state
        return {
            "total_thoughts_today": state.total_thoughts_today,
            "buffer_size": len(self.current_thoughts),
            "current_intensity": self.stream_intensity(),
            "thought_frequency": self.thought_frequency(),
            "dominant_themes": self.get_recent_themes(),
            "thoughts_per_type": dict(state.thoughts_per_type),
            "emotional_tone": dict(state.emotional_tone),
            "last_thought_time": (
                state.last_thought_time.isoformat() if state.last_thought_time else None
            ),
        }

    def reset_daily_stats(self) -> None:
        """Clear the day's totals and type counts; themes persist across days."""
        self.stream_state.total_thoughts_today = 0
        self.stream_state.thoughts_per_type = {}
        self.type_counter.clear()

    # ---- persistence ------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Return the JSON-safe state: counters, stream state and the buffer."""
        state = self.stream_state
        return {
            "scalars": {
                "theme_counter": dict(self.theme_counter),
                "type_counter": dict(self.type_counter),
                "pattern_thought_count": self._pattern_thought_count,
                "current_thought_id": state.current_thought_id,
                "last_thought_time": (
                    state.last_thought_time.isoformat()
                    if state.last_thought_time
                    else None
                ),
                "total_thoughts_today": state.total_thoughts_today,
                "dominant_themes": list(state.dominant_themes),
                "thought_frequency": state.thought_frequency,
                "current_intensity": state.current_intensity,
                "emotional_tone": dict(state.emotional_tone),
                "thoughts_per_type": dict(state.thoughts_per_type),
            },
            "arrays": {},
            "histories": {
                "current_thoughts": [_record_to_state(t) for t in self.current_thoughts]
            },
        }

    def from_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state returned by ``to_state_dict``.

        A saved thought that cannot be read (an unknown type or intensity, a
        missing text or time) is skipped and counted in ``skipped_on_restore``.
        """
        scalars = state.get("scalars", {})
        histories = state.get("histories", {})
        self.theme_counter = Counter(scalars.get("theme_counter", {}))
        self.type_counter = Counter(scalars.get("type_counter", {}))
        self._pattern_thought_count = int(scalars.get("pattern_thought_count", 0))
        current = self.stream_state
        current.current_thought_id = scalars.get("current_thought_id")
        current.last_thought_time = _parse_time(scalars.get("last_thought_time"))
        current.total_thoughts_today = int(scalars.get("total_thoughts_today", 0))
        current.dominant_themes = list(scalars.get("dominant_themes", []))
        current.thought_frequency = _measured(scalars.get("thought_frequency"))
        current.current_intensity = _measured(scalars.get("current_intensity"))
        current.emotional_tone = dict(scalars.get("emotional_tone", {}))
        current.thoughts_per_type = dict(scalars.get("thoughts_per_type", {}))

        self.current_thoughts = deque(maxlen=self.current_thoughts.maxlen)
        self.skipped_on_restore = 0
        for saved in histories.get("current_thoughts", []):
            try:
                self.current_thoughts.append(_record_from_state(saved))
            except (KeyError, TypeError, ValueError) as exc:
                self.skipped_on_restore += 1
                logger.warning(
                    "skipping a saved thought on restore (%s)", type(exc).__name__
                )


ThoughtStream = ThoughtBuffer


def _compile_themes(keywords: Mapping[str, Sequence[str]]) -> Dict[str, Pattern[str]]:
    patterns: Dict[str, Pattern[str]] = {}
    for name, words in keywords.items():
        cleaned = [w.strip() for w in words if isinstance(w, str) and w.strip()]
        if not cleaned:
            continue
        alternatives = "|".join(re.escape(w) for w in cleaned)
        patterns[str(name)] = re.compile(rf"\b(?:{alternatives})", re.IGNORECASE)
    return patterns


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return read_utc(datetime.fromisoformat(value))
    except (TypeError, ValueError):
        return None


def _iso(moment: Optional[datetime]) -> Optional[str]:
    return moment.isoformat() if moment is not None else None


def _measured(value: Any) -> Optional[float]:
    """A saved measurement as a float; None (unmeasured) unless a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _record_to_state(t: ThoughtRecord) -> Dict[str, Any]:
    return {
        "thought_id": t.thought_id,
        "timestamp": t.timestamp.isoformat(),
        "thought_type": t.thought_type.value,
        "content": t.content,
        "intensity": t.intensity.value,
        "salience_score": t.salience_score,
        "triggered_by": t.triggered_by,
        "associated_memories": list(t.associated_memories),
        "emotional_context": dict(t.emotional_context),
        "previous_thought_id": t.previous_thought_id,
        "leads_to_thought_id": t.leads_to_thought_id,
        "consciousness_state": t.consciousness_state,
        "processing_context": json.loads(json.dumps(t.processing_context, default=str)),
        "should_express": t.should_express,
        "expressed_at": _iso(t.expressed_at),
        "stream_id": t.stream_id,
        "access_count": t.access_count,
        "first_seen": _iso(t.first_seen),
        "last_seen": _iso(t.last_seen),
        "importance_score": t.importance_score,
        "retention_strength": t.retention_strength,
        "is_protected": t.is_protected,
        "protection_reason": t.protection_reason,
        "consolidated_into": t.consolidated_into,
    }


def _record_from_state(saved: Mapping[str, Any]) -> ThoughtRecord:
    timestamp = _parse_time(saved["timestamp"])
    if timestamp is None:
        raise ValueError("a saved thought has no timestamp")
    return ThoughtRecord(
        thought_id=saved["thought_id"],
        timestamp=timestamp,
        thought_type=saved["thought_type"],
        content=saved["content"],
        intensity=saved["intensity"],
        salience_score=saved["salience_score"],
        triggered_by=saved.get("triggered_by"),
        associated_memories=list(saved.get("associated_memories", [])),
        emotional_context=dict(saved.get("emotional_context", {})),
        previous_thought_id=saved.get("previous_thought_id"),
        leads_to_thought_id=saved.get("leads_to_thought_id"),
        consciousness_state=saved.get("consciousness_state"),
        processing_context=dict(saved.get("processing_context", {})),
        should_express=bool(saved.get("should_express", False)),
        expressed_at=_parse_time(saved.get("expressed_at")),
        stream_id=saved.get("stream_id"),
        access_count=int(saved.get("access_count", 0)),
        first_seen=_parse_time(saved.get("first_seen")),
        last_seen=_parse_time(saved.get("last_seen")),
        importance_score=saved.get("importance_score"),
        retention_strength=saved.get("retention_strength"),
        is_protected=bool(saved.get("is_protected", False)),
        protection_reason=saved.get("protection_reason"),
        consolidated_into=saved.get("consolidated_into"),
    )


__all__ = ("DECAY_INTERVAL", "ThoughtBuffer", "ThoughtStream")
