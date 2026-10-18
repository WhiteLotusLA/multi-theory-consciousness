"""
Memory Consolidation Pipeline
=============================

Sleep-inspired consolidation over recent memories and thoughts, after
complementary learning systems theory (McClelland et al., 1995) and the active
systems consolidation model (Born and Wilhelm, 2012):

- Memory patterns: recurring learning, social, conceptual, sensory, emotional
  and nature patterns in recent memories (``extract_patterns``), importance
  ranking, strategy selection, and core-memory formation from strong patterns
  (``run_consolidation_cycle``), through the memory manager's
  ``get_recent_memories`` and ``store_memory``.
- Thought consolidation: thoughts older than ``min_thought_age_hours`` are
  clustered by text similarity with temporal burst detection, scored and
  checked for protection, and each unprotected cluster becomes a
  ``ThoughtGist`` summarised by the brain (``run_thought_consolidation``).
  A cluster whose thoughts carry no numeric emotion score gets
  ``dominant_emotion`` None and no emotional arc, its prompt says no emotion
  was measured, and its vector point carries ``emotional_weight`` None. A
  cluster without salience values gets ``avg_salience`` None. A new gist's
  retention is computed at creation (age zero). Gists go to the
  ``ThoughtGistStore`` and are embedded in the vector store with payload type
  ``gist``.
- Retention: the forgetting curve for memories and gists (true half-lives
  ``memory_half_life_days`` and ``gist_half_life_days``), the gist lifecycle
  classes, conflict resolution and connection strengthening. The refresh pass
  walks every stored gist through ``ThoughtGistStore.iter_gists``, so a gist
  stored without a retention gets one on the next pass.
- A memory row without a numeric emotional context has ``emotional_intensity``
  None; strategy selection averages the measured intensities only.

The brain is reached only through ``BrainProtocol``. Without a brain every
model step is skipped and the report says so; ``NoAnswerError`` or an empty
reply stores nothing, and the cluster stays for the next pass (counted under
``no_answer``). Reasoning blocks are stripped from every reply. No summary
text is composed by code in place of a model answer. Prompts are neutral and
third person and name the system only as configured (``SYSTEM_NAME``).

Relationship mentions raise a cluster's importance: the words come from
``RELATIONSHIP_KEYWORDS`` at call time and match whole words only. Identity
markers protect a cluster: the generic phrases in ``protected_markers`` plus
``i am <SYSTEM_NAME>`` rendered at call time. No participant is protected
unless ``protected_participants`` names one. The developmental stage applies
only when ``developmental_enabled()`` is true and the caller passes the stage;
otherwise the stage-independent defaults apply. Dreams belong to the dream
engine and the sleep scheduler, not to this pipeline.

Thoughts, gists and memories hold private conversation content at runtime.
Config keys read: ``SYSTEM_NAME``, ``RELATIONSHIP_KEYWORDS``,
``COMPANION_TYPE`` and ``DEVELOPMENTAL_STAGE_ENABLED`` (through
``developmental_enabled``).
"""

import asyncio
import logging
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from mtc.brain.protocol import BrainProtocol, NoAnswerError, strip_reasoning_blocks
from mtc.consciousness.developmental_modulator import developmental_enabled
from mtc.core.config import Settings, csv_list, get_settings
from mtc.memory.protocols import ThoughtGistStore, VectorStore
from mtc.memory.thought_gist import ThoughtGist

logger = logging.getLogger(__name__)

CORE_MEMORY_TYPE = "core"
CONCEPTUAL_KEYWORDS = ("make", "makes", "because", "why", "how")
CONCEPTUAL_TAGS = ("color", "concept", "learning")


class ConsolidationStrategy(Enum):
    """Strategy for consolidating memories, chosen from their characteristics."""

    SLEEP_CONSOLIDATION = "sleep_consolidation"
    ACTIVE_REHEARSAL = "active_rehearsal"
    SPACED_REPETITION = "spaced_repetition"


@dataclass
class MemoryPattern:
    """A pattern extracted from a set of related memories."""

    pattern_type: str
    content: str
    strength: float
    supporting_memories: List[str] = field(default_factory=list)
    emotional_significance: float = 0.0


@dataclass
class ConsolidationResult:
    """Result of one memory consolidation cycle."""

    memories_processed: int
    patterns_found: int
    core_memories_formed: int
    processing_time: float
    next_consolidation: datetime
    strategy: Optional[str] = None
    skipped: Optional[str] = None


@dataclass
class ThoughtConsolidationReport:
    """Result of one thought consolidation pass.

    ``delete_ids`` lists the thoughts a caller may delete: those of clusters
    whose gist was produced (and stored, when a gist store is configured).
    ``skipped`` names steps that did not run and why; ``errors`` holds the
    error of each part that failed.
    """

    thoughts_processed: int = 0
    clusters: int = 0
    gists_created: int = 0
    gists_stored: int = 0
    gists_embedded: int = 0
    protected_count: int = 0
    no_answer: int = 0
    thoughts_to_delete: int = 0
    gists_refreshed: int = 0
    delete_ids: List[str] = field(default_factory=list)
    gists: List[ThoughtGist] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["gists"] = [gist.to_document() for gist in self.gists]
        return result


@dataclass
class ConsolidationConfig:
    """Thresholds and weights of the pipeline (the reference's design values)."""

    # Thought clustering and bursts
    similarity_threshold: float = 0.85
    burst_gap_minutes: float = 30.0
    min_thought_age_hours: float = 48.0
    max_variations: int = 5
    # Cluster importance weights (they sum to 1)
    weight_emotion: float = 0.30
    weight_size: float = 0.20
    weight_conversation: float = 0.20
    weight_expressed: float = 0.15
    weight_uniqueness: float = 0.15
    size_reference_fraction: float = 0.1
    # Protection
    protect_emotion_above: float = 0.8
    protected_markers: Tuple[str, ...] = (
        "my name is",
        "i exist",
        "who i am",
        "my identity",
    )
    protected_participants: Tuple[str, ...] = ()
    # Memory patterns
    social_tags: Tuple[str, ...] = ("social", "comfort")
    emotional_intensity_pattern: float = 0.6
    core_memory_strength: float = 0.8
    core_memory_importance_floor: float = 0.91
    conceptual_min_stage: int = 2
    # Strategy selection
    strategy_emotion_threshold: float = 0.5
    strategy_repetition_threshold: float = 3.0
    strategy_span_days: float = 3.0
    # Forgetting curve
    memory_half_life_days: float = 7.0
    gist_half_life_days: float = 14.0
    memory_access_boost: float = 0.1
    gist_access_boost: float = 0.15
    max_access_boost: float = 0.5
    importance_floor_factor: float = 0.85
    protected_gist_floor: float = 0.2
    gist_active_at: float = 0.5
    gist_fading_at: float = 0.2
    # Conflict resolution
    conflict_penalty: float = 0.2
    conflict_min_confidence: float = 0.3
    # Brain calls and cadence
    gist_max_tokens: int = 150
    synthesis_max_tokens: int = 300
    consolidation_interval_hours: float = 4.0

    def markers(self, system_name: str) -> Tuple[str, ...]:
        """The generic identity markers plus ``i am <system name>``."""
        rendered = f"i am {system_name.strip().lower()}" if system_name else ""
        return tuple(self.protected_markers) + ((rendered,) if rendered else ())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Any) -> Optional[datetime]:
    """Read a timestamp (datetime or ISO text); a naive value is UTC."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_values(context: Any) -> List[float]:
    if not isinstance(context, Mapping):
        return []
    return [
        float(v)
        for v in context.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]


def _part_status(part: Any) -> Dict[str, Any]:
    """Presence and, when the part reports one, its own status."""
    if part is None:
        return {"present": False, "status": None}
    reader = getattr(part, "status", None)
    if not callable(reader):
        return {"present": True, "status": None}
    try:
        reported = reader()
    except Exception as exc:  # noqa: BLE001 - status never raises
        return {
            "present": True,
            "status": {
                "available": False,
                "last_error": f"{type(exc).__name__}: {exc}",
            },
        }
    return {
        "present": True,
        "status": dict(reported) if isinstance(reported, Mapping) else reported,
    }


def _phrase_pattern(phrases: Sequence[str]) -> Optional["re.Pattern[str]"]:
    cleaned = [p.strip().lower() for p in phrases if p and p.strip()]
    if not cleaned:
        return None
    alternatives = "|".join(re.escape(p) for p in cleaned)
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


class MemoryConsolidationPipeline:
    """Consolidates memories and thoughts; stores and the brain are seams."""

    def __init__(
        self,
        memory_manager: Any = None,
        brain: Optional[BrainProtocol] = None,
        gist_store: Optional[ThoughtGistStore] = None,
        vector_store: Optional[VectorStore] = None,
        config: Optional[ConsolidationConfig] = None,
        settings: Optional[Settings] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.memory_manager = memory_manager
        self.brain = brain
        self.gist_store = gist_store
        self.vector_store = vector_store
        self.config = config or ConsolidationConfig()
        self._settings = settings
        self._clock = clock or _utcnow
        self.developmental_stage: Optional[int] = None
        self.last_brain_error: Optional[str] = None
        self.last_errors: Dict[str, str] = {}
        self.last_unmeasured_gists = 0
        self._last_consolidation: Optional[datetime] = None

    @property
    def settings(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available``, ``last_error`` and the parts.

        ``parts`` names each part (memory manager, brain, gist store, vector
        store) as present or absent with its own status when it has one;
        ``errors`` holds the last error of each part from the latest pass.
        ``available`` is False when no part is present or a present part
        reports itself unavailable.
        """
        parts = {
            "memory_manager": _part_status(self.memory_manager),
            "brain": _part_status(self.brain),
            "gist_store": _part_status(self.gist_store),
            "vector_store": _part_status(self.vector_store),
        }
        errors = dict(self.last_errors)
        if self.last_brain_error and "brain" not in errors:
            errors["brain"] = self.last_brain_error
        present = [name for name, part in parts.items() if part["present"]]
        return {
            "backend": "+".join(present) if present else "none",
            "available": bool(present)
            and not any(
                isinstance(part["status"], Mapping)
                and part["status"].get("available") is False
                for part in parts.values()
            ),
            "last_error": next(iter(errors.values()), None),
            "parts": parts,
            "absent": [name for name, part in parts.items() if not part["present"]],
            "errors": errors,
            "unmeasured_gists": self.last_unmeasured_gists,
            "last_consolidation": (
                self._last_consolidation.isoformat()
                if self._last_consolidation is not None
                else None
            ),
        }

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("the pipeline clock must return aware datetimes")
        return moment.astimezone(timezone.utc)

    def _system_name(self) -> str:
        return self.settings.SYSTEM_NAME

    # ------------------------------------------------------------------
    # Memory patterns
    # ------------------------------------------------------------------

    def _conceptual_enabled(self, stage: Optional[int]) -> bool:
        if stage is None or not developmental_enabled(self.settings):
            return True
        return int(stage) >= self.config.conceptual_min_stage

    async def extract_patterns(
        self, memories: List[Dict[str, Any]], developmental_stage: Optional[int] = None
    ) -> List[MemoryPattern]:
        """Return the recurring patterns in ``memories``.

        Memory dicts may carry ``content``, ``tags``, ``participants``,
        ``emotional_intensity`` and ``memory_id``.
        """
        stage = (
            developmental_stage
            if developmental_stage is not None
            else self.developmental_stage
        )
        total = max(len(memories), 1)
        patterns: List[MemoryPattern] = []

        def tags_of(memory: Mapping[str, Any]) -> List[str]:
            return list(memory.get("tags") or [])

        def ids_of(group: Sequence[Mapping[str, Any]]) -> List[str]:
            return [str(m.get("memory_id", "")) for m in group]

        def joined(group: Sequence[Mapping[str, Any]]) -> str:
            return "; ".join(str(m.get("content", "")) for m in group)

        learning = [m for m in memories if "learning" in tags_of(m)]
        if learning:
            co_tags: Counter = Counter(
                t for m in learning for t in tags_of(m) if t != "learning"
            )
            subject = co_tags.most_common(1)[0][0] if co_tags else "general"
            patterns.append(
                MemoryPattern(
                    "learning",
                    f"Learning about {subject}: " + joined(learning),
                    min(1.0, len(learning) / total + 0.5),
                    ids_of(learning),
                )
            )

        social_tags = set(self.config.social_tags)
        social = [
            m
            for m in memories
            if social_tags.intersection(tags_of(m))
            or len(m.get("participants") or []) > 1
        ]
        if social:
            patterns.append(
                MemoryPattern(
                    "social",
                    "Social interactions: " + joined(social),
                    min(1.0, len(social) / total + 0.4),
                    ids_of(social),
                )
            )

        if self._conceptual_enabled(stage):
            keyword = _phrase_pattern(CONCEPTUAL_KEYWORDS)
            conceptual = [
                m
                for m in memories
                if set(CONCEPTUAL_TAGS).intersection(tags_of(m))
                or (keyword is not None and keyword.search(str(m.get("content", ""))))
            ]
            if conceptual:
                patterns.append(
                    MemoryPattern(
                        "conceptual",
                        "Conceptual understanding: " + joined(conceptual),
                        min(1.0, len(conceptual) / total + 0.3),
                        ids_of(conceptual),
                    )
                )

        sensory = [m for m in memories if "sensation" in tags_of(m)]
        if sensory:
            patterns.append(
                MemoryPattern(
                    "sensory",
                    "Sensory experiences: " + joined(sensory),
                    min(1.0, len(sensory) / total + 0.3),
                    ids_of(sensory),
                )
            )

        emotional = [
            m
            for m in memories
            if "emotion" in tags_of(m)
            or float(m.get("emotional_intensity", 0.0) or 0.0)
            > self.config.emotional_intensity_pattern
        ]
        if emotional:
            significance = max(
                float(m.get("emotional_intensity", 0.0) or 0.0) for m in emotional
            )
            patterns.append(
                MemoryPattern(
                    "emotional",
                    "Emotional experiences: " + joined(emotional),
                    min(1.0, len(emotional) / total + 0.4),
                    ids_of(emotional),
                    emotional_significance=significance,
                )
            )

        nature = [m for m in memories if "nature" in tags_of(m)]
        if nature:
            patterns.append(
                MemoryPattern(
                    "nature",
                    "Nature observations: " + joined(nature),
                    min(1.0, len(nature) / total + 0.3),
                    ids_of(nature),
                )
            )
        return patterns

    def rank_by_importance(self, memories: List[Dict[str, Any]]) -> List[Dict]:
        """Rank memories by emotional intensity, participants, tags and repetition."""
        scored = []
        for memory in memories:
            score = float(memory.get("emotional_intensity", 0.0) or 0.0) * 0.4
            if len(memory.get("participants") or []) > 1:
                score += 0.2
            tags = memory.get("tags") or []
            if "achievement" in tags:
                score += 0.15
            if "learning" in tags:
                score += 0.05
            score += min(int(memory.get("repetitions", 1) or 0) * 0.05, 0.2)
            ranked = dict(memory)
            ranked["importance_score"] = score
            scored.append(ranked)
        scored.sort(key=lambda m: m["importance_score"], reverse=True)
        return scored

    def select_strategy(self, memories: List[Dict[str, Any]]) -> ConsolidationStrategy:
        """High emotion: sleep consolidation; high repetition: active rehearsal;
        otherwise spaced repetition.

        The emotion test averages the memories whose ``emotional_intensity``
        is a number; a memory without one is left out, not read as 0.
        """
        if not memories:
            return ConsolidationStrategy.SLEEP_CONSOLIDATION
        intensities = [
            float(m["emotional_intensity"])
            for m in memories
            if _is_number(m.get("emotional_intensity"))
        ]
        # The mean of the measured intensities only; with none measured the
        # emotion test does not pass.
        if (
            intensities
            and sum(intensities) / len(intensities)
            > self.config.strategy_emotion_threshold
        ):
            return ConsolidationStrategy.SLEEP_CONSOLIDATION
        repetitions = [float(m.get("repetitions", 0) or 0) for m in memories]
        if (
            sum(repetitions) / len(repetitions)
            >= self.config.strategy_repetition_threshold
        ):
            return ConsolidationStrategy.ACTIVE_REHEARSAL
        return ConsolidationStrategy.SPACED_REPETITION

    def prioritize_for_consolidation(
        self, memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort memories by emotional intensity, highest first."""
        return sorted(
            memories,
            key=lambda m: float(m.get("emotional_intensity", 0.0) or 0.0),
            reverse=True,
        )

    async def synthesize_knowledge(self, patterns: List[MemoryPattern]) -> Dict:
        """Ask the brain to synthesise the patterns into one statement.

        Returns ``{"synthesis": text}``, ``{"no_answer": True}`` or
        ``{"skipped": reason}``; nothing is composed without an answer.
        """
        if self.brain is None:
            return {"skipped": "no brain"}
        if not patterns:
            return {"skipped": "no patterns"}
        name = self._system_name()
        listing = "\n".join(
            f"- {p.pattern_type} (strength {p.strength:.2f}): {p.content}"
            for p in patterns
        )
        prompt = (
            f"These are recurring patterns in the recent memories of {name}:\n"
            f"{listing}\n"
            f"State in two or three sentences what {name} has come to understand "
            "from them."
        )
        system_prompt = (
            f"Summarise the learning of {name}, an AI research system, in the "
            "third person. Be concise and stay with what the patterns show."
        )
        text = await self._ask(prompt, system_prompt, self.config.synthesis_max_tokens)
        if text is None:
            return {"no_answer": True}
        return {"synthesis": text}

    async def form_core_memory(self, pattern: MemoryPattern) -> Dict[str, Any]:
        """Store a strong pattern as a core memory row (memory type ``core``).

        The row's importance is ``10 * max(core_memory_importance_floor,
        strength)`` on the episodic 0 to 10 scale, which keeps it above the
        pruner's thresholds.
        """
        importance = max(self.config.core_memory_importance_floor, pattern.strength)
        core = {
            "memory_type": CORE_MEMORY_TYPE,
            "content": pattern.content,
            "pattern_type": pattern.pattern_type,
            "importance": importance,
            "emotional_significance": pattern.emotional_significance,
            "consolidated_from": list(pattern.supporting_memories),
            "formed_at": self._now().isoformat(),
            "stored": False,
        }
        if self.memory_manager is None:
            return core
        result = await self.memory_manager.store_memory(
            pattern.content,
            memory_type=CORE_MEMORY_TYPE,
            importance=min(10.0, 10.0 * importance),
        )
        core["postgres_id"] = getattr(result, "postgres_id", None)
        core["stored"] = core["postgres_id"] is not None or bool(
            getattr(result, "buffered", False)
        )
        return core

    async def strengthen_connections(
        self, related_memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Connect memory pairs that share tags or content words.

        The connections are passed to the memory manager's
        ``update_memory_connections`` when it has one; either way they are
        returned.
        """
        connections: List[Dict[str, Any]] = []
        for i, first in enumerate(related_memories):
            for j in range(i + 1, len(related_memories)):
                second = related_memories[j]
                shared_tags = set(first.get("tags") or []) & set(
                    second.get("tags") or []
                )
                shared_words = {
                    w
                    for w in set(str(first.get("content", "")).lower().split())
                    & set(str(second.get("content", "")).lower().split())
                    if len(w) > 2
                }
                reasons = []
                if shared_tags:
                    reasons.append(f"shared tags: {', '.join(sorted(shared_tags))}")
                if shared_words:
                    reasons.append(
                        f"shared concepts: {', '.join(sorted(shared_words))}"
                    )
                if reasons:
                    connections.append(
                        {
                            "from": first.get("memory_id", f"mem_{i}"),
                            "to": second.get("memory_id", f"mem_{j}"),
                            "reason": "; ".join(reasons),
                            "strength": len(shared_tags) * 0.3
                            + len(shared_words) * 0.1,
                        }
                    )
        updater = getattr(self.memory_manager, "update_memory_connections", None)
        if connections and callable(updater):
            await updater(connections)
        elif connections:
            logger.debug("memory connections computed; no store takes them")
        return connections

    def _as_pipeline_memory(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """Turn a memory manager row (importance 0 to 10) into a pipeline memory.

        ``emotional_intensity`` is the largest numeric value of the row's
        emotional context clamped to [0, 1], or None when it holds no number.
        """
        context = row.get("emotional_context") or {}
        values = _numeric_values(context)
        memory = dict(row)
        memory.setdefault("memory_id", str(row.get("id", "")))
        memory["timestamp"] = _as_utc(row.get("timestamp") or row.get("created_at"))
        memory["importance"] = float(row.get("importance", 0.0) or 0.0) / 10.0
        memory["access_count"] = int(row.get("accessed_count", 0) or 0)
        memory.setdefault(
            "emotional_intensity", max(0.0, min(1.0, max(values))) if values else None
        )
        for key in ("tags", "participants"):
            if key not in memory and isinstance(context, Mapping):
                found = context.get(key)
                if isinstance(found, list):
                    memory[key] = list(found)
        return memory

    async def run_consolidation_cycle(
        self, developmental_stage: Optional[int] = None
    ) -> ConsolidationResult:
        """Extract patterns from recent memories and form core memories."""
        started = time.perf_counter()
        now = self._now()
        next_time = now + timedelta(hours=self.config.consolidation_interval_hours)
        if self.memory_manager is None:
            return ConsolidationResult(
                0, 0, 0, 0.0, next_time, skipped="no memory manager"
            )
        rows = await self.memory_manager.get_recent_memories()
        existing_core = {
            str(r.get("content", ""))
            for r in rows
            if r.get("memory_type") == CORE_MEMORY_TYPE
        }
        memories = [
            self._as_pipeline_memory(r)
            for r in rows
            if r.get("memory_type") != CORE_MEMORY_TYPE
        ]
        patterns = await self.extract_patterns(memories, developmental_stage)
        strategy = self.select_strategy(memories)
        formed = 0
        for pattern in patterns:
            if pattern.strength < self.config.core_memory_strength:
                continue
            if pattern.content in existing_core:
                continue
            core = await self.form_core_memory(pattern)
            if core["stored"]:
                existing_core.add(pattern.content)
                formed += 1
        self._last_consolidation = now
        return ConsolidationResult(
            memories_processed=len(memories),
            patterns_found=len(patterns),
            core_memories_formed=formed,
            processing_time=time.perf_counter() - started,
            next_consolidation=next_time,
            strategy=strategy.value,
        )

    # ------------------------------------------------------------------
    # Thought clustering, scoring and protection
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).lower().strip())

    @staticmethod
    def _text_similarity(first: str, second: str) -> float:
        """Jaccard similarity of the word sets."""
        a, b = set(first.split()), set(second.split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _detect_bursts(
        self, timestamps: Sequence[datetime], gap_minutes: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Split sorted timestamps into bursts separated by gaps over the limit."""
        if not timestamps:
            return []
        gap = self.config.burst_gap_minutes if gap_minutes is None else gap_minutes
        bursts = []
        start, count = timestamps[0], 1
        for previous, current in zip(timestamps, timestamps[1:]):
            if (current - previous).total_seconds() / 60.0 > gap:
                bursts.append(
                    {
                        "start": start.isoformat(),
                        "end": previous.isoformat(),
                        "count": count,
                    }
                )
                start, count = current, 1
            else:
                count += 1
        bursts.append(
            {
                "start": start.isoformat(),
                "end": timestamps[-1].isoformat(),
                "count": count,
            }
        )
        return bursts

    def cluster_thoughts(
        self,
        thoughts: List[Dict[str, Any]],
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Cluster thoughts by normalised text similarity.

        Each cluster carries ``representative``, ``count``, ``thought_ids``,
        ``burst_windows``, ``timestamps``, ``emotional_contexts``,
        ``content_variations``, ``participants``, ``expressed`` and
        ``salience``.
        """
        threshold = (
            self.config.similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        )
        groups: List[Tuple[str, Dict[str, Any]]] = []
        for thought in thoughts:
            content = str(thought.get("content", "")).strip()
            normalized = self._normalize_text(content)
            for key, group in groups:
                if self._text_similarity(normalized, key) >= threshold:
                    group["thoughts"].append(thought)
                    break
            else:
                groups.append(
                    (normalized, {"representative": content, "thoughts": [thought]})
                )

        clusters = []
        for _key, group in groups:
            members = group["thoughts"]
            stamps = sorted(
                t for t in (_as_utc(m.get("timestamp")) for m in members) if t
            )
            variations: List[str] = []
            for member in members:
                text = str(member.get("content", "")).strip()
                if text and text not in variations:
                    variations.append(text)
            participants: List[str] = []
            for member in members:
                for name in member.get("participants") or []:
                    if name not in participants:
                        participants.append(name)
            clusters.append(
                {
                    "representative": group["representative"],
                    "count": len(members),
                    "thought_ids": [
                        str(m.get("thought_id", m.get("_id", ""))) for m in members
                    ],
                    "burst_windows": self._detect_bursts(stamps),
                    "timestamps": stamps,
                    "emotional_contexts": [
                        dict(m.get("emotional_context") or {}) for m in members
                    ],
                    "content_variations": variations,
                    "participants": participants,
                    "expressed": [
                        bool(m["expressed"]) for m in members if "expressed" in m
                    ],
                    "salience": [
                        float(m["salience_score"])
                        for m in members
                        if isinstance(m.get("salience_score"), (int, float))
                    ],
                }
            )
        return clusters

    def _relationship_pattern(self) -> Optional["re.Pattern[str]"]:
        return _phrase_pattern(csv_list(self.settings.RELATIONSHIP_KEYWORDS))

    def score_cluster_importance(
        self, cluster: Mapping[str, Any], total_thoughts: int
    ) -> float:
        """Importance in [0, 1]: emotion, size, conversation link, expression and
        uniqueness, weighted by the config.

        The conversation term is 1 when a thought id names a conversation or
        the representative mentions a relationship keyword as a whole word.
        The expression term is the fraction of the cluster's thoughts marked
        ``expressed`` (0 when none carries the mark). The emotion term is the
        largest numeric emotion value of the cluster clamped to [0, 1], and 0
        when no context holds a number: the score is a documented weighted
        sum, so an absent term contributes nothing.
        """
        cfg = self.config
        emotions = [
            v
            for ctx in cluster.get("emotional_contexts") or []
            for v in _numeric_values(ctx)
        ]
        max_emotion = max(0.0, min(1.0, max(emotions))) if emotions else 0.0
        count = int(cluster.get("count", 1) or 1)
        size_signal = min(
            1.0, count / max(total_thoughts * cfg.size_reference_fraction, 1.0)
        )
        representative = str(cluster.get("representative", ""))
        linked = any(
            "conversation" in str(tid).lower()
            for tid in cluster.get("thought_ids") or []
        )
        relationship = self._relationship_pattern()
        if relationship is not None and relationship.search(representative):
            linked = True
        expressed = list(cluster.get("expressed") or [])
        expressed_score = sum(expressed) / len(expressed) if expressed else 0.0
        score = (
            max_emotion * cfg.weight_emotion
            + size_signal * cfg.weight_size
            + (1.0 if linked else 0.0) * cfg.weight_conversation
            + expressed_score * cfg.weight_expressed
            + (1.0 - size_signal) * cfg.weight_uniqueness
        )
        return min(1.0, max(0.0, score))

    def check_protection(self, cluster: Mapping[str, Any]) -> Tuple[bool, str]:
        """Return ``(protected, reason)`` for a cluster."""
        content = str(cluster.get("representative", ""))
        markers = _phrase_pattern(self.config.markers(self._system_name()))
        if markers is not None and markers.search(content):
            return True, "Identity-defining content"
        for ctx in cluster.get("emotional_contexts") or []:
            for value in _numeric_values(ctx):
                if value > self.config.protect_emotion_above:
                    return True, f"High emotional intensity ({value:.2f})"
        protected = [p for p in self.config.protected_participants if p and p.strip()]
        if protected:
            wanted = {p.strip().lower() for p in protected}
            named = {str(p).strip().lower() for p in cluster.get("participants") or []}
            names = _phrase_pattern(protected)
            if wanted & named or (names is not None and names.search(content)):
                return True, "Protected participant"
        return False, ""

    # ------------------------------------------------------------------
    # Gists
    # ------------------------------------------------------------------

    @staticmethod
    def _emotional_arc(
        contexts: Sequence[Any],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """The dominant emotion and its arc, or ``(None, {})`` when no context
        carries a numeric score: no mood is assumed for unmeasured thoughts.

        The start and end intensities average the first and the last three
        contexts that carry a numeric value for the dominant emotion; a
        context without it is left out, not read as 0.
        """
        sums: Dict[str, float] = {}
        for ctx in contexts:
            if isinstance(ctx, Mapping):
                for emotion, value in ctx.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        sums[emotion] = sums.get(emotion, 0.0) + float(value)
        if not sums:
            return None, {}
        dominant = max(sums, key=sums.get)
        carrying = [
            float(c[dominant])
            for c in contexts
            if isinstance(c, Mapping) and _is_number(c.get(dominant))
        ]
        start = carrying[:3]
        end = carrying[-3:]
        start_avg = sum(start) / len(start) if start else 0.0
        end_avg = sum(end) / len(end) if end else 0.0
        if end_avg > start_avg + 0.1:
            trend = "rising"
        elif end_avg < start_avg - 0.1:
            trend = "falling"
        else:
            trend = "steady"
        return dominant, {
            "start_intensity": round(start_avg, 3),
            "end_intensity": round(end_avg, 3),
            "trend": trend,
        }

    async def _ask(
        self, prompt: str, system_prompt: str, max_tokens: int
    ) -> Optional[str]:
        """The brain's visible answer, or None when there is none.

        A brain failure other than ``NoAnswerError`` is also no answer; its
        description is kept in ``last_brain_error`` for the report.
        """
        self.last_brain_error = None
        try:
            reply = await self.brain.think(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                thinking_mode=False,
            )
        except NoAnswerError as exc:
            logger.info("brain gave no answer: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - reported, nothing composed
            self.last_brain_error = f"{type(exc).__name__}: {exc}"
            logger.warning("brain call failed: %s", self.last_brain_error)
            return None
        text = strip_reasoning_blocks(getattr(reply, "text", "") or "")
        return text or None

    async def _gist_for(
        self, cluster: Mapping[str, Any], total_thoughts: Optional[int] = None
    ) -> Tuple[Optional[ThoughtGist], Optional[str]]:
        if self.brain is None:
            return None, "no_brain"
        count = int(cluster.get("count", 0) or 0)
        sample = str(cluster.get("representative", ""))
        bursts = list(cluster.get("burst_windows") or [])
        dominant, arc = self._emotional_arc(cluster.get("emotional_contexts") or [])
        name = self._system_name()
        if dominant is None:
            tone = (
                "No emotion was measured for these thoughts; add none. Preserve "
                "the temporal shape."
            )
        else:
            tone = (
                f"Dominant emotion: {dominant}. Preserve the emotional tone and "
                "the temporal shape."
            )
        prompt = (
            f"Summarise this recurring pattern in the thinking of {name} in one or "
            f"two sentences. The thought recurred {count} times. "
            f'Content: "{sample}". Temporal pattern: {len(bursts)} burst(s). '
            f"{tone}"
        )
        system_prompt = (
            f"Summarise thought patterns of {name}, an AI research system, in the "
            "third person. Be concise (one or two sentences) and preserve the "
            "emotional tone."
        )
        summary = await self._ask(prompt, system_prompt, self.config.gist_max_tokens)
        if summary is None:
            return None, "no_answer"
        stamps = list(cluster.get("timestamps") or [])
        time_span = (
            {"start": min(stamps).isoformat(), "end": max(stamps).isoformat()}
            if stamps
            else {}
        )
        variations = list(cluster.get("content_variations") or []) or [sample]
        salience = list(cluster.get("salience") or [])
        gist = ThoughtGist(
            gist_id=str(uuid.uuid4()),
            created_at=self._now(),
            summary=summary,
            source_count=count,
            source_thought_ids=list(cluster.get("thought_ids") or []),
            dominant_emotion=dominant,
            emotional_arc=arc,
            time_span=time_span,
            burst_windows=bursts,
            avg_salience=sum(salience) / len(salience) if salience else None,
            total_access_count=0,
            content_variations=variations[: self.config.max_variations],
            importance_score=self.score_cluster_importance(
                cluster, total_thoughts if total_thoughts else max(count, 1)
            ),
        )
        protected, reason = self.check_protection(cluster)
        if protected:
            gist.is_protected = True
            gist.protection_reason = reason
        # Retention at age zero, computed by the forgetting curve, not a default.
        gist.retention_strength = self.compute_gist_retention(gist, gist.created_at)
        return gist, None

    async def generate_gist(
        self, cluster: Mapping[str, Any], total_thoughts: Optional[int] = None
    ) -> Optional[ThoughtGist]:
        """Summarise a cluster through the brain; None without an answer."""
        gist, _reason = await self._gist_for(cluster, total_thoughts)
        return gist

    async def run_thought_consolidation(
        self,
        thoughts: List[Dict[str, Any]],
        min_age_hours: Optional[float] = None,
    ) -> ThoughtConsolidationReport:
        """Cluster old thoughts, protect what must stay, gist the rest.

        The caller deletes ``delete_ids``; this pass stores and embeds the
        gists when the stores are configured, then refreshes stored gist
        retention.
        """
        report = ThoughtConsolidationReport()
        self.last_errors = {}
        age = (
            self.config.min_thought_age_hours
            if min_age_hours is None
            else min_age_hours
        )
        cutoff = self._now() - timedelta(hours=float(age))
        eligible = []
        for thought in thoughts:
            stamp = _as_utc(thought.get("timestamp"))
            if stamp is not None and stamp < cutoff:
                eligible.append(thought)
        report.thoughts_processed = len(eligible)
        if not eligible:
            return report
        clusters = self.cluster_thoughts(eligible)
        report.clusters = len(clusters)
        if self.brain is None:
            report.skipped["gist_generation"] = "no brain"
        protected_ids: List[str] = []
        for cluster in clusters:
            protected, _reason = self.check_protection(cluster)
            if protected:
                protected_ids.extend(cluster["thought_ids"])
                continue
            if self.brain is None:
                continue
            gist, reason = await self._gist_for(cluster, len(eligible))
            if gist is None:
                if reason == "no_answer":
                    report.no_answer += 1
                    if self.last_brain_error:
                        report.errors["brain"] = self.last_brain_error
                continue
            report.gists_created += 1
            report.gists.append(gist)
            if not await self._store_gist(gist, cluster, report):
                continue
            report.delete_ids.extend(cluster["thought_ids"])
        report.protected_count = len(protected_ids)
        report.thoughts_to_delete = len(report.delete_ids)
        if self.gist_store is not None:
            try:
                report.gists_refreshed = await self.refresh_gist_retention()
            except Exception as exc:  # noqa: BLE001 - recorded in the report
                report.errors["gist_retention"] = f"{type(exc).__name__}: {exc}"
        self.last_errors = dict(report.errors)
        return report

    async def _store_gist(
        self,
        gist: ThoughtGist,
        cluster: Mapping[str, Any],
        report: ThoughtConsolidationReport,
    ) -> bool:
        """Store and embed a gist; False when a configured gist store failed."""
        if self.gist_store is not None:
            try:
                await self.gist_store.add_gist(gist)
                report.gists_stored += 1
            except Exception as exc:  # noqa: BLE001 - the cluster stays
                report.errors["gist_store"] = f"{type(exc).__name__}: {exc}"
                logger.warning("gist store failed: %s", exc)
                return False
        if self.vector_store is not None:
            emotions = [
                v
                for ctx in cluster.get("emotional_contexts") or []
                for v in _numeric_values(ctx)
            ]
            try:
                await asyncio.to_thread(
                    self.vector_store.store_memory,
                    text=gist.summary,
                    metadata={
                        "type": "gist",
                        "gist_id": gist.gist_id,
                        "dominant_emotion": gist.dominant_emotion,
                        "source_count": gist.source_count,
                        "importance_score": gist.importance_score,
                    },
                    importance=gist.importance_score,
                    emotional_weight=max(emotions) if emotions else None,
                )
                report.gists_embedded += 1
            except Exception as exc:  # noqa: BLE001 - embedding is optional
                report.errors["vector"] = f"{type(exc).__name__}: {exc}"
                logger.warning("gist embedding failed: %s", exc)
        return True

    # ------------------------------------------------------------------
    # Retention and conflicts
    # ------------------------------------------------------------------

    def _decay(self, age_days: float, half_life_days: float) -> float:
        return math.pow(0.5, max(0.0, age_days) / half_life_days)

    def apply_forgetting_curve(
        self, memories: List[Dict[str, Any]], now: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Add ``retention_strength`` to copies of the memories.

        retention = min(1, max(importance * floor_factor,
        0.5 ** (age / memory_half_life_days) * (1 + access_boost))); importance
        is on a 0 to 1 scale. A memory without an importance has no floor. A
        memory without a readable timestamp has an unknown age: its
        ``retention_strength`` is None and ``retention_unmeasured`` says why,
        so it is never treated as fresh.
        """
        cfg = self.config
        moment = now or self._now()
        results = []
        for memory in memories:
            decayed = dict(memory)
            stamp = _as_utc(memory.get("timestamp"))
            if stamp is None:
                decayed["retention_strength"] = None
                decayed["retention_unmeasured"] = "no timestamp"
                results.append(decayed)
                continue
            age_days = (moment - stamp).total_seconds() / 86400.0
            boost = min(
                int(memory.get("access_count", 0) or 0) * cfg.memory_access_boost,
                cfg.max_access_boost,
            )
            importance = memory.get("importance")
            floor = (
                float(importance) * cfg.importance_floor_factor
                if isinstance(importance, (int, float))
                and not isinstance(importance, bool)
                else 0.0
            )
            retention = max(
                floor, self._decay(age_days, cfg.memory_half_life_days) * (1.0 + boost)
            )
            decayed["retention_strength"] = min(retention, 1.0)
            results.append(decayed)
        return results

    def compute_gist_retention(
        self, gist: Any, now: Optional[datetime] = None
    ) -> Optional[float]:
        """Retention of a gist (a mapping or a ``ThoughtGist``).

        retention = min(1, max(importance * floor_factor,
        0.5 ** (age / gist_half_life_days) * (1 + retrieval_boost))); a
        protected gist never drops below ``protected_gist_floor``. A gist
        without a readable ``created_at`` has an unknown age and returns None.
        """
        cfg = self.config
        doc = gist.to_document() if isinstance(gist, ThoughtGist) else dict(gist)
        moment = now or self._now()
        created = _as_utc(doc.get("created_at"))
        if created is None:
            return None
        age_days = (moment - created).total_seconds() / 86400.0
        boost = min(
            int(doc.get("retrieval_count", 0) or 0) * cfg.gist_access_boost,
            cfg.max_access_boost,
        )
        floor = (
            float(doc.get("importance_score", 0.0) or 0.0) * cfg.importance_floor_factor
        )
        retention = max(
            floor, self._decay(age_days, cfg.gist_half_life_days) * (1.0 + boost)
        )
        if doc.get("is_protected"):
            retention = max(retention, cfg.protected_gist_floor)
        return min(1.0, retention)

    def classify_retention(self, retention: float) -> str:
        """``active``, ``fading`` or ``expired``."""
        if retention >= self.config.gist_active_at:
            return "active"
        if retention >= self.config.gist_fading_at:
            return "fading"
        return "expired"

    async def refresh_gist_retention(self, batch: int = 500) -> int:
        """Recompute and store ``retention_strength`` for every stored gist.

        The pass walks ``ThoughtGistStore.iter_gists`` (``batch`` gists per
        read), never ``active_gists``: a gist stored without a retention is
        never returned under a floor, so only a walk over every gist can
        compute it. The retention store expires gists by this stored value, so
        the pass runs before pruning. Returns the gists updated; a gist whose
        age is unknown keeps its stored value and is counted in ``status()``.
        """
        if self.gist_store is None:
            return 0
        now = self._now()
        updated = 0
        self.last_unmeasured_gists = 0
        async for gist in self.gist_store.iter_gists(batch=batch):
            retention = self.compute_gist_retention(gist, now)
            if retention is None:
                self.last_unmeasured_gists += 1
                continue
            if await self.gist_store.update_gist(
                gist.gist_id, {"retention_strength": retention}
            ):
                updated += 1
        return updated

    def resolve_conflicts(
        self, conflicting_memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prefer the most recent memory, then the most confident among equals.

        Confidence falls by ``conflict_penalty`` per extra version, never below
        ``conflict_min_confidence``.
        """
        if not conflicting_memories:
            return {"resolution": None, "confidence": None, "conflict_history": []}
        floor = datetime.min.replace(tzinfo=timezone.utc)
        ordered = sorted(
            conflicting_memories,
            key=lambda m: (
                _as_utc(m.get("timestamp")) or floor,
                float(m.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
        cfg = self.config
        confidence = max(
            cfg.conflict_min_confidence,
            1.0 - (len(conflicting_memories) - 1) * cfg.conflict_penalty,
        )
        return {
            "resolution": ordered[0].get("content"),
            "confidence": confidence,
            "conflict_history": [m.get("content") for m in ordered],
        }
