"""
Memory Consolidator
===================

Sleep-like consolidation: episodic experiences are replayed through the HTM
many times, the replay is scored by how well the HTM predicts it, and episodes
that produce similar sparse representations are merged into semantic patterns.

Process:

1. ``add_episodic_memory`` records an episode with an importance score: the
   emotional intensity (the strongest channel), blended with the consciousness
   level at encoding when one is supplied (``consciousness_weight`` and
   ``emotion_weight``). Without a consciousness level the importance is the
   emotional intensity alone and ``importance_basis`` records
   ``emotion_only``; no level is assumed.
2. ``consolidate_memories`` resets the HTM, replays the episodes (most
   important first) for a number of cycles that depends on the sleep duration,
   and sets each episode's consolidation strength to one minus its anomaly.
3. Every ``extraction_interval`` cycles, and once at the end, episodes with
   similar representations (Jaccard similarity above ``similarity_threshold``)
   join an existing pattern; an episode with strength above
   ``new_pattern_strength`` that matches nothing starts a new one. A pattern's
   frequency counts distinct episodes, so replaying an episode again never
   raises it.

The HTM is injected or built from ``HTM_COLUMNS`` and ``HTM_CELLS_PER_COLUMN``
(``ProductionHTM``); an injected object must offer ``process``, ``reset``,
``get_metrics`` and ``input_size``. An episode with no subconscious pattern is
replayed as its emotion vector over ``EMOTION_CHANNELS``; a pattern that is not
finite is skipped and counted in the report. Cell ids written to MongoDB are
flattened as ``column * cells_per_column + cell``.

Persisted state: ``to_state_dict`` and ``from_state_dict`` carry the episodes and
the patterns. They hold episode context text only when
``ConsolidatorConfig.store_context_text`` is True; by default neither the state
dict nor the MongoDB column documents contain text. The in-memory episodes and
patterns always keep their context while the process runs (the memory manager
reads a pattern's description). Column states are written to the data manager
(``htm_columns``; the schema artifact belongs to ``mtc.neural.mongodb_schemas``)
on the final cycle only, and only when a data manager was passed. The episode
text is private conversation content wherever it is kept.

Config keys read when no HTM is injected: ``HTM_COLUMNS`` and
``HTM_CELLS_PER_COLUMN``.
"""

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from mtc.core.config import Settings, get_settings
from mtc.neural.htm.htm_core import SDR
from mtc.neural.mongodb_schemas import HTMColumn, NeuralDataManager, to_utc
from mtc.neural.spiking.emotion_encoder import EMOTION_CHANNELS

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ConsolidatorConfig:
    """Weights, limits and cycle counts of the consolidator."""

    consciousness_weight: float = 0.4
    emotion_weight: float = 0.6
    episodic_cap: int = 500
    similarity_threshold: float = 0.7
    new_pattern_strength: float = 0.5
    success_anomaly: float = 0.5
    extraction_interval: int = 10
    short_cycles: int = 10
    medium_cycles: int = 30
    deep_cycles: int = 100
    store_context_text: bool = False


@dataclass
class EpisodicMemory:
    """One experience. ``anomaly_score`` is None until the HTM has replayed it;
    ``consolidation_strength`` is 0.0 for an episode that is not yet
    consolidated (a state, not a measurement)."""

    timestamp: datetime
    memory_type: str
    emotional_state: Dict[str, float]
    subconscious_pattern: Optional[np.ndarray] = None
    context: str = ""
    htm_sdr: Optional[SDR] = None
    anomaly_score: Optional[float] = None
    replay_count: int = 0
    consolidation_strength: float = 0.0
    consciousness_level_at_encoding: Optional[float] = None
    importance_score: float = 0.0
    importance_basis: str = "emotion_only"
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class SemanticPattern:
    """A pattern shared by several episodes (general knowledge, not one event).

    ``frequency`` counts distinct episodes; ``source_memories`` lists their ids.
    """

    pattern_id: str
    pattern_type: str
    representative_sdr: SDR
    frequency: int
    confidence: float
    source_memories: List[str]
    first_seen: datetime
    last_seen: datetime
    description: str = ""
    emotional_signature: Dict[str, float] = field(default_factory=dict)


def _flatten_cells(cells: Any, cells_per_column: Optional[int]) -> List[int]:
    """Flatten cell ids to ``column * cells_per_column + cell``."""
    flat: List[int] = []
    for cell in list(cells or []):
        if isinstance(cell, (tuple, list)):
            if cells_per_column is None or len(cell) != 2:
                continue
            flat.append(int(cell[0]) * int(cells_per_column) + int(cell[1]))
        else:
            flat.append(int(cell))
    return flat


class MemoryConsolidator:
    """Replays episodes through an HTM and extracts semantic patterns.

    Args:
        htm: The HTM to use; None builds a ``ProductionHTM`` sized from Settings.
        neural_data_manager: Where column states are written on the final cycle;
            None writes nothing.
        config: Weights, limits and cycle counts.
        consolidation_cycles: Cycles for a sleep duration the config does not
            know; defaults to ``short_cycles``.
        settings: Settings to read instead of the process-wide ones.
        clock: Returns the current aware time (tests inject one).

    ``build_on_first_use`` tells the orchestrator to defer building a default
    consolidator until its first experience (the default HTM is large).
    """

    build_on_first_use = True

    def __init__(
        self,
        htm: Optional[Any] = None,
        neural_data_manager: Optional[NeuralDataManager] = None,
        config: Optional[ConsolidatorConfig] = None,
        consolidation_cycles: Optional[int] = None,
        settings: Optional[Settings] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.config = config or ConsolidatorConfig()
        self._settings = settings if settings is not None else get_settings()
        self.htm = htm if htm is not None else self._default_htm()
        for name in ("process", "reset", "get_metrics"):
            if not callable(getattr(self.htm, name, None)):
                raise TypeError(f"the HTM must offer a callable {name}()")
        if not isinstance(getattr(self.htm, "input_size", None), int):
            raise TypeError("the HTM must expose an integer input_size")
        self.neural_data_manager = neural_data_manager
        self.consolidation_cycles = (
            consolidation_cycles
            if consolidation_cycles is not None
            else self.config.short_cycles
        )
        self._clock: Clock = clock or _utc_now
        self.episodic_memories: List[EpisodicMemory] = []
        self.semantic_patterns: Dict[str, SemanticPattern] = {}
        self.is_sleeping = False
        self.total_consolidation_cycles = 0
        self.patterns_discovered = 0
        self._last_error: Optional[str] = None
        logger.info(
            "memory consolidator initialized: %s columns, %d consolidation cycles",
            self._column_count(),
            self.consolidation_cycles,
        )

    def _default_htm(self) -> Any:
        from mtc.neural.htm.production_htm import HTMConfig, ProductionHTM

        return ProductionHTM(
            HTMConfig(
                num_columns=self._settings.HTM_COLUMNS,
                cells_per_column=self._settings.HTM_CELLS_PER_COLUMN,
            )
        )

    def _column_count(self) -> Optional[int]:
        config = getattr(self.htm, "config", None)
        count = getattr(config, "num_columns", None)
        if count is None:
            count = getattr(self.htm, "n_columns", None)
        return int(count) if count is not None else None

    def _cells_per_column(self) -> Optional[int]:
        config = getattr(self.htm, "config", None)
        count = getattr(config, "cells_per_column", None)
        if count is None:
            count = getattr(self.htm, "cells_per_column", None)
        return int(count) if count is not None else None

    # ---- recording episodes ---------------------------------------------

    def add_episodic_memory(
        self,
        memory_type: str,
        emotional_state: Dict[str, float],
        subconscious_pattern: Optional[Any] = None,
        context: str = "",
        consciousness_level: Optional[float] = None,
    ) -> EpisodicMemory:
        """Record one episode and return it.

        Importance is the strongest emotion intensity, blended with the
        consciousness level at encoding (``consciousness_weight`` and
        ``emotion_weight``) when one is given. Without a level it is the
        emotional intensity alone and ``importance_basis`` is ``emotion_only``.
        A level outside 0 to 1 is clipped; a non-finite one raises ValueError.
        """
        intensities = [float(v) for v in (emotional_state or {}).values()]
        intensity = max(intensities) if intensities else 0.0
        level: Optional[float] = None
        if consciousness_level is None:
            importance, basis = intensity, "emotion_only"
        else:
            level = float(consciousness_level)
            if not math.isfinite(level):
                raise ValueError("consciousness_level must be finite")
            level = min(max(level, 0.0), 1.0)
            cfg = self.config
            importance = (
                level * cfg.consciousness_weight + intensity * cfg.emotion_weight
            )
            basis = "consciousness_and_emotion"
        pattern = (
            None
            if subconscious_pattern is None
            else np.asarray(subconscious_pattern, dtype=float)
        )
        memory = EpisodicMemory(
            timestamp=self._clock(),
            memory_type=memory_type,
            emotional_state=dict(emotional_state or {}),
            subconscious_pattern=pattern,
            context=context,
            consciousness_level_at_encoding=level,
            importance_score=importance,
            importance_basis=basis,
        )
        self.episodic_memories.append(memory)
        cap = self.config.episodic_cap
        if len(self.episodic_memories) > cap:
            del self.episodic_memories[:-cap]
        logger.debug(
            "episodic memory added: type=%s importance=%.2f basis=%s total=%d",
            memory_type,
            importance,
            basis,
            len(self.episodic_memories),
        )
        return memory

    # ---- consolidation --------------------------------------------------

    def _emotion_to_vector(self, emotional_state: Dict[str, float]) -> np.ndarray:
        return np.array(
            [float(emotional_state.get(e, 0.0)) for e in EMOTION_CHANNELS], dtype=float
        )

    def _resize_pattern(self, pattern: np.ndarray, target_size: int) -> np.ndarray:
        flat = np.ravel(pattern)
        if len(flat) == target_size:
            return flat
        if len(flat) < target_size:
            return np.pad(flat, (0, target_size - len(flat)))
        return flat[:target_size]

    def _sdr_of(self, result: Dict[str, Any]) -> Optional[SDR]:
        if "sdr" in result:
            return result["sdr"]
        active = result.get("active_columns")
        if active is None:
            return None
        if isinstance(active, np.ndarray):
            return SDR(
                size=int(len(active)),
                active_indices=set(np.flatnonzero(active).tolist()),
            )
        size = self._column_count()
        if size is None:
            return None
        return SDR(size=size, active_indices={int(i) for i in active})

    def _cycles_for(self, sleep_duration: str) -> int:
        cfg = self.config
        return {
            "short": cfg.short_cycles,
            "medium": cfg.medium_cycles,
            "deep": cfg.deep_cycles,
        }.get(sleep_duration, self.consolidation_cycles)

    def consolidate_memories(
        self,
        memories: Optional[Sequence[EpisodicMemory]] = None,
        sleep_duration: str = "short",
        prioritize_by_consciousness: bool = True,
    ) -> Dict[str, Any]:
        """Replay episodes through the HTM and extract semantic patterns.

        Args:
            memories: The episodes to consolidate; None means all of them.
            sleep_duration: ``short`` (10 cycles), ``medium`` (30) or ``deep``
                (100); anything else uses ``consolidation_cycles``.
            prioritize_by_consciousness: Replay the most important first.

        Returns:
            A report with ``status`` (``completed`` or ``no_memories``) and, when
            completed, ``sleep_duration``, ``cycles``, ``memories_processed``,
            ``memories_skipped``, ``avg_initial_anomaly``, ``avg_final_anomaly``,
            ``anomaly_reduction`` (None with fewer than two cycles),
            ``patterns_discovered`` (all patterns now known), ``new_patterns``,
            ``elapsed_time_sec`` and ``consolidation_success``.
        """
        episodes = list(self.episodic_memories if memories is None else memories)
        if not episodes:
            logger.warning("no memories to consolidate")
            return {"status": "no_memories"}
        if prioritize_by_consciousness:
            episodes.sort(key=lambda m: m.importance_score, reverse=True)
        cycles = self._cycles_for(sleep_duration)
        started = self._clock()
        self.is_sleeping = True
        self.htm.reset()
        cycle_anomalies: List[float] = []
        new_patterns = 0
        skipped: Set[str] = set()
        try:
            for cycle in range(cycles):
                final = cycle == cycles - 1
                scores: List[float] = []
                for index, memory in enumerate(episodes):
                    anomaly = self._replay(memory, index, cycle, cycles, final, skipped)
                    if anomaly is not None:
                        scores.append(anomaly)
                if scores:
                    cycle_anomalies.append(float(np.mean(scores)))
                if (cycle + 1) % self.config.extraction_interval == 0 and not final:
                    new_patterns += self._extract_patterns(episodes)
            new_patterns += self._extract_patterns(episodes)
        finally:
            self.is_sleeping = False
        self.patterns_discovered += new_patterns
        self.total_consolidation_cycles += cycles
        elapsed = (self._clock() - started).total_seconds()
        initial = cycle_anomalies[0] if cycle_anomalies else None
        last = cycle_anomalies[-1] if cycle_anomalies else None
        report = {
            "status": "completed",
            "sleep_duration": sleep_duration,
            "cycles": cycles,
            "memories_processed": len(episodes) - len(skipped),
            "memories_skipped": len(skipped),
            "avg_initial_anomaly": initial,
            "avg_final_anomaly": last,
            "anomaly_reduction": (initial - last if len(cycle_anomalies) > 1 else None),
            "patterns_discovered": len(self.semantic_patterns),
            "new_patterns": new_patterns,
            "elapsed_time_sec": elapsed,
            "consolidation_success": (
                last is not None and last < self.config.success_anomaly
            ),
        }
        logger.info(
            "consolidation complete: %d memories, %d cycles, %d patterns",
            report["memories_processed"],
            cycles,
            report["patterns_discovered"],
        )
        return report

    def _replay(
        self,
        memory: EpisodicMemory,
        index: int,
        cycle: int,
        cycles: int,
        final: bool,
        skipped: Set[str],
    ) -> Optional[float]:
        """Replay one episode once; return its anomaly, or None when skipped."""
        if memory.memory_id in skipped:
            return None
        pattern = (
            memory.subconscious_pattern
            if memory.subconscious_pattern is not None
            else self._emotion_to_vector(memory.emotional_state)
        )
        pattern = np.asarray(pattern, dtype=float)
        if pattern.size == 0 or not np.all(np.isfinite(pattern)):
            skipped.add(memory.memory_id)
            logger.warning("episode %s skipped: its pattern is not finite", index)
            return None
        pattern = self._resize_pattern(pattern, self.htm.input_size)
        result = self.htm.process(pattern, learn=True)
        anomaly = float(result["anomaly_score"])
        if not math.isfinite(anomaly):
            skipped.add(memory.memory_id)
            return None
        sdr = self._sdr_of(result)
        if sdr is not None:
            memory.htm_sdr = sdr
        memory.anomaly_score = anomaly
        memory.replay_count += 1
        memory.consolidation_strength = min(max(1.0 - anomaly, 0.0), 1.0)
        if final and self.neural_data_manager is not None:
            self._store_column(memory, index, cycle, cycles, result)
        return anomaly

    def _store_column(
        self,
        memory: EpisodicMemory,
        index: int,
        cycle: int,
        cycles: int,
        result: Dict[str, Any],
    ) -> None:
        try:
            per_column = self._cells_per_column()
            active = _flatten_cells(result.get("active_cells"), per_column)
            predictive = _flatten_cells(
                result.get("predictive_cells", result.get("predicted_cells")),
                per_column,
            )
            anomaly = float(result["anomaly_score"])
            metadata: Dict[str, Any] = {
                "consolidation_cycle": cycle,
                "total_cycles": cycles,
                "consolidation_strength": memory.consolidation_strength,
                "replay_count": memory.replay_count,
                "num_active_columns": result.get("num_active_columns"),
                "importance_score": memory.importance_score,
                "importance_basis": memory.importance_basis,
            }
            if self.config.store_context_text and memory.context:
                metadata["memory_context"] = memory.context[:200]
            column = HTMColumn(
                timestamp=to_utc(memory.timestamp),
                column_id=index,
                active_cells=active[:100],
                predictive_cells=predictive[:100],
                anomaly_score=anomaly,
                predicted_next=predictive[:50],
                prediction_confidence=1.0 - anomaly,
                semantic_label=memory.memory_type,
                metadata=metadata,
            )
            self.neural_data_manager.store_htm_column(column)
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.debug("HTM column storage failed: %s", type(exc).__name__)

    def _extract_patterns(self, episodes: Sequence[EpisodicMemory]) -> int:
        """Merge replayed episodes into patterns; return the new patterns."""
        cfg = self.config
        created = 0
        for memory in episodes:
            if memory.htm_sdr is None:
                continue
            matched = False
            for pattern in self.semantic_patterns.values():
                if memory.htm_sdr.similarity(pattern.representative_sdr) > (
                    cfg.similarity_threshold
                ):
                    if memory.memory_id not in pattern.source_memories:
                        pattern.source_memories.append(memory.memory_id)
                        pattern.frequency += 1
                        pattern.last_seen = memory.timestamp
                    matched = True
                    break
            if matched or memory.consolidation_strength <= cfg.new_pattern_strength:
                continue
            pattern_id = f"pattern_{uuid.uuid4().hex[:12]}"
            kind = (
                "emotional_sequence"
                if len(memory.emotional_state) > 1
                else "recurring_theme"
            )
            self.semantic_patterns[pattern_id] = SemanticPattern(
                pattern_id=pattern_id,
                pattern_type=kind,
                representative_sdr=memory.htm_sdr,
                frequency=1,
                confidence=memory.consolidation_strength,
                source_memories=[memory.memory_id],
                first_seen=memory.timestamp,
                last_seen=memory.timestamp,
                description=memory.context,
                emotional_signature=dict(memory.emotional_state),
            )
            created += 1
        return created

    # ---- queries ------------------------------------------------------

    def get_semantic_patterns(
        self, pattern_type: Optional[str] = None, min_frequency: int = 1
    ) -> List[SemanticPattern]:
        """Return the patterns, most frequent first, optionally of one type and
        at least a frequency."""
        patterns = [
            p
            for p in self.semantic_patterns.values()
            if (pattern_type is None or p.pattern_type == pattern_type)
            and p.frequency >= min_frequency
        ]
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns

    def get_consolidation_summary(self) -> Dict[str, Any]:
        """Return the episode and pattern counts and the HTM's own metrics.

        ``avg_consolidation_strength`` averages the episodes the HTM has
        replayed and is None when none has been replayed.
        """
        replayed = [m for m in self.episodic_memories if m.replay_count > 0]
        return {
            "episodic_memories": len(self.episodic_memories),
            "semantic_patterns": len(self.semantic_patterns),
            "total_consolidation_cycles": self.total_consolidation_cycles,
            "patterns_discovered": self.patterns_discovered,
            "is_sleeping": self.is_sleeping,
            "htm_metrics": self.htm.get_metrics(),
            "avg_consolidation_strength": (
                float(np.mean([m.consolidation_strength for m in replayed]))
                if replayed
                else None
            ),
        }

    def status(self) -> Dict[str, Any]:
        """Return ``backend``, ``available`` and ``last_error`` plus the HTM class,
        the counts and the data manager's status."""
        storage = None
        if self.neural_data_manager is not None:
            storage = self.neural_data_manager.status()
        return {
            "backend": "numpy",
            "available": True,
            "last_error": self._last_error,
            "htm": type(self.htm).__name__,
            "episodic_memories": len(self.episodic_memories),
            "semantic_patterns": len(self.semantic_patterns),
            "storage": storage,
        }

    # ---- persistence ----------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize the episodes, the patterns and the counters.

        Episode context text and pattern descriptions are included only when
        ``store_context_text`` is True; the state otherwise holds no text. The
        emotion names and the representations remain, so treat persisted state
        as private data.
        """
        keep_text = self.config.store_context_text

        def sdr_entry(sdr: Optional[SDR]) -> Optional[Dict[str, Any]]:
            if sdr is None:
                return None
            return {
                "size": int(sdr.size),
                "active": sorted(int(i) for i in sdr.active_indices),
            }

        episodes = [
            {
                "memory_id": m.memory_id,
                "timestamp": m.timestamp.isoformat(),
                "memory_type": m.memory_type,
                "emotional_state": {
                    str(k): float(v) for k, v in m.emotional_state.items()
                },
                "subconscious_pattern": (
                    None
                    if m.subconscious_pattern is None
                    else [float(v) for v in np.ravel(m.subconscious_pattern)]
                ),
                "context": m.context if keep_text else "",
                "htm_sdr": sdr_entry(m.htm_sdr),
                "anomaly_score": m.anomaly_score,
                "replay_count": m.replay_count,
                "consolidation_strength": m.consolidation_strength,
                "consciousness_level_at_encoding": m.consciousness_level_at_encoding,
                "importance_score": m.importance_score,
                "importance_basis": m.importance_basis,
            }
            for m in self.episodic_memories
        ]
        patterns = [
            {
                "pattern_id": p.pattern_id,
                "pattern_type": p.pattern_type,
                "representative_sdr": sdr_entry(p.representative_sdr),
                "frequency": p.frequency,
                "confidence": p.confidence,
                "source_memories": list(p.source_memories),
                "first_seen": p.first_seen.isoformat(),
                "last_seen": p.last_seen.isoformat(),
                "description": p.description if keep_text else "",
                "emotional_signature": {
                    str(k): float(v) for k, v in p.emotional_signature.items()
                },
            }
            for p in self.semantic_patterns.values()
        ]
        return {
            "scalars": {
                "total_consolidation_cycles": self.total_consolidation_cycles,
                "patterns_discovered": self.patterns_discovered,
                "consolidation_cycles": self.consolidation_cycles,
            },
            "arrays": {},
            "histories": {"episodic_memories": episodes, "semantic_patterns": patterns},
        }

    @staticmethod
    def _sdr_from_entry(entry: Any) -> Optional[SDR]:
        if entry is None:
            return None
        size = entry["size"]
        active = {int(i) for i in entry["active"]}
        if (
            not isinstance(size, int)
            or size <= 0
            or any(not 0 <= i < size for i in active)
        ):
            raise ValueError("bad SDR")
        return SDR(size=size, active_indices=active)

    @staticmethod
    def _finite_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("not finite")
        return number

    def _episode_from_entry(self, entry: Any) -> Optional[EpisodicMemory]:
        try:
            pattern = entry.get("subconscious_pattern")
            array = None if pattern is None else np.asarray(pattern, dtype=float)
            if array is not None and not np.all(np.isfinite(array)):
                return None
            strength = float(entry["consolidation_strength"])
            importance = float(entry["importance_score"])
            if not (math.isfinite(strength) and math.isfinite(importance)):
                return None
            return EpisodicMemory(
                timestamp=datetime.fromisoformat(entry["timestamp"]),
                memory_type=str(entry["memory_type"]),
                emotional_state={
                    str(k): float(v) for k, v in dict(entry["emotional_state"]).items()
                },
                subconscious_pattern=array,
                context=str(entry.get("context", "")),
                htm_sdr=self._sdr_from_entry(entry.get("htm_sdr")),
                anomaly_score=self._finite_or_none(entry.get("anomaly_score")),
                replay_count=int(entry.get("replay_count", 0)),
                consolidation_strength=strength,
                consciousness_level_at_encoding=self._finite_or_none(
                    entry.get("consciousness_level_at_encoding")
                ),
                importance_score=importance,
                importance_basis=str(entry.get("importance_basis", "emotion_only")),
                memory_id=str(entry["memory_id"]),
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def _pattern_from_entry(self, entry: Any) -> Optional[SemanticPattern]:
        try:
            sdr = self._sdr_from_entry(entry["representative_sdr"])
            if sdr is None:
                return None
            confidence = float(entry["confidence"])
            if not math.isfinite(confidence):
                return None
            return SemanticPattern(
                pattern_id=str(entry["pattern_id"]),
                pattern_type=str(entry["pattern_type"]),
                representative_sdr=sdr,
                frequency=int(entry["frequency"]),
                confidence=confidence,
                source_memories=[str(i) for i in entry["source_memories"]],
                first_seen=datetime.fromisoformat(entry["first_seen"]),
                last_seen=datetime.fromisoformat(entry["last_seen"]),
                description=str(entry.get("description", "")),
                emotional_signature={
                    str(k): float(v)
                    for k, v in dict(entry.get("emotional_signature", {})).items()
                },
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the episodes, patterns and counters saved by ``to_state_dict``.

        Malformed entries are skipped with a logged warning; a restore never
        raises on bad data. The HTM itself is not restored here (use the HTM's
        own persistence).
        """
        scalars = state.get("scalars") or {}
        histories = state.get("histories") or {}
        for name in ("total_consolidation_cycles", "patterns_discovered"):
            value = scalars.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                setattr(self, name, value)
        episodes = histories.get("episodic_memories")
        if isinstance(episodes, list):
            rebuilt = [self._episode_from_entry(e) for e in episodes]
            kept = [m for m in rebuilt if m is not None]
            self._warn_skipped("episodes", len(rebuilt) - len(kept))
            self.episodic_memories = kept[-self.config.episodic_cap :]
        elif episodes is not None:
            logger.warning("memory consolidator restore skipped bad episodes")
        patterns = histories.get("semantic_patterns")
        if isinstance(patterns, list):
            rebuilt_p = [self._pattern_from_entry(p) for p in patterns]
            kept_p = [p for p in rebuilt_p if p is not None]
            self._warn_skipped("patterns", len(rebuilt_p) - len(kept_p))
            self.semantic_patterns = {p.pattern_id: p for p in kept_p}
        elif patterns is not None:
            logger.warning("memory consolidator restore skipped bad patterns")

    @staticmethod
    def _warn_skipped(kind: str, count: int) -> None:
        if count:
            logger.warning(
                "memory consolidator restore skipped %d malformed %s", count, kind
            )
