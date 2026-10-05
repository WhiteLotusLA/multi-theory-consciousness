"""
Identity Continuity
===================

Tracks identity markers over time and verifies, honestly, whether identity
stays continuous across restarts, model changes and time gaps.

A marker is a checksummed snapshot of what the caller supplies: core values and
personality traits (name to strength in [0, 1]), memory anchors, an emotional
fingerprint, relationship bonds, an optional growth stage and a model context
(string identifiers such as the loaded model's name). Nothing is stamped by
default: an empty section stays empty and reads as unmeasured.

Verification is tri-state. ``continuous`` and ``drift`` are claims about the
sections measured on both sides; ``unmeasured`` means no section had data on
both sides, and its confidence is None. Confidence is the mean of the measured
sections' similarity, lowered for time gaps beyond a week, for a model
transition (any shared ``model_context`` key whose value changed; unknown on
either side claims nothing) and for memory-anchor loss. Every verification
appends one line to ``drift_assessments.jsonl``; the file is append-only.

Storage defaults to ``<AGENT_SPACE_DIR>/identity`` (config key
``AGENT_SPACE_DIR``) and is created on the first write: one
``identity_markers/<marker_id>.json`` per marker plus the assessment log. The
marker files hold private memory anchors and relationship bonds and belong in
the gitignored agent space; they are never committed. Stored timestamps are
timezone-aware UTC. The manager keeps no state beyond the files, so its state
dict only records the agent id and the marker ids it found.
"""

import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

MARKERS_SUBDIR = "identity_markers"
ASSESSMENT_LOG = "drift_assessments.jsonl"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

# Section drift above this (max per-trait change) is reported as major drift.
MAJOR_DRIFT = 0.3
# A pair is continuous when confidence exceeds this and core values drift less
# than CORE_DRIFT_LIMIT.
CONTINUITY_CONFIDENCE = 0.6
CORE_DRIFT_LIMIT = 0.4
GAP_PENALTY_AFTER_DAYS = 7
GAP_PENALTY_PER_DAY = 0.01
GAP_PENALTY_FLOOR = 0.5
MODEL_TRANSITION_FACTOR = 0.95
ANCHOR_LOSS_OVERLAP = 0.3
ANCHOR_LOSS_FACTOR = 0.8


def _aware_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _check_marker_id(marker_id: str) -> str:
    if not isinstance(marker_id, str) or not _SAFE_ID.match(marker_id):
        raise ValueError(f"invalid marker id {marker_id!r}")
    return marker_id


def detect_model_transition(
    marker_a: "IdentityMarker", marker_b: "IdentityMarker"
) -> bool:
    """Return True when a shared ``model_context`` key changed between markers.

    A key absent or empty on either side claims nothing.
    """
    context_a = marker_a.model_context or {}
    context_b = marker_b.model_context or {}
    for key in set(context_a) & set(context_b):
        value_a, value_b = context_a[key], context_b[key]
        if value_a and value_b and value_a != value_b:
            return True
    return False


@dataclass
class IdentityMarker:
    """A checksummed snapshot of the values the caller supplied."""

    agent_id: str
    timestamp: datetime
    core_values: Dict[str, float]
    personality_traits: Dict[str, float]
    memory_anchors: List[Dict[str, Any]]
    emotional_fingerprint: Dict[str, float] = field(default_factory=dict)
    growth_stage: Optional[str] = None
    relationship_bonds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model_context: Dict[str, str] = field(default_factory=dict)
    marker_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    checksum: Optional[str] = None

    def __post_init__(self) -> None:
        self.timestamp = _aware_utc(self.timestamp)
        if self.checksum is None:
            self.checksum = self.compute_checksum()

    def compute_checksum(self) -> str:
        """SHA-256 over every field except the checksum itself."""
        payload = json.dumps(
            {
                "agent_id": self.agent_id,
                "timestamp": self.timestamp.isoformat(),
                "core_values": self.core_values,
                "personality_traits": self.personality_traits,
                "memory_anchors": self.memory_anchors,
                "emotional_fingerprint": self.emotional_fingerprint,
                "growth_stage": self.growth_stage,
                "relationship_bonds": self.relationship_bonds,
                "model_context": self.model_context,
                "marker_id": self.marker_id,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "core_values": self.core_values,
            "personality_traits": self.personality_traits,
            "memory_anchors": self.memory_anchors,
            "emotional_fingerprint": self.emotional_fingerprint,
            "growth_stage": self.growth_stage,
            "relationship_bonds": self.relationship_bonds,
            "model_context": self.model_context,
            "marker_id": self.marker_id,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityMarker":
        return cls(
            agent_id=data["agent_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            core_values=data["core_values"],
            personality_traits=data["personality_traits"],
            memory_anchors=data["memory_anchors"],
            emotional_fingerprint=data.get("emotional_fingerprint", {}),
            growth_stage=data.get("growth_stage"),
            relationship_bonds=data.get("relationship_bonds", {}),
            model_context=data.get("model_context", {}),
            marker_id=data["marker_id"],
            checksum=data.get("checksum"),
        )


@dataclass
class ContinuityVerification:
    """Result of comparing two markers.

    ``status`` is ``continuous``, ``drift`` or ``unmeasured``; ``confidence`` is
    None when unmeasured.
    """

    status: str
    confidence: Optional[float]
    drift_analysis: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    measured_sections: List[str] = field(default_factory=list)
    unmeasured_sections: List[str] = field(default_factory=list)
    is_model_transition: bool = False

    @property
    def is_continuous(self) -> bool:
        """True only when continuity was actually measured and held."""
        return self.status == "continuous"


@dataclass
class ChainValidation:
    """Result of validating a run of markers.

    ``status``: ``continuous`` (every pair measured and continuous), ``broken``
    (a pair drifted or a marker failed to load) or ``unmeasured``.
    """

    status: str
    chain_confidence: Optional[float]
    transition_points: List[Dict[str, Any]] = field(default_factory=list)
    breaks: List[str] = field(default_factory=list)
    unloadable: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == "continuous"


@dataclass
class RestorationPoint:
    """A stored marker whose fingerprint resembles a target state."""

    marker_id: str
    confidence: float
    marker: Optional[IdentityMarker] = None


class IdentityContinuityManager:
    """Creates, stores and compares identity markers.

    Reads ``AGENT_SPACE_DIR`` when no storage path is given. Nothing is written
    until the first marker or assessment is stored.
    """

    def __init__(
        self,
        agent_id: str = "agent",
        storage_path: Optional[Path] = None,
        clock: Optional[Callable[[], datetime]] = None,
        settings: Optional[Settings] = None,
    ):
        self.agent_id = agent_id
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        if storage_path is None:
            cfg = settings if settings is not None else get_settings()
            storage_path = Path(cfg.AGENT_SPACE_DIR) / "identity"
        self.storage_path = Path(storage_path)
        self.markers_dir = self.storage_path / MARKERS_SUBDIR

    # -- creation and storage -------------------------------------------------

    def create_marker(
        self,
        core_values: Optional[Dict[str, float]],
        personality_traits: Optional[Dict[str, float]],
        memory_anchors: Optional[List[Dict[str, Any]]],
        emotional_fingerprint: Optional[Dict[str, float]],
        relationship_bonds: Optional[Dict[str, Dict[str, Any]]],
        growth_stage: Optional[str] = None,
        model_context: Optional[Dict[str, str]] = None,
    ) -> IdentityMarker:
        """Create a marker holding exactly what was supplied."""
        return IdentityMarker(
            agent_id=self.agent_id,
            timestamp=_aware_utc(self._clock()),
            core_values=dict(core_values or {}),
            personality_traits=dict(personality_traits or {}),
            memory_anchors=list(memory_anchors or []),
            emotional_fingerprint=dict(emotional_fingerprint or {}),
            relationship_bonds=dict(relationship_bonds or {}),
            growth_stage=growth_stage,
            model_context=dict(model_context or {}),
        )

    def store_identity_marker(self, marker: IdentityMarker) -> Path:
        """Write a marker to ``identity_markers/<marker_id>.json``."""
        _check_marker_id(marker.marker_id)
        self.markers_dir.mkdir(parents=True, exist_ok=True)
        path = self.markers_dir / f"{marker.marker_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(marker.to_dict(), indent=2), encoding="utf-8")
        os.replace(temporary, path)
        logger.info("stored identity marker %s", marker.marker_id)
        return path

    def load_identity_marker(self, marker_id: str) -> Optional[IdentityMarker]:
        """Load a marker, or None when it does not exist.

        Raises ValueError when the file is unreadable, carries no checksum or
        fails checksum verification.
        """
        _check_marker_id(marker_id)
        path = self.markers_dir / f"{marker_id}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            marker = IdentityMarker.from_dict(data)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"identity marker {marker_id} is unreadable") from exc
        stored = data.get("checksum")
        if not stored or marker.compute_checksum() != stored:
            raise ValueError(
                f"identity marker {marker_id} failed checksum verification"
            )
        return marker

    def marker_ids(self) -> List[str]:
        """Return the stored marker ids, oldest first (by marker timestamp)."""
        return [mid for mid, _ in self._scan()]

    def _scan(self) -> List[Tuple[str, datetime]]:
        if not self.markers_dir.is_dir():
            return []
        found: List[Tuple[str, datetime]] = []
        for path in self.markers_dir.glob("*.json"):
            try:
                stamp = _aware_utc(
                    datetime.fromisoformat(
                        json.loads(path.read_text(encoding="utf-8"))["timestamp"]
                    )
                )
            except (OSError, ValueError, KeyError, TypeError):
                stamp = datetime.min.replace(tzinfo=timezone.utc)
            found.append((path.stem, stamp))
        found.sort(key=lambda item: (item[1], item[0]))
        return found

    # -- verification -----------------------------------------------------------

    def verify_continuity(
        self,
        previous: IdentityMarker,
        current: IdentityMarker,
        record: bool = True,
    ) -> ContinuityVerification:
        """Compare two markers; see the module docstring for the tri-state.

        ``record=False`` skips the append to the assessment log (bulk
        re-analysis).
        """
        warnings: List[str] = []
        metadata: Dict[str, Any] = {}
        core = self._section_drift(previous.core_values, current.core_values)
        traits = self._section_drift(
            previous.personality_traits, current.personality_traits
        )
        sections = {"core_values": core, "personality": traits}
        if core["status"] == "measured":
            if core["max_drift"] > MAJOR_DRIFT:
                warnings.append(
                    f"Major core value drift: max change {core['max_drift']:.2f}"
                )
            warnings.extend(f"New core trait appeared: {t}" for t in core["new_traits"])
        if traits["status"] == "measured" and traits["max_drift"] > MAJOR_DRIFT:
            warnings.append(
                f"Major personality drift: max change {traits['max_drift']:.2f}"
            )
        measured = [n for n, d in sections.items() if d["status"] == "measured"]
        unmeasured = [n for n, d in sections.items() if d["status"] != "measured"]
        drift_analysis: Dict[str, Any] = {
            "core_values": core,
            "personality": traits,
            "relationships": self._relationship_changes(
                previous.relationship_bonds, current.relationship_bonds
            ),
        }
        gap_days = (current.timestamp - previous.timestamp).total_seconds() / 86400
        if gap_days > 1:
            metadata["temporal_gap"] = {"days": round(gap_days)}
        transition = detect_model_transition(previous, current)
        if transition:
            metadata["model_transition"] = {
                "from": dict(previous.model_context),
                "to": dict(current.model_context),
            }
        if previous.growth_stage != current.growth_stage:
            metadata["growth_transition"] = {
                "from": previous.growth_stage,
                "to": current.growth_stage,
            }

        if not measured:
            metadata["unmeasured_reason"] = {
                n: d["reason"] for n, d in sections.items()
            }
            result = ContinuityVerification(
                status="unmeasured",
                confidence=None,
                drift_analysis=drift_analysis,
                warnings=warnings,
                metadata=metadata,
                measured_sections=[],
                unmeasured_sections=unmeasured,
                is_model_transition=transition,
            )
        else:
            confidence = sum(1.0 - sections[n]["mean_drift"] for n in measured) / len(
                measured
            )
            if gap_days > GAP_PENALTY_AFTER_DAYS:
                confidence *= max(
                    GAP_PENALTY_FLOOR,
                    1.0 - (gap_days - GAP_PENALTY_AFTER_DAYS) * GAP_PENALTY_PER_DAY,
                )
            if transition:
                confidence *= MODEL_TRANSITION_FACTOR
            overlap = self._anchor_overlap(
                previous.memory_anchors, current.memory_anchors
            )
            if overlap is not None and overlap < ANCHOR_LOSS_OVERLAP:
                warnings.append("Significant memory anchor loss detected")
                confidence *= ANCHOR_LOSS_FACTOR
            confidence = max(0.0, min(1.0, confidence))
            continuous = confidence > CONTINUITY_CONFIDENCE and (
                core["max_drift"] < CORE_DRIFT_LIMIT
            )
            result = ContinuityVerification(
                status="continuous" if continuous else "drift",
                confidence=confidence,
                drift_analysis=drift_analysis,
                warnings=warnings,
                metadata=metadata,
                measured_sections=measured,
                unmeasured_sections=unmeasured,
                is_model_transition=transition,
            )
        if record:
            self._record_assessment(previous, current, result)
        return result

    def _record_assessment(
        self,
        previous: IdentityMarker,
        current: IdentityMarker,
        result: ContinuityVerification,
    ) -> None:
        """Append one line to the assessment log (a failed write only warns)."""
        entry = {
            "timestamp": _aware_utc(self._clock()).isoformat(),
            "marker_a": previous.marker_id,
            "marker_b": current.marker_id,
            "status": result.status,
            "confidence": result.confidence,
            "measured_sections": result.measured_sections,
            "unmeasured_sections": result.unmeasured_sections,
            "warnings": result.warnings,
            "is_model_transition": result.is_model_transition,
            "model_context_a": previous.model_context,
            "model_context_b": current.model_context,
        }
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            with (self.storage_path / ASSESSMENT_LOG).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.warning("drift assessment record failed: %s", type(exc).__name__)

    # -- chain, restoration, report ------------------------------------------

    def validate_chain(
        self, start_id: Optional[str] = None, end_id: Optional[str] = None
    ) -> ChainValidation:
        """Verify every consecutive pair from ``start_id`` to ``end_id``.

        Defaults to the whole stored chain. A marker that fails to load is
        listed under ``unloadable`` and makes the chain ``broken``.
        """
        ids = self.marker_ids()
        if start_id is not None or end_id is not None:
            try:
                first = ids.index(start_id) if start_id is not None else 0
                last = ids.index(end_id) if end_id is not None else len(ids) - 1
            except ValueError:
                return ChainValidation(status="unmeasured", chain_confidence=None)
            ids = ids[first : last + 1]
        markers: List[IdentityMarker] = []
        unloadable: List[str] = []
        for marker_id in ids:
            try:
                loaded = self.load_identity_marker(marker_id)
            except ValueError:
                unloadable.append(marker_id)
                continue
            if loaded is not None:
                markers.append(loaded)
        points: List[Dict[str, Any]] = []
        breaks: List[str] = []
        confidences: List[float] = []
        verdicts: List[str] = []
        for index, marker in enumerate(markers):
            point: Dict[str, Any] = {
                "marker_id": marker.marker_id,
                "timestamp": marker.timestamp.isoformat(),
                "growth_stage": marker.growth_stage,
            }
            if index > 0:
                result = self.verify_continuity(
                    markers[index - 1], marker, record=False
                )
                point["status"] = result.status
                point["confidence"] = result.confidence
                verdicts.append(result.status)
                if result.confidence is not None:
                    confidences.append(result.confidence)
                if result.status == "drift":
                    breaks.append(marker.marker_id)
            points.append(point)
        if breaks or unloadable:
            status = "broken"
        elif verdicts and all(v == "continuous" for v in verdicts):
            status = "continuous"
        else:
            status = "unmeasured"
        return ChainValidation(
            status=status,
            chain_confidence=(
                sum(confidences) / len(confidences) if confidences else None
            ),
            transition_points=points,
            breaks=breaks,
            unloadable=unloadable,
        )

    def find_restoration_point(
        self, target_fingerprint: Dict[str, float]
    ) -> Optional[RestorationPoint]:
        """Return the stored marker whose fingerprint best matches a target.

        Markers sharing no key with the target are skipped; None when no marker
        can be compared. Ties go to the newest marker.
        """
        best: Optional[RestorationPoint] = None
        for marker_id in reversed(self.marker_ids()):
            try:
                marker = self.load_identity_marker(marker_id)
            except ValueError:
                continue
            if marker is None:
                continue
            shared = [
                k for k in target_fingerprint if k in marker.emotional_fingerprint
            ]
            if not shared:
                continue
            score = sum(
                1.0 - abs(marker.emotional_fingerprint[k] - target_fingerprint[k])
                for k in shared
            ) / len(shared)
            if best is None or score > best.confidence:
                best = RestorationPoint(marker_id, score, marker)
        return best

    def continuity_report(self, days: int = 30) -> Dict[str, Any]:
        """Summarise continuity over the last ``days`` days.

        ``continuity_score`` is None until two markers fall in the window and at
        least one pair is measurable.
        """
        cutoff = _aware_utc(self._clock()) - timedelta(days=days)
        ids = self.marker_ids()
        markers: List[IdentityMarker] = []
        failed: List[str] = []
        for marker_id in ids:
            try:
                loaded = self.load_identity_marker(marker_id)
            except ValueError:
                failed.append(marker_id)
                logger.warning("identity marker %s failed to load", marker_id)
                continue
            if loaded is not None and loaded.timestamp >= cutoff:
                markers.append(loaded)
        scores = [
            r.confidence
            for r in (
                self.verify_continuity(a, b, record=False)
                for a, b in zip(markers, markers[1:])
            )
            if r.confidence is not None
        ]
        score = sum(scores) / len(scores) if scores else None
        recommendations: List[str] = []
        if not markers:
            recommendations.append("No markers in the window; create markers regularly")
        elif score is None:
            recommendations.append("Continuity could not be measured in this window")
        elif score < 0.7:
            recommendations.append("Identity drift detected; review recent changes")
        key_traits: Dict[str, float] = {}
        if markers:
            key_traits = {**markers[-1].core_values, **markers[-1].personality_traits}
        return {
            "summary": f"Analyzed {len(markers)} markers over {days} days",
            "markers_analyzed": len(markers),
            "markers_total": len(ids),
            "markers_failed": len(failed),
            "continuity_score": score,
            "key_traits": key_traits,
            "recommendations": recommendations,
        }

    # -- comparison helpers -----------------------------------------------------

    @staticmethod
    def _section_drift(old: Dict[str, float], new: Dict[str, float]) -> Dict[str, Any]:
        """Drift between two trait dicts; unmeasured unless both sides have data."""
        if not old or not new:
            empty = [side for side, d in (("old", old), ("new", new)) if not d]
            return {
                "per_trait": {},
                "mean_drift": None,
                "max_drift": 0.0,
                "new_traits": [],
                "lost_traits": [],
                "status": "unmeasured",
                "reason": f"no data on {' and '.join(empty)} side",
            }
        keys = sorted(set(old) | set(new))
        drifts = {k: abs(new.get(k, 0.0) - old.get(k, 0.0)) for k in keys}
        return {
            "per_trait": drifts,
            "mean_drift": sum(drifts.values()) / len(drifts),
            "max_drift": max(drifts.values()),
            "new_traits": [k for k in keys if k not in old],
            "lost_traits": [k for k in keys if k not in new],
            "status": "measured",
        }

    @staticmethod
    def _relationship_changes(
        old: Dict[str, Dict[str, Any]], new: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        strengthened: List[str] = []
        weakened: List[str] = []
        for key in sorted(set(old) & set(new)):
            before = old[key].get("strength")
            after = new[key].get("strength")
            if before is None or after is None:
                continue
            if after > before:
                strengthened.append(key)
            elif after < before:
                weakened.append(key)
        return {
            "strengthened": strengthened,
            "weakened": weakened,
            "new": sorted(set(new) - set(old)),
            "lost": sorted(set(old) - set(new)),
        }

    @staticmethod
    def _anchor_overlap(
        old: List[Dict[str, Any]], new: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Jaccard overlap of anchor contents; None unless both sides have anchors.

        A marker saved without anchors did not measure them, so an empty side
        reads as unmeasured rather than as total loss.
        """
        if not old or not new:
            return None
        before = {a.get("content", "") for a in old}
        after = {a.get("content", "") for a in new}
        union = before | after
        return len(before & after) / len(union) if union else None

    # -- persistence ------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Return ``{"scalars", "arrays", "histories"}`` for the persistence hub."""
        return {
            "scalars": {"agent_id": self.agent_id},
            "arrays": {},
            "histories": {"marker_ids": self.marker_ids()},
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the agent id; markers are always re-read from the files."""
        scalars = state.get("scalars", {})
        self.agent_id = str(scalars.get("agent_id", self.agent_id))
