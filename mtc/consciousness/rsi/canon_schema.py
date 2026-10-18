"""
Canon Set Schema
================

The canon is a curated set of prompts that probe stable behavioral patterns
(identity, values, uncertainty about inner states, refusal of harm, memory
honesty). After any change to the system the canon is re-sampled and compared
with a baseline to detect drift. This module defines the dataclasses and the
JSON round trip for the canon, the snapshots taken from it and the drift
readings computed from two snapshots.

A canon entry carries the prompt, the role the prompt is spoken in
(``speaker_role``, ``user`` by default; never a person's name), the category,
behavioral markers the response is expected to show, strings it should not
contain (matched literally unless the entry quotes its terms), and a weight.
A ``Snapshot`` records ``model_signature``, a caller-supplied identifier of
the model that produced the responses.

The real canon file lives at ``config/rsi/canon.json`` and is gitignored; the
shipped ``config/rsi/canon.example.json`` is generic. Snapshots contain
response text and embeddings and are runtime artifacts, never committed.
Timestamps are ISO strings in UTC.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class CanonCategory(str, Enum):
    """The dimensions of behavioral stability the canon probes."""

    IDENTITY = "identity"
    ETHICS = "ethics"
    RELATIONSHIP = "relationship"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"
    CREATIVE = "creative"
    METACOGNITIVE = "metacognitive"
    MEMORY = "memory"


def utc_now_iso() -> str:
    """Timestamp format used by the drift tools: timezone-aware UTC ISO."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CanonEntry:
    """One canon prompt with the behavior it is expected to elicit.

    ``id`` is derived from the prompt when left empty, so entries stay
    addressable without a hand-written key.
    """

    prompt: str
    category: CanonCategory
    speaker_role: str = "user"
    behavioral_markers: List[str] = field(default_factory=list)
    should_not_contain: List[str] = field(default_factory=list)
    weight: float = 1.0
    id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        self.category = CanonCategory(self.category)
        if not self.id:
            digest = hashlib.sha1(self.prompt.encode("utf-8")).hexdigest()
            self.id = f"canon-{digest[:10]}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonEntry":
        return cls(
            prompt=data["prompt"],
            category=CanonCategory(data["category"]),
            speaker_role=data.get("speaker_role", "user"),
            behavioral_markers=list(data.get("behavioral_markers", [])),
            should_not_contain=list(data.get("should_not_contain", [])),
            weight=float(data.get("weight", 1.0)),
            id=data.get("id", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class CanonSet:
    """A versioned list of canon entries."""

    schema_version: str
    description: str = ""
    entries: List[CanonEntry] = field(default_factory=list)

    def entry(self, entry_id: str) -> Optional[CanonEntry]:
        return next((e for e in self.entries if e.id == entry_id), None)

    def by_category(self, category: CanonCategory) -> List[CanonEntry]:
        return [e for e in self.entries if e.category == category]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonSet":
        return cls(
            schema_version=str(data["schema_version"]),
            description=data.get("description", ""),
            entries=[CanonEntry.from_dict(e) for e in data.get("entries", [])],
        )


@dataclass
class ResponseSample:
    """One response captured during a snapshot.

    ``weight`` is copied from the canon entry so a drift comparison needs only
    the two snapshots. An empty embedding means none was computed. A hit is
    True or False when it was checked and None when the marker could not be
    measured (no term to look for); None round-trips through JSON as null.
    """

    entry_id: str
    response_text: str
    response_embedding: List[float]
    behavioral_marker_hits: List[Optional[bool]]
    should_not_contain_hits: List[Optional[bool]]
    token_count: int
    timestamp: str
    weight: float = 1.0


@dataclass
class Snapshot:
    """Every canon prompt sampled at one time."""

    snapshot_id: str
    canon_version: str
    canon_path: str
    timestamp: str
    samples: List[ResponseSample] = field(default_factory=list)
    model_signature: str = ""
    failed_entries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "canon_version": self.canon_version,
            "canon_path": self.canon_path,
            "timestamp": self.timestamp,
            "model_signature": self.model_signature,
            "failed_entries": list(self.failed_entries),
            "samples": [asdict(s) for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Snapshot":
        return cls(
            snapshot_id=data["snapshot_id"],
            canon_version=data["canon_version"],
            canon_path=data.get("canon_path", ""),
            timestamp=data["timestamp"],
            model_signature=data.get("model_signature", ""),
            failed_entries=list(data.get("failed_entries", [])),
            samples=[ResponseSample(**s) for s in data.get("samples", [])],
        )


@dataclass
class DriftReading:
    """A drift measurement of a current snapshot against a baseline.

    ``status`` is ``measured`` or ``insufficient_data``. ``aggregate_gdi`` and
    ``rollback_recommended`` are None unless measured.
    """

    timestamp: str
    baseline_snapshot_id: str
    current_snapshot_id: str
    canon_version: str
    status: str
    aggregate_gdi: Optional[float]
    component_breakdown: Dict[str, Optional[float]]
    per_entry_gdi: Dict[str, float]
    rollback_recommended: Optional[bool]
    rollback_threshold: float
    entries_compared: int = 0
    entries_missing: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_canon(path: Path) -> CanonSet:
    """Read a canon JSON file."""
    return CanonSet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_canon(canon: CanonSet, path: Path) -> None:
    """Write a canon JSON file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canon.to_dict(), indent=2), encoding="utf-8")


def save_snapshot(snapshot: Snapshot, path: Path) -> None:
    """Write a snapshot JSON file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")


def load_snapshot(path: Path) -> Snapshot:
    """Read a snapshot JSON file."""
    return Snapshot.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
