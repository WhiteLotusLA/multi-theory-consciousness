"""
Drift Tools
===========

Measurement tools for behavioral drift: a canon of probe prompts
(``canon_schema``), a sampler that runs the canon through any invocation
function into a snapshot (``sampler``), and a detector that compares two
snapshots with the Goal Drift Index (``drift_detector``). The package measures
only; it modifies nothing.

The index follows "SAHOO: Safeguarded Alignment for High-Order Optimization
Objectives in Recursive Self-Improvement" (arXiv 2603.06333).
"""

from mtc.consciousness.rsi.canon_schema import (
    CanonCategory,
    CanonEntry,
    CanonSet,
    DriftReading,
    ResponseSample,
    Snapshot,
    load_canon,
    load_snapshot,
    save_canon,
    save_snapshot,
    utc_now_iso,
)
from mtc.consciousness.rsi.drift_detector import (
    DEFAULT_COMPONENT_WEIGHTS,
    DriftDetector,
)
from mtc.consciousness.rsi.sampler import KeywordMarkerChecker, Sampler

__all__ = [
    "CanonCategory",
    "CanonEntry",
    "CanonSet",
    "DEFAULT_COMPONENT_WEIGHTS",
    "DriftDetector",
    "DriftReading",
    "KeywordMarkerChecker",
    "ResponseSample",
    "Sampler",
    "Snapshot",
    "load_canon",
    "load_snapshot",
    "save_canon",
    "save_snapshot",
    "utc_now_iso",
]
