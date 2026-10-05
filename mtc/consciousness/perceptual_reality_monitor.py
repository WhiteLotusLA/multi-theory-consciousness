"""
Perceptual reality monitor
==========================

A functional reality-monitoring signal after Lau (2019, "Consciousness,
metacognition and perceptual reality monitoring") and Lau and Rosenthal
(2011, Trends in Cognitive Sciences): a system that can tell externally
sourced signals from self-generated ones needs both kinds to be present and
needs a confident source label for each.

Each signal is recorded with a source name. The ``sources`` registry maps a
source name to ``"external"`` or ``"internal"``; the defaults are
``user_message`` and ``sensory`` (external) and ``memory_retrieval``,
``thought_generator``, ``dream`` and ``imagination`` (internal). A source
absent from the registry is counted as unknown and is never treated as
external or internal.

``discrimination()`` is source diversity (1 at an even external and internal
mix, 0 when one-sided) times the mean attribution confidence of the classified
signals in the window. It is None until the window holds at least one
external and one internal signal, because there is nothing to discriminate
before then. When a caller records a signal without an attribution
confidence, the class prior ``DEFAULT_ATTRIBUTION`` stands in for it and the
statistics report the share of such signals as ``default_attribution_fraction``.

``get_context_string()`` is a neutral bracketed line and is empty while
discrimination is None. The Phase 6 orchestrator fills
``ConsciousnessState.prm_discrimination`` from ``discrimination()``; nothing
calls the monitor yet.

Config: constructor arguments ``sources`` and ``window``. Persisted state
(``to_state_dict``): the signal counter and the bounded signal window, whose
summaries are private conversation fragments.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

EXTERNAL = "external"
INTERNAL = "internal"
UNKNOWN = "unknown"

DEFAULT_SOURCES: Dict[str, str] = {
    "user_message": EXTERNAL,
    "sensory": EXTERNAL,
    "memory_retrieval": INTERNAL,
    "thought_generator": INTERNAL,
    "dream": INTERNAL,
    "imagination": INTERNAL,
}


@dataclass
class Signal:
    """One recorded signal with its attributed origin."""

    summary: str
    source: str
    kind: str  # "external", "internal" or "unknown"
    attribution_confidence: Optional[float]
    attribution_is_default: bool = False
    timestamp: float = field(default_factory=time.time)


class PerceptualRealityMonitor:
    """Tracks the external and internal signal mix and how well it is labelled."""

    DEFAULT_ATTRIBUTION = 0.9

    def __init__(
        self, sources: Optional[Dict[str, str]] = None, window: int = 100
    ) -> None:
        registry = dict(DEFAULT_SOURCES if sources is None else sources)
        for name, kind in registry.items():
            if kind not in (EXTERNAL, INTERNAL):
                raise ValueError(
                    f"source {name!r} must map to 'external' or 'internal', "
                    f"got {kind!r}"
                )
        self.sources = registry
        self.window = int(window)
        self._signals: Deque[Signal] = deque(maxlen=self.window)
        self.signal_count = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_signal(
        self,
        source: str,
        summary: str = "",
        attribution_confidence: Optional[float] = None,
    ) -> None:
        """Record one signal; the confidence is clamped to [0, 1]."""
        kind = self.sources.get(source, UNKNOWN)
        confidence: Optional[float]
        is_default = False
        if attribution_confidence is not None:
            confidence = max(0.0, min(1.0, float(attribution_confidence)))
        elif kind != UNKNOWN:
            confidence = self.DEFAULT_ATTRIBUTION
            is_default = True
        else:
            confidence = None
        self._signals.append(
            Signal(
                summary=summary,
                source=source,
                kind=kind,
                attribution_confidence=confidence,
                attribution_is_default=is_default,
            )
        )
        self.signal_count += 1

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _classified(self) -> Dict[str, list]:
        out: Dict[str, list] = {EXTERNAL: [], INTERNAL: [], UNKNOWN: []}
        for signal in self._signals:
            out[signal.kind].append(signal)
        return out

    def discrimination(self) -> Optional[float]:
        """Diversity times mean attribution; None until both kinds were seen."""
        groups = self._classified()
        ext, internal = groups[EXTERNAL], groups[INTERNAL]
        if not ext or not internal:
            return None
        total = len(ext) + len(internal)
        diversity = 1.0 - abs(len(ext) / total - 0.5) * 2.0
        confidences = [
            s.attribution_confidence
            for s in ext + internal
            if s.attribution_confidence is not None
        ]
        if not confidences:
            return None
        return float(diversity * (sum(confidences) / len(confidences)))

    def get_context_string(self) -> str:
        """Neutral bracketed status line; empty while not measurable."""
        score = self.discrimination()
        if score is None:
            return ""
        groups = self._classified()
        return (
            f"[Reality monitoring: external={len(groups[EXTERNAL])}, "
            f"internal={len(groups[INTERNAL])}, "
            f"unknown={len(groups[UNKNOWN])}, discrimination={score:.2f}]"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Counts, diversity, attribution and the discrimination score."""
        groups = self._classified()
        ext, internal = len(groups[EXTERNAL]), len(groups[INTERNAL])
        classified = groups[EXTERNAL] + groups[INTERNAL]
        confidences = [
            s.attribution_confidence
            for s in classified
            if s.attribution_confidence is not None
        ]
        diversity: Optional[float] = None
        if ext and internal:
            diversity = 1.0 - abs(ext / (ext + internal) - 0.5) * 2.0
        defaults = [s for s in classified if s.attribution_is_default]
        return {
            "signal_count": self.signal_count,
            "window_size": len(self._signals),
            "external_count": ext,
            "internal_count": internal,
            "unknown_count": len(groups[UNKNOWN]),
            "source_diversity": diversity,
            "mean_attribution": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "default_attribution_fraction": (
                len(defaults) / len(classified) if classified else None
            ),
            "discrimination": self.discrimination(),
            "unique_sources": sorted({s.source for s in self._signals}),
        }

    def generate_report(self) -> str:
        """Plain-text status report."""
        stats = self.get_statistics()
        score = stats["discrimination"]
        lines = [
            "Perceptual reality monitoring status:",
            f"  Signals recorded: {stats['signal_count']}",
            f"  Window: {stats['window_size']} signals",
            f"  External: {stats['external_count']}, "
            f"internal: {stats['internal_count']}, "
            f"unknown: {stats['unknown_count']}",
        ]
        if score is None:
            lines.append("  Discrimination: not measurable yet")
        else:
            lines.append(f"  Source diversity: {stats['source_diversity']:.3f}")
            lines.append(f"  Mean attribution: {stats['mean_attribution']:.3f}")
            lines.append(f"  Discrimination: {score:.3f}")
        lines.append(f"  Sources seen: {', '.join(stats['unique_sources'])}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Export state for the persistence hub."""
        return {
            "scalars": {"signal_count": self.signal_count},
            "arrays": {},
            "histories": {
                "signals": [
                    {
                        "summary": s.summary,
                        "source": s.source,
                        "kind": s.kind,
                        "attribution_confidence": s.attribution_confidence,
                        "attribution_is_default": s.attribution_is_default,
                        "timestamp": s.timestamp,
                    }
                    for s in self._signals
                ]
            },
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the counter and the bounded signal window."""
        self.signal_count = int(state.get("scalars", {}).get("signal_count", 0))
        self._signals = deque(
            (
                Signal(
                    summary=str(entry.get("summary", "")),
                    source=str(entry["source"]),
                    kind=str(entry["kind"]),
                    attribution_confidence=entry.get("attribution_confidence"),
                    attribution_is_default=bool(
                        entry.get("attribution_is_default", False)
                    ),
                    timestamp=float(entry.get("timestamp", 0.0)),
                )
                for entry in state.get("histories", {}).get("signals", [])
            ),
            maxlen=self.window,
        )
