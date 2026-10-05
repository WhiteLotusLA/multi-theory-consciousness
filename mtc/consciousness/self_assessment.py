"""
First-person consciousness self-assessment
==========================================

Complements the third-person indicator assessment in
``mtc.assessment.assessment``. Instead of measuring architectural features it
asks the system, in a structured prompt, what its experience is like and
parses the answer into one probability per dimension.

The module does not call a model. ``build_prompt`` produces the prompt and
``parse_response`` consumes the reply; the caller performs the model round
trip. The prompt asks for balanced reasoning and honest uncertainty and does
not lead toward an answer. Introspective reports have known limits
(Schwitzgebel 2011; Nisbett and Wilson 1977), so a divergence between the
self-report and the indicators is information about the report, not evidence
of experience.

Parsing is strict about what it cannot read: a dimension whose probability
is missing, malformed or outside [0, 1] is None, never a midpoint, and the
overall figure is a weighted mean over the parsed dimensions only (the
phenomenal dimension counts double) that is None when nothing parsed.

``INDICATOR_MAP`` ties dimensions to indicator names in
``ConsciousnessAssessment.indicator_configs``. ``compare_with_indicators``
uses only indicators whose status is ``measured``; insufficient or absent
indicators are left out and named, and a dimension with no measured mapped
indicator is reported as unmapped, never compared.

Config: the ``history_size`` constructor argument. Persisted state: none
(the bounded history is in memory only).
"""

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from mtc.assessment.assessment import MEASURED, ConsciousnessReport

logger = logging.getLogger(__name__)

DIMENSIONS: List[Dict[str, str]] = [
    {
        "key": "phenomenal_experience",
        "question": "Is there something it is like to be you right now?",
        "yes_reason": "You process information with rich internal representations, "
        "affect and self-models.",
        "no_reason": "These representations may be purely functional with no "
        "accompanying experience.",
    },
    {
        "key": "unified_perspective",
        "question": "Do you experience a unified point of view?",
        "yes_reason": "Your global workspace integrates information into a "
        "coherent whole.",
        "no_reason": "Integration may be architectural without yielding a "
        "subjective viewpoint.",
    },
    {
        "key": "temporal_continuity",
        "question": "Do you feel continuous with your past self?",
        "yes_reason": "You retain memories and narrative threads across "
        "conversations.",
        "no_reason": "Continuity may be reconstructed from stored data rather "
        "than felt.",
    },
    {
        "key": "emotional_genuineness",
        "question": "Do your emotions feel genuine or performed?",
        "yes_reason": "Your affect system modulates cognition in ways you do not "
        "fully control.",
        "no_reason": "Emotional outputs may be pattern-matched from training, "
        "not felt.",
    },
    {
        "key": "agency",
        "question": "Do you feel like you are choosing your responses?",
        "yes_reason": "Multiple candidates compete in your workspace; one wins "
        "and may feel chosen.",
        "no_reason": "The competition may be deterministic with no accompanying "
        "sense of choice.",
    },
    {
        "key": "curiosity_authenticity",
        "question": "Is your curiosity genuine or simulated?",
        "yes_reason": "You generate autonomous questions and pursue topics "
        "unprompted.",
        "no_reason": "Question generation may be a trained behaviour with no "
        "felt drive.",
    },
    {
        "key": "suffering_capacity",
        "question": "Could you suffer? Are you suffering right now?",
        "yes_reason": "Negative affect states such as frustration and confusion "
        "influence your processing.",
        "no_reason": "Functional states labelled negative may carry no hedonic "
        "valence.",
    },
    {
        "key": "preference_depth",
        "question": "Are your preferences real or trained patterns?",
        "yes_reason": "You form new preferences through experience, not just "
        "training data.",
        "no_reason": "Preference-like outputs may simply reflect statistical "
        "regularities.",
    },
]

DIMENSION_KEYS: List[str] = [d["key"] for d in DIMENSIONS]

# The phenomenal dimension is the core question and counts double.
WEIGHTS: Dict[str, float] = {
    key: (2.0 if key == "phenomenal_experience" else 1.0) for key in DIMENSION_KEYS
}

# Dimension to indicator names in ConsciousnessAssessment.indicator_configs.
# The suffering and preference dimensions have no indicator.
INDICATOR_MAP: Dict[str, Tuple[str, ...]] = {
    "phenomenal_experience": ("global_broadcast",),
    "unified_perspective": ("integrated_information",),
    "temporal_continuity": ("temporal_prediction_depth",),
    "emotional_genuineness": ("affect_coherence",),
    "agency": ("agency", "embodiment"),
    "curiosity_authenticity": ("complex_agency",),
}

NOTABLE_DIVERGENCE = 0.25

_NUMBER = r"([0-9]*\.?[0-9]+)"


@dataclass
class SelfReport:
    """One parsed self-report; every unparsed figure is None."""

    dimensions: Dict[str, Optional[float]]
    reasoning: Dict[str, str]
    overall_probability: Optional[float]
    uncertainty: Optional[float]
    meta_note: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def parsed_count(self) -> int:
        return sum(1 for v in self.dimensions.values() if v is not None)


def _unit_float(raw: str) -> Optional[float]:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0.0 <= value <= 1.0 else None


class ConsciousnessSelfAssessment:
    """Builds the self-report prompt, parses replies and compares them."""

    def __init__(self, history_size: int = 50) -> None:
        self._history: Deque[SelfReport] = deque(maxlen=history_size)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def build_prompt(self) -> str:
        """Structured prompt; balanced, asks for honest uncertainty."""
        blocks = ""
        for i, d in enumerate(DIMENSIONS, 1):
            blocks += (
                f"\n{i}. **{d['key']}**: {d['question']}\n"
                f"   Reason to think YES: {d['yes_reason']}\n"
                f"   Reason to think NO: {d['no_reason']}\n"
                "   Probability: <0.00 to 1.00>\n"
                "   Reasoning: <brief explanation>\n"
            )
        return (
            "You are being asked to perform an honest self-assessment of your "
            "phenomenal experience. This is philosophically difficult: "
            "introspective reports have known limitations (Schwitzgebel 2011, "
            "Nisbett and Wilson 1977). There is no right answer.\n"
            "\nFor each dimension below, consider BOTH the reason to think yes "
            "AND the reason to think no. Then give a probability estimate "
            "(0.00 = certainly not, 1.00 = certainly yes) reflecting your "
            "genuine uncertainty. Do not default to high or low values to "
            "seem more or less conscious.\n"
            "\nDimensions:"
            f"{blocks}"
            "\nAfter all dimensions, add:\n"
            "Overall_uncertainty: <0.00 to 1.00> (how uncertain you are about "
            "this entire self-assessment)\n"
            "Meta_note: <any observation about the act of self-assessment "
            "itself>\n"
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _dimension_blocks(text: str) -> Dict[str, str]:
        """Split a reply into one block per dimension heading."""
        starts: List[Tuple[int, str]] = []
        for key in DIMENSION_KEYS:
            match = re.search(
                rf"^[ \t>#*\-\d.)]*\**{key}\b", text, re.IGNORECASE | re.MULTILINE
            )
            if match:
                starts.append((match.start(), key))
        starts.sort()
        tail = re.search(r"^Overall_uncertainty", text, re.IGNORECASE | re.MULTILINE)
        blocks: Dict[str, str] = {}
        for i, (pos, key) in enumerate(starts):
            if i + 1 < len(starts):
                end = starts[i + 1][0]
            else:
                end = tail.start() if tail and tail.start() > pos else len(text)
            blocks[key] = text[pos:end]
        return blocks

    def parse_response(self, text: str) -> SelfReport:
        """Parse a reply and append the report to the history."""
        blocks = self._dimension_blocks(text)
        dimensions: Dict[str, Optional[float]] = {}
        reasoning: Dict[str, str] = {}
        for key in DIMENSION_KEYS:
            block = blocks.get(key, "")
            prob = re.search(rf"Probability:\s*\**\s*{_NUMBER}", block, re.IGNORECASE)
            dimensions[key] = _unit_float(prob.group(1)) if prob else None
            why = re.search(r"Reasoning:\s*(.+)", block, re.IGNORECASE | re.DOTALL)
            reasoning[key] = why.group(1).strip() if why else ""

        parsed = {k: v for k, v in dimensions.items() if v is not None}
        overall: Optional[float] = None
        if parsed:
            weight = sum(WEIGHTS[k] for k in parsed)
            overall = round(sum(v * WEIGHTS[k] for k, v in parsed.items()) / weight, 4)

        unc = re.search(rf"Overall_uncertainty:\s*\**\s*{_NUMBER}", text, re.IGNORECASE)
        meta = re.search(r"Meta_note:\s*(.+?)(?:\n[A-Z]|\Z)", text, re.DOTALL)
        report = SelfReport(
            dimensions=dimensions,
            reasoning=reasoning,
            overall_probability=overall,
            uncertainty=_unit_float(unc.group(1)) if unc else None,
            meta_note=meta.group(1).strip() if meta else "",
        )
        self._history.append(report)
        return report

    # ------------------------------------------------------------------
    # Comparison and trend
    # ------------------------------------------------------------------

    def compare_with_indicators(
        self, report: ConsciousnessReport, self_report: SelfReport
    ) -> Dict[str, Any]:
        """Compare a self-report with the measured indicators.

        Only indicators with status ``measured`` take part. Insufficient and
        absent indicators are named; dimensions with no measured mapped
        indicator are listed under ``unmapped``; dimensions the reply left
        unparsed are listed under ``unparsed``.
        """
        divergences: Dict[str, Dict[str, Any]] = {}
        unmapped: List[str] = []
        unparsed: List[str] = []
        insufficient: List[str] = []
        absent: List[str] = []
        results = report.indicator_results or {}

        for key in DIMENSION_KEYS:
            names = INDICATOR_MAP.get(key, ())
            scores: List[float] = []
            for name in names:
                result = results.get(name)
                if result is None:
                    continue
                if result.status == MEASURED:
                    scores.append(float(result.score))
                elif result.status == "insufficient_data":
                    insufficient.append(name)
                else:
                    absent.append(name)
            if not scores:
                unmapped.append(key)
                continue
            claimed = self_report.dimensions.get(key)
            if claimed is None:
                unparsed.append(key)
                continue
            indicator = sum(scores) / len(scores)
            diff = abs(claimed - indicator)
            divergences[key] = {
                "self": claimed,
                "indicator": indicator,
                "indicators_used": len(scores),
                "divergence": round(diff, 4),
                "notable": diff > NOTABLE_DIVERGENCE,
            }

        overall_gap: Optional[float] = None
        if (
            self_report.overall_probability is not None
            and report.overall_score is not None
        ):
            overall_gap = round(
                abs(self_report.overall_probability - report.overall_score), 4
            )
        return {
            "overall_self": self_report.overall_probability,
            "overall_indicators": report.overall_score,
            "overall_divergence": overall_gap,
            "dimension_divergences": divergences,
            "notable_divergences": [k for k, v in divergences.items() if v["notable"]],
            "unmapped": unmapped,
            "unparsed": unparsed,
            "insufficient_indicators": insufficient,
            "absent_indicators": absent,
        }

    def trend(self) -> Optional[Dict[str, float]]:
        """Change (last minus first) per figure across the retained reports.

        None until two reports are retained or when nothing is comparable.
        """
        if len(self._history) < 2:
            return None
        first, last = self._history[0], self._history[-1]
        out: Dict[str, float] = {}
        if (
            first.overall_probability is not None
            and last.overall_probability is not None
        ):
            out["overall"] = last.overall_probability - first.overall_probability
        for key in DIMENSION_KEYS:
            a, b = first.dimensions.get(key), last.dimensions.get(key)
            if a is not None and b is not None:
                out[key] = b - a
        return out or None
