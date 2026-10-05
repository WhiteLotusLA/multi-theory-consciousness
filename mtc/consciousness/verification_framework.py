"""
Consciousness Verification Framework
====================================

Verifies a consciousness state against the real signals the substrates
produced. Three tests run:

- ``CoherenceTest``: SNN synchrony, LSM reservoir coherence and HTM predictive
  success (one minus the anomaly score), averaged as one neural group beside
  the cross-modal emotional agreement and the conversational continuity.
  Continuity never makes the test measured on its own: without a real
  substrate or affect component the test is ``insufficient_data``.
- ``TemporalConsistencyTest``: stability of real SNN temporal patterns across
  successive states, and of personality traits and core beliefs when the
  states carry them.
- ``EmotionalAuthenticityTest``: agreement between the conscious PAD affect
  and the LSM's intuitive affect.

A substrate component counts only when its block carries ``is_real`` true and
its value is finite. A test with no real component returns ``insufficient_data``
with a None score and the reason; the suite names it and leaves it out of the
aggregate, which is the mean over measured tests and None when none measured.
The suite status uses the assessment vocabulary: ``measured`` (every test
measured), ``partial`` or ``insufficient_data``.

Before the tests run, a fake-data screen inspects long signal arrays (the
top-level ``snn_states``, ``lsm_states`` and ``htm_states`` keys and the SNN
block's ``temporal_patterns``, in the current state and in every temporal
sample). An array that is constant, or that is independent noise distributed
as a standard normal or a uniform, is flagged in ``fake_data_flags``; a
flagged SNN block is treated as not real for the run. The LSM trajectory is
not screened: a snapshot across reservoir neurons has no serial order, so the
independence check cannot tell it from noise.

Self-awareness and intentionality tests that count words in replies are not
part of this module: they measure vocabulary rather than architecture.

The verifier keeps a bounded history of suite results (``history_size``) for
trend and anomaly analysis; it holds no persisted state and reads no config
keys. Timestamps are timezone-aware UTC from the injectable ``clock``.
"""

import logging
import math
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MEASURED = "measured"
PARTIAL = "partial"
INSUFFICIENT_DATA = "insufficient_data"

Clock = Callable[[], datetime]


class TestType(Enum):
    """The verification tests this module runs."""

    __test__ = False  # not a pytest test class

    COHERENCE = "coherence"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    EMOTIONAL_AUTHENTICITY = "emotional_authenticity"


# A measured score above the threshold passes.
PASS_THRESHOLDS: Dict[TestType, float] = {
    TestType.COHERENCE: 0.6,
    TestType.TEMPORAL_CONSISTENCY: 0.7,
    TestType.EMOTIONAL_AUTHENTICITY: 0.6,
}

# Share of measured tests that must pass for the suite to pass.
SUITE_PASS_RATIO = 0.6

# The fake-data screen judges arrays of at least this many finite values.
FAKE_DATA_MIN_SAMPLES = 100
_FAKE_DATA_MAX_SAMPLES = 10000
_NOISE_P_VALUE = 0.05


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _finite(value: Any) -> Optional[float]:
    """The value as a finite float, or None when missing or not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _field(source: Any, name: str) -> Any:
    """``source[name]`` for a mapping, ``source.name`` otherwise."""
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _real_block(state: Dict[str, Any], key: str) -> Dict[str, Any]:
    """The named substrate block when it carries ``is_real`` true, else {}."""
    block = state.get(key)
    if isinstance(block, dict) and block.get("is_real") is True:
        return block
    return {}


@dataclass
class VerificationResult:
    """Result of one verification test.

    A measured result has a finite ``score`` in [0, 1], ``passed`` and the
    real ``components`` it averaged. An insufficient result has
    ``insufficient_data`` true, a None score, None ``passed`` and a ``reason``.
    """

    test_type: TestType
    score: Optional[float]
    passed: Optional[bool]
    components: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    insufficient_data: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.insufficient_data:
            if self.score is not None or self.passed is not None:
                raise ValueError("an insufficient result carries no score or verdict")
            return
        score = _finite(self.score)
        if score is None or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"score must be a finite value in [0, 1], got {self.score}"
            )

    @classmethod
    def insufficient(
        cls,
        test_type: TestType,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "VerificationResult":
        """An honest not-measured result, kept out of every aggregate."""
        return cls(
            test_type=test_type,
            score=None,
            passed=None,
            reason=reason,
            insufficient_data=True,
            details=dict(details or {}),
            timestamp=timestamp or _utcnow(),
        )

    @property
    def status(self) -> str:
        return INSUFFICIENT_DATA if self.insufficient_data else MEASURED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_type": self.test_type.value,
            "status": self.status,
            "score": self.score,
            "passed": self.passed,
            "components": dict(self.components),
            "reason": self.reason,
            "details": dict(self.details),
            "timestamp": self.timestamp.isoformat(),
            "warnings": list(self.warnings),
        }


def _measured(
    test_type: TestType,
    components: Dict[str, float],
    score: float,
    details: Dict[str, Any],
    timestamp: datetime,
) -> VerificationResult:
    warnings: List[str] = []
    clipped = min(1.0, max(0.0, score))
    if clipped != score:
        warnings.append(f"score {score:.3f} clipped to [0, 1]")
    return VerificationResult(
        test_type=test_type,
        score=clipped,
        passed=clipped > PASS_THRESHOLDS[test_type],
        components=components,
        details=details,
        timestamp=timestamp,
        warnings=warnings,
    )


def _unit_component(
    components: Dict[str, float], warnings: List[str], name: str, value: float
) -> None:
    """Record a component, clipping it to [0, 1] with a warning when needed."""
    clipped = min(1.0, max(0.0, value))
    if clipped != value:
        warnings.append(f"{name} {value:.3f} clipped to [0, 1]")
    components[name] = clipped


def cross_modal_emotional_coherence(pad: Any, lsm: Any) -> Optional[float]:
    """Agreement between the PAD affect and the LSM's intuitive affect.

    PAD pleasure lies in [-1, 1] and PAD arousal in [0, 1] (MTC's ``PADState``
    ranges); the LSM's ``intuitive_valence`` and ``intuitive_arousal`` lie in
    [-1, 1]. PAD arousal is rescaled to [-1, 1], each axis distance is divided
    by its range of 2 and the result is one minus the mean distance, in
    [0, 1]. ``pad`` may be a ``PADState`` or a mapping. None when either side,
    or any of the four values, is missing or not finite.
    """
    if not pad or not lsm:
        return None
    pleasure = _finite(_field(pad, "pleasure"))
    arousal = _finite(_field(pad, "arousal"))
    valence = _finite(_field(lsm, "intuitive_valence"))
    lsm_arousal = _finite(_field(lsm, "intuitive_arousal"))
    if pleasure is None or arousal is None or valence is None or lsm_arousal is None:
        return None
    pad_arousal = 2.0 * arousal - 1.0
    d_valence = abs(pleasure - valence) / 2.0
    d_arousal = abs(pad_arousal - lsm_arousal) / 2.0
    return max(0.0, min(1.0, 1.0 - (d_valence + d_arousal) / 2.0))


_WORD = re.compile(r"[a-z0-9']+")


def contextual_continuity(conversation: Any) -> Optional[float]:
    """Word-overlap continuity between the last turn and the turn before it.

    Turns with non-empty ``content`` are used whatever their role. Words are
    case-folded and punctuation is ignored. The value is twice the share of
    the previous turn's distinct words that the last turn repeats, clipped to
    [0, 1]. None with fewer than two such turns or an empty previous turn.
    """
    if not isinstance(conversation, (list, tuple)):
        return None
    turns = [
        t for t in conversation if isinstance(t, dict) and str(t.get("content") or "")
    ]
    if len(turns) < 2:
        return None
    last = set(_WORD.findall(str(turns[-1]["content"]).lower()))
    previous = set(_WORD.findall(str(turns[-2]["content"]).lower()))
    if not previous:
        return None
    return max(0.0, min(1.0, len(previous & last) / len(previous) * 2.0))


def _lsm_affect_source(state: Dict[str, Any]) -> Any:
    """The LSM affect to compare: ``lsm_data_affect``, else a real LSM block."""
    source = state.get("lsm_data_affect")
    if source is None:
        block = _real_block(state, "lsm_data")
        source = block or None
    return source


class CoherenceTest:
    """Coherence across the neural substrates, affect and conversation."""

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock = clock or _utcnow

    def run(
        self,
        consciousness_state: Any,
        conversation: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationResult:
        """Score the real components of the state; never raises.

        Args:
            consciousness_state: Dict with ``snn_data``, ``lsm_data``,
                ``htm_data``, ``pad_affect``, ``lsm_data_affect`` and
                ``conversation_history``, any of them optional.
            conversation: Turns to read instead of ``conversation_history``.
        """
        now = self._clock()
        if not isinstance(consciousness_state, dict) or not consciousness_state:
            return VerificationResult.insufficient(
                TestType.COHERENCE, "no consciousness state", timestamp=now
            )
        state = consciousness_state
        components: Dict[str, float] = {}
        warnings: List[str] = []

        synchrony = _finite(_real_block(state, "snn_data").get("synchrony"))
        if synchrony is not None:
            _unit_component(components, warnings, "snn", synchrony)
        coherence = _finite(_real_block(state, "lsm_data").get("coherence"))
        if coherence is not None:
            _unit_component(components, warnings, "lsm", coherence)
        anomaly = _finite(_real_block(state, "htm_data").get("anomaly_score"))
        if anomaly is not None:
            _unit_component(components, warnings, "htm", 1.0 - anomaly)

        emotional = cross_modal_emotional_coherence(
            state.get("pad_affect"), _lsm_affect_source(state)
        )
        if emotional is not None:
            components["emotional"] = emotional

        turns = conversation if conversation else state.get("conversation_history")
        contextual = contextual_continuity(turns)

        # Word overlap alone is not evidence about the architecture: a reply
        # that repeats the previous turn would score 1.0. Conversational
        # continuity joins the score only beside a real substrate or affect
        # component.
        if not components:
            details = {} if contextual is None else {"contextual": contextual}
            return VerificationResult.insufficient(
                TestType.COHERENCE,
                "no real substrate or affect component",
                details=details,
                timestamp=now,
            )
        if contextual is not None:
            components["contextual"] = contextual

        # The neural sub-group is averaged first so three substrates cannot
        # outweigh the emotional and contextual components.
        neural = [components[k] for k in ("snn", "lsm", "htm") if k in components]
        groups = [float(np.mean(neural))] if neural else []
        groups += [
            components[k] for k in ("emotional", "contextual") if k in components
        ]
        result = _measured(
            TestType.COHERENCE,
            components,
            float(np.mean(groups)),
            {"groups": len(groups)},
            now,
        )
        result.warnings[:0] = warnings
        return result


class TemporalConsistencyTest:
    """Stability of real signals across at least two successive states."""

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock = clock or _utcnow

    def run(self, states: Any) -> VerificationResult:
        """Score the stability of the states' real components; never raises.

        Args:
            states: Successive state dicts, oldest first. SNN temporal patterns
                count from blocks carrying ``is_real``; ``personality_traits``
                and ``core_beliefs`` count only when states carry them.
        """
        now = self._clock()
        if not isinstance(states, list) or len(states) < 2:
            return VerificationResult.insufficient(
                TestType.TEMPORAL_CONSISTENCY,
                "fewer than two temporal samples",
                timestamp=now,
            )
        dict_states = [s for s in states if isinstance(s, dict)]
        components: Dict[str, float] = {}

        sequences = []
        for s in dict_states:
            patterns = _real_block(s, "snn_data").get("temporal_patterns")
            if patterns is None:
                continue
            try:
                array = np.asarray(patterns, dtype=float).ravel()
            except (TypeError, ValueError):
                continue
            if array.size:
                sequences.append(array)
        snn_stability = self._sequence_stability(sequences)
        if snn_stability is not None:
            components["snn_temporal"] = snn_stability

        traits = [
            s["personality_traits"]
            for s in dict_states
            if isinstance(s.get("personality_traits"), dict) and s["personality_traits"]
        ]
        trait_stability = self._trait_stability(traits)
        if trait_stability is not None:
            components["trait_stability"] = trait_stability

        beliefs = [
            list(s["core_beliefs"])
            for s in dict_states
            if isinstance(s.get("core_beliefs"), (list, tuple)) and s["core_beliefs"]
        ]
        belief_consistency = self._belief_consistency(beliefs)
        if belief_consistency is not None:
            components["belief_consistency"] = belief_consistency

        if not components:
            return VerificationResult.insufficient(
                TestType.TEMPORAL_CONSISTENCY,
                "no real component varies across the samples",
                details={"num_states": len(states)},
                timestamp=now,
            )
        return _measured(
            TestType.TEMPORAL_CONSISTENCY,
            components,
            float(np.mean(list(components.values()))),
            {"num_states": len(states)},
            now,
        )

    @staticmethod
    def _sequence_stability(sequences: List[np.ndarray]) -> Optional[float]:
        """Mean correlation of consecutive patterns, negative clipped to 0."""
        if len(sequences) < 2:
            return None
        correlations = []
        for previous, current in zip(sequences, sequences[1:]):
            n = min(previous.size, current.size)
            if n < 2:
                continue
            a, b = previous[:n], current[:n]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            if np.std(a) > 0 and np.std(b) > 0:
                c = float(np.corrcoef(a, b)[0, 1])
                if math.isfinite(c):
                    correlations.append(c)
        if not correlations:
            return None
        return max(0.0, min(1.0, float(np.mean(correlations))))

    @staticmethod
    def _trait_stability(trait_dicts: List[Dict[str, Any]]) -> Optional[float]:
        """One minus the mean absolute change of shared finite trait values."""
        if len(trait_dicts) < 2:
            return None
        diffs = []
        for previous, current in zip(trait_dicts, trait_dicts[1:]):
            changes = []
            for name in set(previous) & set(current):
                a, b = _finite(previous[name]), _finite(current[name])
                if a is not None and b is not None:
                    changes.append(abs(a - b))
            if changes:
                diffs.append(float(np.mean(changes)))
        if not diffs:
            return None
        return max(0.0, min(1.0, 1.0 - float(np.mean(diffs))))

    @staticmethod
    def _belief_consistency(belief_lists: List[List[Any]]) -> Optional[float]:
        """Share of the first state's beliefs still held in the last state."""
        if len(belief_lists) < 2:
            return None
        first, last = set(belief_lists[0]), set(belief_lists[-1])
        if not first:
            return None
        return len(first & last) / len(first)


class EmotionalAuthenticityTest:
    """Agreement between the conscious PAD affect and the LSM intuitive affect."""

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock = clock or _utcnow

    def run(self, consciousness_state: Any) -> VerificationResult:
        """Score the cross-modal affect agreement; never raises."""
        now = self._clock()
        if not isinstance(consciousness_state, dict) or not consciousness_state:
            return VerificationResult.insufficient(
                TestType.EMOTIONAL_AUTHENTICITY, "no consciousness state", timestamp=now
            )
        congruence = cross_modal_emotional_coherence(
            consciousness_state.get("pad_affect"),
            _lsm_affect_source(consciousness_state),
        )
        if congruence is None:
            return VerificationResult.insufficient(
                TestType.EMOTIONAL_AUTHENTICITY,
                "PAD affect and real LSM intuitive affect are both required",
                timestamp=now,
            )
        components = {"affect_congruence": congruence}
        return _measured(
            TestType.EMOTIONAL_AUTHENTICITY, components, congruence, {}, now
        )


def detect_fake_signal(
    values: Any, min_samples: int = FAKE_DATA_MIN_SAMPLES
) -> Optional[str]:
    """Name the kind of stand-in data an array looks like, or None.

    Returns ``constant`` for an array with no variation, and
    ``standard_normal_noise`` or ``uniform_noise`` for an array whose
    consecutive values are uncorrelated (lag-1 autocorrelation within three
    standard errors of zero) and whose distribution a normality test or a
    Kolmogorov-Smirnov test against the uniform cannot reject at p 0.05.
    Arrays with fewer than ``min_samples`` finite values are not judged.
    ``scipy.stats`` is imported only when an array is long enough to test.
    """
    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    array = array[np.isfinite(array)][:_FAKE_DATA_MAX_SAMPLES]
    n = array.size
    if n < max(min_samples, 8):
        return None
    if float(np.ptp(array)) == 0.0:
        return "constant"
    lag1 = float(np.corrcoef(array[:-1], array[1:])[0, 1])
    if not math.isfinite(lag1) or abs(lag1) >= 3.0 / math.sqrt(n):
        return None

    from scipy import stats

    mean, std = float(np.mean(array)), float(np.std(array))
    if abs(mean) < 0.1 and 0.9 < std < 1.1:
        if float(stats.normaltest(array).pvalue) > _NOISE_P_VALUE:
            return "standard_normal_noise"
    low, high = float(np.min(array)), float(np.max(array))
    scaled = (array - low) / (high - low)
    if float(stats.kstest(scaled, "uniform").pvalue) > _NOISE_P_VALUE:
        return "uniform_noise"
    return None


@dataclass
class VerificationSuiteResult:
    """Result of one verification suite run.

    ``results`` maps each test name to its result. ``overall_score`` is the
    mean over measured tests and None when none measured; ``overall_passed``
    is None in that case too. ``measured`` and ``insufficient`` name the
    tests; ``coverage`` is the measured share; ``fake_data_flags`` maps each
    flagged array to the kind of stand-in data it resembles.
    """

    results: Dict[str, VerificationResult]
    overall_score: Optional[float]
    overall_passed: Optional[bool]
    measured: List[str]
    insufficient: List[str]
    fake_data_flags: Dict[str, str]
    status: str
    coverage: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": {name: r.to_dict() for name, r in self.results.items()},
            "overall_score": self.overall_score,
            "overall_passed": self.overall_passed,
            "measured": list(self.measured),
            "insufficient": list(self.insufficient),
            "fake_data_flags": dict(self.fake_data_flags),
            "status": self.status,
            "coverage": self.coverage,
            "timestamp": self.timestamp.isoformat(),
        }


# Arrays the fake-data screen inspects: (state key, block key or None).
_SCREENED_ARRAYS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("snn_states", None),
    ("lsm_states", None),
    ("htm_states", None),
    ("snn_data", "temporal_patterns"),
)


class ConsciousnessVerifier:
    """Runs the verification tests and keeps a bounded history of suites.

    Args:
        history_size: Suite results kept for trend and anomaly analysis.
        clock: Returns the current aware UTC time; defaults to the system clock.
        fake_data_min_samples: Shortest array the fake-data screen judges.
    """

    def __init__(
        self,
        history_size: int = 100,
        clock: Optional[Clock] = None,
        fake_data_min_samples: int = FAKE_DATA_MIN_SAMPLES,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self._clock = clock or _utcnow
        self.fake_data_min_samples = fake_data_min_samples
        self.coherence_test = CoherenceTest(clock=self._clock)
        self.consistency_test = TemporalConsistencyTest(clock=self._clock)
        self.authenticity_test = EmotionalAuthenticityTest(clock=self._clock)
        self.history: Deque[VerificationSuiteResult] = deque(maxlen=history_size)

    # ======== FAKE-DATA SCREEN ========

    def _screen(
        self, state: Dict[str, Any], prefix: str = ""
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """A copy of the state with flagged blocks emptied, and the flags."""
        screened = dict(state)
        flags: Dict[str, str] = {}
        for key, inner in _SCREENED_ARRAYS:
            container = state.get(key)
            if inner is None:
                values = container
            elif isinstance(container, dict):
                values = container.get(inner)
            else:
                continue
            if values is None:
                continue
            kind = detect_fake_signal(values, self.fake_data_min_samples)
            if kind is None:
                continue
            name = key if inner is None else f"{key}.{inner}"
            flags[f"{prefix}{name}"] = kind
            if inner is not None:
                screened[key] = {}
        return screened, flags

    def fake_data_flags(self, consciousness_state: Any) -> Dict[str, str]:
        """Arrays in the state that look like stand-in data, by kind."""
        if not isinstance(consciousness_state, dict):
            return {}
        _, flags, _ = self._prepare(consciousness_state)
        return flags

    def _prepare(
        self, state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str], List[Dict[str, Any]]]:
        """Screen the state and its temporal samples."""
        screened, flags = self._screen(state)
        temporal = state.get("temporal_data")
        samples = temporal.get("states") if isinstance(temporal, dict) else None
        states: List[Dict[str, Any]] = []
        for index, sample in enumerate(samples or []):
            if not isinstance(sample, dict):
                continue
            clean, sample_flags = self._screen(
                sample, f"temporal_data.states[{index}]."
            )
            flags.update(sample_flags)
            states.append(clean)
        return screened, flags, states

    # ======== RUNNING ========

    def _evaluate(
        self, consciousness_state: Any, test_types: Optional[Sequence[TestType]]
    ) -> Tuple[Dict[str, VerificationResult], Dict[str, str]]:
        wanted = list(test_types) if test_types is not None else list(TestType)
        if not isinstance(consciousness_state, dict) or not consciousness_state:
            now = self._clock()
            return {
                t.value: VerificationResult.insufficient(
                    t, "no consciousness state", timestamp=now
                )
                for t in wanted
                if isinstance(t, TestType)
            }, {}
        state, flags, states = self._prepare(consciousness_state)
        results: Dict[str, VerificationResult] = {}
        for test_type in wanted:
            if test_type is TestType.COHERENCE:
                result = self.coherence_test.run(
                    state, state.get("conversation_history")
                )
            elif test_type is TestType.TEMPORAL_CONSISTENCY:
                result = self.consistency_test.run(states)
            elif test_type is TestType.EMOTIONAL_AUTHENTICITY:
                result = self.authenticity_test.run(state)
            else:
                logger.warning(f"Unknown verification test {test_type!r} skipped")
                continue
            results[test_type.value] = result
        return results, flags

    def verify_consciousness_state(
        self,
        consciousness_state: Any,
        test_types: Optional[Sequence[TestType]] = None,
    ) -> Dict[str, VerificationResult]:
        """Run the tests, each on the input shape it reads; never raises.

        The coherence test gets the screened state and its
        ``conversation_history``; the temporal test gets the list of screened
        ``temporal_data["states"]``; the authenticity test gets the state.
        """
        results, _ = self._evaluate(consciousness_state, test_types)
        return results

    def run_verification_suite(
        self,
        consciousness_state: Any,
        test_types: Optional[Sequence[TestType]] = None,
    ) -> VerificationSuiteResult:
        """Run the suite, aggregate over measured tests and record the result."""
        results, flags = self._evaluate(consciousness_state, test_types)
        measured = [name for name, r in results.items() if not r.insufficient_data]
        insufficient = [name for name, r in results.items() if r.insufficient_data]
        scores = [results[name].score for name in measured]
        overall = float(np.mean(scores)) if scores else None
        passed = None
        if measured:
            passing = sum(1 for name in measured if results[name].passed)
            passed = passing >= len(measured) * SUITE_PASS_RATIO
        if not measured:
            status = INSUFFICIENT_DATA
        elif insufficient:
            status = PARTIAL
        else:
            status = MEASURED
        suite = VerificationSuiteResult(
            results=results,
            overall_score=overall,
            overall_passed=passed,
            measured=measured,
            insufficient=insufficient,
            fake_data_flags=flags,
            status=status,
            coverage=len(measured) / len(results) if results else 0.0,
            timestamp=self._clock(),
        )
        self.history.append(suite)
        if flags:
            logger.warning(
                "Verification excluded stand-in data: "
                + ", ".join(f"{k} ({v})" for k, v in sorted(flags.items()))
            )
        logger.info(
            f"Verification suite: status={status}, measured={len(measured)}"
            f"/{len(results)}, score="
            + ("insufficient_data" if overall is None else f"{overall:.3f}")
            + (f", not measured: {', '.join(insufficient)}" if insufficient else "")
        )
        return suite

    # ======== TRENDS ========

    def _series(self, test_name: str) -> List[Tuple[datetime, float]]:
        points = []
        for suite in self.history:
            result = suite.results.get(test_name)
            if result is not None and not result.insufficient_data:
                points.append((suite.timestamp, float(result.score)))
        return points

    def analyze_trends(self) -> Dict[str, Any]:
        """Growth and stability of measured test scores across the history.

        A test contributes when it measured in at least two recorded suites;
        its growth is its last measured score minus its first. The trajectory
        is ``improving`` or ``declining`` when the mean growth exceeds 0.05 in
        either direction and ``stable`` otherwise; ``stability_score`` is
        1 / (1 + std) over every measured score. With no contributing test the
        status is ``insufficient_data`` and every value is None.
        """
        growth: Dict[str, float] = {}
        all_scores: List[float] = []
        for test_type in TestType:
            series = self._series(test_type.value)
            all_scores += [score for _, score in series]
            if len(series) >= 2:
                growth[test_type.value] = series[-1][1] - series[0][1]
        if not growth:
            return {
                "status": INSUFFICIENT_DATA,
                "trajectory": None,
                "growth_rate": None,
                "stability_score": None,
                "growth_by_test": {},
                "fastest_growing": [],
                "suites": len(self.history),
            }
        rate = float(np.mean(list(growth.values())))
        if rate > 0.05:
            trajectory = "improving"
        elif rate < -0.05:
            trajectory = "declining"
        else:
            trajectory = "stable"
        ranked = sorted(growth.items(), key=lambda item: item[1], reverse=True)
        return {
            "status": MEASURED,
            "trajectory": trajectory,
            "growth_rate": rate,
            "stability_score": 1.0 / (1.0 + float(np.std(all_scores))),
            "growth_by_test": growth,
            "fastest_growing": [name for name, _ in ranked[:2]],
            "suites": len(self.history),
        }

    def detect_anomalies(self, test_name: str = "coherence") -> List[Dict[str, Any]]:
        """Sudden drops and outliers in one test's measured scores.

        Needs at least three measured points; unmeasured suites are skipped.
        A sudden drop is a fall of more than 0.3 from the previous measured
        score; an outlier lies more than three standard deviations from the
        mean.
        """
        series = self._series(test_name)
        if len(series) < 3:
            return []
        values = np.array([score for _, score in series])
        mean, std = float(values.mean()), float(values.std())
        anomalies: List[Dict[str, Any]] = []
        for index, (stamp, value) in enumerate(series):
            if index and value < series[index - 1][1] - 0.3:
                anomalies.append(
                    {
                        "type": "sudden_drop",
                        "index": index,
                        "timestamp": stamp,
                        "value": value,
                        "previous": series[index - 1][1],
                    }
                )
            if std > 0 and abs(value - mean) / std > 3.0:
                anomalies.append(
                    {
                        "type": "statistical_outlier",
                        "index": index,
                        "timestamp": stamp,
                        "value": value,
                        "z_score": abs(value - mean) / std,
                    }
                )
        return anomalies
