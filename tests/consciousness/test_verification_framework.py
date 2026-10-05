"""The verification suite scores only real signals: an unreal block, a missing
input or random-looking data makes the affected test insufficient_data, which
is named and kept out of the aggregate, and nothing raises on a None field."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import mtc.consciousness.verification_framework as vf
from mtc.consciousness.verification_framework import (
    ConsciousnessVerifier,
    VerificationResult,
    VerificationSuiteResult,
    detect_fake_signal,
)

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _clock():
    return T0


def _real_snn(synchrony=0.8, patterns=None):
    return {
        "is_real": True,
        "synchrony": synchrony,
        "temporal_patterns": list(patterns) if patterns is not None else [],
    }


def test_unreal_components_are_insufficient_not_scored():
    state = {
        "snn_data": {"is_real": False, "synchrony": 0.9, "temporal_patterns": [1, 2]},
        "lsm_data": {
            "is_real": False,
            "coherence": 0.9,
            "intuitive_valence": 0.5,
            "intuitive_arousal": 0.5,
        },
        "htm_data": {"is_real": False, "anomaly_score": 0.1},
        "pad_affect": {"pleasure": 0.5, "arousal": 0.75},
        "conversation_history": [],
    }
    suite = ConsciousnessVerifier(clock=_clock).run_verification_suite(state)

    assert suite.overall_score is None
    assert suite.overall_passed is None
    assert suite.status == "insufficient_data"
    assert suite.coverage == 0.0
    assert suite.measured == []
    assert sorted(suite.insufficient) == sorted(t.value for t in vf.TestType)
    for result in suite.results.values():
        assert result.insufficient_data is True
        assert result.score is None
        assert result.passed is None
        assert result.status == "insufficient_data"
        assert result.reason


def test_overall_score_excludes_insufficient_and_never_raises_on_none():
    verifier = ConsciousnessVerifier(clock=_clock)
    state = {
        "snn_data": _real_snn(0.8),
        "lsm_data": {"is_real": True, "coherence": None},
        "htm_data": {"is_real": True, "anomaly_score": None},
        "pad_affect": {"pleasure": None, "arousal": None},
        "lsm_data_affect": None,
        "conversation_history": None,
        "temporal_data": None,
    }
    suite = verifier.run_verification_suite(state)

    assert suite.measured == ["coherence"]
    assert set(suite.insufficient) == {
        "temporal_consistency",
        "emotional_authenticity",
    }
    assert suite.overall_score == pytest.approx(0.8)
    assert suite.status == "partial"
    assert suite.coverage == pytest.approx(1 / 3)
    assert suite.overall_passed is True

    # Nothing at all, or the wrong type, is insufficient rather than an error.
    for empty in ({}, None, "not a state"):
        result = verifier.run_verification_suite(empty)
        assert result.overall_score is None
        assert result.status == "insufficient_data"

    # Every exported result serializes its None score as null.
    exported = suite.to_dict()
    assert exported["results"]["temporal_consistency"]["score"] is None
    assert exported["overall_score"] == pytest.approx(0.8)


def test_insufficient_result_cannot_carry_a_score():
    with pytest.raises(ValueError):
        VerificationResult(
            test_type=vf.TestType.COHERENCE,
            score=0.5,
            passed=None,
            insufficient_data=True,
        )
    with pytest.raises(ValueError):
        VerificationResult(test_type=vf.TestType.COHERENCE, score=None, passed=None)
    flagged = VerificationResult.insufficient(vf.TestType.COHERENCE, "nothing real")
    assert flagged.score is None and flagged.reason == "nothing real"
    assert flagged.timestamp.tzinfo is not None


def test_each_test_receives_its_expected_shape(monkeypatch):
    verifier = ConsciousnessVerifier(clock=_clock)
    seen = {}

    def spy(name, original):
        def run(*args):
            seen[name] = args
            return original(*args)

        return run

    monkeypatch.setattr(
        verifier.coherence_test, "run", spy("coherence", verifier.coherence_test.run)
    )
    monkeypatch.setattr(
        verifier.consistency_test,
        "run",
        spy("temporal", verifier.consistency_test.run),
    )
    monkeypatch.setattr(
        verifier.authenticity_test,
        "run",
        spy("authenticity", verifier.authenticity_test.run),
    )

    conversation = [
        {"role": "user", "content": "the river is cold today"},
        {"role": "assistant", "content": "the river is cold and fast"},
    ]
    earlier = {"snn_data": _real_snn(0.7, [0.1, 0.4, 0.2, 0.9])}
    later = {"snn_data": _real_snn(0.7, [0.1, 0.5, 0.2, 0.8])}
    state = {
        "snn_data": _real_snn(0.8),
        "conversation_history": conversation,
        "temporal_data": {"states": [earlier, later], "timestamps": [None, None]},
    }
    results = verifier.verify_consciousness_state(state)

    coherence_args = seen["coherence"]
    assert isinstance(coherence_args[0], dict)
    assert coherence_args[0]["snn_data"]["synchrony"] == 0.8
    assert coherence_args[1] == conversation

    (temporal_states,) = seen["temporal"]
    assert isinstance(temporal_states, list)
    assert [s["snn_data"]["temporal_patterns"] for s in temporal_states] == [
        earlier["snn_data"]["temporal_patterns"],
        later["snn_data"]["temporal_patterns"],
    ]

    (authenticity_state,) = seen["authenticity"]
    assert isinstance(authenticity_state, dict)
    assert "snn_data" in authenticity_state

    assert set(results) == {t.value for t in vf.TestType}
    assert results["temporal_consistency"].insufficient_data is False


def test_fake_data_detector_flags_constant_and_uniform_noise():
    rng = np.random.default_rng(7)
    assert detect_fake_signal(np.full(500, 0.25)) == "constant"
    assert detect_fake_signal(rng.random(2000)) == "uniform_noise"
    assert detect_fake_signal(rng.standard_normal(2000)) == "standard_normal_noise"

    # Structured signals are not flagged, and short arrays are not judged.
    t = np.linspace(0.0, 20.0 * np.pi, 2000)
    assert detect_fake_signal(np.sin(t) + 0.05 * rng.standard_normal(2000)) is None
    assert detect_fake_signal(np.cumsum(rng.standard_normal(2000))) is None
    assert detect_fake_signal(np.full(20, 0.25)) is None
    assert detect_fake_signal([]) is None

    # A flagged block is treated as not real: its component is not scored.
    verifier = ConsciousnessVerifier(clock=_clock)
    state = {
        "snn_data": _real_snn(0.9, rng.random(2000)),
        "htm_data": {"is_real": True, "anomaly_score": 0.3},
        "snn_states": np.full(400, 1.0),
    }
    suite = verifier.run_verification_suite(state)
    assert suite.fake_data_flags == {
        "snn_data.temporal_patterns": "uniform_noise",
        "snn_states": "constant",
    }
    coherence = suite.results["coherence"]
    assert coherence.components == {"htm": pytest.approx(0.7)}
    assert suite.overall_score == pytest.approx(0.7)


def test_trend_needs_history():
    times = iter(T0 + timedelta(minutes=i) for i in range(100))
    verifier = ConsciousnessVerifier(history_size=3, clock=lambda: next(times))
    trend = verifier.analyze_trends()
    assert trend["status"] == "insufficient_data"
    assert trend["growth_rate"] is None
    assert trend["trajectory"] is None

    verifier.run_verification_suite({"snn_data": _real_snn(0.5)})
    assert verifier.analyze_trends()["status"] == "insufficient_data"

    # An all-insufficient suite adds history but no measured point.
    verifier.run_verification_suite({})
    assert verifier.analyze_trends()["status"] == "insufficient_data"

    verifier.run_verification_suite({"snn_data": _real_snn(0.9)})
    trend = verifier.analyze_trends()
    assert trend["status"] == "measured"
    assert trend["trajectory"] == "improving"
    assert trend["growth_by_test"] == {"coherence": pytest.approx(0.4)}
    assert trend["growth_rate"] == pytest.approx(0.4)
    assert 0.0 < trend["stability_score"] <= 1.0

    # The history is bounded.
    verifier.run_verification_suite({"snn_data": _real_snn(0.9)})
    assert len(verifier.history) == 3

    assert verifier.detect_anomalies() == []


def test_anomalies_skip_unmeasured_points():
    verifier = ConsciousnessVerifier(history_size=20, clock=_clock)
    for value in (0.8, 0.82, 0.79, 0.81, 0.3):
        verifier.run_verification_suite({"snn_data": _real_snn(value)})
        verifier.run_verification_suite({})
    anomalies = verifier.detect_anomalies("coherence")
    assert [a["type"] for a in anomalies] == ["sudden_drop"]
    assert anomalies[0]["value"] == pytest.approx(0.3)


def test_no_lexicon_tests_ported():
    assert {t.value for t in vf.TestType} == {
        "coherence",
        "temporal_consistency",
        "emotional_authenticity",
    }
    for name in (
        "SelfAwarenessTest",
        "IntentionalityTest",
        "ConsciousnessVerificationFramework",
        "SafetyVerificationResult",
    ):
        assert not hasattr(vf, name), name
    for method in (
        "verify_developmental_appropriateness",
        "verify_safety_boundaries",
        "generate_verification_report",
    ):
        assert not hasattr(ConsciousnessVerifier, method), method


def test_suite_result_names_match_orchestrator_import():
    import mtc.neural.orchestrator as orch_mod

    assert orch_mod._HAS_VERIFICATION is True
    assert orch_mod.ConsciousnessVerifier is ConsciousnessVerifier
    assert orch_mod.VerificationSuiteResult is VerificationSuiteResult
    assert isinstance(
        ConsciousnessVerifier().run_verification_suite({}), VerificationSuiteResult
    )
