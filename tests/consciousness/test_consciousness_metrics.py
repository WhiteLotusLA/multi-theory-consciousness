"""The legacy 14-indicator framework: no NaN inflation, no defaulted inputs, a
failed indicator is None and stays out of the aggregate."""

import asyncio
import math

import numpy as np
import pytest

import mtc.consciousness.consciousness_metrics as metrics_mod
from mtc.consciousness.consciousness_metrics import (
    INDICATOR_NAMES,
    ConsciousnessIndicator,
    ConsciousnessMetrics,
    GlobalWorkspaceIntegration,
    IntegratedInformationPhi,
    MissingSignal,
)


def _measure(indicator, data):
    return asyncio.run(indicator.measure(data))


def _gwi_data(pattern, tokens, retrieved=(), focus_changes=0, latency=100.0):
    return {
        "neural_activity": {"activation_pattern": list(pattern)},
        "llm_processing": {"token_attention": list(tokens)},
        "memory_access": {"retrieved_memories": list(retrieved)},
        "attention_focus": {"focus_changes": focus_changes},
        "broadcast_latency_ms": latency,
    }


def _full_data():
    return {
        **_gwi_data(
            [0.1, 0.5, 0.3, 0.9], [0.2, 0.4, 0.1, 0.8], ["a", "b"], 3, latency=40.0
        ),
        "attention_focus": {
            "focus_changes": 3,
            "sustained_attention": 0.7,
            "selective_focus": 0.8,
            "divided_attention": 0.5,
            "attention_switching": 0.6,
        },
        "system_state": {
            "active_components": ["snn", "lsm", "htm"],
            "partition_independence": 0.6,
        },
        "causal_connections": {"active_links": [("snn", "lsm"), ("lsm", "htm")]},
        "working_memory": {"items_in_memory": 6, "successful_manipulations": 3},
        "episodic_memory": {
            "recall_attempts": 5,
            "successful_recalls": 4,
            "temporal_ordering_score": 0.8,
            "detail_richness": 0.7,
        },
        "metacognition": {
            "self_monitoring_score": 0.6,
            "reflection_depth": 0.7,
            "error_self_detection": 0.5,
            "confidence_calibration": 0.8,
        },
        "agency": {
            "self_generated_goals": 2,
            "autonomous_decision_ratio": 0.6,
            "self_initiated_actions": 0.5,
        },
        "temporal_continuity": {
            "identity_consistency": 0.8,
            "narrative_coherence": 0.7,
            "future_planning_score": 0.6,
        },
        "embodiment": {
            "boundary_awareness": 0.7,
            "spatial_presence": 0.6,
            "self_other_distinction": 0.8,
        },
        "affective_processing": {
            "emotion_recognition_accuracy": 0.8,
            "emotion_regulation_score": 0.7,
            "emotional_coherence": 0.8,
            "empathy_score": 0.6,
        },
        "social_cognition": {
            "theory_of_mind_score": 0.7,
            "perspective_taking": 0.6,
            "social_context_score": 0.8,
        },
        "unified_experience": {
            "feature_binding_score": 0.7,
            "experience_coherence": 0.8,
            "gestalt_formation": 0.6,
        },
        "flexible_reasoning": {
            "abstract_reasoning_score": 0.7,
            "cognitive_flexibility": 0.6,
            "creative_solutions": 0.5,
        },
        "adaptive_learning": {
            "learning_efficiency": 0.8,
            "knowledge_transfer_score": 0.6,
            "novelty_adaptation": 0.7,
        },
    }


def test_gwi_constant_vector_does_not_inflate():
    data = _gwi_data([0.5] * 10, [0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5, 0.5])
    score = _measure(GlobalWorkspaceIntegration(), data)
    assert score == pytest.approx(0.0)
    assert not math.isnan(score)


def test_gwi_mismatched_lengths_truncate():
    pattern = [0.1, 0.5, 0.3, 0.9, 0.7]
    tokens = [0.2, 0.4, 0.1]
    score = _measure(GlobalWorkspaceIntegration(), _gwi_data(pattern, tokens))
    expected = abs(np.corrcoef(pattern[:3], tokens)[0, 1]) * 0.3
    assert score == pytest.approx(expected)


def test_gwi_missing_token_attention_contributes_zero():
    data = _gwi_data([0.1, 0.5, 0.3], [], retrieved=["m1"], focus_changes=4, latency=50)
    score = _measure(GlobalWorkspaceIntegration(), data)
    assert score == pytest.approx(0.5 * 0.3 + 0.5 * 0.4)


@pytest.mark.parametrize("name", INDICATOR_NAMES)
def test_each_indicator_raises_on_empty_input(name):
    indicator = ConsciousnessMetrics().indicators[name]
    with pytest.raises(MissingSignal):
        _measure(indicator, {})


def test_require_reports_every_missing_path():
    with pytest.raises(MissingSignal) as caught:
        ConsciousnessIndicator._require({"a": {"b": 1, "c": None}}, "a.b", "a.c", "d.e")
    assert "a.c" in str(caught.value) and "d.e" in str(caught.value)
    assert "a.b" not in str(caught.value)
    assert ConsciousnessIndicator._require({"a": {"b": 0}}, "a.b") == [0]


def test_iit_legacy_indicator_needs_partition_measure():
    data = {
        "system_state": {"active_components": ["snn", "lsm", "htm"]},
        "causal_connections": {"active_links": [("snn", "lsm")]},
    }
    with pytest.raises(MissingSignal, match="partition_independence"):
        _measure(IntegratedInformationPhi(), data)

    data["system_state"]["partition_independence"] = 1.0
    score = _measure(IntegratedInformationPhi(), data)
    assert score == pytest.approx((1 / 3) * 1.0 * 0.8)


def test_iit_links_and_components_share_element_space():
    components = ["snn", "lsm", "htm"]
    every_pair = [("snn", "lsm"), ("snn", "htm"), ("lsm", "htm")]
    data = {
        "system_state": {
            "active_components": components,
            "partition_independence": 1.0,
        },
        "causal_connections": {"active_links": every_pair},
    }
    assert _measure(IntegratedInformationPhi(), data) == pytest.approx(0.8)

    # Reversed duplicates, self links and links to elements that are not
    # components (oscillator indices) add nothing.
    data["causal_connections"]["active_links"] = [
        ("snn", "lsm"),
        ("lsm", "snn"),
        ("snn", "snn"),
        (3, 17),
        ("htm", "workspace"),
    ]
    assert _measure(IntegratedInformationPhi(), data) == pytest.approx((1 / 3) * 0.8)


def test_metrics_failed_indicator_is_none_not_zero():
    data = _full_data()
    del data["working_memory"]
    del data["system_state"]["partition_independence"]
    assessment = asyncio.run(ConsciousnessMetrics().assess_consciousness(data))

    assert assessment.working_memory_capacity is None
    assert assessment.integrated_information_phi is None
    assert set(assessment.failed_indicators) == {
        "working_memory_capacity",
        "integrated_information_phi",
    }
    measured = {
        name: getattr(assessment, name)
        for name in INDICATOR_NAMES
        if getattr(assessment, name) is not None
    }
    assert len(measured) == 12
    weights = metrics_mod.INDICATOR_WEIGHTS
    expected = sum(weights[n] * s for n, s in measured.items()) / sum(
        weights[n] for n in measured
    )
    assert assessment.overall_consciousness_score == pytest.approx(expected)


def test_metrics_full_panel_keeps_weighted_mean():
    assessment = asyncio.run(ConsciousnessMetrics().assess_consciousness(_full_data()))
    assert assessment.failed_indicators == {}
    scores = [getattr(assessment, name) for name in INDICATOR_NAMES]
    assert all(score is not None for score in scores)
    weights = [metrics_mod.INDICATOR_WEIGHTS[name] for name in INDICATOR_NAMES]
    assert sum(weights) == pytest.approx(14.5)
    expected = sum(w * s for w, s in zip(weights, scores)) / 14.5
    assert assessment.overall_consciousness_score == pytest.approx(expected)


def test_metrics_nothing_measured_is_none():
    assessment = asyncio.run(ConsciousnessMetrics().assess_consciousness({}))
    assert assessment.overall_consciousness_score is None
    assert len(assessment.failed_indicators) == 14
    analysis = ConsciousnessMetrics()
    analysis.assessment_history.append(assessment)
    trends = analysis.get_longitudinal_analysis(hours=1)["trends"]
    assert trends["overall"]["measured_samples"] == 0
    assert trends["global_workspace_integration"]["mean"] is None


def test_indicator_count_is_fourteen():
    metrics = ConsciousnessMetrics()
    assert len(metrics.indicators) == 14
    assert tuple(metrics.indicators) == INDICATOR_NAMES
    doc = metrics_mod.__doc__.lower()
    assert "legacy 14-indicator" in doc
    assert "30-indicator" in doc
    assert "20 " not in doc
