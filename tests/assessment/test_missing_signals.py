"""Missing-signal semantics: measured, absent and insufficient_data."""

import numpy as np
import pytest

from mtc.assessment import assessment as assessment_module
from mtc.assessment.assessment import (
    ABSENT,
    INSUFFICIENT_DATA,
    MEASURED,
    ConsciousnessAssessment,
)

# The v0.4.0 numbers of the fixture below, captured from the committed
# v0.4.0 assessment (scores, confidences, passes).
V04_FIXTURE_EXPECTED = {
    "global_broadcast": (0.55, 0.4, True),
    "ignition_dynamics": (0.5, 0.7, True),
    "global_ignition_nuanced": (0.5449999999999999, 0.6, True),
    "attention_schema": (0.655, 0.8, True),
    "attention_control": (0.4, 0.6, True),
    "higher_order_representations": (0.6666666666666666, 0.7, True),
    "metacognition": (0.5266666666666666, 0.6, True),
    "sparse_smooth_coding": (0.30000000000000004, 0.5, True),
    "prediction_error_minimization": (0.405, 0.6, True),
    "hierarchical_prediction": (0.6, 0.6, True),
    "temporal_prediction_depth": (0.26, 0.7, False),
    "agent_environment_coupling": (0.40987803063838396, 0.8, True),
    "affect_coherence": (0.62, 0.725, True),
    "integrated_information": (0.41016438174693864, 0.7, True),
    "irreducibility": (0.6881620746666838, 0.5, True),
    "causal_emergence": (0.6224593312018546, 0.7, True),
    "recurrent_processing": (0.65, 0.7, True),
    "algorithmic_recurrence": (0.376, 0.8, True),
    "local_recurrence": (0.42, 0.7, True),
    "oscillatory_binding_coherence": (0.52, 0.8, True),
    "agency": (0.8400000000000001, 0.6, True),
    "embodiment": (0.76, 0.7, True),
    "bayesian_binding_quality": (0.45, 0.8, True),
    "epistemic_depth": (0.4, 0.6, True),
    "genuine_implementation": (1.0, 0.6799999999999999, True),
    "sequential_module_querying": (0.9, 0.8, True),
    "organized_percepts": (0.51, 0.6000000000000001, True),
    "flexible_responsiveness": (0.6833333333333333, 0.5, True),
    "online_reportable_learning": (0.73, 0.6000000000000001, True),
    "complex_agency": (0.44999999999999996, 0.5, True),
}
V04_FIXTURE_OVERALL = 0.569037580589282
V04_FIXTURE_PASSING = 29
V04_FIXTURE_CONFIDENCE = 0.6535000000000001

# The legacy_v04 preset's assessment values
V04_SETTINGS = {
    "saturated_baseline_min_range": 0.0,
    "indicator_formulas": "v04",
    "missing_signal_policy": "v04_constants",
    "irreducibility_method": "spectral_entropy",
    "causal_emergence_threshold": 0.0,
    "phi_source": "calculator",
}


# ---------------------------------------------------------------------------
# A deterministic fixture holding both the producers' keys and the v0.4.0
# keys the legacy formulas read.
# ---------------------------------------------------------------------------


class FixtureWorkspace:
    def get_statistics(self):
        return {
            "cycle_count": 12,
            "broadcast": {"total_broadcasts": 9, "coverage_ratio": 0.55},
            "ignition": {"total_ignitions": 6, "sustain_rate": 0.5, "decay_rate": 0.2},
            "competition": {
                "total_candidates": 8,
                "source_counts": {"sensory": 3, "memory": 1},
            },
            "bottleneck": {
                "total_competitions": 12,
                "avg_candidates": 6.0,
                "source_distribution": {"sensory": 20, "memory": 8, "concept": 4},
            },
        }


class FixtureSchema:
    current_focus = "focus"
    attention_capacity_used = 0.4


class FixtureAttention:
    schema = FixtureSchema()

    def get_statistics(self):
        return {
            "total_updates": 10,
            "history_length": 6,
            "prediction_accuracy": 0.45,
            "capacity_used": 0.4,
            "voluntary_shifts": 4,
            "voluntary_ratio": 0.4,
        }


class FixtureMeta:
    def get_statistics(self):
        return {
            "total_hots_generated": 8,
            "introspection_count": 3,
            "belief_evaluations": 2,
            "overall_confidence": 0.6,
            "first_order_buffer_size": 8,
        }


class FixtureConfig:
    planning_horizon = 4


class FixtureInference:
    config = FixtureConfig()
    learning_rate = 0.1

    def get_statistics(self):
        return {
            "total_inferences": 15,
            "total_model_updates": 7,
            "avg_prediction_error": 0.35,
            "prediction_error_trend": -0.06,
            "hierarchical_prediction_error": 0.4,
            "urgency_level": 0.3,
            "variational_free_energy": 1.5,
            "active_policies": 3,
        }


class FixtureLoop:
    def get_loop_statistics(self):
        return {
            "cycle_count": 12,
            "average_loop_quality": 0.5,
            "average_binding_quality": 0.45,
            "epistemic_depth": {
                "average_depth": 1.6,
                "max_depth_reached": 3,
                "peak_depth": 3,
            },
            "field_evidencing_count": 2,
            "field_evidencing_rate": 0.17,
        }


class FixtureRPT:
    def get_statistics(self):
        return {"average_local": 0.42, "average_global": 0.31, "measurement_count": 12}

    def get_oscillatory_binding_stats(self):
        return {
            "global_order_parameter": 0.6,
            "cross_coherence": {"snn_lsm": 0.3, "snn_htm": 0.5},
            "binding_count": 12,
        }


class FixturePAD:
    def __init__(self):
        self._history = [None] * 15

    def compute_coherence(self):
        return 0.62

    def get_latest(self):
        return None


class FixtureCE:
    def __init__(self, value=0.25):
        self.value = value

    def get_latest_ce(self):
        return self.value


def fixture_modules():
    rng = np.random.default_rng(3)
    raw = rng.random((10, 10))
    connectivity = (raw + raw.T) / 2.0
    np.fill_diagonal(connectivity, 1.0)
    return {
        "global_workspace": FixtureWorkspace(),
        "attention_schema": FixtureAttention(),
        "metacognition": FixtureMeta(),
        "active_inference": FixtureInference(),
        "beautiful_loop": FixtureLoop(),
        "rpt_measurement": FixtureRPT(),
        "pad_affect_model": FixturePAD(),
        "causal_emergence_analyzer": FixtureCE(),
        "neural_states": {
            "combined_state": np.linspace(0.1, 0.9, 10),
            "connectivity": connectivity,
        },
    }


class _ZeroWorkspace:
    """A workspace that was supplied but has run no cycle."""

    def get_statistics(self):
        return {
            "cycle_count": 0,
            "broadcast": {"total_broadcasts": 0, "coverage_ratio": 0.0},
            "ignition": {"total_ignitions": 0},
            "bottleneck": {"total_competitions": 0},
        }


class _ZeroMeta:
    def get_statistics(self):
        return {
            "total_hots_generated": 0,
            "introspection_count": 0,
            "first_order_buffer_size": 0,
            "overall_confidence": 0.5,
        }


class _ZeroInference:
    def get_statistics(self):
        return {
            "total_inferences": 0,
            "total_model_updates": 0,
            "avg_prediction_error": 0.0,
            "prediction_error_trend": "stable",
            "hierarchical_prediction_error": 0.0,
            "variational_free_energy": 0.0,
        }


def _fresh_modules():
    """Real modules, freshly built, with no cycle run."""
    pytest.importorskip("pymdp")
    from mtc.assessment.rpt_measurement import RPTMeasurement
    from mtc.consciousness.active_inference import ActiveInferenceModule
    from mtc.consciousness.attention_schema import AttentionSchemaModule
    from mtc.consciousness.beautiful_loop import BeautifulLoop
    from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace
    from mtc.consciousness.metacognition import MetacognitionModule
    from mtc.consciousness.pad_affect import PADAffectModel
    from mtc.consciousness.phi_tracker import PhiTracker

    class _NoResultYet:
        def get_latest_ce(self):
            return None

    return {
        "global_workspace": EnhancedGlobalWorkspace(),
        "attention_schema": AttentionSchemaModule(),
        "metacognition": MetacognitionModule(),
        "active_inference": ActiveInferenceModule(),
        "beautiful_loop": BeautifulLoop(),
        "rpt_measurement": RPTMeasurement(),
        "pad_affect_model": PADAffectModel(),
        "causal_emergence_analyzer": _NoResultYet(),
        "phi_tracker": PhiTracker(),
    }


def _weighted_mean(assessment, report, names):
    total = 0.0
    weighted = 0.0
    for name in names:
        cfg = assessment.indicator_configs[name]
        weight = cfg["weight"] * assessment.theory_empirical_weights.get(
            cfg["theory"].name, 0.5
        )
        weighted += report.indicator_results[name].score * weight
        total += weight
    return weighted / total


@pytest.mark.asyncio
async def test_no_indicator_fabricates_a_score_before_any_cycle():
    modules = _fresh_modules()
    report = await ConsciousnessAssessment().run_full_assessment(
        **modules, normalize=False
    )
    statuses = {name: r.status for name, r in report.indicator_results.items()}
    assert MEASURED not in statuses.values(), [
        n for n, s in statuses.items() if s == MEASURED
    ]
    assert statuses.pop("irreducibility") == ABSENT
    assert set(statuses.values()) == {INSUFFICIENT_DATA}
    assert report.status == INSUFFICIENT_DATA
    assert report.coverage == 0.0
    assert report.passing_count == 0
    assert sorted(report.insufficient_indicators) == sorted(statuses)
    for neg in report.negative_indicators.values():
        assert neg["status"] == INSUFFICIENT_DATA
        assert neg["penalty"] == 0.0
    exported = report.to_dict()
    assert all(
        exported["indicator_results"][name]["score"] is None for name in statuses
    )


@pytest.mark.asyncio
async def test_absent_modules_score_zero_with_status_absent():
    report = await ConsciousnessAssessment().run_full_assessment(normalize=False)
    for name, result in report.indicator_results.items():
        assert result.status == ABSENT, name
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.passes_threshold is False
    assert report.overall_score == 0.0
    assert set(report.theory_scores) == {
        "GWT",
        "IIT",
        "AST",
        "HOT",
        "FEP",
        "RPT",
        "BLT",
        "AE",
    }
    assert all(score == 0.0 for score in report.theory_scores.values())
    assert report.passing_count == 0
    # Missing modules keep their rule levels (an ablation is penalized)
    assert report.negative_indicators["purely_reactive"]["status"] == ABSENT
    assert report.negative_indicators["purely_reactive"]["score"] == 0.8


@pytest.mark.asyncio
async def test_insufficient_indicators_excluded_from_averages():
    class _ZeroLoop:
        def get_loop_statistics(self):
            return {
                "cycle_count": 0,
                "average_loop_quality": 0.0,
                "average_binding_quality": 0.0,
                "field_evidencing_count": 0,
                "epistemic_depth": {"average_depth": 0.0, "peak_depth": 0},
            }

    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        global_workspace=FixtureWorkspace(),
        beautiful_loop=_ZeroLoop(),
        normalize=False,
    )
    included = [
        name
        for name, r in report.indicator_results.items()
        if r.status in (MEASURED, ABSENT)
    ]
    insufficient = [
        name
        for name, r in report.indicator_results.items()
        if r.status == INSUFFICIENT_DATA
    ]
    assert {"bayesian_binding_quality", "epistemic_depth"} <= set(insufficient)
    assert "BLT" not in report.theory_scores
    penalty = sum(v["penalty"] for v in report.negative_indicators.values())
    expected = max(0.0, _weighted_mean(assessment, report, included) - penalty)
    assert report.overall_score == pytest.approx(expected)
    for theory, score in report.theory_scores.items():
        members = [
            report.indicator_results[n].score
            for n in included
            if report.indicator_results[n].theory.name == theory
        ]
        assert score == pytest.approx(float(np.mean(members)))


@pytest.mark.asyncio
async def test_overall_none_only_when_nothing_included():
    modules = _fresh_modules()
    # Neural states without connectivity or a combined state: present, not
    # usable, so even irreducibility is insufficient.
    report = await ConsciousnessAssessment().run_full_assessment(
        **modules, neural_states={"snn_spikes": np.zeros(4)}, normalize=False
    )
    assert all(r.status == INSUFFICIENT_DATA for r in report.indicator_results.values())
    assert report.overall_score is None
    assert report.status == INSUFFICIENT_DATA
    assert report.architecture_functional is False
    assert report.theory_scores == {}
    assert report.to_dict()["overall_score"] is None

    all_absent = await ConsciousnessAssessment().run_full_assessment(normalize=False)
    assert all_absent.overall_score == 0.0


@pytest.mark.asyncio
async def test_confidence_scales_with_coverage():
    assessment = ConsciousnessAssessment()
    partial = await assessment.run_full_assessment(
        global_workspace=FixtureWorkspace(), normalize=False
    )
    full = await assessment.run_full_assessment(**fixture_modules(), normalize=False)
    for report in (partial, full):
        measured = [
            r.confidence
            for r in report.indicator_results.values()
            if r.status == MEASURED
        ]
        coverage = len(measured) / report.total_indicators
        assert report.coverage == pytest.approx(coverage)
        assert report.confidence == pytest.approx(float(np.mean(measured)) * coverage)
    assert partial.coverage < full.coverage
    assert partial.status == "partial"


@pytest.mark.asyncio
async def test_negative_indicator_without_data_applies_no_penalty():
    report = await ConsciousnessAssessment().run_full_assessment(
        global_workspace=_ZeroWorkspace(),
        metacognition=_ZeroMeta(),
        active_inference=_ZeroInference(),
        normalize=False,
    )
    for name in ("purely_reactive", "fixed_repertoire", "no_metacognitive_access"):
        neg = report.negative_indicators[name]
        assert neg["status"] == INSUFFICIENT_DATA, name
        assert neg["score"] is None
        assert neg["penalty"] == 0.0
    # Under v04_constants the same modules carry the v0.4.0 rule levels.
    legacy = await ConsciousnessAssessment(**V04_SETTINGS).run_full_assessment(
        global_workspace=_ZeroWorkspace(),
        metacognition=_ZeroMeta(),
        active_inference=_ZeroInference(),
        normalize=False,
    )
    assert legacy.negative_indicators["purely_reactive"]["score"] == 0.5
    assert legacy.negative_indicators["fixed_repertoire"]["score"] == 0.4


@pytest.mark.asyncio
async def test_v04_constants_policy_reproduces_old_values():
    assessment = ConsciousnessAssessment(**V04_SETTINGS)
    absent = await assessment.run_full_assessment(normalize=False)
    for name, result in absent.indicator_results.items():
        expected = (
            0.0
            if name in ("integrated_information", "irreducibility", "causal_emergence")
            else 0.1
        )
        assert result.score == expected, name
    assert absent.overall_score == 0.0
    # v0.4.0 passed causal emergence at 0.0 >= 0.0 even with no analyzer
    assert absent.passing_count == 1

    report = await assessment.run_full_assessment(**fixture_modules(), normalize=False)
    for name, (score, confidence, passes) in V04_FIXTURE_EXPECTED.items():
        result = report.indicator_results[name]
        assert result.score == score, name
        assert result.confidence == confidence, name
        assert result.passes_threshold is passes, name
    assert report.overall_score == V04_FIXTURE_OVERALL
    assert report.passing_count == V04_FIXTURE_PASSING
    assert report.confidence == V04_FIXTURE_CONFIDENCE


class _AlternatingRPT:
    """Every second noise iteration has no measurements yet."""

    created = 0

    def __init__(self):
        type(self).created += 1
        self.active = type(self).created % 2 == 0

    def get_statistics(self):
        return {
            "average_local": 0.6,
            "average_global": 0.2,
            "measurement_count": 3 if self.active else 0,
        }

    def get_oscillatory_binding_stats(self):
        return {}


class _NeverReadyAnalyzer:
    def get_latest_ce(self):
        return None


@pytest.mark.asyncio
async def test_noise_baseline_skips_insufficient_runs(monkeypatch):
    _AlternatingRPT.created = 0
    monkeypatch.setattr(assessment_module, "_NoiseRPTMeasurement", _AlternatingRPT)
    monkeypatch.setattr(
        assessment_module, "_NoiseCausalEmergenceAnalyzer", _NeverReadyAnalyzer
    )
    baselines = await ConsciousnessAssessment().run_noise_baseline(iterations=4)
    # Only the two measured iterations count: 0.6, not 0.3
    assert baselines["local_recurrence"] == pytest.approx(0.6)
    assert "oscillatory_binding_coherence" not in baselines
    assert "causal_emergence" not in baselines
    assert "global_broadcast" in baselines


def test_invalid_choices_name_the_field():
    for field_name, value in (
        ("indicator_formulas", "v03"),
        ("missing_signal_policy", "zeros"),
        ("irreducibility_method", "rank"),
        ("phi_source", "oracle"),
        ("phi_normalization", "log"),
    ):
        with pytest.raises(ValueError, match=field_name):
            ConsciousnessAssessment(**{field_name: value})
    with pytest.raises(ValueError, match="causal_emergence_threshold"):
        ConsciousnessAssessment(causal_emergence_threshold=1.5)
    with pytest.raises(ValueError, match="phi_fixed_reference"):
        ConsciousnessAssessment(phi_fixed_reference=0.0)


@pytest.mark.asyncio
async def test_hot_without_workspace_is_absent_not_insufficient():
    # No workspace supplied: HOT coverage and the metacognition activity term
    # have no workspace to be measured against, so both read absent (0.0)
    # and stay in the averages, as an ablation compares.
    report = await ConsciousnessAssessment().run_full_assessment(
        metacognition=FixtureMeta(), normalize=False
    )
    hor = report.indicator_results["higher_order_representations"]
    assert hor.status == ABSENT
    assert hor.score == 0.0
    assert "higher_order_representations" not in report.insufficient_indicators
    meta = report.indicator_results["metacognition"]
    assert meta.status == MEASURED
    assert meta.evidence["activity_status"] == ABSENT
    assert meta.evidence["activity_rate"] == 0.0
    # Only the confidence term counts: 0.6 x 0.6, no invented denominator
    assert meta.score == pytest.approx(0.6 * 0.6)
    assert report.indicator_results["global_broadcast"].status == ABSENT


@pytest.mark.asyncio
async def test_hot_with_idle_workspace_is_insufficient():
    report = await ConsciousnessAssessment().run_full_assessment(
        global_workspace=_ZeroWorkspace(), metacognition=FixtureMeta(), normalize=False
    )
    for name in ("higher_order_representations", "metacognition"):
        result = report.indicator_results[name]
        assert result.status == INSUFFICIENT_DATA, name
        assert result.evidence["missing"] == "global_workspace.cycle_count"
