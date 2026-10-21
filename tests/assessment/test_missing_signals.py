"""Missing-signal semantics: measured, absent and insufficient_data."""

import copy

import numpy as np
import pytest

from mtc.assessment import assessment as assessment_module
from mtc.assessment.assessment import (
    ABSENT,
    INSUFFICIENT_DATA,
    MEASURED,
    ConsciousnessAssessment,
)
from tests.baseline_compare import close_enough

# The v0.4.0 numbers of the fixture below, captured from the committed
# v0.4.0 assessment (scores, confidences, passes), compared within the
# fixture comparator's tolerance: the host's Linux x64 runners read
# integrated_information as 0.4101643817469391 against the
# 0.41016438174693864 captured on macOS arm64 (the first CI run on the
# pushed v0.5.0 main).
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
    # Nothing measured: no overall and no theory score (absent indicators
    # alone are not a measurement of anything)
    assert report.overall_score is None
    assert report.theory_scores == {}
    assert report.status == INSUFFICIENT_DATA
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


@pytest.mark.asyncio
async def test_overall_none_when_nothing_measured():
    # Absent indicators are included in the averages, but an average over
    # nothing measured is not a number: the overall is None and the report
    # exports it as null.
    all_absent = await ConsciousnessAssessment().run_full_assessment(normalize=False)
    assert all(r.status == ABSENT for r in all_absent.indicator_results.values())
    assert all_absent.overall_score is None
    assert all_absent.theory_scores == {}
    assert all_absent.to_dict()["overall_score"] is None
    assert all_absent.architecture_functional is False
    assert isinstance(all_absent.passing_count, int)
    # Present modules without data plus absent ones: still nothing measured
    idle = await ConsciousnessAssessment().run_full_assessment(
        global_workspace=_ZeroWorkspace(), normalize=False
    )
    assert MEASURED not in {r.status for r in idle.indicator_results.values()}
    assert idle.overall_score is None
    assert idle.theory_scores == {}


@pytest.mark.asyncio
async def test_one_measured_indicator_gives_a_number():
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        causal_emergence_analyzer=FixtureCE(0.25), normalize=False
    )
    measured = [n for n, r in report.indicator_results.items() if r.status == MEASURED]
    assert measured == ["causal_emergence"]
    included = [
        name
        for name, r in report.indicator_results.items()
        if r.status in (MEASURED, ABSENT)
    ]
    assert len(included) == len(report.indicator_results)
    # The one measured indicator and every absent one enter the weighted mean
    # (the three absent negative indicators then take their penalties)
    mean = _weighted_mean(assessment, report, included)
    assert mean > 0.0
    penalty = sum(v["penalty"] for v in report.negative_indicators.values())
    assert report.overall_score is not None
    assert report.overall_score == pytest.approx(max(0.0, mean - penalty))
    assert set(report.theory_scores) == {"IIT"}
    iit = [
        report.indicator_results[n].score
        for n in included
        if report.indicator_results[n].theory.name == "IIT"
    ]
    assert report.theory_scores["IIT"] == pytest.approx(float(np.mean(iit)))


@pytest.mark.asyncio
async def test_theory_without_a_measured_member_is_omitted():
    # The RPT indicators are measured, the loop is present but idle (its
    # indicators are insufficient) and every other module is absent: only
    # RPT has a measured member.
    class _ZeroLoop:
        def get_loop_statistics(self):
            return {
                "cycle_count": 0,
                "average_loop_quality": 0.0,
                "average_binding_quality": 0.0,
                "field_evidencing_count": 0,
                "epistemic_depth": {"average_depth": 0.0, "peak_depth": 0},
            }

    report = await ConsciousnessAssessment().run_full_assessment(
        rpt_measurement=FixtureRPT(), beautiful_loop=_ZeroLoop(), normalize=False
    )
    statuses = {n: r.status for n, r in report.indicator_results.items()}
    assert statuses["algorithmic_recurrence"] == MEASURED
    assert statuses["global_broadcast"] == ABSENT
    assert statuses["bayesian_binding_quality"] == INSUFFICIENT_DATA
    assert set(report.theory_scores) == {"RPT"}
    rpt_members = [
        r.score
        for r in report.indicator_results.values()
        if r.theory.name == "RPT" and r.status in (MEASURED, ABSENT)
    ]
    assert report.theory_scores["RPT"] == pytest.approx(float(np.mean(rpt_members)))
    assert report.overall_score is not None


@pytest.mark.asyncio
async def test_v04_constants_keeps_overall_and_theory_scores_with_nothing_measured():
    legacy = ConsciousnessAssessment(**V04_SETTINGS)
    absent = await legacy.run_full_assessment(normalize=False)
    assert absent.overall_score == 0.0
    assert set(absent.theory_scores) == {
        "GWT",
        "IIT",
        "AST",
        "HOT",
        "FEP",
        "RPT",
        "BLT",
        "AE",
    }
    # The v0.4.0 constants: 0.1 for an absent module, 0.0 for the IIT three
    assert absent.theory_scores["IIT"] == pytest.approx(0.0)
    assert absent.theory_scores["GWT"] == pytest.approx(0.1)
    constants = ConsciousnessAssessment(missing_signal_policy="v04_constants")
    report = await constants.run_full_assessment(normalize=False)
    assert report.overall_score == 0.0
    assert len(report.theory_scores) == 8


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
        assert close_enough(result.score, score), (name, result.score, score)
        assert close_enough(result.confidence, confidence), (name, result.confidence)
        assert result.passes_threshold is passes, name
    assert close_enough(report.overall_score, V04_FIXTURE_OVERALL), report.overall_score
    assert report.passing_count == V04_FIXTURE_PASSING
    assert close_enough(report.confidence, V04_FIXTURE_CONFIDENCE), report.confidence


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


# ---------------------------------------------------------------------------
# Real producers that leave a statistic unmeasured
# ---------------------------------------------------------------------------


def _attention_state(summary, module, salience=0.8):
    """A workspace state whose focus moved to ``module`` (a habitual shift)."""
    from mtc.consciousness.enhanced_global_workspace import (
        ConsciousnessState,
        WorkspaceCandidate,
        WorkspaceCandidateSource,
        WorkspaceContent,
    )

    candidate = WorkspaceCandidate(
        content=summary,
        content_type="thought",
        summary=summary,
        source=WorkspaceCandidateSource.CTM,
        source_module=module,
    )
    content = WorkspaceContent(candidate=candidate, salience=salience)
    return ConsciousnessState(
        is_conscious=True,
        primary_content=content,
        workspace_contents=[content],
        ignition_events=1,
        integration_level=0.5,
        broadcast_coverage=1.0,
        attention_focus=module,
        attention_distribution={module: salience},
        stream_position=1,
    )


async def _attention_after_ordinary_updates(updates=6):
    from mtc.consciousness.attention_schema import AttentionSchemaModule

    schema = AttentionSchemaModule()
    for i in range(updates):
        await schema.update_schema(_attention_state(f"topic {i}", f"module_{i}"))
    return schema


async def _workspace_after_ignitions(cycles=7):
    from mtc.consciousness.enhanced_global_workspace import (
        EnhancedGlobalWorkspace,
        WorkspaceCandidate,
        WorkspaceCandidateSource,
    )

    workspace = EnhancedGlobalWorkspace(body_metrics_enabled=False)
    for i in range(cycles):
        candidate = WorkspaceCandidate(
            content=f"message number {i}",
            content_type="conversation",
            summary=f"message number {i}",
            source=WorkspaceCandidateSource.LLM,
            activation_level=0.9,
            emotional_salience=0.5,
        )
        await workspace.process_consciousness_cycle([candidate])
    stats = workspace.get_statistics()
    assert _count_of(stats["ignition"].get("total_ignitions")) > 0
    return workspace


def _count_of(value):
    return int(value) if isinstance(value, (int, float)) else 0


@pytest.mark.asyncio
async def test_embodiment_insufficient_without_a_classifiable_shift():
    schema = await _attention_after_ordinary_updates()
    stats = schema.get_statistics()
    assert stats["total_updates"] == 6
    assert stats["voluntary_shifts"] == 0 and stats["captured_shifts"] == 0
    assert stats["voluntary_ratio"] is None
    report = await ConsciousnessAssessment().run_full_assessment(
        attention_schema=schema, normalize=False
    )
    result = report.indicator_results["embodiment"]
    assert result.status == INSUFFICIENT_DATA
    assert result.evidence["missing"] == "attention_schema.voluntary_ratio"
    constants = await ConsciousnessAssessment(
        missing_signal_policy="v04_constants"
    ).run_full_assessment(attention_schema=schema, normalize=False)
    legacy = constants.indicator_results["embodiment"]
    assert legacy.status == MEASURED
    assert legacy.evidence["voluntary_ratio"] == 0.5


@pytest.mark.asyncio
async def test_attention_schema_insufficient_without_a_scored_prediction():
    schema = await _attention_after_ordinary_updates()
    assert schema.get_statistics()["prediction_accuracy"] is None
    report = await ConsciousnessAssessment().run_full_assessment(
        attention_schema=schema, normalize=False
    )
    result = report.indicator_results["attention_schema"]
    assert result.status == INSUFFICIENT_DATA
    assert result.evidence["missing"] == "attention_schema.prediction_accuracy"
    constants = await ConsciousnessAssessment(
        missing_signal_policy="v04_constants"
    ).run_full_assessment(attention_schema=schema, normalize=False)
    legacy = constants.indicator_results["attention_schema"]
    assert legacy.status == MEASURED
    assert legacy.evidence["prediction_accuracy"] == 0.0


@pytest.mark.asyncio
async def test_global_broadcast_insufficient_without_receivers():
    workspace = await _workspace_after_ignitions()
    stats = workspace.get_statistics()
    assert stats["broadcast"]["coverage_ratio"] is None
    report = await ConsciousnessAssessment().run_full_assessment(
        global_workspace=workspace, normalize=False
    )
    for name in ("global_broadcast", "recurrent_processing"):
        result = report.indicator_results[name]
        assert result.status == INSUFFICIENT_DATA, name
        assert result.evidence["missing"] == "global_workspace.broadcast.coverage_ratio"
        assert name in report.insufficient_indicators
    constants = await ConsciousnessAssessment(
        missing_signal_policy="v04_constants"
    ).run_full_assessment(global_workspace=workspace, normalize=False)
    assert constants.indicator_results["global_broadcast"].status == MEASURED
    assert constants.indicator_results["global_broadcast"].evidence["coverage"] == 0.0


@pytest.mark.asyncio
async def test_ignition_indicators_insufficient_without_sustain_check():
    workspace = await _workspace_after_ignitions()
    stats = workspace.get_statistics()
    assert stats["ignition"]["sustain_rate"] is None
    report = await ConsciousnessAssessment().run_full_assessment(
        global_workspace=workspace, normalize=False
    )
    for name in ("ignition_dynamics", "global_ignition_nuanced"):
        result = report.indicator_results[name]
        assert result.status == INSUFFICIENT_DATA, name
        assert result.evidence["missing"] == "global_workspace.ignition.sustain_rate"
    # A producer that did measure a sustain rate stays measured
    stub = await ConsciousnessAssessment().run_full_assessment(
        global_workspace=FixtureWorkspace(), normalize=False
    )
    for name in ("ignition_dynamics", "global_ignition_nuanced"):
        assert stub.indicator_results[name].status == MEASURED, name
        assert stub.indicator_results[name].evidence["sustain_rate"] == 0.5


@pytest.mark.asyncio
async def test_binding_quality_insufficient_without_candidate_vectors():
    from types import SimpleNamespace

    from mtc.consciousness.beautiful_loop import BeautifulLoop

    def winners(saliences):
        return [
            SimpleNamespace(
                candidate=SimpleNamespace(
                    id=f"c{i}",
                    summary=f"content {i}",
                    content_type="thought",
                    source_module=f"m{i}",
                ),
                salience=salience,
            )
            for i, salience in enumerate(saliences)
        ]

    loop = BeautifulLoop()
    for saliences in ([0.9, 0.4, 0.6], [0.3, 0.8], [0.7, 0.2, 0.5, 0.6]):
        moment = await loop.process_conscious_moment(
            workspace_winners=winners(saliences)
        )
        assert moment.binding_quality is None
    stats = loop.get_loop_statistics()
    assert stats["cycle_count"] == 3
    assert stats["average_binding_quality"] is None
    report = await ConsciousnessAssessment().run_full_assessment(
        beautiful_loop=loop, normalize=False
    )
    result = report.indicator_results["bayesian_binding_quality"]
    assert result.status == INSUFFICIENT_DATA
    assert result.evidence["missing"] == "beautiful_loop.average_binding_quality"
    # The loop's other indicators keep measuring
    assert report.indicator_results["epistemic_depth"].status == MEASURED


@pytest.mark.asyncio
async def test_indicator_failure_is_counted_and_raised_under_debug(monkeypatch):
    # A clean run carries an empty mapping, not a missing key
    clean = await ConsciousnessAssessment().run_full_assessment(
        beautiful_loop=FixtureLoop(), normalize=False
    )
    assert clean.failures == {}
    assert clean.to_dict()["failures"] == {}
    assert "failed" not in clean.indicator_results["epistemic_depth"].evidence

    original = ConsciousnessAssessment._measure_indicator

    async def broken(self, name, config, **kw):
        if name == "epistemic_depth":
            raise RuntimeError("indicator body deleted")
        return await original(self, name, config, **kw)

    monkeypatch.setattr(ConsciousnessAssessment, "_measure_indicator", broken)
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        beautiful_loop=FixtureLoop(), normalize=False
    )
    result = report.indicator_results["epistemic_depth"]
    assert result.status == INSUFFICIENT_DATA
    assert result.evidence["failed"] is True
    assert "indicator body deleted" in result.evidence["error"]
    assert report.failures == {"indicator:epistemic_depth": 1}
    assert assessment.failures["indicator:epistemic_depth"] == 1
    assert report.to_dict()["failures"] == {"indicator:epistemic_depth": 1}
    # The other indicators of the same module are untouched
    assert report.indicator_results["genuine_implementation"].status == MEASURED

    from mtc.core.config import Settings

    monkeypatch.setattr(assessment_module, "get_settings", lambda: Settings(DEBUG=True))
    with pytest.raises(RuntimeError, match="indicator body deleted"):
        await ConsciousnessAssessment().run_full_assessment(
            beautiful_loop=FixtureLoop(), normalize=False
        )


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


# ---------------------------------------------------------------------------
# A statistic the producer reports as None (it has not measured it yet)
# ---------------------------------------------------------------------------

_STAT_METHODS = {
    "global_workspace": "get_statistics",
    "attention_schema": "get_statistics",
    "metacognition": "get_statistics",
    "active_inference": "get_statistics",
    "beautiful_loop": "get_loop_statistics",
    "rpt_measurement": "get_statistics",
    "rpt_measurement.oscillatory_binding": "get_oscillatory_binding_stats",
}


def _with_statistic(module, method, path, value):
    """A copy of a fixture module whose ``method`` reports ``path`` as ``value``."""
    base = getattr(module, method)
    parts = path.split(".")

    def patched():
        stats = copy.deepcopy(base())
        target = stats
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        return stats

    clone = copy.copy(module)
    setattr(clone, method, patched)
    return clone


def _modules_with(contract_key, path, value):
    modules = fixture_modules()
    module_key = contract_key.split(".")[0]
    modules[module_key] = _with_statistic(
        modules[module_key], _STAT_METHODS[contract_key], path, value
    )
    return modules


def _same_result(left, right):
    return (left.status, left.score, left.confidence, left.passes_threshold) == (
        right.status,
        right.score,
        right.confidence,
        right.passes_threshold,
    )


# (contract key, statistic path, indicator that reads it, the v04_constants
# default the v05 formula falls back to). The fixture leaves every activity
# counter above zero, so each statistic is read.
MISSING_STATISTIC_CASES = [
    (
        "global_workspace",
        "bottleneck.avg_candidates",
        "sequential_module_querying",
        0.0,
    ),
    ("global_workspace", "broadcast.coverage_ratio", "global_broadcast", 0.0),
    ("global_workspace", "broadcast.coverage_ratio", "recurrent_processing", 0.0),
    ("global_workspace", "ignition.sustain_rate", "ignition_dynamics", 0.0),
    ("global_workspace", "ignition.sustain_rate", "global_ignition_nuanced", 0.0),
    ("attention_schema", "prediction_accuracy", "attention_schema", 0.0),
    ("attention_schema", "capacity_used", "embodiment", 0.0),
    ("attention_schema", "voluntary_ratio", "embodiment", 0.5),
    ("attention_schema", "voluntary_ratio", "attention_control", 0.0),
    ("metacognition", "overall_confidence", "metacognition", 0.5),
    ("active_inference", "variational_free_energy", "agency", 0.0),
    ("rpt_measurement", "average_local", "local_recurrence", 0.0),
    ("rpt_measurement", "average_local", "algorithmic_recurrence", 0.0),
    ("rpt_measurement", "average_global", "algorithmic_recurrence", 0.0),
    (
        "rpt_measurement.oscillatory_binding",
        "global_order_parameter",
        "oscillatory_binding_coherence",
        0.0,
    ),
    ("beautiful_loop", "average_binding_quality", "bayesian_binding_quality", 0.0),
    ("beautiful_loop", "epistemic_depth.average_depth", "epistemic_depth", 0.0),
    ("beautiful_loop", "average_loop_quality", "genuine_implementation", 0.0),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("contract_key,path,indicator,default", MISSING_STATISTIC_CASES)
async def test_missing_statistic_is_insufficient_and_names_the_key(
    contract_key, path, indicator, default
):
    with_none = _modules_with(contract_key, path, None)
    honest = await ConsciousnessAssessment().run_full_assessment(
        **with_none, normalize=False
    )
    result = honest.indicator_results[indicator]
    assert result.status == INSUFFICIENT_DATA, result.evidence
    assert f"{contract_key}.{path}" in result.evidence["missing"]
    assert indicator in honest.insufficient_indicators
    assert honest.to_dict()["indicator_results"][indicator]["score"] is None

    # v04_constants reads the missing statistic as the v0.4.0 default
    constants = ConsciousnessAssessment(missing_signal_policy="v04_constants")
    seen = await constants.run_full_assessment(**with_none, normalize=False)
    expected = await constants.run_full_assessment(
        **_modules_with(contract_key, path, default), normalize=False
    )
    assert seen.indicator_results[indicator].status == MEASURED
    assert _same_result(
        seen.indicator_results[indicator], expected.indicator_results[indicator]
    )


@pytest.mark.asyncio
async def test_v05_metacognition_reads_none_confidence_as_the_v04_constant():
    # The module reported a constant 0.5 before it measured its confidence,
    # and the v04 body reads a None confidence as that constant; the v05
    # body under v04_constants reads the same stand-in, not 0.0.
    constants = ConsciousnessAssessment(missing_signal_policy="v04_constants")
    with_none = await constants.run_full_assessment(
        **_modules_with("metacognition", "overall_confidence", None),
        normalize=False,
    )
    with_half = await constants.run_full_assessment(
        **_modules_with("metacognition", "overall_confidence", 0.5),
        normalize=False,
    )
    with_zero = await constants.run_full_assessment(
        **_modules_with("metacognition", "overall_confidence", 0.0),
        normalize=False,
    )
    seen = with_none.indicator_results["metacognition"]
    assert seen.evidence["introspections"] > 0
    assert seen.status == MEASURED
    assert _same_result(seen, with_half.indicator_results["metacognition"])
    assert seen.evidence["overall_confidence"] == 0.5
    assert seen.score != with_zero.indicator_results["metacognition"].score


def test_none_as_docstring_names_the_two_stand_ins_that_differ_from_v04():
    # The v04 bodies read a None broadcast.coverage_ratio and a None
    # ignition.sustain_rate as 0.0, where v0.4.0 read 1.0 by construction;
    # the helper's docstring must not claim that every stand-in is the
    # number v0.4.0 saw.
    from mtc.assessment.assessment import _none_as

    doc = _none_as.__doc__
    assert "v0.4.0 saw" not in doc
    assert "coverage_ratio" in doc and "sustain_rate" in doc
    assert "1.0" in doc


# (contract key, statistic path, the stand-in the v04 bodies read a None as:
# the v0.4.0 constant where v0.4.0 had one; 0.0 for broadcast.coverage_ratio
# and ignition.sustain_rate, where v0.4.0 read 1.0 by construction (the
# acknowledgement-only receivers and the sustained flag that defaulted True;
# the legacy preset restores both producers, so it never reads None there))
V04_NONE_READS = [
    ("global_workspace", "broadcast.coverage_ratio", 0.0),
    ("global_workspace", "ignition.sustain_rate", 0.0),
    ("attention_schema", "prediction_accuracy", 0.0),
    ("attention_schema", "voluntary_ratio", 0.5),
    ("metacognition", "overall_confidence", 0.5),
    ("rpt_measurement", "average_local", 0.0),
    ("rpt_measurement", "average_global", 0.0),
    ("beautiful_loop", "average_binding_quality", 0.0),
    ("beautiful_loop", "average_loop_quality", 0.0),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("contract_key,path,stand_in", V04_NONE_READS)
async def test_v04_formulas_read_none_as_the_stand_in(contract_key, path, stand_in):
    legacy = ConsciousnessAssessment(**V04_SETTINGS)
    with_none = await legacy.run_full_assessment(
        **_modules_with(contract_key, path, None), normalize=False
    )
    with_value = await legacy.run_full_assessment(
        **_modules_with(contract_key, path, stand_in), normalize=False
    )
    for name, result in with_value.indicator_results.items():
        seen = with_none.indicator_results[name]
        assert _same_result(seen, result), name
        assert seen.evidence == result.evidence, name
    assert with_none.overall_score == with_value.overall_score
    assert with_none.passing_count == with_value.passing_count
    assert with_none.confidence == with_value.confidence


@pytest.mark.asyncio
async def test_v04_attention_formula_reads_none_as_zero():
    legacy = ConsciousnessAssessment(indicator_formulas="v04")
    with_none = await legacy.run_full_assessment(
        **_modules_with("attention_schema", "prediction_accuracy", None),
        normalize=False,
    )
    with_zero = await legacy.run_full_assessment(
        **_modules_with("attention_schema", "prediction_accuracy", 0.0),
        normalize=False,
    )
    seen = with_none.indicator_results["attention_schema"]
    expected = with_zero.indicator_results["attention_schema"]
    assert expected.status == MEASURED
    assert seen.evidence == expected.evidence
    assert _same_result(seen, expected)
    assert seen.score == pytest.approx(0.15 * 0.8 + 0.15 + 0.20 * 0.8)


@pytest.mark.asyncio
async def test_v04_binding_formula_reads_none_as_zero():
    legacy = ConsciousnessAssessment(indicator_formulas="v04")
    with_none = await legacy.run_full_assessment(
        **_modules_with("beautiful_loop", "average_binding_quality", None),
        normalize=False,
    )
    with_zero = await legacy.run_full_assessment(
        **_modules_with("beautiful_loop", "average_binding_quality", 0.0),
        normalize=False,
    )
    seen = with_none.indicator_results["bayesian_binding_quality"]
    expected = with_zero.indicator_results["bayesian_binding_quality"]
    assert expected.status == MEASURED and expected.score == 0.0
    assert seen.evidence == expected.evidence
    assert _same_result(seen, expected)
    assert "bayesian_binding_quality" not in with_none.insufficient_indicators
