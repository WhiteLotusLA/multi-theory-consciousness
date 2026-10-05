"""v05 indicator inputs: zero floors, weighted Fiedler, upgraded formulas."""

import numpy as np
import pytest

from mtc.assessment.assessment import (
    INSUFFICIENT_DATA,
    MEASURED,
    ConsciousnessAssessment,
    weighted_fiedler_irreducibility,
)


class _Workspace:
    """A workspace stub with configurable statistics and competition history."""

    def __init__(self, stats, history=None):
        self._stats = stats
        self.bottleneck = type("B", (), {"competition_history": history or []})()

    def get_statistics(self):
        return self._stats


def _quiet_workspace_stats(cycles=5):
    return {
        "cycle_count": cycles,
        "broadcast": {"total_broadcasts": 0, "coverage_ratio": 0.0},
        "ignition": {"total_ignitions": 0},
        "bottleneck": {"total_competitions": 0},
    }


class _Inference:
    def __init__(
        self,
        inferences=3,
        updates=0,
        policies=None,
        drives=None,
        fe_history=None,
        horizon=3,
        belief_history=None,
        avg_error=0.2,
        learning_rate=None,
    ):
        self._stats = {
            "total_inferences": inferences,
            "total_model_updates": updates,
            "avg_prediction_error": avg_error,
            "prediction_error_trend": "stable",
            "hierarchical_prediction_error": 0.1,
            "variational_free_energy": 0.0,
        }
        self.agent = type(
            "A", (), {"policies": policies if policies is not None else []}
        )()
        self.homeostatic_drives = (
            type("H", (), {"drives": drives})() if drives is not None else None
        )
        self.free_energy_history = fe_history if fe_history is not None else []
        self.belief_history = belief_history if belief_history is not None else []
        self.config = type("C", (), {"planning_horizon": horizon})()
        self.inference_history = []
        if learning_rate is not None:
            self.learning_rate = learning_rate

    def get_statistics(self):
        return self._stats


class _Meta:
    def __init__(self, introspections=0, buffer=1, hots=0):
        self._stats = {
            "total_hots_generated": hots,
            "introspection_count": introspections,
            "first_order_buffer_size": buffer,
            "overall_confidence": 0.5,
        }

    def get_statistics(self):
        return self._stats


class _Drive:
    def __init__(self, urgency):
        self._urgency = urgency

    def get_urgency(self):
        return self._urgency


class _Event:
    def __init__(self, sources):
        self.winning_sources = sources


class _Attention:
    def __init__(self, updates=0, history=0):
        self.schema = type("S", (), {"current_focus": None})()
        self._stats = {
            "total_updates": updates,
            "history_length": history,
            "prediction_accuracy": 0.0,
            "capacity_used": 0.0,
            "voluntary_shifts": 0,
            "voluntary_ratio": 0.0,
        }

    def get_statistics(self):
        return self._stats


async def _assess(assessment=None, **modules):
    assessment = assessment or ConsciousnessAssessment()
    return await assessment.run_full_assessment(**modules, normalize=False)


@pytest.mark.asyncio
async def test_zero_floors_under_v05():
    """Modules that ran but show no activity score 0.0, measured, no floor."""
    report = await _assess(
        global_workspace=_Workspace(_quiet_workspace_stats()),
        active_inference=_Inference(inferences=3, updates=0, fe_history=[0.5] * 5),
        metacognition=_Meta(introspections=0, buffer=1),
    )
    for name in (
        "sequential_module_querying",
        "organized_percepts",
        "flexible_responsiveness",
        "online_reportable_learning",
        "complex_agency",
    ):
        result = report.indicator_results[name]
        assert result.status == MEASURED, name
        assert result.score == 0.0, (name, result.score)

    # AE agency and embodiment: with zero counters the v05 formulas give 0.0
    # (v0.4.0 floored both at 0.4); under the default policy zero counters
    # are insufficient data, never a floor.
    constants = ConsciousnessAssessment(missing_signal_policy="v04_constants")
    quiet = await _assess(
        constants,
        active_inference=_Inference(inferences=0),
        attention_schema=_Attention(updates=0, history=0),
    )
    for name in ("agency", "embodiment"):
        assert quiet.indicator_results[name].status == MEASURED
        assert quiet.indicator_results[name].score == 0.0, name
    honest = await _assess(
        active_inference=_Inference(inferences=0),
        attention_schema=_Attention(updates=0, history=0),
    )
    for name in ("agency", "embodiment"):
        assert honest.indicator_results[name].status == INSUFFICIENT_DATA


def test_fiedler_zero_for_disconnected_graph():
    block = np.zeros((6, 6))
    block[:3, :3] = 1.0
    block[3:, 3:] = 1.0
    assert weighted_fiedler_irreducibility(block)[0] == 0.0
    isolated = np.ones((4, 4))
    isolated[3, :] = 0.0
    isolated[:, 3] = 0.0
    assert weighted_fiedler_irreducibility(isolated)[0] == 0.0
    assert weighted_fiedler_irreducibility(np.ones((1, 1)))[0] == 0.0


def test_fiedler_one_for_uniform_complete_graph():
    for n in (2, 5, 16):
        complete = np.ones((n, n))
        np.fill_diagonal(complete, 0.0)
        score, details = weighted_fiedler_irreducibility(complete)
        assert score == pytest.approx(1.0)
        assert details["graph_nodes"] == n


def test_fiedler_does_not_saturate_on_random_weights():
    rng = np.random.default_rng(0)
    for _ in range(20):
        raw = rng.random((16, 16))
        weights = (raw + raw.T) / 2.0
        score, _details = weighted_fiedler_irreducibility(weights)
        assert 0.2 < score < 0.95, score


def test_fiedler_symmetrizes_asymmetric_input():
    rng = np.random.default_rng(4)
    asymmetric = rng.random((8, 8)) * (rng.random((8, 8)) > 0.4)
    np.fill_diagonal(asymmetric, 0.0)
    forward = weighted_fiedler_irreducibility(asymmetric)[0]
    backward = weighted_fiedler_irreducibility(asymmetric.T)[0]
    averaged = weighted_fiedler_irreducibility((asymmetric + asymmetric.T) / 2.0)[0]
    assert forward == pytest.approx(backward)
    assert forward == pytest.approx(averaged)
    # Negative weights count by magnitude
    assert weighted_fiedler_irreducibility(-asymmetric)[0] == pytest.approx(forward)


@pytest.mark.asyncio
async def test_irreducibility_method_switch():
    rng = np.random.default_rng(2)
    raw = rng.random((16, 16))
    connectivity = (raw + raw.T) / 2.0
    np.fill_diagonal(connectivity, 1.0)
    neural = {"combined_state": rng.random(16), "connectivity": connectivity}
    fiedler = await _assess(neural_states=neural)
    entropy = await _assess(
        ConsciousnessAssessment(irreducibility_method="spectral_entropy"),
        neural_states=neural,
    )
    assert fiedler.indicator_results["irreducibility"].evidence["method"] == (
        "weighted_fiedler"
    )
    assert fiedler.indicator_results["irreducibility"].score == pytest.approx(
        weighted_fiedler_irreducibility(connectivity)[0]
    )
    assert "spectral_entropy" in entropy.indicator_results["irreducibility"].evidence


@pytest.mark.asyncio
async def test_complex_agency_reads_policies_drives_and_fe_variance():
    fe = list(np.linspace(0.0, 0.6, 20))
    inference = _Inference(
        inferences=12,
        policies=[None] * 3,
        drives={"a": _Drive(0.5), "b": _Drive(0.3), "c": _Drive(0.1)},
        fe_history=[9.9] * 5 + fe,  # only the last 20 count
    )
    report = await _assess(active_inference=inference)
    result = report.indicator_results["complex_agency"]
    expected = 0.4 * min(1.0, 3 / 5) + 0.3 * (2 / 3) + 0.3 * min(1.0, np.var(fe) / 0.05)
    assert result.score == pytest.approx(expected)
    assert result.evidence["policy_count"] == 3
    assert result.evidence["urgent_drives"] == 2


@pytest.mark.asyncio
async def test_temporal_depth_uses_belief_and_fe_histories():
    inference = _Inference(
        inferences=10,
        horizon=5,
        avg_error=0.2,
        belief_history=[None] * 10,
        fe_history=[0.1] * 30,
    )
    report = await _assess(active_inference=inference)
    result = report.indicator_results["temporal_prediction_depth"]
    expected = 0.6 * 0.5 * 0.8 + 0.2 * 0.5 + 0.2 * 1.0
    assert result.score == pytest.approx(expected)


class _RPT:
    def __init__(self, avg_global, count):
        self._stats = {
            "average_local": 0.2,
            "average_global": avg_global,
            "measurement_count": count,
        }

    def get_statistics(self):
        return self._stats

    def get_oscillatory_binding_stats(self):
        return {}


@pytest.mark.asyncio
async def test_recurrent_processing_blends_rpt_when_present():
    stats = {
        "cycle_count": 10,
        "broadcast": {"total_broadcasts": 5, "coverage_ratio": 0.6},
        "ignition": {"total_ignitions": 0},
        "bottleneck": {"total_competitions": 0},
    }
    gwt_term = 0.5 * 0.6 + 0.5 * 0.5
    alone = await _assess(global_workspace=_Workspace(stats))
    blended = await _assess(
        global_workspace=_Workspace(stats), rpt_measurement=_RPT(0.3, 4)
    )
    idle_rpt = await _assess(
        global_workspace=_Workspace(stats), rpt_measurement=_RPT(0.3, 0)
    )
    score = "recurrent_processing"
    assert alone.indicator_results[score].score == pytest.approx(gwt_term)
    assert blended.indicator_results[score].score == pytest.approx(
        0.5 * 0.3 + 0.5 * gwt_term
    )
    assert idle_rpt.indicator_results[score].score == pytest.approx(gwt_term)


@pytest.mark.asyncio
async def test_flexible_responsiveness_counts_winner_changes():
    stats = {
        "cycle_count": 8,
        "broadcast": {"total_broadcasts": 4, "coverage_ratio": 0.5},
        "ignition": {"total_ignitions": 0},
        "bottleneck": {
            "total_competitions": 8,
            "avg_candidates": 5.0,
            "source_distribution": {"sensory": 5, "memory": 3, "concept": 2},
        },
    }
    history = [
        _Event(["sensory"]),
        _Event(["sensory"]),
        {"winning_sources": ["dict events are not typed"]},
        _Event(["memory"]),
        _Event(["memory", "sensory"]),
        _Event(["sensory", "memory"]),
    ]
    report = await _assess(
        global_workspace=_Workspace(stats, history),
        active_inference=_Inference(inferences=5, updates=5),
    )
    result = report.indicator_results["flexible_responsiveness"]
    # 4 typed pairs, 2 with a different winner set
    assert result.evidence["attention_shifts"] == pytest.approx(0.5)
    assert result.score == pytest.approx(0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 0.5)


@pytest.mark.asyncio
async def test_online_learning_has_no_learning_rate_credit():
    report = await _assess(
        active_inference=_Inference(inferences=3, updates=0, learning_rate=0.5),
        metacognition=_Meta(introspections=0, buffer=2),
    )
    result = report.indicator_results["online_reportable_learning"]
    assert result.status == MEASURED
    assert result.score == 0.0
    learned = await _assess(
        active_inference=_Inference(inferences=3, updates=5),
        metacognition=_Meta(introspections=5, buffer=2),
    )
    assert learned.indicator_results[
        "online_reportable_learning"
    ].score == pytest.approx(0.5 * 0.5 + 0.3 * 1.0)


class _PAD:
    history_length = 20
    _history = [None] * 20

    def compute_coherence(self):
        return 0.7

    def get_latest(self):
        return None


@pytest.mark.asyncio
async def test_affect_coherence_does_not_need_active_inference():
    report = await _assess(pad_affect_model=_PAD())
    result = report.indicator_results["affect_coherence"]
    assert result.status == MEASURED
    assert result.score == pytest.approx(0.7)
    legacy = await _assess(
        ConsciousnessAssessment(indicator_formulas="v04"), pad_affect_model=_PAD()
    )
    # v0.4.0 nested the PAD read inside the active inference block
    assert legacy.indicator_results["affect_coherence"].status == "absent"


class _CE:
    def __init__(self, value):
        self.value = value

    def get_latest_ce(self):
        return self.value


@pytest.mark.asyncio
async def test_causal_emergence_needs_macro_to_beat_micro():
    none = await _assess(causal_emergence_analyzer=_CE(0.0))
    result = none.indicator_results["causal_emergence"]
    assert result.score == pytest.approx(0.5)
    assert result.passes_threshold is False
    emerges = await _assess(causal_emergence_analyzer=_CE(0.2))
    assert emerges.indicator_results["causal_emergence"].passes_threshold is True
    loses = await _assess(causal_emergence_analyzer=_CE(-0.1))
    assert loses.indicator_results["causal_emergence"].passes_threshold is False

    # The v04 formulas keep the inclusive comparison
    inclusive = await _assess(
        ConsciousnessAssessment(indicator_formulas="v04"),
        causal_emergence_analyzer=_CE(0.0),
        active_inference=_Inference(),
    )
    assert inclusive.indicator_results["causal_emergence"].passes_threshold is True

    configured = ConsciousnessAssessment(causal_emergence_threshold=0.0)
    assert configured.indicator_configs["causal_emergence"]["threshold"] == 0.0
    assert ConsciousnessAssessment().indicator_configs["causal_emergence"][
        "threshold"
    ] == pytest.approx(0.5)
