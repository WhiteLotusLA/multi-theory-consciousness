"""The binding coherence indicator against the binding layer's zero-stimulus floor."""

from collections import deque

import numpy as np
import pytest

from mtc.assessment.assessment import MEASURED, ConsciousnessAssessment
from mtc.assessment.rpt_measurement import RPTMeasurement

INDICATOR = "oscillatory_binding_coherence"


def _bound_rpt(binds=3):
    from mtc.neural.oscillatory_binding import OscillatoryBinding

    binding = OscillatoryBinding()
    rpt = RPTMeasurement()
    rpt.attach_oscillatory_binding(binding)
    rng = np.random.default_rng(1)
    for _ in range(binds):
        binding.bind(
            {
                "attention_focus": float(rng.random()),
                "valence": float(rng.uniform(-1.0, 1.0)),
                "novelty": float(rng.random()),
            }
        )
    return rpt, binding


def _raw_score(stats):
    cross = [float(v) for v in stats["cross_coherence"].values()]
    return 0.6 * stats["global_order_parameter"] + 0.4 * float(np.mean(cross))


class _FakeBinding:
    def __init__(self, history, null_history, cross, count):
        self._order_history = deque(history, maxlen=100)
        if null_history is not None:
            self._null_order_history = deque(null_history, maxlen=100)
        self._last_cross_coherence = cross
        self._binding_count = count


class _StubRPT:
    def __init__(self, stats):
        self._stats = stats

    def get_statistics(self):
        return {"average_local": 0.3, "average_global": 0.2, "measurement_count": 3}

    def get_oscillatory_binding_stats(self):
        return dict(self._stats)


@pytest.mark.asyncio
async def test_floor_saturated_indicator_scores_zero():
    rpt, binding = _bound_rpt(binds=3)
    stats = rpt.get_oscillatory_binding_stats()
    floor = stats["zero_stimulus_order_parameter"]
    assert floor == pytest.approx(float(np.mean(list(binding._null_order_history))))
    assert 1.0 - floor < 0.1
    report = await ConsciousnessAssessment().run_full_assessment(
        rpt_measurement=rpt, normalize=False
    )
    result = report.indicator_results[INDICATOR]
    assert result.status == MEASURED
    assert result.score == 0.0
    assert result.passes_threshold is False
    assert result.evidence["floor_saturated"] is True
    assert result.evidence["zero_stimulus_order_parameter"] == pytest.approx(floor)
    # The raw inputs stay in the evidence
    assert result.evidence["global_order_parameter"] == pytest.approx(
        stats["global_order_parameter"]
    )
    assert result.evidence["binding_count"] == 3
    assert result.evidence["cross_coherence"] == stats["cross_coherence"]


@pytest.mark.asyncio
async def test_legacy_min_range_keeps_raw_score():
    rpt, _binding = _bound_rpt(binds=3)
    stats = rpt.get_oscillatory_binding_stats()
    legacy = ConsciousnessAssessment(saturated_baseline_min_range=0.0)
    report = await legacy.run_full_assessment(rpt_measurement=rpt, normalize=False)
    result = report.indicator_results[INDICATOR]
    assert result.status == MEASURED
    assert result.score == pytest.approx(_raw_score(stats))
    assert result.score > 0.9
    assert result.evidence["floor_saturated"] is False
    assert result.evidence["zero_stimulus_order_parameter"] == pytest.approx(
        stats["zero_stimulus_order_parameter"]
    )


def test_rpt_binding_stats_carry_the_floor():
    rpt = RPTMeasurement()
    rpt.attach_oscillatory_binding(
        _FakeBinding([0.2, 0.4], [0.9, 0.7], {"snn_lsm": 0.3}, 2)
    )
    stats = rpt.get_oscillatory_binding_stats()
    assert stats["zero_stimulus_order_parameter"] == pytest.approx(0.8)
    assert stats["global_order_parameter"] == pytest.approx(0.3)
    # An empty null history, or a binding without one: the key is absent
    for null_history in ([], None):
        rpt = RPTMeasurement()
        rpt.attach_oscillatory_binding(
            _FakeBinding([0.2, 0.4], null_history, {"snn_lsm": 0.3}, 2)
        )
        assert (
            "zero_stimulus_order_parameter" not in rpt.get_oscillatory_binding_stats()
        )
    # A pushed snapshot passes the key through when it carries it
    pushed = RPTMeasurement()
    pushed.set_oscillatory_binding_stats(
        {
            "global_order_parameter": 0.9,
            "cross_coherence": {"snn_lsm": 0.7},
            "binding_count": 11,
            "zero_stimulus_order_parameter": 0.95,
        }
    )
    assert pushed.get_oscillatory_binding_stats()["zero_stimulus_order_parameter"] == (
        pytest.approx(0.95)
    )
    pushed.set_oscillatory_binding_stats(
        {"global_order_parameter": 0.9, "cross_coherence": {}, "binding_count": 1}
    )
    assert "zero_stimulus_order_parameter" not in pushed.get_oscillatory_binding_stats()


@pytest.mark.asyncio
async def test_known_floor_with_headroom_scores_as_before():
    stats = {
        "global_order_parameter": 0.6,
        "cross_coherence": {"snn_lsm": 0.3, "snn_htm": 0.5},
        "binding_count": 12,
    }
    expected = 0.6 * 0.6 + 0.4 * 0.4
    unknown = await ConsciousnessAssessment().run_full_assessment(
        rpt_measurement=_StubRPT(stats), normalize=False
    )
    result = unknown.indicator_results[INDICATOR]
    assert result.score == pytest.approx(expected)
    assert result.evidence["zero_stimulus_order_parameter"] is None
    assert result.evidence["floor_saturated"] is False

    known = await ConsciousnessAssessment().run_full_assessment(
        rpt_measurement=_StubRPT({**stats, "zero_stimulus_order_parameter": 0.5}),
        normalize=False,
    )
    result = known.indicator_results[INDICATOR]
    assert result.score == pytest.approx(expected)
    assert result.evidence["zero_stimulus_order_parameter"] == pytest.approx(0.5)
    assert result.evidence["floor_saturated"] is False

    # The guard reads the configured minimum headroom
    narrow = await ConsciousnessAssessment(
        saturated_baseline_min_range=0.6
    ).run_full_assessment(
        rpt_measurement=_StubRPT({**stats, "zero_stimulus_order_parameter": 0.5}),
        normalize=False,
    )
    assert narrow.indicator_results[INDICATOR].score == 0.0
    assert narrow.indicator_results[INDICATOR].evidence["floor_saturated"] is True
