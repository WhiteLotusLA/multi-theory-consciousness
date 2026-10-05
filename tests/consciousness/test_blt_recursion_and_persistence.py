"""Beautiful Loop recursion, cessation, multi-level binding and persistence.

Covers the resynced behavior: recursive self-modeling in the hyper-model,
the momentum boost, cessation detection, multi-level binding with ragged
belief vectors, the precision floor, the v0.4.0 formula switch, and typed
rehydration after a save and restore.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mtc.consciousness.beautiful_loop import (
    BayesianBinding,
    BeautifulLoop,
    BindingLevel,
    BoundPercept,
    ConsciousMoment,
    EpistemicDepthTracker,
    HyperModel,
    Inference,
    PrecisionState,
    SelfReferenceMetrics,
)
from mtc.consciousness.beautiful_loop.epistemic_depth import DepthMeasurement
from mtc.consciousness.beautiful_loop.hyper_model import PrecisionUpdate


def _self_model(update_count: int = 5) -> Mock:
    sm = Mock()
    sm.update_count = update_count
    sm.current_focus = "synthetic focus"
    sm.confidence = 0.7
    sm.cognitive_load = 0.3
    sm.curiosity_level = 0.6
    sm.prediction_accuracy = 0.7
    sm.model_confidence = 0.65
    sm.valence = 0.2
    sm.arousal = 0.4
    sm.free_energy = 0.1
    sm.self_calibration_score = 0.7
    sm.prediction_accuracy_history = [0.5, 0.6, 0.65, 0.7]
    return sm


def _meta_state() -> Mock:
    ms = Mock()
    ms.total_predictions = 15
    ms.predicted_accuracy = 0.7
    ms.prediction_accuracy = 0.65
    ms.model_confidence = 0.65
    return ms


def _winner(idx: int, salience: float, content_type: str = "thought"):
    candidate = SimpleNamespace(
        id=f"w{idx}",
        summary=f"synthetic content {idx}",
        content_type=content_type,
        source_module="test",
    )
    return SimpleNamespace(candidate=candidate, salience=salience)


async def _full_cycle(loop: BeautifulLoop, step: int) -> ConsciousMoment:
    hot = SimpleNamespace(meta_content="I notice my own prediction confidence")
    result = SimpleNamespace(
        posterior_beliefs=np.array([0.3, 0.2, 0.1, 0.15, 0.05, 0.05, 0.1, 0.05])
    )
    return await loop.process_conscious_moment(
        prediction_errors=[0.1 + 0.05 * (step % 3), 0.3, 0.2 + 0.01 * step],
        workspace_winners=[_winner(1, 0.7), _winner(2, 0.6), _winner(3, 0.5)],
        inference_result=result,
        self_model=_self_model(),
        meta_state=_meta_state(),
        higher_order_thoughts=[hot],
        context="reflection" if step % 2 else "general",
    )


def _hub_round_trip(state: dict) -> dict:
    """Mimic the hub: JSON for scalars and histories, arrays kept as arrays."""
    return {
        "scalars": json.loads(json.dumps(state["scalars"])),
        "arrays": {k: np.asarray(v) for k, v in state["arrays"].items()},
        "histories": json.loads(json.dumps(state["histories"])),
    }


@pytest.mark.asyncio
async def test_momentum_boost_raises_loop_quality():
    loop = BeautifulLoop()
    first = await loop.process_conscious_moment()
    second = await loop.process_conscious_moment(self_model=_self_model())

    rc = loop.hyper_model.recursive_confidence
    raw = (
        0.25 * second.precision_balance
        + 0.35 * second.binding_quality
        + 0.25 * min(1.0, second.epistemic_depth / 3.0)
        + 0.15 * rc
    )
    assert second.epistemic_depth == 1
    assert raw > first.loop_quality
    assert second.loop_quality > raw
    assert second.loop_quality == pytest.approx(raw + 0.1 * (raw - first.loop_quality))


@pytest.mark.asyncio
async def test_cessation_detected_after_five_quiet_cycles_and_recovers():
    loop = BeautifulLoop()
    moments = [await loop.process_conscious_moment() for _ in range(5)]
    assert all(m.loop_quality < 0.1 for m in moments)
    assert [m.cessation_detected for m in moments] == [False] * 4 + [True]
    stats = loop.get_loop_statistics()
    assert stats["cessation_detected"] is True
    assert stats["cessation_count"] == 1

    recovered = await loop.process_conscious_moment(self_model=_self_model())
    assert recovered.cessation_detected is False
    stats = loop.get_loop_statistics()
    assert stats["cessation_detected"] is False
    assert stats["cessation_count"] == 1


def test_multilevel_dispatch_with_ragged_beliefs_does_not_raise():
    binding = BayesianBinding()
    candidates = [
        Inference(
            id="u1",
            content_summary="unimodal a",
            content_type="perception",
            source_module="test",
            belief_vector=np.array([0.1, 0.5, 0.9]),
            salience=0.7,
            level="unimodal",
        ),
        Inference(
            id="u2",
            content_summary="unimodal b",
            content_type="perception",
            source_module="test",
            belief_vector=np.array([0.1, 0.5, 0.9, 0.3, 0.2]),
            salience=0.6,
            level="unimodal",
        ),
        Inference(
            id="g1",
            content_summary="global a",
            content_type="thought",
            source_module="test",
            belief_vector=np.array([0.2, 0.4, 0.8, 0.1]),
            salience=0.5,
        ),
    ]
    percepts = binding.bind_multilevel(candidates)
    assert set(percepts) == {"unimodal", "global"}
    assert percepts["unimodal"].binding_level == "unimodal"
    assert percepts["global"].binding_level == "global"
    assert any(
        inf.id == "bound_unimodal" for inf in percepts["global"].bound_inferences
    )


@pytest.mark.asyncio
async def test_multilevel_dispatch_through_the_loop():
    loop = BeautifulLoop()
    winners = [_winner(1, 0.8), _winner(2, 0.7)]
    winners[0].binding_level = "crossmodal"
    winners[1].binding_level = BindingLevel.CROSSMODAL
    moment = await loop.process_conscious_moment(workspace_winners=winners)
    assert isinstance(moment.bound_percept, BoundPercept)
    assert moment.bound_percept.binding_level in {"global", "crossmodal"}


def test_precision_floor_applies_when_all_levels_reliable():
    on = BeautifulLoop()
    off = BeautifulLoop(precision_floor_enabled=False)
    for loop in (on, off):
        loop.hyper_model.precision_beliefs = np.full(3, 4.5)
    state = on.hyper_model.get_precision_state()
    assert on._compute_precision_balance(state) == pytest.approx(0.4)
    assert off._compute_precision_balance(state) == pytest.approx(0.0, abs=1e-6)

    # Uniform but low precision never earns the floor
    on.hyper_model.precision_beliefs = np.full(3, 1.0)
    low = on.hyper_model.get_precision_state()
    assert on._compute_precision_balance(low) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_v04_formula_matches_old_weights():
    loop = BeautifulLoop(
        loop_quality_formula="v04",
        precision_floor_enabled=False,
        require_recursive_confidence=False,
    )
    await loop.process_conscious_moment()
    for step in range(3):
        moment = await _full_cycle(loop, step)
        expected = (
            0.3 * moment.precision_balance
            + 0.4 * moment.binding_quality
            + 0.3 * min(1.0, moment.epistemic_depth / 3.0)
        )
        assert moment.loop_quality == pytest.approx(expected)


def test_unknown_formula_is_rejected():
    with pytest.raises(ValueError, match="loop_quality_formula"):
        BeautifulLoop(loop_quality_formula="other")


def test_field_evidencing_threshold_default():
    assert BeautifulLoop().field_evidencing_threshold == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_require_recursive_confidence_gates_field_evidencing():
    gated = BeautifulLoop(field_evidencing_threshold=0.0)
    open_loop = BeautifulLoop(
        field_evidencing_threshold=0.0, require_recursive_confidence=False
    )
    for loop in (gated, open_loop):
        loop.hyper_model._recursive_confidence = 0.1
        await loop.process_conscious_moment(
            self_model=_self_model(), meta_state=_meta_state()
        )
    assert gated.field_evidencing_count == 0
    assert open_loop.field_evidencing_count == 1


def test_recursive_confidence_updates_after_observe():
    hm = HyperModel(num_levels=3)
    assert hm.recursive_confidence == pytest.approx(0.5)
    assert hm.observe_meta_error() == 0.0

    predicted = hm.predict_precision_update([0.2, 0.4, 0.6])
    assert predicted.shape == (3,)
    hm.precision_beliefs = predicted + 0.5
    meta_error = hm.observe_meta_error()
    assert meta_error == pytest.approx(0.5)
    assert hm.recursive_confidence == pytest.approx(1.0 - 0.1 * 0.5)

    before = hm.recursive_confidence
    for _ in range(5):
        hm.update_precision_beliefs([0.9, 0.1, 0.5])
    assert hm.recursive_confidence != before
    assert 0.0 <= hm.recursive_confidence <= 1.0


@pytest.mark.asyncio
async def test_report_after_restore_does_not_raise():
    loop = BeautifulLoop()
    for step in range(12):
        await _full_cycle(loop, step)

    restored = BeautifulLoop()
    restored.from_state_dict(_hub_round_trip(loop.to_state_dict()))

    # Loop level
    assert all(isinstance(m, ConsciousMoment) for m in restored._moment_history)
    assert restored._moment_history.maxlen == loop._moment_history.maxlen
    assert isinstance(restored.generate_report(), str)
    assert "Beautiful Loop" in restored.generate_consciousness_context()
    assert restored.get_loop_statistics()["cycle_count"] == 12

    # Hyper model
    hm = restored.hyper_model
    assert all(isinstance(s, PrecisionState) for s in hm._state_history)
    assert all(isinstance(u, PrecisionUpdate) for u in hm._update_history)
    assert set(hm.get_precision_trend()) == {0, 1, 2}
    assert isinstance(hm.generate_report(), str)

    # Bayesian binding
    bb = restored.bayesian_binding
    assert all(isinstance(p, BoundPercept) for p in bb._percept_history)
    assert all(
        isinstance(inf, Inference) and isinstance(inf.belief_vector, np.ndarray)
        for p in bb._percept_history
        for inf in p.bound_inferences
    )
    assert bb.get_binding_trend() in {"improving", "degrading", "stable"}
    assert "Latest" in bb.generate_report()

    # Epistemic depth
    ed = restored.epistemic_depth
    assert all(isinstance(m, DepthMeasurement) for m in ed._depth_history)
    assert all(isinstance(m, SelfReferenceMetrics) for m in ed._metrics_history)
    assert ed.get_depth_trend() in {"deepening", "shallowing", "stable"}
    assert "Strange loop" in ed.generate_report()

    # And the restored loop keeps running
    moment = await _full_cycle(restored, 13)
    assert isinstance(moment, ConsciousMoment)


@pytest.mark.asyncio
async def test_state_dict_round_trip():
    loop = BeautifulLoop()
    for step in range(8):
        await _full_cycle(loop, step)
    for _ in range(5):
        await loop.process_conscious_moment()

    state = loop.to_state_dict()
    assert all(isinstance(v, np.ndarray) for v in state["arrays"].values())

    restored = BeautifulLoop()
    restored.from_state_dict(_hub_round_trip(state))

    assert restored.cycle_count == loop.cycle_count
    assert restored.field_evidencing_count == loop.field_evidencing_count
    stats_a = loop.get_loop_statistics()
    stats_b = restored.get_loop_statistics()
    for key in (
        "average_loop_quality",
        "average_binding_quality",
        "cessation_detected",
        "cessation_count",
        "recursive_confidence",
    ):
        assert stats_b[key] == pytest.approx(stats_a[key])
    np.testing.assert_allclose(
        restored.hyper_model.precision_beliefs, loop.hyper_model.precision_beliefs
    )
    assert set(restored.hyper_model.context_precision_map) == set(
        loop.hyper_model.context_precision_map
    )
    for key, arr in loop.hyper_model.context_precision_map.items():
        np.testing.assert_allclose(restored.hyper_model.context_precision_map[key], arr)
    assert [len(d) for d in restored.hyper_model._error_history] == [
        len(d) for d in loop.hyper_model._error_history
    ]
    assert restored.epistemic_depth.peak_depth == loop.epistemic_depth.peak_depth
    assert restored.bayesian_binding.bind_count == loop.bayesian_binding.bind_count
    latest_a = loop._moment_history[-1]
    latest_b = restored._moment_history[-1]
    assert latest_b.loop_quality == pytest.approx(latest_a.loop_quality)
    assert latest_b.precision_state.level_precisions == pytest.approx(
        latest_a.precision_state.level_precisions
    )


def test_hyper_model_rejects_wrong_shapes():
    hm = HyperModel(num_levels=3)
    hm.update_precision_beliefs([0.1, 0.2, 0.3])
    state = hm.to_state_dict()
    state["arrays"]["precision_beliefs"] = np.ones(5)
    fresh = HyperModel(num_levels=3)
    fresh.from_state_dict(state)
    np.testing.assert_allclose(fresh.precision_beliefs, np.ones(3))


def test_depth_tracker_round_trip_restores_self_model_states():
    tracker = EpistemicDepthTracker()
    for _ in range(4):
        tracker.measure_depth(self_model=_self_model())
    restored = EpistemicDepthTracker()
    restored.from_state_dict(_hub_round_trip(tracker.to_state_dict()))
    assert restored._self_model_states.maxlen == tracker._self_model_states.maxlen
    assert list(restored._self_model_states) == list(tracker._self_model_states)
    assert restored._compute_self_model_coherence() == pytest.approx(
        tracker._compute_self_model_coherence()
    )
