"""
Persistence for the active inference module.

The state dict follows the module contract (scalars, arrays, histories):
JSON-safe scalars and histories, numpy arrays, typed objects on restore,
shape guards, capped histories and a rebuilt pymdp agent.
"""

import json
import logging

import numpy as np
import pytest

from mtc.consciousness.active_inference import (
    HISTORY_LIMIT,
    STATE_SCHEMA_VERSION,
    ActiveInferenceConfig,
    ActiveInferenceModule,
    ConsciousPosteriorMapping,
    InferenceResult,
)

FAST = dict(num_hidden_states=4, num_actions=2, planning_horizon=2)


def _module(**overrides):
    return ActiveInferenceModule(ActiveInferenceConfig(**{**FAST, **overrides}))


async def _run(module, cycles, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(cycles):
        observation = rng.dirichlet(np.ones(module.num_observations))
        result = await module.infer_and_act(observation)
        module.map_conscious_content(
            [{"summary": "synthetic workspace fragment", "content_type": "thought"}],
            result,
        )
        module.modulate_precision_from_workspace(
            [{"content_type": "emotion", "activation_level": 0.6}]
        )
        await module.update_homeostasis({"type": "reasoning", "intensity": 0.5})
        module.get_statistics()
    return module


def _json_round_trip(state):
    """Scalars and histories through JSON, arrays as numpy arrays."""
    return {
        "scalars": json.loads(json.dumps(state["scalars"])),
        "arrays": {k: np.asarray(v) for k, v in state["arrays"].items()},
        "histories": json.loads(json.dumps(state["histories"])),
    }


@pytest.mark.asyncio
async def test_state_dict_round_trip_restores_agent_and_histories():
    source = await _run(_module(), 30)
    state = _json_round_trip(source.to_state_dict())

    assert state["scalars"]["schema_version"] == STATE_SCHEMA_VERSION
    assert all(isinstance(v, np.ndarray) for v in state["arrays"].values())

    restored = _module()
    restored.from_state_dict(state)

    # Generative model and the rebuilt agent
    for name in ("A", "B", "D"):
        np.testing.assert_array_equal(
            getattr(restored, name)[0], getattr(source, name)[0]
        )
    np.testing.assert_array_equal(restored.C, source.C)
    np.testing.assert_array_equal(restored.agent.A[0], source.A[0])
    np.testing.assert_array_equal(restored.agent.B[0], source.B[0])
    np.testing.assert_array_equal(
        restored.workspace_precision_bias, source.workspace_precision_bias
    )

    # Counters, histories and typed history entries
    assert restored.total_inferences == source.total_inferences
    assert restored.total_model_updates == source.total_model_updates
    assert restored.prediction_errors == source.prediction_errors
    assert restored.free_energy_history == source.free_energy_history
    assert len(restored.belief_history) == len(source.belief_history)
    assert all(isinstance(b, np.ndarray) for b in restored.belief_history)
    np.testing.assert_array_equal(
        restored.belief_history[-1], source.belief_history[-1]
    )
    assert len(restored.inference_history) == len(source.inference_history)
    assert all(isinstance(r, InferenceResult) for r in restored.inference_history)
    assert all(
        isinstance(m, ConsciousPosteriorMapping) for m in restored._posterior_mappings
    )
    assert restored._posterior_mappings[-1].content_summary == (
        "synthetic workspace fragment"
    )

    # Meta-state and ledger
    for name in ("model_confidence", "predicted_accuracy", "total_predictions"):
        assert getattr(restored.meta_state, name) == getattr(source.meta_state, name)
    assert restored.meta_state._recent_outcomes == source.meta_state._recent_outcomes
    assert restored.measured_accuracy() == source.measured_accuracy()
    assert restored._pending_prediction_id == source._pending_prediction_id

    # Drives and hierarchy
    for name, drive in source.homeostatic_drives.drives.items():
        assert restored.homeostatic_drives.drives[name].current_level == pytest.approx(
            drive.current_level
        )
    np.testing.assert_array_equal(
        restored.hierarchical_processor.levels[0].weights_up,
        source.hierarchical_processor.levels[0].weights_up,
    )

    # The restored module keeps running, and scores the restored forecast
    scored_before = restored.prediction_ledger.scored_count
    result = await restored.infer_and_act(np.array([0.3, 0.3, 0.2, 0.1, 0.1]))
    assert np.all(np.isfinite(result.posterior_beliefs))
    assert restored.prediction_ledger.scored_count == scored_before + 1


@pytest.mark.asyncio
async def test_ledger_state_is_flat_and_round_trips():
    source = await _run(_module(), 12)
    state = _json_round_trip(source.to_state_dict())

    assert "prediction_ledger" not in state["histories"]
    assert state["scalars"]["ledger_scored_count"] == (
        source.prediction_ledger.scored_count
    )
    assert isinstance(state["histories"]["ledger_pending"], list)
    assert isinstance(state["histories"]["ledger_scored"], list)
    assert all(isinstance(v, list) for v in state["histories"].values())

    restored = _module()
    restored.from_state_dict(state)
    assert restored.prediction_ledger.scored_count == (
        source.prediction_ledger.scored_count
    )
    assert restored.measured_accuracy() == source.measured_accuracy()
    assert restored._pending_prediction_id == source._pending_prediction_id


@pytest.mark.asyncio
async def test_phase1_nested_ledger_shape_still_restores():
    source = await _run(_module(), 12)
    state = _json_round_trip(source.to_state_dict())
    histories = state["histories"]
    nested = source.prediction_ledger.to_state_dict()
    histories.pop("ledger_pending")
    histories.pop("ledger_scored")
    histories["prediction_ledger"] = json.loads(json.dumps(nested))

    restored = _module()
    restored.from_state_dict(state)
    assert restored.prediction_ledger.scored_count == (
        source.prediction_ledger.scored_count
    )
    assert restored._pending_prediction_id == source._pending_prediction_id


@pytest.mark.asyncio
async def test_report_and_statistics_after_restore_do_not_raise():
    state = (await _run(_module(), 5)).to_state_dict()
    restored = _module()
    restored.from_state_dict(state)

    assert isinstance(await restored.generate_active_inference_report(), str)
    assert restored.get_statistics()["total_inferences"] == 5
    assert restored.get_posterior_mapping_stats()["total_mappings"] == 5


@pytest.mark.asyncio
async def test_restore_with_wrong_shapes_is_rejected(caplog):
    # Saved by a 4-state model with the same world model name
    state = (await _run(_module(), 10)).to_state_dict()
    target = _module(num_hidden_states=6)
    fresh_A = target.A[0].copy()
    fresh_B = target.B[0].copy()

    with caplog.at_level(logging.WARNING, logger="mtc.consciousness.active_inference"):
        target.from_state_dict(state)

    np.testing.assert_array_equal(target.A[0], fresh_A)
    np.testing.assert_array_equal(target.B[0], fresh_B)
    assert target.belief_history == []
    assert target.inference_history == []
    assert target._posterior_mappings == []
    assert any("A_0" in r.getMessage() for r in caplog.records)
    assert any("belief_history" in r.getMessage() for r in caplog.records)

    # Corrupt matrices are rejected as well: non-normalized or non-finite
    bad = (await _run(_module(), 3)).to_state_dict()
    bad["arrays"]["A_0"] = bad["arrays"]["A_0"] * 2.0
    bad["arrays"]["B_0"] = np.full_like(bad["arrays"]["B_0"], np.nan)
    victim = _module()
    fresh_A, fresh_B = victim.A[0].copy(), victim.B[0].copy()
    victim.from_state_dict(bad)
    np.testing.assert_array_equal(victim.A[0], fresh_A)
    np.testing.assert_array_equal(victim.B[0], fresh_B)

    # The module still runs on its own matrices
    result = await target.infer_and_act(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    assert result.posterior_beliefs.shape == (6,)


@pytest.mark.asyncio
async def test_restore_caps_histories():
    source = await _run(_module(), 3)
    state = _json_round_trip(source.to_state_dict())
    big = 5 * HISTORY_LIMIT
    h = state["histories"]
    h["prediction_errors"] = [0.5] * big
    h["free_energy_history"] = [1.5] * big
    h["meta_accuracy_history"] = [0.5] * big
    h["meta_confidence_history"] = [0.5] * big
    h["meta_recent_outcomes"] = [True] * big
    h["homeostatic_free_energy_history"] = [0.2] * big
    h["homeostatic_valence_history"] = [{"curiosity": -0.1}] * big
    h["inference_history"] = h["inference_history"][:1] * big
    h["posterior_mappings"] = h["posterior_mappings"][:1] * big
    state["arrays"]["belief_history"] = np.full((big, 4), 0.25)

    restored = _module()
    restored.from_state_dict(state)

    assert len(restored.prediction_errors) == HISTORY_LIMIT
    assert len(restored.free_energy_history) == HISTORY_LIMIT
    assert len(restored.belief_history) == HISTORY_LIMIT
    assert len(restored.meta_state._accuracy_history) == HISTORY_LIMIT
    assert len(restored.meta_state._confidence_history) == HISTORY_LIMIT
    assert len(restored.meta_state._recent_outcomes) == restored.meta_state._window_size
    drives = restored.homeostatic_drives
    assert len(drives.free_energy_history) == HISTORY_LIMIT
    assert len(drives.valence_history) == HISTORY_LIMIT
    assert len(restored.inference_history) <= 100
    assert len(restored._posterior_mappings) <= 200


@pytest.mark.asyncio
async def test_untrusted_state_restores_counters_and_drives_only(caplog):
    source = await _run(_module(), 12)
    state = source.to_state_dict()
    del state["scalars"]["schema_version"]
    state["scalars"]["homeostatic_drive_levels"]["curiosity"] = 0.33

    restored = _module()
    fresh_A = restored.A[0].copy()
    with caplog.at_level(logging.WARNING, logger="mtc.consciousness.active_inference"):
        restored.from_state_dict(state)

    assert any("schema_version" in r.getMessage() for r in caplog.records)
    # Operational state restores
    assert restored.total_inferences == source.total_inferences
    assert restored.homeostatic_drives.drives["curiosity"].current_level == 0.33
    # Learned state and the verdict history start clean
    np.testing.assert_array_equal(restored.A[0], fresh_A)
    assert restored.total_model_updates == 0
    assert restored.prediction_errors == []
    assert restored.prediction_ledger.scored_count == 0
    assert restored._pending_prediction_id is None
    assert restored.meta_state.total_predictions == 0
    assert restored.meta_state.model_confidence == 0.5
    assert restored.meta_state._accuracy_history == []


@pytest.mark.asyncio
async def test_world_model_mismatch_keeps_fresh_model():
    state = (await _run(_module(), 4)).to_state_dict()
    state["scalars"]["world_model"] = "custom"
    restored = _module()
    fresh_A = restored.A[0].copy()

    restored.from_state_dict(state)

    np.testing.assert_array_equal(restored.A[0], fresh_A)
    assert restored.prediction_errors == []


@pytest.mark.asyncio
async def test_restore_clamps_drive_levels_to_floor():
    state = (await _run(_module(), 2)).to_state_dict()
    state["scalars"]["homeostatic_drive_levels"]["curiosity"] = 0.0
    state["scalars"]["homeostatic_drive_levels"]["safety"] = 3.0

    restored = _module()
    restored.from_state_dict(state)

    curiosity = restored.homeostatic_drives.drives["curiosity"]
    assert curiosity.current_level == pytest.approx(0.1 * curiosity.optimal_level)
    assert restored.homeostatic_drives.drives["safety"].current_level == 1.0


@pytest.mark.asyncio
async def test_belief_history_lists_restore_into_belief_history():
    source = await _run(_module(), 6)
    state = _json_round_trip(source.to_state_dict())
    stacked = state["arrays"].pop("belief_history")
    rows = stacked.tolist()
    rows.insert(0, [0.5, 0.5])  # a row of the wrong width is skipped
    state["histories"]["belief_history_lists"] = rows

    restored = _module()
    restored.from_state_dict(state)

    assert len(restored.belief_history) == stacked.shape[0]
    assert all(isinstance(b, np.ndarray) for b in restored.belief_history)
    assert all(b.shape == (4,) for b in restored.belief_history)
    np.testing.assert_array_equal(restored.belief_history[-1], stacked[-1])
    assert not hasattr(restored, "belief_history_lists")


def test_state_dict_of_fresh_module_is_json_safe():
    module = _module()
    state = module.to_state_dict()
    json.dumps(state["scalars"])
    json.dumps(state["histories"])
    assert state["histories"]["prediction_errors"] == []

    restored = _module()
    restored.from_state_dict(state)
    assert restored.total_inferences == 0
