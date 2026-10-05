"""
Tests for the helpers the experiment and assessment runners share: the warmup
shift-type profiles, the switch families and the one workspace construction.
"""

import pytest

from mtc.consciousness.attention_schema import AttentionShiftType
from mtc.consciousness.enhanced_global_workspace import (
    COMPETITION_FLAT,
    EnhancedGlobalWorkspace,
)
from mtc.consciousness.phi_tracker import PhiTracker
from mtc.core import runner_support
from mtc.core.runner_support import (
    AST_SWITCHES,
    FEP_BEHAVIOUR_SWITCHES,
    GWT_SWITCHES,
    attention_shift_type,
    build_phi_tracker,
    build_workspace,
)


def test_v04_profile_is_all_captured():
    shifts = {attention_shift_type(cycle, "v04") for cycle in range(40)}
    assert shifts == {AttentionShiftType.CAPTURED}


def test_v05_profile_rotation():
    rotation = [attention_shift_type(cycle, "v05") for cycle in range(8)]
    one_pass = [
        AttentionShiftType.VOLUNTARY,
        AttentionShiftType.CAPTURED,
        AttentionShiftType.GOAL_DRIVEN,
        AttentionShiftType.CAPTURED,
    ]
    assert rotation == one_pass + one_pass


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="profile"):
        attention_shift_type(0, "v06")


def test_switch_families_name_real_fields():
    config = EnhancedGlobalWorkspace.default_active_inference_config()
    assert all(hasattr(config, name) for name in FEP_BEHAVIOUR_SWITCHES)
    workspace = build_workspace({}, None)
    assert workspace.competition_mode
    assert len(set(GWT_SWITCHES)) == len(GWT_SWITCHES)
    assert len(set(AST_SWITCHES)) == len(AST_SWITCHES)
    assert set(runner_support._TEMPORAL_FIELDS) == {
        "competition_duration_ms",
        "ignition_threshold_time_ms",
        "broadcast_propagation_ms",
        "refractory_period_ms",
        "timestep_ms",
    }


def test_build_workspace_passes_every_switch_family():
    overrides = {
        "gwt": {
            "capacity": 5,
            "ignition_threshold": 0.4,
            "amplification_factor": 1.5,
            "competition_mode": COMPETITION_FLAT,
            "thalamic_gate_enabled": False,
            "theory_interactions_enabled": False,
            "legacy_coverage_ratio": True,
            "body_metrics_enabled": True,
            "precision_feedback_enabled": False,
            "self_prediction_enabled": False,
            "refractory_period_ms": 123.0,
        },
        "fep": {
            "drive_model": "v04",
            "legacy_double_decay": True,
            "num_hidden_states": 99,  # geometry stays the workspace's own
        },
        "blt": {"field_evidencing_threshold": 0.77},
        "damasio": {"stability_window": 33},
        "hot": {"hot_generation_threshold": 0.55},
        "ast": {
            "provenance_mode": "legacy_strength",
            "voluntary_boost": 0.31,
            "history_size": 12,
            "not_an_ast_switch": 1,
        },
    }
    tracker = build_phi_tracker({}, min_transitions=7)

    workspace = build_workspace(overrides, tracker)

    assert workspace.phi_tracker is tracker
    assert workspace.bottleneck.capacity == 5
    assert workspace.bottleneck.threshold == pytest.approx(0.4)
    assert workspace.ignition_detector.amplification == pytest.approx(1.5)
    assert workspace.competition_mode == COMPETITION_FLAT
    assert workspace.bottleneck.thalamic_gate is None
    assert workspace.theory_interactions is None
    assert workspace.broadcaster.legacy_coverage_ratio is True
    assert workspace.body_metrics_enabled is True
    assert workspace.precision_feedback_enabled is False
    assert workspace.self_prediction_enabled is False
    assert workspace.temporal_config.refractory_period_ms == pytest.approx(123.0)

    fep = workspace.active_inference.config
    assert fep.drive_model == "v04"
    assert fep.legacy_double_decay is True
    assert fep.num_hidden_states != 99

    assert workspace.beautiful_loop.field_evidencing_threshold == pytest.approx(0.77)
    assert workspace.damasio.protoself._stability_history.maxlen == 33
    assert workspace.metacognition.hot_threshold == pytest.approx(0.55)

    schema = workspace.attention_schema
    assert schema.provenance_mode == "legacy_strength"
    assert schema.voluntary_boost == pytest.approx(0.31)
    assert schema.history_size == 12


def test_build_workspace_defaults_and_body_metrics_default():
    workspace = build_workspace({}, None)
    assert workspace.phi_tracker is None
    assert workspace.bottleneck.capacity == 7
    assert workspace.body_metrics_enabled is True
    assert workspace.attention_schema.provenance_mode == "typed"

    seeded = build_workspace({}, None, body_metrics_default=False)
    assert seeded.body_metrics_enabled is False
    # An explicit override beats the default
    explicit = build_workspace(
        {"gwt": {"body_metrics_enabled": True}}, None, body_metrics_default=False
    )
    assert explicit.body_metrics_enabled is True


def test_build_workspace_accepts_nested_temporal_table():
    workspace = build_workspace({"gwt": {"temporal": {"timestep_ms": 5.0}}}, None)
    assert workspace.temporal_config.timestep_ms == pytest.approx(5.0)


def test_build_workspace_does_not_mutate_overrides():
    overrides = {"gwt": {"capacity": 4, "refractory_period_ms": 10.0}, "ast": {}}
    snapshot = {key: dict(value) for key, value in overrides.items()}
    build_workspace(overrides, None)
    assert overrides == snapshot


def test_build_phi_tracker_min_transitions():
    tracker = build_phi_tracker({}, min_transitions=10)
    assert isinstance(tracker, PhiTracker)
    assert tracker.min_transitions == 10

    tuned = build_phi_tracker({"scale": 2.0, "warmup_reference": 3.0}, 24)
    assert tuned.min_transitions == 24
    assert tuned.phi_scale == pytest.approx(2.0)
    assert tuned.phi_warmup_reference == pytest.approx(3.0)
