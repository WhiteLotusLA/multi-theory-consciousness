"""
Tests for the helpers the experiment and assessment runners share: the warmup
shift-type profiles, the switch families and the one workspace construction.
"""

import inspect
from pathlib import Path

import pytest

from mtc.assessment.assessment import ConsciousnessAssessment
from mtc.consciousness.attention_schema import AttentionShiftType
from mtc.consciousness.enhanced_global_workspace import (
    COMPETITION_FLAT,
    EnhancedGlobalWorkspace,
)
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.consciousness.phi_tracker import PhiTracker
from mtc.core import runner_support
from mtc.core.param_overrides import load_overrides
from mtc.core.runner_support import (
    AST_SWITCHES,
    CYCLE_SWITCHES,
    FEP_BEHAVIOUR_SWITCHES,
    GWT_SWITCHES,
    attention_shift_type,
    build_cycle_augmenter,
    build_phi_tracker,
    build_workspace,
)
from mtc.pipeline.cycle_augmenter import CycleAugmenter, CycleAugmenterConfig

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "experiments" / "param_schema.toml"
LEGACY_PRESET = ROOT / "experiments" / "presets" / "legacy_v04.toml"

# The one legacy row that cannot stand alone: the v04 loop quality formula
# reads a binding quality only the v04 binding evidence measures (A6).
LEGACY_ROWS_WITH_A_COMPANION = {"blt.loop_quality_formula": "blt.binding_evidence"}


def _legacy_overrides():
    return load_overrides(str(LEGACY_PRESET), str(SCHEMA))


def _flat_rows(node, prefix=""):
    """Dotted key to value for every leaf of a nested override mapping."""
    rows = {}
    for key, value in node.items():
        if isinstance(value, dict):
            rows.update(_flat_rows(value, prefix + key + "."))
        else:
            rows[prefix + key] = value
    return rows


def _one_row(dotted, value):
    """The nested override mapping that sets one dotted row alone."""
    parts = dotted.split(".")
    overrides = current = {}
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
    return overrides


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
    # Every gwt switch is a constructor field of the workspace, and the two
    # legacy booleans of the broadcast coverage and the sustain rate are in
    constructor = inspect.signature(EnhancedGlobalWorkspace).parameters
    assert all(name in constructor for name in GWT_SWITCHES)
    assert {"legacy_broadcast_receivers", "legacy_sustain_rate"} <= set(GWT_SWITCHES)
    # The refractory clock is a temporal field (amendment A8): it reaches the
    # temporal config and has no entry in the workspace switch family
    assert "refractory_clock" not in GWT_SWITCHES
    assert set(runner_support._TEMPORAL_FIELDS) == {
        "competition_duration_ms",
        "ignition_threshold_time_ms",
        "broadcast_propagation_ms",
        "refractory_period_ms",
        "timestep_ms",
        "refractory_clock",
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
            "legacy_broadcast_receivers": True,
            "legacy_sustain_rate": True,
            "body_metrics_enabled": True,
            "precision_feedback_enabled": False,
            "self_prediction_enabled": False,
            "refractory_period_ms": 123.0,
            "refractory_clock": "elapsed",
        },
        "fep": {
            "drive_model": "v04",
            "legacy_double_decay": True,
            "num_hidden_states": 99,  # geometry stays the workspace's own
        },
        "blt": {
            "field_evidencing_threshold": 0.77,
            "binding_evidence": "v04_salience_posterior",
            "strange_loop_test": "v04_confidence_match",
        },
        "damasio": {"stability_window": 33, "first_delta": "v04_zero"},
        "hot": {
            "hot_generation_threshold": 0.55,
            "overall_confidence_source": "v04_constant",
            "introspection_counting": "v04_per_hot",
        },
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
    assert workspace.legacy_broadcast_receivers is True
    # The sustain-rate switch lives on the ignition detector it is forwarded to
    assert workspace.ignition_detector.legacy_sustain_rate is True
    assert workspace.body_metrics_enabled is True
    assert workspace.precision_feedback_enabled is False
    assert workspace.self_prediction_enabled is False
    assert workspace.temporal_config.refractory_period_ms == pytest.approx(123.0)
    assert workspace.temporal_config.refractory_clock == "elapsed"

    fep = workspace.active_inference.config
    assert fep.drive_model == "v04"
    assert fep.legacy_double_decay is True
    assert fep.num_hidden_states != 99

    loop = workspace.beautiful_loop
    assert loop.field_evidencing_threshold == pytest.approx(0.77)
    assert loop.binding_evidence == "v04_salience_posterior"
    assert loop.bayesian_binding.binding_evidence == "v04_salience_posterior"
    assert loop.strange_loop_test == "v04_confidence_match"
    assert loop.epistemic_depth.strange_loop_test == "v04_confidence_match"
    assert workspace.damasio.protoself._stability_history.maxlen == 33
    assert workspace.damasio.protoself.first_delta == "v04_zero"
    assert workspace.metacognition.hot_threshold == pytest.approx(0.55)
    assert workspace.metacognition.overall_confidence_source == "v04_constant"
    assert workspace.metacognition.introspection_counting == "v04_per_hot"

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


def test_refractory_override_reaches_the_temporal_config():
    """gwt.refractory_clock is forwarded as a temporal field (amendment A8):
    per_cycle unless set, elapsed when the overrides say so, flat or nested."""
    assert build_workspace({}, None).temporal_config.refractory_clock == "per_cycle"
    flat = build_workspace({"gwt": {"refractory_clock": "elapsed"}}, None)
    assert flat.temporal_config.refractory_clock == "elapsed"
    nested = build_workspace(
        {"gwt": {"temporal": {"refractory_clock": "elapsed"}}}, None
    )
    assert nested.temporal_config.refractory_clock == "elapsed"
    with pytest.raises(ValueError, match="refractory_clock"):
        build_workspace({"gwt": {"refractory_clock": "wall"}}, None)


def test_build_workspace_introspection_cooldown_default():
    """The runners pin the metacognition cooldown at 0.0 through
    ``introspection_cooldown_default``; an explicit hot override wins and the
    live runtime, which passes nothing, keeps the module's own default."""
    module_default = build_workspace({}, None).metacognition.introspection_cooldown
    assert module_default == pytest.approx(1.0)
    pinned = build_workspace({}, None, introspection_cooldown_default=0.0)
    assert pinned.metacognition.introspection_cooldown == 0.0
    explicit = build_workspace(
        {"hot": {"introspection_cooldown": 2.5}},
        None,
        introspection_cooldown_default=0.0,
    )
    assert explicit.metacognition.introspection_cooldown == pytest.approx(2.5)
    overrides = {"hot": {"hot_generation_threshold": 0.5}}
    build_workspace(overrides, None, introspection_cooldown_default=0.0)
    assert overrides == {"hot": {"hot_generation_threshold": 0.5}}


def test_legacy_preset_rows_reach_their_constructors():
    """Every switch row of the legacy preset lands on the module it names."""
    overrides = _legacy_overrides()
    workspace = build_workspace(overrides, None, introspection_cooldown_default=0.0)
    assert workspace.competition_mode == COMPETITION_FLAT
    assert workspace.broadcaster.legacy_coverage_ratio is True
    assert workspace.legacy_broadcast_receivers is True
    assert workspace.ignition_detector.legacy_sustain_rate is True
    assert workspace.temporal_config.refractory_clock == "per_cycle"
    hot = workspace.metacognition
    assert hot.overall_confidence_source == "v04_constant"
    assert hot.introspection_counting == "v04_per_hot"
    # The preset's explicit 1.0 s gate wins over the runner pin
    assert hot.introspection_cooldown == pytest.approx(1.0)
    loop = workspace.beautiful_loop
    assert loop.loop_quality_formula == "v04"
    assert loop.binding_evidence == "v04_salience_posterior"
    assert loop.bayesian_binding.binding_evidence == "v04_salience_posterior"
    assert loop.strange_loop_test == "v04_confidence_match"
    assert loop.epistemic_depth.strange_loop_test == "v04_confidence_match"
    assert workspace.damasio.protoself.first_delta == "v04_zero"
    assert workspace.damasio.protoself.circadian == "v04_sawtooth"
    assert workspace.attention_schema.provenance_mode == "legacy_strength"
    assert workspace.active_inference.config.drive_model == "v04"
    assessment = ConsciousnessAssessment(**overrides["assessment"])
    assert assessment.noise_baseline_matching == "v04_any"
    assert assessment.indicator_formulas == "v04"
    assert overrides["runner"] == {
        "warmup_profile": "v04",
        "assess_blt_source": "standalone",
    }


def test_each_legacy_row_alone_builds_or_names_its_companion():
    """Every key of the legacy preset applied alone builds a workspace, except
    the v04 loop quality formula, which raises the dependency error naming
    its companion row (amendment A6)."""
    rows = _flat_rows(_legacy_overrides())
    assert len(rows) >= 52
    raised = {}
    for dotted, value in rows.items():
        try:
            workspace = build_workspace(_one_row(dotted, value), None)
        except ValueError as exc:
            raised[dotted] = str(exc)
            continue
        assert isinstance(workspace, EnhancedGlobalWorkspace), dotted
    assert set(raised) == set(LEGACY_ROWS_WITH_A_COMPANION)
    for dotted, companion in LEGACY_ROWS_WITH_A_COMPANION.items():
        assert dotted in raised[dotted]
        assert companion in raised[dotted]
    # The row and its companion together build
    both = build_workspace(
        {
            "blt": {
                "loop_quality_formula": "v04",
                "binding_evidence": "v04_salience_posterior",
            }
        },
        None,
    )
    assert both.beautiful_loop.loop_quality_formula == "v04"


def test_build_workspace_does_not_mutate_overrides():
    overrides = {"gwt": {"capacity": 4, "refractory_period_ms": 10.0}, "ast": {}}
    snapshot = {key: dict(value) for key, value in overrides.items()}
    build_workspace(overrides, None)
    assert overrides == snapshot


def test_build_cycle_augmenter_pins_off():
    workspace = build_workspace({}, None)
    assert set(CYCLE_SWITCHES) == {
        "drive_candidates",
        "drive_urgency_threshold",
        "prospective_candidates",
        "crossmodal_binding",
        "gate_arousal",
        "pad_from_state",
        "candidate_cap",
    }
    pinned = build_cycle_augmenter({}, workspace)
    assert isinstance(pinned, CycleAugmenter)
    assert pinned.config == CycleAugmenterConfig.all_off()
    assert pinned.workspace is workspace

    # A set switch moves; every other one stays off
    partial = build_cycle_augmenter({"cycle": {"drive_candidates": True}}, workspace)
    assert partial.config.drive_candidates is True
    assert partial.config.pad_from_state is False
    assert partial.config.gate_arousal == "workspace"
    assert partial.config.candidate_cap == 0

    # The live pipeline starts from the defaults instead
    live = build_cycle_augmenter({}, workspace, pinned_off=False)
    assert live.config == CycleAugmenterConfig()

    # Keys outside the cycle family are ignored, the pad model is passed on
    pad = PADAffectModel()
    with_pad = build_cycle_augmenter({"gwt": {"capacity": 4}}, workspace, pad_model=pad)
    assert with_pad.pad_model is pad


def test_build_phi_tracker_min_transitions():
    tracker = build_phi_tracker({}, min_transitions=10)
    assert isinstance(tracker, PhiTracker)
    assert tracker.min_transitions == 10

    tuned = build_phi_tracker({"scale": 2.0, "warmup_reference": 3.0}, 24)
    assert tuned.min_transitions == 24
    assert tuned.phi_scale == pytest.approx(2.0)
    assert tuned.phi_warmup_reference == pytest.approx(3.0)
