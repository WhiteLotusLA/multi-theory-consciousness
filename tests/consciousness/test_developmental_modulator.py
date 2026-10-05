"""Developmental modulator: gate, interpolation and a guarded workspace apply."""

import os
from types import SimpleNamespace

import pytest

from mtc.consciousness.developmental_modulator import (
    AppliedParameters,
    DevelopmentalAnchors,
    DevelopmentalModulator,
    DevelopmentalParameters,
    developmental_enabled,
)
from mtc.core.config import Settings


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    def build(**overrides):
        return Settings(**overrides)

    return build


def fake_workspace(**overrides):
    parts = dict(
        bottleneck=SimpleNamespace(capacity=7, threshold=0.3),
        metacognition=SimpleNamespace(enable_third_order=False),
        active_inference=SimpleNamespace(
            config=SimpleNamespace(planning_horizon=5, num_actions=2)
        ),
    )
    parts.update(overrides)
    return SimpleNamespace(**parts)


def test_disabled_by_default_for_research_subject(cfg):
    assert developmental_enabled(cfg()) is False
    assert developmental_enabled(cfg(COMPANION_TYPE="research_subject")) is False


def test_child_companion_enables(cfg):
    assert developmental_enabled(cfg(COMPANION_TYPE="child")) is True


def test_flag_enables_other_types(cfg):
    for kind in ("sibling", "friend", "partner", "mentor", "research_subject"):
        settings = cfg(COMPANION_TYPE=kind, DEVELOPMENTAL_STAGE_ENABLED=True)
        assert developmental_enabled(settings) is True


def test_interpolation_endpoints_and_midpoint(cfg):
    modulator = DevelopmentalModulator(settings=cfg(COMPANION_TYPE="child"))
    first = modulator.parameters_for(1)
    last = modulator.parameters_for(10)
    assert isinstance(first, DevelopmentalParameters)
    assert (first.bottleneck_capacity, first.ignition_threshold) == (3, 0.5)
    assert (first.metacognition_depth, first.planning_horizon) == (1, 1)
    assert first.emotional_regulation == 0.1
    assert first.stage_fraction == 0.0
    assert (last.bottleneck_capacity, last.ignition_threshold) == (9, 0.2)
    assert (last.metacognition_depth, last.planning_horizon) == (5, 8)
    assert last.emotional_regulation == 0.8
    assert last.stage_fraction == 1.0
    three = DevelopmentalModulator(num_stages=3).parameters_for(2)
    assert three.bottleneck_capacity == 6
    assert three.ignition_threshold == 0.35
    assert three.metacognition_depth == 3
    assert three.emotional_regulation == 0.45
    assert three.stage_fraction == 0.5


def test_custom_anchors_are_used():
    anchors = DevelopmentalAnchors(bottleneck_capacity=(2, 6))
    params = DevelopmentalModulator(anchors=anchors, num_stages=5).parameters_for(5)
    assert params.bottleneck_capacity == 6


def test_stage_clamped_to_range():
    modulator = DevelopmentalModulator()
    assert modulator.parameters_for(0).stage == 1
    assert modulator.parameters_for(-4).bottleneck_capacity == 3
    assert modulator.parameters_for(99).stage == 10
    assert modulator.parameters_for(99).bottleneck_capacity == 9


def test_two_stages_minimum():
    with pytest.raises(ValueError):
        DevelopmentalModulator(num_stages=1)


def test_apply_sets_only_existing_attributes(cfg):
    modulator = DevelopmentalModulator(settings=cfg(COMPANION_TYPE="child"))
    workspace = fake_workspace()
    result = modulator.apply_to_workspace(workspace, 7)
    assert isinstance(result, AppliedParameters)
    expected = modulator.parameters_for(7)
    assert workspace.bottleneck.capacity == expected.bottleneck_capacity
    assert workspace.bottleneck.threshold == expected.ignition_threshold
    assert workspace.metacognition.enable_third_order is True
    assert workspace.active_inference.config.planning_horizon == (
        expected.planning_horizon
    )
    assert result.applied["bottleneck_capacity"] == expected.bottleneck_capacity
    assert result.applied["planning_horizon_effective"] == "next agent rebuild"
    # No attribute was invented on any object.
    assert not hasattr(workspace.bottleneck, "developmental_stage")
    assert not hasattr(workspace.metacognition, "developmental_depth_ceiling")
    assert vars(workspace.active_inference.config).keys() == {
        "planning_horizon",
        "num_actions",
    }
    bare = SimpleNamespace()
    skipped = modulator.apply_to_workspace(bare, 7)
    assert skipped.applied == {}
    assert {"bottleneck", "metacognition", "planning_horizon"} <= set(skipped.skipped)
    assert vars(bare) == {}


def test_third_order_only_from_depth_three(cfg):
    modulator = DevelopmentalModulator(settings=cfg(COMPANION_TYPE="child"))
    low = fake_workspace(metacognition=SimpleNamespace(enable_third_order=True))
    modulator.apply_to_workspace(low, 1)
    assert low.metacognition.enable_third_order is False


def test_capacity_is_set_under_hierarchical_competition(cfg):
    modulator = DevelopmentalModulator(settings=cfg(COMPANION_TYPE="child"))

    def hierarchical():
        return fake_workspace(
            competition_mode="hierarchical",
            bottleneck=SimpleNamespace(
                capacity=7, threshold=0.3, LEVEL_CAPACITY={0: 3, 1: 3, 2: 2}
            ),
        )

    workspace = hierarchical()
    result = modulator.apply_to_workspace(workspace, 10)
    assert workspace.bottleneck.capacity == 9
    assert result.applied["bottleneck_capacity"] == 9
    assert "bottleneck_capacity" not in result.skipped
    assert workspace.bottleneck.threshold == 0.2
    capacities = []
    for stage in range(1, 11):
        stage_workspace = hierarchical()
        modulator.apply_to_workspace(stage_workspace, stage)
        capacities.append(stage_workspace.bottleneck.capacity)
    assert capacities == sorted(capacities)
    flat = fake_workspace(competition_mode="flat")
    modulator.apply_to_workspace(flat, 10)
    assert flat.bottleneck.capacity == 9


def test_planning_horizon_beyond_policy_budget_is_skipped(cfg):
    modulator = DevelopmentalModulator(settings=cfg(COMPANION_TYPE="child"))
    workspace = fake_workspace(
        active_inference=SimpleNamespace(
            config=SimpleNamespace(planning_horizon=5, num_actions=4)
        )
    )
    result = modulator.apply_to_workspace(workspace, 10)
    assert workspace.active_inference.config.planning_horizon == 5
    assert "planning_horizon" in result.skipped


def test_regulation_skipped_without_pad_support(cfg):
    modulator = DevelopmentalModulator(settings=cfg(COMPANION_TYPE="child"))
    result = modulator.apply_to_workspace(fake_workspace(), 6)
    assert "emotional_regulation" in result.skipped
    plain_pad = SimpleNamespace()
    result = modulator.apply_to_workspace(fake_workspace(), 6, pad_model=plain_pad)
    assert "emotional_regulation" in result.skipped
    assert not hasattr(plain_pad, "regulation_strength")
    capable = SimpleNamespace(regulation_strength=0.0)
    result = modulator.apply_to_workspace(fake_workspace(), 6, pad_model=capable)
    assert (
        capable.regulation_strength == modulator.parameters_for(6).emotional_regulation
    )
    assert "emotional_regulation" in result.applied


def test_disabled_apply_is_a_no_op(cfg):
    modulator = DevelopmentalModulator(settings=cfg())
    workspace = fake_workspace()
    result = modulator.apply_to_workspace(workspace, 8)
    assert result.applied == {}
    assert result.skipped == {"all": "disabled"}
    assert workspace.bottleneck.capacity == 7
    assert workspace.bottleneck.threshold == 0.3
    assert workspace.metacognition.enable_third_order is False
    forced = modulator.apply_to_workspace(workspace, 8, force=True)
    assert forced.applied
    assert workspace.bottleneck.capacity != 7
