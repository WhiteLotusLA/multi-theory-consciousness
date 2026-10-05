"""
Homeostatic drive dynamics: activity vocabulary, single decay, decay floor
and urgent-drive ranking, with the v0.4.0 behaviour kept behind switches.
"""

import pytest

from mtc.consciousness.active_inference import (
    ActiveInferenceConfig,
    ActiveInferenceModule,
    HomeostaticDrives,
)


def _level(drives, name):
    return float(drives.drives[name].current_level)


@pytest.mark.asyncio
async def test_attention_budget_depletes_on_consciousness_cycle_under_f4():
    drives = HomeostaticDrives(drive_model="f4")
    before = _level(drives, "attention_budget")

    await drives.update_drives({"type": "consciousness_cycle", "intensity": 0.8})

    # Cognitive load (0.8 * 0.1) plus the natural decay (0.01)
    assert _level(drives, "attention_budget") == pytest.approx(before - 0.08 - 0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("activity", ["conversation", "reasoning"])
async def test_f4_cognitive_activity_depletes_attention(activity):
    drives = HomeostaticDrives(drive_model="f4")
    before = _level(drives, "attention_budget")
    await drives.update_drives({"type": activity, "intensity": 0.5})
    assert _level(drives, "attention_budget") == pytest.approx(before - 0.05 - 0.01)


@pytest.mark.asyncio
async def test_v04_vocabulary_ignores_consciousness_cycle():
    drives = HomeostaticDrives(drive_model="v04")
    before = _level(drives, "attention_budget")

    await drives.update_drives({"type": "consciousness_cycle", "intensity": 0.8})

    # Only the natural decay applies
    assert _level(drives, "attention_budget") == pytest.approx(before - 0.01)


@pytest.mark.asyncio
async def test_rest_recovers_attention_under_f4_only():
    f4 = HomeostaticDrives(drive_model="f4")
    v04 = HomeostaticDrives(drive_model="v04")
    for drives in (f4, v04):
        drives.drives["attention_budget"].current_level = 0.4

    await f4.update_drives({"type": "rest"})
    await v04.update_drives({"type": "rest"})

    assert _level(f4, "attention_budget") == pytest.approx(0.4 + 0.05 - 0.01)
    assert _level(v04, "attention_budget") == pytest.approx(0.4 - 0.01)


@pytest.mark.asyncio
async def test_social_interaction_feeds_social_connection_under_f4():
    drives = HomeostaticDrives(drive_model="f4")
    drives.drives["social_connection"].current_level = 0.3

    await drives.update_drives({"type": "social_interaction", "intensity": 0.5})

    assert _level(drives, "social_connection") == pytest.approx(0.3 + 0.1 - 0.005)


@pytest.mark.asyncio
@pytest.mark.parametrize("activity", ["reasoning", "understanding", "sensemaking"])
async def test_coherence_recovers_on_sensemaking_under_f4(activity):
    drives = HomeostaticDrives(drive_model="f4")
    drives.drives["coherence"].current_level = 0.5

    await drives.update_drives({"type": activity, "intensity": 0.5})

    assert _level(drives, "coherence") == pytest.approx(0.5 + 0.075 - 0.01)


@pytest.mark.asyncio
async def test_curiosity_single_decay():
    drives = HomeostaticDrives()
    curiosity = drives.drives["curiosity"]
    social = drives.drives["social_connection"]
    curiosity_before = float(curiosity.current_level)
    social_before = float(social.current_level)

    await drives.update_drives({"type": "idle", "intensity": 0.1})

    # Idle depletion comes once, from the natural decay step
    assert float(curiosity.current_level) == pytest.approx(
        curiosity_before - curiosity.decay_rate
    )
    assert float(social.current_level) == pytest.approx(
        social_before - social.decay_rate
    )

    # Curiosity still recovers with real activity
    await drives.update_drives({"type": "learning", "intensity": 1.0})
    expected = curiosity_before - curiosity.decay_rate
    expected += curiosity.recovery_rate - curiosity.decay_rate
    assert float(curiosity.current_level) == pytest.approx(expected)


@pytest.mark.asyncio
async def test_legacy_double_decay_reproduces_old_dynamics():
    """The v0.4.0 rules: double idle decay for curiosity and social, floor 0."""
    drives = HomeostaticDrives(
        drive_model="v04", decay_floor=0.0, legacy_double_decay=True
    )
    curiosity, social = 0.5, 0.6
    for _ in range(30):
        await drives.update_drives({"type": "idle", "intensity": 0.1})
        # v0.4.0: impact = -decay_rate, then the natural decay, floor at zero
        curiosity = max(0.0, min(1.0, curiosity - 0.02) - 0.02)
        social = max(0.0, min(1.0, social - 0.005) - 0.005)
        assert _level(drives, "curiosity") == pytest.approx(curiosity)
        assert _level(drives, "social_connection") == pytest.approx(social)

    # Thirty idle cycles sink v0.4.0 curiosity all the way to zero
    assert _level(drives, "curiosity") == 0.0


@pytest.mark.asyncio
async def test_decay_floor():
    floored = HomeostaticDrives(decay_floor=0.1)
    unfloored = HomeostaticDrives(decay_floor=0.0)
    for drives in (floored, unfloored):
        drives.drives["curiosity"].current_level = 0.01

    await floored.update_drives({"type": "idle"})
    await unfloored.update_drives({"type": "idle"})

    optimal = floored.drives["curiosity"].optimal_level
    assert _level(floored, "curiosity") == pytest.approx(0.1 * optimal)
    assert _level(unfloored, "curiosity") == 0.0

    for _ in range(200):
        await floored.update_drives({"type": "idle"})
    for name, drive in floored.drives.items():
        assert drive.current_level >= floored.floor_for(drive) - 1e-12, name


def test_get_urgent_drives_sorted():
    drives = HomeostaticDrives()
    drives.drives["attention_budget"].current_level = 0.0  # 0.7 * 1.0 = 0.70
    drives.drives["curiosity"].current_level = 0.0  # 0.5 * 0.8 = 0.40
    drives.drives["social_connection"].current_level = 0.4  # 0.2 * 0.9 = 0.18

    urgent = drives.get_urgent_drives()

    assert [name for name, _ in urgent] == ["attention_budget", "curiosity"]
    assert urgent[0][1] == pytest.approx(0.7)
    assert urgent[1][1] == pytest.approx(0.4)
    assert all(isinstance(u, float) for _, u in urgent)

    lower = drives.get_urgent_drives(threshold=0.1)
    assert [name for name, _ in lower] == [
        "attention_budget",
        "curiosity",
        "social_connection",
    ]
    assert drives.get_urgent_drives(threshold=0.9) == []


def test_module_passes_drive_switches():
    config = ActiveInferenceConfig(
        num_hidden_states=4,
        num_actions=2,
        planning_horizon=2,
        drive_model="v04",
        drive_decay_floor=0.0,
        legacy_double_decay=True,
    )
    module = ActiveInferenceModule(config)

    assert module.homeostatic_drives.drive_model == "v04"
    assert module.homeostatic_drives.decay_floor == 0.0
    assert module.homeostatic_drives.legacy_double_decay is True


def test_drive_switch_validation():
    with pytest.raises(ValueError, match="drive_model"):
        ActiveInferenceConfig(drive_model="unknown")
    with pytest.raises(ValueError, match="drive_decay_floor"):
        ActiveInferenceConfig(drive_decay_floor=1.5)
    with pytest.raises(ValueError, match="drive_model"):
        HomeostaticDrives(drive_model="unknown")
