"""Damasio layers: persistence, circadian models, body metrics, sensory input.

Synthetic drives only; no conversation text from any real source.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from mtc.consciousness.damasio import (
    BodyState,
    CoreConsciousness,
    CoreExperience,
    DamasioLayers,
    ExtendedConsciousness,
    ExtendedState,
    ProtoSelfState,
    Protoself,
    SomaticMarker,
)
from mtc.consciousness.damasio.protoself import (
    CIRCADIAN_CYCLE_LENGTH,
    gather_system_body_metrics,
    wall_clock_alertness,
)


class _Drives:
    """Deterministic stand-in for HomeostaticDrives."""

    def __init__(self, attention: float = 0.7, valence: float = 0.1, fe: float = 0.2):
        self.attention = attention
        self.valence = valence
        self.fe = fe

    def get_drive_state(self):
        return {
            "attention_budget": {
                "current_level": self.attention,
                "optimal_level": 0.7,
                "deviation": abs(self.attention - 0.7),
                "urgency": 0.1,
            },
            "social_connection": {
                "current_level": 0.4,
                "optimal_level": 0.6,
                "deviation": 0.2,
                "urgency": 0.3,
            },
        }

    def get_overall_valence(self):
        return self.valence

    def get_free_energy(self):
        return self.fe


class _Candidate:
    def __init__(self, summary: str, content_type: str):
        self.summary = summary
        self.content_type = content_type


class _Winner:
    def __init__(self, summary: str, content_type: str, salience: float):
        self.candidate = _Candidate(summary, content_type)
        self.salience = salience


HISTORY = [
    {"role": "user", "content": "a synthetic question about patterns in data"},
    {"role": "assistant", "content": "a synthetic answer about patterns in data"},
    {"role": "user", "content": "another synthetic follow up message here"},
]


def _hub_round_trip(state: dict) -> dict:
    return {
        "scalars": json.loads(json.dumps(state["scalars"])),
        "arrays": {k: np.asarray(v) for k, v in state["arrays"].items()},
        "histories": json.loads(json.dumps(state["histories"])),
    }


def _drive_sequence():
    return [_Drives(attention=0.5 + 0.05 * i, valence=0.1 * (i % 3)) for i in range(6)]


def test_protoself_round_trip():
    ps = Protoself(stability_window=7)
    for drives in _drive_sequence():
        ps.update(drives)

    restored = Protoself(stability_window=7)
    restored.from_state_dict(_hub_round_trip(ps.to_state_dict()))

    assert restored.cycle_count == ps.cycle_count
    np.testing.assert_allclose(restored._previous_body_vector, ps._previous_body_vector)
    for name in (
        "_stability_history",
        "_body_delta_history",
        "_valence_trend",
        "_energy_trend",
    ):
        assert getattr(restored, name).maxlen == 7
        assert list(getattr(restored, name)) == pytest.approx(list(getattr(ps, name)))
    assert isinstance(restored._latest_state, ProtoSelfState)
    assert isinstance(restored._latest_state.body_state, BodyState)
    assert restored.get_statistics() == ps.get_statistics()

    nxt = _Drives(attention=0.9)
    a = ps.update(nxt)
    b = restored.update(nxt)
    assert b.body_delta == pytest.approx(a.body_delta)
    assert b.stability == pytest.approx(a.stability)


def test_protoself_rejects_wrong_vector_shape():
    ps = Protoself()
    ps.update(_Drives())
    state = ps.to_state_dict()
    state["arrays"]["previous_body_vector"] = np.ones(3)
    fresh = Protoself()
    fresh.from_state_dict(state)
    assert fresh._previous_body_vector is None


def _core_and_states():
    ps = Protoself()
    core = CoreConsciousness(history_window=9)
    winners = [
        _Winner("synthetic discovery", "discovery", 0.8),
        _Winner("synthetic warning", "warning", 0.6),
    ]
    for drives in _drive_sequence():
        core.bind(ps.update(drives), workspace_winners=winners)
    return ps, core


def test_core_round_trip():
    _, core = _core_and_states()
    restored = CoreConsciousness(history_window=9)
    restored.from_state_dict(_hub_round_trip(core.to_state_dict()))

    assert restored.cycle_count == core.cycle_count
    assert restored._binding_history.maxlen == 9
    assert list(restored._binding_history) == pytest.approx(list(core._binding_history))
    assert list(restored._fok_history) == pytest.approx(list(core._fok_history))
    latest = restored._latest_experience
    assert isinstance(latest, CoreExperience)
    assert all(isinstance(m, SomaticMarker) for m in latest.somatic_markers)
    assert latest.somatic_valence == pytest.approx(
        core._latest_experience.somatic_valence
    )
    assert restored.get_statistics() == core.get_statistics()


def test_extended_round_trip():
    ps, core = _core_and_states()
    ext = ExtendedConsciousness(history_window=11)
    markers = {"interests": ["patterns", "data"], "core_values": ["care"]}
    experiences = []
    for drives in _drive_sequence():
        exp = core.bind(ps.update(drives), workspace_winners=[])
        experiences.append(exp)
        ext.extend(exp, conversation_history=HISTORY, identity_markers=markers)

    restored = ExtendedConsciousness(history_window=11)
    restored.from_state_dict(_hub_round_trip(ext.to_state_dict()))

    assert restored.cycle_count == ext.cycle_count
    assert restored._themes_seen.maxlen == ext._themes_seen.maxlen
    assert list(restored._themes_seen) == list(ext._themes_seen)
    assert restored._binding_trend.maxlen == 11
    assert isinstance(restored._latest_state, ExtendedState)
    assert restored.get_statistics() == ext.get_statistics()

    a = ext.extend(experiences[-1], conversation_history=HISTORY)
    b = restored.extend(experiences[-1], conversation_history=HISTORY)
    assert b.autobiographic_self.to_dict() == a.autobiographic_self.to_dict()
    assert b.narrative_context == a.narrative_context


@pytest.mark.asyncio
async def test_aggregate_round_trip():
    layers = DamasioLayers(stability_window=6)
    winners = [_Winner("synthetic topic", "thought", 0.7)]
    for drives in _drive_sequence():
        await layers.process(
            homeostatic_drives=drives,
            workspace_winners=winners,
            conversation_history=HISTORY,
        )

    state = layers.to_state_dict()
    assert all(isinstance(v, np.ndarray) for v in state["arrays"].values())
    restored = DamasioLayers(stability_window=6)
    restored.from_state_dict(_hub_round_trip(state))

    assert restored.cycle_count == layers.cycle_count
    assert restored.generate_context() == layers.generate_context()
    assert restored.generate_report() == layers.generate_report()
    assert restored.get_statistics() == layers.get_statistics()


def test_sinusoid_circadian_is_continuous():
    ps = Protoself()
    phases = []
    for _ in range(CIRCADIAN_CYCLE_LENGTH * 2 + 5):
        phases.append(ps.update(_Drives()).body_state.circadian_phase)
    jumps = np.abs(np.diff(phases))
    assert jumps.max() <= 0.05
    assert min(phases) >= 0.0 and max(phases) <= 1.0
    assert max(phases) > 0.99  # the full range is still covered

    legacy = Protoself(circadian="v04_sawtooth")
    old = [legacy.update(_Drives()).body_state.circadian_phase for _ in range(1001)]
    assert old[0] == pytest.approx(1 / CIRCADIAN_CYCLE_LENGTH)
    assert np.abs(np.diff(old)).max() > 0.9  # the pinned discontinuity


def test_wall_clock_circadian_uses_injected_now():
    afternoon = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)
    ps = Protoself(
        circadian="wall_clock",
        timezone="UTC",
        wake_hour=6,
        sleep_hour=22,
        clock=lambda: afternoon,
    )
    state = ps.update(_Drives())
    assert state.body_state.circadian_phase == pytest.approx(
        wall_clock_alertness(15.0, 6, 22)
    )
    assert ps.time_of_day == "afternoon"
    assert ps.alertness == pytest.approx(state.body_state.circadian_phase)
    assert state.body_state.to_dict()["time_period"] == "afternoon"

    # The daily peak falls in the afternoon
    hours = np.arange(0.0, 24.0, 1.0 / 60.0)
    curve = np.array([wall_clock_alertness(h, 6, 22) for h in hours])
    assert 12.0 <= hours[int(np.argmax(curve))] < 17.0

    # Sleep pressure ramps down over the last waking hour: no jump at 22:00
    evening = (hours >= 20.0) & (hours <= 23.0)
    assert np.abs(np.diff(curve[evening])).max() < 0.01
    assert np.abs(np.diff(curve)).max() < 0.01


def test_wall_clock_only_properties_absent_otherwise():
    ps = Protoself()
    assert ps.time_of_day is None
    assert ps.alertness is None
    assert "time_period" not in ps.update(_Drives()).body_state.to_dict()


def test_invalid_circadian_mode_is_rejected():
    with pytest.raises(ValueError, match="circadian"):
        Protoself(circadian="lunar")


def test_system_metrics_do_not_fabricate_latency():
    metrics = gather_system_body_metrics()
    assert "response_latency_ms" not in metrics
    assert "error_rate" not in metrics
    timed = gather_system_body_metrics(cycle_latency_ms=120.0)
    assert timed["response_latency_ms"] == pytest.approx(120.0)

    drives = _Drives(attention=0.7, fe=0.2)
    without = Protoself().update(drives, system_metrics={"cpu_load": 0.1})
    baseline = Protoself().update(drives)
    # Energy stays on the attention drive; metabolic is not capped by an
    # invented memory reading.
    assert without.body_state.energy_level == pytest.approx(0.7)
    assert without.body_state.metabolic_state == pytest.approx(
        baseline.body_state.metabolic_state
    )

    with_latency = Protoself().update(drives, system_metrics={"response_latency_ms": 0})
    assert with_latency.body_state.energy_level == pytest.approx(1.0)


def test_sensory_emotion_valence_shift_reaches_valence():
    ps = Protoself()
    ps.register_sensory_emotion(
        source="audio", arousal=0.2, valence_shift=0.4, pain=0.1, confidence=1.0
    )
    first = ps.update(_Drives(attention=0.5, valence=0.1))
    assert first.primordial_feelings["pleasure_pain"] == pytest.approx(0.5)
    assert first.body_state.energy_level == pytest.approx(0.7)
    assert first.body_state.interoceptive_state["sensory_emotion"]["source"] == "audio"

    # Consumed once: the next cycle carries no shift
    second = ps.update(_Drives(attention=0.5, valence=0.1))
    assert "sensory_emotion" not in second.body_state.interoceptive_state
    assert second.primordial_feelings["pleasure_pain"] == pytest.approx(0.1)

    # Label lookup scales the mapped shift by confidence
    labelled = Protoself()
    assert labelled.register_sensory_label("joy", confidence=0.5) is True
    assert labelled.register_sensory_label("unmapped label") is False
    state = labelled.update(_Drives(attention=0.5, valence=0.0))
    expected = Protoself.SENSORY_EMOTION_BODY_MAP["joy"]["valence_shift"] * 0.5
    assert state.primordial_feelings["pleasure_pain"] == pytest.approx(expected)


def test_somatic_valence_mean_of_markers():
    markers = [
        SomaticMarker(option="a", valence=0.2, intensity=0.5, body_source="test"),
        SomaticMarker(option="b", valence=-0.4, intensity=0.5, body_source="test"),
        SomaticMarker(option="c", valence=0.8, intensity=0.5, body_source="test"),
    ]
    exp = CoreExperience(
        self_world_binding=0.5,
        perturbation_source="synthetic",
        feeling_of_knowing=0.5,
        somatic_markers=markers,
        here_now_salience=0.5,
    )
    assert exp.somatic_valence == pytest.approx(0.2)
    assert exp.to_dict()["somatic_valence"] == pytest.approx(0.2)

    empty = CoreExperience(
        self_world_binding=0.1,
        perturbation_source="none",
        feeling_of_knowing=0.1,
        somatic_markers=[],
        here_now_salience=0.0,
    )
    assert empty.somatic_valence == 0.0
