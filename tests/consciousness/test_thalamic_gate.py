"""
Thalamic gate: the pre-competition salience filter of the workspace.

Safety content always passes, low arousal raises the activation threshold,
repeated content habituates, goal-aligned content gets a small boost, and the
running statistics count what was filtered and why.
"""

import json

import pytest

from mtc.consciousness.enhanced_global_workspace import (
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.thalamic_gate import (
    GatingStats,
    ThalamicGate,
    ThalamicGateConfig,
)


def _candidate(
    activation: float,
    summary: str = "synthetic percept",
    source: WorkspaceCandidateSource = WorkspaceCandidateSource.SENSORY,
    content_type: str = "perception",
) -> WorkspaceCandidate:
    return WorkspaceCandidate(
        content=summary,
        content_type=content_type,
        summary=summary,
        source=source,
        activation_level=activation,
    )


def test_config_defaults_carry_the_tuned_constants():
    config = ThalamicGateConfig()
    assert config.arousal_threshold_base == 0.05
    assert config.arousal_threshold_range == 0.30
    assert config.habituation_window == 50
    assert config.habituation_penalty == 0.15
    assert config.habituation_max_penalty == 0.40
    assert config.relevance_boost == 0.10
    assert config.irrelevance_threshold == 0.08


def test_config_rejects_an_empty_habituation_window():
    with pytest.raises(ValueError):
        ThalamicGateConfig(habituation_window=0)


def test_safety_source_always_passes():
    gate = ThalamicGate(ThalamicGateConfig())
    safety = _candidate(
        0.0, summary="safety check", source=WorkspaceCandidateSource.SAFETY
    )
    # Zero activation at zero arousal would fail every filter
    passed = gate.gate([safety], arousal=0.0, goals=[])
    assert passed == [safety]
    assert gate.stats.total_safety_bypassed == 1
    assert gate.stats.total_gated == 0


def test_low_arousal_raises_threshold():
    config = ThalamicGateConfig()
    gate = ThalamicGate(config)
    assert gate.arousal_threshold(1.0) == pytest.approx(0.05)
    assert gate.arousal_threshold(0.0) == pytest.approx(0.35)
    assert gate.arousal_threshold(0.5) == pytest.approx(0.20)
    # Out-of-range arousal is clamped
    assert gate.arousal_threshold(5.0) == pytest.approx(0.05)

    weak = _candidate(0.25, summary="a faint percept")
    assert gate.gate([weak], arousal=1.0) == [weak]
    fresh = ThalamicGate(config)
    assert fresh.gate([_candidate(0.25, summary="a faint percept")], arousal=0.0) == []
    assert fresh.stats.gated_by_arousal == 1


def test_habituation_penalises_repeats():
    gate = ThalamicGate(ThalamicGateConfig())
    # Arousal 1.0 puts the threshold at 0.05; activation 0.30 passes once,
    # then each repeat costs 0.15 until the 0.40 cap
    first = gate.gate([_candidate(0.30, summary="same percept")], arousal=1.0)
    assert len(first) == 1
    second = gate.gate([_candidate(0.30, summary="same percept")], arousal=1.0)
    assert len(second) == 1  # 0.30 - 0.15 = 0.15 still clears 0.05
    third = gate.gate([_candidate(0.30, summary="same percept")], arousal=1.0)
    assert third == []  # 0.30 - 0.30 = 0.0 falls below 0.05
    assert gate.stats.gated_by_habituation == 1

    # Different content is not habituated
    other = gate.gate([_candidate(0.30, summary="another percept")], arousal=1.0)
    assert len(other) == 1


def test_habituation_penalty_is_capped():
    config = ThalamicGateConfig(habituation_penalty=0.3, habituation_max_penalty=0.4)
    gate = ThalamicGate(config)
    for _ in range(5):
        gate.gate([_candidate(0.9, summary="loud percept")], arousal=1.0)
    # Five repeats would cost 1.5 uncapped; the cap keeps 0.9 - 0.4 = 0.5
    assert len(gate.gate([_candidate(0.9, summary="loud percept")], arousal=1.0)) == 1


def test_goal_relevance_boost():
    # Arousal 1.0 (threshold 0.05) and an activation of 0.06 clear the arousal
    # gate but sit under the 0.08 irrelevance floor; the goal boost lifts it
    config = ThalamicGateConfig()
    irrelevant = ThalamicGate(config).gate(
        [_candidate(0.06, summary="weather report")], arousal=1.0, goals=["chess"]
    )
    assert irrelevant == []

    gate = ThalamicGate(config)
    relevant = gate.gate(
        [_candidate(0.06, summary="a chess opening")], arousal=1.0, goals=["chess"]
    )
    assert len(relevant) == 1
    assert gate.stats.gated_by_irrelevance == 0


def test_upstream_goal_relevance_is_trusted():
    gate = ThalamicGate(ThalamicGateConfig())
    candidate = _candidate(0.06, summary="unrelated words")
    candidate.goal_relevance = 1.0
    assert gate.gate([candidate], arousal=1.0, goals=["chess"]) == [candidate]


def test_stats_count_gated():
    gate = ThalamicGate(ThalamicGateConfig())
    candidates = [
        _candidate(0.9, summary="strong percept"),
        _candidate(0.01, summary="faint percept"),
        _candidate(0.0, summary="alarm", source=WorkspaceCandidateSource.SAFETY),
    ]
    passed = gate.gate(candidates, arousal=0.5)
    assert len(passed) == 2
    stats = gate.stats
    assert isinstance(stats, GatingStats)
    assert stats.total_candidates == 3
    assert stats.total_gated == 1
    assert stats.gated_by_arousal == 1
    assert stats.total_safety_bypassed == 1
    summary = stats.as_dict()
    assert summary["gate_rate"] == pytest.approx(1 / 3)


def test_empty_input_leaves_stats_untouched():
    gate = ThalamicGate(ThalamicGateConfig())
    assert gate.gate([], arousal=0.5) == []
    assert gate.stats.total_candidates == 0
    assert gate.stats.as_dict()["gate_rate"] == 0.0


def test_state_dict_round_trip():
    gate = ThalamicGate(ThalamicGateConfig())
    gate.gate([_candidate(0.9, summary="repeat me")], arousal=1.0)
    gate.gate([_candidate(0.01, summary="faint")], arousal=0.0)
    state = gate.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    state = json.loads(json.dumps(state))

    restored = ThalamicGate(ThalamicGateConfig())
    restored.from_state_dict(state)
    assert restored.stats.as_dict() == gate.stats.as_dict()
    # The habituation window came back: the repeat is penalised again
    assert restored.repeat_count(_candidate(0.9, summary="repeat me")) == 1
