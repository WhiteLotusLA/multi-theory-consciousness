"""
Hierarchical competition, the thalamic gate, cross-theory bias, the extended
consciousness state and persistence of the enhanced global workspace.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from mtc.consciousness.active_inference import ActiveInferenceConfig
from mtc.consciousness import enhanced_global_workspace as egw
from mtc.consciousness.enhanced_global_workspace import (
    HIERARCHY_LEVELS,
    SOURCE_LEVEL_MAP,
    AttentionBottleneck,
    BroadcastEvent,
    CompetitionEvent,
    ConsciousnessState,
    EnhancedGlobalWorkspace,
    IgnitionEvent,
    TemporalWorkspaceConfig,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
    WorkspaceContent,
)
from mtc.consciousness.thalamic_gate import ThalamicGate
from mtc.consciousness.theory_interaction import (
    TheoryInteractionManager,
    TheoryModulations,
)

FAST_FEP = dict(num_hidden_states=4, num_actions=2, planning_horizon=2)


def _workspace(**kwargs) -> EnhancedGlobalWorkspace:
    kwargs.setdefault("active_inference_config", ActiveInferenceConfig(**FAST_FEP))
    kwargs.setdefault(
        "temporal_config", TemporalWorkspaceConfig(refractory_period_ms=0.0)
    )
    return EnhancedGlobalWorkspace(**kwargs)


def _candidate(
    summary: str,
    source: WorkspaceCandidateSource = WorkspaceCandidateSource.SENSORY,
    activation: float = 0.9,
    emotional: float = 0.6,
    content_type: str = "perception",
) -> WorkspaceCandidate:
    return WorkspaceCandidate(
        content=summary,
        content_type=content_type,
        summary=summary,
        source=source,
        source_module=source.value,
        activation_level=activation,
        emotional_salience=emotional,
    )


def _layered_candidates(tag: str = "") -> list:
    """Six moderate candidates per level, none salient enough to promote."""
    candidates = []
    for i in range(6):
        candidates.append(
            _candidate(f"percept {tag}{i}", activation=0.5, emotional=0.0)
        )
        candidates.append(
            _candidate(
                f"memory {tag}{i}",
                source=WorkspaceCandidateSource.MEMORY,
                activation=0.5,
                emotional=0.0,
                content_type="memory",
            )
        )
        candidates.append(
            _candidate(
                f"reasoning {tag}{i}",
                source=WorkspaceCandidateSource.LLM,
                activation=0.5,
                emotional=0.0,
                content_type="reasoning",
            )
        )
    return candidates


# ============================================================================
# Sources and levels
# ============================================================================


def test_new_sources_and_level_map():
    for name in ("CONCEPT", "GIST", "HOMEOSTATIC", "INTERNAL"):
        assert hasattr(WorkspaceCandidateSource, name)
    assert HIERARCHY_LEVELS == {"perceptual": 0, "integrative": 1, "executive": 2}
    assert set(SOURCE_LEVEL_MAP) == set(WorkspaceCandidateSource)


def test_hierarchy_level_auto_assigned_from_source():
    assert _candidate("x").hierarchy_level == 0
    memory = _candidate("x", source=WorkspaceCandidateSource.MEMORY)
    assert memory.hierarchy_level == 1
    reasoning = _candidate("x", source=WorkspaceCandidateSource.LLM)
    assert reasoning.hierarchy_level == 2
    explicit = WorkspaceCandidate(
        source=WorkspaceCandidateSource.SENSORY, hierarchy_level=2
    )
    assert explicit.hierarchy_level == 2


def test_competition_event_has_metadata():
    event = CompetitionEvent(
        timestamp=0.0,
        candidates_count=1,
        winners_count=1,
        threshold_used=0.3,
        winning_sources=["sensory"],
        top_salience=0.5,
        avg_salience=0.5,
    )
    assert event.metadata == {}


# ============================================================================
# Hierarchical competition
# ============================================================================


@pytest.mark.asyncio
async def test_promotion_does_not_duplicate_winner():
    bottleneck = AttentionBottleneck(capacity=7, ignition_threshold=0.3)
    strong = _candidate("a vivid percept", activation=1.0, emotional=0.8)

    winners = await bottleneck.compete_hierarchical([strong])

    assert [w.candidate.id for w in winners] == [strong.id]
    assert bottleneck.last_competition.metadata["promotions"] >= 1
    assert bottleneck.last_competition.winners_count == 1

    # Through the workspace: one slot, one first-order registration, one
    # stream entry
    workspace = _workspace()
    before = len(workspace.metacognition.first_order_states)
    state = await workspace.process_consciousness_cycle(
        [_candidate("a vivid percept", activation=1.0, emotional=0.8)]
    )
    ids = [w.candidate.id for w in state.workspace_contents]
    assert len(ids) == len(set(ids)) == 1
    assert len(workspace.metacognition.first_order_states) - before == 1
    assert len(workspace.stream_of_consciousness) == 1
    assert state.cross_level_promotions >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "activation, emotional",
    [(0.8, 0.5), (0.9, 0.5), (1.0, 0.5), (1.0, 1.0), (0.95, 0.8)],
)
async def test_promotion_never_lowers_final_salience(activation, emotional):
    """A promoted winner keeps its unpenalized salience, as in flat mode."""
    flat = AttentionBottleneck(
        capacity=7, ignition_threshold=0.3, lateral_inhibition=0.0
    )
    hierarchical = AttentionBottleneck(
        capacity=7, ignition_threshold=0.3, lateral_inhibition=0.0
    )
    flat_winners = await flat.compete_for_access(
        [_candidate("a percept", activation=activation, emotional=emotional)]
    )
    hier_winners = await hierarchical.compete_hierarchical(
        [_candidate("a percept", activation=activation, emotional=emotional)]
    )
    assert len(flat_winners) == len(hier_winners) == 1
    assert hier_winners[0].salience == pytest.approx(flat_winners[0].salience)


@pytest.mark.asyncio
async def test_final_salience_monotonic_in_raw_salience():
    """A stronger percept never ranks below a weaker one after promotion."""
    saliences = []
    for activation in (0.6, 0.7, 0.8, 0.9, 1.0):
        bottleneck = AttentionBottleneck(
            capacity=7, ignition_threshold=0.3, lateral_inhibition=0.0
        )
        winners = await bottleneck.compete_hierarchical(
            [_candidate("a percept", activation=activation, emotional=0.5)]
        )
        saliences.append(winners[0].salience)
    assert saliences == sorted(saliences)
    assert len(set(saliences)) == len(saliences)


@pytest.mark.asyncio
async def test_level_capacity_respected():
    bottleneck = AttentionBottleneck(capacity=20, ignition_threshold=0.3)
    winners = await bottleneck.compete_hierarchical(_layered_candidates())

    per_level = {level: 0 for level in AttentionBottleneck.LEVEL_CAPACITY}
    for w in winners:
        per_level[w.candidate.hierarchy_level] += 1
    assert per_level == AttentionBottleneck.LEVEL_CAPACITY
    assert len(winners) == sum(AttentionBottleneck.LEVEL_CAPACITY.values())
    metadata = bottleneck.last_competition.metadata
    assert metadata["hierarchical"] is True
    assert metadata["promotions"] == 0
    assert metadata["level_candidates"] == {0: 6, 1: 6, 2: 6}

    # Ranks are 1..N by salience after the merge
    assert [w.competition_rank for w in winners] == list(range(1, len(winners) + 1))


@pytest.mark.asyncio
async def test_total_capacity_caps_the_merge():
    bottleneck = AttentionBottleneck(capacity=4, ignition_threshold=0.3)
    winners = await bottleneck.compete_hierarchical(_layered_candidates())
    assert len(winners) == 4
    # Executive winners are taken first in the merge
    assert sum(1 for w in winners if w.candidate.hierarchy_level == 2) == 2


@pytest.mark.asyncio
async def test_avg_salience_covers_all_scored_candidates(monkeypatch):
    bottleneck = AttentionBottleneck(capacity=20, ignition_threshold=0.3)
    scored = []
    score = bottleneck._calculate_composite_salience

    async def spy(candidate):
        value = await score(candidate)
        scored.append(value)
        return value

    monkeypatch.setattr(bottleneck, "_calculate_composite_salience", spy)
    await bottleneck.compete_hierarchical(_layered_candidates())
    event = bottleneck.last_competition
    # 18 candidates were scored and 8 won; the average spans all 18, as in
    # the flat competition
    assert len(scored) == event.candidates_count == 18
    assert event.avg_salience == pytest.approx(float(np.mean(scored)))


@pytest.mark.asyncio
async def test_hierarchical_competition_leaves_novelty_window_alone():
    bottleneck = AttentionBottleneck(capacity=7, ignition_threshold=0.3)
    candidates = _layered_candidates()
    await bottleneck.compete_hierarchical(candidates)
    # One novelty hash per scored candidate, nothing appended afterwards
    assert len(bottleneck._recent_content_hashes) == len(candidates)


@pytest.mark.asyncio
async def test_safety_bypasses_hierarchy():
    bottleneck = AttentionBottleneck(capacity=7, ignition_threshold=0.3)
    safety = _candidate(
        "safety check",
        source=WorkspaceCandidateSource.SAFETY,
        activation=0.1,
        emotional=0.0,
        content_type="alert",
    )
    percepts = [
        _candidate(f"percept {i}", activation=0.5, emotional=0.0) for i in range(5)
    ]
    winners = await bottleneck.compete_hierarchical(percepts + [safety])

    assert safety.id in {w.candidate.id for w in winners}
    # The safety winner does not consume the perceptual level's capacity
    perceptual = [
        w
        for w in winners
        if w.candidate.source == WorkspaceCandidateSource.SENSORY
        and w.candidate.hierarchy_level == 0
    ]
    assert len(perceptual) == AttentionBottleneck.LEVEL_CAPACITY[0]
    assert bottleneck.last_competition.metadata["safety_bypass"] == 1


@pytest.mark.asyncio
async def test_flat_mode_matches_compete_for_access(monkeypatch):
    # Salience carries a recency term read from the wall clock, so the two
    # candidate sets below must see the same age or a near-tie can flip under load.
    import mtc.consciousness.enhanced_global_workspace as egw_module

    monkeypatch.setattr(egw_module.time, "time", lambda: 1_700_000_000.0)
    workspace = _workspace(competition_mode="flat", thalamic_gate_enabled=False)
    calls = {"flat": 0, "hierarchical": 0, "winners": None}
    flat = workspace.bottleneck.compete_for_access
    hierarchical = workspace.bottleneck.compete_hierarchical

    async def spy_flat(candidates):
        calls["flat"] += 1
        calls["winners"] = await flat(candidates)
        return calls["winners"]

    async def spy_hierarchical(candidates):
        calls["hierarchical"] += 1
        return await hierarchical(candidates)

    monkeypatch.setattr(workspace.bottleneck, "compete_for_access", spy_flat)
    monkeypatch.setattr(workspace.bottleneck, "compete_hierarchical", spy_hierarchical)

    await workspace.process_consciousness_cycle(_layered_candidates())
    assert calls == {"flat": 1, "hierarchical": 0, "winners": calls["winners"]}

    # Same ranking as a bare bottleneck running the v0.4.0 flat competition
    reference = AttentionBottleneck(capacity=7, ignition_threshold=0.3)
    expected = await reference.compete_for_access(_layered_candidates())
    assert [w.candidate.summary for w in calls["winners"]] == [
        w.candidate.summary for w in expected
    ]
    assert len(calls["winners"]) == 7


def test_unknown_competition_mode_rejected():
    with pytest.raises(ValueError):
        _workspace(competition_mode="tournament")


# ============================================================================
# Thalamic gate in the workspace
# ============================================================================


@pytest.mark.asyncio
async def test_gate_disabled_passes_everything():
    candidates = [
        _candidate("strong percept", activation=0.9),
        _candidate("faint percept", activation=0.01, emotional=0.0),
    ]
    ungated = _workspace(thalamic_gate_enabled=False)
    assert ungated.bottleneck.thalamic_gate is None
    state = await ungated.process_consciousness_cycle(list(candidates))
    assert ungated.bottleneck.competition_history[-1].candidates_count == 2
    assert state.thalamic_gate_rate is None
    assert state.thalamic_gated_count is None

    gated = _workspace()
    assert isinstance(gated.bottleneck.thalamic_gate, ThalamicGate)
    state = await gated.process_consciousness_cycle(
        [
            _candidate("strong percept", activation=0.9),
            _candidate("faint percept", activation=0.01, emotional=0.0),
        ]
    )
    assert gated.bottleneck.competition_history[-1].candidates_count == 1
    assert state.thalamic_gated_count == 1
    assert state.thalamic_gate_rate == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_gate_rate_is_per_cycle():
    workspace = _workspace()
    await workspace.process_consciousness_cycle(
        [_candidate("faint percept", activation=0.01, emotional=0.0)]
    )
    state = await workspace.process_consciousness_cycle(
        [_candidate("another strong percept", activation=0.9)]
    )
    assert state.thalamic_gated_count == 0
    assert state.thalamic_gate_rate == 0.0


@pytest.mark.asyncio
async def test_arousal_derived_from_emotional_state_unless_supplied():
    workspace = _workspace()
    await workspace.process_consciousness_cycle(
        [_candidate("percept")], emotional_state={"arousal": 0.9}
    )
    assert workspace.bottleneck.arousal == pytest.approx(0.9)

    workspace.set_arousal(0.2)
    await workspace.process_consciousness_cycle(
        [_candidate("percept two")], emotional_state={"arousal": 0.9}
    )
    assert workspace.bottleneck.arousal == pytest.approx(0.2)

    workspace.set_arousal(7.0)
    assert workspace.bottleneck.arousal == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["caller", "emotional_state", "pad", "default"])
async def test_arousal_source_recorded(source):
    workspace = _workspace()
    emotional_state = None
    if source == "caller":
        workspace.set_arousal(0.3)
        emotional_state = {"arousal": 0.9}
    elif source == "emotional_state":
        emotional_state = {"arousal": 0.8}
    elif source == "pad":
        first = await workspace.process_consciousness_cycle([_candidate("first")])
        assert first.arousal_source == "default"
        first.pad_arousal = 0.9  # a caller records PAD arousal on the state

    state = await workspace.process_consciousness_cycle(
        [_candidate("percept")], emotional_state=emotional_state
    )

    assert state.arousal_source == source
    expected = {"caller": 0.3, "emotional_state": 0.8, "pad": 0.9, "default": 0.5}
    assert workspace.bottleneck.arousal == pytest.approx(expected[source])


@pytest.mark.asyncio
async def test_arousal_falls_through_unusable_sources():
    workspace = _workspace()
    first = await workspace.process_consciousness_cycle(
        [_candidate("first")], emotional_state={"arousal": 0.9}
    )
    assert first.arousal_source == "emotional_state"
    first.pad_arousal = 0.2

    # A non-finite entry is skipped in favour of the recorded PAD arousal
    second = await workspace.process_consciousness_cycle(
        [_candidate("second")], emotional_state={"arousal": float("nan")}
    )
    assert second.arousal_source == "pad"
    assert workspace.bottleneck.arousal == pytest.approx(0.2)

    # With no source at all the gate returns to its neutral default instead of
    # keeping the last cycle's value
    third = await workspace.process_consciousness_cycle([_candidate("third")])
    assert third.arousal_source == "default"
    assert workspace.bottleneck.arousal == pytest.approx(0.5)

    # A workspace without a gate has no arousal to source
    ungated = _workspace(thalamic_gate_enabled=False)
    state = await ungated.process_consciousness_cycle([_candidate("percept")])
    assert state.arousal_source is None


@pytest.mark.asyncio
async def test_gate_rate_none_on_empty_cycle():
    workspace = _workspace()
    state = await workspace.process_consciousness_cycle([])
    assert state.thalamic_gated_count == 0
    assert state.thalamic_gate_rate is None  # 0 of 0 is not a rate

    state = await workspace.process_consciousness_cycle([_candidate("a percept")])
    assert state.thalamic_gate_rate == 0.0  # a real zero: one seen, none gated


@pytest.mark.asyncio
async def test_reset_clears_theory_interactions_and_gate_stats():
    workspace = _workspace()
    for i in range(3):
        await workspace.process_consciousness_cycle(
            [_candidate(f"percept {i}", activation=0.9)]
        )
    manager = workspace.theory_interactions
    assert manager.get_status()["cycle_count"] == 3
    assert workspace.bottleneck.thalamic_gate.stats.total_candidates > 0

    workspace.reset()

    status = workspace.theory_interactions.get_status()
    assert status["cycle_count"] == 0
    assert status["active_interactions"] == 0
    assert all(value == 0.0 for value in status["modulations"].values())
    stats = workspace.bottleneck.thalamic_gate.stats
    assert stats.total_candidates == 0
    assert stats.total_gated == 0
    assert workspace.bottleneck.cross_theory_bias == 0.0
    # The reset modules keep their configuration and stay persistable
    assert workspace.theory_interactions.config is manager.config
    assert "theory_interactions" in workspace._persisted_submodules()
    assert "thalamic_gate" in workspace._persisted_submodules()

    # The next cycle starts from a fresh arousal history
    state = await workspace.process_consciousness_cycle([_candidate("after reset")])
    assert state.arousal_source == "default"


def test_attention_schema_kwargs_reach_the_owned_module():
    workspace = _workspace(
        attention_schema_kwargs={
            "provenance_mode": "legacy_strength",
            "voluntary_boost": 0.35,
        }
    )
    assert workspace.attention_schema.provenance_mode == "legacy_strength"
    assert workspace.attention_schema.schema.provenance_mode == "legacy_strength"
    assert workspace.attention_schema.voluntary_boost == pytest.approx(0.35)

    # Omitted, the construction is the module's own defaults
    default = _workspace()
    assert default.attention_schema.provenance_mode == "typed"
    assert default.attention_schema.voluntary_boost == pytest.approx(0.2)


# ============================================================================
# Cross-theory bias
# ============================================================================


@pytest.mark.asyncio
async def test_cross_theory_bias_applies_next_cycle(monkeypatch):
    workspace = _workspace()
    assert isinstance(workspace.theory_interactions, TheoryInteractionManager)
    assert workspace.bottleneck.cross_theory_bias == 0.0

    seen_states = []

    def fake_update(state):
        seen_states.append(state)
        return TheoryModulations(somatic_salience_bias=0.1, phi_competition_bias=0.05)

    monkeypatch.setattr(workspace.theory_interactions, "update_from_state", fake_update)

    bias_during = []
    hierarchical = workspace.bottleneck.compete_hierarchical

    async def spy(candidates):
        bias_during.append(workspace.bottleneck.cross_theory_bias)
        return await hierarchical(candidates)

    monkeypatch.setattr(workspace.bottleneck, "compete_hierarchical", spy)

    first = await workspace.process_consciousness_cycle([_candidate("one")])
    assert seen_states == [first]
    assert bias_during == [0.0]
    assert workspace.bottleneck.cross_theory_bias == pytest.approx(0.15)

    await workspace.process_consciousness_cycle([_candidate("two")])
    assert bias_during[1] == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_cross_theory_bias_scales_with_novelty():
    plain = AttentionBottleneck()
    biased = AttentionBottleneck()
    biased.cross_theory_bias = 0.2
    candidate_a = _candidate("same text", activation=0.5, emotional=0.0)
    candidate_b = _candidate("same text", activation=0.5, emotional=0.0)
    candidate_b.timestamp = candidate_a.timestamp
    s_plain = await plain._calculate_composite_salience(candidate_a)
    s_biased = await biased._calculate_composite_salience(candidate_b)
    assert s_biased - s_plain == pytest.approx(
        0.2 * candidate_b.novelty_score, abs=1e-4
    )


def test_cross_theory_bias_setter_guards():
    bottleneck = AttentionBottleneck()
    bottleneck.cross_theory_bias = float("nan")
    assert bottleneck.cross_theory_bias == 0.0
    bottleneck.cross_theory_bias = 5.0
    assert bottleneck.cross_theory_bias == 1.0


@pytest.mark.asyncio
async def test_theory_interactions_disabled_leave_bias_at_zero():
    workspace = _workspace(theory_interactions_enabled=False)
    assert workspace.theory_interactions is None
    for i in range(3):
        await workspace.process_consciousness_cycle([_candidate(f"percept {i}")])
    assert workspace.bottleneck.cross_theory_bias == 0.0


# ============================================================================
# Extended consciousness state
# ============================================================================


def test_extended_state_slots_default_to_none():
    state = ConsciousnessState(
        is_conscious=False,
        primary_content=None,
        workspace_contents=[],
        ignition_events=0,
        integration_level=0.0,
        broadcast_coverage=0.0,
        attention_focus="none",
        attention_distribution={},
        stream_position=0,
    )
    for name in (
        "prediction_error_rank",
        "dominant_hierarchy_level",
        "level_winner_counts",
        "cross_level_promotions",
        "thalamic_gate_rate",
        "thalamic_gated_count",
        "arousal_source",
        "architectural_phi",
        "phi_warming_up",
        "rpt_deep_recurrence",
        "rpt_superficial_recurrence",
        "causal_emergence",
        "causal_emergence_warming_up",
        "prm_discrimination",
        "pad_pleasure",
        "pad_arousal",
        "pad_dominance",
        "pad_label",
    ):
        assert getattr(state, name) is None, name


@pytest.mark.asyncio
async def test_hierarchical_fields_filled_in_hierarchical_mode():
    workspace = _workspace(capacity=20)
    state = await workspace.process_consciousness_cycle(_layered_candidates())
    assert state.level_winner_counts == {0: 3, 1: 3, 2: 2}
    assert state.dominant_hierarchy_level in (0, 1, 2)
    assert state.cross_level_promotions == 0

    flat = _workspace(competition_mode="flat")
    flat_state = await flat.process_consciousness_cycle(_layered_candidates())
    assert flat_state.level_winner_counts is None
    assert flat_state.dominant_hierarchy_level is None
    assert flat_state.cross_level_promotions is None


@pytest.mark.asyncio
async def test_dominant_level_reflects_admitted_level():
    """A winner promoted from level 0 to 1 counts under level 1, not 0."""
    workspace = _workspace(thalamic_gate_enabled=False)
    state = await workspace.process_consciousness_cycle(
        [_candidate("a vivid percept", activation=0.9, emotional=0.5)]
    )
    (winner,) = state.workspace_contents
    assert winner.candidate.hierarchy_level == 0  # its source level
    assert winner.admitted_level == 1
    assert state.cross_level_promotions == 1
    assert state.dominant_hierarchy_level == 1
    assert state.level_winner_counts == {0: 0, 1: 1, 2: 0}

    # Promoted twice, it counts under the executive level
    twice = _workspace(thalamic_gate_enabled=False)
    state = await twice.process_consciousness_cycle(
        [_candidate("a vivid percept", activation=1.0, emotional=0.8)]
    )
    assert state.dominant_hierarchy_level == 2
    assert state.level_winner_counts == {0: 0, 1: 0, 2: 1}

    # An unpromoted winner stays at its own level
    plain = _workspace(thalamic_gate_enabled=False)
    state = await plain.process_consciousness_cycle(
        [_candidate("a quiet percept", activation=0.5, emotional=0.0)]
    )
    assert state.dominant_hierarchy_level == 0
    assert state.level_winner_counts == {0: 1, 1: 0, 2: 0}


@pytest.mark.asyncio
async def test_admitted_level_survives_persistence():
    workspace = _workspace(thalamic_gate_enabled=False)
    await workspace.process_consciousness_cycle(
        [_candidate("a vivid percept", activation=0.9, emotional=0.5)]
    )
    saved = workspace.to_state_dict()
    saved["histories"] = json.loads(json.dumps(saved["histories"]))
    restored = _workspace(thalamic_gate_enabled=False)
    restored.from_state_dict(saved)
    assert [c.admitted_level for c in restored.stream_of_consciousness] == [1]


@pytest.mark.asyncio
async def test_prediction_error_rank_none_until_warm():
    workspace = _workspace()
    state = await workspace.process_consciousness_cycle([_candidate("percept")])
    assert state.prediction_error_rank is None

    baseline = workspace.active_inference.BASELINE_MIN_SAMPLES
    workspace.active_inference.prediction_errors.extend([0.5] * baseline)
    state = await workspace.process_consciousness_cycle([_candidate("percept two")])
    assert state.prediction_error_rank is not None
    assert 0.0 <= state.prediction_error_rank <= 1.0


class _FakePhiTracker:
    def __init__(self, phi):
        self.phi = phi
        self.recorded_states = []
        self.recorded_phi = []

    def record_module_states(self, state):
        self.recorded_states.append(state)

    @property
    def latest_phi(self):
        return self.phi

    @property
    def warming_up(self):
        return self.phi is None

    def record_phi(self, value):
        self.recorded_phi.append(value)

    def normalize_phi(self, value):
        return 1.0


@pytest.mark.asyncio
async def test_phi_tracker_fills_phi_slots():
    tracker = _FakePhiTracker(0.42)
    workspace = _workspace(phi_tracker=tracker)
    state = await workspace.process_consciousness_cycle([_candidate("percept")])
    assert tracker.recorded_states == [state]
    assert tracker.recorded_phi == [0.42]
    assert state.architectural_phi == 0.42
    assert state.phi_warming_up is False
    # The tracker's normalizer feeds the IIT to GWT interaction
    phi_bias = workspace.theory_interactions.modulations.phi_competition_bias
    assert phi_bias == pytest.approx(0.7 * 0.10)

    warming = _FakePhiTracker(None)
    workspace = _workspace(phi_tracker=warming)
    state = await workspace.process_consciousness_cycle([_candidate("percept")])
    assert state.architectural_phi is None
    assert state.phi_warming_up is True
    assert warming.recorded_phi == []

    bare = _workspace()
    state = await bare.process_consciousness_cycle([_candidate("percept")])
    assert state.architectural_phi is None
    assert state.phi_warming_up is None


# ============================================================================
# Persistence
# ============================================================================


def _json_round_trip(state):
    """Scalars and histories through strict JSON, arrays as numpy arrays."""
    return {
        "scalars": json.loads(json.dumps(state["scalars"], allow_nan=False)),
        "arrays": {k: np.asarray(v) for k, v in state["arrays"].items()},
        "histories": json.loads(json.dumps(state["histories"], allow_nan=False)),
    }


async def _run(workspace, cycles):
    for i in range(cycles):
        candidates = _layered_candidates(tag=f"c{i}-")[:7]
        candidates.append(
            _candidate(
                "safety check",
                source=WorkspaceCandidateSource.SAFETY,
                activation=0.4,
                content_type="alert",
            )
        )
        await workspace.process_consciousness_cycle(candidates)
    return workspace


@pytest.mark.asyncio
async def test_state_dict_round_trip():
    source = await _run(_workspace(), 4)
    state = source.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    # Hub contract: histories map names to lists. Sub-module tiers are
    # flattened with a "<name>." prefix rather than nested under the name, so
    # the workspace adds no nesting of its own; a non-list value can only be
    # one the sub-module itself exports as a non-list (the composite beautiful
    # loop and Damasio modules nest their components).
    submodules = source._persisted_submodules()
    for name in submodules:
        assert name not in state["scalars"]
        assert name not in state["histories"]
        assert any(
            key.startswith(f"{name}.") for tier in state.values() for key in tier
        )
    for key, value in state["histories"].items():
        if isinstance(value, list):
            continue
        name, _, sub_key = key.partition(".")
        assert name in submodules, key
        own = submodules[name].to_state_dict()["histories"][sub_key]
        assert not isinstance(own, list), key
    for name in ("active_inference", "metacognition", "theory_interactions"):
        assert all(
            isinstance(v, list)
            for k, v in state["histories"].items()
            if k.startswith(f"{name}.")
        )
    state = _json_round_trip(state)

    restored = _workspace()
    restored.from_state_dict(state)

    assert restored.cycle_count == source.cycle_count == 4
    assert restored._simulated_time_ms == source._simulated_time_ms
    assert restored._last_ignition_time_ms == source._last_ignition_time_ms
    assert restored.bottleneck._prior_prediction_error == pytest.approx(
        source.bottleneck._prior_prediction_error
    )
    assert restored.bottleneck.cross_theory_bias == pytest.approx(
        source.bottleneck.cross_theory_bias
    )
    for counter in ("total_broadcasts", "successful_broadcasts", "failed_broadcasts"):
        assert getattr(restored.broadcaster, counter) == getattr(
            source.broadcaster, counter
        )

    # Typed histories
    assert len(restored.stream_of_consciousness) == len(source.stream_of_consciousness)
    for item, original in zip(
        restored.stream_of_consciousness, source.stream_of_consciousness
    ):
        assert isinstance(item, WorkspaceContent)
        assert isinstance(item.candidate, WorkspaceCandidate)
        assert isinstance(item.candidate.source, WorkspaceCandidateSource)
        assert item.id == original.id
        assert item.candidate.summary == original.candidate.summary
        assert item.candidate.hierarchy_level == original.candidate.hierarchy_level
    assert all(
        isinstance(c, WorkspaceCandidate) for c in restored.unconscious_buffer
    ) and len(restored.unconscious_buffer) == len(source.unconscious_buffer)
    history = restored.bottleneck.competition_history
    assert all(isinstance(e, CompetitionEvent) for e in history)
    assert len(history) == 4
    assert history[-1].metadata["level_candidates"] == (
        source.bottleneck.competition_history[-1].metadata["level_candidates"]
    )
    assert all(
        isinstance(e, BroadcastEvent) for e in restored.broadcaster.broadcast_log
    )
    events = restored.ignition_detector.ignition_events
    assert len(events) == len(source.ignition_detector.ignition_events) > 0
    assert all(isinstance(e, IgnitionEvent) for e in events)
    assert all(isinstance(e.content, WorkspaceContent) for e in events)
    assert (
        restored.bottleneck._recent_content_hashes
        == source.bottleneck._recent_content_hashes
    )
    if source.current_conscious_content is not None:
        assert restored.current_conscious_content.id == (
            source.current_conscious_content.id
        )

    # Sub-modules restored through their own contracts
    assert (
        restored.active_inference.total_inferences
        == source.active_inference.total_inferences
    )
    assert (
        restored.metacognition.total_hots_generated
        == source.metacognition.total_hots_generated
    )
    assert restored.beautiful_loop.cycle_count == source.beautiful_loop.cycle_count
    assert restored.damasio.cycle_count == source.damasio.cycle_count
    assert (
        restored.theory_interactions.modulations.to_dict()
        == source.theory_interactions.modulations.to_dict()
    )
    assert (
        restored.bottleneck.thalamic_gate.stats.as_dict()
        == source.bottleneck.thalamic_gate.stats.as_dict()
    )


@pytest.mark.asyncio
async def test_statistics_after_restore_do_not_raise():
    source = await _run(_workspace(), 3)
    restored = _workspace()
    restored.from_state_dict(_json_round_trip(source.to_state_dict()))

    stats = restored.get_statistics()
    original = source.get_statistics()
    assert stats["cycle_count"] == 3
    assert stats["bottleneck"]["total_competitions"] == 3
    assert stats["bottleneck"]["source_distribution"] == (
        original["bottleneck"]["source_distribution"]
    )
    assert stats["ignition"]["total_ignitions"] == (
        original["ignition"]["total_ignitions"]
    )
    assert stats["broadcast"]["avg_latency_ms"] == pytest.approx(
        original["broadcast"]["avg_latency_ms"]
    )

    # The restored workspace keeps running
    state = await restored.process_consciousness_cycle([_candidate("after restore")])
    assert restored.cycle_count == 4
    assert isinstance(state, ConsciousnessState)


@pytest.mark.asyncio
async def test_sustain_check_after_restore():
    source = await _run(_workspace(), 2)
    restored = _workspace()
    restored.from_state_dict(_json_round_trip(source.to_state_dict()))

    detector = restored.ignition_detector
    ignited = detector.ignition_events[-1].content
    sustained = await detector.check_sustain(ignited)
    assert isinstance(sustained, bool)

    # Content the detector never saw falls back to the threshold
    fresh = WorkspaceContent(candidate=_candidate("fresh"), salience=0.9)
    assert await detector.check_sustain(fresh) is True

    # An event without content never matches and never raises
    detector.ignition_events.append(
        IgnitionEvent(
            content=None,
            pre_ignition_salience=0.5,
            post_ignition_salience=1.0,
            amplification_factor=2.0,
            ignition_time=0.0,
            expected_duration=1.0,
        )
    )
    assert await detector.check_sustain(fresh) is True
    detector.get_statistics()


@pytest.mark.asyncio
async def test_sustain_checked_survives_a_round_trip():
    """Whether a sustain check ran on an ignition event persists, so a
    restored detector reports the same sustain statistics."""
    source = await _run(_workspace(), 2)
    detector = source.ignition_detector
    events = detector.ignition_events
    assert len(events) >= 2
    assert all(e.sustain_checked is False for e in events)
    assert detector.get_statistics()["sustain_rate"] is None

    await detector.check_sustain(events[-1].content)
    assert events[-1].sustain_checked is True
    expected = detector.get_statistics()

    restored = _workspace()
    restored.from_state_dict(_json_round_trip(source.to_state_dict()))
    restored_events = restored.ignition_detector.ignition_events
    assert [e.sustain_checked for e in restored_events] == [
        e.sustain_checked for e in events
    ]
    stats = restored.ignition_detector.get_statistics()
    for key in ("sustain_rate", "sustain_checked_count", "sustained_count"):
        assert stats[key] == expected[key]

    # A record saved before the field existed restores as unchecked
    saved = _json_round_trip(source.to_state_dict())
    for record in saved["histories"]["ignition_detector.ignition_events"]:
        del record["sustain_checked"]
    older = _workspace()
    older.from_state_dict(saved)
    assert all(
        e.sustain_checked is False for e in older.ignition_detector.ignition_events
    )


class _ManualClock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_last_cycle_end_survives_a_round_trip():
    """The end time of the last cycle persists, so a workspace restored on
    the elapsed clock credits its downtime to the refractory period."""
    clock = _ManualClock(now=2000.0)
    source = _workspace(
        temporal_config=TemporalWorkspaceConfig(refractory_clock="elapsed"),
        clock=clock,
    )
    await source.process_consciousness_cycle([_candidate("before the save")])
    saved = _json_round_trip(source.to_state_dict())
    assert saved["scalars"]["last_cycle_end_s"] == 2000.0

    clock.now = 2060.0  # a minute of downtime
    restored = _workspace(
        temporal_config=TemporalWorkspaceConfig(refractory_clock="elapsed"),
        clock=clock,
    )
    restored.from_state_dict(saved)
    assert restored._last_cycle_end_s == 2000.0
    before = restored._simulated_time_ms
    state = await restored.process_consciousness_cycle([_candidate("after restore")])
    assert state.ignition_events == 1
    assert restored._simulated_time_ms == pytest.approx(before + 60000.0 + 250.0)

    # A state saved before the field existed restores with no end time
    del saved["scalars"]["last_cycle_end_s"]
    older = _workspace()
    older.from_state_dict(saved)
    assert older._last_cycle_end_s is None


@pytest.mark.asyncio
async def test_submitted_time_survives_a_round_trip():
    """A buffered candidate keeps the simulated time the bottleneck first saw
    it at through a save and restore, a record without the key restores
    None, and the restored bottleneck reads the restored simulated clock."""
    source = _workspace(thalamic_gate_enabled=False)
    faint = _candidate("a faint percept", activation=0.05, emotional=0.0)
    # First seen in cycle 1 (it wins on novelty), seen again in cycle 2 (it
    # loses and joins the unconscious buffer with its original stamp)
    await source.process_consciousness_cycle([faint])
    await source.process_consciousness_cycle([faint, _candidate("a vivid percept")])
    assert faint.submitted_at_s == 0.0
    # Dropped by the temporal competition in cycle 1, a loser in cycle 2:
    # the buffer holds it twice, both entries with the cycle 1 stamp
    assert {c.id for c in source.unconscious_buffer} == {faint.id}
    assert source._simulated_time_ms == pytest.approx(500.0)

    state = _json_round_trip(source.to_state_dict())
    restored = _workspace(thalamic_gate_enabled=False)
    restored.from_state_dict(state)
    assert len(restored.unconscious_buffer) == len(source.unconscious_buffer) >= 1
    assert {c.submitted_at_s for c in restored.unconscious_buffer} == {0.0}
    assert restored.bottleneck.simulated_time_s == pytest.approx(0.5)

    record = egw._candidate_to_record(faint)
    assert record["submitted_at_s"] == 0.0
    del record["submitted_at_s"]
    assert egw._candidate_from_record(record).submitted_at_s is None


def test_restore_skips_malformed_records_and_foreign_hashes():
    workspace = _workspace()
    workspace.from_state_dict(
        {
            "scalars": {"cycle_count": 2},
            "histories": {
                "stream_of_consciousness": [{"id": "x"}, "not a record", None],
                "bottleneck.competition_history": [{"timestamp": "bad"}],
                "bottleneck.recent_content_hashes": [12345, "abc123"],
                "ignition_detector.ignition_events": [{"content": None}],
            },
        }
    )
    assert workspace.cycle_count == 2
    assert all(
        isinstance(item, WorkspaceContent) for item in workspace.stream_of_consciousness
    )
    assert workspace.bottleneck._recent_content_hashes == ["abc123"]
    assert all(
        isinstance(e, CompetitionEvent)
        for e in workspace.bottleneck.competition_history
    )
    workspace.get_statistics()


@pytest.mark.asyncio
async def test_restore_from_empty_state_keeps_defaults():
    workspace = _workspace()
    workspace.from_state_dict({})
    assert workspace.cycle_count == 0
    await workspace.process_consciousness_cycle([_candidate("percept")])
    assert workspace.cycle_count == 1


def test_attention_precision_map_per_encoder():
    focus = SimpleNamespace(content_type="emotion", attention_strength=0.8)
    compact = _workspace()
    vector = compact._attention_precision(focus, ignited_count=1)
    assert len(vector) == 5
    assert vector[1] == pytest.approx(0.8)  # emotional_arousal
    assert vector[3] == pytest.approx(0.4)  # ignition_strength

    conversational = _workspace(
        active_inference_config=ActiveInferenceConfig(
            world_model="conversational_16x10x8"
        )
    )
    index = conversational.active_inference.observation_index
    vector = conversational._attention_precision(focus, ignited_count=1)
    assert len(vector) == 10
    assert vector[index["emote_intent"]] == pytest.approx(0.8)
    assert vector[index["high_engagement"]] == pytest.approx(0.4)
    social = SimpleNamespace(content_type="social", attention_strength=0.6)
    vector = conversational._attention_precision(social, ignited_count=0)
    assert vector[index["warm_tone"]] == pytest.approx(0.6)
    other = SimpleNamespace(content_type="thought", attention_strength=0.5)
    vector = conversational._attention_precision(other, ignited_count=0)
    assert vector[index["neutral_tone"]] == pytest.approx(0.5)
    assert vector.sum() == pytest.approx(0.5)
