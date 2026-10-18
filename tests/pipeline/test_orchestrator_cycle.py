"""Orchestrator cycle: signals in, measurements at their cadence, hooks called once."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.consciousness.perceptual_reality_monitor import PerceptualRealityMonitor
from mtc.consciousness.phi_tracker import PhiTracker
from mtc.consciousness.prospective_memory import ProspectiveSimulator
from mtc.core.config import Settings
from mtc.core.narration import NarrationFlags
from mtc.pipeline.consciousness_orchestrator import (
    CONSUMED_MODULATIONS,
    ConsciousnessOrchestrator,
    OrchestratorConfig,
)
from mtc.pipeline.context_types import RawConversationContext
from mtc.pipeline.cycle_augmenter import CycleAugmenter, CycleAugmenterConfig
from mtc.pipeline.speaker_trust import SpeakerTrust


class _RecordingWorkspace(EnhancedGlobalWorkspace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signal_calls = []

    async def process_consciousness_cycle(self, candidates, neural_signals=None, **kw):
        self.signal_calls.append(neural_signals)
        return await super().process_consciousness_cycle(
            candidates, neural_signals=neural_signals, **kw
        )


class _Neural:
    def __init__(self):
        self.latest_neural_signals = {
            "snn": np.linspace(0.1, 0.9, 16),
            "lsm": np.linspace(0.2, 0.8, 16),
        }
        self.latest_substrate_states = {"snn_state": {"metrics": {"synchrony": 0.4}}}
        # The experience's emotions are the caller's input to the substrate;
        # snn_result is what the SNN produced (the encoder path's shape).
        self.state = SimpleNamespace(
            current_experience=SimpleNamespace(
                emotions={"joy": 0.6, "sadness": 0.1, "trust": 0.4},
                snn_result={
                    "output_emotions": {"joy": 0.3, "sadness": 0.05, "fear": 0.1},
                    "firing_rates": {"output": [30.0, 5.0, 10.0]},
                    "output_unit": "hz",
                    "total_spikes": 45,
                },
                subconscious_state=None,
            )
        )
        self.published = []

    def publish_retrieved_memories(self, memories):
        self.published.append(list(memories))

    def get_binding_stats(self):
        return {"global_order_parameter": 0.5, "cross_coherence": 0.3}

    def get_binding_context(self):
        return "Binding context line."


class _RPT:
    def __init__(self):
        self.calls = []
        self.stats = []
        self.measurement_count = 0

    def set_oscillatory_binding_stats(self, stats):
        self.stats.append(stats)

    def measure_full(self, snn_state=None, lsm_state=None, workspace_state=None):
        self.calls.append((snn_state, lsm_state, workspace_state))
        self.measurement_count += 1
        return SimpleNamespace(
            global_recurrence_score=0.42, local_recurrence_score=0.21
        )


class _CausalEmergence:
    def __init__(self, warming=False):
        self.calls = []
        self.warming = warming

    def analyze_modules(self, tracker):
        self.calls.append(tracker)
        if self.warming:
            return None
        return SimpleNamespace(causal_emergence=0.33, warming_up=False)


def _build(workspace=None, augmenter_config=None, flags=None, **kwargs):
    cfg = kwargs.pop("settings", None) or Settings(DEBUG=False)
    workspace = workspace or EnhancedGlobalWorkspace(
        phi_tracker=PhiTracker(min_transitions=3)
    )
    augmenter = CycleAugmenter(
        augmenter_config or CycleAugmenterConfig(),
        workspace,
        pad_model=PADAffectModel(),
    )
    orchestrator = ConsciousnessOrchestrator(
        workspace,
        augmenter=augmenter,
        narration=flags or NarrationFlags(),
        speaker_trust=SpeakerTrust(settings=cfg),
        settings=cfg,
        **kwargs,
    )
    return orchestrator


def _raw(orchestrator, message="hello there", **fields):
    return RawConversationContext(
        message=message,
        identity=orchestrator.speaker_trust.verify("Alice", None),
        **fields,
    )


def _turns(orchestrator, n, **fields):
    async def run():
        out = []
        for i in range(n):
            out.append(
                await orchestrator.process(_raw(orchestrator, f"turn {i}", **fields))
            )
        return out

    return asyncio.run(run())


def test_cycle_receives_neural_signals():
    workspace = _RecordingWorkspace(phi_tracker=PhiTracker(min_transitions=3))
    neural = _Neural()
    orchestrator = _build(workspace=workspace, neural_orchestrator=neural)
    (context,) = _turns(orchestrator, 1)
    assert context.cycle_ok
    assert workspace.signal_calls[0] is neural.latest_neural_signals
    assert context.consciousness_state.integration_level > 0.0
    # SNN affect reached the PAD sources
    assert "snn_valence" in context.pad_sources_measured
    assert "snn_spike_rate" in context.pad_sources_measured
    # Without a neural orchestrator the cycle gets no signals, honestly
    bare = _RecordingWorkspace()
    plain = _build(workspace=bare)
    _turns(plain, 1)
    assert bare.signal_calls == [None]


def test_integration_level_unmeasured_without_neural_signals():
    from mtc.pipeline.message_persistence import build_turn_snapshot

    # No neural orchestrator: nothing produces the signals
    orchestrator = _build()
    (context,) = _turns(orchestrator, 1)
    assert context.cycle_ok
    assert context.consciousness_state.integration_level is None
    section = context.theory_sections.global_workspace_context
    assert "integration level unmeasured" in section
    assert "integration level 0" not in section
    assert context.absent_signals["integration_level"] == "no_producer"
    assert context.extras["momentary_consciousness"] is None
    snapshot = build_turn_snapshot(context, provider_id="mock")
    assert snapshot["computed_state"]["integration_level"] is None
    assert snapshot["absent_signals"]["integration_level"] == "no_producer"

    # A neural orchestrator whose latest signals are missing: no input
    neural = _Neural()
    neural.latest_neural_signals = None
    (context,) = _turns(_build(neural_orchestrator=neural), 1)
    assert context.consciousness_state.integration_level is None
    assert context.absent_signals["integration_level"] == "no_input"
    assert context.extras["momentary_consciousness"] is None

    # Two usable signals: a measured level, not named absent
    (context,) = _turns(_build(neural_orchestrator=_Neural()), 1)
    level = context.consciousness_state.integration_level
    assert level is not None and level > 0.0
    assert "integration_level" not in context.absent_signals
    assert f"integration level {level:.2f}" in (
        context.theory_sections.global_workspace_context
    )


def test_integration_unmeasured_without_substrate():
    from mtc.pipeline.message_persistence import build_turn_snapshot

    orchestrator = _build()
    for context in _turns(orchestrator, 3):
        assert "integration level unmeasured" in (
            context.theory_sections.global_workspace_context
        )
        snapshot = build_turn_snapshot(context)
        assert snapshot["computed_state"]["integration_level"] is None


def test_newly_optional_state_fields_are_named_when_absent():
    """The state fields that became Optional (they have a non-None default or
    none at all, so the default scan does not see them) are named with a
    reason whenever the cycle leaves them None."""
    orchestrator = _build()
    first, second = _turns(orchestrator, 2)
    expected_first = {
        "integration_level": "no_producer",
        "broadcast_coverage": "no_input",
        "binding_quality": "no_input",
        "protoself_stability": "warming_up",
        "feeling_of_knowing": "warming_up",
        "autobiographic_continuity": "no_input",
    }
    from mtc.pipeline.consciousness_orchestrator import NAMED_STATE_FIELDS

    assert set(expected_first) <= set(NAMED_STATE_FIELDS)
    state = first.consciousness_state
    for name, reason in expected_first.items():
        assert getattr(state, name) is None, name
        assert first.absent_signals[name] == reason, name
    # After the first cycle every None field is named without warming up; a
    # measured one is not named
    state = second.consciousness_state
    for name in expected_first:
        value = getattr(state, name)
        if value is None:
            assert second.absent_signals[name] in ("no_input", "no_producer"), name
        else:
            assert name not in second.absent_signals, name
    # After the first cycle the body state has a previous reading
    assert second.consciousness_state.protoself_stability is not None
    assert "protoself_stability" not in second.absent_signals


class _FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now


def _minute_apart(refractory_clock):
    from mtc.consciousness.enhanced_global_workspace import TemporalWorkspaceConfig

    clock = _FakeClock()
    workspace = EnhancedGlobalWorkspace(
        phi_tracker=PhiTracker(min_transitions=3),
        temporal_config=TemporalWorkspaceConfig(refractory_clock=refractory_clock),
        clock=clock,
    )
    orchestrator = _build(
        workspace=workspace, augmenter_config=CycleAugmenterConfig.all_off()
    )

    async def run():
        out = []
        for i in range(3):
            raw = _raw(orchestrator, f"What should we plant in bed {i}?")
            out.append(await orchestrator.process(raw))
            clock.now += 60.0
        return out

    return asyncio.run(run())


def test_turns_a_minute_apart_all_focus_on_the_message():
    contexts = _minute_apart("elapsed")
    assert [c.winner_is_message for c in contexts] == [True, True, True]
    for context in contexts:
        assert context.cycle_ok
        assert context.consciousness_state.ignition_events > 0
        assert context.prompt_context.splitlines()[0] == (
            "The workspace is focused on the incoming message."
        )
        assert context.workspace_winner == "incoming message"
    # The simulated per-cycle clock keeps the refractory cadence whatever the
    # real time between turns
    simulated = _minute_apart("per_cycle")
    assert [c.winner_is_message for c in simulated] == [True, False, False]


def test_phi_recorded_once_per_cycle():
    tracker = PhiTracker(min_transitions=3)
    calls = []
    real_record = tracker.record_module_states

    def spy(state):
        calls.append(state)
        return real_record(state)

    tracker.record_module_states = spy
    workspace = EnhancedGlobalWorkspace(phi_tracker=tracker)
    orchestrator = _build(workspace=workspace)
    contexts = _turns(orchestrator, 4)
    assert len(calls) == 4
    assert tracker.transitions_recorded == 3
    assert contexts[-1].consciousness_state.architectural_phi is not None
    assert contexts[0].absent_signals["architectural_phi"] == "warming_up"
    assert "architectural_phi" not in contexts[-1].absent_signals
    # The orchestrator never records on its own; the idle sampler does
    assert orchestrator.sample_phi_state() is True
    assert len(calls) == 5


def test_rpt_every_n_cycles():
    rpt = _RPT()
    neural = _Neural()
    orchestrator = _build(
        rpt_measurement=rpt,
        neural_orchestrator=neural,
        config=OrchestratorConfig(rpt_every_n_cycles=2),
    )
    contexts = _turns(orchestrator, 4)
    assert len(rpt.calls) == 2
    snn_state, lsm_state, workspace_state = rpt.calls[0]
    assert snn_state == neural.latest_substrate_states["snn_state"]
    assert lsm_state is None
    assert "cycle_count" in workspace_state
    assert len(rpt.stats) == 4  # binding stats refreshed every cycle
    states = [c.consciousness_state for c in contexts]
    assert states[0].rpt_deep_recurrence is None
    assert contexts[0].absent_signals["rpt_deep_recurrence"] == "cadence_not_reached"
    assert states[1].rpt_deep_recurrence == pytest.approx(0.42)
    assert states[1].rpt_superficial_recurrence == pytest.approx(0.21)
    assert "rpt_deep_recurrence" not in contexts[1].absent_signals
    assert states[3].rpt_deep_recurrence == pytest.approx(0.42)
    without = _build()
    (context,) = _turns(without, 1)
    assert context.absent_signals["rpt_deep_recurrence"] == "no_producer"


def test_causal_emergence_every_n_cycles():
    analyzer = _CausalEmergence()
    orchestrator = _build(
        causal_emergence=analyzer,
        config=OrchestratorConfig(causal_emergence_every_n_cycles=3),
    )
    contexts = _turns(orchestrator, 3)
    assert len(analyzer.calls) == 1
    assert analyzer.calls[0] is orchestrator.workspace.phi_tracker
    assert contexts[0].absent_signals["causal_emergence"] == "cadence_not_reached"
    final = contexts[2].consciousness_state
    assert final.causal_emergence == pytest.approx(0.33)
    assert final.causal_emergence_warming_up is False

    warming = _CausalEmergence(warming=True)
    cold = _build(
        causal_emergence=warming,
        config=OrchestratorConfig(causal_emergence_every_n_cycles=1),
    )
    (context,) = _turns(cold, 1)
    assert context.consciousness_state.causal_emergence is None
    assert context.consciousness_state.causal_emergence_warming_up is True
    assert context.absent_signals["causal_emergence"] == "warming_up"


def test_pad_on_state_every_cycle():
    orchestrator = _build()
    contexts = _turns(orchestrator, 3)
    for index, context in enumerate(contexts):
        state = context.consciousness_state
        assert state.pad_pleasure is not None
        assert 0.0 <= state.pad_arousal <= 1.0
        assert state.pad_label
        # The first body update has no previous body state, so its arousal is
        # not a measured input; every later one is
        if index == 0:
            assert "protoself_arousal" not in context.pad_sources_measured
        else:
            assert "protoself_arousal" in context.pad_sources_measured
        assert not any(name.startswith("pad_") for name in context.absent_signals)
    assert orchestrator.augmenter.pad_model.history_length == 3
    # Pinned off: nothing recorded and the absence is named as disabled
    off = _build(augmenter_config=CycleAugmenterConfig.all_off())
    (context,) = _turns(off, 1)
    assert context.consciousness_state.pad_arousal is None
    assert context.absent_signals["pad_arousal"] == "disabled"
    assert context.pad_sources_measured == ()


def test_prm_external_and_internal_signals():
    monitor = PerceptualRealityMonitor()
    orchestrator = _build(reality_monitor=monitor)
    thoughts = [{"content": f"thought {i}"} for i in range(5)]
    (context,) = _turns(orchestrator, 1, ctm_thoughts=thoughts)
    kinds = [(s.source, s.kind) for s in monitor._signals]
    assert kinds.count(("user_message", "external")) == 1
    assert kinds.count(("thought_generator", "internal")) == 3
    assert len(kinds) == 4
    # Both kinds were seen but the pipeline supplies no attribution
    # confidence, so nothing measures the discrimination: no prior stands in
    assert monitor.get_statistics()["default_attribution_fraction"] == 1.0
    assert monitor.discrimination() is None
    assert context.consciousness_state.prm_discrimination is None
    assert context.absent_signals["prm_discrimination"] == "no_input"
    # Only the message: nothing internal yet, so the signal is warming up
    fresh = _build(reality_monitor=PerceptualRealityMonitor())
    (context,) = _turns(fresh, 1)
    assert context.consciousness_state.prm_discrimination is None
    assert context.absent_signals["prm_discrimination"] == "warming_up"


def test_theory_interactions_not_updated_twice():
    orchestrator = _build()
    _turns(orchestrator, 3)
    manager = orchestrator.workspace.theory_interactions
    assert (
        manager.get_status()["cycle_count"] == orchestrator.workspace.cycle_count == 3
    )
    status = orchestrator.status()["theory_modulations"]
    assert status["enabled"] is True
    names = [m["name"] for m in status["modulations"]]
    assert names == [
        "somatic_salience_bias",
        "phi_competition_bias",
        "metacognitive_precision_bias",
        "free_energy_arousal",
        "recurrence_binding_boost",
        "epistemic_attention_expansion",
    ]
    consumed = {m["name"] for m in status["modulations"] if m["consumed"]}
    assert consumed == set(CONSUMED_MODULATIONS)
    disabled = _build(
        workspace=EnhancedGlobalWorkspace(theory_interactions_enabled=False)
    )
    _turns(disabled, 1)
    assert disabled.status()["theory_modulations"] == {
        "enabled": False,
        "modulations": [],
    }


def test_goal_source_assigned_and_read_each_cycle():
    reads = []

    class _Goals:
        def get_top_goals(self, n):
            reads.append(n)
            return [
                SimpleNamespace(description=f"goal {len(reads)}"),
                SimpleNamespace(description=""),
            ]

    class _Prospective:
        def __init__(self):
            self.goals = []

        def set_goals(self, goals):
            self.goals.append(list(goals))

        def update_topics(self, text):
            pass

        def maybe_generate(self, context):
            return []

    simulator = _Prospective()
    orchestrator = _build(goal_source=_Goals(), prospective=simulator)
    _turns(orchestrator, 2)
    assert reads == [5, 5]
    assert simulator.goals == [["goal 1"], ["goal 2"]]


def test_one_prospective_simulator_shared_with_the_augmenter():
    class _Goals:
        def get_top_goals(self, n):
            return [SimpleNamespace(description="tidy the notes")]

    workspace = EnhancedGlobalWorkspace(phi_tracker=PhiTracker(min_transitions=3))
    augmenter = CycleAugmenter(
        CycleAugmenterConfig(), workspace, pad_model=PADAffectModel()
    )
    assert augmenter.prospective is None
    simulator = ProspectiveSimulator()
    cfg = Settings(DEBUG=False)
    orchestrator = ConsciousnessOrchestrator(
        workspace,
        augmenter=augmenter,
        narration=NarrationFlags(),
        speaker_trust=SpeakerTrust(settings=cfg),
        settings=cfg,
        prospective=simulator,
        goal_source=_Goals(),
    )
    # One simulator: the goals the orchestrator sets and the generation the
    # augmenter runs meet in the same object
    assert orchestrator.prospective is simulator
    assert augmenter.prospective is simulator
    _turns(orchestrator, 2, recent_messages=[{"content": "tulips blooming nicely"}])
    assert augmenter.prospective is simulator
    histories = simulator.to_state_dict()["histories"]
    assert histories["active_goals"] == ["tidy the notes"]
    assert "tulips" in histories["recent_topics"]
    assert histories["simulation_history"]  # the augmenter generated through it

    # Without one, the orchestrator reads the augmenter's own, lazily built
    plain = _build(goal_source=_Goals())
    assert plain.prospective is None
    _turns(plain, 2)
    assert plain.prospective is plain.augmenter.prospective
    assert plain.prospective is not None
    assert plain.prospective.to_state_dict()["histories"]["active_goals"] == [
        "tidy the notes"
    ]

    # Two different simulators can never both be in use
    with pytest.raises(ValueError, match="prospective"):
        ConsciousnessOrchestrator(
            workspace,
            augmenter=CycleAugmenter(
                CycleAugmenterConfig(),
                workspace,
                prospective=ProspectiveSimulator(),
            ),
            narration=NarrationFlags(),
            speaker_trust=SpeakerTrust(settings=cfg),
            settings=cfg,
            prospective=ProspectiveSimulator(),
        )


def test_attention_observer_called_when_enabled():
    class _Observer:
        def __init__(self, enabled):
            self.enabled = enabled
            self.seen = []

        async def observe(self, focus):
            self.seen.append(focus)
            return None

    on = _Observer(True)
    orchestrator = _build(attention_observer=on)
    _turns(orchestrator, 2)
    assert len(on.seen) == 2
    assert on.seen[-1] is orchestrator.workspace.attention_schema.schema.current_focus
    off = _Observer(False)
    quiet = _build(attention_observer=off)
    _turns(quiet, 2)
    assert off.seen == []


def test_state_manager_on_cycle_complete_called():
    class _Manager:
        def __init__(self):
            self.calls = 0

        def on_cycle_complete(self):
            self.calls += 1
            return {}

    manager = _Manager()
    orchestrator = _build(state_manager=manager)
    _turns(orchestrator, 3)
    assert manager.calls == 3


def test_developmental_modulator_only_when_enabled():
    class _Modulator:
        def __init__(self):
            self.calls = []

        def apply_to_workspace(self, workspace, stage, pad_model=None, force=False):
            self.calls.append((stage, pad_model))

    disabled = _Modulator()
    orchestrator = _build(
        developmental=disabled,
        settings=Settings(DEBUG=False, COMPANION_TYPE="research_subject"),
    )
    orchestrator.set_developmental_stage(3)
    _turns(orchestrator, 2)
    assert disabled.calls == []

    enabled = _Modulator()
    live = _build(
        developmental=enabled,
        settings=Settings(DEBUG=False, DEVELOPMENTAL_STAGE_ENABLED=True),
    )
    _turns(live, 1)
    assert enabled.calls == []  # no stage known yet
    live.set_developmental_stage(4)
    _turns(live, 2)
    assert enabled.calls == [(4.0, live.pad_model)]  # once per stage change
    live.set_developmental_stage(5)
    _turns(live, 1)
    assert [stage for stage, _ in enabled.calls] == [4.0, 5.0]
    assert live.status()["developmental_stage"] == 5.0


def _affect(experience):
    neural = _Neural()
    neural.state.current_experience = experience
    return _build(neural_orchestrator=neural)._neural_affect()


def test_neural_affect_reads_substrate_output_not_input_emotions():
    # Input emotions alone: nothing was measured, so nothing is reported
    assert _affect(SimpleNamespace(emotions={"joy": 0.9, "trust": 0.4})) is None
    assert _affect(None) is None
    # The encoder path: valence is the SNN's decoded output, not the input
    encoder = SimpleNamespace(
        emotions={"joy": 0.9, "sadness": 0.0, "trust": 0.4},
        snn_result={
            "output_emotions": {"joy": 0.3, "sadness": 0.05, "fear": 0.1},
            "firing_rates": {"output": [30.0, 5.0, 10.0]},
            "output_unit": "hz",
        },
        subconscious_state=None,
    )
    affect = _affect(encoder)
    assert affect["snn_valence"] == pytest.approx(0.25)
    assert affect["snn_spike_rate"] == pytest.approx((0.3 + 0.05 + 0.1) / 3)
    # The production path: spike fractions, with valence from a trained readout
    production = SimpleNamespace(
        emotions={"joy": 0.9},
        snn_result={
            "firing_rates": {"output": [0.2, 0.4]},
            "output_unit": "spike_fraction",
            "metrics": {},
        },
        subconscious_state=SimpleNamespace(
            readout_trained=True, intuitive_valence=-0.3, emotional_coherence=0.5
        ),
    )
    affect = _affect(production)
    assert affect == {
        "snn_valence": pytest.approx(-0.3),
        "snn_spike_rate": pytest.approx(0.3),
    }
    # An untrained readout measures no valence; an unknown unit measures no rate
    untrained = SimpleNamespace(
        emotions={"joy": 0.9},
        snn_result={"firing_rates": {"output": [0.2, 0.4]}},
        subconscious_state=SimpleNamespace(
            readout_trained=False, intuitive_valence=0.7
        ),
    )
    assert _affect(untrained) is None
    # Through a turn: only measured sources are named, never the trust constant
    neural = _Neural()
    neural.state.current_experience = SimpleNamespace(
        emotions={"joy": 0.6, "sadness": 0.1, "trust": 0.4}
    )
    (context,) = _turns(_build(neural_orchestrator=neural), 1)
    assert "snn_valence" not in context.pad_sources_measured
    assert "snn_spike_rate" not in context.pad_sources_measured


def test_pad_model_follows_the_augmenters_lazy_build():
    workspace = EnhancedGlobalWorkspace(phi_tracker=PhiTracker(min_transitions=3))
    augmenter = CycleAugmenter(CycleAugmenterConfig(), workspace, pad_model=None)
    cfg = Settings(DEBUG=False)
    orchestrator = ConsciousnessOrchestrator(
        workspace,
        augmenter=augmenter,
        narration=NarrationFlags(affect_line=True),
        speaker_trust=SpeakerTrust(settings=cfg),
        settings=cfg,
    )
    assert orchestrator.pad_model is None  # not yet built, and not pretended
    (context,) = _turns(orchestrator, 1)
    assert isinstance(augmenter.pad_model, PADAffectModel)
    assert orchestrator.pad_model is augmenter.pad_model
    # The affect line reads the model PAD was computed through
    assert augmenter.pad_model.generate_context() in context.prompt_context
    # An injected model is kept, and it is the one PAD is computed through
    own = PADAffectModel()
    injected = ConsciousnessOrchestrator(
        workspace,
        augmenter=CycleAugmenter(CycleAugmenterConfig(), workspace, pad_model=None),
        narration=NarrationFlags(),
        speaker_trust=SpeakerTrust(settings=cfg),
        settings=cfg,
        pad_model=own,
    )
    (context,) = _turns(injected, 1)
    assert injected.pad_model is own
    assert injected.augmenter.pad_model is own
    assert own.history_length == 1
    assert own.get_latest().arousal == pytest.approx(
        context.consciousness_state.pad_arousal
    )
    # Two different PAD models can never both be in use
    with pytest.raises(ValueError, match="PAD model"):
        ConsciousnessOrchestrator(
            workspace,
            augmenter=CycleAugmenter(
                CycleAugmenterConfig(), workspace, pad_model=PADAffectModel()
            ),
            narration=NarrationFlags(),
            speaker_trust=SpeakerTrust(settings=cfg),
            settings=cfg,
            pad_model=PADAffectModel(),
        )
