"""Cycle augmenter: the per-cycle additions, each behind its switch."""

import asyncio
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.pad_affect import PADAffectModel, PADSources
from mtc.pipeline.cycle_augmenter import (
    GATE_AROUSAL_SOURCES,
    AugmentRecord,
    CycleAugmenter,
    CycleAugmenterConfig,
    CycleInputs,
    extract_pad_sources,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


class _Drives:
    def __init__(self, urgent):
        self.urgent = urgent
        self.thresholds = []
        self.drives = {
            name: SimpleNamespace(get_deviation=lambda d=0.2: d) for name, _ in urgent
        }

    def get_urgent_drives(self, threshold=0.3):
        self.thresholds.append(threshold)
        return [(n, u) for n, u in self.urgent if u > threshold]


class _Workspace:
    def __init__(self, urgent=()):
        self.active_inference = SimpleNamespace(
            homeostatic_drives=_Drives(list(urgent))
        )
        self.arousals = []

    def set_arousal(self, value):
        self.arousals.append(value)


def _state(**overrides):
    values = dict(
        damasio_state={
            "core": {"somatic_valence": 0.25},
            "protoself": {"primordial_feelings": {"arousal": 0.42}},
        },
        ignition_events=2,
        inference_result=object(),
        prediction_error=1.7,
        active_inference_state=SimpleNamespace(meta_confidence=0.6),
        attention_schema=SimpleNamespace(voluntary_ratio=0.75),
        beautiful_loop_moment=object(),
        epistemic_depth=3,
        pad_pleasure=None,
        pad_arousal=None,
        pad_dominance=None,
        pad_label=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _cand(i, act=0.5, emo=0.0, content_type="perception"):
    return WorkspaceCandidate(
        content=f"stimulus_{i}",
        content_type=content_type,
        summary=f"stimulus {i}",
        source=WorkspaceCandidateSource.SENSORY,
        activation_level=act,
        emotional_salience=emo,
        timestamp=NOW.timestamp(),
    )


def test_drive_candidates_from_urgent_drives():
    workspace = _Workspace(urgent=[("curiosity", 0.8), ("safety", 0.55), ("x", 0.2)])
    config = CycleAugmenterConfig.from_overrides(
        {"drive_candidates": True, "drive_urgency_threshold": 0.5}, pinned_off=True
    )
    augmenter = CycleAugmenter(config, workspace, clock=lambda: NOW)
    out = augmenter.before_cycle([_cand(0)])
    drives = [c for c in out if c.source is WorkspaceCandidateSource.HOMEOSTATIC]
    assert workspace.active_inference.homeostatic_drives.thresholds == [0.5]
    assert [c.content["drive"] for c in drives] == ["curiosity", "safety"]
    assert [c.activation_level for c in drives] == [
        pytest.approx(0.8),
        pytest.approx(0.55),
    ]
    assert all(c.content_type == "homeostatic_drive" for c in drives)
    assert all(c.hierarchy_level == 2 for c in drives)
    record = augmenter.after_cycle(_state())
    assert record.drive_candidates == 2


def test_prospective_pending_injected_next_cycle():
    pending = _cand(99, content_type="prospective_simulation")
    pending.source = WorkspaceCandidateSource.INTERNAL
    calls = []

    class _Prospective:
        def update_topics(self, text):
            calls.append(("topics", text))

        def maybe_generate(self, context):
            calls.append(("generate", list(context.goals), context.arousal))
            return [pending]

    config = CycleAugmenterConfig.from_overrides(
        {"prospective_candidates": True}, pinned_off=True
    )
    augmenter = CycleAugmenter(
        config, _Workspace(), prospective=_Prospective(), clock=lambda: NOW
    )
    inputs = CycleInputs(
        recent_messages=[{"content": "planning the garden tomorrow"}],
        goals=["learn more about soil"],
        now=NOW,
    )
    first = augmenter.before_cycle([_cand(0)], inputs)
    assert pending not in first
    augmenter.after_cycle(_state(pad_arousal=0.3), inputs)
    assert calls == [
        ("topics", "planning the garden tomorrow"),
        ("generate", ["learn more about soil"], 0.3),
    ]
    second = augmenter.before_cycle([_cand(1)], inputs)
    assert pending in second
    record = augmenter.after_cycle(_state(), inputs)
    assert record.prospective_injected == 1
    third = augmenter.before_cycle([_cand(2)], inputs)
    assert third.count(pending) == 1  # generated again after the second cycle


def test_crossmodal_binding_runs_when_enabled():
    bound = _cand(50, content_type="crossmodal_bound")
    seen = []

    class _Binder:
        def bind(self, candidates, now=None):
            seen.append((len(candidates), now))
            return list(candidates) + [bound]

    on = CycleAugmenter(
        CycleAugmenterConfig.from_overrides(
            {"crossmodal_binding": True}, pinned_off=True
        ),
        _Workspace(),
        binder=_Binder(),
        clock=lambda: NOW,
    )
    out = on.before_cycle([_cand(0), _cand(1)])
    assert bound in out
    assert seen == [(2, NOW.timestamp())]
    assert on.after_cycle(_state()).bound == 1

    off = CycleAugmenter(CycleAugmenterConfig.all_off(), _Workspace(), binder=_Binder())
    assert bound not in off.before_cycle([_cand(0), _cand(1)])
    assert len(seen) == 1


def test_candidate_cap_keeps_highest():
    candidates = [_cand(i, act=0.1 + 0.03 * i, emo=0.02 * (24 - i)) for i in range(25)]
    augmenter = CycleAugmenter(
        CycleAugmenterConfig.from_overrides({"candidate_cap": 20}, pinned_off=True),
        _Workspace(),
    )
    out = augmenter.before_cycle(candidates)
    assert len(out) == 20
    ranked = sorted(
        candidates,
        key=lambda c: c.activation_level + c.emotional_salience,
        reverse=True,
    )
    assert {c.id for c in out} == {c.id for c in ranked[:20]}
    assert augmenter.after_cycle(_state()).capped == 5
    # 0 means no cap
    uncapped = CycleAugmenter(CycleAugmenterConfig.all_off(), _Workspace())
    assert len(uncapped.before_cycle(candidates)) == 25


def test_gate_arousal_protoself_calls_set_arousal():
    workspace = _Workspace()
    augmenter = CycleAugmenter(
        CycleAugmenterConfig.from_overrides(
            {"gate_arousal": "protoself"}, pinned_off=True
        ),
        workspace,
    )
    augmenter.before_cycle([])
    record = augmenter.after_cycle(_state())
    assert workspace.arousals == [pytest.approx(0.42)]
    assert record.gate_arousal_value == pytest.approx(0.42)
    # No protoself reading: the gate is left alone
    augmenter.before_cycle([])
    record = augmenter.after_cycle(_state(damasio_state={}))
    assert len(workspace.arousals) == 1
    assert record.gate_arousal_value is None


def test_gate_arousal_pad_records_on_state_for_next_cycle():
    workspace = EnhancedGlobalWorkspace(thalamic_gate_enabled=True)
    config = CycleAugmenterConfig.from_overrides(
        {"gate_arousal": "pad", "pad_from_state": True}, pinned_off=True
    )
    augmenter = CycleAugmenter(config, workspace, pad_model=PADAffectModel())

    async def two_cycles():
        first = await workspace.process_consciousness_cycle(
            augmenter.before_cycle([_cand(i) for i in range(5)])
        )
        record = augmenter.after_cycle(first)
        second = await workspace.process_consciousness_cycle(
            augmenter.before_cycle([_cand(i + 5) for i in range(5)])
        )
        return first, record, second

    first, record, second = asyncio.run(two_cycles())
    assert first.pad_arousal is not None
    assert record.gate_arousal_value == pytest.approx(first.pad_arousal)
    assert record.pad_state is not None
    assert first.arousal_source == "default"
    assert second.arousal_source == "pad"
    assert workspace.bottleneck.arousal == pytest.approx(first.pad_arousal)


def test_gate_arousal_workspace_leaves_gate_alone():
    workspace = _Workspace()
    augmenter = CycleAugmenter(CycleAugmenterConfig.all_off(), workspace)
    augmenter.before_cycle([])
    record = augmenter.after_cycle(_state())
    assert workspace.arousals == []
    assert record.gate_arousal_value is None
    assert GATE_AROUSAL_SOURCES == ("protoself", "pad", "workspace")
    with pytest.raises(ValueError, match="gate_arousal"):
        CycleAugmenterConfig(gate_arousal="camera")


def test_pad_sources_measured_named():
    workspace = _Workspace(urgent=[("curiosity", 0.8)])
    full, measured = extract_pad_sources(
        _state(), workspace, {"snn_valence": 0.5, "snn_spike_rate": 0.9}
    )
    assert set(measured) == {
        "somatic_valence",
        "homeostatic_satisfaction",
        "snn_valence",
        "snn_spike_rate",
        "protoself_arousal",
        "ignition_rate",
        "prediction_error",
        "agency_score",
        "voluntary_attention_ratio",
        "epistemic_depth_normalized",
    }
    assert full.prediction_error == pytest.approx(1.0)  # clamped
    assert full.ignition_rate == pytest.approx(0.4)
    assert full.epistemic_depth_normalized == pytest.approx(0.6)
    assert full.snn_valence == pytest.approx(0.5)
    assert full.snn_spike_rate == pytest.approx(0.9)
    assert full.homeostatic_satisfaction == pytest.approx(2 * 0.8 - 1)

    sparse_state = _state(
        damasio_state=None,
        inference_result=None,
        active_inference_state=None,
        attention_schema=None,
        beautiful_loop_moment=None,
    )
    sparse, measured = extract_pad_sources(sparse_state, None, None)
    assert measured == ("ignition_rate",)
    # The emotions a caller fed the substrate are its input, not its output:
    # input-shaped keys measure no SNN source.
    _, from_input = extract_pad_sources(
        sparse_state, None, {"joy": 0.6, "sadness": 0.1, "trust": 0.4}
    )
    assert from_input == ("ignition_rate",)
    defaults = PADSources()
    for name in (
        "somatic_valence",
        "homeostatic_satisfaction",
        "snn_valence",
        "snn_spike_rate",
        "protoself_arousal",
        "prediction_error",
        "agency_score",
        "voluntary_attention_ratio",
        "epistemic_depth_normalized",
    ):
        assert getattr(sparse, name) == getattr(defaults, name), name

    # The record names the measured inputs; an absent neural affect names none
    augmenter = CycleAugmenter(
        CycleAugmenterConfig.from_overrides({"pad_from_state": True}, pinned_off=True),
        workspace,
    )
    augmenter.before_cycle([])
    state = _state()
    record = augmenter.after_cycle(state)
    assert "snn_valence" not in record.pad_sources_measured
    assert "somatic_valence" in record.pad_sources_measured
    assert state.pad_label == record.pad_state.label
    assert augmenter.pad_model is not None


def test_unclassifiable_ratio_is_not_a_measured_pad_input():
    """A typed attention schema with no classifiable shift has no voluntary
    ratio, so the ratio is not named measured and PAD keeps its default."""
    from mtc.consciousness.attention_schema import AttentionSchemaModule

    schema = AttentionSchemaModule().schema
    assert schema.voluntary_ratio is None
    sources, measured = extract_pad_sources(_state(attention_schema=schema))
    assert "voluntary_attention_ratio" not in measured
    assert sources.voluntary_attention_ratio == PADSources().voluntary_attention_ratio
    # A classified ratio is a measured input
    _, measured = extract_pad_sources(
        _state(attention_schema=SimpleNamespace(voluntary_ratio=0.25))
    )
    assert "voluntary_attention_ratio" in measured


def test_agency_is_unmeasured_without_a_meta_reading():
    """The meta confidence before any scored forecast is the prior, not a
    reading: agency is not named measured and PAD keeps its 0.5 default."""
    from mtc.consciousness.active_inference import ActiveInferenceModule

    absent = _state(active_inference_state=SimpleNamespace(meta_confidence=None))
    sources, measured = extract_pad_sources(absent)
    assert "agency_score" not in measured
    assert sources.agency_score == 0.5
    # The real module's state before a scored forecast carries no reading
    fresh = ActiveInferenceModule().get_state()
    assert fresh.meta_confidence is None
    sources, measured = extract_pad_sources(_state(active_inference_state=fresh))
    assert "agency_score" not in measured
    assert sources.agency_score == 0.5


def _drives():
    from mtc.consciousness.active_inference import HomeostaticDrives

    return HomeostaticDrives()


class _Winner:
    def __init__(self, summary, content_type, salience):
        self.candidate = SimpleNamespace(summary=summary, content_type=content_type)
        self.salience = salience


def _damasio_state(result):
    return _state(damasio_state=result)


def test_somatic_valence_without_markers_is_not_measured():
    from mtc.consciousness.damasio import DamasioLayers

    async def run():
        layers = DamasioLayers()
        empty = await layers.process(homeostatic_drives=_drives(), workspace_winners=[])
        winner = await layers.process(
            homeostatic_drives=_drives(),
            workspace_winners=[_Winner("a synthetic topic", "thought", 0.7)],
        )
        return empty, winner

    empty, winner = asyncio.run(run())
    assert empty["core"]["somatic_valence"] is None
    sources, measured = extract_pad_sources(_damasio_state(empty))
    assert "somatic_valence" not in measured
    assert sources.somatic_valence == PADSources().somatic_valence
    assert winner["core"]["somatic_valence"] is not None
    _, measured = extract_pad_sources(_damasio_state(winner))
    assert "somatic_valence" in measured


def test_first_protoself_update_names_no_arousal():
    from mtc.consciousness.damasio import DamasioLayers

    async def run(layers):
        first = await layers.process(homeostatic_drives=_drives(), workspace_winners=[])
        second = await layers.process(
            homeostatic_drives=_drives(), workspace_winners=[]
        )
        return first, second

    first, second = asyncio.run(run(DamasioLayers()))
    assert first["protoself"]["primordial_feelings"]["arousal"] is None
    _, measured = extract_pad_sources(_damasio_state(first))
    assert "protoself_arousal" not in measured
    # The second update has a previous body state: a measured arousal
    _, measured = extract_pad_sources(_damasio_state(second))
    assert "protoself_arousal" in measured
    # The legacy first update records 0.0, which the extractor names
    legacy, _ = asyncio.run(run(DamasioLayers(first_delta="v04_zero")))
    _, measured = extract_pad_sources(_damasio_state(legacy))
    assert "protoself_arousal" in measured


class _SpyPAD(PADAffectModel):
    def __init__(self):
        super().__init__()
        self.measured_args = []

    def compute(self, sources, measured=None):
        self.measured_args.append(None if measured is None else tuple(measured))
        return super().compute(sources, measured)


def test_after_cycle_hands_measured_inputs_to_the_pad_state():
    pad = _SpyPAD()
    augmenter = CycleAugmenter(
        CycleAugmenterConfig.from_overrides({"pad_from_state": True}, pinned_off=True),
        _Workspace(),
        pad_model=pad,
    )
    augmenter.before_cycle([])
    state = _state()
    record = augmenter.after_cycle(state)
    sources, measured = extract_pad_sources(state, augmenter.workspace, None)
    # No drives and no substrate: homeostatic satisfaction and both SNN inputs
    # are not measured and contribute nothing
    assert "homeostatic_satisfaction" not in measured
    assert "snn_valence" not in measured
    assert pad.measured_args == [measured]
    assert record.pad_sources_measured == measured
    assert record.pad_state is not None
    assert record.pad_state.measured_inputs == measured
    expected = PADAffectModel().compute(sources, measured)
    assert (state.pad_pleasure, state.pad_arousal, state.pad_dominance) == (
        expected.pleasure,
        expected.arousal,
        expected.dominance,
    )
    assert state.pad_label == expected.label
    # Pleasure is the mean over its one measured input, not diluted by the
    # unmeasured defaults
    assert state.pad_pleasure == pytest.approx(0.25)
    assert PADAffectModel().compute(sources).pleasure == pytest.approx(0.1)


def test_no_pad_recorded_without_measured_sources():
    pad = PADAffectModel()
    augmenter = CycleAugmenter(
        CycleAugmenterConfig.from_overrides({"pad_from_state": True}, pinned_off=True),
        _Workspace(),
        pad_model=pad,
    )
    augmenter.before_cycle([])
    bare = SimpleNamespace(
        pad_pleasure=None, pad_arousal=None, pad_dominance=None, pad_label=None
    )
    record = augmenter.after_cycle(bare)
    assert bare.pad_label is None
    assert (bare.pad_pleasure, bare.pad_arousal, bare.pad_dominance) == (
        None,
        None,
        None,
    )
    assert record.pad_state is None
    assert record.pad_sources_measured == ()
    # One dimension measured is not enough: no coordinate, the list is kept
    augmenter.before_cycle([])
    arousal_only = SimpleNamespace(
        ignition_events=2,
        pad_pleasure=None,
        pad_arousal=None,
        pad_dominance=None,
        pad_label=None,
    )
    record = augmenter.after_cycle(arousal_only)
    assert record.pad_state is None
    assert record.pad_sources_measured == ("ignition_rate",)
    assert arousal_only.pad_arousal is None
    assert pad.history_length == 0


def test_all_off_is_identity():
    workspace = _Workspace(urgent=[("curiosity", 0.9)])
    augmenter = CycleAugmenter(CycleAugmenterConfig.all_off(), workspace)
    candidates = [_cand(i, act=0.9, emo=0.9) for i in range(30)]
    out = augmenter.before_cycle(candidates)
    assert out == candidates
    assert out is not candidates
    state = _state()
    record = augmenter.after_cycle(state)
    assert record == AugmentRecord()
    assert (state.pad_pleasure, state.pad_arousal, state.pad_dominance) == (
        None,
        None,
        None,
    )
    assert workspace.arousals == []
    assert augmenter.binder is None and augmenter.prospective is None
    assert augmenter.pad_model is None


def test_config_defaults_match_schema():
    with open(ROOT / "experiments" / "param_schema.toml", "rb") as handle:
        schema = tomllib.load(handle)["cycle"]
    config = CycleAugmenterConfig()
    assert set(schema) == set(config.as_dict())
    for name, spec in schema.items():
        assert spec["default"] == getattr(config, name), name
        assert "pin" in spec["description"] and "v0.4.0" in spec["description"], name
    assert list(schema["gate_arousal"]["choices"]) == list(GATE_AROUSAL_SOURCES)
    assert schema["candidate_cap"]["min"] == 0
    assert schema["drive_urgency_threshold"]["min"] == 0.0
    assert schema["drive_urgency_threshold"]["max"] == 1.0
    with pytest.raises(ValueError, match="unknown cycle switches"):
        CycleAugmenterConfig.from_overrides({"camera": True})


def test_state_dict_round_trip():
    config = CycleAugmenterConfig.from_overrides(
        {
            "crossmodal_binding": True,
            "prospective_candidates": True,
            "candidate_cap": 3,
        },
        pinned_off=True,
    )
    augmenter = CycleAugmenter(config, _Workspace(), clock=lambda: NOW)
    inputs = CycleInputs(recent_messages=[{"content": "watching the weather turn"}])
    for i in range(3):
        augmenter.before_cycle([_cand(j) for j in range(5)], inputs)
        augmenter.after_cycle(_state(), inputs)
    saved = augmenter.to_state_dict()
    assert set(saved) == {"scalars", "arrays", "histories"}
    assert saved["scalars"]["before_calls"] == 3
    # Five candidates per cycle capped to three, plus the one prospective
    # candidate injected in the second cycle (the simulator's interval then
    # holds it quiet at this frozen clock): 2 + 3 + 2 dropped.
    assert saved["scalars"]["total_capped"] == 7
    assert saved["scalars"]["total_prospective"] == 1
    assert "binder" in saved["histories"] and "prospective" in saved["histories"]

    restored = CycleAugmenter(config, _Workspace(), clock=lambda: NOW)
    restored.from_state_dict(saved)
    assert restored.status()["before_calls"] == 3
    assert restored.status()["totals"] == augmenter.status()["totals"]
    assert restored.to_state_dict()["histories"]["prospective"] == (
        saved["histories"]["prospective"]
    )


def _bare_pad_state():
    return SimpleNamespace(
        pad_pleasure=None, pad_arousal=None, pad_dominance=None, pad_label=None
    )


def _pad_gate_on():
    return CycleAugmenter(
        CycleAugmenterConfig.from_overrides(
            {"gate_arousal": "pad", "pad_from_state": True}, pinned_off=True
        ),
        _Workspace(),
        pad_model=PADAffectModel(),
    )


def test_pad_gate_warning_names_why_no_arousal_was_recorded(caplog):
    gate_logger = "mtc.pipeline.cycle_augmenter"
    # Arousal is measured (the ignition count, as in every real cycle) while
    # pleasure and dominance have no measured input: the cause names those two
    on = _pad_gate_on()
    on.before_cycle([])
    state = _bare_pad_state()
    state.ignition_events = 2
    with caplog.at_level("WARNING", logger=gate_logger):
        record = on.after_cycle(state)
    assert record.gate_arousal_value is None
    assert "ignition_rate" in record.pad_sources_measured
    messages = [r.getMessage() for r in caplog.records if r.name == gate_logger]
    assert len(messages) == 1
    assert "pad_from_state off" not in messages[0]
    assert "no measured arousal input" not in messages[0]
    assert "no measured input this cycle for pleasure, dominance" in messages[0]

    # Pleasure and arousal measured, dominance not: only dominance is named
    caplog.clear()
    on = _pad_gate_on()
    on.before_cycle([])
    state = _bare_pad_state()
    state.ignition_events = 2
    state.damasio_state = {"core": {"somatic_valence": 0.25}}
    with caplog.at_level("WARNING", logger=gate_logger):
        on.after_cycle(state)
    messages = [r.getMessage() for r in caplog.records if r.name == gate_logger]
    assert len(messages) == 1
    assert "no measured input this cycle for dominance)" in messages[0]
    assert "pleasure" not in messages[0]

    caplog.clear()
    off = CycleAugmenter(
        CycleAugmenterConfig.from_overrides(
            {"gate_arousal": "pad", "pad_from_state": False}, pinned_off=True
        ),
        _Workspace(),
    )
    off.before_cycle([])
    with caplog.at_level("WARNING", logger=gate_logger):
        off.after_cycle(_bare_pad_state())
    messages = [r.getMessage() for r in caplog.records if r.name == gate_logger]
    assert len(messages) == 1
    assert "pad_from_state off" in messages[0]
    assert "no measured arousal input" not in messages[0]


def test_pad_computed_reports_whether_this_call_computed_the_pad():
    augmenter = CycleAugmenter(
        CycleAugmenterConfig.from_overrides({"pad_from_state": True}, pinned_off=True),
        _Workspace(),
        pad_model=PADAffectModel(),
    )
    assert augmenter.pad_computed is False
    augmenter.before_cycle([])
    record = augmenter.after_cycle(_bare_pad_state())
    # The computation returned without a coordinate: it still ran
    assert record.pad_state is None
    assert augmenter.pad_computed is True

    class _BrokenPAD(PADAffectModel):
        def compute(self, sources, measured=None):
            raise RuntimeError("pad broke")

    augmenter._pad_model = _BrokenPAD()
    augmenter.before_cycle([])
    with pytest.raises(RuntimeError):
        augmenter.after_cycle(_bare_pad_state())
    assert augmenter.pad_computed is False

    off = CycleAugmenter(
        CycleAugmenterConfig.from_overrides({"pad_from_state": False}, pinned_off=True),
        _Workspace(),
    )
    off.before_cycle([])
    off.after_cycle(_bare_pad_state())
    assert off.pad_computed is False
