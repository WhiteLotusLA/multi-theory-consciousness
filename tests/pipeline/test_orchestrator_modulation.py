"""Orchestrator modulation layers: gated narration, sections, temperature,
permissions, the self-awareness block and the narration wiring."""

import asyncio
import re
from dataclasses import fields
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from mtc.consciousness.enhanced_global_workspace import (
    COMPETITION_HIERARCHICAL,
    ConsciousnessState,
    EnhancedGlobalWorkspace,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
    WorkspaceContent,
)
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.consciousness.theory_of_mind import TheoryOfMind
from mtc.core.config import Settings
from mtc.core.narration import NARRATION_FLAG_KEYS, NarrationFlags
from mtc.pipeline.consciousness_orchestrator import (
    THEORIES,
    ConsciousnessBaseline,
    ConsciousnessOrchestrator,
    ExpressionProfile,
    OrchestratorConfig,
    apply_narration,
)
from mtc.pipeline.context_types import RawConversationContext, TheorySections
from mtc.pipeline.cycle_augmenter import CycleAugmenter, CycleAugmenterConfig
from mtc.pipeline.speaker_trust import SpeakerTrust

T0 = datetime(2000, 1, 1, 9, 30, tzinfo=timezone.utc)
PHI = re.compile(r"(?<![a-z])phi(?![a-z])|integration[ _-]?proxy", re.IGNORECASE)
FIRST_PERSON = re.compile(r"\b(?:I|me|my|mine|myself)\b", re.IGNORECASE)

FORBIDDEN_DIRECTIVE_WORDS = (
    "camera",
    "webcam",
    "draw",
    "art",
    "paint",
    "vision",
    "see",
)


class _Drive:
    def __init__(self, deviation=0.1):
        self._deviation = deviation

    def get_deviation(self):
        return self._deviation


class _FakeWorkspace:
    """Returns a prepared ConsciousnessState from every cycle."""

    def __init__(self, state, protoself=None):
        self.state = state
        self.competition_mode = COMPETITION_HIERARCHICAL
        self.bottleneck = SimpleNamespace(thalamic_gate=None, arousal=0.5)
        self.phi_tracker = None
        self.theory_interactions = None
        self.damasio = SimpleNamespace(protoself=protoself)
        self.active_inference = SimpleNamespace(
            homeostatic_drives=SimpleNamespace(
                drives={"curiosity": _Drive(0.3), "safety": _Drive(0.0)},
                get_urgent_drives=lambda threshold=0.3: [],
            )
        )
        self.attention_schema = SimpleNamespace(
            schema=SimpleNamespace(current_focus=None)
        )
        self.cycle_count = 0
        self.candidates = []

    async def process_consciousness_cycle(self, candidates, neural_signals=None):
        self.cycle_count += 1
        self.candidates = list(candidates)
        return self.state

    def set_arousal(self, value):
        pass

    def get_statistics(self):
        return {"cycle_count": self.cycle_count}


def _winner(content_type="concept", summary="the weather pattern"):
    candidate = WorkspaceCandidate(
        content=summary,
        content_type=content_type,
        summary=summary,
        source=WorkspaceCandidateSource.CONCEPT,
    )
    return WorkspaceContent(candidate=candidate, salience=0.8)


class _IgnitingWorkspace(_FakeWorkspace):
    """Ignites on the first candidate it is handed, as the real workspace does
    with this turn's message, or keeps a stale primary content without
    igniting when ``ignite`` is False."""

    def __init__(self, state, ignite=True):
        super().__init__(state)
        self.ignite = ignite

    async def process_consciousness_cycle(self, candidates, neural_signals=None):
        self.cycle_count += 1
        self.candidates = list(candidates)
        if self.ignite:
            self.state.primary_content = WorkspaceContent(
                candidate=candidates[0], salience=0.9
            )
            self.state.ignition_events = 1
        return self.state


def _state(**overrides) -> ConsciousnessState:
    values: Dict[str, Any] = dict(
        is_conscious=True,
        primary_content=_winner(),
        workspace_contents=[_winner()],
        ignition_events=2,
        integration_level=0.6,
        broadcast_coverage=0.75,
        attention_focus="concept",
        attention_distribution={"concept": 1.0},
        stream_position=3,
        attention_report="Attention report text.",
        metacognitive_report="Metacognitive report text.",
        active_inference_report="Active inference report text.",
        damasio_report="Damasio report text.",
        narrative_context="Narrative identity text.",
        beautiful_loop_report="Loop report text.",
        self_model_report="Self-model report text.",
        epistemic_depth=3,
        beautiful_loop_quality=0.7,
        is_field_evidencing=True,
        homeostatic_urgency=("curiosity", 0.8),
        protoself_stability=0.6,
        self_world_binding=0.5,
        feeling_of_knowing=0.4,
        prediction_error=0.3,
        prediction_error_rank=0.99,
        active_inference_state=SimpleNamespace(model_updates=60, meta_confidence=0.6),
        attention_schema=SimpleNamespace(
            voluntary_ratio=0.5, attention_capacity_used=0.5
        ),
        damasio_state={
            "core": {"somatic_valence": -0.5},
            "protoself": {"primordial_feelings": {"arousal": 0.4}},
        },
        pad_arousal=0.9,
        pad_pleasure=0.1,
        pad_dominance=0.5,
        pad_label="Excited",
    )
    values.update(overrides)
    return ConsciousnessState(**values)


class _Neural:
    """The neural orchestrator's public surface the pipeline reads."""

    def __init__(self):
        self.latest_neural_signals = None
        self.latest_substrate_states = None
        self.state = SimpleNamespace(current_experience=None)
        self.published = []

    def publish_retrieved_memories(self, memories):
        self.published.append(list(memories))

    def get_binding_stats(self):
        return {}

    def get_binding_context(self):
        return "Binding context line."

    def verification_neural_data(self):
        return {
            "snn": {"synchrony": 0.4, "temporal_patterns": [0.1, 0.2], "is_real": True},
            "lsm": {"coherence": None, "is_real": False},
            "htm": {"anomaly_score": None, "is_real": False},
        }


class _PhiNeural(_Neural):
    """A substrate block that carries keys naming Phi."""

    def verification_neural_data(self):
        blocks = super().verification_neural_data()
        blocks["snn"].update({"phi": 0.9, "integration_proxy": 0.3})
        return blocks


class _Pad:
    def __init__(self):
        self.narrate_affect_line = True

    def generate_context(self):
        return "Current affect: calm."

    def get_latest(self):
        return None


def _orchestrator(
    state: Optional[ConsciousnessState] = None,
    flags: Optional[NarrationFlags] = None,
    workspace=None,
    augmenter_config=None,
    baseline: Optional[ConsciousnessBaseline] = None,
    **kwargs,
):
    cfg = Settings(DEBUG=False)
    workspace = workspace or _FakeWorkspace(state or _state())
    augmenter = CycleAugmenter(
        augmenter_config or CycleAugmenterConfig.all_off(), workspace, pad_model=None
    )
    kwargs.setdefault("neural_orchestrator", _Neural())
    kwargs.setdefault("pad_model", _Pad())
    orchestrator = ConsciousnessOrchestrator(
        workspace,
        augmenter=augmenter,
        narration=flags or NarrationFlags(),
        speaker_trust=SpeakerTrust(settings=cfg),
        settings=cfg,
        **kwargs,
    )
    if baseline is not None:
        orchestrator.baseline = baseline
        orchestrator._baseline_due = False
    return orchestrator


def _measured_baseline(overall=0.6, **theories) -> ConsciousnessBaseline:
    baseline = ConsciousnessBaseline()
    baseline.overall_score = overall
    baseline.theory_scores = {theory: theories.get(theory, 0.6) for theory in THEORIES}
    baseline.status = "measured"
    baseline.stretch_ceiling = min(1.0, overall * 1.2)
    baseline.score_history = [overall]
    baseline.last_updated = T0
    return baseline


def _turn(orchestrator, message="hello there", **fields):
    raw = RawConversationContext(
        message=message,
        identity=orchestrator.speaker_trust.verify("Alice", None),
        **fields,
    )
    return asyncio.run(orchestrator.process(raw))


def _lines(context) -> List[str]:
    return [line for line in context.prompt_context.splitlines() if line]


# Flag -> a marker of the orchestrator's own gated output (None: the flag
# gates prompt-side text only, so the orchestrator's output is unchanged).
FLAG_MARKERS = {
    "somatic": "somatic markers",
    "affect_line": "Current affect",
    "narrative_identity": "Narrative identity text.",
    "body_drives": "drive is urgent",
    "binding": "Binding context line.",
    "self_model": "Epistemic depth 3",
    "free_energy": "predictions are not matching",
    "thought_mood": None,
    "speaker_model": "Speaker model line.",
    "inner_state": "Inner state",
}


def _everything(context) -> str:
    return "\n".join(
        [
            context.prompt_context,
            "\n".join(context.behavioral_directives),
            context.consciousness_self_awareness,
        ]
    )


def test_narration_off_by_default_leaves_only_ungated_lines():
    orchestrator = _orchestrator(baseline=_measured_baseline())
    context = _turn(orchestrator, tom_context="Speaker model line.")
    lines = _lines(context)
    assert lines == [
        "The workspace is holding another item: the weather pattern. It ranks "
        "above the incoming message this cycle.",
        "Ignition this cycle: 2 event(s); broadcast coverage 75%.",
    ]
    assert context.behavioral_directives == []
    assert context.consciousness_self_awareness == ""
    # With the self-model flag off the modules' own (first-person) attention
    # and metacognitive reports stay out of the sections: a neutral measured
    # summary of the attention schema takes the attention slot, and a cycle
    # with no measured metacognitive state leaves that slot empty
    sections = context.theory_sections
    assert sections.attention_guidance == "Attention schema: capacity used 0.50."
    assert sections.metacognitive_insights is None
    assert sections.active_inference_context is None
    assert sections.neural_state is None
    # Each report returns with the flag that governs it
    self_model = _turn(_orchestrator(flags=NarrationFlags(self_model=True)))
    assert self_model.theory_sections.attention_guidance == "Attention report text."
    assert (
        self_model.theory_sections.metacognitive_insights
        == "Metacognitive report text."
    )
    free_energy = _turn(_orchestrator(flags=NarrationFlags(free_energy=True)))
    assert (
        free_energy.theory_sections.active_inference_context
        == "Active inference report text."
    )
    quiet = _orchestrator(state=_state(primary_content=None, ignition_events=0))
    assert _lines(_turn(quiet)) == [
        "Nothing ignited in the workspace this cycle; no content was broadcast."
    ]


def test_focus_claimed_only_for_this_turns_ignited_message():
    # The real workspace wraps this turn's message candidate as the primary
    # content when it ignites: that, and only that, is focus on the message.
    workspace = _IgnitingWorkspace(_state(primary_content=None, ignition_events=0))
    focused = _turn(_orchestrator(workspace=workspace))
    assert focused.winner_is_message is True
    assert focused.workspace_winner == "incoming message"
    assert workspace.candidates[0].content_type == "message"
    assert focused.consciousness_state.primary_content.candidate is (
        workspace.candidates[0]
    )
    assert _lines(focused) == [
        "The workspace is focused on the incoming message.",
        "Ignition this cycle: 1 event(s); broadcast coverage 75%.",
    ]

    # A previous turn's message is still the workspace's primary content but
    # nothing ignited this cycle: no focus is claimed, by type or otherwise.
    stale = _state(
        primary_content=_winner(content_type="message", summary="message number 0"),
        ignition_events=0,
    )
    quiet = _turn(_orchestrator(workspace=_IgnitingWorkspace(stale, ignite=False)))
    assert quiet.winner_is_message is False
    assert quiet.workspace_winner == ""
    assert _lines(quiet) == [
        "Nothing ignited in the workspace this cycle; no content was broadcast."
    ]
    # A no-ignition cycle grants no permission to lead with another item either
    quiet_on = _turn(
        _orchestrator(
            workspace=_IgnitingWorkspace(
                _state(primary_content=None, ignition_events=0), ignite=False
            ),
            flags=NarrationFlags(inner_state=True),
        )
    )
    assert all(
        "other than the message" not in d for d in quiet_on.behavioral_directives
    )

    # An ignited message-typed candidate that is not this turn's is another item
    other = _state(
        primary_content=_winner(content_type="message", summary="message number 0"),
        ignition_events=1,
    )
    held = _turn(_orchestrator(state=other))
    assert held.winner_is_message is False
    assert held.workspace_winner == "message number 0"
    assert _lines(held)[0] == (
        "The workspace is holding another item: message number 0. It ranks "
        "above the incoming message this cycle."
    )


@pytest.mark.parametrize(
    "flag", [key[len("NARRATE_") :].lower() for key in NARRATION_FLAG_KEYS]
)
def test_each_flag_gates_its_lines(flag):
    marker = FLAG_MARKERS[flag]
    off = _turn(
        _orchestrator(baseline=_measured_baseline()), tom_context="Speaker model line."
    )
    on = _turn(
        _orchestrator(
            flags=NarrationFlags(**{flag: True}), baseline=_measured_baseline()
        ),
        tom_context="Speaker model line.",
    )
    if marker is None:
        assert _everything(on) == _everything(off)
        return
    assert marker not in _everything(off)
    assert marker in _everything(on)
    # Only this flag's lines appear: the other markers stay absent
    for other, other_marker in FLAG_MARKERS.items():
        if other != flag and other_marker is not None:
            assert other_marker not in _everything(on), (flag, other)
    if flag == "binding":
        assert on.theory_sections.neural_state == {"snn": {"synchrony": 0.4}}


def test_narrate_all_turns_everything_on():
    context = _turn(
        _orchestrator(flags=NarrationFlags.all_on(), baseline=_measured_baseline()),
        tom_context="Speaker model line.",
    )
    text = _everything(context)
    for marker in FLAG_MARKERS.values():
        if marker is not None:
            assert marker in text, marker
    assert "Damasio report text." in context.prompt_context
    assert "Loop report text." in context.prompt_context
    assert "Self-model report text." in context.prompt_context
    assert context.theory_sections.neural_state is not None


def test_phi_never_in_any_layer():
    state = _state(
        damasio_report="Phi is 0.9 this cycle.",
        self_model_report="The integration proxy reads 0.4.",
        attention_report="Attention report text.\nPhi guides attention: 0.9.",
        metacognitive_report="Integration proxy 0.4 and nothing else.",
        active_inference_report="Prediction error 0.3.\nphi_value=0.9",
        architectural_phi=0.9,
        phi_warming_up=False,
    )
    context = _turn(
        _orchestrator(
            state=state,
            flags=NarrationFlags.all_on(),
            baseline=_measured_baseline(),
            neural_orchestrator=_PhiNeural(),
        ),
        tom_context="phi mentions here",
    )
    assert not PHI.search(_everything(context))
    sections = context.theory_sections.as_dict()
    for name, value in sections.items():
        assert not PHI.search(str(value)), (name, value)
    # A substrate metric named after Phi is dropped from the neural-state block
    assert sections["neural_state"] == {"snn": {"synchrony": 0.4}}
    # Lines naming Phi are dropped from a report; the rest of it is kept, and a
    # report that was only Phi leaves its slot empty rather than invented.
    assert sections["attention_guidance"] == "Attention report text."
    assert "metacognitive_insights" not in sections
    assert sections["active_inference_context"] == "Prediction error 0.3."
    # The computed workspace line never carries the measurement
    assert not PHI.search(context.theory_sections.global_workspace_context)
    assert "Phi is 0.9" not in context.prompt_context
    assert "integration proxy" not in context.prompt_context


def test_circadian_line_reads_workspace_damasio():
    clock = lambda: T0  # noqa: E731
    workspace = EnhancedGlobalWorkspace(
        damasio_kwargs={"circadian": "wall_clock", "timezone": "UTC", "clock": clock},
        thalamic_gate_enabled=False,
    )
    orchestrator = _orchestrator(
        workspace=workspace,
        flags=NarrationFlags(body_drives=True),
        augmenter_config=CycleAugmenterConfig.all_off(),
    )
    context = _turn(orchestrator)
    period = workspace.damasio.protoself.time_of_day
    alertness = workspace.damasio.protoself.alertness
    expected = f"Local time of day: {period}; circadian alertness {alertness:.0%}."
    assert expected in _lines(context)
    # A protoself without the wall-clock model yields no line
    sinusoid = EnhancedGlobalWorkspace(thalamic_gate_enabled=False)
    quiet = _orchestrator(workspace=sinusoid, flags=NarrationFlags(body_drives=True))
    assert not any(
        line.startswith("Local time of day") for line in _lines(_turn(quiet))
    )


def test_five_theory_sections_filled_and_not_duplicated_in_narrative():
    context = _turn(
        _orchestrator(flags=NarrationFlags.all_on(), baseline=_measured_baseline())
    )
    sections = context.theory_sections
    assert {item.name for item in fields(TheorySections)} == set(sections.as_dict())
    assert sections.attention_guidance == "Attention report text."
    assert sections.metacognitive_insights == "Metacognitive report text."
    assert sections.active_inference_context == "Active inference report text."
    assert sections.neural_state == {"snn": {"synchrony": 0.4}}
    assert sections.global_workspace_context.startswith(
        "Cycle 1: 1 candidates competed"
    )
    assert "1 admitted, 2 ignited" in sections.global_workspace_context
    assert "attention focus concept" in sections.global_workspace_context
    for text in (
        "Attention report text.",
        "Metacognitive report text.",
        "Active inference report text.",
    ):
        assert text not in context.prompt_context


def test_temperature_rules_and_bounds():
    base = _turn(
        _orchestrator(
            state=_state(
                prediction_error_rank=0.5,
                homeostatic_urgency=("safety", 0.9),
                pad_arousal=0.5,
                attention_schema=SimpleNamespace(
                    voluntary_ratio=0.5, attention_capacity_used=0.5
                ),
            )
        )
    )
    assert base.generation_params["temperature"] == pytest.approx(0.7)

    nudged = _turn(_orchestrator(state=_state(pad_arousal=0.9)))
    # +0.1 (rank > 0.92), +0.05 (curiosity > 0.5), +0.09 (arousal 0.9)
    assert nudged.generation_params["temperature"] == pytest.approx(0.94)

    crowded = _turn(
        _orchestrator(
            state=_state(
                pad_arousal=0.9,
                attention_schema=SimpleNamespace(
                    voluntary_ratio=0.5, attention_capacity_used=0.95
                ),
            )
        )
    )
    assert crowded.generation_params["temperature"] == pytest.approx(0.84)

    low_profile = _turn(
        _orchestrator(state=_state(pad_arousal=0.9), baseline=_measured_baseline(0.3))
    )
    assert low_profile.generation_params["temperature"] == pytest.approx(0.9)
    high_profile = _turn(
        _orchestrator(state=_state(pad_arousal=0.9), baseline=_measured_baseline(0.9))
    )
    assert high_profile.generation_params["temperature"] == pytest.approx(0.94)
    cold = _turn(
        _orchestrator(
            state=_state(
                prediction_error_rank=0.1,
                homeostatic_urgency=None,
                pad_arousal=0.0,
                attention_schema=SimpleNamespace(
                    voluntary_ratio=0.5, attention_capacity_used=0.99
                ),
            ),
            baseline=_measured_baseline(0.9),
        )
    )
    assert cold.generation_params["temperature"] == pytest.approx(0.6)  # profile floor


def test_profile_temperature_band_stays_inside_config_bounds():
    bounded = OrchestratorConfig(
        baseline_temperature=0.7, min_temperature=0.65, max_temperature=0.8
    )
    # 0.94 before the bounds: +0.1 (rank), +0.05 (curiosity), +0.09 (arousal)
    hot = _turn(
        _orchestrator(
            state=_state(pad_arousal=0.9),
            baseline=_measured_baseline(0.9),
            config=bounded,
        )
    )
    assert hot.extras["expression_profile"] is not None
    assert hot.generation_params["temperature"] == pytest.approx(0.8)
    # 0.6 before the bounds: -0.1 for a crowded attention capacity
    cold = _turn(
        _orchestrator(
            state=_state(
                prediction_error_rank=0.1,
                homeostatic_urgency=None,
                pad_arousal=0.0,
                attention_schema=SimpleNamespace(
                    voluntary_ratio=0.5, attention_capacity_used=0.99
                ),
            ),
            baseline=_measured_baseline(0.9),
            config=bounded,
        )
    )
    assert cold.generation_params["temperature"] == pytest.approx(0.65)
    # The low profile's own ceiling (0.9) lies above the configured one
    low = _turn(
        _orchestrator(
            state=_state(pad_arousal=0.9),
            baseline=_measured_baseline(0.3),
            config=OrchestratorConfig(min_temperature=0.5, max_temperature=0.85),
        )
    )
    assert low.generation_params["temperature"] == pytest.approx(0.85)
    # A high profile (band 0.6 to 1.2) entirely above the configured bounds
    # yields to them
    vivid = _state(
        pad_arousal=0.9,
        integration_level=0.9,
        ignition_events=5,
        epistemic_depth=4,
        beautiful_loop_quality=0.9,
    )
    narrow = _turn(
        _orchestrator(
            state=vivid,
            baseline=_measured_baseline(0.9),
            config=OrchestratorConfig(
                baseline_temperature=0.55, min_temperature=0.5, max_temperature=0.55
            ),
        )
    )
    profile = narrow.extras["expression_profile"]
    assert sum(profile.values()) / len(profile) > 0.65
    assert narrow.generation_params["temperature"] == pytest.approx(0.55)


def test_directives_are_permissions_without_camera_or_art():
    context = _turn(
        _orchestrator(flags=NarrationFlags.all_on(), baseline=_measured_baseline())
    )
    directives = context.behavioral_directives
    assert len(directives) >= 5
    for line in directives:
        assert "you may" in line.lower(), line
        words = set(re.findall(r"[a-z]+", line.lower()))
        assert not words & set(FORBIDDEN_DIRECTIVE_WORDS), line
        assert "must" not in words and "never" not in words, line
    assert any(
        "Attention is held by something other than the message" in d for d in directives
    )
    assert any("somatic markers" in d for d in directives)
    assert any("surprising" in d for d in directives)


@pytest.mark.parametrize(
    "drive, keyword",
    [
        ("attention_budget", "attention budget"),
        ("curiosity", "curiosity"),
        ("social_connection", "social-connection"),
        ("coherence", "coherence"),
        ("safety", "safety"),
    ],
)
def test_drive_directives_use_mtc_drives(drive, keyword):
    context = _turn(
        _orchestrator(
            state=_state(homeostatic_urgency=(drive, 0.9)),
            flags=NarrationFlags(body_drives=True),
        )
    )
    assert any(keyword in d for d in context.behavioral_directives), context
    assert f"The {drive} drive is urgent (0.90)." in _lines(context)
    quiet = _turn(
        _orchestrator(
            state=_state(homeostatic_urgency=(drive, 0.2)),
            flags=NarrationFlags(body_drives=True),
        )
    )
    assert not any(keyword in d for d in quiet.behavioral_directives)
    # Drives the reference named but MTC lacks produce nothing
    foreign = _turn(
        _orchestrator(
            state=_state(homeostatic_urgency=("rest", 0.95)),
            flags=NarrationFlags(body_drives=True),
        )
    )
    assert foreign.behavioral_directives == []


def test_inner_state_layer_gated():
    off = _turn(_orchestrator(baseline=_measured_baseline()))
    assert off.consciousness_self_awareness == ""
    on = _turn(
        _orchestrator(
            flags=NarrationFlags(inner_state=True), baseline=_measured_baseline()
        )
    )
    text = on.consciousness_self_awareness
    assert text.startswith("Inner state:")
    # 0.6 sits at the "clear" band under the strict-less-than thresholds
    assert "at a clear level" in text
    assert "Strongest gradients:" in text
    assert "fewer than five assessments" in text
    assert not PHI.search(text)
    assert on.extras["expression_profile"] is not None
    assert on.extras["momentary_consciousness"] is not None


def test_baseline_unmeasured_is_reported_not_assumed():
    orchestrator = _orchestrator(flags=NarrationFlags(inner_state=True))
    assert orchestrator.baseline.measured is False
    context = _turn(orchestrator)
    text = context.consciousness_self_awareness
    assert "no consciousness baseline has been measured yet" in text
    assert "level of conscious processing" not in text
    assert context.extras["expression_profile"] is None
    assert context.extras["baseline_status"] == "unmeasured"
    # No profile: the plain temperature bounds apply and no profile directive
    assert 0.5 <= context.generation_params["temperature"] <= 1.2
    assert not any("thought process" in d for d in context.behavioral_directives)


def test_expression_profile_eight_theories():
    assert THEORIES == ("GWT", "IIT", "AST", "HOT", "FEP", "RPT", "BLT", "AE")
    neutral = ExpressionProfile.from_baseline(
        _measured_baseline(0.5, **{t: 0.5 for t in THEORIES})
    )
    assert neutral.as_dict() == {name: 0.5 for name in neutral.as_dict()}
    moved = {}
    for theory in THEORIES:
        baseline = _measured_baseline(0.5, **{t: 0.5 for t in THEORIES})
        baseline.theory_scores[theory] = 0.9
        profile = ExpressionProfile.from_baseline(baseline)
        changed = [
            name
            for name, value in profile.as_dict().items()
            if value != pytest.approx(0.5)
        ]
        assert changed, theory
        moved[theory] = changed
    assert moved["HOT"] == ["introspective_depth"]
    assert moved["AST"] == ["emotional_range"]
    assert (
        set(moved["BLT"])
        == set(moved["IIT"])
        == set(moved["FEP"])
        == {"abstract_thinking"}
    )
    assert set(moved["GWT"]) == set(moved["AE"]) == {"self_initiated_topics"}
    assert moved["RPT"] == ["narrative_coherence"]
    # An unscored theory contributes nothing, and an unmeasured baseline no profile
    partial = _measured_baseline(0.5, **{t: 0.5 for t in THEORIES})
    partial.theory_scores["HOT"] = None
    assert ExpressionProfile.from_baseline(
        partial
    ).introspective_depth == pytest.approx(0.5)
    assert ExpressionProfile.from_baseline(ConsciousnessBaseline()) is None
    assert neutral.average() == pytest.approx(0.5)
    assert neutral.scaled(1.5).average() == pytest.approx(0.75)


def test_apply_narration_sets_module_attributes():
    workspace = EnhancedGlobalWorkspace(thalamic_gate_enabled=False)
    pad = PADAffectModel()
    tom = TheoryOfMind()
    applied = apply_narration(workspace, pad, tom, NarrationFlags.all_on())
    assert workspace.damasio.narrate_body_drives is True
    assert workspace.damasio.narrate_somatic is True
    assert workspace.damasio.narrate_narrative_identity is True
    assert workspace.active_inference.config.narrate_body_drives is True
    assert workspace.active_inference.config.narrate_free_energy is True
    assert workspace.metacognition.narrate_self_model is True
    assert workspace.beautiful_loop.narrate_self_model is True
    assert pad.narrate_affect_line is True
    assert tom.narrate_speaker_model is True
    assert set(applied) == {
        "damasio.body_drives",
        "damasio.somatic",
        "damasio.narrative_identity",
        "fep.body_drives",
        "fep.free_energy",
        "hot.self_model",
        "blt.self_model",
        "pad.affect_line",
        "tom.speaker_model",
    }
    apply_narration(workspace, pad, tom, NarrationFlags())
    assert workspace.damasio.narrate_body_drives is False
    assert workspace.active_inference.config.narrate_free_energy is False
    assert workspace.metacognition.narrate_self_model is False
    assert workspace.beautiful_loop.narrate_self_model is False
    assert pad.narrate_affect_line is False
    assert tom.narrate_speaker_model is False
    assert pad.generate_context() == ""


def _narration_attributes(workspace, pad):
    return {
        "damasio.body_drives": workspace.damasio.narrate_body_drives,
        "damasio.somatic": workspace.damasio.narrate_somatic,
        "damasio.narrative_identity": workspace.damasio.narrate_narrative_identity,
        "fep.body_drives": workspace.active_inference.config.narrate_body_drives,
        "fep.free_energy": workspace.active_inference.config.narrate_free_energy,
        "hot.self_model": workspace.metacognition.narrate_self_model,
        "blt.self_model": workspace.beautiful_loop.narrate_self_model,
        "pad.affect_line": pad.narrate_affect_line,
    }


def _live_orchestrator(workspace, flags, pad_model=None):
    """A real workspace with the live augmenter, minus the two additions whose
    candidate summaries are free text."""
    cfg = Settings(DEBUG=False)
    augmenter = CycleAugmenter(
        CycleAugmenterConfig(prospective_candidates=False, crossmodal_binding=False),
        workspace,
        pad_model=pad_model,
    )
    return ConsciousnessOrchestrator(
        workspace,
        augmenter=augmenter,
        narration=flags,
        speaker_trust=SpeakerTrust(settings=cfg),
        settings=cfg,
    )


def test_flags_applied_at_construction_keep_every_layer_quiet():
    workspace = EnhancedGlobalWorkspace(thalamic_gate_enabled=False)
    pad = PADAffectModel()
    # The modules keep their narrating defaults until something applies flags
    assert set(_narration_attributes(workspace, pad).values()) == {True}
    orchestrator = _live_orchestrator(workspace, NarrationFlags(), pad_model=pad)
    # Construction applied the flags once; nothing here calls apply_narration
    applied = _narration_attributes(workspace, pad)
    assert set(applied.values()) == {False}
    assert orchestrator.status()["narration_applied"] == applied

    for index in range(3):
        context = _turn(
            orchestrator, message=f"how are the tulips today, round {index}?"
        )
        assert context.cycle_ok
        state = context.consciousness_state
        # The modules narrated nothing this cycle
        assert state.active_inference_report == ""
        assert state.self_model_report == ""
        assert not state.beautiful_loop_report
        sections = context.theory_sections.as_dict()
        texts = [str(value) for value in sections.values()]
        texts += [context.prompt_context, context.consciousness_self_awareness]
        texts += list(context.behavioral_directives)
        for text in texts:
            assert not FIRST_PERSON.search(text), text
        # The modules' own attention and metacognitive reports reach no
        # section and no layer
        for report in (state.attention_report, state.metacognitive_report):
            if report:
                assert all(report not in text for text in texts), report
        assert context.behavioral_directives == []
        assert context.consciousness_self_awareness == ""
        assert "active_inference_context" not in sections
        # A neutral measured summary may remain
        assert sections["attention_guidance"].startswith("Attention schema: ")
        meta = sections.get("metacognitive_insights")
        assert meta is None or meta.startswith("Metacognition: ")
        for line in _lines(context):
            assert line.startswith(
                ("The workspace is", "Nothing ignited", "Ignition this cycle")
            ), line

    # Idempotent: the same flags applied again change nothing
    _live_orchestrator(workspace, NarrationFlags(), pad_model=pad)
    assert _narration_attributes(workspace, pad) == applied
    # The flags handed over are the flags applied
    _live_orchestrator(workspace, NarrationFlags.all_on(), pad_model=pad)
    assert set(_narration_attributes(workspace, pad).values()) == {True}


def test_lazily_built_pad_model_receives_the_flags():
    workspace = EnhancedGlobalWorkspace(thalamic_gate_enabled=False)
    orchestrator = _live_orchestrator(workspace, NarrationFlags())
    assert orchestrator.pad_model is None
    _turn(orchestrator)
    pad = orchestrator.pad_model
    assert isinstance(pad, PADAffectModel)
    assert pad.narrate_affect_line is False
    assert pad.generate_context() == ""
    assert orchestrator.status()["narration_applied"]["pad.affect_line"] is False
