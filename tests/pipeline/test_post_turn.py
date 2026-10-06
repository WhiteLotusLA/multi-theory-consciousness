"""Post-turn hooks: order, isolation, cadences, trust source and honesty."""

import secrets
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtc.consciousness.emotion_dynamics import EmotionDynamics
from mtc.consciousness.emotion_extractor import ExtractedEmotion
from mtc.consciousness.enhanced_global_workspace import ConsciousnessState
from mtc.consciousness.pad_affect import PADState
from mtc.consciousness.theory_of_mind import TheoryOfMind
from mtc.consciousness.verification_framework import ConsciousnessVerifier, TestType
from mtc.core.config import Settings
from mtc.memory.enhanced_memory_manager import EnhancedMemoryManager
from mtc.memory.episodic_store import InMemoryEpisodicStore
from mtc.memory.inmemory import InMemoryConversationRepository, InMemoryGraphStore
from mtc.pipeline.message_persistence import MessagePersistence
from mtc.pipeline.post_turn import (
    STEP_ORDER,
    PostTurnConfig,
    PostTurnIntegrator,
    PostTurnReport,
    TurnRecord,
)
from mtc.pipeline.speaker_trust import SpeakerEntry, SpeakerTrust

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
TOKEN_ENV = "MTC_TEST_POST_TURN_CREATOR_TOKEN"


def _settings(**overrides):
    values = {
        "SYSTEM_NAME": "Testagent",
        "CREATOR_NAME": "Operator",
        "TRUSTED_SPEAKERS": "Operator",
        "EPISODIC_IMPORTANCE": 6.5,
        "DEBUG": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def cfg():
    return _settings()


@pytest.fixture
def trust(monkeypatch, cfg):
    token = secrets.token_hex(16)
    monkeypatch.setenv(TOKEN_ENV, token)
    source = SpeakerTrust(
        [SpeakerEntry("Operator", "creator", 0.9, token_env=TOKEN_ENV)], settings=cfg
    )
    return source, token


def _state(**overrides):
    values = dict(
        is_conscious=True,
        primary_content=None,
        workspace_contents=[],
        ignition_events=1,
        integration_level=0.6,
        broadcast_coverage=0.5,
        attention_focus="message",
        attention_distribution={"message": 3.0},
        stream_position=1,
        pad_pleasure=0.4,
        pad_arousal=0.6,
        pad_dominance=0.5,
    )
    values.update(overrides)
    return ConsciousnessState(**values)


def _record(
    identity, turn=1, message="How is the garden?", spoken="It is green.", **kw
):
    values = dict(
        identity=identity,
        message=message,
        spoken=spoken,
        thinking=None,
        state=_state(),
        modulated=None,
        conversation_id=str(uuid.uuid4()),
        turn_number=turn,
        topics=("garden",),
    )
    values.update(kw)
    return TurnRecord(**values)


class Recorder:
    """Records calls of every listed method, in one shared order list."""

    def __init__(self, order, name, fail=False, result=None):
        self.order, self.name, self.fail, self.result = order, name, fail, result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.order.append(self.name)
        self.calls.append((args, kwargs))
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return self.result


class AsyncRecorder(Recorder):
    async def __call__(self, *args, **kwargs):
        return Recorder.__call__(self, *args, **kwargs)


def _fakes(order, failing=()):
    def sync(name, result=None):
        return Recorder(order, name, name in failing, result)

    def coro(name, result=None):
        return AsyncRecorder(order, name, name in failing, result)

    suite = SimpleNamespace(
        status="measured", overall_score=0.7, measured=["coherence"], insufficient=[]
    )
    verifier = SimpleNamespace(run_verification_suite=sync("verification", suite))
    assembler = SimpleNamespace(assemble=lambda **kw: {"pad_affect": None})
    integration = SimpleNamespace(
        append_temporal_sample=sync("temporal_append"),
        get_temporal_data=lambda: {"states": [], "timestamps": []},
    )
    neural = SimpleNamespace(process_experience=sync("neural_experience"))
    extractor = SimpleNamespace(
        extract=coro(
            "emotion",
            ExtractedEmotion(
                label="curious",
                secondary=None,
                confidence=0.5,
                baseline="prior",
                raw={},
            ),
        )
    )
    learning = SimpleNamespace(record_turn=coro("learning"))
    tom = SimpleNamespace(
        update_from_turn=sync("theory_of_mind", SimpleNamespace(relationship=None))
    )
    memory = SimpleNamespace(
        store_memory=coro(
            "episodic_memory",
            SimpleNamespace(postgres_id=1, buffered=False, errors={}),
        )
    )
    graph = SimpleNamespace(
        merge_concept=sync("concepts"), link_concept_to_conversation=lambda *a: None
    )
    workspace = SimpleNamespace(
        active_inference=SimpleNamespace(update_homeostasis=coro("homeostasis"))
    )
    safety = SimpleNamespace(post_message_log=coro("safety_log"))
    shadow = sync("affect_shadow")
    identity = SimpleNamespace(
        create_marker=sync("identity_marker", SimpleNamespace(marker_id="m1")),
        store_identity_marker=lambda marker: None,
    )
    orchestrator = SimpleNamespace(
        baseline_due=True,
        refresh_baseline=coro("baseline", SimpleNamespace(overall_score=0.5)),
    )
    return dict(
        verifier=verifier,
        state_assembler=assembler,
        integration=integration,
        neural_orchestrator=neural,
        emotion_extractor=extractor,
        learning=learning,
        theory_of_mind=tom,
        memory_manager=memory,
        graph=graph,
        workspace=workspace,
        safety=safety,
        identity_continuity=identity,
        orchestrator=orchestrator,
        affect_shadow_tap=shadow,
    )


@pytest.mark.asyncio
async def test_hooks_run_in_order_and_isolated(trust, cfg):
    source, _ = trust
    order = []
    failing = {"verification", "learning", "safety_log"}
    integrator = PostTurnIntegrator(
        speaker_trust=source, settings=cfg, **_fakes(order, failing)
    )
    report = await integrator.run(_record(source.verify("Alice", None), turn=20))
    assert isinstance(report, PostTurnReport)
    assert list(report.steps) == list(STEP_ORDER)
    for step in failing:
        assert report.steps[step] == "failed: RuntimeError"
    assert report.steps["baseline"] == "ok"
    assert report.steps["identity_marker"] == "ok"
    # hooks ran in the documented order, later ones despite earlier failures
    expected = [
        "verification",
        "temporal_append",
        "verification",
        "neural_experience",
        "emotion",
        "learning",
        "theory_of_mind",
        "episodic_memory",
        "concepts",
        "homeostasis",
        "safety_log",
        "affect_shadow",
        "identity_marker",
        "baseline",
    ]
    collapsed = [name for i, name in enumerate(order) if i == 0 or order[i - 1] != name]
    assert collapsed == expected
    # turn 20 is a temporal turn, so the failing verifier fails that step too
    assert report.steps["temporal_verification"] == "failed: RuntimeError"
    assert integrator.status()["failures"] == {
        "verification": 1,
        "temporal_verification": 1,
        "learning": 1,
        "safety_log": 1,
    }
    # with nothing attached, every step reports itself honestly
    bare = PostTurnIntegrator(
        speaker_trust=source,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    steps = (await bare.run(_record(source.verify("Alice", None)))).steps
    assert steps["affect_shadow"] == "disabled"
    assert {v for k, v in steps.items() if k != "affect_shadow"} == {"absent"}


@pytest.mark.asyncio
async def test_debug_setting_raises(trust):
    source, _ = trust
    order = []
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        settings=_settings(DEBUG=True),
        **_fakes(order, {"learning"}),
    )
    with pytest.raises(RuntimeError, match="learning failed"):
        await integrator.run(_record(source.verify("Alice", None)))


@pytest.mark.asyncio
async def test_verification_every_turn_temporal_every_ten(trust, cfg):
    source, _ = trust
    order = []
    fakes = _fakes(order)
    integrator = PostTurnIntegrator(speaker_trust=source, settings=cfg, **fakes)
    identity = source.verify("Alice", None)
    temporal_turns = []
    for turn in range(1, 21):
        report = await integrator.run(_record(identity, turn=turn))
        assert report.steps["verification"] == "ok"
        if report.steps["temporal_verification"] == "ok":
            temporal_turns.append(turn)
        else:
            assert report.steps["temporal_verification"] == "not_due"
    assert temporal_turns == [10, 20]
    calls = fakes["verifier"].run_verification_suite.calls
    per_turn = [
        c for c in calls if c[1]["test_types"] != [TestType.TEMPORAL_CONSISTENCY]
    ]
    temporal = [
        c for c in calls if c[1]["test_types"] == [TestType.TEMPORAL_CONSISTENCY]
    ]
    assert len(per_turn) == 20 and len(temporal) == 2
    assert per_turn[0][1]["test_types"] == [
        TestType.COHERENCE,
        TestType.EMOTIONAL_AUTHENTICITY,
    ]
    assert "temporal_data" in temporal[0][0][0]
    # a sample is appended every turn
    assert len(fakes["integration"].append_temporal_sample.calls) == 20


@pytest.mark.asyncio
async def test_insufficient_verification_reported_not_scored(trust, cfg):
    source, _ = trust

    class Integration:
        def __init__(self):
            self.samples = []

        def collect_neural_data(self):
            return {"snn": {"is_real": False}, "lsm": {}, "htm": {}}

        def append_temporal_sample(self, sample):
            self.samples.append(sample)

        def get_temporal_data(self):
            return {"states": list(self.samples), "timestamps": []}

    verifier = ConsciousnessVerifier()
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        verifier=verifier,
        integration=Integration(),
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    report = await integrator.run(
        _record(source.verify("Alice", None), state=_state(pad_pleasure=None))
    )
    assert report.steps["verification"] == "insufficient_data"
    details = report.details["verification"]
    assert details["overall_score"] is None
    assert details["status"] == "insufficient_data"
    assert verifier.history[-1].overall_score is None


@pytest.mark.asyncio
async def test_neural_experience_emotions_from_pad_and_trust_from_speaker_trust(
    trust, cfg
):
    source, token = trust
    calls = []
    neural = SimpleNamespace(process_experience=lambda **kw: calls.append(kw))
    winner = SimpleNamespace(
        candidate=SimpleNamespace(metadata={"emotions": {"joy": 0.2, "trust": 1.0}})
    )
    # The workspace still holds a winner from an earlier cycle; it is never
    # consulted, only this cycle's state is
    workspace = SimpleNamespace(current_conscious_content=winner)
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        neural_orchestrator=neural,
        workspace=workspace,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    # no winner this cycle: the documented PAD mapping plus the verified trust
    spoofed = source.verify(cfg.CREATOR_NAME, None)
    await integrator.run(_record(spoofed, state=_state(pad_pleasure=-0.3)))
    assert calls[-1]["emotions"] == {
        "joy": 0.0,
        "sadness": pytest.approx(0.3),
        "trust": cfg.DEFAULT_EMOTION_TRUST,
    }
    assert calls[-1]["valence"] == pytest.approx(-0.3)
    assert calls[-1]["arousal"] == pytest.approx(0.6)
    assert cfg.DEFAULT_EMOTION_TRUST == 0.4
    # the authenticated, allowlisted creator gets the trusted emotion trust
    verified = source.verify(cfg.CREATOR_NAME, token)
    await integrator.run(_record(verified))
    assert calls[-1]["emotions"]["trust"] == cfg.TRUSTED_EMOTION_TRUST
    # this cycle's ignited winner lends its emotions, but never the trust
    ignited = _state(primary_content=winner, ignition_events=1)
    await integrator.run(_record(spoofed, state=ignited))
    assert calls[-1]["emotions"] == {"joy": 0.2, "trust": cfg.DEFAULT_EMOTION_TRUST}
    # a primary content kept from an earlier cycle (nothing ignited this
    # cycle) is not this cycle's winner
    kept = _state(primary_content=winner, ignition_events=0, pad_pleasure=0.5)
    await integrator.run(_record(spoofed, state=kept))
    assert calls[-1]["emotions"] == {
        "joy": pytest.approx(0.5),
        "sadness": 0.0,
        "trust": cfg.DEFAULT_EMOTION_TRUST,
    }
    # no winner emotions and no PAD: nothing is invented
    report = await integrator.run(
        _record(spoofed, state=_state(pad_pleasure=None, pad_arousal=None))
    )
    assert report.steps["neural_experience"] == "skipped_no_affect"
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_pad_is_this_cycles_never_a_stale_model_reading(trust, cfg):
    source, _ = trust
    calls = []
    neural = SimpleNamespace(process_experience=lambda **kw: calls.append(kw))

    class Pad:
        def __init__(self, latest):
            self.latest = latest

        def get_latest(self):
            return self.latest

    # A reading taken long before this turn's cycle
    pad = Pad(PADState(pleasure=-0.9, arousal=0.1, dominance=0.1, timestamp=1.0))
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        neural_orchestrator=neural,
        pad_model=pad,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    identity = source.verify("Alice", None)
    # This cycle's state carries PAD: it wins over the model's older reading
    await integrator.run(_record(identity))
    assert calls[-1]["valence"] == pytest.approx(0.4)
    assert calls[-1]["arousal"] == pytest.approx(0.6)
    # No PAD on this cycle's state and only a stale reading: nothing measured
    unmeasured = dict(pad_pleasure=None, pad_arousal=None, pad_dominance=None)
    report = await integrator.run(_record(identity, state=_state(**unmeasured)))
    assert report.steps["neural_experience"] == "skipped_no_affect"
    assert len(calls) == 1
    # A reading taken after this cycle's state was formed is this cycle's
    state = _state(**unmeasured)
    pad.latest = PADState(
        pleasure=0.2, arousal=0.3, dominance=0.4, timestamp=state.timestamp + 0.5
    )
    await integrator.run(_record(identity, state=state))
    assert calls[-1]["valence"] == pytest.approx(0.2)
    assert calls[-1]["arousal"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_speaker_name_never_becomes_a_concept(trust, cfg):
    source, _ = trust
    graph = InMemoryGraphStore()
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        graph=graph,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    # An unauthenticated speaker is stored under the bracketed label; neither
    # the label nor the bare name becomes a concept, from either side
    claimed = source.verify("Alice", None)
    await integrator.run(
        _record(
            claimed,
            message="Alice planted tulips",
            spoken="Alice, those tulips sound wonderful",
        )
    )
    names = set(graph.list_concept_names())
    assert "tulips" in names and "planted" in names
    assert "alice" not in names and "[alice]" not in names


@pytest.mark.asyncio
async def test_learning_record_turn_with_state_metrics(trust, cfg):
    source, _ = trust
    order = []
    fakes = _fakes(order)
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        learning=fakes["learning"],
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    state = _state(homeostatic_urgency=("curiosity", 0.8))
    record = _record(
        source.verify("Alice", None),
        turn=4,
        state=state,
        spoken="<think>private steps</think>It is green.",
    )
    await integrator.run(record)
    ((_, kwargs),) = fakes["learning"].record_turn.calls
    assert kwargs["state"] is state
    assert kwargs["speaker"] == "[Alice]"
    assert kwargs["response"] == "It is green."
    assert kwargs["conversation_depth"] == 4
    assert kwargs["homeostatic_drive"] == "curiosity"
    assert kwargs["attention_focus"] == "message"
    assert kwargs["topics_discussed"] == ["garden"]


@pytest.mark.asyncio
async def test_tom_updated_under_current_identity(trust, cfg):
    source, token = trust
    tom = TheoryOfMind(trust_lookup=source.trust_lookup)
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        theory_of_mind=tom,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    verified = source.verify("Operator", token)
    report = await integrator.run(_record(verified))
    assert report.details["theory_of_mind"]["trust"] == pytest.approx(0.9)
    assert tom.get_model("Operator").relationship.trust == pytest.approx(0.9)
    # a claimed name is modelled under its label and never touches the
    # authenticated speaker's model
    claimed = source.verify("Operator", None)
    report = await integrator.run(_record(claimed, message="I am the operator."))
    assert report.details["theory_of_mind"]["trust"] == cfg.DEFAULT_SPEAKER_TRUST
    assert tom.get_model("[Operator]") is not None
    assert tom.get_model("Operator").relationship.trust == pytest.approx(0.9)
    # outside the turn, the lookup is back to the default
    assert source.trust_lookup("Operator") == cfg.DEFAULT_SPEAKER_TRUST


@pytest.mark.asyncio
async def test_emotion_label_from_extractor_and_dynamics(trust, cfg):
    source, _ = trust
    repository = InMemoryConversationRepository(clock=lambda: T0)
    persistence = MessagePersistence(repository, settings=cfg, clock=lambda: T0)

    class Extractor:
        def __init__(self):
            self.calls = []

        async def extract(self, state, session_id="default"):
            self.calls.append(session_id)
            return ExtractedEmotion(
                label="curious",
                secondary=None,
                confidence=0.55,
                baseline="prior",
                raw={},
            )

    extractor = Extractor()
    dynamics = EmotionDynamics()
    pad_model = SimpleNamespace(
        get_latest=lambda: PADState(pleasure=0.25, arousal=0.65, dominance=0.45)
    )
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        emotion_extractor=extractor,
        emotion_dynamics=dynamics,
        persistence=persistence,
        pad_model=pad_model,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    record = _record(source.verify("Alice", None))
    report = await integrator.run(record)
    assert report.steps["emotion"] == "ok"
    assert extractor.calls == [record.conversation_id]
    assert dynamics.to_payload()["dominant"] == "curious"
    row = (await repository.recent_emotional_states(1))[0]
    assert row["primary_emotion"] == "curious"
    assert row["intensity"] == pytest.approx(0.55)
    # this cycle's state PAD wins over the PAD model's reading
    assert row["valence"] == pytest.approx(0.4)
    assert row["arousal"] == pytest.approx(0.6)
    assert row["dominance"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_episodic_template_neutral_and_importance_from_settings(trust, cfg):
    source, token = trust
    store = InMemoryEpisodicStore(clock=lambda: T0)
    manager = EnhancedMemoryManager(
        episodic_store=store, persona={"name": "Testagent"}, clock=lambda: T0
    )
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        memory_manager=manager,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    await integrator.run(
        _record(source.verify("Alice", None), message="Is it raining?", spoken="No.")
    )
    await integrator.run(
        _record(source.verify("Operator", token), message="Thanks.", spoken="Sure.")
    )
    rows = await store.recent(24 * 365 * 30, 10)
    contents = sorted(row["content"] for row in rows)
    assert contents == [
        "Operator said: Thanks.\nTestagent responded: Sure.",
        "[Alice] said: Is it raining?\nTestagent responded: No.",
    ]
    assert {row["importance"] for row in rows} == {6.5}
    assert rows[0]["emotional_context"] == {
        "pleasure": 0.4,
        "arousal": 0.6,
        "dominance": 0.5,
    }


@pytest.mark.asyncio
async def test_concepts_for_message_and_spoken_only(trust, cfg):
    source, _ = trust
    graph = InMemoryGraphStore()
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        graph=graph,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    record = _record(
        source.verify("Alice", None),
        message="The telescope arrived",
        spoken="<think>planetarium secretly</think>Wonderful telescope",
        thinking="observatory reasoning",
    )
    report = await integrator.run(record)
    names = set(graph.list_concept_names())
    assert {"telescope", "arrived", "wonderful"} <= names
    for hidden in ("planetarium", "secretly", "observatory", "reasoning"):
        assert hidden not in names
    assert report.details["concepts"]["stored"] == 4


@pytest.mark.asyncio
async def test_homeostasis_activity_type(trust, cfg):
    source, _ = trust
    activities = []

    async def update(activity):
        activities.append(activity)

    workspace = SimpleNamespace(
        active_inference=SimpleNamespace(update_homeostasis=update)
    )
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        workspace=workspace,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    identity = source.verify("Alice", None)
    await integrator.run(_record(identity, message="What grows here?"))
    await integrator.run(_record(identity, message="The roses grow here."))
    assert activities == [{"type": "exploring"}, {"type": "conversation"}]


@pytest.mark.asyncio
async def test_safety_post_message_log_called(trust, cfg):
    source, _ = trust
    calls = []

    async def post_message_log(response, emotional_state=None):
        calls.append((response, emotional_state))
        return {}

    integrator = PostTurnIntegrator(
        speaker_trust=source,
        safety=SimpleNamespace(post_message_log=post_message_log),
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    await integrator.run(
        _record(source.verify("Alice", None), spoken="<think>x</think>Visible reply.")
    )
    assert calls == [
        ("Visible reply.", {"pleasure": 0.4, "arousal": 0.6, "dominance": 0.5})
    ]


@pytest.mark.asyncio
async def test_identity_marker_every_n_with_supplied_values_only(trust, cfg):
    source, _ = trust
    created = []

    class Continuity:
        def create_marker(self, **kwargs):
            created.append(kwargs)
            return SimpleNamespace(marker_id=f"m{len(created)}")

        def store_identity_marker(self, marker):
            return None

    integrator = PostTurnIntegrator(
        speaker_trust=source,
        identity_continuity=Continuity(),
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    identity = source.verify("Alice", None)
    steps = []
    for turn in (1, 19, 20, 21, 40):
        report = await integrator.run(_record(identity, turn=turn, provider_id="mock"))
        steps.append(report.steps["identity_marker"])
    assert steps == ["not_due", "not_due", "ok", "not_due", "ok"]
    marker = created[0]
    assert marker["core_values"] is None
    assert marker["personality_traits"] is None
    assert marker["relationship_bonds"] is None
    assert marker["growth_stage"] is None
    assert marker["memory_anchors"][0]["type"] == "conversation"
    assert marker["memory_anchors"][0]["turn"] == 20
    assert marker["emotional_fingerprint"] == {
        "pleasure": 0.4,
        "arousal": 0.6,
        "dominance": 0.5,
    }
    assert marker["model_context"] == {"provider_id": "mock"}


@pytest.mark.asyncio
async def test_baseline_refreshed_when_due(trust, cfg):
    source, _ = trust
    refreshed = []

    class Orchestrator:
        baseline_due = False
        baseline_due_reason = None
        fail = False

        async def refresh_baseline(self, trigger="heartbeat"):
            refreshed.append(trigger)
            return None if self.fail else SimpleNamespace(overall_score=None)

    orchestrator = Orchestrator()
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        orchestrator=orchestrator,
        settings=cfg,
        config=PostTurnConfig(affect_shadow=False),
    )
    identity = source.verify("Alice", None)
    report = await integrator.run(_record(identity))
    assert report.steps["baseline"] == "not_due" and refreshed == []
    orchestrator.baseline_due = True
    orchestrator.baseline_due_reason = "ignition_spike"
    report = await integrator.run(_record(identity))
    assert report.steps["baseline"] == "ok" and refreshed == ["ignition_spike"]
    assert report.details["baseline"] == {
        "trigger": "ignition_spike",
        "refreshed": True,
        "overall_score": None,
    }
    orchestrator.fail = True
    orchestrator.baseline_due_reason = None
    report = await integrator.run(_record(identity))
    assert report.steps["baseline"] == "failed: assessment"
    assert refreshed[-1] == "heartbeat"


@pytest.mark.asyncio
async def test_affect_shadow_tap_gets_reply_message_and_pad(trust, cfg):
    source, _ = trust
    taps = []
    integrator = PostTurnIntegrator(
        speaker_trust=source,
        settings=cfg,
        affect_shadow_tap=lambda *args: taps.append(args),
    )
    await integrator.run(
        _record(source.verify("Alice", None), spoken="<think>x</think>Shown.")
    )
    assert taps == [
        (
            "Shown.",
            "How is the garden?",
            {"pleasure": 0.4, "arousal": 0.6, "dominance": 0.5},
        )
    ]


def test_turn_record_refuses_a_bare_name():
    with pytest.raises(TypeError):
        TurnRecord(
            identity="Operator",
            message="m",
            spoken="s",
            thinking=None,
            state=None,
            modulated=None,
            conversation_id="c",
            turn_number=1,
        )
