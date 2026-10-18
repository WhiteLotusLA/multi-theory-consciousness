"""The machine's hooks and controls: honest quiet reasons, the contention
guard around the model-written form, broadcasts, injection, pause, the
scheduler's vocabulary, start and stop, persistence and loud failures."""

import inspect
from dataclasses import fields

import pytest

from mtc.consciousness.continuous.machine import ContinuousThoughtMachine
from mtc.consciousness.continuous.schemas import (
    CONSCIOUSNESS_STATES,
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
    new_thought_id,
)
from mtc.consciousness.continuous.store import InMemoryThoughtStore
from mtc.consciousness.sleep_scheduler import THOUGHT_MODES, ThoughtStreamControl
from mtc.core.config import Settings
from mtc.core.embeddings import HashEmbedder
from mtc.neural.mongodb_schemas import ThoughtStream as MongoThoughtStream
from tests.consciousness.continuous.test_machine_cadence import (
    START,
    Clock,
    cfg_with,
    make_machine,
    run,
)

FAST = {"max_thoughts_per_minute": 1000}


class FakeEnhancer:
    def __init__(self, text="A model-written line.", miss=False):
        self.text = text
        self.miss = miss
        self.calls = []

    async def generate(self, category, context, state="awake"):
        self.calls.append((category, dict(context), state))
        if self.miss:
            return None
        return ThoughtRecord(
            thought_id=new_thought_id(),
            timestamp=START,
            thought_type=ThoughtType.REFLECTION,
            content=self.text,
            intensity=ThoughtIntensity.MODERATE,
            salience_score=0.8,
            triggered_by=f"llm_{category}",
            consciousness_state=state,
        )


def test_quiet_without_embedder_and_status_names_it():
    store = InMemoryThoughtStore()
    machine = make_machine(["river", "garden"], embedder=None, store=store)
    assert machine.get_status()["quiet_reason"] == "no_embedder"
    result = run(machine.tick())
    assert result.generated and result.persisted
    assert result.expressed is False and result.quiet_reason == "no_embedder"
    status = machine.get_status()
    assert status["quiet_reason"] == "no_embedder"
    assert status["novelty"]["embedder"] == "none"
    assert run(store.expressible(5)) == []
    (stored,) = run(store.recent(5))
    assert stored.should_express is False  # the provisional flag was replaced
    assert stored.processing_context["provisional_express"] is True

    no_filter = make_machine(["river"], novelty=None)
    assert run(no_filter.tick()).quiet_reason == "no_embedder"

    lexical = make_machine(["river"], embedder=HashEmbedder(dim=32))
    assert run(lexical.tick()).quiet_reason == "non_semantic_embedder"
    assert lexical.get_status()["quiet_reason"] == "non_semantic_embedder"
    opted = make_machine(
        ["river"], embedder=HashEmbedder(dim=32), allow_non_semantic=True
    )
    result = run(opted.tick())
    assert result.expressed is True and result.quiet_reason is None
    assert opted.get_status()["novelty"]["embedder"] == "non_semantic"


def test_quiet_machine_warns_once_at_construction(caplog):
    with caplog.at_level("WARNING", logger="mtc.consciousness.continuous.machine"):
        make_machine(embedder=None)
        make_machine()
    lines = [r.getMessage() for r in caplog.records if "stay quiet" in r.getMessage()]
    assert len(lines) == 1 and "no_embedder" in lines[0]


def test_llm_upgrade_only_when_uncontended():
    for guard, expect_upgrade in ((lambda: True, False), (lambda: False, True)):
        enhancer = FakeEnhancer()
        store = InMemoryThoughtStore()
        machine = make_machine(
            ["river"], store=store, enhancer=enhancer, contention_guard=guard
        )
        result = run(machine.tick())
        assert result.expressed is True
        assert result.upgraded is expect_upgrade
        (stored,) = run(store.recent(5))
        if expect_upgrade:
            assert enhancer.calls and enhancer.calls[0][0] == "reflection"
            assert stored.content == "A model-written line."
            assert stored.should_express is True
            assert stored.processing_context["upgraded_from"]
            assert result.quiet_reason is None
        else:
            assert enhancer.calls == []
            assert stored.content == "river"
            assert result.quiet_reason == "contended"

    unguarded = make_machine(["river"], enhancer=FakeEnhancer())
    result = run(unguarded.tick())
    assert result.upgraded is False and result.quiet_reason == "contended"
    assert unguarded.get_status()["contention_guard"] is False


def test_llm_miss_keeps_template_and_counts():
    from tests.consciousness.continuous.test_generator import make as make_generator

    # A real seed generator drafts the thought, so the template form checked
    # below is the generator's own marking that the machine keeps on a miss.
    drafted = []
    store = InMemoryThoughtStore()
    machine = make_machine(
        store=store,
        generator=make_generator(),
        enhancer=FakeEnhancer(miss=True),
        contention_guard=lambda: False,
        on_thought_generated=lambda thought: drafted.append(thought.content),
    )
    result = run(machine.tick())
    assert result.upgraded is False and result.expressed is True
    (stored,) = run(store.recent(5))
    assert drafted and stored.content == drafted[0]  # nothing composed instead
    assert stored.content != "A model-written line."
    assert stored.processing_context["form"] == "template"
    assert "template_salience" not in stored.processing_context
    assert "upgraded_from" not in stored.processing_context
    assert machine.get_status()["counters"]["llm_misses"] == 1


def test_upgraded_thought_names_its_salience_basis():
    from mtc.brain.protocol import FakeBrain
    from mtc.consciousness.continuous.config import CTMConfig
    from mtc.consciousness.continuous.enhancer import LLMThoughtEnhancer
    from mtc.consciousness.continuous.persona import CTMPersona

    def measured_template(thought):
        thought.salience_score = 0.58
        thought.processing_context["salience_basis"] = "affect"

    enhancer = LLMThoughtEnhancer(
        FakeBrain(["The light keeps changing."]),
        CTMPersona(name="Testagent", system_prompt="You are {name}."),
        CTMConfig(),
        now=lambda: START,
        narrate_mood=False,
    )
    store = InMemoryThoughtStore()
    machine = make_machine(
        ["river"],
        store=store,
        enhancer=enhancer,
        contention_guard=lambda: False,
        on_thought_generated=measured_template,
    )
    result = run(machine.tick())
    assert result.upgraded is True
    (stored,) = run(store.recent(5))
    assert stored.content == "The light keeps changing."
    assert stored.salience_score == 0.8
    context = stored.processing_context
    assert context["salience_basis"] == "llm_design_constant"
    assert context["form"] == "model"
    assert context["template_salience"] == 0.58
    assert context["template_salience_basis"] == "affect"
    assert context["upgraded_from"]


def test_broadcast_only_expressed():
    messages = []

    async def broadcaster(message):
        messages.append(message)

    machine = make_machine(
        ["river", "the river", "garden"],
        broadcaster=broadcaster,
        config=cfg_with(**FAST),
    )
    results = [run(machine.tick()) for _ in range(3)]
    assert [r.expressed for r in results] == [True, False, True]
    assert [m["thought"]["content"] for m in messages] == ["river", "garden"]
    assert all(m["type"] == "spontaneous_thought" for m in messages)
    assert "stream_state" in messages[0]
    for message in messages:
        assert message["thought"]["form"] == "template"
        assert message["thought"]["salience_basis"] == "strategy_default"

    quiet = []
    silent = make_machine(["river"], embedder=None, broadcaster=quiet.append)
    run(silent.tick())
    assert quiet == []


def test_broadcast_names_the_model_form():
    from mtc.brain.protocol import FakeBrain
    from mtc.consciousness.continuous.config import CTMConfig
    from mtc.consciousness.continuous.enhancer import LLMThoughtEnhancer
    from mtc.consciousness.continuous.persona import CTMPersona

    messages = []
    machine = make_machine(
        ["river"],
        broadcaster=messages.append,
        enhancer=LLMThoughtEnhancer(
            FakeBrain(["A model-written line."]),
            CTMPersona(name="Testagent", system_prompt="You are {name}."),
            CTMConfig(),
            now=lambda: START,
            narrate_mood=False,
            settings=Settings(DEBUG=False),
        ),
        contention_guard=lambda: False,
    )
    result = run(machine.tick())
    assert result.upgraded is True
    (message,) = messages
    assert message["thought"]["content"] == "A model-written line."
    assert message["thought"]["form"] == "model"
    assert message["thought"]["salience_basis"] == "llm_design_constant"
    assert message["thought"]["salience"] == 0.8


def test_inject_thought_stores_unexpressed():
    store = InMemoryThoughtStore()
    machine = make_machine(store=store)
    machine.pause()
    record = run(machine.inject_thought("A loose dream.", "dream_creative", "calm"))
    assert record.thought_type is ThoughtType.DREAM
    assert record.should_express is False
    assert record.triggered_by == "dream_creative"
    assert record.processing_context["emotional_tone"] == "calm"
    assert record.processing_context["form"] == "injected"
    (stored,) = run(store.recent(5))
    assert stored.thought_id == record.thought_id
    assert run(store.expressible(5)) == []
    assert run(machine.inject_thought("  ", "dream", "")) is None
    other = run(machine.inject_thought("A plan.", "planning", ""))
    assert other.thought_type is ThoughtType.PLANNING
    odd = run(machine.inject_thought("Something.", "unknown_kind", ""))
    assert odd.thought_type is ThoughtType.SPONTANEOUS


def test_hooks_called():
    generated, persisted = [], []

    async def on_persisted(thought):
        persisted.append(thought.thought_id)

    machine = make_machine(
        ["river", "the river"],
        on_thought_generated=lambda t: generated.append(t.thought_id),
        on_thought_persisted=on_persisted,
        config=cfg_with(**FAST),
    )
    first = run(machine.tick())
    duplicate = run(machine.tick())
    assert generated == [first.thought_id, duplicate.thought_id]
    assert persisted == [first.thought_id]


def test_pause_resume():
    machine = make_machine(["river"])
    machine.pause()
    result = run(machine.tick())
    assert result.generated is False and result.quiet_reason == "paused"
    assert machine.get_status()["quiet_reason"] == "paused"
    assert machine.generator.calls == []
    machine.resume()
    assert run(machine.tick()).generated is True


def test_set_state_vocabulary_matches_sleep_scheduler():
    assert tuple(THOUGHT_MODES) == tuple(CONSCIOUSNESS_STATES)
    machine = make_machine()
    for mode in THOUGHT_MODES:
        machine.set_consciousness_state(mode)
        assert machine.get_status()["consciousness_state"] == mode
    with pytest.raises(ValueError):
        machine.set_consciousness_state("napping")


def test_satisfies_thought_stream_control():
    machine = make_machine()
    assert isinstance(machine, ThoughtStreamControl)
    expected = list(inspect.signature(ThoughtStreamControl.inject_thought).parameters)
    actual = list(inspect.signature(ContinuousThoughtMachine.inject_thought).parameters)
    assert actual == expected


def test_stop_cancels_cleanly():
    async def scenario():
        machine = make_machine(["river"])
        await machine.start()
        assert machine.running is True
        task = machine._task
        await machine.start()  # a second start makes no second task
        assert machine._task is task
        await machine.stop()
        assert machine.running is False
        assert task.done()
        await machine.stop()  # stopping twice is harmless
        return machine

    machine = run(scenario())
    assert machine.get_status()["running"] is False


def test_state_dict_round_trip():
    clock = Clock()
    machine = make_machine(
        ["river", "garden", "the garden"], clock=clock, config=cfg_with(**FAST)
    )
    for _ in range(3):
        run(machine.tick())
    machine.set_consciousness_state("drowsy")
    state = machine.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    restored = make_machine(clock=clock)
    restored.from_state_dict(state)
    assert restored.counters == machine.counters
    assert restored.consciousness_state == "drowsy"
    assert restored.to_state_dict() == state
    restored.from_state_dict(
        {"scalars": {"consciousness_state": "napping", "counters": {"x": -1}}}
    )
    assert restored.consciousness_state == "drowsy"
    assert "x" not in restored.counters


def test_no_depth_level_field():
    assert "depth_level" not in {f.name for f in fields(ThoughtRecord)}
    assert "depth_level" not in {f.name for f in fields(MongoThoughtStream)}
    store = InMemoryThoughtStore()
    machine = make_machine(["river"], store=store)
    run(machine.tick())
    (stored,) = run(store.recent(1))
    assert "depth_level" not in stored.to_doc()
    assert "depth_level" not in stored.to_doc()["metadata"]
    assert not hasattr(ContinuousThoughtMachine, "_compute_depth_level")


def test_step_failure_counted_and_raised_under_debug():
    class BrokenStore(InMemoryThoughtStore):
        async def insert(self, record):
            raise RuntimeError("store down")

    soft = make_machine(["river"], store=BrokenStore())
    result = run(soft.tick())
    assert result.generated is True and result.persisted is False
    assert soft.get_status()["failures"] == {"persist": 1}

    loud = make_machine(["river"], store=BrokenStore(), settings=Settings(DEBUG=True))
    with pytest.raises(RuntimeError):
        run(loud.tick())


def _broken_summary():
    raise RuntimeError("summary down")


def test_stream_summary_failure_counted_and_raised_under_debug():
    messages = []
    soft = make_machine(["river"], broadcaster=messages.append)
    soft.buffer.get_stream_summary = _broken_summary
    result = run(soft.tick())
    assert result.expressed is True
    (message,) = messages
    assert message["stream_state"] == {}  # the fallback is unchanged
    assert dict(soft.failures) == {"stream_summary": 1}  # counted in the broadcast
    status = soft.get_status()  # and in the status read
    assert status["stream_summary"] == {}
    assert status["failures"] == {"stream_summary": 2}

    loud = make_machine(
        ["river"], broadcaster=messages.append, settings=Settings(DEBUG=True)
    )
    loud.buffer.get_stream_summary = _broken_summary
    with pytest.raises(RuntimeError):
        run(loud.tick())
    with pytest.raises(RuntimeError):
        loud.get_status()


def test_status_carries_the_enhancer_miss_reasons():
    from mtc.brain.protocol import FakeBrain
    from mtc.consciousness.continuous.config import CTMConfig
    from mtc.consciousness.continuous.enhancer import LLMThoughtEnhancer
    from mtc.consciousness.continuous.persona import CTMPersona

    enhancer = LLMThoughtEnhancer(
        FakeBrain(["", "<think>only reasoning</think>"]),
        CTMPersona(name="Testagent", system_prompt="You are {name}."),
        CTMConfig(),
        now=lambda: START,
        narrate_mood=False,
        settings=Settings(DEBUG=False),
    )
    machine = make_machine(
        ["river", "garden"],
        enhancer=enhancer,
        contention_guard=lambda: False,
        config=cfg_with(**FAST),
    )
    run(machine.tick())
    run(machine.tick())
    status = machine.get_status()
    assert status["enhancer"] is True  # the boolean key stays
    assert status["enhancer_status"]["miss_reasons"] == {
        "empty_reply": 1,
        "reasoning_only": 1,
    }
    assert status["enhancer_status"]["last_miss_reason"] == "reasoning_only"
    assert status["counters"]["llm_misses"] == 2
    assert "enhancer_status" not in make_machine(["river"]).get_status()


def test_status_keys_for_the_integration():
    machine = make_machine(["river"])
    run(machine.tick())
    status = machine.get_status()
    assert status["total_thoughts_generated"] == 1
    assert status["consciousness_state"] == "awake"
    assert isinstance(status["stream_summary"], dict)
    assert inspect.iscoroutinefunction(machine.tick)
