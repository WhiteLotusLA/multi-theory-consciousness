"""The thought generator: strategies from config, material only from real
sources, seed text only from the packs, and no model call of its own."""

import asyncio
import inspect
import logging
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtc.consciousness.continuous import generator as generator_module
from mtc.consciousness.continuous.config import CTMConfig
from mtc.consciousness.continuous.generator import (
    BridgeEmotionSource,
    EmotionSource,
    MemoryManagerSource,
    MemorySource,
    MemoryWeightRule,
    PadEmotionSource,
    ThoughtGenerator,
    name_affect,
)
from mtc.consciousness.continuous.schemas import ThoughtIntensity, ThoughtType
from mtc.consciousness.continuous.seeds import SeedLibrary
from mtc.core.config import Settings

ROOT = Path(__file__).resolve().parents[3]
START = datetime(2000, 1, 1, 9, 0, tzinfo=timezone.utc)

PACK = {
    "memory_templates": ["Back to {memory}, {observation}."],
    "emotion_templates": [
        "I notice {current_emotion}.",
        "Some {emotion} is here.",
    ],
    "association_templates": ["{first} meets {second}."],
    "questions": ["What is next?"],
    "spontaneous_lines": ["A {descriptor} moment."],
    "dream_consolidation_templates": ["Pieces settle, {reflection}."],
    "dream_creative_templates": ["{concept} turns {descriptor}."],
    "dream_emotion_templates": ["A {current_emotion} tone.", "A {feeling} tone."],
    "dream_spontaneous_templates": ["Something {descriptor}."],
    "concept_pairs": [["rest", "attention"], ["habit", "change"]],
    "creative_concepts": ["a quiet garden"],
    "memory_query_terms": ["a plan", "a routine"],
    "reflection": ["a pattern"],
    "feeling": ["steady"],
    "emotion": ["calm"],
    "descriptor": ["small"],
    "assessment": ["worth a look"],
    "observation": ["what was left out"],
}


def run(coro):
    return asyncio.run(coro)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


class FakeMemory:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    async def search(self, query, limit, min_importance):
        self.queries.append((query, limit, min_importance))
        return [dict(r) for r in self.rows]


class FixedAffect:
    def __init__(self, valence, arousal):
        self.reading = {"valence": valence, "arousal": arousal}

    def get_affect(self):
        return dict(self.reading)


def config(**weights) -> CTMConfig:
    waking = {k: 0.0 for k in CTMConfig().waking_weights()}
    data = {}
    if weights:
        waking.update(weights)
        data["strategy_weights"] = waking
    return CTMConfig.model_validate(data)


def make(cfg=None, seed=1, pack=PACK, **kwargs) -> ThoughtGenerator:
    kwargs.setdefault("weight_rules", [])
    return ThoughtGenerator(
        SeedLibrary(pack, [Path("test-pack.yaml")]),
        cfg or CTMConfig(),
        rng=random.Random(seed),
        clock=Clock(),
        **kwargs,
    )


def test_strategy_weights_from_config():
    only_questions = make(config(question=1.0))
    for _ in range(10):
        assert (
            run(only_questions.generate_thought()).thought_type is ThoughtType.QUESTION
        )

    halves = make(config(association=0.5, question=0.5), seed=3)
    types = Counter(run(halves.generate_thought()).thought_type for _ in range(2000))
    assert set(types) == {ThoughtType.ASSOCIATION, ThoughtType.QUESTION}
    assert types[ThoughtType.ASSOCIATION] / 2000 == pytest.approx(0.5, abs=0.05)
    assert halves.status()["by_strategy"] == {
        "association": types[ThoughtType.ASSOCIATION],
        "question": types[ThoughtType.QUESTION],
    }


def test_anti_repeat_guard_after_three_same_types():
    memory = FakeMemory([{"id": 1, "content": "a long walk", "importance": 5}])
    gen = make(
        config(memory=0.5, emotion=0.5, spontaneous=0.0001),
        memory_source=memory,
        emotion_source=FixedAffect(0.5, 0.7),
    )
    gen.recent_thought_types.extend(["memory_replay"] * 3)
    guarded = run(gen.generate_thought())
    assert guarded.thought_type is ThoughtType.SPONTANEOUS
    # for one choice only: the run is broken, so memory and emotion return
    after = [run(gen.generate_thought()).thought_type for _ in range(3)]
    assert ThoughtType.SPONTANEOUS not in after

    # dream thoughts share one type, so a run of dreams guards nothing
    dreams = make(seed=2, emotion_source=FixedAffect(0.5, 0.7))
    strategies = Counter(
        run(dreams.generate_thought("dreaming")).processing_context["strategy"]
        for _ in range(300)
    )
    assert strategies["emotion"] > 20
    woken = make(
        config(memory=0.5, emotion=0.5, spontaneous=0.0001),
        memory_source=memory,
        emotion_source=FixedAffect(0.5, 0.7),
    )
    woken.recent_thought_types.extend(["dream"] * 3)
    assert run(woken.generate_thought()).thought_type is not ThoughtType.SPONTANEOUS


def test_dream_state_uses_dream_strategies():
    gen = make(seed=5, emotion_source=FixedAffect(0.1, 0.3))
    seen = set()
    for state in ("dreaming", "deep_sleep"):
        for _ in range(200):
            thought = run(gen.generate_thought(state))
            assert thought.thought_type is ThoughtType.DREAM
            assert thought.triggered_by.startswith("dream_")
            assert thought.processing_context["template_key"].startswith("dream_")
            seen.add(thought.processing_context["strategy"])
    assert seen == {"consolidation", "creative", "emotion", "spontaneous"}
    awake = run(gen.generate_thought("awake"))
    assert awake.thought_type is not ThoughtType.DREAM
    with pytest.raises(ValueError):
        run(gen.generate_thought("asleep"))


def test_memory_strategy_uses_memory_source():
    memory = FakeMemory(
        [{"id": 7, "content": "We planted seeds in the garden.", "importance": 8}]
    )
    gen = make(config(memory=1.0), memory_source=memory)
    thought = run(gen.generate_thought())
    assert thought.thought_type is ThoughtType.MEMORY_REPLAY
    assert thought.content.startswith('Back to "We planted seeds in the garden"')
    assert thought.associated_memories == ["7"]
    assert thought.salience_score == pytest.approx(0.8)
    assert thought.processing_context["salience_basis"] == "memory_importance"
    assert memory.queries and memory.queries[0][0] in PACK["memory_query_terms"]
    assert gen.llm_context()["recent_memory"] == "We planted seeds in the garden"
    assert isinstance(memory, MemorySource)


def test_memory_manager_adapter_calls_search_memories():
    calls = []

    class Manager:
        async def search_memories(self, query, limit=10, min_importance=0.0):
            calls.append((query, limit, min_importance))
            return [{"id": 3, "content": "a note", "importance": 4.0}]

    # A manager with no episodic store to read is searched through
    # search_memories, which counts the access, and the source says so.
    source = MemoryManagerSource(Manager())
    assert source.touches_access is True
    rows = run(source.search("a plan", 5, 1.0))
    assert rows == [{"id": 3, "content": "a note", "importance": 4.0}]
    assert calls == [("a plan", 5, 1.0)]


def test_memory_manager_source_replays_without_touching(tmp_path):
    from mtc.memory.enhanced_memory_manager import EnhancedMemoryManager
    from mtc.memory.episodic_store import InMemoryEpisodicStore

    store = InMemoryEpisodicStore()
    settings = Settings(
        PROJECT_ROOT=str(tmp_path), SYSTEM_NAME="Unit", CREATOR_NAME="Alice"
    )
    manager = EnhancedMemoryManager(episodic_store=store, settings=settings)
    text = "We planted seeds in the garden."
    memory_id = run(store.add_episodic(text, importance=6.0))
    source = MemoryManagerSource(manager)
    assert isinstance(source, MemorySource)
    assert source.touches_access is False
    rows = run(source.search("garden", 5, 0.0))
    assert rows == [
        {
            "id": memory_id,
            "content": text,
            "importance": 6.0,
            "memory_type": "episodic",
        }
    ]
    gen = make(
        config(memory=1.0),
        pack=dict(PACK, memory_query_terms=["garden"]),
        memory_source=source,
    )
    for _ in range(3):
        thought = run(gen.generate_thought())
        assert thought.associated_memories == [str(memory_id)]
    # A replay is not a recall: the stored access count is untouched.
    (row,) = run(store.get_episodic([memory_id]))
    assert row["accessed_count"] == 0
    # The manager's own search does count one, which is why replay avoids it.
    run(manager.search_memories("garden", limit=5))
    (row,) = run(store.get_episodic([memory_id]))
    assert row["accessed_count"] == 1


def test_no_memory_source_skips_memory_strategy_and_says_so(caplog):
    alone = make(config(memory=1.0))
    assert run(alone.generate_thought()) is None
    assert alone.last_miss_reason == "no_seeds"
    status = alone.status()
    assert status["memory_source"] is False
    assert status["skipped"]["memory"] == 1

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=generator_module.__name__):
        mixed = make(config(memory=0.5, question=0.5))
        types = {run(mixed.generate_thought()).thought_type for _ in range(20)}
    assert types == {ThoughtType.QUESTION}
    lines = [r for r in caplog.records if "no memory source" in r.getMessage()]
    assert len(lines) == 1
    assert mixed.status()["skipped"]["memory"] == 20


def test_emotion_maps_valence_arousal_quadrants():
    assert name_affect(0.6, 0.9)[0] == "excitement"
    assert name_affect(0.6, 0.1)[0] == "contentment"
    assert name_affect(-0.6, 0.9)[0] == "tension"
    assert name_affect(-0.6, 0.1)[0] == "sadness"
    assert name_affect(0.0, 0.5)[1] == 0.0
    assert name_affect(1.0, 1.0)[1] == 1.0

    gen = make(config(emotion=1.0), emotion_source=FixedAffect(-0.8, 0.9))
    thought = run(gen.generate_thought())
    assert thought.thought_type is ThoughtType.EMOTION
    assert thought.content == "I notice tension."  # the measured name, not a slot word
    name, strength = name_affect(-0.8, 0.9)
    assert thought.emotional_context == {name: strength}
    assert thought.processing_context["affect"] == {"valence": -0.8, "arousal": 0.9}
    assert thought.processing_context["salience_basis"] == "affect"
    assert isinstance(FixedAffect(0, 0), EmotionSource)


def _emotion_only() -> CTMConfig:
    waking = {k: 0.0 for k in CTMConfig().waking_weights()}
    dream = {k: 0.0 for k in CTMConfig().dream_weights()}
    waking["emotion"] = dream["emotion"] = 1.0
    return CTMConfig.model_validate(
        {"strategy_weights": waking, "dream_strategy_weights": dream}
    )


def test_emotion_thoughts_name_the_measured_emotion_only():
    pack = dict(PACK)
    pack["emotion_templates"] = [
        "I notice {current_emotion}.",
        "Some {emotion} is here.",
        "I notice {current_emotion}, and it feels {feeling}.",
    ]
    pack["dream_emotion_templates"] = [
        "A {current_emotion} tone.",
        "A {feeling} tone.",
        "Some {emotion} drifts by.",
    ]
    gen = make(_emotion_only(), pack=pack, emotion_source=FixedAffect(-0.8, 0.9))
    texts = set()
    for state in ("awake", "drowsy", "dreaming", "deep_sleep"):
        for _ in range(60):
            thought = run(gen.generate_thought(state))
            assert thought.processing_context["strategy"] == "emotion"
            assert thought.emotional_context == dict([name_affect(-0.8, 0.9)])
            texts.add(thought.content)
    # Only the templates that name the measured emotion and draw no random
    # emotion or feeling word were used.
    assert texts == {"I notice tension.", "A tension tone."}


def test_emotion_strategy_without_a_naming_template_has_no_material():
    pack = dict(PACK)
    pack["emotion_templates"] = ["Some {emotion} is here."]
    pack["dream_emotion_templates"] = ["A {feeling} tone."]
    gen = make(_emotion_only(), pack=pack, emotion_source=FixedAffect(0.5, 0.5))
    for state in ("awake", "dreaming"):
        assert run(gen.generate_thought(state)) is None
        assert gen.last_miss_reason == "no_seeds"


def test_every_shipped_template_can_be_chosen():
    """Each shipped template uses only the context its strategy supplies, so
    none is dead; no shipped template draws a random emotion or feeling word,
    so none puts an unmeasured feeling into a thought; and the emotion
    families name the measured emotion."""
    from mtc.consciousness.continuous.seeds import (
        SLOT_KEYS,
        TEMPLATE_KEYS,
        template_fields,
    )

    supplied = {
        "memory_templates": {"memory"},
        "emotion_templates": {"current_emotion"},
        "association_templates": {"first", "second"},
        "questions": set(),
        "spontaneous_lines": set(),
        "dream_consolidation_templates": {"memory"},
        "dream_creative_templates": {"first", "second", "concept"},
        "dream_emotion_templates": {"current_emotion"},
        "dream_spontaneous_templates": set(),
    }
    assert set(supplied) == set(TEMPLATE_KEYS)  # every family is checked
    assert set(generator_module.AFFECT_SLOTS) == {"emotion", "feeling"}
    seeds = SeedLibrary.from_config_dir(ROOT / "config" / "ctm")
    for key, context in supplied.items():
        assert seeds.entries(key), key
        for template in seeds.entries(key):
            names = set(template_fields(template))
            assert names - set(SLOT_KEYS) <= context, (key, template)
            assert not names & set(generator_module.AFFECT_SLOTS), (key, template)
            if key in ("emotion_templates", "dream_emotion_templates"):
                assert "current_emotion" in names, template


def _bridge(trained, valence=0.5, arousal=0.6):
    state = SimpleNamespace(
        intuitive_valence=valence, intuitive_arousal=arousal, readout_trained=trained
    )
    return SimpleNamespace(readout_trained=lambda: trained, state_history=[state])


def test_untrained_bridge_gives_no_emotion():
    untrained = BridgeEmotionSource(_bridge(False, None, None))
    assert untrained.get_affect() is None
    assert BridgeEmotionSource(lambda: None).get_affect() is None
    trained = BridgeEmotionSource(lambda: _bridge(True, 0.5, 0.6))
    assert trained.get_affect() == {"valence": 0.5, "arousal": 0.6}

    gen = make(config(emotion=0.5, spontaneous=0.5), emotion_source=untrained)
    for _ in range(30):
        thought = run(gen.generate_thought())
        assert thought.thought_type is ThoughtType.SPONTANEOUS
        assert thought.emotional_context == {}
        assert "affect" not in thought.processing_context
    assert gen.status()["skipped"]["emotion"] == 30


def test_pad_source_reads_pleasure_and_arousal():
    class Pad:
        def __init__(self, latest):
            self.latest = latest

        def get_latest(self):
            return self.latest

    assert PadEmotionSource(Pad(None)).get_affect() is None
    reading = PadEmotionSource(
        Pad(SimpleNamespace(pleasure=-0.2, arousal=0.7, dominance=0.5))
    ).get_affect()
    assert reading == {"valence": -0.2, "arousal": 0.7}


def test_sensitive_rule_downweights_never_zero():
    settings = Settings(
        SENSITIVE_MEMORY_PHRASES="garden", SENSITIVE_MEMORY_REPLAY_WEIGHT=0.35
    )
    rules = MemoryWeightRule.from_settings(settings)
    assert [(r.name, r.phrases, r.weight) for r in rules] == [
        ("sensitive", ("garden",), 0.35)
    ]
    memory = FakeMemory(
        [
            {"id": 1, "content": "A talk about the Garden", "importance": 5},
            {"id": 2, "content": "A walk in the rain", "importance": 5},
        ]
    )
    gen = make(config(memory=1.0), seed=11, memory_source=memory, weight_rules=rules)
    picks = Counter(
        run(gen.generate_thought()).processing_context["memory_id"] for _ in range(1000)
    )
    assert picks["1"] > 0
    assert picks["1"] / picks["2"] == pytest.approx(0.35, abs=0.1)

    with pytest.raises(ValueError):
        MemoryWeightRule("x", ("garden",), 0.0)
    assert MemoryWeightRule.from_settings(Settings(SENSITIVE_MEMORY_PHRASES="")) == []


def test_lifecycle_noise_patterns_filter():
    cfg = config(memory=1.0)
    cfg.lifecycle_noise_patterns = [r"messages exchanged"]
    memory = FakeMemory(
        [
            {"id": 1, "content": "Conversation ended, 12 messages exchanged"},
            {"id": 2, "content": "The light through the window", "importance": 6},
        ]
    )
    gen = make(cfg, memory_source=memory)
    for _ in range(20):
        thought = run(gen.generate_thought())
        assert "messages exchanged" not in thought.content
        assert thought.associated_memories == ["2"]
    assert gen.status()["noise_filtered"] == 20

    only_noise = make(cfg, memory_source=FakeMemory([memory.rows[0]]))
    assert run(only_noise.generate_thought()) is None
    assert only_noise.last_miss_reason == "no_seeds"


def test_intensity_by_state():
    gen = make(config(spontaneous=1.0))
    expected = {
        "awake": ThoughtIntensity.MODERATE,
        "drowsy": ThoughtIntensity.QUIET,
    }
    for state, intensity in expected.items():
        assert run(gen.generate_thought(state)).intensity is intensity
    dreams = make(emotion_source=None)
    assert run(dreams.generate_thought("dreaming")).intensity is ThoughtIntensity.QUIET
    assert (
        run(dreams.generate_thought("deep_sleep")).intensity is ThoughtIntensity.WHISPER
    )
    strong = make(config(emotion=1.0), emotion_source=FixedAffect(0.9, 0.95))
    assert run(strong.generate_thought("awake")).intensity is ThoughtIntensity.STRONG
    mild = make(config(emotion=1.0), emotion_source=FixedAffect(0.1, 0.55))
    assert run(mild.generate_thought("awake")).intensity is ThoughtIntensity.MODERATE


def test_should_express_is_provisional():
    always = config(spontaneous=1.0)
    always.express_probability.spontaneous = 1.0
    thought = run(make(always).generate_thought())
    assert thought.should_express is True
    assert thought.processing_context["express_basis"] == "provisional"
    never = config(spontaneous=1.0)
    never.express_probability.spontaneous = 0.0
    assert run(make(never).generate_thought()).should_express is False
    doc = " ".join(generator_module.__doc__.split())
    assert "the machine replaces it with its novelty decision" in doc


def test_no_llm_branch():
    parameters = inspect.signature(ThoughtGenerator).parameters
    assert not {"brain", "enhancer", "llm"} & set(parameters)
    source = Path(generator_module.__file__).read_text(encoding="utf-8")
    assert ".think(" not in source and "mtc.brain" not in source
    assert run(make().generate_thought()) is not None


def test_no_seeds_returns_none_with_reason():
    empty = ThoughtGenerator(SeedLibrary({}, []), CTMConfig(), weight_rules=[])
    assert run(empty.generate_thought()) is None
    assert empty.last_miss_reason == "no_seeds"
    assert empty.status()["misses"] == 1
    assert empty.status()["seed_source"] == "missing"
    assert empty.llm_context() == {
        "recent_memory": "",
        "emotions": {},
        "recent_topic": "",
    }


def test_state_dict_round_trip():
    gen = make(config(association=0.5, question=0.5))
    for _ in range(6):
        run(gen.generate_thought())
    state = gen.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    restored = make()
    restored.from_state_dict(state)
    assert restored.generated == 6
    assert list(restored.recent_thought_types) == list(gen.recent_thought_types)
    assert restored.strategy_counts == gen.strategy_counts
    restored.from_state_dict(
        {"scalars": {"generated": "x"}, "histories": {"recent_thought_types": [3]}}
    )
    assert restored.generated == 6 and list(restored.recent_thought_types) == []


def test_seeded_reproducible():
    memory_rows = [{"id": 1, "content": "a quiet morning", "importance": 5}]

    def texts(seed):
        gen = make(
            seed=seed,
            memory_source=FakeMemory(memory_rows),
            emotion_source=FixedAffect(0.3, 0.4),
        )
        return [run(gen.generate_thought()).content for _ in range(25)]

    assert texts(42) == texts(42)
    assert texts(42) != texts(43)


def test_default_pack_generates_in_every_state():
    seeds = SeedLibrary.from_config_dir(ROOT / "config" / "ctm")
    gen = ThoughtGenerator(
        seeds,
        CTMConfig(),
        memory_source=FakeMemory([{"id": 1, "content": "a short walk"}]),
        emotion_source=FixedAffect(0.2, 0.6),
        weight_rules=[],
        rng=random.Random(9),
        clock=Clock(),
    )
    for state in ("awake", "drowsy", "dreaming", "deep_sleep"):
        for _ in range(40):
            thought = run(gen.generate_thought(state))
            assert thought is not None and thought.content
            assert "{" not in thought.content
            assert thought.timestamp.tzinfo is not None


class BrokenMemory:
    async def search(self, query, limit, min_importance):
        raise RuntimeError("memory store down")


class BrokenAffect:
    def get_affect(self):
        raise RuntimeError("emotion source down")


def test_failing_sources_counted_and_logged_first_and_every_hundredth(caplog):
    gen = make(
        config(memory=1.0, question=0.0001),
        memory_source=BrokenMemory(),
        emotion_source=BrokenAffect(),
        settings=Settings(DEBUG=False),
    )
    with caplog.at_level(logging.WARNING, logger=generator_module.__name__):
        for _ in range(200):
            thought = run(gen.generate_thought())
            assert thought.thought_type is ThoughtType.QUESTION  # no memory: skipped
    errors = gen.status()["source_errors"]
    # the anti-repeat guard rests memory after three questions, so fewer tries
    assert errors["emotion"] == 200 and errors["memory"] >= 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    emotion_lines = [r for r in warnings if r.getMessage().startswith("emotion")]
    memory_lines = [r for r in warnings if r.getMessage().startswith("memory")]
    assert [r.args[1] for r in emotion_lines] == [1, 100, 200]
    expected = [c for c in range(1, errors["memory"] + 1) if c == 1 or c % 100 == 0]
    assert [r.args[1] for r in memory_lines] == expected
    assert all("down" not in r.getMessage() for r in warnings)  # type name only


def test_failing_source_raises_under_debug():
    loud_memory = make(
        config(memory=1.0),
        memory_source=BrokenMemory(),
        settings=Settings(DEBUG=True),
    )
    with pytest.raises(RuntimeError):
        run(loud_memory.generate_thought())
    assert loud_memory.status()["source_errors"] == {"memory": 1}

    loud_affect = make(
        config(question=1.0),
        emotion_source=BrokenAffect(),
        settings=Settings(DEBUG=True),
    )
    with pytest.raises(RuntimeError):
        run(loud_affect.generate_thought())
    assert loud_affect.status()["source_errors"] == {"emotion": 1}
