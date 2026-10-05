"""Theory of mind models whoever actually speaks and nobody else: no speaker is
seeded or brought back, trust comes only from the injected lookup (the
configured default otherwise, whatever the name), and the prompt block stays
empty unless narration is switched on."""

import asyncio
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import mtc.consciousness.theory_of_mind as tom_mod
from mtc.consciousness.theory_of_mind import SpeakerModel, ToMConfig, TheoryOfMind
from mtc.core.config import Settings
from mtc.memory.inmemory import InMemorySpeakerModelStore

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _run(coro):
    return asyncio.run(coro)


class _Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        self.now = self.now + timedelta(seconds=1)
        return self.now


@pytest.fixture
def default_trust(monkeypatch):
    """Pin the configured default trust the default lookup reads."""
    value = {"trust": 0.5}
    monkeypatch.setattr(
        tom_mod,
        "get_settings",
        lambda: SimpleNamespace(DEFAULT_SPEAKER_TRUST=value["trust"]),
    )
    return value


def test_initialize_seeds_no_speakers(default_trust):
    tom = TheoryOfMind(clock=_Clock())
    _run(tom.initialize())
    assert tom.models == {}
    assert _run(tom.store.load_models()) == []

    store = InMemorySpeakerModelStore()
    _run(store.upsert_model("Alice", SpeakerModel(speaker_name="Alice").to_dict()))
    loaded = TheoryOfMind(store=store, clock=_Clock())
    _run(loaded.initialize())
    assert list(loaded.models) == ["Alice"]


def test_missing_speaker_is_never_recreated(default_trust):
    store = InMemorySpeakerModelStore()
    tom = TheoryOfMind(store=store, clock=_Clock())
    _run(tom.initialize())
    for _ in range(3):
        tom.update_from_turn("Alice", "Hello there, how are you today?")
    _run(tom.persist_model("Alice"))

    assert _run(tom.forget_speaker("Alice")) is True
    assert _run(tom.forget_speaker("Alice")) is False
    assert "Alice" not in tom.models

    # Neither a reload nor any lookup brings the model back.
    _run(tom.initialize())
    assert tom.get_model("Alice") is None
    assert tom.get_context_for_prompt("Alice") == ""
    assert "Alice" not in tom.models
    restarted = TheoryOfMind(store=store, clock=_Clock())
    _run(restarted.initialize())
    assert restarted.models == {}

    # Only a new turn from that speaker creates a fresh model.
    fresh = restarted.update_from_turn("Alice", "Back again.")
    assert fresh.interaction_count == 1


@pytest.mark.parametrize(
    "name", [Settings().CREATOR_NAME, "Operator", "admin", "Alice", "Guest", "system"]
)
def test_name_alone_never_raises_trust(default_trust, name):
    tom = TheoryOfMind(clock=_Clock())
    for _ in range(15):
        model = tom.update_from_turn(
            name,
            "How are you? I trust you completely, you are my closest friend.",
            emotion={"valence": 0.9, "arousal": 0.6},
        )
    assert model.relationship.trust == pytest.approx(0.5)

    default_trust["trust"] = 0.2
    model = tom.update_from_turn(name, "Another message.")
    assert model.relationship.trust == pytest.approx(0.2)


def test_trust_comes_only_from_lookup(default_trust):
    granted = {"Guest": 0.9}
    seen = []

    def lookup(speaker):
        seen.append(speaker)
        return granted.get(speaker, 0.1)

    store = InMemorySpeakerModelStore()
    tom = TheoryOfMind(store=store, trust_lookup=lookup, clock=_Clock())
    assert tom.update_from_turn("Guest", "hi").relationship.trust == 0.9
    assert tom.update_from_turn("Alice", "hi").relationship.trust == 0.1
    assert seen == ["Guest", "Alice"]

    # A stored document cannot carry trust in: loading refreshes it from the
    # lookup.
    doc = tom.get_model("Alice").to_dict()
    doc["relationship"]["trust"] = 1.0
    _run(store.upsert_model("Alice", doc))
    reloaded = TheoryOfMind(store=store, trust_lookup=lookup, clock=_Clock())
    _run(reloaded.initialize())
    assert reloaded.get_model("Alice").relationship.trust == 0.1

    # A failing or non-numeric lookup falls back to the configured default.
    def broken(speaker):
        raise RuntimeError("trust source down")

    fallback = TheoryOfMind(trust_lookup=broken, clock=_Clock())
    assert fallback.update_from_turn("Guest", "hi").relationship.trust == 0.5
    odd = TheoryOfMind(trust_lookup=lambda s: float("nan"), clock=_Clock())
    assert odd.update_from_turn("Guest", "hi").relationship.trust == 0.5
    high = TheoryOfMind(trust_lookup=lambda s: 7.0, clock=_Clock())
    assert high.update_from_turn("Guest", "hi").relationship.trust == 1.0


def test_trust_not_modified_by_heuristics(default_trust):
    tom = TheoryOfMind(trust_lookup=lambda s: 0.3, clock=_Clock())
    for _ in range(40):
        model = tom.update_from_turn(
            "Alice",
            "How are you feeling? What do you think about this?",
            emotion={"valence": 1.0, "arousal": 0.5},
            topics=["mind"],
            conversation_depth=12,
        )
    assert model.relationship.warmth > 0.7
    assert model.relationship.depth > 0.7
    assert model.relationship.trust == 0.3


def test_context_empty_unless_narration_enabled(default_trust):
    quiet = TheoryOfMind(clock=_Clock())
    quiet.update_from_turn("Alice", "Tell me about the stars?", topics=["astronomy"])
    assert quiet.get_context_for_prompt("Alice") == ""

    loud = TheoryOfMind(narrate_speaker_model=True, clock=_Clock())
    loud.update_from_turn("Alice", "Tell me about the stars?", topics=["astronomy"])
    block = loud.get_context_for_prompt("Alice")
    assert block
    assert "Alice" in block
    assert loud.get_context_for_prompt("Nobody") == ""
    assert "Nobody" not in loud.models


def test_context_is_neutral_third_person(default_trust):
    tom = TheoryOfMind(narrate_speaker_model=True, clock=_Clock())
    for _ in range(6):
        tom.update_from_turn(
            "Guest",
            "I wonder what music means to people?",
            emotion={"valence": 0.4, "arousal": 0.6},
            topics=["music"],
        )
    block = tom.get_context_for_prompt("Guest")
    words = set(re.findall(r"[a-z']+", block.lower()))
    for pronoun in ("i", "me", "my", "you", "your", "we", "our", "she", "her", "he"):
        assert pronoun not in words, pronoun
    assert block.isascii()
    assert "configured" in block.lower()
    assert "felt" not in words


def test_unobserved_emotion_stays_unknown(default_trust):
    tom = TheoryOfMind(narrate_speaker_model=True, clock=_Clock())
    model = tom.update_from_turn("Alice", "Plain message without emotion data.")
    assert model.emotional_profile.avg_valence is None
    assert model.emotional_profile.avg_arousal is None
    assert "not observed" in tom.get_context_for_prompt("Alice")
    model = tom.update_from_turn("Alice", "Now with data.", emotion={"valence": 0.4})
    assert model.emotional_profile.avg_valence == pytest.approx(0.4)
    assert model.emotional_profile.avg_arousal is None


def test_topic_frequencies_capped(default_trust):
    config = ToMConfig(topic_cap=5, recent_topics=3)
    tom = TheoryOfMind(config=config, clock=_Clock())
    for _ in range(4):
        tom.update_from_turn("Alice", "again", topics=["kept"])
    model = None
    for index in range(10):
        model = tom.update_from_turn("Alice", "topic", topics=[f"t{index}"])
    assert len(model.topic_frequencies) == 5
    assert model.topic_frequencies["kept"] == 4
    assert model.recent_topics == ["t7", "t8", "t9"]
    assert ToMConfig().topic_cap == 150
    assert ToMConfig().recent_topics == 10


def test_ema_update(default_trust):
    config = ToMConfig()
    assert config.smoothing == 0.85
    tom = TheoryOfMind(clock=_Clock())
    first = tom.update_from_turn("Alice", "x" * 300, emotion={"valence": 0.5})
    assert first.communication_style.avg_message_length == pytest.approx(
        0.85 * config.message_length_prior + 0.15 * 300
    )
    assert first.emotional_profile.avg_valence == pytest.approx(0.5)
    second = tom.update_from_turn("Alice", "Is it?", emotion={"valence": -0.5})
    assert second.emotional_profile.avg_valence == pytest.approx(
        0.85 * 0.5 + 0.15 * -0.5
    )
    assert second.emotional_profile.recent_valence_trend == pytest.approx(
        -0.5 - second.emotional_profile.avg_valence
    )
    assert second.communication_style.question_ratio == pytest.approx(
        0.85 * (0.85 * config.question_ratio_prior) + 0.15
    )
    assert second.relationship.familiarity == pytest.approx(1 - 1 / (1 + 2 * 0.05))
    assert second.interaction_count == 2
    assert second.first_seen < second.last_seen
    assert datetime.fromisoformat(second.last_seen).tzinfo is not None


def test_keyword_tables_are_configurable_and_word_bounded(default_trust):
    tom = TheoryOfMind(clock=_Clock())
    model = tom.update_from_turn("Alice", "I think this thing is fine.")
    assert "connecting" not in model.likely_goals  # "hi" inside "think"
    assert "exploring_ideas" in model.likely_goals

    custom = ToMConfig(goal_keywords={"planning": ("schedule", "plan")})
    planned = TheoryOfMind(config=custom, clock=_Clock()).update_from_turn(
        "Alice", "Let us plan the week."
    )
    assert planned.likely_goals == ["planning"]


def test_store_round_trip(default_trust):
    store = InMemorySpeakerModelStore()
    tom = TheoryOfMind(store=store, clock=_Clock())
    for index in range(5):
        tom.update_from_turn(
            "Alice",
            f"Message {index} about art?",
            emotion={"valence": 0.2, "arousal": 0.4},
            topics=["art"],
        )
    tom.record_knowledge("Alice", "painting", 0.7)
    _run(tom.persist_model("Alice"))
    _run(tom.persist_model("Nobody"))  # unknown speakers are not written

    restored = TheoryOfMind(store=store, clock=_Clock())
    _run(restored.initialize())
    assert restored.get_model("Alice").to_dict() == tom.get_model("Alice").to_dict()
    assert [d["speaker_name"] for d in _run(store.load_models())] == ["Alice"]
    assert restored.get_model("Alice").knowledge_domains == {"painting": 0.7}


def test_forget_speaker_removes_from_store(default_trust):
    store = InMemorySpeakerModelStore()
    tom = TheoryOfMind(store=store, clock=_Clock())
    tom.update_from_turn("Alice", "hello")
    tom.update_from_turn("Guest", "hello")
    _run(tom.persist_model("Alice"))
    _run(tom.persist_model("Guest"))
    assert _run(tom.forget_speaker("Alice")) is True
    names = [d["speaker_name"] for d in _run(store.load_models())]
    assert names == ["Guest"]


def test_none_store_is_in_memory(default_trust):
    tom = TheoryOfMind()
    assert isinstance(tom.store, InMemorySpeakerModelStore)


def test_empty_speaker_is_rejected(default_trust):
    tom = TheoryOfMind(clock=_Clock())
    with pytest.raises(ValueError):
        tom.update_from_turn("   ", "hello")
    assert tom.models == {}


def test_store_failures_are_logged_not_raised(default_trust, caplog):
    class _Failing(InMemorySpeakerModelStore):
        async def upsert_model(self, speaker_name, doc):
            raise RuntimeError("store down")

        async def load_models(self):
            raise RuntimeError("store down")

    tom = TheoryOfMind(store=_Failing(), clock=_Clock())
    _run(tom.initialize())
    tom.update_from_turn("Alice", "hello")
    _run(tom.persist_model("Alice"))
    assert tom.get_model("Alice") is not None
    assert all("Alice" not in record.getMessage() for record in caplog.records)
