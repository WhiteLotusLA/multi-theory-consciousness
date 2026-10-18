"""Learning system: gating, periodic learning, consolidation, store seam."""

import logging
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mtc.consciousness.learning import (
    AffectModulatedRate,
    BehavioralAdapter,
    ExperienceTracker,
    LearningSystem,
    PatternLearner,
)
from mtc.consciousness.pad_affect import PADState
from mtc.memory.inmemory import InMemoryLearningStore

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def settings(enabled=True, rate=0.02, novelty=0.7, zone="UTC"):
    return SimpleNamespace(
        LEARNING_ENABLED=enabled,
        LEARNING_RATE=rate,
        NOVELTY_THRESHOLD=novelty,
        TIMEZONE=zone,
    )


def make_system(store=None, **kwargs):
    return LearningSystem(store=store, settings=settings(**kwargs), clock=lambda: T0)


async def turn(system, speaker="Alice", **kwargs):
    defaults = dict(
        emotion_before={"valence": 0.0, "arousal": 0.5, "label": "calm"},
        emotion_after={"valence": 0.3, "arousal": 0.5, "label": "calm"},
        conversation_depth=2,
    )
    defaults.update(kwargs)
    return await system.record_turn(speaker, "hello", "a reply", **defaults)


def test_package_exports_the_five_classes():
    assert {
        AffectModulatedRate,
        ExperienceTracker,
        PatternLearner,
        BehavioralAdapter,
        LearningSystem,
    }
    import mtc.consciousness.learning as pkg

    assert set(pkg.__all__) == {
        "AffectModulatedRate",
        "ExperienceTracker",
        "PatternLearner",
        "BehavioralAdapter",
        "LearningSystem",
    }
    doc = pkg.__doc__ or ""
    assert "learning_experiences" not in doc and "learning_beliefs" not in doc


def test_none_store_means_in_memory_and_no_client():
    code = (
        "import asyncio, sys\n"
        "from mtc.consciousness.learning import LearningSystem\n"
        "from mtc.memory.inmemory import InMemoryLearningStore\n"
        "s = LearningSystem()\n"
        "asyncio.run(s.initialize())\n"
        "assert isinstance(s.store, InMemoryLearningStore)\n"
        "bad = [m for m in ('motor', 'pymongo', 'asyncpg', 'redis', 'neo4j',"
        " 'qdrant_client') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**__import__("os").environ, "PYPHI_WELCOME_OFF": "yes"},
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.asyncio
async def test_learning_disabled_records_nothing():
    system = make_system(enabled=False)
    await system.initialize()
    assert await turn(system) is None
    assert await system.store.recent_experiences(10) == []
    assert await system.consolidate() == {"experiences": 0, "beliefs_changed": 0}
    modulation = system.get_modulation(speaker="Alice")
    assert modulation.directives == [] and modulation.temperature_delta == 0.0
    assert system.get_belief_summary() == ""


@pytest.mark.asyncio
async def test_periodic_learning_every_ten_turns():
    system = make_system()
    await system.initialize()
    for _ in range(9):
        await turn(system)
    assert system.learner.beliefs == {}
    await turn(system)
    assert "speaker_Alice_positive" in system.learner.beliefs
    count = system.learner.beliefs["speaker_Alice_positive"].evidence_count
    for _ in range(9):
        await turn(system)
    assert system.learner.beliefs["speaker_Alice_positive"].evidence_count == count
    await turn(system)
    assert system.learner.beliefs["speaker_Alice_positive"].evidence_count == count + 1


@pytest.mark.asyncio
async def test_consolidate_runs_learning_pass():
    system = make_system()
    await system.initialize()
    for _ in range(4):
        await turn(system)
    assert system.learner.beliefs == {}
    result = await system.consolidate()
    assert result["experiences"] == 4 and result["beliefs_changed"] > 0
    assert system.learner.beliefs

    empty = make_system()
    await empty.initialize()
    assert await empty.consolidate() == {"experiences": 0, "beliefs_changed": 0}


@pytest.mark.asyncio
async def test_attach_store_reloads_beliefs():
    source = make_system()
    await source.initialize()
    for _ in range(4):
        await turn(source)
    await source.consolidate()
    shared = source.store
    assert await shared.load_beliefs(10)

    fresh = make_system()
    await fresh.initialize()
    assert fresh.learner.beliefs == {}
    fresh.attach_store(shared)
    assert fresh.store is shared
    await fresh.initialize()
    assert set(fresh.learner.beliefs) == set(source.learner.beliefs)
    assert len(fresh.tracker.get_recent(10)) == 4
    # turns recorded now go to the attached store
    await turn(fresh)
    assert len(await shared.recent_experiences(10)) == 5


@pytest.mark.asyncio
async def test_modulation_uses_relevant_beliefs():
    system = make_system()
    await system.initialize()
    for _ in range(4):
        await turn(system, emotion_after={"valence": 0.5, "label": "joy"})
    await system.consolidate()
    mod = system.get_modulation(speaker="Alice")
    assert mod.source_beliefs
    other = system.get_modulation(speaker="Guest")
    assert not any("Alice" in b for b in other.source_beliefs)


@pytest.mark.asyncio
async def test_pad_rate_feeds_learning_passes():
    low, high = make_system(rate=0.02), make_system(rate=0.02)
    await low.initialize()
    await high.initialize()
    calm = PADState(pleasure=0.0, arousal=0.0, dominance=0.0)
    peak = PADState(pleasure=1.0, arousal=0.6, dominance=1.0)
    for _ in range(4):
        await turn(low, pad=calm)
        await turn(high, pad=peak)
    await low.consolidate()
    await high.consolidate()
    key = "speaker_Alice_positive"
    assert high.learner.beliefs[key].confidence > low.learner.beliefs[key].confidence


@pytest.mark.asyncio
async def test_belief_summary_is_neutral_third_person():
    system = make_system()
    await system.initialize()
    assert system.get_belief_summary() == ""
    for _ in range(4):
        await turn(system)
    for _ in range(30):
        await system.consolidate()
    summary = system.get_belief_summary()
    assert summary.startswith("Learned patterns:")
    assert summary.isascii()
    lowered = summary.lower()
    assert " i " not in f" {lowered} " and "my " not in lowered


@pytest.mark.asyncio
async def test_store_failure_is_logged_not_raised(caplog):
    class Broken(InMemoryLearningStore):
        async def add_experience(self, doc):
            raise RuntimeError("down")

        async def upsert_belief(self, belief_id, doc):
            raise RuntimeError("down")

        async def recent_experiences(self, limit):
            raise RuntimeError("down")

        async def load_beliefs(self, limit):
            raise RuntimeError("down")

    system = make_system(store=Broken())
    with caplog.at_level(logging.WARNING):
        await system.initialize()
        for _ in range(4):
            await turn(system)
        await system.consolidate()
    assert system.learner.beliefs
    assert "down" in caplog.text


@pytest.mark.asyncio
async def test_stats_report_settings():
    system = make_system(rate=0.05, novelty=0.9)
    await system.initialize()
    stats = system.get_stats()
    assert stats["enabled"] is True
    assert stats["learning_rate"] == 0.05 and stats["novelty_threshold"] == 0.9
    assert "experiences" in stats and "beliefs" in stats


@pytest.mark.asyncio
async def test_state_argument_is_mapped_to_metrics():
    state = SimpleNamespace(
        ignition_events=3,
        beautiful_loop_quality=0.4,
        architectural_phi=None,
        higher_order_thoughts=[1, 2],
        prediction_error=0.1,
        pad_pleasure=0.2,
        pad_arousal=0.6,
        pad_dominance=0.5,
        broadcast_coverage=0.7,
        integration_level=0.6,
        epistemic_depth=2,
        binding_quality=0.3,
        protoself_stability=0.8,
        feeling_of_knowing=0.5,
        homeostatic_urgency=("curiosity", 0.4),
    )
    system = make_system()
    await system.initialize()
    experience = await turn(system, state=state)
    assert experience.gwt_ignition_count == 3.0
    assert experience.architectural_phi is None
    assert experience.metacognitive_events == 2.0
    assert experience.homeostatic_urgency == pytest.approx(0.4)
    # the state's PAD fields drive the learning rate when no pad is given
    assert system._pad is not None and system._pad.arousal == 0.6


def pad_state(pleasure, arousal=0.5, dominance=0.5, **extra):
    """A cycle state as the post-turn step hands it over (PAD fields set)."""
    values = dict(
        ignition_events=None,
        beautiful_loop_quality=None,
        architectural_phi=None,
        higher_order_thoughts=[],
        prediction_error=None,
        pad_pleasure=pleasure,
        pad_arousal=arousal,
        pad_dominance=dominance,
    )
    values.update(extra)
    return SimpleNamespace(**values)


async def state_turn(system, state, conversation_id="c1", depth=1, **kwargs):
    """A turn in the post-turn shape: no emotion mapping, only the state."""
    return await system.record_turn(
        speaker="Alice",
        message="a synthetic message",
        response=kwargs.pop("response", "a synthetic reply"),
        state=state,
        conversation_depth=depth,
        conversation_id=conversation_id,
        topics_discussed=kwargs.pop("topics_discussed", ["music"]),
        homeostatic_drive="curiosity",
        attention_focus="music",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_consecutive_pad_states_give_shift_and_outcome():
    system = make_system()
    await system.initialize()
    first = await state_turn(system, pad_state(0.0))
    second = await state_turn(system, pad_state(0.4))
    assert first.valence_shift is None and first.outcome is None
    assert second.valence_shift == pytest.approx(0.4)
    assert second.arousal_shift == pytest.approx(0.0)
    assert second.outcome == "repair"
    # another conversation starts without a shift; the first one goes on
    other = await state_turn(system, pad_state(0.9), conversation_id="c2")
    assert other.valence_shift is None
    third = await state_turn(system, pad_state(0.1))
    assert third.valence_shift == pytest.approx(-0.3)
    # a turn without a complete PAD records no shift and breaks the chain
    gap = await state_turn(system, pad_state(None))
    assert gap.valence_shift is None and gap.outcome is None
    after_gap = await state_turn(system, pad_state(0.5))
    assert after_gap.valence_shift is None
    # an explicit pad counts as the turn's PAD as well
    explicit = await state_turn(
        system, None, pad=PADState(pleasure=0.7, arousal=0.5, dominance=0.5)
    )
    assert explicit.valence_shift == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_state_only_turns_form_beliefs():
    system = make_system()
    await system.initialize()
    for index in range(1, 13):
        await state_turn(system, pad_state(0.1 * (index % 5)), depth=index)
        if index == 10:
            assert system.learner.beliefs, "the tenth turn's pass formed no belief"
    recent = system.tracker.get_recent(12)
    assert sum(e.valence_shift is not None for e in recent) == 11
    assert all(e.outcome is not None for e in recent[1:])


@pytest.mark.asyncio
async def test_previous_pad_is_kept_for_at_most_a_hundred_conversations():
    system = make_system()
    await system.initialize()
    for index in range(101):
        await state_turn(system, pad_state(0.0), conversation_id=f"c{index}")
    # the oldest conversation was forgotten, the newest hundred are kept
    oldest = await state_turn(system, pad_state(0.4), conversation_id="c0")
    newest = await state_turn(system, pad_state(0.4), conversation_id="c100")
    assert oldest.valence_shift is None
    assert newest.valence_shift == pytest.approx(0.4)
    assert len(system._previous_affect) <= LearningSystem.PREVIOUS_AFFECT_CONVERSATIONS
    assert LearningSystem.PREVIOUS_AFFECT_CONVERSATIONS == 100


async def deepening_turn(system, topic="music"):
    return await system.record_turn(
        "Alice",
        "a synthetic message",
        "a synthetic reply " * 40,
        emotion_before={"valence": 0.0, "arousal": 0.5},
        emotion_after={"valence": 0.3, "arousal": 0.5},
        conversation_depth=6,
        topic=topic,
    )


@pytest.mark.asyncio
async def test_learning_pass_proposes_goals():
    from mtc.consciousness.intrinsic_goals import (
        MIN_PROPOSAL_STRENGTH,
        IntrinsicGoalSystem,
    )

    goals = IntrinsicGoalSystem(clock=lambda: T0)
    system = LearningSystem(settings=settings(rate=0.2), clock=lambda: T0, goals=goals)
    await system.initialize()
    for _ in range(4):
        await deepening_turn(system)
    assert goals.get_top_goals() == []
    for _ in range(20):
        await system.consolidate()
        belief = system.learner.beliefs.get("topic_music_deepens")
        if (
            belief is not None
            and system.learner.strength(belief) > MIN_PROPOSAL_STRENGTH
        ):
            break
    top = goals.get_top_goals()
    assert top, "a learning pass that changed beliefs proposed no goal"
    assert any(goal.related_topics == ["music"] for goal in top)
    assert all("Alice" not in goal.description for goal in top)

    # the periodic pass every ten turns proposes as well
    periodic_goals = IntrinsicGoalSystem(clock=lambda: T0)
    periodic = LearningSystem(
        settings=settings(rate=0.5), clock=lambda: T0, goals=periodic_goals
    )
    await periodic.initialize()
    for _ in range(LearningSystem.LEARNING_INTERVAL):
        await deepening_turn(periodic, topic="gardens")
    assert any(g.related_topics == ["gardens"] for g in periodic_goals.get_top_goals())


class BrokenGoals:
    def __init__(self):
        self.calls = 0

    async def generate_goals(self, beliefs):
        self.calls += 1
        raise RuntimeError("goals down")


@pytest.mark.asyncio
async def test_goal_generation_failure_is_counted_and_raised_under_debug(caplog):
    broken = BrokenGoals()
    soft = LearningSystem(settings=settings(), clock=lambda: T0, goals=broken)
    await soft.initialize()
    for _ in range(4):
        await turn(soft)
    with caplog.at_level(logging.WARNING):
        result = await soft.consolidate()
    assert result["beliefs_changed"] > 0  # the pass itself still counts
    assert broken.calls == 1
    assert soft.get_stats()["failures"] == {"goal_generation": 1}
    assert "RuntimeError" in caplog.text and "goals down" not in caplog.text

    loud_settings = settings()
    loud_settings.DEBUG = True
    loud = LearningSystem(settings=loud_settings, clock=lambda: T0, goals=BrokenGoals())
    await loud.initialize()
    for _ in range(4):
        await turn(loud)
    with pytest.raises(RuntimeError):
        await loud.consolidate()

    # no goals attached: nothing is called and nothing is counted
    plain = make_system()
    await plain.initialize()
    for _ in range(4):
        await turn(plain)
    await plain.consolidate()
    assert plain.get_stats()["failures"] == {}


class RecordingGoals:
    def __init__(self):
        self.calls = []

    async def generate_goals(self, beliefs):
        self.calls.append(list(beliefs))
        return 0


@pytest.mark.asyncio
async def test_pass_that_changed_no_beliefs_proposes_no_goal():
    goals = RecordingGoals()
    system = LearningSystem(settings=settings(), clock=lambda: T0, goals=goals)
    await system.initialize()
    await turn(system)  # one experience is below the evidence a belief needs
    result = await system.consolidate()
    assert result["experiences"] == 1 and result["beliefs_changed"] == 0
    assert goals.calls == []

    for _ in range(3):
        await turn(system)
    result = await system.consolidate()
    assert result["beliefs_changed"] > 0
    assert len(goals.calls) == 1  # the guard opens once a belief changed
