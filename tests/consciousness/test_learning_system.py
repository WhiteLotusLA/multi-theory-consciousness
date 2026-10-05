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
    other = system.get_modulation(speaker="Bob")
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
