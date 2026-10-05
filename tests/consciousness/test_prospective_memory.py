"""Prospective simulation: rate limit, internal candidates, configured time."""

import os
import random
from datetime import datetime, timedelta, timezone

import pytest

from mtc.consciousness.enhanced_global_workspace import WorkspaceCandidateSource
from mtc.consciousness.prospective_memory import (
    ProspectiveContext,
    ProspectiveSimulator,
    SIMULATION_TYPES,
)
from mtc.core.config import Settings

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return lambda **overrides: Settings(**overrides)


def messages(*texts, speaker="Alice"):
    return [{"content": t, "speaker": speaker, "role": "user"} for t in texts]


def context(**kwargs):
    kwargs.setdefault("recent_messages", messages("We talked about gardening plans"))
    kwargs.setdefault("now", T0)
    return ProspectiveContext(**kwargs)


def simulator(cfg, clock=None, seed=1, **kwargs):
    return ProspectiveSimulator(
        clock=clock or Clock(),
        settings=cfg(),
        rng=random.Random(seed),
        **kwargs,
    )


def test_rate_limited(cfg):
    clock = Clock()
    sim = simulator(cfg, clock, min_interval_s=120.0)
    assert len(sim.maybe_generate(context())) == 1
    clock.now = T0 + timedelta(seconds=119)
    assert sim.maybe_generate(context()) == []
    clock.now = T0 + timedelta(seconds=121)
    sim._simulation_history.clear()  # an exact repeat is a separate guard
    assert len(sim.maybe_generate(context(now=clock.now))) == 1


def test_no_context_no_candidates(cfg):
    sim = simulator(cfg)
    empty = ProspectiveContext(recent_messages=[], now=T0)
    assert sim.maybe_generate(empty) == []


def test_candidates_are_internal_source(cfg):
    candidate = simulator(cfg).maybe_generate(context())[0]
    assert candidate.source is WorkspaceCandidateSource.INTERNAL
    assert candidate.source is not WorkspaceCandidateSource.CTM
    assert candidate.content_type == "prospective_simulation"
    assert candidate.source_module == "prospective_simulator"
    assert candidate.metadata["sim_type"] in SIMULATION_TYPES
    assert 0.0 < candidate.activation_level <= 0.85
    assert candidate.summary and candidate.content == candidate.summary
    assert candidate.timestamp == T0.timestamp()


def test_confidence_is_deterministic(cfg):
    first = simulator(cfg, seed=1).maybe_generate(context())[0]
    second = simulator(cfg, seed=1).maybe_generate(context())[0]
    assert first.activation_level == second.activation_level
    assert first.summary == second.summary


def test_speaker_label_passes_through_unchanged(cfg):
    creator = cfg().CREATOR_NAME
    texts = {}
    for label in (creator, "Guest", "unlisted-speaker"):
        sim = simulator(cfg)
        sim._recent_topics.append("gardening")
        content = sim._empathic_content(
            "hello there",
            context(recent_messages=messages("hello there", speaker=label)),
        )
        assert label in content
        texts[label] = content.replace(label, "<speaker>")
    assert len(set(texts.values())) == 1  # no label gets different treatment


def test_time_of_day_uses_configured_timezone(cfg):
    sim = ProspectiveSimulator(clock=Clock(), settings=cfg(TIMEZONE="Asia/Tokyo"))
    # 12:00 UTC is 21:00 in Tokyo: evening; the weights follow the local hour.
    assert sim.period_for(T0) == "evening"
    utc_sim = ProspectiveSimulator(clock=Clock(), settings=cfg(TIMEZONE="UTC"))
    assert utc_sim.period_for(T0) == "afternoon"
    berlin = ProspectiveSimulator(clock=Clock(), settings=cfg(TIMEZONE="Europe/Berlin"))
    assert berlin.period_for(T0) == "afternoon"
    night = ProspectiveSimulator(
        clock=Clock(), settings=cfg(TIMEZONE="UTC", WAKE_HOUR=13, SLEEP_HOUR=4)
    )
    assert night.period_for(T0) == "night"
    naive = datetime(2000, 1, 1, 12, 0)
    assert utc_sim.period_for(naive) == "afternoon"
    morning_weights = utc_sim.type_weights(context(now=T0.replace(hour=8)))
    evening_weights = utc_sim.type_weights(context(now=T0.replace(hour=19)))
    assert morning_weights["planning"] > evening_weights["planning"]
    assert evening_weights["empathic"] > morning_weights["empathic"]


def test_goal_and_emotion_boosts(cfg):
    sim = simulator(cfg)
    plain = sim.type_weights(context())
    with_goals = sim.type_weights(context(goals=["finish the report"]))
    assert with_goals["planning"] > plain["planning"]
    feeling = sim.type_weights(
        context(recent_messages=messages("I feel worried today"))
    )
    assert feeling["empathic"] > plain["empathic"]
    custom = sim.type_weights(
        context(recent_messages=messages("such a gloomy day"), emotion_words=["gloomy"])
    )
    assert custom["empathic"] > plain["empathic"]
    ignored = sim.type_weights(
        context(recent_messages=messages("I feel worried"), emotion_words=["gloomy"])
    )
    assert ignored["empathic"] == pytest.approx(plain["empathic"])
    aroused = sim.type_weights(context(arousal=0.9))
    assert aroused["counterfactual"] > plain["counterfactual"]
    unknown = sim.type_weights(context(arousal=None))
    assert unknown == plain
    for weights in (plain, with_goals, feeling, aroused):
        assert sum(weights.values()) == pytest.approx(1.0)


def test_planning_confidence_rises_with_goals(cfg):
    sim = simulator(cfg)
    base = sim.confidence("planning", context())
    goal = sim.confidence("planning", context(goals=["a goal"]))
    assert goal == pytest.approx(base + 0.15)
    assert sim.confidence("empathic", context(arousal=0.5)) > sim.confidence(
        "empathic", context()
    )
    many = context(recent_messages=messages(*["m"] * 20))
    assert sim.confidence("temporal", many) <= 0.85


def test_templates_are_neutral(cfg):
    banned = {
        "bedtime",
        "bath",
        "dinner",
        "homework",
        "mommy",
        "family",
        "kids",
        "child",
        "dad",
        "father",
        "mother",
    }
    seen = []
    for period in ("morning", "afternoon", "evening", "night"):
        for seed in range(30):
            sim = simulator(cfg, seed=seed)
            sim._recent_topics.extend(["gardening", "planning"])
            when = {"morning": 8, "afternoon": 13, "evening": 19, "night": 2}[period]
            ctx = context(
                goals=["finish the report"],
                now=T0.replace(hour=when),
            )
            for sim_type in SIMULATION_TYPES:
                content, prompt = sim.build(sim_type, ctx)
                seen.append(content + " " + prompt)
    words = {w.strip(".,!?'\"():;").lower() for text in seen for w in text.split()}
    assert not words & banned


def test_state_dict_round_trip(cfg):
    clock = Clock()
    sim = simulator(cfg, clock)
    sim.update_topics("gardening schedules planning thoroughly")
    sim.set_goals(["goal one", "goal two"])
    assert sim.maybe_generate(context())
    state = sim.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    assert state["scalars"]["last_simulation_time"] == T0.timestamp()
    restored = simulator(cfg, clock)
    restored.from_state_dict(state)
    assert restored.to_state_dict() == state
    assert restored._recent_topics.maxlen == ProspectiveSimulator.MAX_RECENT_TOPICS
    assert (
        restored._simulation_history.maxlen
        == ProspectiveSimulator.MAX_SIMULATION_HISTORY
    )
    assert restored.maybe_generate(context()) == []  # rate limit survives a restore
    fresh = simulator(cfg, clock)
    assert fresh.to_state_dict()["scalars"]["last_simulation_time"] is None
