"""Behavioral adapter: caps, strength weighting, drives, neutral templates."""

import re
from datetime import datetime, timezone

import pytest

from mtc.consciousness.active_inference import HomeostaticDrives
from mtc.consciousness.learning.behavioral_adapter import (
    DEFAULT_TEMPLATES,
    BehavioralAdapter,
    BehavioralModulation,
)
from mtc.consciousness.learning.pattern_learner import LearnedBelief

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def belief(
    belief_id="b",
    category="topic",
    pattern=None,
    effect=None,
    confidence=0.8,
    last_reinforced=T0,
) -> LearnedBelief:
    return LearnedBelief(
        belief_id=belief_id,
        category=category,
        description="d",
        pattern=pattern or {},
        confidence=confidence,
        evidence_count=5,
        counter_evidence=0,
        first_observed=T0.isoformat(),
        last_reinforced=last_reinforced.isoformat(),
        behavioral_effect=effect or {},
    )


def adapter(**kwargs) -> BehavioralAdapter:
    return BehavioralAdapter(clock=lambda: T0, **kwargs)


def test_empty_beliefs_give_neutral_modulation():
    mod = adapter().adapt([])
    assert isinstance(mod, BehavioralModulation)
    assert mod.temperature_delta == 0.0 and mod.directives == []
    assert mod.drive_adjustments == {} and mod.unapplied_effects == {}


def test_modulation_caps():
    beliefs = [
        belief(
            f"b{i}",
            "speaker",
            {"speaker": "Alice", "dominant_outcome": "deepening"},
            {"warmth": 5.0, "curiosity": 5.0, "social_connection": -5.0},
            confidence=0.9,
        )
        for i in range(6)
    ]
    mod = adapter().adapt(beliefs)
    assert mod.temperature_delta == pytest.approx(0.15)
    assert mod.drive_adjustments["curiosity"] == pytest.approx(0.2)
    assert mod.drive_adjustments["social_connection"] == pytest.approx(-0.2)
    assert len(mod.directives) == 3


def test_strength_weighted_accumulation():
    strong = belief("s", confidence=0.8, effect={"curiosity": 0.1, "verbosity": 0.1})
    weak = belief("w", confidence=0.2, effect={"curiosity": 0.1, "verbosity": 0.1})
    mod = adapter().adapt([strong, weak])
    assert mod.drive_adjustments["curiosity"] == pytest.approx(0.1 * 0.8 + 0.1 * 0.2)
    assert mod.verbosity_delta == pytest.approx(0.1 * 0.8 + 0.1 * 0.2)
    # a belief below the strength floor contributes nothing
    faint = belief("f", confidence=0.05, effect={"curiosity": 0.1})
    assert adapter().adapt([faint]).drive_adjustments == {}
    # recency decays the weight
    old = belief("o", confidence=0.8, effect={"curiosity": 0.1})
    later = BehavioralAdapter(
        clock=lambda: datetime(2000, 1, 8, 12, 0, tzinfo=timezone.utc)
    )
    assert later.adapt([old]).drive_adjustments["curiosity"] == pytest.approx(0.04)


def test_unknown_drive_effects_reported_not_applied():
    drives = HomeostaticDrives()
    mod = adapter().adapt(
        [
            belief(
                "x",
                effect={
                    "curiosity": 0.1,
                    "social_connection": 0.05,
                    "patience": 0.1,
                    "playfulness": 0.1,
                    "openness": 0.05,
                },
                confidence=1.0,
            )
        ],
        drives=drives,
    )
    assert set(mod.drive_adjustments) == {"curiosity", "social_connection"}
    assert set(mod.unapplied_effects) == {"patience", "playfulness", "openness"}
    assert mod.unapplied_effects["patience"] == pytest.approx(0.1)

    class NoSocial:
        def get_drive_state(self):
            return {"curiosity": {}}

    mod = adapter().adapt(
        [belief("y", effect={"curiosity": 0.1, "social_connection": 0.1})],
        drives=NoSocial(),
    )
    assert set(mod.drive_adjustments) == {"curiosity"}
    assert "social_connection" in mod.unapplied_effects


def test_directives_come_from_the_template_table():
    deep = belief(
        "sp",
        "speaker",
        {"speaker": "Alice", "dominant_outcome": "deepening"},
        confidence=0.9,
    )
    topic = belief("topic_music_deepens", "topic", {"topic": "music"}, confidence=0.5)
    mod = adapter().adapt([deep, topic])
    assert mod.directives[0] == DEFAULT_TEMPLATES["speaker_deepening"].format(
        speaker="Alice"
    )
    assert any("music" in d for d in mod.directives)
    custom = adapter(templates={"speaker_deepening": "Custom line for {speaker}."})
    assert custom.adapt([deep]).directives == ["Custom line for Alice."]
    # a broken template is skipped, never raised
    broken = adapter(templates={"speaker_deepening": "Needs {missing_key}."})
    assert broken.adapt([deep]).directives == []
    # a belief below the directive strength floor yields no directive
    soft = belief(
        "sp2",
        "speaker",
        {"speaker": "Alice", "dominant_outcome": "deepening"},
        confidence=0.2,
    )
    assert adapter().adapt([soft]).directives == []


def test_templates_are_neutral_third_person():
    for key, text in DEFAULT_TEMPLATES.items():
        assert text.isascii(), key
        assert not re.search(r"\b(you|your|i|my|me|we)\b", text, re.I), key
    assert "Conversations with {speaker} have tended to go deep." in set(
        DEFAULT_TEMPLATES.values()
    )


def test_topic_bias_and_summary():
    mod = adapter().adapt([belief("t", "topic", {"topic": "music"}, confidence=0.5)])
    assert mod.topic_bias == {"music": pytest.approx(0.1)}
    summary = adapter().summary(mod)
    assert isinstance(summary, str)
    assert adapter().summary(BehavioralModulation()) == "no learning modulation"
