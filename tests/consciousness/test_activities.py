"""Activities: registry, drive map, brain-only dispatchers, failure handling."""

import pytest

from mtc.brain.protocol import BrainReply, FakeBrain, NoAnswerError
from mtc.consciousness.activities import (
    Activity,
    ActivityContext,
    ActivityRegistry,
    ActivityResult,
    default_registry,
)


def ctx(drive="curiosity", **kwargs) -> ActivityContext:
    return ActivityContext(drive=drive, urgency=kwargs.pop("urgency", 0.5), **kwargs)


async def noop(context):
    return ActivityResult(activity="noop", success=True, text="x")


def test_registry_maps_drives_to_activities():
    async def recall(query, n):
        return []

    registry = default_registry(FakeBrain(), recall=recall)
    assert registry.names() == ["reflect", "write_creative", "review_memories"]
    assert [a.name for a in registry.for_drive("curiosity")] == [
        "reflect",
        "write_creative",
        "review_memories",
    ]
    assert [a.name for a in registry.for_drive("coherence")] == [
        "reflect",
        "review_memories",
    ]
    for drive in ("attention_budget", "social_connection", "safety"):
        assert registry.for_drive(drive) == []


def test_registry_register_unregister_and_duplicates():
    registry = ActivityRegistry()
    activity = Activity("a", "curiosity", "does a", noop)
    registry.register(activity)
    with pytest.raises(ValueError):
        registry.register(Activity("a", "coherence", "again", noop))
    assert registry.get("a") is activity
    assert registry.unregister("a") is True
    assert registry.unregister("a") is False
    assert registry.names() == []


def test_review_memories_registered_only_with_recall():
    without = default_registry(FakeBrain())
    assert "review_memories" not in without.names()

    async def recall(query, n):
        return []

    with_recall = default_registry(FakeBrain(), recall=recall)
    assert "review_memories" in with_recall.names()


@pytest.mark.asyncio
async def test_reflect_uses_brain_and_strips_reasoning():
    brain = FakeBrain(replies=["<think>plan the answer</think>A quiet thought."])
    registry = default_registry(brain)
    reflect = registry.get("reflect")
    result = await reflect.run(ctx(topic="the shape of attention"))
    assert result.success is True
    assert result.text == "A quiet thought."
    assert result.activity == "reflect"
    prompt = brain.calls[0]["prompt"]
    assert "the shape of attention" in prompt
    assert "no addressee" in prompt


@pytest.mark.asyncio
async def test_reflect_without_topic_names_the_drive_and_lists_goals():
    brain = FakeBrain(replies=["ok"])
    reflect = default_registry(brain).get("reflect")
    await reflect.run(ctx(drive="coherence", goals=["map the territory"]))
    prompt = brain.calls[0]["prompt"]
    assert "pull toward coherence" in prompt
    assert "map the territory" in prompt


@pytest.mark.asyncio
async def test_write_creative_uses_brain():
    brain = FakeBrain(replies=["A short poem."])
    result = await default_registry(brain).get("write_creative").run(ctx(topic="tides"))
    assert result.success is True
    assert result.text == "A short poem."
    assert "tides" in brain.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_review_memories_recalls_then_reflects():
    asked = []

    async def recall(query, n):
        asked.append((query, n))
        return ["first memory", "  ", "second memory"]

    brain = FakeBrain(replies=["Both circle the same idea."])
    result = (
        await default_registry(brain, recall=recall)
        .get("review_memories")
        .run(ctx(topic="rivers"))
    )
    assert asked == [("rivers", 3)]
    assert result.success is True
    assert result.metadata["memory_count"] == 2
    prompt = brain.calls[0]["prompt"]
    assert "first memory" in prompt and "second memory" in prompt


@pytest.mark.asyncio
async def test_review_memories_without_memories_fails_without_calling_brain():
    async def recall(query, n):
        return []

    brain = FakeBrain(replies=["unused"])
    result = (
        await default_registry(brain, recall=recall).get("review_memories").run(ctx())
    )
    assert result.success is False
    assert result.error == "no_memories"
    assert result.text == ""
    assert brain.calls == []


@pytest.mark.asyncio
async def test_review_memories_recall_failure_is_reported():
    async def recall(query, n):
        raise RuntimeError("store down")

    result = (
        await default_registry(FakeBrain(), recall=recall)
        .get("review_memories")
        .run(ctx())
    )
    assert result.success is False
    assert result.error == "recall_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["reflect", "write_creative"])
@pytest.mark.parametrize(
    "script",
    [
        [NoAnswerError("none")],
        [""],
        ["<think>only reasoning</think>"],
        [BrainReply(text="   ")],
    ],
)
async def test_no_answer_is_failure_without_text(name, script):
    brain = FakeBrain(replies=script)
    result = await default_registry(brain).get(name).run(ctx())
    assert result.success is False
    assert result.error == "no_answer"
    assert result.text == ""


@pytest.mark.asyncio
async def test_unexpected_brain_error_is_a_failure_not_a_raise():
    brain = FakeBrain(replies=[RuntimeError("boom")])
    result = await default_registry(brain).get("reflect").run(ctx())
    assert result.success is False
    assert result.error == "brain_error"
    assert result.text == ""


def test_unknown_drive_has_no_activity():
    registry = default_registry(FakeBrain())
    assert registry.for_drive("expression") == []
    assert registry.for_drive("") == []


def test_excluded_activities_absent():
    async def recall(query, n):
        return []

    registry = default_registry(FakeBrain(), recall=recall)
    banned = (
        "art",
        "garden",
        "webcam",
        "look",
        "book",
        "read",
        "shelf",
        "code",
        "notif",
        "meditate",
        "svg",
    )
    for name in registry.names():
        assert not any(word in name for word in banned), name
    for activity in registry.all():
        assert not any(word in activity.description.lower() for word in banned)
