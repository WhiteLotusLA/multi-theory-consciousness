"""Autonomous life: selection, step, drives, goals, pacing, turns, rest, share."""

import asyncio
import random
import re
from datetime import datetime, timezone

import numpy as np
import pytest

from mtc.brain.protocol import FakeBrain, NoAnswerError
from mtc.consciousness.activities import (
    Activity,
    ActivityContext,
    ActivityRegistry,
    ActivityResult,
    default_registry,
)
from mtc.consciousness.active_inference import ActiveInferenceModule
from mtc.consciousness.autonomous_life import (
    ActivitySelector,
    AutonomousLife,
    AutonomousLifeConfig,
    BrainActivitySelector,
    DriveBasedSelector,
    Selection,
    most_urgent_drive,
)
from mtc.consciousness.enhanced_global_workspace import WorkspaceCandidateSource
from mtc.consciousness.intrinsic_goals import (
    GoalSource,
    IntrinsicGoalSystem,
    new_goal,
)
from mtc.core.config import Settings

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


class FakeDrives:
    def __init__(self, urgencies, free_energy=0.5):
        self.state = {
            name: {"urgency": urgency, "current_level": 0.3, "optimal_level": 0.7}
            for name, urgency in urgencies.items()
        }
        self.free_energy = free_energy

    def get_drive_state(self):
        return self.state

    def get_free_energy(self):
        return self.free_energy


class FakeSource:
    """Anything exposing homeostatic_drives and update_homeostasis."""

    def __init__(self, urgencies=None, free_energy=0.5):
        self.homeostatic_drives = FakeDrives(
            urgencies if urgencies is not None else {"curiosity": 0.6}, free_energy
        )
        self.updates = []

    async def update_homeostasis(self, activity):
        self.updates.append(activity)
        return {}


def stub(name, drive, text="done", calls=None, also=()):
    async def run(context: ActivityContext) -> ActivityResult:
        if calls is not None:
            calls.append((name, context))
        if text is None:
            return ActivityResult(activity=name, success=False, error="no_answer")
        return ActivityResult(activity=name, success=True, text=text)

    return Activity(name, drive, f"does {name}", run, also_drives=tuple(also))


def registry_of(*activities) -> ActivityRegistry:
    registry = ActivityRegistry()
    for activity in activities:
        registry.register(activity)
    return registry


def make_life(source=None, registry=None, brain=None, **kwargs) -> AutonomousLife:
    config = kwargs.pop("config", None) or AutonomousLifeConfig(
        exploration_probability=0.0
    )
    brain = brain or FakeBrain()
    return AutonomousLife(
        brain=brain,
        drives_source=source or FakeSource(),
        registry=registry if registry is not None else default_registry(brain),
        config=config,
        rng=kwargs.pop("rng", random.Random(1)),
        clock=kwargs.pop("clock", Clock()),
        # DEBUG is pinned off unless a test asks for it, so a handler that
        # raises under DEBUG never depends on the ambient environment.
        settings=kwargs.pop("settings", Settings(DEBUG=False)),
        **kwargs,
    )


# -- selection ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_selects_most_urgent_drive():
    calls = []
    registry = registry_of(
        stub("a", "curiosity", calls=calls), stub("b", "coherence", calls=calls)
    )
    source = FakeSource({"curiosity": 0.4, "coherence": 0.9, "safety": 0.95})
    life = make_life(source, registry)
    result = await life.step()
    # safety is the most urgent but has no activity, so coherence wins
    assert result.activity == "b"
    assert calls[0][1].drive == "coherence"
    assert calls[0][1].urgency == pytest.approx(0.9)
    assert most_urgent_drive(source.homeostatic_drives.state, registry, 0.3) == (
        "coherence",
        0.9,
    )


@pytest.mark.asyncio
async def test_drive_based_selector_takes_the_first_registered_option():
    options = [stub("first", "curiosity"), stub("second", "curiosity")]
    selector = DriveBasedSelector()
    assert isinstance(selector, ActivitySelector)
    picked = await selector.select(ActivityContext("curiosity", 0.5), options)
    assert picked.activity.name == "first"
    assert picked.source == "drive"


@pytest.mark.asyncio
async def test_below_threshold_selects_nothing():
    calls = []
    registry = registry_of(stub("a", "curiosity", calls=calls))
    source = FakeSource({"curiosity": 0.3})  # equal to the threshold: not above it
    life = make_life(source, registry)
    assert await life.step() is None
    assert calls == []
    assert life.get_status()["last_skip"] == "no_urgent_drive"
    assert life.steps_run == 0


def test_most_urgent_drive_ignores_missing_and_non_finite_urgency():
    registry = registry_of(stub("a", "curiosity"), stub("b", "coherence"))
    state = {
        "curiosity": {"urgency": float("nan")},
        "coherence": {"urgency": None},
        "other": {},
    }
    assert most_urgent_drive(state, registry, 0.0) is None
    assert most_urgent_drive({}, registry, 0.0) is None


@pytest.mark.asyncio
async def test_brain_choice_parsed_to_known_option():
    options = [stub("reflect", "curiosity"), stub("write_creative", "curiosity")]
    brain = FakeBrain(replies=["<think>hmm</think>I would pick write_creative now."])
    selector = BrainActivitySelector(brain)
    picked = await selector.select(ActivityContext("curiosity", 0.5), options)
    assert picked.activity.name == "write_creative"
    assert picked.source == "brain"
    prompt = brain.calls[0]["prompt"]
    assert "reflect" in prompt and "write_creative" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        "something else entirely",
        "reflect or write_creative",
        "reflective",
        "",
        NoAnswerError("none"),
    ],
)
async def test_unparseable_choice_falls_back_and_says_so(reply):
    options = [stub("reflect", "curiosity"), stub("write_creative", "curiosity")]
    selector = BrainActivitySelector(FakeBrain(replies=[reply]))
    picked = await selector.select(ActivityContext("curiosity", 0.5), options)
    assert picked.activity.name == "reflect"
    assert picked.source == "drive_fallback"


@pytest.mark.asyncio
async def test_single_option_needs_no_brain_call():
    brain = FakeBrain()
    picked = await BrainActivitySelector(brain).select(
        ActivityContext("curiosity", 0.5), [stub("only", "curiosity")]
    )
    assert picked.activity.name == "only"
    assert brain.calls == []


@pytest.mark.asyncio
async def test_exploration_roll_is_seeded():
    async def run_with(seed):
        registry = registry_of(
            stub("a", "curiosity"), stub("b", "coherence"), stub("c", "safety")
        )
        life = make_life(
            FakeSource({"curiosity": 0.0}),
            registry,
            config=AutonomousLifeConfig(exploration_probability=1.0),
            rng=random.Random(seed),
        )
        names = []
        for _ in range(6):
            names.append((await life.step()).activity)
        return names, [h["source"] for h in life.history]

    first = await run_with(7)
    second = await run_with(7)
    assert first == second
    assert set(first[1]) == {"exploration"}
    assert len(set(first[0])) > 1  # the roll really varies the choice


@pytest.mark.asyncio
async def test_exploration_probability_zero_never_explores():
    calls = []
    registry = registry_of(stub("a", "curiosity", calls=calls))
    life = make_life(FakeSource({"curiosity": 0.0}), registry)
    for _ in range(5):
        assert await life.step() is None
    assert calls == []


# -- the step ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_enters_sink_as_internal_candidate():
    seen = []

    async def sink(candidate):
        seen.append(candidate)

    registry = registry_of(stub("reflect", "curiosity", text="A  thought\nabout x."))
    life = make_life(registry=registry, experience_sink=sink)
    result = await life.step()
    assert result.success is True
    assert len(seen) == 1
    candidate = seen[0]
    assert candidate.source is WorkspaceCandidateSource.INTERNAL
    assert candidate.source_module == "autonomous_life"
    assert candidate.content_type == "autonomous_activity"
    assert candidate.content == "A  thought\nabout x."
    assert candidate.summary == "A thought about x."
    assert candidate.metadata["activity"] == "reflect"
    assert candidate.timestamp == T0.timestamp()
    assert len(life.recent_candidates) == 0


@pytest.mark.asyncio
async def test_without_a_sink_candidates_are_kept_bounded():
    registry = registry_of(stub("reflect", "curiosity"))
    life = make_life(
        registry=registry,
        config=AutonomousLifeConfig(exploration_probability=0.0, history_size=2),
    )
    for _ in range(4):
        await life.step()
    assert len(life.recent_candidates) == 2
    assert life.get_status()["has_sink"] is False


@pytest.mark.asyncio
async def test_sink_failure_is_counted_not_raised():
    async def sink(candidate):
        raise RuntimeError("down")

    life = make_life(
        registry=registry_of(stub("reflect", "curiosity")), experience_sink=sink
    )
    result = await life.step()
    assert result.success is True
    assert life.get_status()["sink_errors"] == 1


@pytest.mark.asyncio
async def test_drive_update_uses_mapped_activity_type():
    source = FakeSource({"curiosity": 0.6})
    registry = registry_of(
        stub("reflect", "curiosity"),
        stub("write_creative", "curiosity"),
        stub("unmapped", "coherence"),
    )
    life = make_life(source, registry, selector=_Pick("write_creative"))
    await life.step()
    assert source.updates == [{"type": "exploring", "intensity": 0.6}]

    source.updates.clear()
    life2 = make_life(source, registry, selector=_Pick("reflect"))
    await life2.step()
    assert source.updates == [{"type": "sensemaking", "intensity": 0.6}]

    source.updates.clear()
    source.homeostatic_drives = FakeDrives({"coherence": 0.8})
    await make_life(source, registry).step()
    assert source.updates == []  # no mapping, no credit


@pytest.mark.asyncio
async def test_activity_mapping_without_intensity_still_credits_the_drive():
    source = FakeSource()
    config = AutonomousLifeConfig(
        exploration_probability=0.0,
        activity_types={"reflect": {"type": "sensemaking"}},
    )
    registry = registry_of(stub("reflect", "curiosity"))
    life = make_life(source, registry, config=config, settings=Settings(DEBUG=True))
    await life.step()
    assert source.updates == [{"type": "sensemaking"}]
    assert life.get_status()["step_failures"] == {}

    # A non-finite or non-numeric intensity is left out the same way.
    for bad in (float("nan"), "high", "0.6", True, None):
        source.updates.clear()
        config.activity_types["reflect"]["intensity"] = bad
        await life.step()
        assert source.updates == [{"type": "sensemaking"}]

    # A real drive model credits the drive at its default event intensity.
    module = ActiveInferenceModule()
    drives = module.homeostatic_drives
    drives.drives["coherence"].current_level = 0.1
    before = drives.drives["coherence"].current_level
    config.activity_types["reflect"] = {"type": "sensemaking"}
    real = make_life(module, registry_of(stub("reflect", "coherence")), config=config)
    await real.step()
    assert drives.drives["coherence"].current_level > before
    assert real.get_status()["step_failures"] == {}


@pytest.mark.asyncio
async def test_finite_numpy_intensity_is_kept():
    source = FakeSource()
    config = AutonomousLifeConfig(
        exploration_probability=0.0,
        activity_types={"reflect": {"type": "sensemaking"}},
    )
    registry = registry_of(stub("reflect", "curiosity"))
    life = make_life(source, registry, config=config, settings=Settings(DEBUG=True))
    for good, expected in (
        (np.float32(0.25), 0.25),
        (np.float64(0.7), 0.7),
        (np.int64(1), 1.0),
    ):
        source.updates.clear()
        config.activity_types["reflect"]["intensity"] = good
        await life.step()
        assert len(source.updates) == 1
        assert source.updates[0]["type"] == "sensemaking"
        assert source.updates[0]["intensity"] == pytest.approx(expected)

    # A non-finite numpy number and a numpy boolean are still left out.
    for bad in (np.float64("nan"), np.float32("inf"), np.bool_(True)):
        source.updates.clear()
        config.activity_types["reflect"]["intensity"] = bad
        await life.step()
        assert source.updates == [{"type": "sensemaking"}]
    assert life.get_status()["step_failures"] == {}


@pytest.mark.asyncio
async def test_drive_update_failure_is_counted_and_raised_under_debug():
    class _Failing(FakeSource):
        async def update_homeostasis(self, activity):
            raise RuntimeError("drive store down")

    registry = registry_of(stub("reflect", "curiosity"))
    quiet = make_life(_Failing(), registry, settings=Settings(DEBUG=False))
    result = await quiet.step()
    # The step completes as before; the failed credit is counted.
    assert result.success is True
    assert quiet.completed == 1
    assert len(quiet.history) == 1
    assert quiet.get_status()["step_failures"] == {"drive_update": 1}
    await quiet.step()
    assert quiet.get_status()["step_failures"] == {"drive_update": 2}

    loud = make_life(_Failing(), registry, settings=Settings(DEBUG=True))
    with pytest.raises(RuntimeError, match="drive store down"):
        await loud.step()
    assert loud.get_status()["step_failures"] == {"drive_update": 1}


@pytest.mark.asyncio
async def test_selection_failure_is_counted_and_raised_under_debug():
    class _Broken(FakeBrain):
        async def think(self, *args, **kwargs):
            raise ConnectionError("brain unavailable")

    options = [stub("first", "curiosity"), stub("second", "curiosity")]
    context = ActivityContext("curiosity", 0.5)
    quiet = BrainActivitySelector(_Broken(), settings=Settings(DEBUG=False))
    picked = await quiet.select(context, options)
    assert (picked.activity.name, picked.source) == ("first", "drive_fallback")
    assert quiet.failures == {"selection": 1}
    life = make_life(registry=registry_of(*options), selector=quiet)
    assert life.get_status()["step_failures"] == {"selection": 1}

    loud = BrainActivitySelector(_Broken(), settings=Settings(DEBUG=True))
    with pytest.raises(ConnectionError):
        await loud.select(context, options)


@pytest.mark.asyncio
async def test_activity_type_mapping_is_configurable():
    source = FakeSource()
    config = AutonomousLifeConfig(
        exploration_probability=0.0,
        activity_types={"reflect": {"type": "understanding", "intensity": 0.2}},
    )
    life = make_life(source, registry_of(stub("reflect", "curiosity")), config=config)
    await life.step()
    assert source.updates == [{"type": "understanding", "intensity": 0.2}]


class _Pick:
    """A selector that returns the named option."""

    def __init__(self, name):
        self.name = name

    async def select(self, context, options):
        return Selection(next(a for a in options if a.name == self.name), "test")


@pytest.mark.asyncio
async def test_drive_credit_moves_real_drives_with_the_f4_vocabulary():
    module = ActiveInferenceModule()
    drives = module.homeostatic_drives
    drives.drives["coherence"].current_level = 0.1
    before = drives.drives["coherence"].current_level
    registry = registry_of(stub("reflect", "coherence"))
    life = make_life(module, registry)
    result = await life.step()
    assert result.success is True
    assert drives.drives["coherence"].current_level > before


@pytest.mark.asyncio
async def test_goals_record_activity_when_attached():
    goals = IntrinsicGoalSystem(clock=Clock())
    goal = new_goal(
        description="understand how attention works",
        source=GoalSource.LEARNING,
        category="understand",
        priority=0.6,
        emotional_investment=0.5,
        now=T0,
    )
    assert await goals.adopt_goal(goal)
    calls = []
    registry = registry_of(stub("reflect", "curiosity", calls=calls))
    life = make_life(registry=registry, goals=goals)
    await life.step()
    assert goal.progress == pytest.approx(0.1)
    assert goal.steps_taken == 1
    (stored,) = await goals.store.load_goals(["active"])
    assert stored["steps_taken"] == 1
    context = calls[0][1]
    assert context.topic == "understand how attention works"
    assert context.goals == ["understand how attention works"]


@pytest.mark.asyncio
async def test_goal_without_matching_category_is_untouched():
    goals = IntrinsicGoalSystem(clock=Clock())
    goal = new_goal(
        description="connect with others",
        source=GoalSource.LEARNING,
        category="connect",
        priority=0.6,
        emotional_investment=0.5,
        now=T0,
    )
    await goals.adopt_goal(goal)
    life = make_life(registry=registry_of(stub("reflect", "curiosity")), goals=goals)
    await life.step()
    assert goal.progress == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [NoAnswerError("none"), ""])
async def test_no_answer_records_no_activity_result(reply):
    brain = FakeBrain(replies=[reply])
    source = FakeSource({"curiosity": 0.8})
    goals = IntrinsicGoalSystem(clock=Clock())
    goal = new_goal(
        description="understand the loop",
        source=GoalSource.LEARNING,
        category="understand",
        priority=0.6,
        emotional_investment=0.5,
        now=T0,
    )
    await goals.adopt_goal(goal)
    seen, shared = [], []

    async def sink(candidate):
        seen.append(candidate)

    async def share(result):
        shared.append(result)

    registry = registry_of(default_registry(brain).get("reflect"))
    life = make_life(
        source,
        registry,
        brain=brain,
        goals=goals,
        experience_sink=sink,
        share=share,
    )
    result = await life.step()
    assert result.success is False
    assert result.error == "no_answer"
    assert result.text == ""
    assert seen == [] and shared == []
    assert source.updates == []
    assert goal.progress == 0.0 and goal.steps_taken == 0
    assert len(life.history) == 0 and len(life.recent_candidates) == 0
    assert len(brain.calls) == 1  # no share decision either
    status = life.get_status()
    assert status["failures"] == {"no_answer": 1}
    assert status["failed"] == 1 and status["completed"] == 0
    assert status["steps_run"] == 1


@pytest.mark.asyncio
async def test_activity_that_raises_or_returns_empty_text_is_a_failure():
    async def boom(context):
        raise RuntimeError("x")

    async def empty(context):
        return ActivityResult(activity="empty", success=True, text="  ")

    registry = registry_of(
        Activity("boom", "curiosity", "d", boom),
        Activity("empty", "coherence", "d", empty),
    )
    source = FakeSource({"curiosity": 0.9, "coherence": 0.5})
    life = make_life(source, registry)
    first = await life.step()
    source.homeostatic_drives.state["curiosity"]["urgency"] = 0.0
    second = await life.step()
    assert first.error == "activity_error" and second.error == "empty_text"
    assert life.failures == {"activity_error": 1, "empty_text": 1}
    assert source.updates == []


@pytest.mark.asyncio
async def test_history_bounded():
    registry = registry_of(stub("reflect", "curiosity", text="x" * 500))
    life = make_life(
        registry=registry,
        config=AutonomousLifeConfig(exploration_probability=0.0, history_size=3),
    )
    for _ in range(7):
        await life.step()
    assert len(life.history) == 3
    assert life.completed == 7
    assert len(life.history[-1]["summary"]) == 100
    assert life.history[-1]["activity"] == "reflect"


@pytest.mark.asyncio
async def test_missing_homeostatic_drives_skips():
    class Bare:
        homeostatic_drives = None

    life = make_life(Bare(), registry_of(stub("reflect", "curiosity")))
    assert await life.step() is None
    assert life.get_status()["last_skip"] == "no_drives"


# -- pacing, turns, rest -------------------------------------------------------


def test_tempo_scales_with_free_energy():
    config = AutonomousLifeConfig(min_interval_s=60.0, max_interval_s=1800.0)
    life = make_life(config=config)
    third = (1800.0 - 60.0) / 3.0
    for _ in range(25):
        high = life.next_interval_s(3.0)
        medium = life.next_interval_s(1.5)
        low = life.next_interval_s(0.5)
        assert 60.0 <= high <= 60.0 + third
        assert 60.0 + third <= medium <= 60.0 + 2 * third
        assert 60.0 + 2 * third <= low <= 1800.0
        assert high <= medium <= low


def test_tempo_without_a_reading_is_the_maximum_and_uses_the_last_reading():
    life = make_life(config=AutonomousLifeConfig(max_interval_s=900.0))
    assert life.next_interval_s() == 900.0
    assert life.next_interval_s(float("nan")) == 900.0
    life.last_free_energy = 5.0
    assert life.next_interval_s() < 900.0 / 3 + 60.0


def test_tempo_is_seeded():
    a = make_life(rng=random.Random(5)).next_interval_s(3.0)
    b = make_life(rng=random.Random(5)).next_interval_s(3.0)
    assert a == b


def test_config_validation():
    with pytest.raises(ValueError):
        AutonomousLifeConfig(min_interval_s=10.0, max_interval_s=5.0)
    with pytest.raises(ValueError):
        AutonomousLifeConfig(exploration_probability=1.5)
    with pytest.raises(ValueError):
        AutonomousLifeConfig(history_size=0)
    with pytest.raises(ValueError):
        AutonomousLifeConfig(high_free_energy=0.5, medium_free_energy=1.0)


@pytest.mark.asyncio
async def test_pauses_during_turn():
    calls = []
    life = make_life(registry=registry_of(stub("reflect", "curiosity", calls=calls)))
    life.notify_turn_started()
    assert await life.step() is None
    assert calls == []
    assert life.get_status()["turn_in_progress"] is True
    assert life.get_status()["last_skip"] == "turn_in_progress"
    life.notify_turn_finished()
    result = await life.step()
    assert result is not None and result.success
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_nested_turns_are_counted():
    calls = []
    life = make_life(registry=registry_of(stub("reflect", "curiosity", calls=calls)))
    life.notify_turn_started()
    life.notify_turn_started()
    life.notify_turn_finished()
    assert await life.step() is None
    life.notify_turn_finished()
    life.notify_turn_finished()  # an extra finish never goes negative
    assert await life.step() is not None


@pytest.mark.asyncio
async def test_turn_opening_during_selection_stops_the_step():
    calls = []
    holder = {}

    class Opens:
        async def select(self, context, options):
            holder["life"].notify_turn_started()
            return await DriveBasedSelector().select(context, options)

    life = make_life(
        registry=registry_of(stub("reflect", "curiosity", calls=calls)),
        selector=Opens(),
    )
    holder["life"] = life
    assert await life.step() is None
    assert calls == []


@pytest.mark.asyncio
async def test_turn_opening_during_the_activity_skips_the_share_decision():
    holder = {}
    shared = []

    async def run(context):
        holder["life"].notify_turn_started()
        return ActivityResult(activity="reflect", success=True, text="t")

    async def share(result):
        shared.append(result)

    brain = FakeBrain(replies=["SHARE"])
    life = make_life(
        registry=registry_of(Activity("reflect", "curiosity", "d", run)),
        brain=brain,
        share=share,
    )
    holder["life"] = life
    await life.step()
    assert brain.calls == [] and shared == []


@pytest.mark.asyncio
async def test_turn_opening_during_the_share_call_skips_the_share():
    holder = {}
    shared = []

    class TurnOpeningBrain(FakeBrain):
        async def think(self, prompt, *args, **kwargs):
            reply = await super().think(prompt, *args, **kwargs)
            holder["life"].notify_turn_started()
            return reply

    async def share(result):
        shared.append(result)

    brain = TurnOpeningBrain(replies=["SHARE"])
    life = make_life(
        registry=registry_of(stub("reflect", "curiosity")),
        brain=brain,
        share=share,
    )
    holder["life"] = life
    await life.step()
    assert len(brain.calls) == 1
    assert shared == []


@pytest.mark.asyncio
async def test_no_activity_during_rest():
    calls = []
    resting = {"now": True}
    life = make_life(
        registry=registry_of(stub("reflect", "curiosity", calls=calls)),
        is_rest_time=lambda: resting["now"],
    )
    assert await life.step() is None
    assert calls == []
    assert life.get_status()["last_skip"] == "rest"
    resting["now"] = False
    assert await life.step() is not None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_failing_rest_check_means_no_activity():
    def broken():
        raise RuntimeError("x")

    calls = []
    life = make_life(
        registry=registry_of(stub("reflect", "curiosity", calls=calls)),
        is_rest_time=broken,
    )
    assert await life.step() is None
    assert calls == []
    assert life.get_status()["last_skip"] == "rest_check_failed"


# -- sharing, consolidation, persistence -----------------------------------------


@pytest.mark.asyncio
async def test_share_only_with_callback():
    brain = FakeBrain(replies=["SHARE"])
    life = make_life(registry=registry_of(stub("reflect", "curiosity")), brain=brain)
    await life.step()
    assert brain.calls == []  # no callback: no share decision at all

    shared = []

    async def share(result):
        shared.append(result)

    for reply, expected in [
        ("share", ["share"]),
        ("DISCUSS", ["discuss"]),
        ("<think>keep it?</think>I would SHARE this.", ["share"]),
        ("KEEP", []),
        ("maybe", []),
        ("KEEP or SHARE", []),
        ("sharing", []),
        ("", []),
    ]:
        shared.clear()
        brain = FakeBrain(replies=[reply])
        life = make_life(
            registry=registry_of(stub("reflect", "curiosity", text="body")),
            brain=brain,
            share=share,
        )
        await life.step()
        assert [r.metadata.get("share_choice") for r in shared] == expected, reply
        prompt = brain.calls[0]["prompt"]
        assert "KEEP, SHARE or DISCUSS" in prompt


@pytest.mark.asyncio
async def test_share_no_answer_keeps_private():
    shared = []

    async def share(result):
        shared.append(result)

    brain = FakeBrain(replies=[NoAnswerError("none")])
    life = make_life(
        registry=registry_of(stub("reflect", "curiosity")), brain=brain, share=share
    )
    await life.step()
    assert shared == []


@pytest.mark.asyncio
async def test_share_prompt_is_neutral():
    brain = FakeBrain(replies=["KEEP"])

    async def share(result):
        return None

    life = make_life(
        registry=registry_of(stub("reflect", "curiosity")), brain=brain, share=share
    )
    await life.step()
    prompt = brain.calls[0]["prompt"]
    allowed = {"KEEP", "SHARE", "DISCUSS"}
    for sentence in re.split(r"[.\n:?]", prompt):
        words = sentence.split()[1:]
        assert not [w for w in words if w[:1].isupper() and w not in allowed], sentence


@pytest.mark.asyncio
async def test_consolidation_every_n_steps():
    counts = []

    async def consolidate():
        counts.append(1)
        return 4

    life = make_life(
        registry=registry_of(stub("reflect", "curiosity")),
        consolidate_thoughts=consolidate,
        config=AutonomousLifeConfig(exploration_probability=0.0, consolidate_every=3),
    )
    for _ in range(7):
        await life.step()
    assert len(counts) == 2
    assert life.get_status()["last_consolidation_step"] == 6
    assert life.last_consolidated == 4


@pytest.mark.asyncio
async def test_consolidation_failure_is_not_fatal_and_zero_disables():
    async def consolidate():
        raise RuntimeError("x")

    life = make_life(
        registry=registry_of(stub("reflect", "curiosity")),
        consolidate_thoughts=consolidate,
        config=AutonomousLifeConfig(exploration_probability=0.0, consolidate_every=1),
    )
    assert (await life.step()).success is True

    ran = []

    async def counted():
        ran.append(1)
        return 0

    off = make_life(
        registry=registry_of(stub("reflect", "curiosity")),
        consolidate_thoughts=counted,
        config=AutonomousLifeConfig(exploration_probability=0.0, consolidate_every=0),
    )
    for _ in range(3):
        await off.step()
    assert ran == []


@pytest.mark.asyncio
async def test_state_dict_round_trip():
    config = AutonomousLifeConfig(exploration_probability=0.0, history_size=3)
    registry = registry_of(
        stub("reflect", "curiosity"), stub("bad", "coherence", text=None)
    )
    source = FakeSource({"curiosity": 0.9, "coherence": 0.0}, free_energy=1.4)
    life = make_life(source, registry, config=config)
    for _ in range(5):
        await life.step()
    source.homeostatic_drives.state["curiosity"]["urgency"] = 0.0
    source.homeostatic_drives.state["coherence"]["urgency"] = 0.9
    await life.step()

    state = life.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    restored = make_life(FakeSource(), registry, config=config)
    restored.from_state_dict(state)
    assert restored.steps_run == 6
    assert restored.completed == 5
    assert restored.failures == {"no_answer": 1}
    assert restored.last_activity == "bad"
    assert restored.last_activity_time == T0
    assert restored.last_activity_time.tzinfo is not None
    assert restored.last_free_energy == pytest.approx(1.4)
    assert list(restored.history) == list(life.history)
    assert restored.history.maxlen == 3
    assert restored.to_state_dict() == state


@pytest.mark.asyncio
async def test_empty_state_restores_defaults():
    life = make_life()
    life.from_state_dict({})
    assert life.steps_run == 0 and life.last_activity_time is None
    assert life.history.maxlen == life.config.history_size


@pytest.mark.asyncio
async def test_stop_is_clean():
    life = make_life()
    assert life.get_status()["running"] is False
    life.start()
    task = life._task
    life.start()  # a second start does not add a task
    assert life._task is task
    assert life.get_status()["running"] is True
    await life.stop()
    assert task.done()
    assert life.get_status()["running"] is False
    await life.stop()  # stopping twice is fine


@pytest.mark.asyncio
async def test_run_loop_steps_then_waits_by_free_energy():
    waits = []
    reached = asyncio.Event()

    async def fake_sleep(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            reached.set()
            await asyncio.Event().wait()  # park until stop() cancels

    calls = []
    source = FakeSource({"curiosity": 0.6}, free_energy=3.0)
    life = AutonomousLife(
        brain=FakeBrain(),
        drives_source=source,
        registry=registry_of(stub("reflect", "curiosity", calls=calls)),
        config=AutonomousLifeConfig(
            min_interval_s=60.0, max_interval_s=1800.0, exploration_probability=0.0
        ),
        rng=random.Random(1),
        clock=Clock(),
        sleep=fake_sleep,
    )
    life.start()
    await reached.wait()
    await life.stop()
    assert len(calls) == 2
    assert all(60.0 <= w <= 640.0 for w in waits)


@pytest.mark.asyncio
async def test_run_loop_survives_a_failing_step():
    waits = []
    reached = asyncio.Event()

    class Exploding:
        @property
        def homeostatic_drives(self):
            raise RuntimeError("x")

    async def fake_sleep(seconds):
        waits.append(seconds)
        reached.set()
        await asyncio.Event().wait()

    life = AutonomousLife(
        brain=FakeBrain(),
        drives_source=Exploding(),
        registry=registry_of(stub("reflect", "curiosity")),
        config=AutonomousLifeConfig(min_interval_s=7.0, exploration_probability=0.0),
        sleep=fake_sleep,
        settings=Settings(DEBUG=False),
    )
    life.start()
    await reached.wait()
    await life.stop()
    assert waits == [7.0]


@pytest.mark.asyncio
async def test_tick_is_step():
    calls = []
    life = make_life(registry=registry_of(stub("reflect", "curiosity", calls=calls)))
    result = await life.tick()
    assert result.success is True and len(calls) == 1


@pytest.mark.asyncio
async def test_default_registry_runs_through_the_brain_end_to_end():
    brain = FakeBrain(replies=["<think>x</think>A private thought."])
    seen = []

    async def sink(candidate):
        seen.append(candidate)

    source = FakeSource({"coherence": 0.7})
    life = make_life(source, default_registry(brain), brain=brain, experience_sink=sink)
    result = await life.step()
    assert result.activity == "reflect"
    assert seen[0].content == "A private thought."
    assert source.updates == [{"type": "sensemaking", "intensity": 0.6}]


def test_loop_has_no_name_comparisons_or_network_imports():
    import inspect

    import mtc.consciousness.activities as acts
    import mtc.consciousness.autonomous_life as life

    for module in (acts, life):
        source = inspect.getsource(module)
        for banned in ("motor", "pymongo", "asyncpg", "redis", "neo4j", "qdrant"):
            assert f"import {banned}" not in source
        assert "CREATOR_NAME" not in source
