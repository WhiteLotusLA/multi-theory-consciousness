"""Sleep scheduler dreams: context, the brain seam, trimming, storage, floor."""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from mtc.brain.protocol import BrainReply, FakeBrain, NoAnswerError
from mtc.consciousness.dream_engine import DreamEngine, dream_sentence_allowance
from mtc.consciousness.pad_affect import PADAffectModel, PADSources, PADState
from mtc.consciousness.sleep_scheduler import (
    AffectSource,
    GoalSource,
    SleepScheduler,
    SleepSchedulerConfig,
    SleepState,
    ThemeSource,
)
from mtc.memory.inmemory import InMemoryDreamStore

UTC = timezone.utc
RECORD_FIELDS = {
    "content",
    "dream_type",
    "timestamp",
    "state",
    "parameters",
    "requested_max_tokens",
    "finish_reason",
    "cap_hit",
    "trimmed_chars",
    "trim_result",
    "output_tokens",
}


# Inputs that cover every PAD dimension: a state that carries them is a
# measured reading.
MEASURED = ("somatic_valence", "protoself_arousal", "agency_score")


def measured_pad(pleasure, arousal, dominance, label):
    return PADState(
        pleasure=pleasure,
        arousal=arousal,
        dominance=dominance,
        label=label,
        measured_inputs=MEASURED,
    )


def at(hour, minute=0, zone="UTC", day=1):
    local = datetime(2000, 1, day, hour, minute, tzinfo=ZoneInfo(zone))
    return local.astimezone(UTC)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, minutes=0.0):
        self.now = self.now + timedelta(minutes=minutes)
        return self.now


class FakeStream:
    def __init__(self):
        self.calls = []

    def set_consciousness_state(self, mode):
        self.calls.append(("mode", mode))

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def inject_thought(self, text, thought_type, emotional_tone):
        self.calls.append(("inject", text, thought_type, emotional_tone))

    @property
    def injected(self):
        return [call for call in self.calls if call[0] == "inject"]


class Themes:
    def __init__(self, themes):
        self.themes = list(themes)
        self.requests = []

    def get_recent_themes(self, n):
        self.requests.append(n)
        return self.themes[:n]


class Affect:
    def __init__(self, pad):
        self.pad = pad

    def get_latest(self):
        return self.pad


@dataclass
class Goal:
    description: str


class Goals:
    def __init__(self, goals):
        self.goals = list(goals)

    def get_top_goals(self, n):
        return self.goals[:n]


class SpyEngine(DreamEngine):
    """Records the comfort bias each prompt was built with."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.biases = []

    def build_prompt(self, context, *, dream_type=None, comfort_bias=None):
        self.biases.append(comfort_bias)
        return super().build_prompt(
            context, dream_type=dream_type, comfort_bias=comfort_bias
        )


def engine(stage=7, cls=DreamEngine):
    return cls(stage=stage, rng=random.Random(3))


def make(clock, **kwargs):
    kwargs.setdefault("dream_engine", engine())
    kwargs.setdefault("dream_store", InMemoryDreamStore())
    return SleepScheduler(
        clock=clock,
        settings=SimpleNamespace(TIMEZONE="UTC", WAKE_HOUR=6, SLEEP_HOUR=22),
        **kwargs,
    )


async def enter_night(scheduler, clock):
    clock.now = at(23, 0)
    scheduler.notify_activity()
    clock.advance(5)
    assert await scheduler.tick() is SleepState.DREAMING


async def enter_nap(scheduler, clock):
    clock.now = at(12, 0)
    scheduler.notify_activity()
    clock.advance(30)
    assert await scheduler.tick() is SleepState.NAPPING


@pytest.mark.asyncio
async def test_dream_interval_at_night():
    clock = Clock(at(22, 0))
    store = InMemoryDreamStore()
    brain = FakeBrain()
    scheduler = make(clock, brain=brain, dream_store=store)
    clock.advance(5)
    assert await scheduler.tick() is SleepState.DREAMING
    assert len(await store.recent_dreams(10)) == 1
    clock.now = at(23, 0)
    await scheduler.tick()
    clock.now = at(23, 34)
    await scheduler.tick()
    assert len(await store.recent_dreams(10)) == 1
    clock.now = at(23, 35)
    await scheduler.tick()
    assert len(await store.recent_dreams(10)) == 2
    assert len(brain.calls) == 2
    dreams = await store.recent_dreams(10)
    assert [d["timestamp"] for d in dreams] == [at(23, 35), at(22, 5)]

    # Naps get the entry dream only; morning ends the night dreams.
    clock.now = at(6, 0, day=2)
    assert await scheduler.tick() is SleepState.AWAKE
    clock.advance(200)
    await scheduler.tick()
    assert len(await store.recent_dreams(10)) == 3
    assert (await store.recent_dreams(1))[0]["state"] == "napping"


@pytest.mark.asyncio
async def test_dream_context_from_themes_affect_and_goals():
    clock = Clock(at(12, 0))
    themes = Themes(["tide pools", "a chess endgame", "rain on glass", "maps"])
    pad = measured_pad(0.6, 0.8, 0.7, "Exhilarated")
    goals = Goals([Goal("understand recursion"), Goal("finish the essay")])
    assert isinstance(themes, ThemeSource)
    assert isinstance(Affect(pad), AffectSource)
    assert isinstance(goals, GoalSource)
    brain = FakeBrain()
    scheduler = make(clock, brain=brain, themes=themes, affect=Affect(pad), goals=goals)
    await enter_nap(scheduler, clock)
    prompt = brain.calls[0]["prompt"]
    assert themes.requests == [scheduler.config.context_items]
    assert "tide pools, a chess endgame, rain on glass" in prompt
    assert "maps" not in prompt
    assert "Something unresolved: understand recursion." in prompt
    assert "Emotional undertone: exhilarated." in prompt
    assert brain.calls[0]["thinking_mode"] is False

    faint = measured_pad(0.1, 0.5, 0.5, "Content")
    quiet_brain = FakeBrain()
    quiet_clock = Clock(at(12, 0))
    quiet = make(quiet_clock, brain=quiet_brain, affect=Affect(faint))
    await enter_nap(quiet, quiet_clock)
    assert "Emotional undertone" not in quiet_brain.calls[0]["prompt"]
    assert "Recent experiences" not in quiet_brain.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_failing_context_source_degrades_only_itself():
    class BrokenThemes:
        def get_recent_themes(self, n):
            raise RuntimeError("stream offline")

    clock = Clock(at(12, 0))
    brain = FakeBrain()
    scheduler = make(
        clock,
        brain=brain,
        themes=BrokenThemes(),
        goals=Goals([Goal("understand recursion")]),
    )
    await enter_nap(scheduler, clock)
    assert "Something unresolved: understand recursion." in brain.calls[0]["prompt"]
    assert scheduler.get_status()["errors"]["context"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        NoAnswerError("no answer"),
        "",
        "   ",
        BrainReply(text="<think>only reasoning here</think>"),
    ],
)
async def test_no_answer_stores_no_dream(reply):
    clock = Clock(at(12, 0))
    store = InMemoryDreamStore()
    stream = FakeStream()
    scheduler = make(
        clock, brain=FakeBrain([reply]), dream_store=store, thought_stream=stream
    )
    await enter_nap(scheduler, clock)
    assert await store.recent_dreams(10) == []
    assert stream.injected == []
    dreams = scheduler.get_status()["dreams"]
    assert dreams["attempted"] == 1
    assert dreams["no_answer"] == 1
    assert dreams["stored"] == 0
    assert scheduler.get_status()["last_dream_at"] is None


@pytest.mark.asyncio
async def test_brain_error_is_counted_not_raised():
    clock = Clock(at(12, 0))
    store = InMemoryDreamStore()
    scheduler = make(
        clock, brain=FakeBrain([RuntimeError("timeout")]), dream_store=store
    )
    await enter_nap(scheduler, clock)
    assert await store.recent_dreams(10) == []
    assert scheduler.get_status()["errors"]["dream"] == 1
    assert scheduler.get_status()["dreams"]["no_answer"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason,output_tokens",
    [("length", 40), ("max_tokens", None), (None, "requested")],
)
async def test_capped_dream_trimmed_to_sentence_end(finish_reason, output_tokens):
    cap = 500
    requested = cap + dream_sentence_allowance(cap)
    if output_tokens == "requested":
        output_tokens = requested
    text = "The hall is long. The door opens. And then the"
    kept = "The hall is long. The door opens."
    reply = BrainReply(
        text=text, finish_reason=finish_reason, output_tokens=output_tokens
    )
    clock = Clock(at(22, 0))
    store = InMemoryDreamStore()
    stream = FakeStream()
    brain = FakeBrain([reply])
    scheduler = make(clock, brain=brain, dream_store=store, thought_stream=stream)
    clock.advance(5)
    await scheduler.tick()
    assert brain.calls[0]["max_tokens"] == requested
    (record,) = await store.recent_dreams(10)
    assert record["parameters"]["max_tokens"] == cap
    assert record["requested_max_tokens"] == requested
    assert record["content"] == kept
    assert record["cap_hit"] is True
    assert record["trimmed_chars"] == len(text) - len(kept)
    assert record["trim_result"] == "trimmed"
    assert record["finish_reason"] == finish_reason
    assert record["output_tokens"] == output_tokens
    assert stream.injected[0][1] == kept


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason,output_tokens", [("stop", 12), (None, None), (None, 30)]
)
async def test_uncapped_dream_is_stored_whole(finish_reason, output_tokens):
    text = "The hall is long. And then the"
    reply = BrainReply(
        text=text, finish_reason=finish_reason, output_tokens=output_tokens
    )
    clock = Clock(at(22, 0))
    store = InMemoryDreamStore()
    scheduler = make(clock, brain=FakeBrain([reply]), dream_store=store)
    clock.advance(5)
    await scheduler.tick()
    (record,) = await store.recent_dreams(10)
    assert record["content"] == text
    assert record["cap_hit"] is False
    assert record["trimmed_chars"] == 0
    assert record["trim_result"] == "not_needed"
    assert record["output_tokens"] == output_tokens


@pytest.mark.asyncio
async def test_reasoning_stripped_from_dream():
    reply = "<think>plan the imagery first</think>The garden folds into itself."
    clock = Clock(at(12, 0))
    store = InMemoryDreamStore()
    stream = FakeStream()
    scheduler = make(
        clock, brain=FakeBrain([reply]), dream_store=store, thought_stream=stream
    )
    await enter_nap(scheduler, clock)
    (record,) = await store.recent_dreams(1)
    assert record["content"] == "The garden folds into itself."
    assert "think" not in stream.injected[0][1]


HEAVY = measured_pad(-0.8, 0.8, 0.2, "Anxious")


@pytest.mark.asyncio
async def test_gentle_floor_off_by_default():
    assert SleepSchedulerConfig().gentle_floor_enabled is False
    clock = Clock(at(12, 0))
    spy = engine(cls=SpyEngine)
    brain = FakeBrain()
    store = InMemoryDreamStore()
    scheduler = make(
        clock, brain=brain, dream_engine=spy, affect=Affect(HEAVY), dream_store=store
    )
    await enter_nap(scheduler, clock)
    assert "Emotional undertone: anxious." in brain.calls[0]["prompt"]
    assert spy.biases == [None]
    (record,) = await store.recent_dreams(1)
    assert record["parameters"]["gentle_floor"] is False


@pytest.mark.asyncio
async def test_gentle_floor_softens_heavy_valence_when_enabled():
    config = SleepSchedulerConfig(gentle_floor_enabled=True)
    clock = Clock(at(12, 0))
    spy = engine(cls=SpyEngine)
    brain = FakeBrain()
    store = InMemoryDreamStore()
    scheduler = make(
        clock,
        brain=brain,
        dream_engine=spy,
        affect=Affect(HEAVY),
        dream_store=store,
        config=config,
    )
    await enter_nap(scheduler, clock)
    prompt = brain.calls[0]["prompt"]
    assert "anxious" not in prompt.lower()
    assert "Emotional undertone" not in prompt
    assert spy.biases == [True]
    (record,) = await store.recent_dreams(1)
    assert record["parameters"]["gentle_floor"] is True

    mild = measured_pad(-0.3, 0.8, 0.2, "Anxious")
    spy_mild = engine(cls=SpyEngine)
    mild_brain = FakeBrain()
    other_clock = Clock(at(12, 0))
    other = make(
        other_clock,
        brain=mild_brain,
        dream_engine=spy_mild,
        affect=Affect(mild),
        config=config,
    )
    await enter_nap(other, other_clock)
    assert spy_mild.biases == [None]
    assert "Emotional undertone: anxious." in mild_brain.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_neutral_label_gives_no_undertone():
    """A measured state labelled Neutral names no emotion, however strong."""
    neutral = measured_pad(0.05, 0.95, 0.95, "Neutral")
    clock = Clock(at(12, 0))
    brain = FakeBrain()
    stream = FakeStream()
    scheduler = make(clock, brain=brain, affect=Affect(neutral), thought_stream=stream)
    await enter_nap(scheduler, clock)
    assert "Emotional undertone" not in brain.calls[0]["prompt"]
    assert "neutral" not in brain.calls[0]["prompt"].lower()
    assert stream.injected[0][3] == ""

    # The label the PAD model gives a near-zero pleasure is Neutral too.
    computed = PADAffectModel().compute(
        PADSources(somatic_valence=0.05, protoself_arousal=1.0, agency_score=1.0),
        measured=MEASURED,
    )
    assert computed.label == "Neutral"
    other_clock = Clock(at(12, 0))
    other_brain = FakeBrain()
    other = make(other_clock, brain=other_brain, affect=Affect(computed))
    await enter_nap(other, other_clock)
    assert "Emotional undertone" not in other_brain.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_state_of_unknown_provenance_gives_no_undertone():
    """A state without measured inputs is not a reading: no emotion, no floor."""
    unknown = PADState(pleasure=-0.8, arousal=0.8, dominance=0.2, label="Anxious")
    assert unknown.measured_inputs is None
    clock = Clock(at(12, 0))
    spy = engine(cls=SpyEngine)
    brain = FakeBrain()
    stream = FakeStream()
    store = InMemoryDreamStore()
    scheduler = make(
        clock,
        brain=brain,
        dream_engine=spy,
        affect=Affect(unknown),
        thought_stream=stream,
        dream_store=store,
        config=SleepSchedulerConfig(gentle_floor_enabled=True),
    )
    await enter_nap(scheduler, clock)
    prompt = brain.calls[0]["prompt"]
    assert "Emotional undertone" not in prompt
    assert "anxious" not in prompt.lower()
    assert spy.biases == [None]
    assert stream.injected[0][3] == ""
    (record,) = await store.recent_dreams(1)
    assert record["parameters"]["gentle_floor"] is False

    # Measured inputs that leave out pleasure or arousal do not make a reading.
    no_arousal = PADState(
        pleasure=-0.8,
        arousal=0.8,
        dominance=0.2,
        label="Anxious",
        measured_inputs=("somatic_valence", "agency_score"),
    )
    other_clock = Clock(at(12, 0))
    other_brain = FakeBrain()
    other = make(other_clock, brain=other_brain, affect=Affect(no_arousal))
    await enter_nap(other, other_clock)
    assert "Emotional undertone" not in other_brain.calls[0]["prompt"]

    # The PAD model's own unlisted state is of unknown provenance as well.
    unlisted = PADAffectModel().compute(PADSources(somatic_valence=-0.9))
    assert unlisted.measured_inputs is None
    third_clock = Clock(at(12, 0))
    third_brain = FakeBrain()
    third = make(third_clock, brain=third_brain, affect=Affect(unlisted))
    await enter_nap(third, third_clock)
    assert "Emotional undertone" not in third_brain.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_gentle_floor_passes_comfort_bias_and_no_emotion():
    """Under the gentle floor the dream context holds no emotion word at all."""
    config = SleepSchedulerConfig(gentle_floor_enabled=True)
    clock = Clock(at(12, 0))
    spy = engine(cls=SpyEngine)
    brain = FakeBrain()
    stream = FakeStream()
    scheduler = make(
        clock,
        brain=brain,
        dream_engine=spy,
        affect=Affect(HEAVY),
        thought_stream=stream,
        config=config,
    )
    context, gentle = scheduler._dream_context("nap")
    assert gentle is True
    assert context.dominant_emotion is None
    assert not hasattr(config, "gentle_floor_emotion")
    await enter_nap(scheduler, clock)
    assert spy.biases == [True]
    assert "Emotional undertone" not in brain.calls[0]["prompt"]
    assert stream.injected[0][3] == ""


@pytest.mark.asyncio
async def test_dream_record_fields():
    clock = Clock(at(12, 0))
    store = InMemoryDreamStore()
    eng = engine()
    scheduler = make(
        clock,
        brain=FakeBrain(["A bridge of paper."]),
        dream_engine=eng,
        dream_store=store,
    )
    await enter_nap(scheduler, clock)
    (record,) = await store.recent_dreams(1)
    assert set(record) == RECORD_FIELDS
    assert record["content"] == "A bridge of paper."
    assert record["dream_type"] in eng.config.templates
    assert record["dream_type"] == eng.recent_types[-1]
    assert record["timestamp"] == clock.now
    assert record["timestamp"].utcoffset() == timedelta(0)
    assert record["state"] == "napping"
    params = record["parameters"]
    assert set(params) == {"temperature", "max_tokens", "mode", "gentle_floor"}
    assert params["mode"] == "nap"
    assert params["max_tokens"] < eng.parameters()["max_tokens"]
    assert record["finish_reason"] is None
    assert record["output_tokens"] is None
    status = scheduler.get_status()
    assert status["dreams"]["stored"] == 1
    assert status["last_dream_at"] == clock.now.isoformat()


@pytest.mark.asyncio
async def test_dream_injected_into_thought_stream():
    clock = Clock(at(12, 0))
    stream = FakeStream()
    pad = measured_pad(0.6, 0.8, 0.7, "Exhilarated")
    store = InMemoryDreamStore()
    scheduler = make(
        clock,
        brain=FakeBrain(["The lanterns drift upward."]),
        thought_stream=stream,
        affect=Affect(pad),
        dream_store=store,
    )
    await enter_nap(scheduler, clock)
    (record,) = await store.recent_dreams(1)
    (call,) = stream.injected
    assert call == (
        "inject",
        "The lanterns drift upward.",
        f"dream_{record['dream_type']}",
        "exhilarated",
    )

    plain_stream = FakeStream()
    plain_clock = Clock(at(12, 0))
    plain = make(
        plain_clock,
        brain=FakeBrain(["A field of clocks."]),
        thought_stream=plain_stream,
    )
    await enter_nap(plain, plain_clock)
    assert plain_stream.injected[0][3] == ""


@pytest.mark.asyncio
async def test_store_failure_is_counted_and_dream_still_recorded_in_stream():
    class BrokenStore:
        async def add_dream(self, doc):
            raise RuntimeError("disk full")

        async def recent_dreams(self, limit):
            return []

    clock = Clock(at(12, 0))
    stream = FakeStream()
    scheduler = make(
        clock,
        brain=FakeBrain(["A stair of water."]),
        dream_store=BrokenStore(),
        thought_stream=stream,
    )
    await enter_nap(scheduler, clock)
    status = scheduler.get_status()
    assert status["errors"]["dream_store"] == 1
    assert status["dreams"]["stored"] == 0
    assert len(stream.injected) == 1


@pytest.mark.asyncio
async def test_none_store_means_in_memory():
    clock = Clock(at(12, 0))
    scheduler = SleepScheduler(
        brain=FakeBrain(["A quiet tide."]),
        dream_engine=engine(),
        clock=clock,
        settings=SimpleNamespace(TIMEZONE="UTC", WAKE_HOUR=6, SLEEP_HOUR=22),
    )
    assert isinstance(scheduler.dream_store, InMemoryDreamStore)
    await enter_nap(scheduler, clock)
    assert len(await scheduler.dream_store.recent_dreams(5)) == 1


class WakingBrain(FakeBrain):
    """A brain whose reply arrives after a conversation turn woke the system."""

    def __init__(self, replies=None):
        super().__init__(replies)
        self.scheduler = None

    async def think(self, *args, **kwargs):
        reply = await super().think(*args, **kwargs)
        self.scheduler.notify_activity()
        await self.scheduler.drain()
        return reply


@pytest.mark.asyncio
@pytest.mark.parametrize("enter", [enter_nap, enter_night])
async def test_dream_arriving_after_wake_is_neither_stored_nor_injected(enter):
    clock = Clock(at(12, 0))
    store = InMemoryDreamStore()
    stream = FakeStream()
    brain = WakingBrain(["A lantern drifting over a quiet field."])
    scheduler = make(clock, brain=brain, dream_store=store, thought_stream=stream)
    brain.scheduler = scheduler
    clock.now = at(23, 0) if enter is enter_night else at(12, 0)
    scheduler.notify_activity()
    clock.advance(30 if enter is enter_nap else 5)
    assert await scheduler.tick() is SleepState.AWAKE
    await scheduler.drain()
    assert len(brain.calls) == 1
    assert stream.calls == [
        ("mode", "dreaming"),
        ("pause",),
        ("mode", "awake"),
        ("resume",),
    ]
    assert stream.injected == []
    assert await store.recent_dreams(10) == []
    status = scheduler.get_status()
    assert status["dreams"]["attempted"] == 1
    assert status["dreams"]["stored"] == 0
    assert status["dreams"]["woken"] == 1
    assert status["last_dream_at"] is None
