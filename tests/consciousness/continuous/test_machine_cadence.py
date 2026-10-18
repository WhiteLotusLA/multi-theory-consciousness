"""The machine's cadence: state rates, jitter and floor, the rate limit, the
measured level's modulation, the non-blocking resource guard, and a tick that
needs no sleeping. The helpers here are shared by the other machine tests."""

import asyncio
import random
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pytest

from mtc.consciousness.continuous import machine as machine_module
from mtc.consciousness.continuous.buffer import ThoughtBuffer
from mtc.consciousness.continuous.config import CTMConfig
from mtc.consciousness.continuous.machine import (
    RATE_LIMIT_BACKOFF_S,
    ContinuousThoughtMachine,
)
from mtc.consciousness.continuous.novelty import NoveltyFilter
from mtc.consciousness.continuous.schemas import (
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
    new_thought_id,
)
from mtc.consciousness.continuous.store import InMemoryThoughtStore
from mtc.core.config import Settings
from mtc.memory.inmemory import InMemoryThoughtGistStore

START = datetime(2000, 1, 1, 8, 0, tzinfo=timezone.utc)
WORDS = (
    "river",
    "bridge",
    "garden",
    "stone",
    "lantern",
    "window",
    "cloud",
    "field",
    "harbor",
    "meadow",
    "orchard",
    "valley",
)


def run(coro):
    return asyncio.run(coro)


class Clock:
    """A wall clock that moves only when told to."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class LookupEmbedder:
    """Each word of ``WORDS`` owns one axis (a semantic-flagged stand-in)."""

    semantic = True

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                if word in WORDS:
                    out[row, WORDS.index(word)] += 1.0
            if not out[row].any():
                out[row, self.dim - 1] = 1.0
            out[row] /= np.linalg.norm(out[row])
        return out


class ScriptedGenerator:
    """Returns one scripted thought per call (None once the script is empty).

    Like the seed generator's, each thought is a template thought whose
    salience is a design constant (``form`` and ``salience_basis``).
    """

    def __init__(self, texts: Iterable[str], clock: Clock, kind=ThoughtType.REFLECTION):
        self.texts = deque(texts)
        self.clock = clock
        self.kind = kind
        self.calls: List[str] = []
        self.last_miss_reason: Optional[str] = None

    async def generate_thought(self, state: str = "awake") -> Optional[ThoughtRecord]:
        self.calls.append(state)
        if not self.texts:
            self.last_miss_reason = "no_seeds"
            return None
        self.last_miss_reason = None
        return ThoughtRecord(
            thought_id=new_thought_id(),
            timestamp=self.clock(),
            thought_type=self.kind,
            content=self.texts.popleft(),
            intensity=ThoughtIntensity.MODERATE,
            salience_score=0.5,
            consciousness_state=state,
            processing_context={
                "form": "template",
                "salience_basis": "strategy_default",
            },
            should_express=True,
        )

    def llm_context(self):
        return {"recent_memory": "", "emotions": {}, "recent_topic": ""}

    def status(self):
        return {"scripted_left": len(self.texts)}


def endless(count: int = 400) -> List[str]:
    """Distinct texts (each a different pair of words) for long runs."""
    texts = []
    for i in range(count):
        texts.append(f"{WORDS[i % len(WORDS)]} {WORDS[(i // len(WORDS)) % len(WORDS)]}")
    return texts


def make_machine(
    texts: Iterable[str] = (),
    *,
    clock: Optional[Clock] = None,
    config: Optional[CTMConfig] = None,
    embedder="lookup",
    store=None,
    gist_store="memory",
    settings: Optional[Settings] = None,
    allow_non_semantic: bool = False,
    seed: int = 3,
    **kwargs,
) -> ContinuousThoughtMachine:
    clock = clock or Clock()
    cfg = config or CTMConfig()
    if embedder == "lookup":
        embedder = LookupEmbedder()
    novelty = kwargs.pop("novelty", "build")
    if novelty == "build":
        novelty = NoveltyFilter(
            embedder,
            cache_size=cfg.novelty.cache,
            dedup_threshold=cfg.novelty.dedup,
            express_threshold=cfg.novelty.express,
            allow_non_semantic=allow_non_semantic,
        )
    return ContinuousThoughtMachine(
        kwargs.pop("generator", None) or ScriptedGenerator(texts, clock),
        ThoughtBuffer(max_buffer_size=cfg.buffer_size),
        store if store is not None else InMemoryThoughtStore(clock=clock),
        gist_store=(
            InMemoryThoughtGistStore() if gist_store == "memory" else gist_store
        ),
        novelty=novelty,
        config=cfg,
        clock=clock,
        rng=random.Random(seed),
        settings=settings or Settings(DEBUG=False),
        **kwargs,
    )


def cfg_with(**changes) -> CTMConfig:
    return CTMConfig.model_validate(changes)


class FakeProcess:
    def __init__(self, readings: Sequence[float]) -> None:
        self.readings = deque(readings)
        self.intervals: List[object] = []

    def cpu_percent(self, interval=0.1):
        self.intervals.append(interval)
        return self.readings.popleft() if self.readings else 0.0


# ---------------------------------------------------------------------------


def test_frequency_by_state():
    machine = make_machine()
    expected = {"awake": 0.008, "drowsy": 0.005, "dreaming": 0.003, "deep_sleep": 0.002}
    for state, rate in expected.items():
        machine.set_consciousness_state(state)
        assert machine.current_frequency == pytest.approx(rate)
        assert machine.get_status()["current_frequency"] == pytest.approx(rate)
    assert expected["dreaming"] < expected["awake"]
    peak_per_minute = 0.008 * (0.4 + 1.2 * 1.0) * 60
    assert machine.config.max_thoughts_per_minute > peak_per_minute


def test_jitter_bounds_and_floor():
    machine = make_machine()
    base = 1 / 0.008
    delays = [machine.next_delay() for _ in range(500)]
    assert min(delays) >= base * 0.7 - 1e-9
    assert max(delays) <= base * 1.3 + 1e-9
    assert max(delays) - min(delays) > base * 0.3  # the spread is used

    fast = make_machine(config=cfg_with(cadence={"awake": 5.0}))
    assert {fast.next_delay() for _ in range(50)} == {1.0}  # min_sleep_s floor
    floor = make_machine(config=cfg_with(cadence={"awake": 5.0}, min_sleep_s=0.0))
    assert all(0.0 < floor.next_delay() < 0.3 for _ in range(50))


def test_rate_limit_backoff():
    clock = Clock()
    machine = make_machine(endless(10), clock=clock)
    assert machine.config.max_thoughts_per_minute == 2
    for _ in range(2):
        assert run(machine.tick()).generated is True
        clock.advance(1)
    limited = run(machine.tick())
    assert limited.generated is False and limited.quiet_reason == "rate_limited"
    assert machine.next_delay(limited) == RATE_LIMIT_BACKOFF_S
    assert machine.get_status()["quiet_reason"] == "rate_limited"
    assert len(machine.generator.calls) == 2  # the generator was not asked
    clock.advance(61)
    assert run(machine.tick()).generated is True
    assert machine.get_status()["counters"]["rate_limited"] == 1


def test_level_modulation_factor():
    calls = []

    async def level():
        calls.append(1)
        return 0.75

    clock = Clock()
    machine = make_machine(
        endless(20),
        clock=clock,
        config=cfg_with(max_thoughts_per_minute=100),
        consciousness_level_provider=level,
    )
    run(machine.tick())
    assert machine.consciousness_level == 0.75
    assert machine.modulation_factor() == pytest.approx(0.4 + 1.2 * 0.75)
    assert machine.current_frequency == pytest.approx(0.008 * 1.3)
    assert machine.get_status()["modulation"] == "measured"
    for _ in range(9):
        clock.advance(1)
        run(machine.tick())
    # refreshed at the first tick and after every five generated thoughts
    assert len(calls) == 2


def test_level_unavailable_means_no_modulation_and_status_says_so():
    async def unknown():
        return None

    machine = make_machine(["river"], consciousness_level_provider=unknown)
    run(machine.tick())
    status = machine.get_status()
    assert machine.consciousness_level is None
    assert status["consciousness_level"] is None
    assert status["modulation"] == "unavailable"
    assert status["modulation_factor"] == 1.0
    assert machine.current_frequency == pytest.approx(0.008)

    absent = make_machine(["river"])
    run(absent.tick())
    assert absent.get_status()["modulation"] == "no_provider"
    assert absent.modulation_factor() == 1.0

    async def high():
        return 1.0

    off = make_machine(
        ["river"],
        config=cfg_with(consciousness_modulation={"enabled": False}),
        consciousness_level_provider=high,
    )
    run(off.tick())
    assert off.get_status()["modulation"] == "disabled"
    assert off.modulation_factor() == 1.0


def test_unmeasured_level_leaves_the_cadence_unmodulated():
    readings = deque([0.0, None, float("nan")])

    async def level():
        return readings.popleft()

    clock = Clock()
    machine = make_machine(
        endless(20),
        clock=clock,
        config=cfg_with(
            max_thoughts_per_minute=100, consciousness_modulation={"refresh_every": 1}
        ),
        consciousness_level_provider=level,
    )
    run(machine.tick())
    # a measured 0.0 slows the cadence: that is a measurement
    assert machine.get_status()["modulation"] == "measured"
    assert machine.modulation_factor() == pytest.approx(0.4)
    for _ in range(2):
        clock.advance(1)
        run(machine.tick())
        status = machine.get_status()
        # the provider has no measured level: nothing is assumed in its place
        assert status["consciousness_level"] is None
        assert status["modulation"] == "unavailable"
        assert status["modulation_factor"] == 1.0
        assert machine.modulation_factor() == 1.0
        assert machine.current_frequency == pytest.approx(0.008)
    assert readings == deque()


def test_resource_check_is_non_blocking():
    process = FakeProcess([1.0, 1.0])
    clock = Clock()
    machine = make_machine(
        endless(30),
        clock=clock,
        config=cfg_with(max_thoughts_per_minute=100, resource={"check_every": 10}),
        process=process,
    )
    for _ in range(20):
        run(machine.tick())
        clock.advance(1)
    assert process.intervals == [None, None]  # no blocking sample interval
    status = machine.get_status()["resource"]
    assert status["available"] is True and status["checks"] == 2
    source = machine_module.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "interval=0." not in text and "time.sleep" not in text


def test_frequency_restored_after_breach_clears():
    process = FakeProcess([50.0, 1.0])
    clock = Clock()
    machine = make_machine(
        endless(30),
        clock=clock,
        config=cfg_with(max_thoughts_per_minute=100, resource={"check_every": 5}),
        process=process,
    )
    for _ in range(5):
        run(machine.tick())
        clock.advance(1)
    assert machine.get_status()["resource"]["throttled"] is True
    assert machine.current_frequency == pytest.approx(0.008 * 0.5)
    for _ in range(5):
        run(machine.tick())
        clock.advance(1)
    assert machine.get_status()["resource"]["throttled"] is False
    assert machine.current_frequency == pytest.approx(0.008)
    assert machine.get_status()["resource"]["breaches"] == 1


def test_tick_needs_no_sleeping(monkeypatch):
    async def no_sleep(*_args, **_kwargs):
        raise AssertionError("tick must not sleep")

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    clock = Clock()
    machine = make_machine(endless(5), clock=clock)
    first = run(machine.tick())
    clock.advance(1)
    second = run(machine.tick())
    assert first.generated and second.generated
    assert first.persisted and second.persisted
