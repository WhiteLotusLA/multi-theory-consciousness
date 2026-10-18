"""Dedup-on-write, thought ids and child links, burst ids, gists with one
count key, memory associations, and blocking work kept off the event loop."""

import asyncio
import threading

import pytest

from mtc.brain.prompt import PromptAssembler, PromptInputs
from mtc.consciousness.continuous.store import InMemoryThoughtStore
from mtc.consciousness.continuous.surfacing import gist_items
from mtc.memory.inmemory import InMemoryThoughtGistStore
from tests.consciousness.continuous.test_machine_cadence import (
    WORDS,
    Clock,
    LookupEmbedder,
    ScriptedGenerator,
    cfg_with,
    make_machine,
    run,
)

FAST = {"max_thoughts_per_minute": 1000}


async def _docs(store: InMemoryThoughtStore):
    return {r.thought_id: r for r in await store.recent(1000)}


def test_near_duplicate_bumps_access_not_insert():
    store = InMemoryThoughtStore()
    machine = make_machine(
        ["river bridge", "the river bridge"], store=store, config=cfg_with(**FAST)
    )
    first = run(machine.tick())
    second = run(machine.tick())
    assert first.persisted is True and first.duplicate is False
    assert second.generated is True
    assert second.persisted is False and second.duplicate is True
    assert second.expressed is False
    assert run(store.count()) == 1
    (stored,) = run(store.recent(5))
    assert stored.content == "river bridge"
    assert stored.access_count == 1
    counters = machine.get_status()["counters"]
    assert counters["duplicates_filtered"] == 1
    assert counters.get("access_unmatched", 0) == 0


def test_unique_inserted_with_thought_id():
    store = InMemoryThoughtStore()
    machine = make_machine(["river", "garden"], store=store, config=cfg_with(**FAST))
    results = [run(machine.tick()) for _ in range(2)]
    docs = run(_docs(store))
    assert set(docs) == {r.thought_id for r in results}
    for result in results:
        assert docs[result.thought_id].to_doc()["thought_id"] == result.thought_id


def test_parent_child_link_resolves_by_thought_id():
    store = InMemoryThoughtStore()
    machine = make_machine(
        ["river", "garden", "the garden", "stone"],
        store=store,
        config=cfg_with(**FAST),
    )
    first, second, duplicate, third = [run(machine.tick()) for _ in range(4)]
    assert duplicate.duplicate is True
    docs = run(_docs(store))
    assert docs[second.thought_id].previous_thought_id == first.thought_id
    # the duplicate was never stored, so the next thought links to the last
    # stored one and the link resolves by thought_id
    assert docs[third.thought_id].previous_thought_id == second.thought_id
    raw = {d.thought_id: d.to_doc() for d in docs.values()}
    assert raw[first.thought_id]["child_thoughts"] == [second.thought_id]
    assert raw[second.thought_id]["child_thoughts"] == [third.thought_id]
    counters = machine.get_status()["counters"]
    assert counters["links_resolved"] == 2
    assert counters.get("links_unresolved", 0) == 0


class SlowStore(InMemoryThoughtStore):
    """An insert that announces itself and then takes a while."""

    def __init__(self, delay: float = 0.05):
        super().__init__()
        self.delay = delay
        self.entered = None

    async def insert(self, record):
        if self.entered is not None:
            self.entered.set()
        await asyncio.sleep(self.delay)
        return await super().insert(record)


def test_injection_during_a_tick_does_not_fork_the_chain():
    store = SlowStore()
    machine = make_machine(["river", "garden"], store=store, config=cfg_with(**FAST))

    async def scenario():
        first = await machine.tick()
        store.entered = asyncio.Event()

        async def inject_while_the_tick_inserts():
            await store.entered.wait()
            return await machine.inject_thought("A loose dream.", "dream", "")

        tick, injected = await asyncio.gather(
            machine.tick(), inject_while_the_tick_inserts()
        )
        return first, tick, injected

    first, tick, injected = run(scenario())
    assert tick.persisted is True
    docs = run(_docs(store))
    assert set(docs) == {first.thought_id, tick.thought_id, injected.thought_id}
    ticked = docs[tick.thought_id]
    dream = docs[injected.thought_id]
    assert ticked.previous_thought_id != dream.previous_thought_id
    # one chain: the tick's thought follows the first, the dream follows it
    assert ticked.previous_thought_id == first.thought_id
    assert dream.previous_thought_id == tick.thought_id
    assert raw_children(docs, first.thought_id) == [tick.thought_id]
    assert raw_children(docs, tick.thought_id) == [injected.thought_id]
    counters = machine.get_status()["counters"]
    assert counters["links_resolved"] == 2
    assert counters.get("links_unresolved", 0) == 0


def test_concurrent_injections_form_one_chain():
    store = SlowStore()
    machine = make_machine(store=store)

    async def scenario():
        return await asyncio.gather(
            *(machine.inject_thought(f"Dream {n}.", "dream", "") for n in range(4))
        )

    records = run(scenario())
    docs = run(_docs(store))
    assert set(docs) == {r.thought_id for r in records}
    parents = [docs[r.thought_id].previous_thought_id for r in records]
    assert parents.count(None) == 1  # one head
    assert len(set(parents)) == 4  # no two thoughts share a parent
    children = {}
    for record in records:
        parent = docs[record.thought_id].previous_thought_id
        if parent is not None:
            assert parent in docs
            children[parent] = record.thought_id
    head = records[parents.index(None)].thought_id
    chain = [head]
    while chain[-1] in children:
        chain.append(children[chain[-1]])
    assert len(chain) == 4  # every stored thought is on the one chain
    assert machine.get_status()["counters"]["links_resolved"] == 3


def test_persisted_hook_may_inject():
    injected = []

    async def on_persisted(thought):
        if thought.processing_context.get("form") != "injected":
            record = await machine.inject_thought("A follow-up.", "dream", "")
            injected.append(record)

    store = InMemoryThoughtStore()
    machine = make_machine(["river"], store=store, on_thought_persisted=on_persisted)

    async def scenario():
        return await asyncio.wait_for(machine.tick(), timeout=5)

    result = run(scenario())
    assert result.persisted is True
    (record,) = injected
    docs = run(_docs(store))
    assert docs[record.thought_id].previous_thought_id == result.thought_id


def test_unstored_thought_names_no_parent():
    generated = []
    store = InMemoryThoughtStore()
    machine = make_machine(
        ["river", "the river"],
        store=store,
        config=cfg_with(**FAST),
        on_thought_generated=generated.append,
    )
    first = run(machine.tick())
    duplicate = run(machine.tick())
    assert first.persisted is True and duplicate.duplicate is True
    assert generated[1].previous_thought_id is None  # never stored, no link

    class BrokenStore(InMemoryThoughtStore):
        async def insert(self, record):
            raise RuntimeError("store down")

    broken = make_machine(store=BrokenStore())
    broken._last_persisted_id = first.thought_id
    record = run(broken.inject_thought("A loose dream.", "dream", ""))
    assert record.previous_thought_id is None
    assert broken.get_status()["failures"] == {"persist": 1}


def test_burst_stream_id_rotates_on_gap_and_state_change():
    clock = Clock()
    store = InMemoryThoughtStore()
    machine = make_machine(
        ["river", "garden", "stone", "lantern"],
        clock=clock,
        store=store,
        config=cfg_with(**FAST),
    )
    a = run(machine.tick())
    clock.advance(10)
    b = run(machine.tick())
    clock.advance(301)  # longer than burst_gap_s
    c = run(machine.tick())
    clock.advance(5)
    machine.set_consciousness_state("drowsy")
    d = run(machine.tick())
    docs = run(_docs(store))
    ids = [docs[r.thought_id].stream_id for r in (a, b, c, d)]
    assert ids[0] == ids[1]
    assert ids[2] != ids[1]
    assert ids[3] != ids[2]
    assert all(i != docs[r.thought_id].thought_id for i, r in zip(ids, (a, b, c, d)))


def test_gist_every_n_with_source_count():
    machine = make_machine(WORDS[:7], config=cfg_with(gist={"every": 5}, **FAST))
    results = [run(machine.tick()) for _ in range(7)]
    gists = run(machine.gist_store.active_gists(min_retention=0.0))
    assert len(gists) == 1
    gist = gists[0]
    assert gist.source_count == 5
    assert set(gist.source_thought_ids) == {r.thought_id for r in results[:5]}
    assert gist.summary.startswith("reflection thoughts (5): ")
    for word in WORDS[2:5]:
        assert word in gist.summary  # composed from the window's own texts
    assert gist.retention_strength == 0.8
    assert gist.dominant_emotion is None  # no thought carried an emotion
    assert "source_count" in gist.to_document()
    assert "thought_count" not in gist.to_document()


def test_gist_importance_is_measured_or_none():
    constant = make_machine(WORDS[:5], config=cfg_with(gist={"every": 5}, **FAST))
    for _ in range(5):
        run(constant.tick())
    (gist,) = run(constant.gist_store.active_gists(min_retention=0.0))
    # every thought's salience was a design constant: nothing was measured
    assert gist.avg_salience is None
    assert gist.importance_score is None
    assert gist.retention_strength == 0.8  # the configured starting retention

    # the measured mean (0.8) differs from the mean over every thought (0.62),
    # so averaging the design constants in as well cannot pass
    measured = {WORDS[6]: 0.9, WORDS[8]: 0.7}

    def tag(thought):
        if thought.content in measured:
            thought.salience_score = measured[thought.content]
            thought.processing_context["salience_basis"] = "memory_importance"

    mixed = make_machine(
        WORDS[5:10],
        config=cfg_with(gist={"every": 5}, **FAST),
        on_thought_generated=tag,
    )
    for _ in range(5):
        run(mixed.tick())
    (gist,) = run(mixed.gist_store.active_gists(min_retention=0.0))
    assert gist.avg_salience == pytest.approx(0.8)
    assert gist.importance_score == pytest.approx(0.8)


def test_gist_mood_is_measured_or_none_and_none_renders_no_mood():
    machine = make_machine(WORDS[:5], config=cfg_with(gist={"every": 5}, **FAST))
    for _ in range(5):
        run(machine.tick())
    (gist,) = run(machine.gist_store.active_gists(min_retention=0.0))
    assert gist.dominant_emotion is None
    assert gist.to_document()["dominant_emotion"] is None
    items = gist_items([gist])
    assert items[0]["dominant_emotion"] is None
    loud = PromptAssembler(narrate_thought_mood=True)
    text = loud.render(PromptInputs(thought_gists=items)).text
    (line,) = [row for row in text.splitlines() if row.startswith("- reflection")]
    assert line.endswith("(from 5 thoughts)")  # no mood, no placeholder word
    assert "[mood:" not in text and "unmeasured" not in text and "None" not in text

    def tag(thought):
        thought.emotional_context = {"calm": 0.6, "unease": 0.1, "flag": True}

    tagged = make_machine(
        WORDS[5:10],
        config=cfg_with(gist={"every": 5}, **FAST),
        on_thought_generated=tag,
    )
    for _ in range(5):
        run(tagged.tick())
    (gist,) = run(tagged.gist_store.active_gists(min_retention=0.0))
    assert gist.dominant_emotion == "calm"  # unease is below the floor
    text = loud.render(PromptInputs(thought_gists=gist_items([gist]))).text
    assert "(from 5 thoughts) [mood: calm]" in text


def test_unstored_duplicate_never_enters_the_buffer():
    store = InMemoryThoughtStore()
    machine = make_machine(
        ["river", "the river", "garden"], store=store, config=cfg_with(**FAST)
    )
    first = run(machine.tick())
    duplicate = run(machine.tick())
    assert duplicate.duplicate is True and duplicate.persisted is False
    held = list(machine.buffer.current_thoughts)
    assert [t.thought_id for t in held] == [first.thought_id]
    assert held[0].leads_to_thought_id is None  # no link to an unstored thought
    assert machine.buffer.stream_state.current_thought_id == first.thought_id
    assert machine.buffer.stream_state.total_thoughts_today == 1
    third = run(machine.tick())
    docs = run(_docs(store))
    held = list(machine.buffer.current_thoughts)
    assert [t.thought_id for t in held] == [first.thought_id, third.thought_id]
    for record in held:
        assert record.thought_id in docs
        for link in (record.previous_thought_id, record.leads_to_thought_id):
            assert link is None or link in docs
    assert held[0].leads_to_thought_id == third.thought_id
    assert raw_children(docs, first.thought_id) == [third.thought_id]


def raw_children(docs, thought_id):
    return docs[thought_id].to_doc()["child_thoughts"]


def test_failed_insert_leaves_the_buffer_unchanged():
    class BrokenStore(InMemoryThoughtStore):
        async def insert(self, record):
            raise RuntimeError("store down")

    machine = make_machine(["river"], store=BrokenStore(), config=cfg_with(**FAST))
    result = run(machine.tick())
    assert result.generated is True and result.persisted is False
    assert list(machine.buffer.current_thoughts) == []
    record = run(machine.inject_thought("A loose dream.", "dream", "calm"))
    assert record is not None
    assert list(machine.buffer.current_thoughts) == []
    assert machine.get_status()["failures"] == {"persist": 2}

    store = InMemoryThoughtStore()
    working = make_machine(store=store)
    injected = run(working.inject_thought("A plan.", "planning", ""))
    (held,) = working.buffer.current_thoughts
    assert held.thought_id == injected.thought_id
    assert run(store.count()) == 1


def test_thought_without_state_changes_no_burst():
    clock = Clock()
    store = InMemoryThoughtStore()

    def drop_state(thought):
        if thought.content == "garden":
            thought.consciousness_state = None

    machine = make_machine(
        ["river", "garden", "stone"],
        clock=clock,
        store=store,
        config=cfg_with(**FAST),
        on_thought_generated=drop_state,
    )
    a = run(machine.tick())
    clock.advance(5)
    b = run(machine.tick())
    clock.advance(5)
    c = run(machine.tick())
    docs = run(_docs(store))
    assert docs[b.thought_id].consciousness_state is None
    assert docs[c.thought_id].consciousness_state == "awake"
    assert len({docs[r.thought_id].stream_id for r in (a, b, c)}) == 1
    assert machine.get_status()["counters"].get("bursts", 0) == 0


def test_gist_cooldown_and_min_new_fraction():
    clock = Clock()
    machine = make_machine(
        WORDS[:12],
        clock=clock,
        config=cfg_with(gist={"every": 3, "cooldown_s": 60}, **FAST),
    )
    for _ in range(3):
        run(machine.tick())
    assert machine.get_status()["counters"]["gists"] == 1
    for _ in range(4):  # inside the cooldown: no gist, however many thoughts
        clock.advance(5)
        run(machine.tick())
    assert machine.get_status()["counters"]["gists"] == 1
    clock.advance(60)
    run(machine.tick())  # first tick after the cooldown mints
    assert machine.get_status()["counters"]["gists"] == 2

    class StaleWindow(InMemoryThoughtStore):
        """Recent reads keep returning the first window (an overlapping one)."""

        def __init__(self):
            super().__init__()
            self.first = None

        async def recent(self, limit):
            window = await super().recent(limit)
            if self.first is None:
                self.first = window
            return self.first

    stale = StaleWindow()
    overlapping = make_machine(
        WORDS[:9],
        store=stale,
        config=cfg_with(
            gist={"every": 3, "cooldown_s": 0, "min_new_fraction": 0.5}, **FAST
        ),
    )
    for _ in range(9):
        run(overlapping.tick())
    counters = overlapping.get_status()["counters"]
    assert counters["gists"] == 1
    assert counters["gists_skipped"] == 2


def test_failed_gist_store_does_not_record_window():
    class FlakyGists(InMemoryThoughtGistStore):
        def __init__(self):
            super().__init__()
            self.fail = True

        async def add_gist(self, gist):
            if self.fail:
                self.fail = False
                raise RuntimeError("gist store down")
            return await super().add_gist(gist)

    gists = FlakyGists()
    machine = make_machine(
        WORDS[:6],
        gist_store=gists,
        config=cfg_with(gist={"every": 3, "cooldown_s": 0}, **FAST),
    )
    for _ in range(3):
        run(machine.tick())
    assert machine.get_status()["failures"] == {"gist_store": 1}
    assert machine.to_state_dict()["histories"]["last_gist_window"] == []
    for _ in range(3):
        run(machine.tick())
    assert len(run(gists.active_gists(min_retention=0.0))) == 1


def test_association_links_only_with_vector_search():
    calls = []

    class Vectors:
        def search_similar(self, query, top_k=None, min_score=0.7):
            calls.append((query, top_k, min_score))
            return [("p1", 0.9, {}), ("p2", 0.7, {})]

    store = InMemoryThoughtStore()
    linked = make_machine(["river"], store=store, vector_search=Vectors())
    result = run(linked.tick())
    assert run(_docs(store))[result.thought_id].associated_memories == ["p1", "p2"]
    assert calls == [("river", 3, 0.5)]

    plain_store = InMemoryThoughtStore()
    plain = make_machine(["river"], store=plain_store)
    result = run(plain.tick())
    assert run(_docs(plain_store))[result.thought_id].associated_memories == []


def test_store_calls_off_loop():
    threads = {}

    class RecordingEmbedder(LookupEmbedder):
        def encode(self, texts):
            threads["embed"] = threading.get_ident()
            return super().encode(texts)

    class Vectors:
        def search_similar(self, query, top_k=None, min_score=0.7):
            threads["vectors"] = threading.get_ident()
            return []

    class RecordingStore(InMemoryThoughtStore):
        async def insert(self, record):
            threads["store"] = threading.get_ident()
            return await super().insert(record)

    async def scenario():
        threads["loop"] = threading.get_ident()
        clock = Clock()
        machine = make_machine(
            clock=clock,
            generator=ScriptedGenerator(["river"], clock),
            embedder=RecordingEmbedder(),
            vector_search=Vectors(),
            store=RecordingStore(),
        )
        return await machine.tick()

    result = run(scenario())
    assert result.persisted is True
    assert threads["embed"] != threads["loop"]  # embedding off the loop
    assert threads["vectors"] != threads["loop"]  # blocking vector search too
    assert threads["store"] == threads["loop"]  # the store protocol is async
