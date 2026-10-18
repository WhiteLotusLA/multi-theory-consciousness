"""The install date is written once in the configured time zone's calendar,
the seed blocks keep the memory package's neutral rules plus one install-date
sentence, existing core blocks are kept, and only enabled stores are seeded."""

import asyncio
import re
import stat
from datetime import datetime, timezone

import pytest

from mtc.core.config import Settings
from mtc.core.envfile import read_env
from mtc.onboarding import seed_db
from mtc.onboarding.seed_db import (
    INSTALL_DATE_SENTENCE,
    record_birth_date,
    render_seed_blocks,
    seed_stores,
)

INSTANT = datetime(2000, 1, 1, 23, 30, tzinfo=timezone.utc)
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def make_settings(tmp_path, **values) -> Settings:
    base = dict(
        PROJECT_ROOT=str(tmp_path),
        SYSTEM_NAME="Tessera",
        CREATOR_NAME="Alice",
        PERSONA_PATH=str(tmp_path / "absent-persona.yaml"),
    )
    base.update(values)
    return Settings(_env_file=None, **base)


def test_birth_date_written_once_never_overwritten(tmp_path):
    env = tmp_path / "project.env"
    env.write_text("# kept comment\nMTC_SYSTEM_NAME=Tessera\n", encoding="utf-8")
    settings = make_settings(tmp_path)
    day, written = record_birth_date(env, settings=settings, clock=lambda: INSTANT)
    assert (day, written) == ("2000-01-01", True)
    assert read_env(env)["MTC_SYSTEM_BIRTH_DATE"] == "2000-01-01"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert env.read_text(encoding="utf-8").startswith("# kept comment\n")
    before = env.read_bytes()
    later = datetime(2000, 6, 15, 12, 0, tzinfo=timezone.utc)
    again = record_birth_date(env, settings=settings, clock=lambda: later)
    assert again == ("2000-01-01", False)
    assert env.read_bytes() == before
    # A date already in the loaded Settings is kept too; the file is untouched.
    other = tmp_path / "other.env"
    preset = make_settings(tmp_path, SYSTEM_BIRTH_DATE="2000-02-02")
    assert record_birth_date(other, settings=preset, clock=lambda: later) == (
        "2000-02-02",
        False,
    )
    assert not other.exists()


@pytest.mark.parametrize(
    "zone, expected",
    [
        ("UTC", "2000-01-01"),
        ("Europe/Berlin", "2000-01-02"),
        ("Asia/Tokyo", "2000-01-02"),
    ],
)
def test_birth_date_uses_timezone_calendar_day(tmp_path, zone, expected):
    env = tmp_path / "zone.env"
    settings = make_settings(tmp_path, TIMEZONE=zone)
    assert record_birth_date(env, settings=settings, clock=lambda: INSTANT) == (
        expected,
        True,
    )
    with pytest.raises(ValueError):
        record_birth_date(
            tmp_path / "naive.env",
            settings=settings,
            clock=lambda: datetime(2000, 1, 1, 12, 0),
        )


def test_identity_block_gets_install_date_sentence(tmp_path):
    settings = make_settings(tmp_path)
    plain = render_seed_blocks(None, None, settings=settings)
    dated = render_seed_blocks(None, "2000-01-01", settings=settings)
    sentence = INSTALL_DATE_SENTENCE.format(date="2000-01-01")
    assert sentence == "This system was first initialized on 2000-01-01."
    assert dated["identity"] == f"{plain['identity']} {sentence}"
    assert dated["persona"] == plain["persona"]
    assert dated["human"] == plain["human"]
    assert "birthday" not in dated["identity"].lower()
    with pytest.raises(ValueError):
        render_seed_blocks(None, "not-a-date", settings=settings)


def test_seed_blocks_keep_the_neutral_rules(tmp_path):
    settings = make_settings(tmp_path)
    blocks = render_seed_blocks(None, "2000-01-01", settings=settings)
    for name, text in blocks.items():
        lowered = text.lower()
        assert "trust" not in lowered, name
        assert "bond" not in lowered, name
    dates = [
        (name, found)
        for name, text in blocks.items()
        for found in ISO_DATE.findall(text)
    ]
    assert dates == [("identity", "2000-01-01")]
    assert blocks["identity"].count("2000") == 1
    assert "does not claim certainty about its own consciousness" in blocks["identity"]


class _Manager:
    def __init__(self, existing=None):
        self.blocks = dict(existing or {})
        self.calls = []

    async def seed_core_memory(self, blocks=None, overwrite=False):
        self.calls.append((dict(blocks), overwrite))
        written = {}
        for name, text in blocks.items():
            if name in self.blocks and not overwrite:
                written[name] = False
                continue
            self.blocks[name] = text
            written[name] = True
        return written


class _Stack:
    def __init__(self, manager):
        self.manager = manager
        self.closed = False

    async def close(self):
        self.closed = True


def test_seed_keeps_existing_blocks(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, POSTGRES_ENABLED=True)
    manager = _Manager(existing={"human": "kept by the operator"})
    stacks = []

    async def build(active):
        assert active.POSTGRES_ENABLED and active.SYSTEM_BIRTH_DATE == "2000-01-01"
        stacks.append(_Stack(manager))
        return stacks[-1]

    monkeypatch.setattr(seed_db, "_build_stack", build)
    report = asyncio.run(seed_stores(settings, birth_date="2000-01-01"))
    assert report.core_blocks == {"identity": True, "persona": True, "human": False}
    assert manager.blocks["human"] == "kept by the operator"
    [(blocks, overwrite)] = manager.calls
    assert overwrite is False
    assert blocks["identity"].endswith("first initialized on 2000-01-01.")
    assert stacks[0].closed
    assert "graph" in report.skipped


def test_seed_keeps_existing_blocks_in_a_real_store(tmp_path, monkeypatch):
    from mtc.memory.factory import build_memory_manager

    settings = make_settings(tmp_path, POSTGRES_ENABLED=True)
    built = {}

    async def build(active):
        stack = await build_memory_manager(
            active, postgres=False, vectors=False, graph=False
        )
        await stack.manager.seed_core_memory(blocks={"human": "kept"}, overwrite=True)
        built["stack"] = stack
        return stack

    monkeypatch.setattr(seed_db, "_build_stack", build)
    report = asyncio.run(seed_stores(settings, birth_date="2000-01-01"))
    assert report.core_blocks["human"] is False
    assert report.core_blocks["identity"] is True
    core = built["stack"].manager.core_memory
    assert core["human"] == "kept"
    assert "first initialized on 2000-01-01" in core["identity"]


def test_graph_seed_only_when_neo4j_enabled(tmp_path, monkeypatch):
    calls = []

    class Graph:
        def __init__(self, active):
            self.active = active

        def connect(self):
            calls.append("connect")

        def seed_core_entities(self, include_creator=True):
            calls.append(("seed", include_creator, self.active.SYSTEM_BIRTH_DATE))
            return {"persons_created": 2, "concepts_created": 3}

        def close(self):
            calls.append("close")

    monkeypatch.setattr(seed_db, "_build_graph", Graph)
    off = asyncio.run(seed_stores(make_settings(tmp_path), birth_date="2000-01-01"))
    assert calls == [] and "Neo4j is disabled" in off.skipped["graph"]
    on = make_settings(tmp_path, NEO4J_ENABLED=True)
    report = asyncio.run(seed_stores(on, birth_date="2000-01-01"))
    assert calls == ["connect", ("seed", True, "2000-01-01"), "close"]
    assert report.graph == {"persons_created": 2, "concepts_created": 3}
    calls.clear()
    unasked = asyncio.run(seed_stores(on, birth_date="2000-01-01", graph=False))
    assert calls == [] and unasked.skipped["graph"] == "graph seeding was not requested"


def test_no_store_enabled_seeds_nothing_and_says_so(tmp_path, monkeypatch):
    async def never(active):
        raise AssertionError("no stack may be built without PostgreSQL")

    def no_graph(active):
        raise AssertionError("no graph may be built without Neo4j")

    monkeypatch.setattr(seed_db, "_build_stack", never)
    monkeypatch.setattr(seed_db, "_build_graph", no_graph)
    report = asyncio.run(seed_stores(make_settings(tmp_path), birth_date="2000-01-01"))
    assert report.core_blocks == {} and report.graph == {}
    assert "seeded at run time from the persona" in report.skipped["core_memory"]
    assert "Neo4j is disabled" in report.skipped["graph"]
