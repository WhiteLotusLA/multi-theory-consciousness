"""The runtime composition root: in-memory by default, enable keys select
backends, presets, narration, the trust lookup, the flight sides, degraded
brains, background tasks, the PyPhi home and the lifecycle."""

import json
import os
import secrets
import subprocess
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import pytest

from mtc.brain.brain import LLMBrain
from mtc.brain.protocol import ConfigurationError
from mtc.brain.providers.mock_provider import MockProvider
from mtc.brain.providers.registry import ProviderRegistry
from mtc.consciousness import pyphi_bridge
from mtc.consciousness.emotion_extractor import InMemoryEmotionAverages
from mtc.core import config as config_module
from mtc.core.config import Settings
from mtc.core.embeddings import HashEmbedder
from mtc.core.logger import reset_logging
from mtc.memory.inmemory import (
    InMemoryGraphStore,
    InMemoryVectorStore,
    InMemoryWorkingMemory,
)
from mtc.memory.protocols import StoreUnavailable
from mtc.pipeline.conversation_pipeline import GatedBrain
from mtc.pipeline.cycle_augmenter import CycleAugmenterConfig
from mtc.pipeline.runtime import (
    HUB_NAMES,
    BackgroundTask,
    Runtime,
    StoreBuilders,
    build_runtime,
)

ROOT = Path(__file__).resolve().parents[2]
TOKEN_ENV = "MTC_TEST_RUNTIME_CREATOR_TOKEN"
CLIENTS = (
    "asyncpg",
    "pymongo",
    "redis",
    "neo4j",
    "qdrant_client",
    "httpx",
    "anthropic",
)


def _settings(tmp_path: Path, **overrides) -> Settings:
    speakers = tmp_path / "speakers.yaml"
    if not speakers.exists():
        speakers.write_text(
            "speakers:\n"
            "  - name: Operator\n"
            "    role: creator\n"
            "    trust: 0.9\n"
            f"    token_env: {TOKEN_ENV}\n",
            encoding="utf-8",
        )
    values: Dict[str, Any] = dict(
        _env_file=None,
        PROJECT_ROOT=str(ROOT),
        DATA_DIRECTORY=str(tmp_path / "data"),
        STATE_DIR=str(tmp_path / "state"),
        LOG_DIRECTORY=str(tmp_path / "logs"),
        RESULTS_DIR=str(tmp_path / "results"),
        AGENT_SPACE_DIR=str(tmp_path / "agent_space"),
        IDENTITY_STATEMENT_PATH=str(tmp_path / "absent.md"),
        SPEAKERS_PATH=str(speakers),
        LLM_PROVIDER="mock",
        CTM_PROFILE="demo",
        SYSTEM_NAME="Testagent",
        CREATOR_NAME="Operator",
        TRUSTED_SPEAKERS="Operator",
        DEVICE="cpu",
        DEBUG=True,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(autouse=True)
def _release_logging():
    """build_runtime configures the mtc logger; give it back to the suite."""
    yield
    reset_logging()


@pytest.fixture(autouse=True)
def _restore_pyphi_home_record(monkeypatch):
    """build_runtime records its Settings in the PyPhi bridge for the first
    import; each test starts with no record and leaves none behind."""
    monkeypatch.setattr(pyphi_bridge, "_home_settings", None)


@pytest.fixture
def token(monkeypatch):
    value = secrets.token_hex(16)
    monkeypatch.setenv(TOKEN_ENV, value)
    return value


async def _build(tmp_path: Path, **kwargs) -> Runtime:
    settings = kwargs.pop("settings", None) or _settings(tmp_path)
    return await build_runtime(
        settings,
        embedder=kwargs.pop("embedder", HashEmbedder(dim=128)),
        allow_non_semantic=kwargs.pop("allow_non_semantic", True),
        substrates=kwargs.pop("substrates", False),
        **kwargs,
    )


_SUBPROCESS = r"""
import asyncio, json, sys
from mtc.pipeline.runtime import build_runtime

async def main():
    runtime = await build_runtime(substrates=False)
    report = {
        "clients": [c for c in json.loads(sys.argv[1]) if c in sys.modules],
        "stores": {k: v["backend"] for k, v in runtime.stores.items()},
        "embedder": runtime.embedder_kind,
        "provider": runtime.brain.registry.active_id,
        "registered": sorted(runtime.registered),
    }
    await runtime.stop()
    print("REPORT " + json.dumps(report))

asyncio.run(main())
"""


def test_default_runtime_is_in_memory_and_builds_no_client(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTC_")}
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
            "MTC_PROJECT_ROOT": str(ROOT),
            "MTC_DATA_DIRECTORY": str(tmp_path / "data"),
            "MTC_STATE_DIR": str(tmp_path / "state"),
            "MTC_LOG_DIRECTORY": str(tmp_path / "logs"),
            "MTC_RESULTS_DIR": str(tmp_path / "results"),
            "MTC_AGENT_SPACE_DIR": str(tmp_path / "agent_space"),
            "MTC_IDENTITY_STATEMENT_PATH": str(tmp_path / "absent.md"),
            "MTC_SPEAKERS_PATH": str(tmp_path / "absent.yaml"),
            "MTC_LLM_PROVIDER": "mock",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS, json.dumps(list(CLIENTS))],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    line = [ln for ln in completed.stdout.splitlines() if ln.startswith("REPORT ")]
    assert len(line) == 1, completed.stdout[-2000:]
    report = json.loads(line[0][len("REPORT ") :])
    assert report["clients"] == [], report
    assert set(report["stores"].values()) == {"memory"}
    assert report["provider"] == "mock"
    assert "gwt" in report["registered"] and "orchestrator" in report["registered"]


class FakeConnection:
    async def execute(self, *args, **kwargs):
        return "INSERT 0 1"

    def transaction(self):
        return _null_context()


@asynccontextmanager
async def _null_context():
    yield None


class FakePool:
    """Answers like an asyncpg pool with nothing in it."""

    def __init__(self):
        self.calls: List[str] = []
        self.closed = False

    def acquire(self, *args, **kwargs):
        self.calls.append("acquire")
        return _connection_context()

    async def execute(self, query, *args, **kwargs):
        self.calls.append("execute")
        return "INSERT 0 1"

    async def fetch(self, query, *args, **kwargs):
        self.calls.append("fetch")
        return []

    async def fetchrow(self, query, *args, **kwargs):
        self.calls.append("fetchrow")
        return None

    async def fetchval(self, query, *args, **kwargs):
        self.calls.append("fetchval")
        return 1

    async def close(self):
        self.closed = True


@asynccontextmanager
async def _connection_context():
    yield FakeConnection()


class FakeRedisManager(InMemoryWorkingMemory):
    """A working memory that also serves emotion averages, like the real one."""

    def __init__(self):
        super().__init__(capacity=7)
        self._averages = InMemoryEmotionAverages()
        self.disconnected = False

    async def record(self, session_id, sample):
        await self._averages.record(session_id, sample)

    async def get_averages(self, session_id):
        return await self._averages.get_averages(session_id)

    async def sample_count(self, session_id):
        return await self._averages.sample_count(session_id)

    async def disconnect(self):
        self.disconnected = True


@pytest.mark.asyncio
async def test_enable_keys_select_backends(tmp_path, token):
    from tests.memory.test_mongo_stores import FakeMongoClient

    from mtc.memory.conversation_store import PostgresConversationRepository
    from mtc.memory.resilient_wrappers import ResilientMongoStore, ResilientRedisManager

    calls: Dict[str, int] = {}
    pool = FakePool()
    mongo = FakeMongoClient()
    redis = FakeRedisManager()
    embedder = HashEmbedder(dim=128)
    vectors = InMemoryVectorStore(embedder, allow_non_semantic=True)
    graph = InMemoryGraphStore()

    def count(name):
        calls[name] = calls.get(name, 0) + 1

    async def make_pool(cfg):
        count("postgres")
        return pool

    def make_mongo(cfg):
        count("mongodb")
        return mongo

    def make_redis(cfg):
        count("redis")
        return redis

    def make_vectors(cfg, emb):
        count("qdrant")
        assert emb is embedder
        return vectors

    def make_graph(cfg):
        count("neo4j")
        return graph

    builders = StoreBuilders(
        postgres_pool=make_pool,
        mongo_client=make_mongo,
        redis_manager=make_redis,
        vector=make_vectors,
        graph=make_graph,
    )
    # every key off: no builder is called, every backend is memory
    off = await _build(tmp_path, builders=builders, embedder=embedder)
    assert calls == {}
    assert {v["backend"] for v in off.stores.values()} == {"memory"}
    assert all(v["enabled"] is False for v in off.stores.values())
    assert off.store_clients == []
    await off.stop()

    on = await _build(
        tmp_path,
        settings=_settings(
            tmp_path,
            POSTGRES_ENABLED=True,
            MONGODB_ENABLED=True,
            REDIS_ENABLED=True,
            QDRANT_ENABLED=True,
            NEO4J_ENABLED=True,
        ),
        builders=builders,
        embedder=embedder,
    )
    assert calls == {"postgres": 1, "mongodb": 1, "redis": 1, "qdrant": 1, "neo4j": 1}
    assert isinstance(on.repository, PostgresConversationRepository)
    assert isinstance(on.turn_store, ResilientMongoStore)
    assert isinstance(on.working_memory, ResilientRedisManager)
    assert on.memory.vector_store is vectors
    assert on.memory.graph is graph
    assert on.data_manager is not None
    assert on.stores["postgres"]["backend"] == "postgres"
    assert on.stores["mongodb"]["backend"] == "mongodb"
    assert on.stores["redis"]["backend"] == "redis"
    assert on.stores["qdrant"]["error"] is None and on.stores["qdrant"]["enabled"]
    assert on.stores["neo4j"]["error"] is None and on.stores["neo4j"]["enabled"]
    assert [name for name, _ in on.store_clients] == [
        "postgres_pool",
        "mongo_client",
        "redis_manager",
    ]
    await on.stop()
    assert redis.disconnected is True


@pytest.mark.asyncio
async def test_failed_store_build_is_reported_and_falls_back(tmp_path, token):
    async def broken_pool(cfg):
        raise StoreUnavailable("no server (test)")

    runtime = await _build(
        tmp_path,
        settings=_settings(tmp_path, POSTGRES_ENABLED=True),
        builders=StoreBuilders(postgres_pool=broken_pool),
    )
    report = runtime.stores["postgres"]
    assert report["enabled"] is True
    assert report["backend"] == "memory"
    assert "StoreUnavailable" in report["error"]
    assert runtime.memory is not None
    await runtime.stop()


@pytest.mark.asyncio
async def test_concept_graph_in_memory_unless_neo4j_enabled(tmp_path, token):
    """With NEO4J_ENABLED false the concept graph lives in process memory (a
    None graph would mean no graph at all); a Neo4j build that fails is
    reported with its error and replaced by the in-memory graph, loudly."""
    runtime = await _build(tmp_path)
    graph = runtime.memory.graph
    assert isinstance(graph, InMemoryGraphStore)
    assert runtime.memory.manager.graph is graph
    assert runtime.memory.manager.concept_matcher is not None
    assert runtime.pipeline.post_turn.graph is graph
    assert runtime.pruner.graph is graph
    report = runtime.stores["neo4j"]
    assert report["enabled"] is False
    assert report["backend"] == "memory" and report["error"] is None
    assert runtime.memory.components["graph"]["backend"] == "memory"
    result = await runtime.pipeline.send_message(
        "The tulips in the garden opened today.", speaker="Alice"
    )
    assert result.success is True
    assert "tulips" in graph.list_concept_names()
    await runtime.stop()

    def broken_graph(cfg):
        raise StoreUnavailable("no graph server (test)")

    fallback = await _build(
        tmp_path,
        settings=_settings(tmp_path, NEO4J_ENABLED=True),
        builders=StoreBuilders(graph=broken_graph),
    )
    report = fallback.stores["neo4j"]
    assert report["enabled"] is True
    assert report["backend"] == "memory"
    assert "StoreUnavailable" in report["error"]
    assert isinstance(fallback.memory.graph, InMemoryGraphStore)
    assert fallback.memory.manager.graph is fallback.memory.graph
    await fallback.stop()


@pytest.fixture
def process_settings_reads(monkeypatch):
    """Make every ``get_settings`` binding in the ``mtc`` package record its
    caller and raise; restore every binding afterwards, including those of
    modules first imported while the fixture was active."""
    original = config_module.get_settings
    reads: List[str] = []

    def forbidden():
        caller = sys._getframe(1)
        reads.append(f"{Path(caller.f_code.co_filename).name}:{caller.f_lineno}")
        raise AssertionError("the process Settings were read")

    monkeypatch.setattr(config_module, "get_settings", forbidden)
    for name, module in list(sys.modules.items()):
        if name == "mtc" or name.startswith("mtc."):
            if getattr(module, "get_settings", None) is original:
                monkeypatch.setattr(module, "get_settings", forbidden)
    yield reads
    for module in list(sys.modules.values()):
        if getattr(module, "get_settings", None) is forbidden:
            setattr(module, "get_settings", original)


@pytest.mark.asyncio
async def test_runtime_reads_only_the_settings_it_was_built_with(
    tmp_path, token, process_settings_reads
):
    """Built from a non-default Settings object, the runtime never falls back
    to the process Settings: not while it builds, not in an unauthenticated or
    an authenticated turn (the brain's one budget pass, the neural
    orchestrator's accelerator release and its thought machine status), not in
    the background ticks, the status page or the stop."""
    try:
        import torch  # noqa: F401  (loaded, so the accelerator release runs)
    except ImportError:
        pass
    cfg = _settings(tmp_path, HISTORY_TOKEN_BUDGET=321, CHARS_PER_TOKEN=2.5)
    runtime = await _build(tmp_path, settings=cfg)
    claimed = await runtime.pipeline.send_message(
        "The tulips in the garden opened today.", speaker="Alice"
    )
    verified = await runtime.pipeline.send_message(
        "Which flowers come next?", speaker="Operator", speaker_token=token
    )
    assert claimed.success is True and verified.success is True
    budget = runtime.brain.last_budget
    assert budget is not None and budget.budget_tokens <= 321
    for task in runtime.background:
        if task.name in ("ctm", "baseline_heartbeat", "wellbeing_sampler"):
            await task.tick()
    status = runtime.status()
    assert status["ctm"]["status"] in ("running", "stopped", "paused")
    await runtime.stop()
    assert process_settings_reads == []


@pytest.mark.asyncio
async def test_runtime_puts_the_pyphi_home_under_its_own_data_directory(
    tmp_path, token, monkeypatch
):
    """build_runtime records its own Settings in the PyPhi bridge: nothing is
    imported or created while it builds, and PyPhi's first import (its config
    copy, cache and log) lands in the runtime's <DATA_DIRECTORY>/pyphi, never
    under the process Settings' directories or the working directory."""
    # The bridge as before PyPhi's first import, with a stand-in import that
    # records the directory it ran in and writes a cache there, as PyPhi does.
    monkeypatch.delitem(sys.modules, "pyphi", raising=False)
    monkeypatch.setattr(pyphi_bridge, "_pyphi_checked", False)
    monkeypatch.setattr(pyphi_bridge, "_pyphi_module", None)
    monkeypatch.setattr(pyphi_bridge, "_home_in_use", None)
    imported_in: List[Path] = []
    fake = types.ModuleType("pyphi")

    def fake_import():
        imported_in.append(Path.cwd().resolve())
        (Path.cwd() / "__pyphi_cache__").mkdir()
        return fake

    monkeypatch.setattr(pyphi_bridge, "_import_pyphi", fake_import)
    process_root = tmp_path / "process"
    process = Settings(
        _env_file=None, PROJECT_ROOT=str(process_root), DATA_DIRECTORY="data"
    )
    monkeypatch.setattr(config_module, "_settings", process)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    runtime = await _build(tmp_path)
    home = tmp_path / "data" / "pyphi"
    assert imported_in == []
    assert not home.exists()

    assert pyphi_bridge.pyphi_available() is True
    assert imported_in == [home.resolve()]
    assert (home / "pyphi_config.yml").is_file()
    assert (home / "__pyphi_cache__").is_dir()
    assert Path.cwd().resolve() == work.resolve()
    assert sorted(work.iterdir()) == []
    assert not process_root.exists()
    await runtime.stop()


@pytest.mark.asyncio
async def test_ctm_store_mongo_requires_mongodb_enabled(tmp_path, token):
    with pytest.raises(ConfigurationError) as info:
        await _build(tmp_path, settings=_settings(tmp_path, CTM_STORE="mongo"))
    assert "CTM_STORE" in str(info.value)
    assert "MONGODB_ENABLED" in str(info.value)


@pytest.mark.asyncio
async def test_preset_applies_to_workspace_and_cycle(tmp_path, token):
    runtime = await _build(
        tmp_path, settings=_settings(tmp_path, PIPELINE_PRESET="legacy_v04")
    )
    assert runtime.workspace.competition_mode == "flat"
    assert runtime.workspace.bottleneck.thalamic_gate is None
    assert runtime.augmenter.config == CycleAugmenterConfig.all_off()
    assert runtime.workspace.damasio.protoself.circadian == "v04_sawtooth"
    assert runtime.status()["preset"] == "legacy_v04"
    await runtime.stop()
    with pytest.raises(ConfigurationError) as info:
        await _build(tmp_path, settings=_settings(tmp_path, PIPELINE_PRESET="no_such"))
    assert "PIPELINE_PRESET" in str(info.value)


@pytest.mark.asyncio
async def test_live_pipeline_uses_wall_clock_circadian_unless_overridden(
    tmp_path, token
):
    runtime = await _build(
        tmp_path, settings=_settings(tmp_path, TIMEZONE="Asia/Tokyo")
    )
    protoself = runtime.workspace.damasio.protoself
    assert protoself.circadian == "wall_clock"
    assert protoself.timezone == "Asia/Tokyo"
    assert runtime.workspace.competition_mode == "hierarchical"
    assert runtime.augmenter.config == CycleAugmenterConfig()
    await runtime.stop()
    overridden = await _build(
        tmp_path, overrides={"damasio": {"circadian": "sinusoid"}}
    )
    assert overridden.workspace.damasio.protoself.circadian == "sinusoid"
    await overridden.stop()


@pytest.mark.asyncio
async def test_narration_flags_applied_to_modules(tmp_path, token):
    quiet = await _build(tmp_path)
    assert quiet.workspace.damasio.narrate_somatic is False
    assert quiet.workspace.damasio.narrate_body_drives is False
    assert quiet.pad_model.narrate_affect_line is False
    assert quiet.theory_of_mind.narrate_speaker_model is False
    assert quiet.narration_applied and not any(quiet.narration_applied.values())
    assert quiet.orchestrator.narration.as_dict() == quiet.narration.as_dict()
    await quiet.stop()
    loud = await _build(tmp_path, settings=_settings(tmp_path, NARRATE_ALL=True))
    assert loud.workspace.damasio.narrate_somatic is True
    assert loud.pad_model.narrate_affect_line is True
    assert loud.theory_of_mind.narrate_speaker_model is True
    assert all(loud.narration_applied.values())
    await loud.stop()


@pytest.mark.asyncio
async def test_speaker_trust_lookup_wired_to_tom_and_compass(tmp_path, token):
    runtime = await _build(tmp_path)
    trust = runtime.speaker_trust
    default = runtime.settings.DEFAULT_SPEAKER_TRUST
    assert runtime.theory_of_mind._trust("Operator") == default
    assert runtime.compass.trust_for("Operator") == default
    verified = trust.verify("Operator", token)
    assert verified.authenticated and verified.trust == 0.9
    with trust.use_identity(verified):
        assert runtime.theory_of_mind._trust("Operator") == 0.9
        assert runtime.compass.trust_for("Operator") == 0.9
        assert runtime.theory_of_mind._trust("Alice") == default
    claimed = trust.verify("Operator", None)
    with trust.use_identity(claimed):
        assert runtime.theory_of_mind._trust("Operator") == default
        assert runtime.compass.trust_for("Operator") == default
    await runtime.stop()


@pytest.mark.asyncio
async def test_background_users_get_background_flight(tmp_path, token):
    runtime = await _build(
        tmp_path, settings=_settings(tmp_path, AUTONOMOUS_LIFE_ENABLED=True)
    )
    for user in (
        runtime.ctm.enhancer.brain,
        runtime.sleep_scheduler.brain,
        runtime.autonomous_life.brain,
        runtime.consolidation.brain,
        runtime.neural.brain,
    ):
        assert isinstance(user, GatedBrain) and user.background is True
        assert user.brain is runtime.brain
    for user in (runtime.compass.brain, runtime.pipeline.summarizer._brain):
        assert isinstance(user, GatedBrain) and user.background is False
    assert runtime.pipeline.brain is runtime.brain
    assert runtime.pipeline.flight is runtime.flight
    assert runtime.ctm.contention_guard is not None
    await runtime.stop()


@pytest.mark.asyncio
async def test_brain_degraded_still_builds(tmp_path, token):
    cfg = _settings(tmp_path, DEBUG=False)
    registry = ProviderRegistry(
        state_path=tmp_path / "data" / "provider_state.json", settings=cfg
    )
    registry.register(MockProvider(healthy=False))
    registry.set_active("mock")
    brain = LLMBrain(registry=registry, settings=cfg)
    runtime = await _build(tmp_path, settings=cfg, brain=brain)
    assert runtime.brain is brain
    assert runtime.status()["brain"]["available"] is False
    await runtime.start()
    result = await runtime.pipeline.send_message(
        "hello", speaker="Operator", speaker_token=token
    )
    assert result.success is False
    assert result.error_code == "brain_unavailable"
    await runtime.stop()


@pytest.mark.asyncio
async def test_background_tasks_registered_per_settings(tmp_path, token):
    default = await _build(tmp_path)
    names = [task.name for task in default.background]
    assert names == [
        "ctm",
        "sleep_scheduler",
        "baseline_heartbeat",
        "wellbeing_sampler",
        "state_save",
        "session_pruning",
    ]
    assert all(isinstance(task, BackgroundTask) for task in default.background)
    await default.stop()
    full = await _build(
        tmp_path,
        settings=_settings(
            tmp_path,
            AUTONOMOUS_LIFE_ENABLED=True,
            PHI_IDLE_SAMPLE_S=5.0,
            SELF_ASSESSMENT_INTERVAL_S=7.0,
        ),
    )
    names = {task.name: task.interval_s for task in full.background}
    assert "autonomous_life" in names
    assert names["phi_idle_sampler"] == 5.0
    assert names["self_assessment"] == 7.0
    await full.stop()
    lean = await _build(
        tmp_path,
        settings=_settings(tmp_path, CTM_ENABLED=False, SLEEP_SCHEDULER_ENABLED=False),
    )
    names = [task.name for task in lean.background]
    assert "ctm" not in names and "sleep_scheduler" not in names
    assert lean.ctm is None and lean.sleep_scheduler is None
    assert lean.pipeline.thought_store is None
    await lean.stop()


@pytest.mark.asyncio
async def test_background_task_counts_failures_and_runs_once(tmp_path):
    calls = []

    async def run_once():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("tick bug")
        return len(calls)

    task = BackgroundTask("probe", 60.0, run_once)
    assert await task.tick() == 1
    assert await task.tick() is None
    assert task.failures == 1 and task.runs == 2
    assert task.last_error.startswith("RuntimeError")
    await task.start()
    assert task.running
    await task.stop()
    assert not task.running


@pytest.mark.asyncio
async def test_start_and_stop_order(tmp_path, token):
    runtime = await _build(tmp_path)
    restored = await runtime.start()
    assert restored == {name: False for name in runtime.registered}
    assert runtime.ctm.running is True
    assert all(task.status()["running"] for task in runtime.background)
    report = await runtime.stop()
    assert runtime.last_stop_order == [
        "background_tasks",
        "conversations",
        "hub_save",
        "neural",
        "memory",
        "store_clients",
        "brain",
    ]
    assert runtime.ctm.running is False
    assert not any(task.running for task in runtime.background)
    assert runtime.stopped is True
    assert "saved" in report


@pytest.mark.asyncio
async def test_stop_saves_state_and_ends_conversations(tmp_path, token):
    runtime = await _build(tmp_path)
    await runtime.start()
    result = await runtime.pipeline.send_message("Hello there", speaker="Alice")
    assert result.success is True
    assert runtime.pipeline.sessions.status()["sessions"] == 1
    report = await runtime.stop()
    assert report["conversations_ended"] == 1
    assert runtime.pipeline.sessions.status()["sessions"] == 0
    state_dir = Path(runtime.settings.STATE_DIR)
    assert (state_dir / "manifest.json").is_file()
    assert runtime.hub.get_status()["has_saved_state"] is True
    saved = report["saved"]["modules"]
    assert "orchestrator" in saved and "gwt" in saved
    assert set(saved) <= set(HUB_NAMES)


@pytest.mark.asyncio
async def test_background_task_raises_under_debug(tmp_path):
    def run_once():
        raise RuntimeError("tick bug")

    loud = BackgroundTask("probe", 60.0, run_once, settings=_settings(tmp_path))
    with pytest.raises(RuntimeError, match="tick bug"):
        await loud.tick()
    assert loud.failures == 1 and loud.runs == 1
    assert loud.status()["failures"] == 1
    quiet = BackgroundTask(
        "probe", 60.0, run_once, settings=_settings(tmp_path, DEBUG=False)
    )
    assert await quiet.tick() is None
    assert quiet.failures == 1


@pytest.mark.asyncio
async def test_thought_listener_failure_counted_and_raises_under_debug(tmp_path):
    received = []

    def failing(payload):
        raise ValueError("listener bug")

    async def working(payload):
        received.append(payload)

    quiet = Runtime(settings=_settings(tmp_path, DEBUG=False))
    quiet.add_thought_listener(failing)
    quiet.add_thought_listener(working)
    await quiet.broadcast_thought({"content": "a thought"})
    assert received == [{"content": "a thought"}]
    assert quiet.listener_failures == 1
    assert quiet.status()["thought_listener_failures"] == 1

    loud = Runtime(settings=_settings(tmp_path))
    loud.add_thought_listener(failing)
    with pytest.raises(ValueError, match="listener bug"):
        await loud.broadcast_thought({"content": "a thought"})
    assert loud.listener_failures == 1


@pytest.mark.asyncio
async def test_turns_succeed_once_the_provider_recovers_without_a_restart(
    tmp_path, token
):
    cfg = _settings(tmp_path, DEBUG=False)
    registry = ProviderRegistry(
        state_path=tmp_path / "data" / "provider_state.json", settings=cfg
    )
    provider = MockProvider(healthy=False)
    registry.register(provider)
    registry.set_active("mock")
    brain = LLMBrain(registry=registry, settings=cfg)
    # The interval itself is pinned in tests/brain; here the next call may
    # re-check at once.
    brain.health_recheck_s = 0.0
    runtime = await _build(tmp_path, settings=cfg, brain=brain)
    await runtime.start()
    try:
        first = await runtime.pipeline.send_message(
            "hello", speaker="Operator", speaker_token=token
        )
        assert first.success is False
        assert first.error_code == "brain_unavailable"
        provider.healthy = True
        second = await runtime.pipeline.send_message(
            "hello again", speaker="Operator", speaker_token=token
        )
        assert second.success is True
        assert second.response
        assert runtime.status()["brain"]["available"] is True
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_ctm_level_is_unavailable_until_the_baseline_is_measured(tmp_path, token):
    runtime = await _build(tmp_path)
    try:
        assert runtime.ctm is not None and runtime.level_cache is not None
        baseline = runtime.orchestrator.baseline
        # the shape an insufficient assessment leaves before any guard: a
        # 0.0 overall under status insufficient_data
        baseline.overall_score = 0.0
        baseline.status = "insufficient_data"
        await runtime.ctm.tick()
        status = runtime.ctm.get_status()
        assert status["consciousness_level"] is None
        assert status["modulation"] == "unavailable"
        assert status["modulation_factor"] == 1.0
        # a baseline that measured nothing yet reads no level either
        runtime.level_cache.min_interval_s = 0.0
        baseline.overall_score = None
        baseline.status = "unmeasured"
        assert await runtime.level_cache.level() is None
        # a measured baseline hands its score to the machine
        baseline.overall_score = 0.42
        baseline.status = "partial"
        assert await runtime.level_cache.level() == pytest.approx(0.42)
    finally:
        await runtime.stop()


@pytest.mark.asyncio
async def test_sleep_prune_dry_run_setting_reaches_the_scheduler(tmp_path, token):
    default = await _build(tmp_path)
    try:
        assert default.settings.SLEEP_PRUNE_DRY_RUN is False
        assert default.sleep_scheduler is not None
        assert default.sleep_scheduler.config.prune_dry_run is False
    finally:
        await default.stop()
    dry = await _build(tmp_path, settings=_settings(tmp_path, SLEEP_PRUNE_DRY_RUN=True))
    try:
        assert dry.sleep_scheduler.config.prune_dry_run is True
    finally:
        await dry.stop()


@pytest.mark.asyncio
async def test_compass_has_no_brain_unless_deep_mode_is_on(tmp_path, token):
    deep = await _build(tmp_path, settings=_settings(tmp_path, DEBUG=False))
    try:
        assert deep.settings.TRUST_COMPASS_DEEP_MODE is True
        assert isinstance(deep.compass.brain, GatedBrain)
        assert deep.compass.brain.background is False
        assert deep.compass.status()["deep_mode"] is True
        provider = deep.brain.registry.active()
        before = len(provider.requests)
        result = await deep.pipeline.send_message("What is the capital of France?")
        assert result.success is True
        # the neutral message passed the keyword screen: no reflection call,
        # only the turn's own request
        assert len(provider.requests) - before == 1
        compass = deep.pipeline.status()["last_turn"]["safety_compass"]
        assert compass == {"status": "not_needed", "method": None}
    finally:
        await deep.stop()
    keyword = await _build(
        tmp_path,
        settings=_settings(tmp_path, DEBUG=False, TRUST_COMPASS_DEEP_MODE=False),
    )
    try:
        assert keyword.compass.brain is None
        status = keyword.compass.status()
        assert status["deep_mode"] is False and status["backend"] == "keyword"
    finally:
        await keyword.stop()


@pytest.mark.asyncio
async def test_runtime_hands_its_goal_system_to_learning(tmp_path, token):
    runtime = await _build(tmp_path)
    try:
        assert runtime.goals is not None
        assert runtime.learning.goals is runtime.goals
    finally:
        await runtime.stop()


async def _failing_neural_runtime(tmp_path, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime = await _build(tmp_path, settings=_settings(tmp_path, **overrides))
    calls: List[str] = []

    async def broken_shutdown():
        calls.append("neural")
        raise RuntimeError("shutdown bug")

    memory_close = runtime.memory.close
    brain_shutdown = runtime.brain.shutdown

    async def close_memory():
        calls.append("memory")
        await memory_close()

    async def shut_brain():
        calls.append("brain")
        await brain_shutdown()

    runtime.neural.shutdown = broken_shutdown
    runtime.memory.close = close_memory
    runtime.brain.shutdown = shut_brain
    pool = FakePool()
    runtime.store_clients.append(("postgres_pool", pool))
    return runtime, calls, pool


@pytest.mark.asyncio
async def test_stop_continues_after_a_failing_step(tmp_path, token):
    runtime, calls, pool = await _failing_neural_runtime(tmp_path, DEBUG=False)
    await runtime.start()
    report = await runtime.stop()
    assert calls == ["neural", "memory", "brain"]
    assert pool.closed is True
    assert report["errors"] == {"neural": "RuntimeError"}
    assert runtime.stopped is True
    assert runtime.last_stop_order == [
        "background_tasks",
        "conversations",
        "hub_save",
        "memory",
        "store_clients",
        "brain",
    ]
    assert not any(task.running for task in runtime.background)
    # under DEBUG every step still runs, then the first error is raised
    strict, strict_calls, strict_pool = await _failing_neural_runtime(
        tmp_path / "strict", DEBUG=True
    )
    with pytest.raises(RuntimeError, match="shutdown bug"):
        await strict.stop()
    assert strict_calls == ["neural", "memory", "brain"]
    assert strict_pool.closed is True
    assert strict.stopped is True
    assert "neural" not in strict.last_stop_order


@pytest.mark.asyncio
async def test_repository_fallback_is_reported(tmp_path, monkeypatch):
    from mtc.memory import conversation_store
    from mtc.memory.inmemory import InMemoryConversationRepository
    from mtc.pipeline import runtime as runtime_module

    report: Dict[str, Any] = {"enabled": True, "backend": "postgres", "error": None}
    built = runtime_module._conversation_stores(FakePool(), None, report)
    assert isinstance(built[0], conversation_store.PostgresConversationRepository)
    assert report["in_memory_parts"] == []
    assert report["error"] is None

    class Broken:
        def __init__(self, *args, **kwargs):
            raise TypeError("constructor bug")

    monkeypatch.setattr(conversation_store, "PostgresConversationRepository", Broken)
    report = {"enabled": True, "backend": "postgres", "error": None}
    repository, snapshots, audit = runtime_module._conversation_stores(
        FakePool(), None, report
    )
    assert isinstance(repository, InMemoryConversationRepository)
    assert report["in_memory_parts"] == ["conversations", "snapshots", "safety_audit"]
    assert "TypeError" in report["error"]
    # the memory stack still uses Postgres, so the backend keeps its name
    assert report["backend"] == "postgres"
    assert snapshots is not None and audit is not None
    # without a pool every part is in memory by configuration, not by fallback
    report = {"enabled": False, "backend": "memory", "error": None}
    repository, _snapshots, _audit = runtime_module._conversation_stores(
        None, None, report
    )
    assert isinstance(repository, InMemoryConversationRepository)
    assert "in_memory_parts" not in report


@pytest.mark.asyncio
async def test_build_runtime_reports_the_repository_fallback(tmp_path, monkeypatch):
    # Every enable key stays false; the memory factory is wrapped so that the
    # stack carries a fake pool, which is what build_runtime hands to the
    # conversation stores when Postgres is on.
    import dataclasses

    from mtc.memory import conversation_store
    from mtc.memory.inmemory import InMemoryConversationRepository
    from mtc.pipeline import runtime as runtime_module

    real_factory = runtime_module.build_memory_manager

    async def with_fake_pool(*args, **kwargs):
        stack = await real_factory(*args, **kwargs)
        return dataclasses.replace(stack, pool=FakePool())

    monkeypatch.setattr(runtime_module, "build_memory_manager", with_fake_pool)
    runtime = await _build(tmp_path)
    try:
        postgres = runtime.stores["postgres"]
        assert postgres["in_memory_parts"] == []
        assert isinstance(
            runtime.repository, conversation_store.PostgresConversationRepository
        )
    finally:
        await runtime.stop()

    class Broken:
        def __init__(self, *args, **kwargs):
            raise TypeError("constructor bug")

    monkeypatch.setattr(conversation_store, "PostgresConversationRepository", Broken)
    fallback = await _build(tmp_path)
    try:
        postgres = fallback.stores["postgres"]
        assert postgres["in_memory_parts"] == [
            "conversations",
            "snapshots",
            "safety_audit",
        ]
        assert "TypeError" in postgres["error"]
        assert isinstance(fallback.repository, InMemoryConversationRepository)
    finally:
        await fallback.stop()


@pytest.mark.asyncio
async def test_live_workspace_runs_on_the_elapsed_clock(tmp_path, token):
    from datetime import datetime, timedelta, timezone

    moments = [datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)]
    runtime = await _build(tmp_path, clock=lambda: moments[0])
    try:
        workspace = runtime.workspace
        assert workspace.refractory_clock == "elapsed"
        # the workspace reads the runtime's clock, in seconds
        assert workspace.clock() == moments[0].timestamp()
        moments[0] += timedelta(minutes=1)
        assert workspace.clock() == moments[0].timestamp()
    finally:
        await runtime.stop()
    # another gwt key keeps the elapsed default; the operator's choice wins
    other = await _build(tmp_path, overrides={"gwt": {"capacity": 5}})
    try:
        assert other.workspace.refractory_clock == "elapsed"
    finally:
        await other.stop()
    pinned = await _build(
        tmp_path, overrides={"gwt": {"refractory_clock": "per_cycle"}}
    )
    try:
        assert pinned.workspace.refractory_clock == "per_cycle"
    finally:
        await pinned.stop()
