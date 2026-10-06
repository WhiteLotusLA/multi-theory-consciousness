"""Every stateful component is a hub module once, or named as not registered,
and the hub restores what it saved."""

import secrets
from pathlib import Path
from typing import Any, Dict

import pytest

from mtc.core.config import Settings
from mtc.core.embeddings import HashEmbedder
from mtc.core.logger import reset_logging
from mtc.pipeline.runtime import HUB_NAMES, NOT_REGISTERED, Runtime, build_runtime

ROOT = Path(__file__).resolve().parents[2]
TOKEN_ENV = "MTC_TEST_RUNTIME_HUB_TOKEN"


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


@pytest.fixture
def token(monkeypatch):
    value = secrets.token_hex(16)
    monkeypatch.setenv(TOKEN_ENV, value)
    return value


async def _build(tmp_path: Path, **overrides) -> Runtime:
    return await build_runtime(
        _settings(tmp_path, **overrides),
        embedder=HashEmbedder(dim=128),
        allow_non_semantic=True,
        substrates=False,
    )


def _stateful(obj: Any) -> bool:
    return callable(getattr(obj, "to_state_dict", None)) and callable(
        getattr(obj, "from_state_dict", None)
    )


@pytest.mark.asyncio
async def test_every_stateful_component_registered(tmp_path, token):
    runtime = await _build(tmp_path, AUTONOMOUS_LIFE_ENABLED=True)
    excused = {name for name, _ in NOT_REGISTERED}
    assert all(reason.strip() for _, reason in NOT_REGISTERED)
    registered_ids = {id(module) for module in runtime.registered.values()}
    missing = []
    for name, component in runtime.components().items():
        if not _stateful(component):
            continue
        if id(component) in registered_ids or name in excused:
            continue
        missing.append(name)
    assert missing == [], f"stateful runtime fields not registered: {missing}"
    # every registered name is one the plan lists and maps to a built part
    assert set(runtime.registered) <= set(HUB_NAMES)
    for name in ("gwt", "phi", "pad_affect", "orchestrator", "augmenter", "neural"):
        assert name in runtime.registered
    assert {"ctm", "ctm_buffer", "ctm_generator", "sleep_scheduler"} <= set(
        runtime.registered
    )
    assert "autonomous_life" in runtime.registered
    assert "wellbeing" in runtime.registered
    # the excused parts really are stateful parts of the runtime (or absent)
    components = runtime.components()
    for name in (
        "prospective",
        "crossmodal",
        "emotion_averages",
        "identity_continuity",
    ):
        assert _stateful(components[name]), name
    # the hub's own view agrees
    assert runtime.hub.get_status()["registered_modules"] == sorted(runtime.registered)
    await runtime.stop()


@pytest.mark.asyncio
async def test_restore_round_trip_through_temp_state_dir(tmp_path, token):
    first = await _build(tmp_path)
    await first.start()
    result = await first.pipeline.send_message("Hello from the first run", "Alice")
    assert result.success is True
    cycles = first.orchestrator.status()["cycles"]
    assert cycles == 1
    report = await first.stop()
    assert isinstance(report["saved"]["modules"]["orchestrator"], list)

    second = await _build(tmp_path)
    assert second.orchestrator.status()["cycles"] == 0
    restored = await second.start()
    assert restored["orchestrator"] is True
    assert restored["gwt"] is True
    assert second.orchestrator.status()["cycles"] == cycles
    assert second.workspace.cycle_count == first.workspace.cycle_count
    await second.stop()


@pytest.mark.asyncio
async def test_no_double_registration(tmp_path, token):
    runtime = await _build(tmp_path)
    modules = runtime.registered
    assert len({id(m) for m in modules.values()}) == len(modules)
    assert len(set(modules)) == len(modules)
    assert runtime.register("gwt", runtime.workspace) is False
    assert runtime.register("pad_affect", object()) is False  # name already taken
    with pytest.raises(ValueError):
        runtime.register("not_a_hub_name", runtime.pad_model)
    # the late components are absent without substrates and never duplicated
    assert runtime.register_late_components() == []
    assert runtime.register_late_components() == []
    hub_names = runtime.hub.get_status()["registered_modules"]
    assert len(hub_names) == len(set(hub_names)) == len(modules)
    await runtime.stop()
