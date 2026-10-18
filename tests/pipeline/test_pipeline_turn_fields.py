"""A real turn through the runtime on the mock provider: every Optional state
field is set or explained, nothing fails, the provenance snapshot carries the
provider, and the theory slots reach the brain."""

import secrets
import typing
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from mtc.consciousness.enhanced_global_workspace import ConsciousnessState
from mtc.core.config import Settings
from mtc.core.embeddings import HashEmbedder
from mtc.core.logger import reset_logging
from mtc.pipeline.context_types import ABSENT_REASONS
from mtc.pipeline.message_persistence import INTEGRATION_PROXY_LABEL
from mtc.pipeline.runtime import Runtime, build_runtime

ROOT = Path(__file__).resolve().parents[2]
TOKEN_ENV = "MTC_TEST_TURN_FIELDS_TOKEN"
_HINTS = typing.get_type_hints(ConsciousnessState)
# The Optional state fields: those that default to None, and the measurements
# annotated Optional[float] that keep a numeric default or none (a measured
# number or None, never a stand-in).
OPTIONAL_FIELDS = tuple(
    item.name
    for item in fields(ConsciousnessState)
    if item.default is None or _HINTS[item.name] == Optional[float]
)
SLOT_SECTIONS = {
    "neural_state": "neural_state",
    "attention_guidance": "attention_guidance",
    "metacognitive_insights": "metacognitive_insights",
    "active_inference_context": "active_inference",
    "global_workspace_context": "global_workspace",
}


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


def _no_failures(runtime: Runtime) -> None:
    assert runtime.orchestrator.status()["failures"] == {}
    assert runtime.pipeline.status()["failures"] == {}
    assert runtime.pipeline.post_turn.status()["failures"] == {}
    assert runtime.pipeline.persistence.status()["failures"] == {}


@pytest.mark.asyncio
async def test_mock_turn_populates_consciousness_fields(tmp_path, token):
    runtime = await _build(tmp_path)
    await runtime.start()
    for _ in range(4):
        await runtime.ctm.tick()
    result = await runtime.pipeline.send_message(
        "What do you notice about this conversation so far?", speaker="Alice"
    )
    assert result.success is True
    assert result.cycle_ok is True
    assert result.response and result.response.startswith("[mock]")
    assert result.provider_id == "mock"
    _no_failures(runtime)
    # the state the cycle produced (the orchestrator keeps its last state for
    # the idle Phi sampler) against what the turn named as absent
    state = runtime.orchestrator._last_state
    assert isinstance(state, ConsciousnessState)
    absent = runtime.pipeline.status()["last_turn"]["absent_signals"]
    unexplained = [
        name
        for name in OPTIONAL_FIELDS
        if getattr(state, name, None) is None and name not in absent
    ]
    assert unexplained == []
    assert set(absent) <= set(OPTIONAL_FIELDS)
    assert all(reason in ABSENT_REASONS for reason in absent.values())
    assert state.attention_report is not None
    assert state.metacognitive_report is not None
    assert state.active_inference_report is not None
    assert state.pad_arousal is not None  # the augmenter records the PAD coordinate
    steps = runtime.pipeline.status()["last_turn"]["post_turn"]
    assert not any(value.startswith("failed") for value in steps.values())
    assert steps["episodic_memory"] == "ok"
    assert steps["neural_experience"] == "ok"
    await runtime.stop()


@pytest.mark.asyncio
async def test_provenance_snapshot_stored_with_provider_mock(tmp_path, token):
    runtime = await _build(tmp_path)
    await runtime.start()
    result = await runtime.pipeline.send_message("Tell me about the roses.", "Alice")
    assert result.success is True
    turns = await runtime.turn_store.get_conversation_turns(result.conversation_id)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    snapshot = turns[1]["metadata"]["consciousness_snapshot"]
    assert snapshot["provider_id"] == "mock"
    assert snapshot["cycle_ok"] is True
    assert snapshot["computed_state"]["integration_proxy"]["label"] == (
        INTEGRATION_PROXY_LABEL
    )
    assert "narrative" in snapshot and "directives" in snapshot
    assert snapshot["absent_signals"] == (
        runtime.pipeline.status()["last_turn"]["absent_signals"]
    )
    assert turns[1]["metadata"]["to_speaker"] == "[Alice]"
    assert turns[0]["metadata"]["to_speaker"] == "Testagent"
    _no_failures(runtime)
    await runtime.stop()


@pytest.mark.asyncio
async def test_five_theory_slots_reach_the_brain(tmp_path, token):
    runtime = await _build(tmp_path, NARRATE_ALL=True)
    await runtime.start()
    result = await runtime.pipeline.send_message("How does attention feel?", "Alice")
    assert result.success is True
    # the mock provider records every request; the first message is the
    # system prompt the assembler rendered
    request = runtime.brain.registry.active().requests[-1]
    system_prompt = request.messages[0]["content"]
    assert request.messages[0]["role"] == "system"
    assert "candidates competed" in system_prompt
    rendered = {name for name, _ in runtime.brain.last_sections}
    filled = runtime.pipeline.persistence.turn_store
    turns = await filled.get_conversation_turns(result.conversation_id)
    sections = turns[1]["metadata"]["consciousness_snapshot"]["theory_sections"]
    assert "global_workspace_context" in sections
    for slot, section in SLOT_SECTIONS.items():
        if slot in sections:
            assert section in rendered, (slot, sorted(rendered))
    assert {"attention_guidance", "metacognitive_insights", "global_workspace"} <= (
        rendered
    )
    assert "phi" not in system_prompt.lower().replace("philosoph", "")
    _no_failures(runtime)
    await runtime.stop()
