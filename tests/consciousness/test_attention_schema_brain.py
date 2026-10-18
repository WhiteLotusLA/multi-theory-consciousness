"""The attention schema's other-agent inference through LLMBrain.generate()."""

from pathlib import Path
from typing import Any, Optional, Tuple

import pytest

from mtc.brain.brain import LLMBrain
from mtc.brain.prompt import PromptAssembler
from mtc.brain.providers.mock_provider import MockEvent, MockProvider
from mtc.brain.providers.registry import ProviderRegistry
from mtc.consciousness.attention_schema import AttentionSchemaModule, AttentionState
from mtc.core import persona as persona_module
from mtc.core.config import Settings


@pytest.fixture(autouse=True)
def _fresh_warning_memory(monkeypatch):
    monkeypatch.setattr(persona_module, "_WARNED_MISSING", set())


def _brain(tmp_path: Path, provider: Optional[Any] = None) -> Tuple[LLMBrain, Any]:
    settings = Settings(
        PROJECT_ROOT=str(tmp_path),
        SYSTEM_NAME="Testagent",
        CREATOR_NAME="Operator",
        LLM_PROVIDER="mock",
        DATA_DIRECTORY=str(tmp_path / "data"),
        IDENTITY_STATEMENT_PATH=str(tmp_path / "absent.md"),
    )
    provider = provider if provider is not None else MockProvider()
    registry = ProviderRegistry(state_path=tmp_path / "data" / "provider_state.json")
    registry.register(provider)
    registry.set_active(provider.model_info().id)
    assembler = PromptAssembler(settings=settings, identity_statement="")
    return LLMBrain(registry=registry, assembler=assembler, settings=settings), provider


@pytest.mark.asyncio
async def test_model_other_attention_uses_brain(tmp_path):
    brain, mock = _brain(tmp_path)
    await brain.initialize()
    schema = AttentionSchemaModule()
    context = "Alice is looking at the harbour chart"
    other = await schema.model_other_attention("Alice", context, llm_brain=brain)
    assert other.agent_name == "Alice"
    assert other.inferred_state == AttentionState.FOCUSED
    assert other.reasoning.startswith("[mock] reply to: ")
    assert "<think>" not in other.reasoning
    assert other.confidence == pytest.approx(0.9)  # the heuristic 0.7 plus 0.2
    assert len(mock.requests) == 1
    assert mock.requests[0].max_tokens == 100
    assert context in mock.requests[0].messages[-1]["content"]


@pytest.mark.asyncio
async def test_no_answer_keeps_heuristic_reasoning(tmp_path):
    brain, mock = _brain(tmp_path, MockProvider(replies=[MockEvent.EMPTY]))
    await brain.initialize()
    schema = AttentionSchemaModule()
    context = "Alice is looking at the harbour chart"
    heuristic = await schema.model_other_attention("Alice", context)
    with_brain = await schema.model_other_attention("Alice", context, llm_brain=brain)
    assert with_brain.reasoning == heuristic.reasoning
    assert "Alice" in with_brain.reasoning
    assert with_brain.confidence == pytest.approx(heuristic.confidence)
    assert len(mock.requests) == 1
    assert brain.status()["no_answers"]["empty"] == 1


@pytest.mark.asyncio
async def test_no_answer_is_a_plain_miss_under_debug(tmp_path):
    brain, mock = _brain(tmp_path, MockProvider(replies=[MockEvent.EMPTY]))
    await brain.initialize()
    schema = AttentionSchemaModule(settings=Settings(DEBUG=True))
    context = "Guest is reading the tide table"
    heuristic = await schema.model_other_attention("Guest", context)
    with_brain = await schema.model_other_attention("Guest", context, llm_brain=brain)
    assert with_brain.reasoning == heuristic.reasoning
    assert with_brain.confidence == pytest.approx(heuristic.confidence)
    assert schema.get_statistics()["failures"] == {}
    assert len(mock.requests) == 1
    assert brain.status()["no_answers"]["empty"] == 1
