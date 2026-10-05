"""generate(): the spoken text only, and an honest error when there is none."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from mtc.brain.brain import LLMBrain
from mtc.brain.prompt import PromptAssembler
from mtc.brain.protocol import NoAnswerError
from mtc.brain.providers.base import (
    CompletionParams,
    CompletionResult,
    HealthStatus,
    ModelInfo,
)
from mtc.brain.providers.mock_provider import MockEvent, MockProvider
from mtc.brain.providers.registry import ProviderRegistry
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


class ReasoningProvider:
    def __init__(self, result: CompletionResult):
        self.result = result
        self.requests: List[CompletionParams] = []

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id="fake",
            name="scripted",
            provider_type="local",
            context_window=4096,
            cost="local",
            supports_thinking=True,
        )

    @property
    def info(self) -> ModelInfo:
        return self.model_info()

    def supports_thinking(self) -> bool:
        return True

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def health_check(self) -> HealthStatus:
        return HealthStatus(True, "scripted", datetime.now(timezone.utc), True)

    async def complete(self, params: CompletionParams) -> CompletionResult:
        self.requests.append(params)
        return self.result

    async def stream(self, params: CompletionParams):
        self.requests.append(params)
        yield self.result.text


@pytest.mark.asyncio
async def test_generate_returns_spoken_text_only(tmp_path):
    provider = ReasoningProvider(
        CompletionResult(text="The answer is the tide.", reasoning="private reasoning")
    )
    brain, _p = _brain(tmp_path, provider)
    spoken = await brain.generate("What moves the water?", max_tokens=100)
    assert spoken == "The answer is the tide."
    assert "<think>" not in spoken and "private reasoning" not in spoken
    request = provider.requests[-1]
    assert request.max_tokens == 100
    assert request.thinking_mode is False
    assert [m["role"] for m in request.messages] == ["user"]

    mocked, _m = _brain(tmp_path / "m")
    text = await mocked.generate("What moves the water?")
    assert text.startswith("[mock] reply to: What moves the water?")
    assert "<think>" not in text


@pytest.mark.asyncio
async def test_generate_no_answer_raises(tmp_path):
    empty, _e = _brain(tmp_path / "e", MockProvider(replies=[MockEvent.EMPTY]))
    with pytest.raises(NoAnswerError) as caught:
        await empty.generate("anything")
    assert caught.value.reason == "empty"
    reasoning_only, _r = _brain(
        tmp_path / "r", MockProvider(replies=[MockEvent.REASONING_ONLY])
    )
    with pytest.raises(NoAnswerError) as caught:
        await reasoning_only.generate("anything")
    assert caught.value.reason == "reasoning_only"
    assert reasoning_only.status()["no_answers"]["reasoning_only"] == 1
