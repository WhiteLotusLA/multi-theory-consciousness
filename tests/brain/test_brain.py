"""LLMBrain: honest replies, one budget pass, no silent fallback, singleton."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from mtc.brain import brain as brain_module
from mtc.brain.brain import LLMBrain, get_brain, reset_brain_for_tests
from mtc.brain.budget import estimate_tokens
from mtc.brain.prompt import PromptAssembler, SpeakerView
from mtc.brain.protocol import (
    BrainProtocol,
    BrainUnavailableError,
    CapabilityUnavailableError,
    ConfigurationError,
    ConversationalBrain,
    NoAnswerError,
)
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

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_warning_memory(monkeypatch):
    monkeypatch.setattr(persona_module, "_WARNED_MISSING", set())


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        PROJECT_ROOT=str(tmp_path),
        SYSTEM_NAME="Testagent",
        CREATOR_NAME="Operator",
        LLM_PROVIDER="mock",
        DATA_DIRECTORY=str(tmp_path / "data"),
        IDENTITY_STATEMENT_PATH=str(tmp_path / "absent.md"),
    )
    values.update(overrides)
    return Settings(**values)


def _brain(
    tmp_path: Path, provider: Optional[Any] = None, **overrides
) -> Tuple[LLMBrain, Any]:
    settings = _settings(tmp_path, **overrides)
    provider = provider if provider is not None else MockProvider()
    registry = ProviderRegistry(state_path=tmp_path / "data" / "provider_state.json")
    registry.register(provider)
    registry.set_active(provider.model_info().id)
    assembler = PromptAssembler(settings=settings, identity_statement="")
    return LLMBrain(registry=registry, assembler=assembler, settings=settings), provider


class ScriptedProvider:
    """A provider that returns the given results in order and records requests."""

    def __init__(self, results: List[CompletionResult], provider_id: str = "fake"):
        self.results = list(results)
        self.provider_id = provider_id
        self.requests: List[CompletionParams] = []

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            id=self.provider_id,
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
        return self.results.pop(0)

    async def stream(self, params: CompletionParams):
        result = await self.complete(params)
        if result.reasoning:
            yield "<think>"
            yield result.reasoning
            yield "</think>"
        yield result.text


def test_satisfies_brain_protocol_and_conversational_brain(tmp_path):
    brain, _mock = _brain(tmp_path)
    assert isinstance(brain, BrainProtocol)
    assert isinstance(brain, ConversationalBrain)


@pytest.mark.asyncio
async def test_think_returns_canonical_reply(tmp_path):
    brain, _mock = _brain(tmp_path)
    await brain.initialize()
    with_reasoning = await brain.think("hello there", thinking_mode=True)
    assert with_reasoning.text.startswith("<think>mock reasoning ")
    assert "</think>\n\n[mock] reply to: hello there (" in with_reasoning.text
    assert with_reasoning.finish_reason == "stop"
    plain = await brain.think("hello there", thinking_mode=False)
    assert plain.text.startswith("[mock] reply to: hello there (")
    assert "<think>" not in plain.text
    assert brain.status()["calls"] == 2


@pytest.mark.asyncio
async def test_reasoning_only_raises_no_answer(tmp_path):
    brain, _mock = _brain(tmp_path, MockProvider(replies=[MockEvent.REASONING_ONLY]))
    with pytest.raises(NoAnswerError) as caught:
        await brain.think("hello", thinking_mode=True)
    assert caught.value.reason == "reasoning_only"
    assert brain.status()["no_answers"]["reasoning_only"] == 1


@pytest.mark.asyncio
async def test_empty_raises_no_answer(tmp_path):
    brain, _mock = _brain(tmp_path, MockProvider(replies=[MockEvent.EMPTY]))
    with pytest.raises(NoAnswerError) as caught:
        await brain.think_with_context("hello")
    assert caught.value.reason == "empty"
    assert brain.status()["no_answers"]["empty"] == 1
    assert brain.status()["no_answers"]["loop"] == 0


@pytest.mark.asyncio
async def test_one_budget_pass(tmp_path, monkeypatch):
    brain, mock = _brain(tmp_path, HISTORY_TOKEN_BUDGET=30)
    calls: List[int] = []
    original = brain_module.fit_history

    def counting(history, budget_tokens, **kwargs):
        calls.append(budget_tokens)
        return original(history, budget_tokens, **kwargs)

    monkeypatch.setattr(brain_module, "fit_history", counting)
    history = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": str(i) * 35,
            "timestamp": T0,
        }
        for i in range(6)
    ]
    assert estimate_tokens("0" * 35) == 11  # three entries would need 33 > 30
    reply = await brain.think_with_context(
        "the current turn", conversation_history=history
    )
    assert reply.text
    assert calls == [30]
    request = mock.requests[-1]
    assert request.messages[0]["role"] == "system"
    assert request.messages[1:] == [
        {"role": "user", "content": "4" * 35},
        {"role": "assistant", "content": "5" * 35},
        {"role": "user", "content": "the current turn"},
    ]
    status = brain.status()["last_budget"]
    assert status == {
        "kept": 2,
        "dropped": 4,
        "estimated_tokens": 22,
        "budget_tokens": 30,
    }
    # A small context window wins over the history budget.
    small, small_mock = _brain(tmp_path / "small", CONTEXT_SIZE=600, LLM_MAX_TOKENS=100)
    await small.think_with_context("x", conversation_history=history)
    system_tokens = estimate_tokens(small_mock.requests[-1].messages[0]["content"])
    assert small.status()["last_budget"]["budget_tokens"] == 600 - 100 - system_tokens


@pytest.mark.asyncio
async def test_temperature_from_generation_params(tmp_path):
    brain, mock = _brain(tmp_path)
    await brain.think_with_context(
        "hi", generation_params={"temperature": 0.21, "top_p": 0.5}
    )
    assert mock.requests[-1].temperature == pytest.approx(0.21)
    assert mock.requests[-1].top_p == pytest.approx(0.5)
    await brain.think_with_context("hi")
    assert mock.requests[-1].temperature == pytest.approx(brain._cfg().TEMPERATURE)
    assert mock.requests[-1].top_p == pytest.approx(brain._cfg().TOP_P)
    assert mock.requests[-1].max_tokens == brain._cfg().LLM_MAX_TOKENS
    await brain.think_with_context(
        "hi", max_tokens=77, generation_params={"max_tokens": 5}
    )
    assert mock.requests[-1].max_tokens == 77


@pytest.mark.asyncio
async def test_five_slots_reach_the_system_prompt(tmp_path):
    brain, mock = _brain(tmp_path)
    await brain.think_with_context(
        "hi",
        neural_state={"binding_strength": 0.31},
        attention_guidance="Attention rests on the harbour question.",
        metacognitive_insights="Confidence was moderate last turn.",
        active_inference_context={"surprise": 0.2},
        global_workspace_context="Workspace focus: ferries.",
        speaker=SpeakerView(
            name="Alice", role="friend", authenticated=True, label="verified"
        ),
        conversation_summary="Earlier: tide tables.",
    )
    system = mock.requests[-1].messages[0]["content"]
    for marker in (
        "binding_strength: 0.310",
        "harbour question",
        "Confidence was moderate",
        "surprise: 0.200",
        "Workspace focus: ferries.",
        "Alice",
        "# Earlier in this conversation",
    ):
        assert marker in system, marker
    # The summary lives in the system prompt, not in a second system message.
    assert [m["role"] for m in mock.requests[-1].messages] == ["system", "user"]
    assert "neural_state" in [name for name, _ in brain.status()["last_sections"]]


@pytest.mark.asyncio
async def test_image_argument_refused(tmp_path):
    brain, mock = _brain(tmp_path)
    with pytest.raises(CapabilityUnavailableError, match="not part of this release"):
        await brain.think_with_context("look", image=b"\x89PNG")
    assert mock.requests == []
    with pytest.raises(TypeError, match="SpeakerView"):
        await brain.think_with_context("hi", speaker="Operator")
    assert mock.requests == []


@pytest.mark.asyncio
async def test_no_silent_provider_fallback(tmp_path):
    settings = _settings(tmp_path)
    unhealthy = MockProvider(healthy=False, provider_id="mock")
    spare = MockProvider(provider_id="spare")
    registry = ProviderRegistry(state_path=tmp_path / "data" / "state.json")
    registry.register(unhealthy)
    registry.register(spare)
    registry.set_active("mock")
    brain = LLMBrain(
        registry=registry,
        assembler=PromptAssembler(settings=settings, identity_statement=""),
        settings=settings,
    )
    health = await brain.initialize()
    assert health.ok is False
    assert brain.available is False
    with pytest.raises(BrainUnavailableError):
        await brain.think("hello")
    with pytest.raises(BrainUnavailableError):
        await brain.think_with_context("hello")
    with pytest.raises(BrainUnavailableError):
        await brain.generate("hello")
    assert unhealthy.requests == [] and spare.requests == []
    assert registry.active_id == "mock"
    status = brain.status()
    assert status["available"] is False
    assert status["health"]["ok"] is False
    assert status["providers"] == ["mock", "spare"]
    # Recovery is reported, never assumed: a later check that passes re-enables.
    unhealthy.healthy = True
    assert (await brain.refresh_health()).ok is True
    assert (await brain.think("hello")).text


@pytest.mark.asyncio
async def test_get_brain_is_locked_singleton(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    built: List[int] = []
    original = LLMBrain.from_settings

    def counting(active=None):
        built.append(1)
        return original(active)

    monkeypatch.setattr(LLMBrain, "from_settings", staticmethod(counting))
    reset_brain_for_tests()
    try:
        first, second = await asyncio.gather(get_brain(settings), get_brain(settings))
        assert first is second
        assert built == [1]
        assert first.status()["initialized"] is True
        assert first.status()["active_provider"] == "mock"
        assert await get_brain(settings) is first
    finally:
        reset_brain_for_tests()
    assert (tmp_path / "data" / "provider_state.json").is_file()


def test_get_brain_contends_in_two_successive_loops(tmp_path, monkeypatch):
    """The singleton lock is bound to the loop that first contends on it; a
    fresh loop (the runtime, or the API under a test client) must not inherit
    a lock bound to a closed one, with or without a reset in between."""
    settings = _settings(tmp_path)
    built: List[int] = []
    original = LLMBrain.from_settings

    def counting(active=None):
        built.append(1)
        return original(active)

    monkeypatch.setattr(LLMBrain, "from_settings", staticmethod(counting))

    async def contend() -> LLMBrain:
        first, second = await asyncio.gather(get_brain(settings), get_brain(settings))
        assert first is second
        return first

    reset_brain_for_tests()
    try:
        one = asyncio.run(contend())
        reset_brain_for_tests()
        two = asyncio.run(contend())
        assert one is not two
        # A loop that merely finds the singleton gone (no reset) also contends
        # safely: the lock is replaced for the running loop.
        brain_module._brain = None
        three = asyncio.run(contend())
        assert three is not two
        assert built == [1, 1, 1]
    finally:
        reset_brain_for_tests()


def test_from_settings_registers_configured_providers(tmp_path):
    brain = LLMBrain.from_settings(_settings(tmp_path))
    assert brain.status()["providers"] == ["local", "mock"]
    assert brain.status()["active_provider"] == "mock"
    assert brain.status()["persona_source"] == "example"
    local_only = LLMBrain.from_settings(_settings(tmp_path, LLM_PROVIDER="local"))
    assert local_only.status()["providers"] == ["local"]
    with pytest.raises(ConfigurationError, match="ANTHROPIC"):
        LLMBrain.from_settings(_settings(tmp_path, LLM_PROVIDER="anthropic"))


@pytest.mark.asyncio
async def test_usage_passthrough(tmp_path):
    scripted = ScriptedProvider(
        [
            CompletionResult(
                text="spoken",
                reasoning=None,
                input_tokens=12,
                output_tokens=3,
                total_tokens=15,
                finish_reason="length",
            ),
            CompletionResult(text="again", reasoning=None),
        ]
    )
    brain, _p = _brain(tmp_path, scripted)
    reply = await brain.think("hi")
    assert (reply.input_tokens, reply.output_tokens, reply.finish_reason) == (
        12,
        3,
        "length",
    )
    none = await brain.think("hi")
    assert none.input_tokens is None and none.output_tokens is None
    mocked, _m = _brain(tmp_path / "m")
    usage = await mocked.think("hello")
    assert isinstance(usage.input_tokens, int) and usage.input_tokens > 0
    assert isinstance(usage.output_tokens, int) and usage.output_tokens > 0


@pytest.mark.asyncio
async def test_reply_carries_provider_id(tmp_path):
    brain, _mock = _brain(tmp_path)
    assert (await brain.think("hi")).provider_id == "mock"
    other, _p = _brain(tmp_path / "o", ScriptedProvider([CompletionResult(text="x")]))
    assert (await other.think("hi")).provider_id == "fake"
    assert brain.model_name == "mock provider"


def test_no_synthesis_method():
    assert not hasattr(LLMBrain, "synthesize_knowledge")
    assert not hasattr(LLMBrain, "synthesize")
