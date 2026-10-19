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
async def test_budget_pass_reads_the_brains_own_settings(tmp_path, monkeypatch):
    """CHARS_PER_TOKEN, HISTORY_TOKEN_BUDGET and CONTEXT_SIZE of the budget pass
    come from the brain's Settings; the process Settings are never read."""
    from mtc.brain import budget as budget_module

    def forbidden():
        raise AssertionError("the process Settings were read")

    brain, _mock = _brain(tmp_path, CHARS_PER_TOKEN=7.0, HISTORY_TOKEN_BUDGET=30)
    monkeypatch.setattr(budget_module, "get_settings", forbidden)
    history = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": str(i) * 35,
            "timestamp": T0,
        }
        for i in range(6)
    ]
    await brain.think_with_context("the current turn", conversation_history=history)
    # 35 characters at 7.0 per token are 6 tokens: five entries fill the 30
    assert brain.status()["last_budget"] == {
        "kept": 5,
        "dropped": 1,
        "estimated_tokens": 30,
        "budget_tokens": 30,
    }


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


class CountingMock(MockProvider):
    """A mock provider that counts its health checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.health_checks = 0

    async def health_check(self) -> HealthStatus:
        self.health_checks += 1
        return await super().health_check()


class FakeMonotonic:
    def __init__(self, start: float = 1000.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now


def _recheck_brain(tmp_path: Path, provider: Any, **kwargs) -> LLMBrain:
    settings = _settings(tmp_path)
    registry = ProviderRegistry(state_path=tmp_path / "data" / "provider_state.json")
    registry.register(provider)
    registry.set_active(provider.model_info().id)
    return LLMBrain(
        registry=registry,
        assembler=PromptAssembler(settings=settings, identity_statement=""),
        settings=settings,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_degraded_brain_recovers_when_the_provider_does(tmp_path):
    provider = CountingMock(healthy=False)
    brain = _recheck_brain(tmp_path, provider, health_recheck_s=0.0)
    assert (await brain.initialize()).ok is False
    with pytest.raises(BrainUnavailableError, match="set unhealthy"):
        await brain.think("hello")
    assert provider.requests == []
    provider.healthy = True
    reply = await brain.think("hello")
    assert reply.text
    assert len(provider.requests) == 1
    assert brain.available is True
    # the conversational path recovers the same way
    provider.healthy = False
    await brain.refresh_health()
    provider.healthy = True
    assert (await brain.think_with_context("hi")).text
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_degraded_brain_probes_at_most_once_per_interval(tmp_path):
    clock = FakeMonotonic()
    provider = CountingMock(healthy=False)
    brain = _recheck_brain(tmp_path, provider, health_recheck_s=5.0, monotonic=clock)
    assert brain.health_recheck_s == 5.0
    await brain.initialize()
    assert provider.health_checks == 1
    for _ in range(5):
        with pytest.raises(BrainUnavailableError):
            await brain.think_with_context("hello")
    assert provider.health_checks == 1
    clock.now += 5.0
    for _ in range(3):
        with pytest.raises(BrainUnavailableError):
            await brain.think("hello")
    assert provider.health_checks == 2
    assert provider.requests == []
    # a provider that was healthy at start is never probed on the call path
    healthy = CountingMock()
    steady = _recheck_brain(tmp_path, healthy, health_recheck_s=0.0, monotonic=clock)
    await steady.initialize()
    for _ in range(3):
        clock.now += 60.0
        assert (await steady.think("hello")).text
    assert healthy.health_checks == 1
    assert len(healthy.requests) == 3
    # the default interval is the code constant
    assert _recheck_brain(tmp_path, MockProvider()).health_recheck_s == 5.0
    with pytest.raises(ValueError, match="health_recheck_s"):
        _recheck_brain(tmp_path, MockProvider(), health_recheck_s=-1.0)


class GatedHealthMock(CountingMock):
    """Healthy for none of its checks; every check after the first waits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, healthy=False, **kwargs)
        self.release = asyncio.Event()

    async def health_check(self) -> HealthStatus:
        if self.health_checks >= 1:
            self.health_checks += 1
            await self.release.wait()
            return await MockProvider.health_check(self)
        return await super().health_check()


@pytest.mark.asyncio
async def test_concurrent_calls_on_a_degraded_brain_probe_once(tmp_path):
    clock = FakeMonotonic()
    provider = GatedHealthMock()
    brain = _recheck_brain(tmp_path, provider, health_recheck_s=5.0, monotonic=clock)
    await brain.initialize()
    assert provider.health_checks == 1
    clock.now += 5.0
    calls = [asyncio.create_task(brain.think(f"call {i}")) for i in range(3)]
    for _ in range(5):
        await asyncio.sleep(0)
    # one re-check is in flight; the other calls saw it started and did not
    # start their own
    assert provider.health_checks == 2
    provider.release.set()
    outcomes = await asyncio.gather(*calls, return_exceptions=True)
    assert provider.health_checks == 2
    assert all(isinstance(o, BrainUnavailableError) for o in outcomes)
    assert provider.requests == []


class RaisingHealthMock(CountingMock):
    """A mock whose health check raises a plain error while ``fault`` is set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault: Optional[Exception] = None

    async def health_check(self) -> HealthStatus:
        if self.fault is not None:
            self.health_checks += 1
            raise self.fault
        return await super().health_check()


@pytest.mark.asyncio
async def test_recheck_that_raises_leaves_the_brain_degraded_and_unavailable(
    tmp_path,
):
    provider = RaisingHealthMock(healthy=False)
    brain = _recheck_brain(tmp_path, provider, health_recheck_s=0.0)
    assert (await brain.initialize()).ok is False
    assert provider.health_checks == 1
    provider.fault = OSError("connection closed under the health probe")
    with pytest.raises(BrainUnavailableError, match="OSError"):
        await brain.think("hello")
    assert provider.health_checks == 2
    assert brain.available is False
    health = brain.status()["health"]
    assert health["ok"] is False
    assert health["detail"].startswith("OSError: connection closed")
    # the conversational path refuses the same way
    provider.fault = RuntimeError("health probe bug")
    with pytest.raises(BrainUnavailableError, match="RuntimeError"):
        await brain.think_with_context("hello")
    assert provider.health_checks == 3
    assert brain.available is False
    assert provider.requests == []
    # an explicit refresh records the same unhealthy state instead of raising
    recorded = await brain.refresh_health()
    assert recorded.ok is False and recorded.detail.startswith("RuntimeError:")
    # once the probe stops raising and the provider is healthy, calls are served
    provider.fault = None
    provider.healthy = True
    assert (await brain.think("hello")).text
    assert brain.available is True
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_last_call_outcome_is_recorded_and_never_gates(tmp_path):
    from mtc.brain.providers.base import ProviderError, ProviderTimeoutError

    provider = MockProvider(
        replies=[
            "first answer",
            ProviderError("server error", status_code=500),
            "second answer",
            ProviderTimeoutError("stalled"),
            MockEvent.EMPTY,
        ]
    )
    brain = _recheck_brain(tmp_path, provider)
    await brain.initialize()
    status = brain.status()
    assert status["last_call_ok"] is None
    assert status["last_call_at"] is None
    assert (await brain.think("one")).text
    status = brain.status()
    assert status["last_call_ok"] is True
    assert isinstance(status["last_call_at"], str)
    with pytest.raises(BrainUnavailableError):
        await brain.think("two")
    assert brain.status()["last_call_ok"] is False
    # the record never refuses a call: the next call is sent and succeeds
    assert brain.available is True
    assert (await brain.think("three")).text
    assert len(provider.requests) == 3
    assert brain.status()["last_call_ok"] is True
    with pytest.raises(ProviderTimeoutError):
        await brain.think("four")
    assert brain.status()["last_call_ok"] is False
    # a completion that came back empty is a returned completion
    with pytest.raises(NoAnswerError):
        await brain.think("five")
    assert brain.status()["last_call_ok"] is True
    assert len(provider.requests) == 5
    # the recorded health keeps its own fields
    assert brain.status()["health"]["ok"] is True


@pytest.mark.asyncio
async def test_given_last_turn_at_wins_over_the_history_tail(tmp_path):
    brain, provider = _brain(tmp_path)
    await brain.initialize()
    evening = datetime(2000, 1, 1, 21, 0, tzinfo=timezone.utc)
    history = [
        {"role": "user", "content": "good evening", "timestamp": evening},
        {
            "role": "user",
            "speaker": "clock",
            "content": "[Clock note, not a person: a night passed at this point.]",
            "timestamp": None,
            "is_clock_note": True,
        },
    ]
    await brain.think_with_context(
        "good morning", conversation_history=history, last_turn_at=evening
    )
    system = provider.requests[-1].messages[0]["content"]
    assert "Time since the last exchange:" in system
    # without it the tail (a clock note with no time) gives no gap line
    await brain.think_with_context("good morning", conversation_history=history)
    assert (
        "Time since the last exchange:"
        not in provider.requests[-1].messages[0]["content"]
    )


@pytest.mark.asyncio
async def test_image_refusal_names_no_removed_key(tmp_path):
    brain, mock = _brain(tmp_path)
    with pytest.raises(CapabilityUnavailableError) as refused:
        await brain.think_with_context("look", image=b"\x89PNG")
    assert str(refused.value) == "image input is not part of this release"
    assert mock.requests == []
