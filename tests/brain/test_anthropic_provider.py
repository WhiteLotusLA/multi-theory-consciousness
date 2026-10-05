"""Anthropic provider against a fake SDK client; no network, no SDK needed."""

import inspect
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mtc.brain.protocol import CapabilityUnavailableError, ConfigurationError
from mtc.brain.providers.anthropic_provider import (
    STOP_REASON_MAP,
    AnthropicProvider,
)
from mtc.brain.providers.base import (
    CompletionParams,
    ModelProvider,
    ProviderError,
    ProviderTimeoutError,
)
from mtc.core.config import Settings

MODEL = "test-model-id"


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _events(thinking="weighing it", text="The answer.", stop="end_turn"):
    events = [_ns(type="message_start", message=_ns(usage=_ns(input_tokens=11)))]
    index = 0
    if thinking is not None:
        events += [
            _ns(
                type="content_block_start", index=0, content_block=_ns(type="thinking")
            ),
            _ns(
                type="content_block_delta",
                index=0,
                delta=_ns(type="thinking_delta", thinking=thinking),
            ),
            _ns(
                type="content_block_delta",
                index=0,
                delta=_ns(type="signature_delta", signature="sig"),
            ),
            _ns(type="content_block_stop", index=0),
        ]
        index = 1
    events += [
        _ns(type="content_block_start", index=index, content_block=_ns(type="text")),
        _ns(
            type="content_block_delta",
            index=index,
            delta=_ns(type="text_delta", text=text[:4]),
        ),
        _ns(
            type="content_block_delta",
            index=index,
            delta=_ns(type="text_delta", text=text[4:]),
        ),
        _ns(type="content_block_stop", index=index),
        _ns(
            type="message_delta",
            delta=_ns(stop_reason=stop),
            usage=_ns(output_tokens=7),
        ),
        _ns(type="message_stop"),
    ]
    return events


def _final(thinking="weighing it", text="The answer.", stop="end_turn", usage=None):
    content = []
    if thinking is not None:
        content.append(_ns(type="thinking", thinking=thinking, signature="sig"))
    content.append(_ns(type="text", text=text))
    usage = usage or _ns(
        input_tokens=11,
        output_tokens=7,
        cache_creation_input_tokens=3,
        cache_read_input_tokens=5,
    )
    return _ns(content=content, stop_reason=stop, usage=usage)


class FakeStream:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def generate():
            for event in self._events:
                yield event

        return generate()

    async def get_final_message(self):
        return self._final


class FakeMessages:
    def __init__(self, events=None, final=None, error=None):
        self.events = events if events is not None else _events()
        self.final = final if final is not None else _final()
        self.error = error
        self.stream_calls = []
        self.create_calls = []

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeStream(self.events, self.final)

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.final


class FakeClient:
    def __init__(self, **kwargs):
        self.messages = FakeMessages(**kwargs)
        self.closed = False

    async def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.now = datetime(2000, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def _key():
    return secrets.token_hex(16)


def _provider(client=None, **kwargs):
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("api_key", _key())
    kwargs.setdefault("max_tokens", 8192)
    return AnthropicProvider(client=client or FakeClient(), **kwargs)


def _params(**overrides):
    values = dict(
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "system", "content": "Be kind."},
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        ],
        max_tokens=512,
        temperature=0.7,
    )
    values.update(overrides)
    return CompletionParams(**values)


async def _collect(provider, params):
    return [chunk async for chunk in provider.stream(params)]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_model_id_required_no_default(clean_env):
    parameter = inspect.signature(AnthropicProvider).parameters["model"]
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(ConfigurationError, match="ANTHROPIC_MODEL"):
        AnthropicProvider(model="", api_key=_key(), max_tokens=1024)
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider(model=MODEL, api_key="", max_tokens=1024)
    with pytest.raises(ConfigurationError, match="ANTHROPIC_MODEL"):
        AnthropicProvider.from_settings(Settings(ANTHROPIC_API_KEY=_key()))
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider.from_settings(Settings(ANTHROPIC_MODEL=MODEL))
    key = _key()
    provider = AnthropicProvider.from_settings(
        Settings(ANTHROPIC_MODEL=MODEL, ANTHROPIC_API_KEY=key)
    )
    assert provider.model == MODEL and provider.max_tokens == 8192
    assert key not in repr(provider)
    info = provider.model_info()
    assert (info.id, info.name, info.provider_type, info.cost) == (
        "anthropic",
        MODEL,
        "anthropic",
        "paid",
    )
    assert isinstance(provider, ModelProvider)


@pytest.mark.asyncio
async def test_system_message_translated():
    client = FakeClient()
    await _provider(client).complete(_params())
    call = client.messages.stream_calls[0]
    assert call["system"] == "Be brief.\n\nBe kind."
    assert call["model"] == MODEL
    assert [m["role"] for m in call["messages"]] == ["user", "assistant", "user"]
    assert call["messages"][2]["content"] == [{"type": "text", "text": "How are you?"}]
    assert all(m["role"] != "system" for m in call["messages"])
    with pytest.raises(ValueError, match="role"):
        await _provider().complete(_params(messages=[{"role": "tool", "content": "x"}]))


@pytest.mark.asyncio
async def test_thinking_budget_below_max_tokens():
    with pytest.raises(ConfigurationError, match="ANTHROPIC_THINKING_BUDGET"):
        _provider(thinking_budget=8192, max_tokens=8192)
    with pytest.raises(ConfigurationError, match="1024"):
        _provider(thinking_budget=100)
    client = FakeClient()
    provider = _provider(client, thinking_budget=2048, max_tokens=8192)
    # The reference set the budget equal to max tokens at one size; every
    # requested size here leaves room for the answer.
    for requested in (None, 256, 2048, 4000, 8192):
        await provider.complete(_params(thinking_mode=True, max_tokens=requested))
        call = client.messages.stream_calls[-1]
        assert call["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert call["thinking"]["budget_tokens"] < call["max_tokens"]
        assert "temperature" not in call and "top_p" not in call
    adaptive = FakeClient()
    await _provider(adaptive).complete(_params(thinking_mode=True))
    assert adaptive.messages.stream_calls[0]["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    quiet = FakeClient()
    await _provider(quiet).complete(_params(thinking_mode=False))
    assert "thinking" not in quiet.messages.stream_calls[0]


@pytest.mark.asyncio
async def test_sampling_parameters_only_when_enabled():
    default = FakeClient()
    await _provider(default).complete(_params(top_p=0.9, top_k=5))
    call = default.messages.stream_calls[0]
    for field in ("temperature", "top_p", "top_k"):
        assert field not in call, field
    enabled = FakeClient()
    await _provider(enabled, sampling=True).complete(_params(top_p=0.9, top_k=5))
    call = enabled.messages.stream_calls[0]
    # One of temperature or top_p, never both.
    assert call["temperature"] == 0.7 and "top_p" not in call
    assert call["top_k"] == 5


@pytest.mark.asyncio
async def test_complete_reads_thinking_and_text_blocks():
    result = await _provider().complete(_params(thinking_mode=True))
    assert result.text == "The answer."
    assert result.reasoning == "weighing it"
    omitted = FakeClient(final=_final(thinking=""))
    result = await _provider(omitted).complete(_params(thinking_mode=True))
    assert result.reasoning is None and result.text == "The answer."


@pytest.mark.asyncio
async def test_thinking_maps_to_think_tags_in_stream():
    provider = _provider()
    chunks = await _collect(provider, _params(thinking_mode=True))
    assert chunks == ["<think>", "weighing it", "</think>", "The ", "answer."]
    assert provider.last_stream_usage["finish_reason"] == "stop"
    plain = _provider(FakeClient(events=_events(thinking=None)))
    assert await _collect(plain, _params()) == ["The ", "answer."]
    cut = [e for e in _events() if e.type != "content_block_stop"][:4]
    unclosed = _provider(FakeClient(events=cut))
    chunks = await _collect(unclosed, _params(thinking_mode=True))
    assert chunks[-1] == "</think>" and chunks.count("<think>") == 1


@pytest.mark.asyncio
async def test_stream_honours_stop_and_top_p():
    client = FakeClient()
    provider = _provider(client, sampling=True)
    params = _params(temperature=None, top_p=0.9, stop=["END"])
    await _collect(provider, params)
    await provider.complete(params)
    streamed, completed = client.messages.stream_calls
    assert streamed["stop_sequences"] == ["END"] == completed["stop_sequences"]
    assert streamed["top_p"] == 0.9 == completed["top_p"]
    assert streamed == completed


@pytest.mark.asyncio
async def test_finish_reason_from_stop_reason():
    expected = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
        "tool_use": "tool_calls",
        "pause_turn": "pause_turn",
        "refusal": "refusal",
    }
    assert STOP_REASON_MAP == expected
    for stop, finish in expected.items():
        client = FakeClient(final=_final(stop=stop))
        assert (await _provider(client).complete(_params())).finish_reason == finish
    novel = FakeClient(final=_final(stop="something_new"))
    assert (await _provider(novel).complete(_params())).finish_reason == "something_new"


@pytest.mark.asyncio
async def test_health_check_makes_no_completion():
    clock = Clock()
    client = FakeClient()
    provider = _provider(client, clock=clock, health_ttl_s=600.0)
    first = await provider.health_check()
    assert first.ok is True and first.verified is False
    assert "unverified" in first.detail
    await provider.complete(_params())
    verified = await provider.health_check()
    assert verified.ok is True and verified.verified is True
    clock.now += timedelta(seconds=601)
    stale = await provider.health_check()
    assert stale.verified is False
    assert client.messages.create_calls == []
    assert len(client.messages.stream_calls) == 1  # only the real call
    failing = FakeClient(
        error=type("APIStatusError", (Exception,), {"status_code": 529})()
    )
    broken = _provider(failing, clock=clock)
    with pytest.raises(ProviderError) as info:
        await broken.complete(_params())
    assert info.value.status_code == 529
    status = await broken.health_check()
    assert status.ok is False and status.verified is True


@pytest.mark.asyncio
async def test_timeout_raises_provider_timeout():
    slow = FakeClient(error=type("APITimeoutError", (Exception,), {})())
    with pytest.raises(ProviderTimeoutError):
        await _provider(slow).complete(_params())


@pytest.mark.asyncio
async def test_image_parts_refused():
    client = FakeClient()
    image = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
        ],
    }
    provider = _provider(client)
    with pytest.raises(CapabilityUnavailableError):
        await provider.complete(_params(messages=[image]))
    with pytest.raises(CapabilityUnavailableError):
        await _collect(provider, _params(messages=[image]))
    assert client.messages.stream_calls == []
    assert provider.model_info().supports_vision is False


@pytest.mark.asyncio
async def test_usage_totals():
    result = await _provider().complete(_params())
    # Cached prompt tokens count as input: 11 uncached + 3 written + 5 read.
    assert result.input_tokens == 19
    assert result.output_tokens == 7
    assert result.total_tokens == 26
    bare = FakeClient(final=_final(usage=_ns(input_tokens=4, output_tokens=2)))
    result = await _provider(bare).complete(_params())
    assert (result.input_tokens, result.total_tokens) == (4, 6)
    missing = FakeClient(final=_ns(content=[], stop_reason="end_turn", usage=None))
    result = await _provider(missing).complete(_params())
    assert result.input_tokens is None and result.total_tokens is None
    assert result.text == ""


@pytest.mark.asyncio
async def test_injected_client_is_not_closed():
    client = FakeClient()
    provider = _provider(client)
    await provider.initialize()
    await provider.shutdown()
    assert client.closed is False


@pytest.mark.asyncio
async def test_missing_sdk_names_the_anthropic_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    provider = AnthropicProvider(model=MODEL, api_key=_key(), max_tokens=1024)
    with pytest.raises(ImportError, match="anthropic"):
        await provider.initialize()
    with pytest.raises(ImportError, match="'anthropic' extra"):
        await provider.complete(_params())
    status = await provider.health_check()
    assert status.ok is False and "anthropic" in status.detail
