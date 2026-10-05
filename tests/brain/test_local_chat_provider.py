"""Local chat-completions provider over an injected mock transport; no network."""

import json
import os
import sys

import pytest

httpx = pytest.importorskip("httpx")

from mtc.brain.protocol import ConfigurationError  # noqa: E402
from mtc.brain.providers.base import (  # noqa: E402
    CompletionParams,
    ModelProvider,
    ProviderError,
    ProviderTimeoutError,
)
from mtc.brain.providers.local_chat import LocalChatCompletionsProvider  # noqa: E402
from mtc.core.config import Settings  # noqa: E402

BASE = "http://host.invalid:8080"


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class Recorder:
    """A mock transport handler that records requests and replays responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(request)
        # A fresh copy each time: a response object serves one request only.
        return httpx.Response(
            item.status_code, headers=item.headers, content=item.content
        )

    def body(self, index=-1):
        return json.loads(self.requests[index].content)


def _completion(content="Hello.", reasoning=None, usage=None, finish="stop", key=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message[key or "reasoning_content"] = reasoning
    data = {"choices": [{"message": message, "finish_reason": finish}]}
    if usage is not None:
        data["usage"] = usage
    return httpx.Response(200, json=data)


def _sse(*events):
    lines = []
    for event in events:
        if event == "[DONE]":
            lines.append("data: [DONE]")
        else:
            lines.append("data: " + json.dumps(event))
        lines.append("")
    return httpx.Response(
        200,
        content=("\n".join(lines) + "\n").encode(),
        headers={"content-type": "text/event-stream"},
    )


def _delta(content=None, reasoning=None, key="reasoning_content", finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta[key] = reasoning
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def _provider(handler, **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs.setdefault("base_url", BASE)
    return LocalChatCompletionsProvider(client=client, **kwargs)


def _params(**overrides):
    values = dict(
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Say hello."},
        ],
        temperature=0.5,
        max_tokens=64,
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        repetition_penalty=1.1,
        frequency_penalty=0.2,
    )
    values.update(overrides)
    return CompletionParams(**values)


async def _collect(provider, params):
    return [chunk async for chunk in provider.stream(params)]


def test_satisfies_model_provider_protocol():
    provider = _provider(Recorder(_completion()))
    assert isinstance(provider, ModelProvider)
    info = provider.model_info()
    assert info.id == "local" and info.provider_type == "local"
    assert info.cost == "local"
    assert provider.info == info
    assert provider.supports_thinking() is False
    assert _provider(Recorder(_completion()), family="qwen").supports_thinking()


def test_empty_base_url_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="LLM_BASE_URL"):
        LocalChatCompletionsProvider(base_url="")


@pytest.mark.asyncio
async def test_request_fields_follow_family():
    generic = Recorder(_completion())
    await _provider(generic).complete(_params(thinking_mode=True))
    body = generic.body()
    assert generic.requests[0].url == f"{BASE}/v1/chat/completions"
    assert set(body) == {
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stream",
        "frequency_penalty",
    }
    assert body["stream"] is False
    qwen = Recorder(_completion())
    await _provider(qwen, family="qwen").complete(_params(thinking_mode=True))
    body = qwen.body()
    assert body["top_k"] == 40 and body["min_p"] == 0.05
    assert body["repetition_penalty"] == 1.1
    assert "frequency_penalty" not in body
    assert "<|im_end|>" in body["stop"]
    path = Recorder(_completion())
    await _provider(path, completions_path="/api/chat").complete(_params())
    assert path.requests[0].url == f"{BASE}/api/chat"


@pytest.mark.asyncio
async def test_model_field_only_when_configured():
    unset = Recorder(_completion())
    await _provider(unset).complete(_params())
    assert "model" not in unset.body()
    named = Recorder(_completion())
    await _provider(named, model="served-model").complete(_params())
    assert named.body()["model"] == "served-model"
    override = Recorder(_completion())
    await _provider(override, model="served-model").complete(_params(model="other"))
    assert override.body()["model"] == "other"


@pytest.mark.asyncio
async def test_thinking_off_uses_chat_template_kwargs_for_qwen_only():
    for family, expected in (("qwen", True), ("gemma", False), ("generic", False)):
        recorder = Recorder(_completion())
        await _provider(recorder, family=family).complete(_params(thinking_mode=False))
        body = recorder.body()
        assert ("chat_template_kwargs" in body) is expected, family
        if expected:
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
        on = Recorder(_completion())
        await _provider(on, family=family).complete(_params(thinking_mode=True))
        assert "chat_template_kwargs" not in on.body(), family


@pytest.mark.asyncio
async def test_parses_reasoning_content_and_usage():
    usage = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    recorder = Recorder(_completion("Hello.", reasoning="greet back", usage=usage))
    result = await _provider(recorder, family="qwen").complete(_params())
    assert result.text == "Hello."
    assert result.reasoning == "greet back"
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (
        12,
        5,
        17,
    )
    assert result.finish_reason == "stop"
    other_key = Recorder(_completion("Hi.", reasoning="r", key="reasoning"))
    assert (await _provider(other_key).complete(_params())).reasoning == "r"
    bare = Recorder(_completion("Hi."))
    result = await _provider(bare).complete(_params())
    assert result.reasoning is None
    assert result.input_tokens is None and result.total_tokens is None


@pytest.mark.asyncio
async def test_inline_think_tags_normalized():
    recorder = Recorder(_completion("<think>plan it</think>\n\nThe reply."))
    result = await _provider(recorder).complete(_params())
    assert (result.reasoning, result.text) == ("plan it", "The reply.")
    channel = Recorder(_completion("<|channel>thought\nweigh<channel|>Answer."))
    result = await _provider(channel, family="gemma").complete(_params())
    assert (result.reasoning, result.text) == ("weigh", "Answer.")
    only = Recorder(_completion("<think>nothing but reasoning"))
    result = await _provider(only).complete(_params())
    assert result.text == "" and result.reasoning == "nothing but reasoning"


@pytest.mark.asyncio
async def test_stream_rebuilds_reasoning_deltas_as_think_chunks():
    recorder = Recorder(
        _sse(
            _delta(reasoning="first "),
            _delta(reasoning="second"),
            _delta(content="Hello"),
            _delta(content=" there.", finish="stop"),
            "[DONE]",
        )
    )
    provider = _provider(recorder, family="qwen")
    chunks = await _collect(provider, _params(thinking_mode=True))
    assert chunks == ["<think>", "first ", "second", "</think>", "Hello", " there."]
    assert recorder.body()["stream"] is True
    assert provider.last_stream_usage["finish_reason"] == "stop"
    alt = Recorder(_sse(_delta(reasoning="r", key="reasoning"), _delta(content="c")))
    assert await _collect(_provider(alt), _params()) == [
        "<think>",
        "r",
        "</think>",
        "c",
    ]


@pytest.mark.asyncio
async def test_stream_closes_unclosed_block():
    recorder = Recorder(_sse(_delta(reasoning="cut off mid"), "[DONE]"))
    chunks = await _collect(_provider(recorder), _params(thinking_mode=True))
    assert chunks == ["<think>", "cut off mid", "</think>"]
    inline = Recorder(_sse(_delta(content="<think>unfinished"), "[DONE]"))
    chunks = await _collect(_provider(inline), _params())
    assert "".join(chunks) == "<think>unfinished</think>"


@pytest.mark.asyncio
async def test_stream_passthrough_when_server_inlines_tags():
    recorder = Recorder(
        _sse(
            _delta(content="<think>"),
            _delta(content="weigh"),
            _delta(content="</think>"),
            _delta(content="Reply."),
            "[DONE]",
        )
    )
    chunks = await _collect(_provider(recorder), _params())
    assert chunks == ["<think>", "weigh", "</think>", "Reply."]
    plain = Recorder(_sse(_delta(content="Just "), _delta(content="words."), "[DONE]"))
    assert await _collect(_provider(plain), _params()) == ["Just ", "words."]
    channel = Recorder(
        _sse(
            _delta(content="<|chan"),
            _delta(content="nel>thought\nhm<channel|>"),
            _delta(content="Done."),
        )
    )
    chunks = await _collect(_provider(channel, family="gemma"), _params())
    assert "".join(chunks) == "<think>\nhm</think>Done."


@pytest.mark.asyncio
async def test_stream_usage_requested_only_when_family_supports_it():
    usage = {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}
    qwen = Recorder(_sse(_delta(content="ok"), {"choices": [], "usage": usage}))
    provider = _provider(qwen, family="qwen")
    await _collect(provider, _params())
    assert qwen.body()["stream_options"] == {"include_usage": True}
    assert provider.last_stream_usage["input_tokens"] == 9
    assert provider.last_stream_usage["total_tokens"] == 11
    generic = Recorder(_sse(_delta(content="ok")))
    provider = _provider(generic)
    await _collect(provider, _params())
    assert "stream_options" not in generic.body()
    assert provider.last_stream_usage["input_tokens"] is None


@pytest.mark.asyncio
async def test_non_200_raises_provider_error_with_status():
    recorder = Recorder(httpx.Response(503, text="loading model"))
    with pytest.raises(ProviderError) as info:
        await _provider(recorder).complete(_params())
    assert info.value.status_code == 503
    streaming = Recorder(httpx.Response(500, text="boom"))
    with pytest.raises(ProviderError) as info:
        await _collect(_provider(streaming), _params())
    assert info.value.status_code == 500
    malformed = Recorder(httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ProviderError):
        await _provider(malformed).complete(_params())
    refused = Recorder(httpx.ConnectError("refused"))
    with pytest.raises(ProviderError) as info:
        await _provider(refused).complete(_params())
    assert info.value.status_code is None
    slow = Recorder(httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderTimeoutError):
        await _provider(slow).complete(_params())


@pytest.mark.asyncio
async def test_health_tries_paths_in_order():
    recorder = Recorder(httpx.Response(404), httpx.Response(200, json={"data": []}))
    status = await _provider(recorder).health_check()
    assert [r.url.path for r in recorder.requests] == ["/health", "/v1/models"]
    assert status.ok is True and status.verified is True
    assert "/v1/models" in status.detail
    assert status.checked_at.tzinfo is not None
    assert recorder.requests[0].extensions["timeout"]["read"] == 5.0
    first = Recorder(httpx.Response(200))
    assert (await _provider(first).health_check()).ok
    assert len(first.requests) == 1
    down = Recorder(httpx.ConnectError("refused"))
    status = await _provider(down, health_paths=("/a", "/b")).health_check()
    assert status.ok is False and len(down.requests) == 2
    # Health never makes a completion.
    for request in recorder.requests + down.requests:
        assert request.method == "GET"


@pytest.mark.asyncio
async def test_timeouts_from_settings_at_call_time(clean_env):
    settings = Settings(
        LLM_BASE_URL=BASE,
        LLM_CONNECT_TIMEOUT_S=3.0,
        LLM_REQUEST_TIMEOUT_S=40.0,
        GENERATION_TIMEOUT_S=100.0,
        LLM_MODEL="served-model",
        LLM_TEMPLATE_FAMILY="qwen",
        LLM_HEALTH_PATHS="/ready",
    )
    recorder = Recorder(_completion(), _sse(_delta(content="x")))
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    provider = LocalChatCompletionsProvider.from_settings(settings, client=client)
    await provider.complete(_params())
    timeout = recorder.requests[0].extensions["timeout"]
    assert timeout["connect"] == 3.0 and timeout["read"] == 40.0
    await _collect(provider, _params())
    assert recorder.requests[1].extensions["timeout"]["read"] == 160.0
    settings.GENERATION_TIMEOUT_S = 200.0
    await _collect(provider, _params())
    assert recorder.requests[2].extensions["timeout"]["read"] == 260.0
    assert recorder.body(0)["model"] == "served-model"
    assert provider.model_info().context_window == settings.CONTEXT_SIZE
    assert provider.health_paths == ("/ready",)
    pinned = Recorder(_sse(_delta(content="x")))
    fixed = _provider(pinned, stream_read_timeout_s=7.0)
    await _collect(fixed, _params())
    assert pinned.requests[0].extensions["timeout"]["read"] == 7.0


@pytest.mark.asyncio
async def test_base_url_derives_from_host_and_port(clean_env):
    settings = Settings(LLM_HOST="host.invalid", LLM_PORT=9100)
    provider = LocalChatCompletionsProvider.from_settings(settings)
    assert provider.base_url == "http://host.invalid:9100"


@pytest.mark.asyncio
async def test_extra_body_merged_last(clean_env):
    recorder = Recorder(_completion())
    provider = _provider(
        recorder, extra_body={"temperature": 0.1, "cache_prompt": True}
    )
    await provider.complete(_params(extra_body={"seed": 3, "cache_prompt": False}))
    body = recorder.body()
    assert body["temperature"] == 0.1
    assert body["cache_prompt"] is True
    assert body["seed"] == 3
    settings = Settings(LLM_BASE_URL=BASE, LLM_EXTRA_BODY='{"n_probs": 0}')
    from_settings = Recorder(_completion())
    client = httpx.AsyncClient(transport=httpx.MockTransport(from_settings))
    provider = LocalChatCompletionsProvider.from_settings(settings, client=client)
    await provider.complete(_params())
    assert from_settings.body()["n_probs"] == 0


@pytest.mark.asyncio
async def test_injected_client_is_not_closed(clean_env):
    provider = _provider(Recorder(_completion()))
    await provider.initialize()
    await provider.shutdown()
    assert provider._client.is_closed is False


@pytest.mark.asyncio
async def test_missing_httpx_names_the_llm_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    provider = LocalChatCompletionsProvider(base_url=BASE)
    with pytest.raises(ImportError, match="llm"):
        await provider.initialize()
    with pytest.raises(ImportError, match="llm"):
        await provider.complete(_params())
    status = await provider.health_check()
    assert status.ok is False and "llm" in status.detail
