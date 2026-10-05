"""Provider base types: the protocol, model info vocabulary, params and results."""

import inspect
from datetime import datetime, timezone

import pytest

from mtc.brain.providers.base import (
    CompletionParams,
    CompletionResult,
    HealthStatus,
    ModelInfo,
    ModelProvider,
    ProviderError,
    ProviderTimeoutError,
)


class _Minimal:
    """A provider with every protocol member and nothing else."""

    @property
    def info(self):
        return self.model_info()

    async def initialize(self):
        return None

    async def shutdown(self):
        return None

    async def complete(self, params):
        return CompletionResult(text="x")

    async def stream(self, params):
        yield "x"

    async def health_check(self):
        return HealthStatus(True, "ok", datetime.now(timezone.utc), True)

    def model_info(self):
        return ModelInfo("p", "p", "mock", None, "none", False)

    def supports_thinking(self):
        return False


def test_protocol_declares_initialize_and_shutdown():
    for name in (
        "info",
        "initialize",
        "shutdown",
        "complete",
        "stream",
        "health_check",
        "model_info",
        "supports_thinking",
    ):
        assert hasattr(ModelProvider, name), name
    assert inspect.iscoroutinefunction(ModelProvider.initialize)
    assert inspect.iscoroutinefunction(ModelProvider.shutdown)
    assert isinstance(_Minimal(), ModelProvider)

    lacking = type(
        "Lacking",
        (),
        {k: v for k, v in vars(_Minimal).items() if k not in ("initialize",)},
    )
    assert not isinstance(lacking(), ModelProvider)


def test_model_info_vocabulary():
    info = ModelInfo(
        id="local",
        name="server",
        provider_type="local",
        context_window=8192,
        cost="local",
        supports_thinking=True,
    )
    assert info.supports_vision is False
    for provider_type in ("local", "anthropic", "mock"):
        ModelInfo("x", "x", provider_type, None, "none", False)
    for cost in ("local", "paid", "none"):
        ModelInfo("x", "x", "local", None, cost, False)
    for bad_type in ("remote", "cloud", ""):
        with pytest.raises(ValueError, match="provider_type"):
            ModelInfo("x", "x", bad_type, None, "none", False)
    for bad_cost in ("free", "cheap", ""):
        with pytest.raises(ValueError, match="cost"):
            ModelInfo("x", "x", "local", None, bad_cost, False)
    with pytest.raises(ValueError, match="context_window"):
        ModelInfo("x", "x", "local", 0, "local", False)


def test_completion_params_extra_body_copied():
    body = {"seed": 1}
    messages = [{"role": "user", "content": "hello"}]
    params = CompletionParams(messages=messages, extra_body=body)
    body["seed"] = 2
    messages.append({"role": "user", "content": "later"})
    assert params.extra_body == {"seed": 1}
    assert len(params.messages) == 1
    defaults = CompletionParams(messages=[])
    assert defaults.extra_body == {}
    other = CompletionParams(messages=[])
    defaults.extra_body["x"] = 1
    assert other.extra_body == {}
    assert defaults.stream is False and defaults.thinking_mode is False
    for name in (
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "frequency_penalty",
        "stop",
        "model",
    ):
        assert getattr(defaults, name) is None, name


def test_completion_result_usage_optional():
    result = CompletionResult(text="hi")
    assert result.reasoning is None
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.total_tokens is None
    assert result.finish_reason is None
    full = CompletionResult("hi", "why", 3, 2, 5, "stop")
    assert (full.input_tokens, full.output_tokens, full.total_tokens) == (3, 2, 5)


def test_health_status_and_errors():
    now = datetime(2000, 1, 1, tzinfo=timezone.utc)
    status = HealthStatus(ok=False, detail="down", checked_at=now, verified=True)
    assert status.as_dict() == {
        "ok": False,
        "detail": "down",
        "checked_at": "2000-01-01T00:00:00+00:00",
        "verified": True,
    }
    with pytest.raises(ValueError, match="timezone"):
        HealthStatus(
            ok=True, detail="x", checked_at=datetime(2000, 1, 1), verified=True
        )
    error = ProviderError("server said no", status_code=503)
    assert isinstance(error, RuntimeError) and error.status_code == 503
    assert ProviderError("no status").status_code is None
    assert issubclass(ProviderTimeoutError, ProviderError)
