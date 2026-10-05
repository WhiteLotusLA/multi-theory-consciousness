"""The DCM evaluator completes through the provider layer; scoring is unchanged."""

import ast
import functools
import json
import os
import secrets
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mtc.assessment import dcm_evaluator
from mtc.assessment.dcm_evaluator import EVALUATION_SYSTEM_PROMPT, DCMEvaluator
from mtc.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "mtc" / "assessment" / "dcm_evaluator.py"
REPLY = json.dumps(
    {
        "Curiosity": {"score": 0.8, "reasoning": "asks questions"},
        "Play": {"score": 1.7, "reasoning": "clamped to one"},
        "Agency": {"score": "0.25", "reasoning": "numeric text"},
    }
)
NAMES = ["Curiosity", "Play", "Agency", "Memory"]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _use_settings(monkeypatch, **values):
    settings = Settings(**values)
    monkeypatch.setattr(dcm_evaluator, "get_settings", lambda: settings)
    return settings


def test_local_mode_completes_through_local_provider(clean_env, monkeypatch):
    httpx = pytest.importorskip("httpx")
    _use_settings(monkeypatch, LLM_BASE_URL="http://host.invalid:8080")
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        message = {"role": "assistant", "content": "<think>score them</think>" + REPLY}
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        functools.partial(real_client, transport=httpx.MockTransport(handler)),
    )
    evaluator = DCMEvaluator(cache_dir=str(clean_env / "dcm"))
    scores = evaluator.evaluate_indicators(NAMES, None, "a test system", ["hello"])
    assert scores == evaluator._parse_response(REPLY, NAMES)
    request = requests_seen[0]
    assert str(request.url) == "http://host.invalid:8080/v1/chat/completions"
    body = json.loads(request.content)
    assert body["messages"][0] == {
        "role": "system",
        "content": EVALUATION_SYSTEM_PROMPT,
    }
    assert body["messages"][1]["role"] == "user"
    assert "a test system" in body["messages"][1]["content"]
    assert body["temperature"] == 0.3 and body["max_tokens"] == 4096
    # The literal model name is gone: an unset LLM_MODEL sends no model field.
    assert "model" not in body


def test_anthropic_mode_requires_model_and_key(clean_env, monkeypatch):
    evaluator = DCMEvaluator(cache_dir=str(clean_env / "dcm"))
    _use_settings(monkeypatch, ANTHROPIC_API_KEY=secrets.token_hex(16))
    with pytest.raises(ValueError, match="ANTHROPIC_MODEL"):
        evaluator.evaluate_indicators(["Play"], None, "d", ["s"], mode="claude")
    _use_settings(monkeypatch, ANTHROPIC_MODEL="test-model-id")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        evaluator.evaluate_indicators(["Play"], None, "d", ["s"], mode="claude")

    calls = []

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_final_message(self):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=REPLY)],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            )

    class FakeAsyncAnthropic:
        def __init__(self, api_key):
            self.api_key = api_key
            self.messages = SimpleNamespace(stream=self._stream)

        def _stream(self, **kwargs):
            calls.append(kwargs)
            return FakeStream()

        async def close(self):
            return None

    fake_sdk = ModuleType("anthropic")
    fake_sdk.AsyncAnthropic = FakeAsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)
    _use_settings(
        monkeypatch,
        ANTHROPIC_MODEL="test-model-id",
        ANTHROPIC_API_KEY=secrets.token_hex(16),
    )
    scores = evaluator.evaluate_indicators(NAMES, None, "d", ["s"], mode="claude")
    assert scores == evaluator._parse_response(REPLY, NAMES)
    assert calls[0]["model"] == "test-model-id"
    assert calls[0]["system"] == EVALUATION_SYSTEM_PROMPT
    assert calls[0]["max_tokens"] == 4096
    assert "temperature" not in calls[0]


@pytest.mark.asyncio
async def test_sync_entry_inside_running_loop_raises_naming_aevaluate(clean_env):
    async def completer(system, prompt):
        return REPLY

    evaluator = DCMEvaluator(cache_dir=str(clean_env / "dcm"), completer=completer)
    with pytest.raises(RuntimeError, match="aevaluate"):
        evaluator.evaluate_indicators(NAMES, None, "d", ["s"])
    scores = await evaluator.aevaluate(NAMES, None, "d", ["s"])
    assert scores == evaluator._parse_response(REPLY, NAMES)


def test_no_requests_import():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "requests" not in imported
    assert "anthropic" not in imported
    assert "httpx" not in imported
    assert '"model": "default"' not in SOURCE.read_text(encoding="utf-8")


def test_scoring_unchanged_for_same_reply(clean_env):
    seen = []

    async def completer(system, prompt):
        seen.append((system, prompt))
        return REPLY

    evaluator = DCMEvaluator(cache_dir=str(clean_env / "dcm"), completer=completer)
    scores = evaluator.evaluate_indicators(NAMES, None, "d", ["s"], force=True)
    expected = evaluator._parse_response(REPLY, NAMES)
    assert scores == expected
    assert scores == {
        "Curiosity": pytest.approx(0.8),
        "Play": pytest.approx(1.0),
        "Agency": pytest.approx(0.25),
    }
    assert seen[0][0] == EVALUATION_SYSTEM_PROMPT
    assert seen[0][1] == evaluator._build_batch_prompt(NAMES, "d", ["s"])
    # Memory was not scored, so nothing is cached and the next call asks again.
    evaluator.evaluate_indicators(NAMES, None, "d", ["s"])
    assert len(seen) == 2


def test_empty_reply_leaves_the_batch_unscored(clean_env):
    async def completer(system, prompt):
        return "   "

    evaluator = DCMEvaluator(cache_dir=str(clean_env / "dcm"), completer=completer)
    assert evaluator.evaluate_indicators(["Play"], None, "d", ["s"]) == {}
