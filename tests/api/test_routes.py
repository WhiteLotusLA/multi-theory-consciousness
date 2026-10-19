"""The HTTP routes over a runtime built with the mock provider and in-memory
stores (no network, no store client, no model)."""

import asyncio
import functools
import json
import os
import secrets
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from mtc.api.auth import API_KEY_HEADER, SPEAKER_TOKEN_HEADER
from mtc.api.schemas import (
    ERROR_CODES,
    INTEGRATION_PROXY_LABEL,
    HealthResponse,
    TimeoutPayload,
    TurnPayload,
    brain_verified,
    health_status,
)
from mtc.api.server import ERROR_STATUS, create_app
from mtc.assessment.assessment import (
    ConsciousnessAssessment,
    ConsciousnessReport,
    ConsciousnessTheory,
)
from mtc.brain.brain import LLMBrain
from mtc.brain.providers.base import (
    HealthStatus,
    ProviderError,
    ProviderTimeoutError,
)
from mtc.brain.providers.mock_provider import MockEvent, MockProvider, _last_user_text
from mtc.brain.providers.registry import ProviderRegistry
from mtc.core import config as config_module
from mtc.core.embeddings import HashEmbedder
from mtc.core.logger import reset_logging
from mtc.memory.inmemory import InMemoryGraphStore
from mtc.pipeline.context_types import SAFETY_GATE_STATES, TurnResult
from mtc.pipeline.runtime import build_runtime
from tests.api.test_auth import (
    FakePipeline,
    FakeRuntime,
    api_settings,
    isolated_env,
    make_app,
)

ROOT = Path(__file__).resolve().parents[2]
TOKEN_ENV = "MTC_TEST_ROUTES_TOKEN_ALICE"
ORIGIN = "http://ui.invalid"

# Markers in a chat message that make the keyed mock misbehave on that turn
# only; background brain users never send them.
EMPTY = "[[reply-empty]]"
TIMEOUT = "[[reply-timeout]]"
DOWN = "[[provider-down]]"

pytestmark = pytest.mark.timeout(900)


class KeyedMock(MockProvider):
    """The mock provider, with a scripted failure for marked messages."""

    def __init__(self) -> None:
        super().__init__()
        self.keyed = {
            EMPTY: MockEvent.EMPTY,
            TIMEOUT: ProviderTimeoutError("the mock timed out"),
            DOWN: ProviderError("the mock is down", status_code=503),
        }

    async def _reply(self, params):
        text = _last_user_text(params.messages)
        for marker, item in self.keyed.items():
            if marker in text:
                self._script = [item]
                break
        return await super()._reply(params)


class Served:
    def __init__(self, client, app, cfg, key, token) -> None:
        self.client = client
        self.app = app
        self.cfg = cfg
        self.key = key
        self.token = token

    @property
    def runtime(self):
        return self.app.state.runtime

    def get(self, path: str, **kwargs: Any):
        return self.client.get(path, headers={API_KEY_HEADER: self.key}, **kwargs)

    def post(self, path: str, body: Any = None, headers=None):
        return self.client.post(
            path, json=body, headers={API_KEY_HEADER: self.key, **(headers or {})}
        )

    def chat(self, message: str, speaker=None, token=None):
        headers = {SPEAKER_TOKEN_HEADER: token} if token else None
        body: Dict[str, Any] = {"message": message}
        if speaker is not None:
            body["speaker"] = speaker
        return self.post("/api/chat", body, headers)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("routes")
    key = secrets.token_hex(24)
    token = secrets.token_hex(24)
    with pytest.MonkeyPatch.context() as mp:
        for name in list(os.environ):
            if name.startswith("MTC_"):
                mp.delenv(name, raising=False)
        mp.setenv("MTC_DEVICE", "cpu")
        mp.setenv(TOKEN_ENV, token)
        mp.chdir(tmp)
        mp.setattr(config_module, "_settings", None)
        speakers = tmp / "speakers.yaml"
        speakers.write_text(
            "speakers:\n"
            "  - name: Alice\n"
            "    role: creator\n"
            "    trust: 0.9\n"
            f"    token_env: {TOKEN_ENV}\n",
            encoding="utf-8",
        )
        cfg = api_settings(
            tmp,
            API_KEY=key,
            SYSTEM_NAME="Testagent",
            SPEAKERS_PATH=str(speakers),
            TRUSTED_SPEAKERS="Alice",
            METRICS_UPDATE_INTERVAL=0,
            API_CORS_ORIGINS=ORIGIN,
            CTM_CONFIG_DIR=str(ROOT / "config" / "ctm"),
        )

        async def builder(settings):
            brain = LLMBrain.from_settings(settings)
            brain.registry.register(KeyedMock())
            brain.registry.set_active("mock")
            return await build_runtime(
                settings,
                brain=brain,
                embedder=HashEmbedder(),
                allow_non_semantic=True,
                substrates=False,
            )

        app = create_app(settings=cfg, runtime_builder=builder)
        with TestClient(app) as client:
            yield Served(client, app, cfg, key, token)
        reset_logging()


def test_meta_lists_theories_and_indicator_metadata_from_assessment(served):
    body = served.get("/api/meta").json()
    assessment = ConsciousnessAssessment()
    assert body["system_name"] == "Testagent"
    assert {t["id"] for t in body["theories"]} == {t.name for t in ConsciousnessTheory}
    assert len(body["theories"]) == 8
    names = [entry["name"] for entry in body["indicators"]]
    assert names == list(assessment.indicator_configs)
    assert len(names) == 30
    negative = [entry["name"] for entry in body["negative_indicators"]]
    assert negative == list(assessment.negative_indicator_configs)
    assert len(negative) == 3
    for entry in body["indicators"]:
        config = assessment.indicator_configs[entry["name"]]
        assert entry["theory"] == config["theory"].name
        assert entry["threshold"] == config["threshold"]
        assert entry["butlin_id"] == config["butlin_id"]
        assert entry["polarity"] == "positive"
    assert all(e["polarity"] == "negative" for e in body["negative_indicators"])
    assert body["capabilities"]["vision"] is False
    assert body["capabilities"]["tools"] == []
    assert body["narration"] and not any(body["narration"].values())
    assert body["auth"]["enabled"] is True
    assert body["providers"]["active"] == "mock"


def test_chat_round_trip(served):
    response = served.chat("Hello, how are you?")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["response"].startswith("[mock]")
    assert body["provider_id"] == "mock"
    assert body["error_code"] is None
    assert body["message_count"] >= 2
    assert "<think>" not in body["response"]


def test_chat_no_answer_502(served):
    response = served.chat(f"Say nothing {EMPTY}")
    assert response.status_code == 502
    body = response.json()
    assert body["error_code"] == "no_answer"
    assert body["success"] is False and body["response"] is None


def test_chat_brain_unavailable_503(served):
    response = served.chat(f"Anyone there {DOWN}")
    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "brain_unavailable"
    assert body["response"] is None


def test_chat_timeout_504(served):
    response = served.chat(f"Take forever {TIMEOUT}")
    assert response.status_code == 504
    body = response.json()
    assert body["error_code"] == "generation_timeout"
    assert body["response"] is None


class HangingPipeline(FakePipeline):
    """A turn that never streams (``first_chunk`` False) or stalls after one
    chunk; it only ends when the bound cancels it."""

    def __init__(self, first_chunk: bool) -> None:
        super().__init__()
        self.first_chunk = first_chunk
        self.cancelled = False

    async def send_message(
        self,
        message,
        speaker=None,
        speaker_token=None,
        *,
        stream_callback=None,
        request_id=None,
        timeout_s=None,
        on_generated=None,
    ):
        self.calls.append({"message": message})
        if self.first_chunk and stream_callback is not None:
            await stream_callback("[fake] partial")
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("the generation bound should have cancelled the turn")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    yield from isolated_env(monkeypatch, tmp_path)


@pytest.mark.parametrize(
    "first_chunk, bound", [(False, "first_token"), (True, "stall")]
)
def test_bound_timeout_reports_only_what_is_known(clean_env, first_chunk, bound):
    """A turn the first-token or stall bound cancelled has no reply, session,
    speaker label or count to report: the 504 body carries the bound, its
    limit, the elapsed time, when the first token came (None when none did)
    and the turn's state, and nothing stands in for the rest."""
    key = secrets.token_hex(24)
    cfg = api_settings(
        clean_env,
        API_KEY=key,
        GENERATION_TIMEOUT_S=0.05,
        GENERATION_STALL_TIMEOUT_S=0.05,
    )
    runtime = FakeRuntime(cfg)
    runtime.pipeline = HangingPipeline(first_chunk)
    with TestClient(make_app(cfg, runtime)) as client:
        response = client.post(
            "/api/chat", json={"message": "hello"}, headers={API_KEY_HEADER: key}
        )
    assert response.status_code == 504
    body = response.json()
    assert set(body) == {
        "success",
        "error",
        "error_code",
        "bound",
        "limit_s",
        "elapsed_s",
        "first_token_s",
        "state",
    }
    TimeoutPayload.model_validate(body)
    assert body["success"] is False
    assert body["error_code"] == "generation_timeout"
    assert body["bound"] == bound
    assert body["limit_s"] == 0.05
    assert body["elapsed_s"] >= 0.05
    assert body["state"] == "cancelled"
    if first_chunk:
        assert 0.0 <= body["first_token_s"] <= body["elapsed_s"]
        assert "after the first token" in body["error"]
    else:
        assert body["first_token_s"] is None
        assert "no first token" in body["error"]
    assert runtime.pipeline.cancelled is True


def test_chat_returns_the_reply_when_post_turn_is_slow(clean_env):
    """A reply that is stored before the post-turn work runs is returned with
    200 even when that work outlasts the stall bound: the route lifts its
    generation bounds once the pipeline reports the reply stored."""
    key = secrets.token_hex(24)
    cfg = api_settings(
        clean_env,
        API_KEY=key,
        GENERATION_TIMEOUT_S=60.0,
        GENERATION_STALL_TIMEOUT_S=0.2,
        METRICS_UPDATE_INTERVAL=0,
        CTM_CONFIG_DIR=str(ROOT / "config" / "ctm"),
    )

    async def builder(settings):
        return await build_runtime(
            settings, embedder=HashEmbedder(), allow_non_semantic=True, substrates=False
        )

    app = create_app(settings=cfg, runtime_builder=builder)
    with TestClient(app) as client:
        pipeline = app.state.runtime.pipeline
        post_turn = pipeline.post_turn
        real_run = post_turn.run
        ran = []

        async def slow_run(record):
            # Five times the stall bound after the last streamed chunk.
            await asyncio.sleep(1.0)
            ran.append(record.turn_number)
            return await real_run(record)

        post_turn.run = slow_run
        try:
            response = client.post(
                "/api/chat",
                json={"message": "Tell me about the garden."},
                headers={API_KEY_HEADER: key},
            )
        finally:
            post_turn.run = real_run
        status = pipeline.status()
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["success"] is True
    assert body["response"].startswith("[mock] reply to: Tell me about the garden.")
    assert ran == [1], "the post-turn work ran to its end"
    assert status["cancelled"] == 0
    assert "post_turn" not in status["failures"]


class UnverifiedMock(KeyedMock):
    """A provider whose health check reflects configuration only (ok, not
    verified by an exchange), as a hosted provider's does before any call."""

    async def health_check(self) -> HealthStatus:
        return HealthStatus(True, "configured", datetime.now(timezone.utc), False)


def _health_app(tmp_path, provider):
    """An application over a runtime whose brain has ``provider`` only; no
    continuous thought, so no background user calls the brain."""
    key = secrets.token_hex(24)
    cfg = api_settings(
        tmp_path,
        API_KEY=key,
        CTM_ENABLED=False,
        METRICS_UPDATE_INTERVAL=0,
    )

    async def builder(settings):
        registry = ProviderRegistry(
            state_path=Path(settings.DATA_DIRECTORY) / "provider_state.json",
            settings=settings,
        )
        registry.register(provider)
        registry.set_active("mock")
        return await build_runtime(
            settings,
            brain=LLMBrain(registry=registry, settings=settings),
            embedder=HashEmbedder(),
            allow_non_semantic=True,
            substrates=False,
        )

    return create_app(settings=cfg, runtime_builder=builder), key


def test_health_reads_unverified_until_a_real_call_succeeds(clean_env):
    """A provider whose health only reflects configuration is not reported as
    ok: /health reads unverified (brain_verified false) until a real call
    returns a completion, then ok (brain_verified true)."""
    app, key = _health_app(clean_env, UnverifiedMock())
    with TestClient(app) as client:
        before = client.get("/health")
        turn = client.post(
            "/api/chat", json={"message": "hello"}, headers={API_KEY_HEADER: key}
        )
        after = client.get("/health")
    assert before.status_code == 200
    HealthResponse.model_validate(before.json())
    assert before.json()["status"] == "unverified"
    assert before.json()["ready"] is True
    assert before.json()["brain_verified"] is False
    assert turn.status_code == 200
    assert after.json()["status"] == "ok"
    assert after.json()["brain_verified"] is True


def test_health_reads_degraded_after_a_failed_call(clean_env):
    """A verified, healthy provider that then fails a real call makes /health
    read degraded (still 200, still unauthenticated); the next successful
    call makes it ok again, and the record never refused a call."""
    app, key = _health_app(clean_env, KeyedMock())
    with TestClient(app) as client:
        healthy = client.get("/health").json()
        failed = client.post(
            "/api/chat",
            json={"message": f"Anyone there {DOWN}"},
            headers={API_KEY_HEADER: key},
        )
        degraded = client.get("/health")
        again = client.post(
            "/api/chat", json={"message": "hello"}, headers={API_KEY_HEADER: key}
        )
        recovered = client.get("/health").json()
    assert healthy["status"] == "ok" and healthy["brain_verified"] is True
    assert failed.status_code == 503
    assert degraded.status_code == 200
    assert degraded.json()["status"] == "degraded"
    assert again.status_code == 200
    assert recovered["status"] == "ok"


def test_health_keeps_the_old_rule_for_a_brain_without_a_health_record(clean_env):
    """A runtime brain that keeps no health record (only ``available``) reads
    ok when available and ready, and brain_verified is unknown (null)."""
    key = secrets.token_hex(24)
    cfg = api_settings(clean_env, API_KEY=key)
    runtime = FakeRuntime(cfg)
    with TestClient(make_app(cfg, runtime)) as client:
        body = client.get("/health").json()
        runtime.brain.available = False
        down = client.get("/health").json()
    assert body["status"] == "ok" and body["brain_verified"] is None
    assert down["status"] == "degraded" and down["brain_verified"] is None


class _RecordedBrain:
    """A brain stand-in whose ``status()`` returns a fixed record or raises."""

    def __init__(self, available, status=None, raises=False):
        self.available = available
        self._status = status
        self._raises = raises

    def status(self):
        if self._raises:
            raise RuntimeError("status unreadable")
        return self._status


def test_brain_verified_is_false_for_a_failed_verified_exchange():
    """A health record that came from a real exchange that failed (verified
    but not ok) does not confirm the provider answers: brain_verified False,
    and /health reads degraded."""
    brain = _RecordedBrain(
        False, {"health": {"ok": False, "verified": True}, "last_call_ok": None}
    )
    assert brain_verified(brain) is False
    assert health_status(True, brain) == ("degraded", False)
    answered = _RecordedBrain(
        True, {"health": {"ok": True, "verified": True}, "last_call_ok": None}
    )
    assert brain_verified(answered) is True
    assert health_status(True, answered) == ("ok", True)


def test_unreadable_health_record_reads_unverified(caplog):
    """A brain whose status cannot be read is not reported ok on availability
    alone: /health reads unverified (degraded when unavailable), brain_verified
    is False, and the failure is logged at DEBUG."""
    import logging

    brain = _RecordedBrain(True, raises=True)
    with caplog.at_level(logging.DEBUG, logger="mtc.api.schemas"):
        assert health_status(True, brain) == ("unverified", False)
        assert brain_verified(brain) is False
    assert any(
        record.levelno == logging.DEBUG and "RuntimeError" in record.getMessage()
        for record in caplog.records
    )
    assert health_status(True, _RecordedBrain(False, raises=True)) == (
        "degraded",
        False,
    )
    assert health_status(False, brain) == ("degraded", False)


def test_meta_reports_missing_indicator_metadata_as_null(clean_env, monkeypatch):
    """An indicator without a threshold, a weight, a description or a theory
    reads null, never a stand-in 0.0, empty text or the word None; a
    non-finite number is not a value either. The polarity comes from the list
    the indicator belongs to when its config names none."""
    from mtc.assessment import assessment as assessment_module

    theory = next(iter(ConsciousnessTheory))

    class SparseAssessment:
        indicator_configs = {
            "complete": {
                "theory": theory,
                "threshold": 0.5,
                "weight": 1.0,
                "description": "a complete entry",
                "polarity": "positive",
            },
            "no_threshold": {"theory": theory, "weight": 0.2},
            "nan_weight": {"theory": theory, "threshold": 0.3, "weight": float("nan")},
        }
        negative_indicator_configs = {"no_polarity": {"threshold": 0.1}}

    monkeypatch.setattr(assessment_module, "ConsciousnessAssessment", SparseAssessment)
    key = secrets.token_hex(24)
    cfg = api_settings(clean_env, API_KEY=key)
    with TestClient(make_app(cfg)) as client:
        response = client.get("/api/meta", headers={API_KEY_HEADER: key})
    assert response.status_code == 200
    body = response.json()
    entries = {entry["name"]: entry for entry in body["indicators"]}
    assert entries["complete"]["threshold"] == 0.5
    assert entries["complete"]["weight"] == 1.0
    assert entries["complete"]["description"] == "a complete entry"
    assert entries["complete"]["theory"] == theory.name
    assert entries["no_threshold"]["threshold"] is None
    assert entries["no_threshold"]["weight"] == 0.2
    assert entries["no_threshold"]["description"] is None
    assert entries["no_threshold"]["polarity"] == "positive"
    assert entries["nan_weight"]["weight"] is None
    assert entries["nan_weight"]["threshold"] == 0.3
    (negative,) = body["negative_indicators"]
    assert negative["name"] == "no_polarity"
    assert negative["polarity"] == "negative"
    assert negative["theory"] is None
    assert negative["weight"] is None
    assert negative["threshold"] == 0.1


def test_chat_blocked_403(served):
    class Denying:
        async def pre_message_check(self, message, sender, speaker_role=None):
            return {"safe": False, "decision": "deny", "checked": True}

    pipeline = served.runtime.pipeline
    original = pipeline.safety
    pipeline.safety = Denying()
    try:
        response = served.chat("anything at all", speaker="Guest")
    finally:
        pipeline.safety = original
    assert response.status_code == 403
    body = response.json()
    assert body["blocked"] is True and body["error_code"] == "blocked"
    assert body["response"] is None


class Unavailable:
    """A safety check that cannot run."""

    async def pre_message_check(self, message, sender, speaker_role=None):
        raise RuntimeError("the safety stack is down")


def test_safety_unavailable_is_503(served):
    """With the default fail-closed gate a check that could not run refuses an
    untrusted speaker's turn: 503 with the turn's payload (error code
    ``safety_unavailable``, gate ``unchecked``, no reply) and nothing stored.
    The trusted creator is not refused: the gate is bypassed for that turn."""
    pipeline = served.runtime.pipeline
    repository = pipeline.persistence.repository
    text = "Hello from a speaker the gate could not check"
    original = pipeline.safety
    memory_writes = pipeline.persistence.status()["written"]["working_memory"]
    pipeline.safety = Unavailable()
    try:
        refused = served.chat(text, speaker="Guest")
        memory_after_refusal = pipeline.persistence.status()["written"][
            "working_memory"
        ]
        rows = served.client.portal.call(
            functools.partial(repository.timeline, 1.0, 1000, addressee="[Guest]")
        )
        trusted = served.chat("Hello there", speaker="Alice", token=served.token)
    finally:
        pipeline.safety = original
    assert served.cfg.SAFETY_GATE_FAIL_CLOSED is True
    assert refused.status_code == 503
    body = refused.json()
    TurnPayload.model_validate(body)
    assert body["success"] is False and body["response"] is None
    assert body["error_code"] == "safety_unavailable"
    assert body["safety_gate"] == "unchecked"
    assert body["blocked"] is False and body["safety_decision"] is None
    assert body["error"] == (
        "The safety check could not run; the message was not processed and "
        "nothing was stored."
    )
    # The turn stored nothing: no message row, no working-memory entry. (The
    # conversation record is opened only after the gate let a turn through;
    # this test checks the message row and the working-memory write.)
    assert not [row for row in rows if row["content"] == text]
    assert memory_after_refusal == memory_writes
    assert trusted.status_code == 200
    assert trusted.json()["safety_gate"] == "bypassed"


def test_turn_payload_model_carries_the_gate_fields():
    """The payload model names exactly the gate states the pipeline reports,
    and every key of ``TurnResult.to_payload()`` is a field of the model."""
    gate = TurnPayload.model_fields["safety_gate"].annotation
    assert set(typing.get_args(typing.get_args(gate)[0])) == set(SAFETY_GATE_STATES)
    result = TurnResult(
        success=False,
        response=None,
        thinking=None,
        conversation_id=None,
        message_count=0,
        timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),
        error_code="blocked",
        blocked=True,
        safety_gate="applied",
        safety_decision="emergency_shutdown",
    )
    payload = result.to_payload()
    assert set(payload) <= set(TurnPayload.model_fields)
    model = TurnPayload.model_validate(payload)
    assert model.safety_gate == "applied"
    assert model.safety_decision == "emergency_shutdown"
    assert "safety_unavailable" in ERROR_CODES
    assert ERROR_STATUS["safety_unavailable"] == 503


def test_status_exposes_counters_timings_first_run_and_safety_components(served):
    assert served.chat("A turn before the status page.").status_code == 200
    body = served.get("/api/consciousness/status").json()
    orchestrator = body["orchestrator"]
    assert isinstance(orchestrator["failures"], dict)
    assert orchestrator["cycles"] >= 1
    assert orchestrator["timings"], "stage timings are reported"
    first_run = body["first_run"]
    assert {"phi", "baseline", "rpt", "causal_emergence"} <= set(first_run)
    modulations = body["theory_modulations"]["modulations"]
    assert modulations and all("consumed" in item for item in modulations)
    assert "enabled" in body["gate"]
    assert "quiet_reason" in body["ctm"]
    assert body["stores"]
    assert body["provider"]["active_provider"] == "mock"
    safety = body["safety"]
    for component in ("privacy", "wellbeing", "alerts", "safety_monitor"):
        assert component in safety
    assert body["background"] is not None
    assert body["pipeline"]["turns"] >= 1


def test_state_labels_phi_as_integration_proxy(served):
    assert served.chat("A turn so the state exists.").status_code == 200
    body = served.get("/api/consciousness/state").json()
    assert body["available"] is True
    proxy = body["integration_proxy"]
    assert proxy["label"] == INTEGRATION_PROXY_LABEL
    assert "not IIT Phi" in proxy["description"]
    assert proxy["available"] is True
    assert "value" in proxy and "warming_up" in proxy
    # The Phi slots appear only under the labelled proxy (a field whose name
    # holds "phi" as a word; "autobiographic" is not one).
    assert not [n for n in body["state"] if "phi" in n.lower().split("_")]
    assert "architectural_phi" not in body["state"]
    last = body["last_turn"]
    assert last["cycle_ok"] is True
    assert "layers_available" in last and "absent_signals" in last


def test_state_carries_the_layers_a_pipeline_keeps():
    from types import SimpleNamespace

    from mtc.api.schemas import state_payload
    from mtc.pipeline.context_types import ModulatedContext, TheorySections

    modulated = ModulatedContext(
        prompt_context="Attention rests on the incoming message.",
        theory_sections=TheorySections(attention_guidance="focus: message"),
        generation_params={"temperature": 0.7},
        behavioral_directives=["Answer briefly."],
        consciousness_self_awareness="",
        consciousness_state=None,
        workspace_winner="incoming message",
        winner_is_message=True,
        cycle_ok=True,
        absent_signals={"rpt_deep_recurrence": "cadence_not_reached"},
        pad_sources_measured=("emotion",),
    )
    pipeline = SimpleNamespace(
        last_modulated=modulated, status=lambda: {"last_turn": {"cycle_ok": True}}
    )
    body = state_payload(SimpleNamespace(pipeline=pipeline, orchestrator=None))
    last = body["last_turn"]
    assert last["layers_available"] is True
    layers = last["layers"]
    assert layers["narrative"] == "Attention rests on the incoming message."
    assert layers["theory_sections"] == {"attention_guidance": "focus: message"}
    assert layers["generation_params"] == {"temperature": 0.7}
    assert layers["behavioral_directives"] == ["Answer briefly."]
    assert last["absent_signals"] == {"rpt_deep_recurrence": "cadence_not_reached"}
    assert body["available"] is False and body["state"] is None
    assert body["integration_proxy"]["label"] == INTEGRATION_PROXY_LABEL


def _insufficient_report() -> ConsciousnessReport:
    return ConsciousnessReport(
        session_id="test",
        timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),
        overall_score=None,
        architecture_functional=False,
        confidence=0.0,
        indicator_results={},
        passing_count=0,
        total_indicators=30,
        theory_scores={},
        phi_measurement=None,
        status="insufficient_data",
        coverage=0.0,
    )


def test_assessment_reports_insufficient_data_as_null_not_zero(served, monkeypatch):
    served.app.state.assessment = None
    before = served.get("/api/consciousness/assessment").json()
    assert before["status"] == "not_run" and before["overall_score"] is None

    async def assess_nothing():
        return _insufficient_report()

    monkeypatch.setattr(served.runtime.orchestrator, "assess_now", assess_nothing)
    fresh = served.get("/api/consciousness/assessment?refresh=true").json()
    assert fresh["status"] == "insufficient_data"
    assert fresh["overall_score"] is None
    assert fresh["report"]["overall_score"] is None
    assert fresh["cached"] is False
    cached = served.get("/api/consciousness/assessment").json()
    assert cached["overall_score"] is None and cached["cached"] is True
    # A second refresh inside the cache window reuses the report.
    again = served.get("/api/consciousness/assessment?refresh=true").json()
    assert again["cached"] is True

    # A report that measured nothing but carries 0.0 from absent indicators
    # is still reported as null.
    zero = _insufficient_report()
    zero.overall_score = 0.0

    async def assess_zero():
        return zero

    served.app.state.assessment = None
    monkeypatch.setattr(served.runtime.orchestrator, "assess_now", assess_zero)
    withheld = served.get("/api/consciousness/assessment?refresh=true").json()
    assert withheld["status"] == "insufficient_data"
    assert withheld["overall_score"] is None and withheld["overall_withheld"] is True
    assert withheld["report"]["overall_score"] is None

    async def assess_fails():
        return None

    served.app.state.assessment = None
    monkeypatch.setattr(served.runtime.orchestrator, "assess_now", assess_fails)
    failed = served.get("/api/consciousness/assessment?refresh=true").json()
    assert failed["status"] == "failed" and failed["overall_score"] is None
    # No report exists, so it has no age, coverage or body either.
    assert failed["age_s"] is None
    assert failed["coverage"] is None and failed["report"] is None
    served.app.state.assessment = None


def test_real_assessment_never_reports_unmeasured_zero(served):
    served.app.state.assessment = None
    body = served.get("/api/consciousness/assessment?refresh=true").json()
    assert body["status"] in ("measured", "partial", "insufficient_data", "failed")
    if body["status"] == "insufficient_data":
        assert body["overall_score"] is None
    elif body["overall_score"] is not None:
        assert 0.0 <= body["overall_score"] <= 1.0
    served.app.state.assessment = None


def test_snapshots_history_and_events(served):
    runtime = served.runtime
    served.client.portal.call(runtime.orchestrator.refresh_baseline, "test")
    body = served.get("/api/consciousness/snapshots?hours=24&limit=10").json()
    assert body["available"] is True
    assert body["hours"] == 24 and body["limit"] == 10
    assert isinstance(body["history"], list) and isinstance(body["events"], list)
    assert len(body["history"]) >= 1
    record = body["history"][0]
    assert {"timestamp", "trigger", "overall_score", "status"} <= set(record)
    for record in body["history"]:
        if record["status"] == "insufficient_data":
            assert record["overall_score"] is None
    assert served.get("/api/consciousness/snapshots?hours=0").status_code == 422
    assert served.get("/api/consciousness/snapshots?limit=0").status_code == 422


def test_memory_search_labels_method(served):
    assert served.chat("Remember that the garden has tulips.").status_code == 200
    response = served.post("/api/memory/search", {"query": "garden tulips", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["method"] in ("vector", "text", "none")
    assert "query" not in body["stats"], "the search text is not echoed"
    assert isinstance(body["memories"], list)
    assert served.post("/api/memory/search", {"query": ""}).status_code == 422
    assert (
        served.post("/api/memory/search", {"query": "x", "top_k": 51}).status_code
        == 422
    )


def test_concepts_list_and_related(served):
    # NEO4J_ENABLED is false: the runtime keeps the concept graph in process
    # memory, and the routes read that very store.
    graph = served.runtime.memory.manager.graph
    assert isinstance(graph, InMemoryGraphStore)
    assert served.runtime.memory.graph is graph
    first = served.get("/api/knowledge/concepts").json()
    assert first["available"] is True and first["backend"] == "memory"
    graph.merge_concept("gardening")
    graph.merge_concept("tulips")
    graph.relate_concepts("gardening", "tulips", "relates_to", 0.8)
    listing = served.get("/api/knowledge/concepts?limit=50").json()
    related = served.post("/api/knowledge/concepts", {"name": "gardening"}).json()
    invalid = served.post("/api/knowledge/concepts", {"name": ""})
    assert listing["available"] is True and listing["backend"] == "memory"
    assert {"gardening", "tulips"} <= set(listing["concepts"])
    assert related["available"] is True and related["concept"] == "gardening"
    assert [row["concept"] for row in related["related"]] == ["tulips"]
    assert invalid.status_code == 422


def test_ctm_start_stop_status_with_quiet_reason(served):
    status = served.get("/api/ctm/status").json()
    assert status["available"] is True
    assert "quiet_reason" in status
    stopped = served.post("/api/ctm/stop").json()
    assert stopped["running"] is False
    started = served.post("/api/ctm/start", {"state": "drowsy"}).json()
    assert started["running"] is True
    assert started["status"]["consciousness_state"] == "drowsy"
    assert served.post("/api/ctm/start", {"state": "hyper"}).status_code == 422
    served.post("/api/ctm/stop")
    restarted = served.post("/api/ctm/start", {"state": "awake"}).json()
    assert restarted["running"] is True


def test_models_lists_registered_providers_and_active(served):
    body = served.get("/api/models").json()
    assert {"local", "mock"} <= set(body["providers"])
    assert body["active"] == "mock"
    assert body["available"] is True
    assert body["health"]["ok"] is True


def test_cors_from_settings(served):
    allowed = served.client.options(
        "/api/meta",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": API_KEY_HEADER,
        },
    )
    assert allowed.headers.get("access-control-allow-origin") == ORIGIN
    assert "access-control-allow-credentials" not in allowed.headers
    refused = served.client.options(
        "/api/meta",
        headers={
            "Origin": "http://other.invalid",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in refused.headers
    simple = served.get("/api/models", **{})
    assert "access-control-allow-credentials" not in simple.headers


def test_responses_never_contain_tokens_or_keys(served):
    texts = []
    turn = served.chat("Hello again", speaker="Alice", token=served.token)
    assert turn.status_code == 200
    # Authenticated: the plain name; a bare claim is shown bracketed.
    assert turn.json()["speaker_label"] == "Alice"
    claim = served.chat("Hello again", speaker="Alice")
    assert claim.json()["speaker_label"] == "[Alice]"
    texts.append(turn.text)
    texts.append(json.dumps(dict(turn.headers)))
    for path in (
        "/health",
        "/api/meta",
        "/api/consciousness/status",
        "/api/consciousness/state",
        "/api/consciousness/assessment",
        "/api/consciousness/snapshots",
        "/api/knowledge/concepts",
        "/api/ctm/status",
        "/api/models",
    ):
        response = served.get(path)
        assert response.status_code == 200, path
        texts.append(response.text)
        texts.append(json.dumps(dict(response.headers)))
    # Refusals and validation errors never echo what was sent.
    refused = served.client.get(
        "/api/meta", headers={API_KEY_HEADER: secrets.token_hex(24)}
    )
    texts.append(refused.text)
    invalid = served.post(
        "/api/chat",
        {"message": "x", "speaker_token": served.token, "api_key": served.key},
    )
    assert invalid.status_code == 422
    texts.append(invalid.text)
    blob = "\n".join(texts)
    for secret in (served.key, served.token):
        assert secret not in blob
        pieces = [secret[i : i + 8] for i in range(len(secret) - 7)]
        assert not [piece for piece in pieces if piece in blob]
