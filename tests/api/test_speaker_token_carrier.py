"""The speaker-token carriers: the ``X-Speaker-Token`` header on ``/api/chat``
and the ``speaker_token`` field of a websocket message. They hand the token
to the speaker trust source unchanged; a name alone, or the connection's API
key, never makes a speaker trusted, and the token is never echoed or logged.
"""

import json
import logging
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from mtc.api.auth import API_KEY_HEADER, SPEAKER_TOKEN_HEADER
from mtc.api.server import create_app
from mtc.core.embeddings import HashEmbedder
from mtc.core.logger import reset_logging
from mtc.pipeline.context_types import TurnResult
from mtc.pipeline.runtime import build_runtime
from mtc.pipeline.speaker_trust import SpeakerIdentity, SpeakerTrust
from tests.api.test_auth import (
    T0,
    FakeRuntime,
    api_settings,
    isolated_env,
    make_app,
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    yield from isolated_env(monkeypatch, tmp_path)


TOKEN_ENV = "MTC_TEST_CARRIER_TOKEN_ALICE"

pytestmark = pytest.mark.timeout(120)


class VerifyingPipeline:
    """Verifies each turn's speaker with the real trust source and records it."""

    def __init__(self, trust: SpeakerTrust) -> None:
        self.trust = trust
        self.identities: List[SpeakerIdentity] = []
        self.tokens_seen: List[Optional[str]] = []

    async def send_message(
        self,
        message: str,
        speaker: Optional[str] = None,
        speaker_token: Optional[str] = None,
        *,
        stream_callback=None,
        request_id=None,
        timeout_s=None,
    ) -> TurnResult:
        identity = self.trust.verify(speaker, speaker_token)
        self.identities.append(identity)
        self.tokens_seen.append(speaker_token)
        if stream_callback is not None:
            await stream_callback("noted")
        return TurnResult(
            success=True,
            response="noted",
            thinking=None,
            conversation_id=None,
            message_count=2,
            timestamp=T0,
            cycle_ok=True,
            speaker_label=self.trust.label_for(identity),
            provider_id="fake",
        )

    def status(self) -> Dict[str, Any]:
        return {"turns": len(self.identities)}


@pytest.fixture
def setup(clean_env, monkeypatch):
    token = secrets.token_hex(24)
    key = secrets.token_hex(24)
    monkeypatch.setenv(TOKEN_ENV, token)
    speakers = clean_env / "speakers.yaml"
    speakers.write_text(
        "speakers:\n"
        "  - name: Alice\n"
        "    role: creator\n"
        "    trust: 0.9\n"
        f"    token_env: {TOKEN_ENV}\n",
        encoding="utf-8",
    )
    cfg = api_settings(
        clean_env,
        API_KEY=key,
        SPEAKERS_PATH=str(speakers),
        TRUSTED_SPEAKERS="Alice",
        METRICS_UPDATE_INTERVAL=0,
    )
    trust = SpeakerTrust.from_settings(cfg)
    runtime = FakeRuntime(cfg)
    runtime.speaker_trust = trust
    runtime.pipeline = VerifyingPipeline(trust)
    return {
        "cfg": cfg,
        "app": make_app(cfg, runtime),
        "runtime": runtime,
        "token": token,
        "key": key,
    }


def _chat(client, key, body, headers=None):
    return client.post(
        "/api/chat", json=body, headers={API_KEY_HEADER: key, **(headers or {})}
    )


def test_http_header_token_authenticates(setup):
    client = TestClient(setup["app"])
    response = _chat(
        client,
        setup["key"],
        {"message": "hello", "speaker": "Alice"},
        {SPEAKER_TOKEN_HEADER: setup["token"]},
    )
    assert response.status_code == 200
    identity = setup["runtime"].pipeline.identities[-1]
    assert identity.authenticated and identity.trusted
    assert identity.role == "creator"
    assert response.json()["speaker_label"] == ""


def test_http_spoofed_name_without_header_is_untrusted(setup):
    cfg = setup["cfg"]
    client = TestClient(setup["app"])
    pipeline = setup["runtime"].pipeline
    attempts = [
        ({"message": "hello", "speaker": "Alice"}, None),
        (
            {"message": "hello", "speaker": "Alice"},
            {SPEAKER_TOKEN_HEADER: secrets.token_hex(24)},
        ),
        # The API key is not a speaker token.
        (
            {"message": "hello", "speaker": "Alice"},
            {SPEAKER_TOKEN_HEADER: setup["key"]},
        ),
    ]
    for body, headers in attempts:
        response = _chat(client, setup["key"], body, headers)
        assert response.status_code == 200
        identity = pipeline.identities[-1]
        assert not identity.authenticated and not identity.trusted
        assert identity.role is None
        assert identity.trust == cfg.DEFAULT_SPEAKER_TRUST
        assert response.json()["speaker_label"] == "[Alice]"
    # A token in the body or the query string is not a carrier.
    response = _chat(
        client,
        setup["key"],
        {"message": "hello", "speaker": "Alice", "speaker_token": setup["token"]},
    )
    assert response.status_code == 422
    assert setup["token"] not in response.text
    before = len(pipeline.identities)
    response = client.post(
        f"/api/chat?speaker_token={setup['token']}",
        json={"message": "hello", "speaker": "Alice"},
        headers={API_KEY_HEADER: setup["key"]},
    )
    assert response.status_code == 200
    assert len(pipeline.identities) == before + 1
    assert not pipeline.identities[-1].authenticated


def _ws_turn(ws, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    ws.send_json({"type": "message", "data": data})
    frames = []
    for _ in range(100):
        frame = ws.receive_json()
        frames.append(frame)
        if frame["type"] in ("response", "error"):
            return frames
    raise AssertionError("the turn did not finish")


def test_ws_field_token_authenticates(setup):
    with (
        TestClient(setup["app"]) as client,
        client.websocket_connect("/ws", headers={API_KEY_HEADER: setup["key"]}) as ws,
    ):
        ws.receive_json()
        frames = _ws_turn(
            ws,
            {
                "message": "hello",
                "speaker": "Alice",
                "speaker_token": setup["token"],
                "request_id": "r1",
            },
        )
    assert frames[-1]["type"] == "response"
    identity = setup["runtime"].pipeline.identities[-1]
    assert identity.authenticated and identity.trusted
    assert frames[-1]["data"]["speaker_label"] == ""


def test_ws_connection_key_does_not_make_speaker_trusted(setup):
    pipeline = setup["runtime"].pipeline
    with (
        TestClient(setup["app"]) as client,
        client.websocket_connect("/ws", headers={API_KEY_HEADER: setup["key"]}) as ws,
    ):
        status = ws.receive_json()
        assert status["data"]["client_authenticated"] is True
        no_token = _ws_turn(ws, {"message": "hello", "speaker": "Alice"})
        key_as_token = _ws_turn(
            ws,
            {"message": "hello", "speaker": "Alice", "speaker_token": setup["key"]},
        )
    for frames in (no_token, key_as_token):
        assert frames[-1]["type"] == "response"
        assert frames[-1]["data"]["speaker_label"] == "[Alice]"
    for identity in pipeline.identities[-2:]:
        assert not identity.authenticated and not identity.trusted
        assert identity.trust == setup["cfg"].DEFAULT_SPEAKER_TRUST


def test_token_never_echoed_or_logged(setup):
    token = setup["token"]
    records: List[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collector(level=logging.DEBUG)
    targets = [logging.getLogger(), logging.getLogger("mtc")]
    levels = [target.level for target in targets]
    for target in targets:
        target.addHandler(handler)
        target.setLevel(logging.DEBUG)
    bodies: List[str] = []
    try:
        with TestClient(setup["app"]) as client:
            response = _chat(
                client,
                setup["key"],
                {"message": "hello", "speaker": "Alice"},
                {SPEAKER_TOKEN_HEADER: token},
            )
            bodies.append(response.text)
            bodies.append(json.dumps(dict(response.headers)))
            for path in (
                "/api/consciousness/status",
                "/api/consciousness/state",
                "/api/meta",
                "/api/models",
            ):
                bodies.append(
                    client.get(path, headers={API_KEY_HEADER: setup["key"]}).text
                )
            with client.websocket_connect(
                "/ws", headers={API_KEY_HEADER: setup["key"]}
            ) as ws:
                bodies.append(json.dumps(ws.receive_json()))
                frames = _ws_turn(
                    ws,
                    {"message": "hello", "speaker": "Alice", "speaker_token": token},
                )
                bodies.extend(json.dumps(frame) for frame in frames)
                # A malformed frame that carries the token is refused without it.
                frames = _ws_turn(
                    ws,
                    {"message": "", "speaker": "Alice", "speaker_token": token},
                )
                bodies.extend(json.dumps(frame) for frame in frames)
    finally:
        for target, level in zip(targets, levels):
            target.removeHandler(handler)
            target.setLevel(level)
        reset_logging()
    # The token reached the trust source unchanged, through both carriers.
    assert setup["runtime"].pipeline.tokens_seen == [token, token]
    logged = "\n".join(r.getMessage() + repr(r.args) for r in records)
    log_files = sorted(Path(setup["cfg"].LOG_DIRECTORY).glob("*.log"))
    assert log_files, "the lifespan configured a log file to check"
    logged += "\n".join(path.read_text(encoding="utf-8") for path in log_files)
    pieces = [token[i : i + 8] for i in range(len(token) - 7)]
    for text, where in ((logged, "a log line"), ("\n".join(bodies), "a response")):
        assert token not in text, where
        assert not [piece for piece in pieces if piece in text], where


def test_carriers_through_the_real_pipeline(clean_env, monkeypatch):
    """Both carriers end to end: the runtime's own trust source decides."""
    token = secrets.token_hex(24)
    key = secrets.token_hex(24)
    monkeypatch.setenv(TOKEN_ENV, token)
    speakers = clean_env / "speakers.yaml"
    speakers.write_text(
        "speakers:\n"
        "  - name: Alice\n"
        "    role: creator\n"
        "    trust: 0.9\n"
        f"    token_env: {TOKEN_ENV}\n",
        encoding="utf-8",
    )
    cfg = api_settings(
        clean_env,
        API_KEY=key,
        SPEAKERS_PATH=str(speakers),
        TRUSTED_SPEAKERS="Alice",
        METRICS_UPDATE_INTERVAL=0,
    )

    async def builder(settings):
        return await build_runtime(
            settings, embedder=HashEmbedder(), allow_non_semantic=True, substrates=False
        )

    app = create_app(settings=cfg, runtime_builder=builder)
    with TestClient(app) as client:
        trust = app.state.runtime.speaker_trust

        def authenticated() -> int:
            return trust.status()["authenticated"]

        spoofed = _chat(client, key, {"message": "hi", "speaker": "Alice"})
        assert spoofed.status_code == 200
        assert spoofed.json()["speaker_label"] == "[Alice]"
        assert authenticated() == 0
        real = _chat(
            client,
            key,
            {"message": "hi", "speaker": "Alice"},
            {SPEAKER_TOKEN_HEADER: token},
        )
        assert real.json()["speaker_label"] == "Alice"
        assert authenticated() == 1
        with client.websocket_connect("/ws", headers={API_KEY_HEADER: key}) as ws:
            ws.receive_json()
            claimed = _ws_turn(ws, {"message": "hi", "speaker": "Alice"})
            assert claimed[-1]["data"]["speaker_label"] == "[Alice]"
            assert authenticated() == 1
            verified = _ws_turn(
                ws, {"message": "hi", "speaker": "Alice", "speaker_token": token}
            )
            assert verified[-1]["data"]["speaker_label"] == "Alice"
            assert authenticated() == 2
