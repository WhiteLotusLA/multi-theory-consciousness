"""Smoke test: the whole server over the mock provider and in-memory stores,
three turns, with every outbound connection refused."""

import os
import secrets
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mtc.api.auth import API_KEY_HEADER, SPEAKER_TOKEN_HEADER
from mtc.api.schemas import INTEGRATION_PROXY_LABEL
from mtc.api.server import create_app
from mtc.core import config as config_module
from mtc.core.embeddings import HashEmbedder
from mtc.core.logger import reset_logging
from mtc.pipeline.runtime import build_runtime
from tests.api.test_auth import api_settings

ROOT = Path(__file__).resolve().parents[2]
TOKEN_ENV = "MTC_TEST_SMOKE_TOKEN_OPERATOR"

pytestmark = pytest.mark.timeout(900)


class _NoNetwork:
    """Refuses every socket connect while installed."""

    def __init__(self) -> None:
        self.attempts = []

    def install(self, monkeypatch) -> None:
        def refuse(sock, address, *args, **kwargs):
            self.attempts.append(repr(address))
            raise ConnectionRefusedError("network access is refused in this test")

        monkeypatch.setattr(socket.socket, "connect", refuse)
        monkeypatch.setattr(socket.socket, "connect_ex", refuse)
        monkeypatch.setattr(
            socket,
            "create_connection",
            lambda address, *a, **k: refuse(None, address),
        )


def test_smoke_meta_chat_state_status_ctm_models(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("MTC_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "_settings", None)
    guard = _NoNetwork()
    guard.install(monkeypatch)

    key = secrets.token_hex(24)
    token = secrets.token_hex(24)
    monkeypatch.setenv(TOKEN_ENV, token)
    speakers = tmp_path / "speakers.yaml"
    speakers.write_text(
        "speakers:\n"
        "  - name: Operator\n"
        "    role: companion\n"
        "    trust: 0.7\n"
        f"    token_env: {TOKEN_ENV}\n",
        encoding="utf-8",
    )
    cfg = api_settings(
        tmp_path,
        API_KEY=key,
        SPEAKERS_PATH=str(speakers),
        METRICS_UPDATE_INTERVAL=0,
        CTM_CONFIG_DIR=str(ROOT / "config" / "ctm"),
    )

    async def builder(settings):
        # substrates=False: no full-size emotion bridge or consolidator
        return await build_runtime(
            settings,
            embedder=HashEmbedder(),
            allow_non_semantic=True,
            substrates=False,
        )

    headers = {API_KEY_HEADER: key}
    try:
        with TestClient(create_app(settings=cfg, runtime_builder=builder)) as client:
            assert client.get("/health").json()["status"] == "ok"
            meta = client.get("/api/meta", headers=headers).json()
            assert len(meta["indicators"]) == 30
            turns = [
                ({"message": "Hello there.", "speaker": "Alice"}, {}),
                (
                    {"message": "It is the operator.", "speaker": "Operator"},
                    {SPEAKER_TOKEN_HEADER: token},
                ),
                ({"message": "What do you notice right now?"}, {}),
            ]
            for body, extra in turns:
                response = client.post(
                    "/api/chat", json=body, headers={**headers, **extra}
                )
                assert response.status_code == 200, response.text
                assert response.json()["response"].startswith("[mock]")
            state = client.get("/api/consciousness/state", headers=headers).json()
            assert state["available"] is True
            assert state["integration_proxy"]["label"] == INTEGRATION_PROXY_LABEL
            status = client.get("/api/consciousness/status", headers=headers).json()
            assert status["pipeline"]["turns"] == 3
            assert status["orchestrator"]["cycles"] >= 3
            ctm = client.get("/api/ctm/status", headers=headers).json()
            assert ctm["available"] is True and "quiet_reason" in ctm
            models = client.get("/api/models", headers=headers).json()
            assert models["active"] == "mock"
    finally:
        reset_logging()
    assert guard.attempts == [], "no outbound connection was attempted"
