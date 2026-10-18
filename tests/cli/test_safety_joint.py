"""``mtc safety status`` and ``mtc safety resume`` against the real application.

The commands are built on the fake server of ``test_http_client``; this module
routes the same requests, in process, into the application ``create_app``
builds (over a small stand-in runtime), so the request bodies, headers, status
codes and payload shapes the commands rely on are the ones the server
validates and sends. No socket is opened: the client's request method is
replaced by a call into the application's test client.
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from mtc.cli import main as cli
from mtc.cli.http import ApiClient
from mtc.core.config import Settings
from mtc.core.envfile import read_env
from mtc.pipeline.speaker_trust import token_for
from tests.cli.test_http_client import project, setup_project  # noqa: F401
from tests.cli.test_http_client import unused_port
from tests.onboarding.test_wizard_noninteractive import ENV_NAME

fastapi = pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="the api extra is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from mtc.api.server import create_app  # noqa: E402


class StubTrust:
    """Trusts exactly one name presenting exactly one token."""

    def __init__(self, name: str, token: str) -> None:
        self._name = name
        self._token = token
        self.seen: List[Tuple[str, Optional[str]]] = []

    def verify(self, name: str, token: Optional[str] = None) -> Any:
        self.seen.append((name, token))
        trusted = name == self._name and token == self._token
        return SimpleNamespace(trusted=trusted)

    def status(self) -> Dict[str, Any]:
        return {"speakers": 1}


class StubSafety:
    """A safety stack with one suspension to lift."""

    def __init__(self) -> None:
        self.suspended = True
        self.roles: List[str] = []

    async def resume_conversations(self, role: str) -> Dict[str, bool]:
        self.roles.append(role)
        resumed = self.suspended
        self.suspended = False
        return {"resumed": resumed, "suspended": self.suspended}

    def status(self) -> Dict[str, Any]:
        return {
            "available": True,
            "last_error": None,
            "safety_monitor": {
                "safety_mode": "standard",
                "suspended": self.suspended,
            },
        }


class StubRuntime:
    def __init__(self, trust: StubTrust, safety: StubSafety) -> None:
        self.speaker_trust = trust
        self.safety = safety

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def status(self) -> Dict[str, Any]:
        return {}


def test_safety_commands_round_trip_through_the_real_application(
    project, tmp_path, monkeypatch, capsys  # noqa: F811
):
    env = setup_project(project, tmp_path, unused_port())
    cfg = Settings(_env_file=str(project / ENV_NAME), PROJECT_ROOT=str(project))
    creator_token = token_for("Alice", cfg)
    assert isinstance(creator_token, str)
    key = read_env(project / ENV_NAME)["MTC_API_KEY"]
    assert key == env["MTC_API_KEY"]
    trust = StubTrust("Alice", creator_token)
    safety = StubSafety()
    served = Settings(
        _env_file=None,
        PROJECT_ROOT=str(project),
        DATA_DIRECTORY=str(tmp_path / "data"),
        LOG_DIRECTORY=str(tmp_path / "logs"),
        STATE_DIR=str(tmp_path / "state"),
        RESULTS_DIR=str(tmp_path / "results"),
        AGENT_SPACE_DIR=str(tmp_path / "agent_space"),
        LLM_PROVIDER="mock",
        DEVICE="cpu",
        API_KEY=key,
        METRICS_UPDATE_INTERVAL=0,
        SLEEP_SCHEDULER_ENABLED=False,
    )
    app = create_app(settings=served, runtime=StubRuntime(trust, safety))
    with TestClient(app) as client:

        def routed(
            self, method, path, body=None, *, extra_headers=None, timeout_s=None
        ):
            headers = {"X-API-Key": self._api_key, **(extra_headers or {})}
            response = client.request(method, path, json=body, headers=headers)
            payload = response.json()
            return response.status_code, payload if isinstance(payload, dict) else {}

        monkeypatch.setattr(ApiClient, "_request", routed)
        base = ["--project-dir", str(project), "safety"]
        streams: List[str] = []

        def run(*arguments):
            code = cli.main(base + list(arguments))
            captured = capsys.readouterr()
            streams.extend([captured.out, captured.err])
            return code, captured.out, captured.err

        code, out, _err = run("status")
        assert code == 0
        assert "safety stack: available" in out
        assert "conversations suspended: yes" in out and "mtc safety resume" in out

        # A name the server does not verify as trusted lifts nothing.
        code, _out, err = run("resume", "--speaker", "Guest")
        assert code == 1
        assert "needs a trusted speaker" in err
        assert safety.suspended is True and safety.roles == []

        # The creator, with the creator's token, does.
        code, out, _err = run("resume")
        assert code == 0
        assert "conversations resumed" in out
        assert safety.suspended is False
        assert safety.roles == [served.SAFETY_ESCALATION_ROLE]
        assert trust.seen[-1] == ("Alice", creator_token)

        # A second call finds nothing to lift, and says so.
        code, out, _err = run("resume")
        assert code == 0 and "were not suspended" in out
        code, out, _err = run("status")
        assert code == 0 and "conversations suspended: no" in out

        # No secret reached a standard stream.
        shown = "\n".join(streams)
        assert creator_token not in shown and key not in shown
