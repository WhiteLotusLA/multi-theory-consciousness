"""The command-line API client and the two commands built on it, against a
standard-library fake server on an ephemeral loopback port.

``fake_api`` and ``unused_port`` are shared with the sibling CLI test modules.
"""

import http.server
import json
import logging
import socket
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import pytest

from mtc.cli import http as http_module
from mtc.cli import main as cli
from mtc.cli.http import ApiClient, ApiUnreachable
from mtc.core.config import Settings
from mtc.core.envfile import read_env
from mtc.core.logger import reset_logging
from mtc.pipeline.speaker_trust import token_for
from tests.onboarding.test_wizard_noninteractive import (
    ENV_NAME,
    make_project,
    run,
    write_answers,
)

LOOPBACK = "127.0.0.1"
API_KEY = "test-api-key-not-a-real-one"


class Recorder:
    """What the fake server saw, and what it answers."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.replies: Dict[Tuple[str, str], Tuple[int, Any]] = {}
        self.port = 0

    def reply(self, method: str, path: str, status: int, body: Any) -> None:
        self.replies[(method, path)] = (status, body)

    @property
    def last(self) -> Dict[str, Any]:
        return self.requests[-1]


@contextmanager
def fake_api():
    recorder = Recorder()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _serve(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            recorder.requests.append(
                {
                    "method": method,
                    "path": self.path,
                    "headers": {
                        key.lower(): value for key, value in self.headers.items()
                    },
                    "body": json.loads(raw) if raw else None,
                }
            )
            route = self.path.split("?", 1)[0]
            status, body = recorder.replies.get(
                (method, route), (404, {"detail": "no"})
            )
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - the handler API's name
            self._serve("GET")

        def do_POST(self):  # noqa: N802 - the handler API's name
            self._serve("POST")

        def log_message(self, *args):
            return

    server = http.server.ThreadingHTTPServer((LOOPBACK, 0), Handler)
    server.daemon_threads = True
    recorder.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield recorder
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1]


def client_for(recorder: Recorder, key: str = API_KEY) -> ApiClient:
    return ApiClient(f"http://{LOOPBACK}:{recorder.port}", key, timeout_s=5.0)


TURN = {"success": True, "response": "[mock] hello back", "thinking": "weighing it"}


@pytest.fixture
def project(monkeypatch, tmp_path):
    """A project the wizard set up, pointed at no server yet."""
    target = make_project(monkeypatch, tmp_path)
    yield target
    reset_logging()


def setup_project(project_dir, tmp_path, port: int) -> Dict[str, str]:
    answers = write_answers(tmp_path, api={"port": port})
    assert run(project_dir, answers)[0] == 0
    return read_env(project_dir / ENV_NAME)


def test_chat_sends_api_key_and_speaker_token_headers():
    with fake_api() as server:
        server.reply("POST", "/api/chat", 200, TURN)
        client = client_for(server)
        status, payload = client.chat("hello there", "Alice", "token-for-alice")
        assert (status, payload) == (200, TURN)
        seen = server.last
        assert seen["method"] == "POST" and seen["path"] == "/api/chat"
        assert seen["headers"]["x-api-key"] == API_KEY
        assert seen["headers"]["x-speaker-token"] == "token-for-alice"
        assert seen["headers"]["content-type"] == "application/json"
        assert seen["body"] == {"message": "hello there", "speaker": "Alice"}

        # No key configured: no key header is invented.
        client_for(server, key="").chat("hi")
        assert "x-api-key" not in server.last["headers"]

        server.reply("GET", "/health", 200, {"status": "ok", "ready": True})
        assert client.health() == (200, {"status": "ok", "ready": True})
        assert "x-speaker-token" not in server.last["headers"]
    assert http_module.API_KEY_HEADER == "X-API-Key"
    assert http_module.SPEAKER_TOKEN_HEADER == "X-Speaker-Token"
    with pytest.raises(ValueError):
        ApiClient("ftp://example", API_KEY)
    with pytest.raises(ValueError):
        ApiClient(f"http://{LOOPBACK}:1/?key=value", API_KEY)
    assert API_KEY not in repr(ApiClient(f"http://{LOOPBACK}:1", API_KEY))


def test_header_names_match_the_server():
    auth = pytest.importorskip(
        "mtc.api.auth", reason="the api extra (fastapi) is not installed"
    )
    assert http_module.API_KEY_HEADER == auth.API_KEY_HEADER
    assert http_module.SPEAKER_TOKEN_HEADER == auth.SPEAKER_TOKEN_HEADER


def test_chat_status_codes_map_to_exit_codes(project, tmp_path, capsys):
    with fake_api() as server:
        setup_project(project, tmp_path, server.port)
        base = ["--project-dir", str(project), "chat", "--message", "hello"]
        server.reply("POST", "/api/chat", 200, TURN)
        assert cli.main(base) == 0
        out = capsys.readouterr()
        assert "[mock] hello back" in out.out
        assert "weighing it" not in out.out
        assert cli.main(base + ["--show-thinking"]) == 0
        assert "weighing it" in capsys.readouterr().out

        expected_words = {
            403: "safety gate blocked",
            502: "gave no answer",
            503: "brain is unavailable",
            504: "took too long",
            401: "refused the API key",
            500: "HTTP 500",
        }
        for status, words in expected_words.items():
            body = {"success": False, "error_code": "x", "response": None}
            server.reply("POST", "/api/chat", status, body)
            assert cli.main(base) == 1, status
            captured = capsys.readouterr()
            assert words in captured.err, status
            assert captured.out == "", status
        # A server that answers with something that is not JSON still maps.
        server.reply("POST", "/api/chat", 502, b"<html>bad gateway</html>")
        assert cli.main(base) == 1
        capsys.readouterr()
        # An unconfigured server key is told apart from a model that is down.
        server.reply(
            "POST",
            "/api/chat",
            503,
            {"detail": "server authentication is not configured"},
        )
        assert cli.main(base) == 1
        assert "no API key configured" in capsys.readouterr().err

    # Nothing listens any more: a refused connection is exit 3.
    assert cli.main(base) == 3
    assert "mtc serve" in capsys.readouterr().err
    with pytest.raises(ApiUnreachable):
        ApiClient(f"http://{LOOPBACK}:{unused_port()}", API_KEY, timeout_s=2).health()
    assert cli.main(base[:3] + ["--message", "  "]) == 2


def test_chat_token_from_speaker_trust_for_the_creator(project, tmp_path, capsys):
    with fake_api() as server:
        env = setup_project(project, tmp_path, server.port)
        server.reply("POST", "/api/chat", 200, TURN)
        assert cli.main(["--project-dir", str(project), "chat", "--message", "hi"]) == 0
        seen = server.last
        cfg = Settings(_env_file=str(project / ENV_NAME), PROJECT_ROOT=str(project))
        expected = token_for("Alice", cfg)
        assert isinstance(expected, str) and len(expected) == 64
        # The creator is the default speaker, with the creator's own token.
        assert seen["body"] == {"message": "hi", "speaker": "Alice"}
        assert (seen["headers"]["x-speaker-token"] == expected) is True
        assert (seen["headers"]["x-api-key"] == env["MTC_API_KEY"]) is True
        output = capsys.readouterr()
        for secret_value in (expected, env["MTC_API_KEY"]):
            assert (secret_value in output.out + output.err) is False

        # Another name gets no token: the server will treat it as unverified.
        assert (
            cli.main(
                [
                    "--project-dir",
                    str(project),
                    "chat",
                    "--speaker",
                    "Guest",
                    "--message",
                    "hi",
                ]
            )
            == 0
        )
        assert server.last["body"] == {"message": "hi", "speaker": "Guest"}
        assert "x-speaker-token" not in server.last["headers"]
        assert "unverified" in capsys.readouterr().err


def test_anonymous_sends_no_token(project, tmp_path, capsys):
    with fake_api() as server:
        setup_project(project, tmp_path, server.port)
        server.reply("POST", "/api/chat", 200, TURN)
        arguments = ["--project-dir", str(project), "chat", "--anonymous"]
        assert cli.main(arguments + ["--message", "hi"]) == 0
        seen = server.last
        assert seen["body"] == {"message": "hi"}
        assert "x-speaker-token" not in seen["headers"]
        assert "x-api-key" in seen["headers"]
        # A token without a name is never sent either.
        client_for(server).chat("hi", None, "stray-token")
        assert "x-speaker-token" not in server.last["headers"]
        assert server.last["body"] == {"message": "hi"}
        assert cli.main(arguments + ["--speaker", "Alice", "--message", "hi"]) == 2
    capsys.readouterr()


def test_chat_repl_ends_on_quit_or_end_of_input(project, tmp_path, monkeypatch, capsys):
    with fake_api() as server:
        setup_project(project, tmp_path, server.port)
        server.reply("POST", "/api/chat", 200, TURN)
        typed = iter(["first", "", "second", "/quit", "never sent"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))
        assert cli.main(["--project-dir", str(project), "chat"]) == 0
        sent = [r["body"]["message"] for r in server.requests]
        assert sent == ["first", "second"]

        def ended(prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", ended)
        assert cli.main(["--project-dir", str(project), "chat"]) == 0
        assert len(server.requests) == 2
    assert capsys.readouterr().out.count("[mock] hello back") == 2


def test_ctm_commands_round_trip(project, tmp_path, capsys):
    status_body = {
        "available": True,
        "running": True,
        "quiet_reason": "no_embedder",
        "status": {"consciousness_state": "awake", "current_frequency": 0.05},
    }
    with fake_api() as server:
        setup_project(project, tmp_path, server.port)
        base = ["--project-dir", str(project), "ctm"]
        server.reply("POST", "/api/ctm/start", 200, status_body)
        server.reply("POST", "/api/ctm/stop", 200, {**status_body, "running": False})
        server.reply("GET", "/api/ctm/status", 200, status_body)

        assert cli.main(base + ["start", "--state", "drowsy"]) == 0
        assert server.last["path"] == "/api/ctm/start"
        assert server.last["body"] == {"state": "drowsy"}
        out = capsys.readouterr().out
        assert "running: yes" in out and "state: awake" in out
        assert "frequency: 0.05" in out and "quiet reason: no_embedder" in out

        assert cli.main(base + ["start"]) == 0
        assert server.last["body"] == {}
        assert cli.main(base + ["stop"]) == 0
        assert server.last["method"] == "POST"
        assert server.last["path"] == "/api/ctm/stop"
        assert "running: no" in capsys.readouterr().out
        assert cli.main(base + ["status"]) == 0
        assert server.last["method"] == "GET"
        assert "x-speaker-token" not in server.last["headers"]
        capsys.readouterr()

        # Every status the server may give is reported, none as success.
        server.reply(
            "POST", "/api/ctm/start", 503, {"detail": "continuous thought is off"}
        )
        assert cli.main(base + ["start"]) == 1
        assert "continuous thought is off" in capsys.readouterr().err
        server.reply("GET", "/api/ctm/status", 401, {"detail": "invalid API key"})
        assert cli.main(base + ["status"]) == 1
        server.reply("GET", "/api/ctm/status", 200, {"available": False})
        assert cli.main(base + ["status"]) == 0
        assert "not built" in capsys.readouterr().out
        assert cli.main(base + ["start", "--state", "hyper"]) == 2
        assert cli.main(base) == 2
    assert cli.main(base + ["status"]) == 3
    capsys.readouterr()


def test_headers_never_logged(project, tmp_path, capsys):
    records: List[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.DEBUG)
    target = logging.getLogger("mtc")
    previous = target.level
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        with fake_api() as server:
            env = setup_project(project, tmp_path, server.port)
            server.reply("POST", "/api/chat", 200, TURN)
            server.reply("GET", "/api/ctm/status", 500, {"detail": "boom"})
            base = ["--project-dir", str(project)]
            assert cli.main(base + ["chat", "--message", "a private sentence"]) == 0
            assert cli.main(base + ["ctm", "status"]) == 1
            token = server.requests[0]["headers"]["x-speaker-token"]
        assert cli.main(base + ["chat", "--message", "a private sentence"]) == 3
    finally:
        target.removeHandler(handler)
        target.setLevel(previous)
    logged = "\n".join(records)
    shown = capsys.readouterr()
    assert any("answered" in line for line in records)
    for secret_value in (env["MTC_API_KEY"], token):
        assert (secret_value in logged) is False
        assert (secret_value in shown.out + shown.err) is False
    assert "a private sentence" not in logged
    assert "x-api-key" not in logged.lower()
    assert "x-speaker-token" not in logged.lower()


def test_no_query_string_carries_text(project, tmp_path, capsys):
    with fake_api() as server:
        setup_project(project, tmp_path, server.port)
        server.reply("POST", "/api/chat", 200, TURN)
        server.reply("POST", "/api/ctm/start", 200, {"available": True})
        server.reply("POST", "/api/ctm/stop", 200, {"available": True})
        server.reply("GET", "/api/ctm/status", 200, {"available": True})
        server.reply("GET", "/health", 200, {"status": "ok"})
        base = ["--project-dir", str(project)]
        cli.main(base + ["chat", "--message", "where is it"])
        cli.main(base + ["chat", "--anonymous", "--message", "where is it"])
        cli.main(base + ["ctm", "start", "--state", "awake"])
        cli.main(base + ["ctm", "stop"])
        cli.main(base + ["ctm", "status"])
        client_for(server).health()
        assert len(server.requests) == 6
        for seen in server.requests:
            assert "?" not in seen["path"] and "#" not in seen["path"]
            assert "where" not in seen["path"] and "Alice" not in seen["path"]
            assert "awake" not in seen["path"]
        paths = sorted({seen["path"] for seen in server.requests})
        assert paths == [
            "/api/chat",
            "/api/ctm/start",
            "/api/ctm/status",
            "/api/ctm/stop",
            "/health",
        ]
        with pytest.raises(ValueError):
            client_for(server)._request("GET", "/api/chat?message=hello")
    capsys.readouterr()


def test_from_settings_reaches_a_wildcard_bind_over_loopback(project):
    cfg = Settings(_env_file=None, API_HOST="0.0.0.0", API_PORT=18080, API_KEY="k")
    client = ApiClient.from_settings(cfg)
    assert client.base_url == f"http://{LOOPBACK}:18080"
    assert client.chat_timeout_s > cfg.GENERATION_TIMEOUT_S
    named = Settings(_env_file=None, API_HOST="example.internal", API_PORT=18081)
    assert ApiClient.from_settings(named).where == "example.internal:18081"
