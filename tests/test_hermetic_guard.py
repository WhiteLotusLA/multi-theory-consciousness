"""The hermetic network guard: default service ports are refused in process.

No test here reaches a server. Every attempt at a default service port is
refused before it leaves the process, and the one real connection goes to a
listener the test owns on an ephemeral loopback port.
"""

import asyncio
import errno
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from mtc.core.config import Settings
from tests import conftest as fixtures

ROOT = Path(__file__).resolve().parents[1]
LOOPBACK = "127.0.0.1"
GUARDED_FIELDS = ("API_PORT", "LLM_PORT", "POSTGRES_PORT", "REDIS_PORT")


def _default(name: str) -> int:
    return int(Settings.model_fields[name].default)


@pytest.fixture
def guarded(monkeypatch, hermetic_network):
    """The test's record, with the store opt-in off and the guard checked."""
    monkeypatch.delenv(fixtures.OPT_IN_VARIABLE, raising=False)
    assert fixtures.hermetic_guard_installed()
    return hermetic_network


def test_api_and_model_ports_are_guarded_with_or_without_the_opt_in(monkeypatch):
    monkeypatch.delenv(fixtures.OPT_IN_VARIABLE, raising=False)
    without = fixtures.guarded_ports()
    monkeypatch.setenv(fixtures.OPT_IN_VARIABLE, "1")
    with_opt_in = fixtures.guarded_ports()
    for name in fixtures.ALWAYS_GUARDED_PORT_FIELDS:
        assert _default(name) in without
        assert _default(name) in with_opt_in


def test_store_ports_are_guarded_without_the_opt_in(monkeypatch):
    monkeypatch.delenv(fixtures.OPT_IN_VARIABLE, raising=False)
    ports = fixtures.guarded_ports()
    for name in fixtures.STORE_PORT_FIELDS:
        assert _default(name) in ports
    for name in fixtures.STORE_URI_FIELDS:
        assert urlsplit(str(Settings.model_fields[name].default)).port in ports


def test_store_ports_open_only_under_the_opt_in(monkeypatch):
    monkeypatch.setenv(fixtures.OPT_IN_VARIABLE, "1")
    ports = fixtures.guarded_ports()
    assert ports == {_default(name) for name in fixtures.ALWAYS_GUARDED_PORT_FIELDS}
    monkeypatch.setenv(fixtures.OPT_IN_VARIABLE, "yes")
    assert _default("POSTGRES_PORT") in fixtures.guarded_ports()


@pytest.mark.parametrize("field", GUARDED_FIELDS)
def test_connect_is_refused_and_recorded(guarded, field):
    port = _default(field)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(ConnectionRefusedError) as raised:
            sock.connect((LOOPBACK, port))
    assert fixtures.GUARD_MESSAGE in str(raised.value)
    attempts = guarded.attempts()
    assert [refused for refused, _ in attempts] == [port]
    assert "test_connect_is_refused_and_recorded" in attempts[0][1]
    guarded.forgive()
    assert guarded.attempts() == []


def test_connect_ex_reports_refused(guarded):
    port = _default("API_PORT")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        assert sock.connect_ex((LOOPBACK, port)) == errno.ECONNREFUSED
    assert [refused for refused, _ in guarded.attempts()] == [port]
    guarded.forgive()


def test_create_connection_is_refused(guarded):
    port = _default("LLM_PORT")
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection((LOOPBACK, port), timeout=1.0)
    assert [refused for refused, _ in guarded.attempts()] == [port]
    guarded.forgive()


@pytest.mark.asyncio
async def test_event_loop_connection_is_refused(guarded):
    port = _default("API_PORT")
    with pytest.raises(OSError) as raised:
        await asyncio.open_connection(LOOPBACK, port)
    assert fixtures.GUARD_MESSAGE in str(raised.value)
    assert [refused for refused, _ in guarded.attempts()] == [port]
    guarded.forgive()


def test_an_owned_ephemeral_listener_is_reachable(guarded):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK, 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert port not in fixtures.guarded_ports()
        with socket.create_connection((LOOPBACK, port), timeout=2.0):
            pass
    assert guarded.attempts() == []


def test_addresses_without_a_guarded_port_pass_through(guarded):
    assert fixtures._guarded_port("relative.sock") is None
    assert fixtures._guarded_port((LOOPBACK, "not-a-port")) is None
    assert fixtures._guarded_port((LOOPBACK,)) is None
    assert fixtures._guarded_port((LOOPBACK, 0)) is None
    assert fixtures._guarded_port(("::1", _default("API_PORT"), 0, 0)) == _default(
        "API_PORT"
    )


def test_unforgiven_attempt_yields_a_failure_message(guarded):
    port = _default("API_PORT")
    assert fixtures.hermetic_failure(guarded) is None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(ConnectionRefusedError):
            sock.connect((LOOPBACK, port))
    message = fixtures.hermetic_failure(guarded)
    assert message is not None
    assert str(port) in message and fixtures.GUARD_MESSAGE in message
    guarded.forgive()
    assert fixtures.hermetic_failure(guarded) is None


PROBE_SOURCE = """
import socket

import pytest

from mtc.core.config import Settings
from tests import conftest as fixtures


def test_probe_reaches_for_the_default_api_port():
    assert fixtures.hermetic_guard_installed()
    port = int(Settings.model_fields["API_PORT"].default)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(ConnectionRefusedError):
            sock.connect(("127.0.0.1", port))
"""


def test_a_test_that_reaches_for_a_default_port_fails_and_is_listed(tmp_path):
    probe = tmp_path / "test_probe.py"
    probe.write_text(PROBE_SOURCE, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(probe),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "1 passed, 1 error" in output, output
    assert fixtures.GUARD_MESSAGE in output
    assert "hermetic network guard" in output
    assert f"refused a connection to port {_default('API_PORT')}" in output
    assert "[forgiven]" not in output
