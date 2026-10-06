"""start_stack.sh and stop_stack.sh against stub commands and dummy processes.

No real service manager, store tool or server is touched: every tool is a
stub on a temporary PATH, every port is an unused ephemeral loopback port and
the only processes signalled are dummies these tests start.
"""

import socket
import sys
import time

import pytest

from .conftest import (
    LOOPBACK,
    Ops,
    pid_alive,
    stop_dummy,
    wait_gone,
    write_fake_service,
)

pytestmark = pytest.mark.slow

COMPOSE = ["docker", "compose", "--env-file", ".env", "-f", "docker/docker-compose.yml"]

# A health stub that answers once the service manager stub has "started" the
# service: the manager stub leaves a flag file, the health stub looks for it.
UP_WHEN = 'if [ -f "${STATE}/%s" ]; then exit 0; fi; exit 1'


def flagging(flag_expression: str) -> str:
    """A manager stub body: flag the service named by its last argument."""
    return f'for last in "$@"; do :; done; : >"${{STATE}}/{flag_expression}"'


def manager_calls(ops: Ops, name: str):
    return [call[1:] for call in ops.stubs.calls(name)]


def test_dry_run_plans_health_or_start_per_enabled_store(ops):
    ops.set(
        SERVICE_MANAGER="brew",
        POSTGRES_ENABLED=True,
        QDRANT_ENABLED=True,
        NEO4J_ENABLED=True,
        LLM_PROVIDER="local",
        LLM_BACKEND="llama_cpp",
    )
    result = ops.run("start_stack.sh", "--dry-run")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    port = ops.ports["POSTGRES_PORT"]
    assert f"postgres: would check: pg_isready -q -h {LOOPBACK} -p {port}" in out
    assert (
        "postgres: if it does not answer, would run: brew services start postgresql@17"
        in out
    )
    assert (
        f"qdrant: would check: GET http://{LOOPBACK}:{ops.ports['QDRANT_PORT']}/readyz"
        in out
    )
    assert "would run: qdrant --config-path config/qdrant.yaml" in out
    assert f"neo4j: would check: a TCP probe of the bolt port {ops.neo4j_port}" in out
    assert "would run: brew services start neo4j" in out
    assert "start_llm.sh --background" in out and "up to 420 seconds" in out
    assert "would run: mtc serve" in out and "up to 150 seconds" in out
    assert "Plan (nothing was started):" in out
    for store in ("mongodb", "redis"):
        assert f"{store}: disabled" in out
    # The order is PostgreSQL, Qdrant, MongoDB, Redis, Neo4j, brain, API.
    positions = [
        out.index(f"  {name}:")
        for name in ("postgres", "qdrant", "mongodb", "redis", "neo4j", "brain", "api")
    ]
    assert positions == sorted(positions)
    # Nothing ran: not a check, not a start.
    assert ops.stubs.calls() == []
    assert not (ops.project / "data").exists()

    # The waits a dry run announces follow the options.
    custom = ops.run(
        "start_stack.sh", "--dry-run", "--brain-wait", "7", "--api-wait", "9"
    )
    assert "up to 7 seconds" in custom.stdout and "up to 9 seconds" in custom.stdout
    assert ops.run("start_stack.sh", "--brain-wait", "soon").returncode == 2


def test_running_store_is_not_restarted(ops):
    ops.set(SERVICE_MANAGER="brew", POSTGRES_ENABLED=True, MONGODB_ENABLED=True)
    ops.stubs.add("pg_isready")
    ops.stubs.add("mongosh")
    ops.stubs.add("brew")
    result = ops.run("start_stack.sh", "--no-brain", "--no-api")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "postgres: healthy (already running)" in result.stdout
    assert "mongodb: healthy (already running)" in result.stdout
    assert manager_calls(ops, "brew") == []
    port = str(ops.ports["POSTGRES_PORT"])
    assert ops.stubs.calls("pg_isready") == [
        ["pg_isready", "-q", "-h", LOOPBACK, "-p", port]
    ]
    mongo_call = ops.stubs.calls("mongosh")[0]
    assert mongo_call[1:6] == [
        "--quiet",
        "--host",
        LOOPBACK,
        "--port",
        str(ops.ports["MONGODB_PORT"]),
    ]


def test_brew_manager_commands(ops):
    ops.set(
        SERVICE_MANAGER="brew",
        POSTGRES_ENABLED=True,
        MONGODB_ENABLED=True,
        REDIS_ENABLED=True,
        QDRANT_ENABLED=True,
    )
    ops.stubs.add("brew", flagging("${last}"))
    ops.stubs.add("pg_isready", UP_WHEN % "postgresql@17")
    ops.stubs.add("mongosh", UP_WHEN % "mongodb-community")
    ops.stubs.add(
        "redis-cli", 'if [ -f "${STATE}/redis" ]; then echo PONG; exit 0; fi; exit 1'
    )
    # Qdrant has no formula: its release binary is started in the background.
    ops.stubs.add("qdrant", ': >"${STATE}/qdrant-binary"; sleep 30')
    ops.stubs.add("curl", UP_WHEN % "qdrant-binary")
    result = ops.run("start_stack.sh", "--no-brain", "--no-api", "--store-wait", "10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert manager_calls(ops, "brew") == [
        ["services", "start", "postgresql@17"],
        ["services", "start", "mongodb-community"],
        ["services", "start", "redis"],
    ]
    assert manager_calls(ops, "qdrant") == [["--config-path", "config/qdrant.yaml"]]
    for store in ("postgres", "qdrant", "mongodb", "redis"):
        assert f"{store}: started" in result.stdout
    pid = ops.adopt_pid_file("qdrant.pid")
    assert pid is not None and pid_alive(pid)
    assert (ops.project / "logs" / "qdrant.log").is_file()

    # A manager that fails leaves the store down, and the run says so.
    ops.stubs.add("brew", exit_code=1)
    ops.stubs.add("pg_isready", exit_code=1)
    failed = ops.run("start_stack.sh", "--no-brain", "--no-api", "--store-wait", "1")
    assert failed.returncode == 1
    assert "postgres: down" in failed.stdout


def test_docker_manager_commands(ops):
    ops.set(
        SERVICE_MANAGER="docker",
        POSTGRES_ENABLED=True,
        MONGODB_ENABLED=True,
        QDRANT_ENABLED=True,
    )
    ops.stubs.add("docker", flagging("${last}"))
    ops.stubs.add("pg_isready", UP_WHEN % "postgres")
    ops.stubs.add("mongosh", UP_WHEN % "mongo")
    ops.stubs.add("curl", UP_WHEN % "qdrant")
    result = ops.run("start_stack.sh", "--no-brain", "--no-api", "--store-wait", "10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert ops.stubs.calls("docker") == [
        COMPOSE + ["up", "-d", "postgres"],
        COMPOSE + ["up", "-d", "qdrant"],
        COMPOSE + ["up", "-d", "mongo"],
    ]
    # The stores are started one service at a time and volumes are untouched.
    flat = " ".join(" ".join(call) for call in ops.stubs.calls("docker"))
    assert " down" not in flat and "-v" not in flat.split()


def test_systemd_manager_commands(ops):
    ops.set(
        SERVICE_MANAGER="systemd",
        POSTGRES_ENABLED=True,
        MONGODB_ENABLED=True,
        REDIS_ENABLED=True,
        QDRANT_ENABLED=True,
    )
    ops.stubs.add("systemctl", flagging("${last}"))
    ops.stubs.add("pg_isready", UP_WHEN % "postgresql")
    ops.stubs.add("mongosh", UP_WHEN % "mongod")
    ops.stubs.add("curl", UP_WHEN % "qdrant")
    ops.stubs.add(
        "redis-cli",
        'if [ -f "${STATE}/redis-server" ]; then echo PONG; exit 0; fi; exit 1',
    )
    result = ops.run("start_stack.sh", "--no-brain", "--no-api", "--store-wait", "10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert manager_calls(ops, "systemctl") == [
        ["start", "postgresql"],
        ["start", "qdrant"],
        ["start", "mongod"],
        ["start", "redis-server"],
    ]


def test_none_manager_prints_command_and_fails(ops):
    ops.set(SERVICE_MANAGER="none", POSTGRES_ENABLED=True, REDIS_ENABLED=True)
    ops.stubs.add("pg_isready", exit_code=1)
    ops.stubs.add("redis-cli", exit_code=1)
    started = time.monotonic()
    result = ops.run("start_stack.sh", "--no-brain", "--no-api")
    # Nothing to wait for: the stores were never started.
    assert time.monotonic() - started < 30
    assert result.returncode == 1
    assert "postgres: SERVICE_MANAGER is none, so start it yourself" in result.stdout
    assert "brew services start postgresql@17" in result.stdout
    assert "postgres: down" in result.stdout and "redis: down" in result.stdout
    for manager in ("brew", "docker", "systemctl"):
        assert manager_calls(ops, manager) == []


def test_redis_and_neo4j_steps_present(ops):
    ops.set(SERVICE_MANAGER="brew", REDIS_ENABLED=True, NEO4J_ENABLED=True)
    ops.stubs.add("brew")
    # Redis that asks for a password is up; so is one that answers PONG.
    ops.stubs.add("redis-cli", 'echo "(error) NOAUTH Authentication required."')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bolt:
        bolt.bind((LOOPBACK, ops.neo4j_port))
        bolt.listen(8)
        result = ops.run("start_stack.sh", "--no-brain", "--no-api")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "redis: healthy (already running)" in result.stdout
    # Neo4j is checked by its bolt port only, so it is never called healthy.
    assert (
        "neo4j: port open, already running (Neo4j is checked by its bolt port "
        "only, so health was not measured)" in result.stdout
    )
    assert manager_calls(ops, "brew") == []
    redis_call = ops.stubs.calls("redis-cli")[0]
    assert redis_call == [
        "redis-cli",
        "-h",
        LOOPBACK,
        "-p",
        str(ops.ports["REDIS_PORT"]),
        "ping",
    ]

    # Down, they are started through the manager like the other stores.
    ops.stubs.add("redis-cli", exit_code=1)
    down = ops.run("start_stack.sh", "--no-brain", "--no-api", "--store-wait", "1")
    assert down.returncode == 1
    assert manager_calls(ops, "brew") == [
        ["services", "start", "redis"],
        ["services", "start", "neo4j"],
    ]
    assert "redis: down (not healthy after 1 seconds)" in down.stdout
    assert "neo4j: down (not healthy after 1 seconds)" in down.stdout


def _local_brain(ops: Ops, ready_after: int) -> None:
    """A llama-server stub that is a stand-in HTTP server the test owns."""
    service = write_fake_service(ops.base)
    model = ops.project / "models" / "model.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"")
    ops.set(
        LLM_PROVIDER="local",
        LLM_BACKEND="llama_cpp",
        MODEL_PATH="models/model.gguf",
        LLM_BASE_URL="",
    )
    ops.stubs.add(
        "llama-server",
        f"""
        case "$1" in --help) echo "usage: llama-server"; exit 0 ;; esac
        exec "{sys.executable}" "{service}" "$@" --ready-after {ready_after}
        """,
    )
    # The probes go to this test's own ephemeral port, so the real client
    # may be used.
    ops.stubs.remove("curl")


def test_brain_wait_is_bounded(ops):
    # A model server that answers on a later poll: the wait ends as soon as
    # it does.
    _local_brain(ops, ready_after=1)
    started = time.monotonic()
    result = ops.run("start_stack.sh", "--no-api", "--brain-wait", "40")
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stdout + result.stderr
    assert "brain: started" in result.stdout
    assert elapsed < 35
    launched = ops.stubs.calls("llama-server")
    assert ["llama-server", "--help"] in launched
    server_call = [call for call in launched if call[1:2] == ["-m"]]
    assert len(server_call) == 1
    assert str(ops.ports["LLM_PORT"]) in server_call[0]
    pid = ops.adopt_pid_file("llm.pid")
    assert pid is not None and pid_alive(pid)
    assert (ops.project / "logs" / "llm.log").is_file()
    # Running again finds it healthy and starts nothing more.
    again = ops.run("start_stack.sh", "--no-api")
    assert again.returncode == 0
    assert "brain: healthy (already running)" in again.stdout
    assert len([c for c in ops.stubs.calls("llama-server") if c[1:2] == ["-m"]]) == 1


def test_brain_wait_gives_up_at_its_limit(ops):
    # A model server that never becomes ready: the wait stops at its limit.
    _local_brain(ops, ready_after=100000)
    started = time.monotonic()
    result = ops.run("start_stack.sh", "--no-api", "--brain-wait", "2")
    elapsed = time.monotonic() - started
    assert result.returncode == 1
    assert "brain: down (no answer after 2 seconds" in result.stdout
    assert elapsed < 25

    # With no backend named there is nothing to start, and it says so.
    ops.cleanup()
    ops.set(LLM_BACKEND="")
    idle = ops.run("start_stack.sh", "--no-api", "--brain-wait", "2")
    assert idle.returncode == 1
    assert "brain: down (LLM_BACKEND is empty" in idle.stdout
    # Another provider needs no model server.
    ops.set(LLM_PROVIDER="mock")
    mock = ops.run("start_stack.sh", "--no-api")
    assert mock.returncode == 0
    assert "brain: LLM_PROVIDER=mock, nothing to start" in mock.stdout


def _fake_api(ops: Ops, ready_after: int) -> None:
    service = write_fake_service(ops.base)
    port = ops.ports["API_PORT"]
    ops.stubs.add(
        "mtc",
        f'exec "{sys.executable}" "{service}" {port} --ready-after {ready_after}',
    )
    ops.stubs.remove("curl")


def test_api_health_polling_bounded(ops):
    _fake_api(ops, ready_after=2)
    started = time.monotonic()
    result = ops.run("start_stack.sh", "--no-brain", "--api-wait", "40")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "api: started" in result.stdout
    assert time.monotonic() - started < 35
    assert ops.stubs.calls("mtc") == [["mtc", "serve"]]
    pid = ops.adopt_pid_file("api.pid")
    assert pid is not None and pid_alive(pid)
    assert (ops.project / "logs" / "api.log").is_file()
    # Already ready: it is not started twice.
    again = ops.run("start_stack.sh", "--no-brain")
    assert again.returncode == 0
    assert "api: healthy (already running)" in again.stdout
    assert ops.stubs.calls("mtc") == [["mtc", "serve"]]


def test_api_wait_gives_up_at_its_limit(ops):
    _fake_api(ops, ready_after=100000)
    started = time.monotonic()
    result = ops.run("start_stack.sh", "--no-brain", "--api-wait", "2")
    assert result.returncode == 1
    assert "api: down (not ready after 2 seconds" in result.stdout
    assert time.monotonic() - started < 25
    # The port is now held by something that never becomes ready: the next
    # run reports the conflict and starts nothing on top of it.
    conflict = ops.run("start_stack.sh", "--no-brain", "--api-wait", "1")
    assert conflict.returncode == 1
    assert "is in use by something that is not a ready API" in conflict.stdout
    assert ops.stubs.calls("mtc") == [["mtc", "serve"]]


POLITE = """
trap 'echo stopped >"%s"; exit 0' TERM
while :; do sleep 0.2; done
"""
STUBBORN = """
trap '' TERM
while :; do sleep 0.2; done
"""


def test_stop_sends_sigterm_and_waits_grace(ops):
    marker = ops.base / "api-got-term"
    api = ops.spawn_dummy("mtc", POLITE % marker)
    brain = ops.spawn_dummy("llama-server", POLITE % (ops.base / "brain-got-term"))
    run_dir = ops.run_dir()
    (run_dir / "api.pid").write_text(f"{api}\n")
    (run_dir / "llm.pid").write_text(f"{brain}\n")

    planned = ops.run("stop_stack.sh", "--dry-run")
    assert planned.returncode == 0
    assert f"would send the terminate signal to process {api}" in planned.stdout
    assert pid_alive(api) and pid_alive(brain)

    result = ops.run("stop_stack.sh", "--grace", "10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "api: stopped" in result.stdout and "brain: stopped" in result.stdout
    # The API went first, and each process ended by handling the signal.
    assert result.stdout.index("api:") < result.stdout.index("brain:")
    assert wait_gone(api) and wait_gone(brain)
    assert marker.read_text().strip() == "stopped"
    assert (ops.base / "brain-got-term").read_text().strip() == "stopped"
    assert not (run_dir / "api.pid").exists() and not (run_dir / "llm.pid").exists()

    # Nothing running is not an error; a stale PID file is cleared.
    (run_dir / "api.pid").write_text(f"{api}\n")
    idle = ops.run("stop_stack.sh")
    assert idle.returncode == 0
    assert "api: not running (the PID file was stale" in idle.stdout
    assert "brain: not running (no PID file)" in idle.stdout
    assert not (run_dir / "api.pid").exists()


def test_stop_never_forces_and_reports_survivor(ops):
    survivor = ops.spawn_dummy("mtc", STUBBORN)
    run_dir = ops.run_dir()
    (run_dir / "api.pid").write_text(f"{survivor}\n")
    time.sleep(0.3)
    started = time.monotonic()
    result = ops.run("stop_stack.sh", "--grace", "2")
    waited = time.monotonic() - started
    assert result.returncode == 1
    assert f"process {survivor} is still running after 2 seconds" in result.stderr
    assert "It was not forced" in result.stderr
    # It waited the grace period and the process is still there, untouched.
    assert waited >= 2
    assert pid_alive(survivor)
    assert (run_dir / "api.pid").read_text().strip() == str(survivor)
    # The test owns the dummy and stops it itself.
    stop_dummy(survivor)
    assert wait_gone(survivor)

    # A PID that now belongs to another program is never signalled.
    other = ops.spawn_dummy("unrelated-tool", POLITE % (ops.base / "other-got-term"))
    (run_dir / "api.pid").write_text(f"{other}\n")
    refused = ops.run("stop_stack.sh", "--grace", "2")
    assert refused.returncode == 1
    assert "is another program now; nothing was signalled" in refused.stderr
    assert pid_alive(other)
    assert not (ops.base / "other-got-term").exists()
    (run_dir / "api.pid").write_text("not-a-number\n")
    garbled = ops.run("stop_stack.sh")
    assert garbled.returncode == 1
    assert "does not hold a process number" in garbled.stderr


def test_stop_all_stops_stores_in_reverse_order(ops):
    ops.set(
        SERVICE_MANAGER="brew",
        POSTGRES_ENABLED=True,
        MONGODB_ENABLED=True,
        QDRANT_ENABLED=True,
        REDIS_ENABLED=True,
        NEO4J_ENABLED=True,
    )
    ops.stubs.add("brew")
    # Without --all the stores are left running.
    assert ops.run("stop_stack.sh").returncode == 0
    assert manager_calls(ops, "brew") == []

    result = ops.run("stop_stack.sh", "--all")
    assert result.returncode == 0, result.stdout + result.stderr
    assert manager_calls(ops, "brew") == [
        ["services", "stop", "neo4j"],
        ["services", "stop", "redis"],
        ["services", "stop", "mongodb-community"],
        ["services", "stop", "postgresql@17"],
    ]
    # Qdrant runs from its binary under brew: its PID file decides.
    assert "qdrant: not running (no PID file)" in result.stdout

    ops.set(SERVICE_MANAGER="docker", QDRANT_ENABLED=False, NEO4J_ENABLED=False)
    ops.stubs.add("docker")
    assert ops.run("stop_stack.sh", "--all").returncode == 0
    assert ops.stubs.calls("docker") == [
        COMPOSE + ["stop", "redis"],
        COMPOSE + ["stop", "mongo"],
        COMPOSE + ["stop", "postgres"],
    ]
    ops.set(SERVICE_MANAGER="systemd")
    ops.stubs.add("systemctl", exit_code=1)
    failed = ops.run("stop_stack.sh", "--all")
    assert failed.returncode == 1
    assert manager_calls(ops, "systemctl") == [
        ["stop", "redis-server"],
        ["stop", "mongod"],
        ["stop", "postgresql"],
    ]
    ops.set(SERVICE_MANAGER="none")
    manual = ops.run("stop_stack.sh", "--all", "--dry-run")
    assert manual.returncode == 0
    assert "postgres: SERVICE_MANAGER is none, so stop it yourself" in manual.stdout


def test_settings_that_cannot_be_read_exit_2(ops):
    for script in ("start_stack.sh", "stop_stack.sh", "start_llm.sh"):
        result = ops.run(script, MTC_WAKE_HOUR="99")
        assert result.returncode == 2, script
        assert "virtual environment" in result.stderr
    assert ops.stubs.calls() == []


def test_stop_api_signals_only_the_server_command(ops):
    """The API PID file is trusted only while the process is still the server
    (``mtc serve`` or the server module): a recycled PID whose command line
    merely contains the letters of the package name is never signalled."""
    run_dir = ops.run_dir()
    marker = ops.base / "lookalike-got-term"
    lookalike = ops.spawn_dummy("mtc_notes.txt", POLITE % marker)
    (run_dir / "api.pid").write_text(f"{lookalike}\n")
    time.sleep(0.3)
    refused = ops.run("stop_stack.sh", "--grace", "2")
    assert refused.returncode == 1
    assert "is another program now; nothing was signalled" in refused.stderr
    assert pid_alive(lookalike)
    assert not marker.exists()
    stop_dummy(lookalike)
    assert wait_gone(lookalike)

    # The server itself is still recognised and stopped.
    served = ops.base / "server-got-term"
    server = ops.spawn_dummy("mtc", POLITE % served)
    (run_dir / "api.pid").write_text(f"{server}\n")
    result = ops.run("stop_stack.sh", "--grace", "10")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "api: stopped" in result.stdout
    assert wait_gone(server)
    assert served.read_text().strip() == "stopped"


def test_bare_port_probe_is_labelled_port_open_not_healthy(ops):
    """Without pg_isready the stack's check is a TCP connect: a store whose
    port is open reads "port open" with the reason, never "healthy"."""
    ops.set(SERVICE_MANAGER="brew", POSTGRES_ENABLED=True)
    ops.stubs.add("brew")
    ops.stubs.remove("pg_isready")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as pg:
        pg.bind((LOOPBACK, ops.ports["POSTGRES_PORT"]))
        pg.listen(8)
        result = ops.run("start_stack.sh", "--no-brain", "--no-api")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "postgres: healthy" not in result.stdout
    assert (
        "postgres: port open, already running (pg_isready is not installed, "
        "so health was not measured)" in result.stdout
    )
    assert manager_calls(ops, "brew") == []
