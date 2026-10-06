"""harden_services.sh and health_check.sh: they report and change nothing
(except file modes, on request). ``lsof`` and every probe are stubs."""

import re
import stat
from pathlib import Path

import pytest

from .conftest import OPS_DIR, Ops, install_store_stubs

pytestmark = pytest.mark.slow

LSOF_HEADER = "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME"
ENV_NAME = "." + "env"


def lsof_line(command: str, address: str, port: int) -> str:
    kind = "IPv6" if address.startswith("[") else "IPv4"
    return f"{command} 4242 user 7u {kind} 0x1 0t0 TCP {address}:{port} (LISTEN)"


def stub_lsof(ops: Ops, lines) -> None:
    listing = "\n".join([LSOF_HEADER, *lines])
    ops.stubs.add("lsof", f"cat <<'EOF'\n{listing}\nEOF")


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def private_paths(ops: Ops) -> dict:
    """The four private paths, created with modes other users can read."""
    paths = {
        ops.project / ENV_NAME: (0o644, "MTC_SYSTEM_NAME=Testagent\n"),
        ops.project / "config" / "speakers.yaml": (0o644, "speakers: []\n"),
        ops.project / "agent_space": (0o755, None),
        ops.project / "backups": (0o755, None),
    }
    for path, (mode, text) in paths.items():
        if text is None:
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        path.chmod(mode)
    return {path: mode for path, (mode, _text) in paths.items()}


def test_harden_flags_wildcard_listener(ops):
    pg_port = ops.ports["POSTGRES_PORT"]
    stub_lsof(
        ops,
        [
            lsof_line("postgres", "*", pg_port),
            lsof_line("redis-ser", "127.0.0.1", ops.ports["REDIS_PORT"]),
            # An unrelated program on every interface is not MTC's business.
            lsof_line("other", "*", 1),
        ],
    )
    result = ops.run("harden_services.sh")
    assert result.returncode == 1
    assert (
        f"FINDING: postgres listens on * port {pg_port}, which other machines can reach"
        in result.stdout
    )
    assert "listen_addresses = '127.0.0.1'" in result.stdout
    assert result.stdout.count("FINDING:") == 1
    assert "listener: redis on 127.0.0.1" in result.stdout
    assert ops.stubs.calls("lsof") == [["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"]]

    # A named non-loopback address is a finding too.
    # (a documentation address, assembled here so this file holds none)
    lan_address = ".".join(["192", "0", "2", "10"])
    stub_lsof(ops, [lsof_line("mongod", lan_address, ops.ports["MONGODB_PORT"])])
    lan = ops.run("harden_services.sh")
    assert lan.returncode == 1
    assert f"FINDING: mongodb listens on {lan_address}" in lan.stdout
    assert "net.bindIp: 127.0.0.1" in lan.stdout

    # Without lsof the listeners were not checked, and that is not a pass.
    ops.stubs.remove("lsof")
    blind = ops.run(
        "harden_services.sh",
        PATH=f"{ops.stubs.bin}:{Path(ops.env['PATH'].split(':')[1])}:/bin",
    )
    assert blind.returncode == 1
    assert "listeners: not checked: lsof is not installed" in blind.stderr


def test_harden_accepts_loopback_listeners(ops):
    stub_lsof(
        ops,
        [
            lsof_line("postgres", "127.0.0.1", ops.ports["POSTGRES_PORT"]),
            lsof_line("postgres", "[::1]", ops.ports["POSTGRES_PORT"]),
            lsof_line("mongod", "127.0.0.1", ops.ports["MONGODB_PORT"]),
            lsof_line("qdrant", "127.0.0.1", ops.ports["QDRANT_PORT"]),
            lsof_line("redis-ser", "[::1]", ops.ports["REDIS_PORT"]),
            lsof_line("java", "127.0.0.1", ops.neo4j_port),
            lsof_line("python3", "127.0.0.1", ops.ports["LLM_PORT"]),
            lsof_line("python3", "127.0.0.1", ops.ports["API_PORT"]),
        ],
    )
    result = ops.run("harden_services.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FINDING" not in result.stdout
    assert result.stdout.count("(loopback)") == 8
    assert "no finding" in result.stdout
    # Nothing listening at all is not a finding either.
    ops.stubs.add("lsof", exit_code=1)
    assert ops.run("harden_services.sh").returncode == 0

    planned = ops.run("harden_services.sh", "--dry-run")
    assert planned.returncode == 0
    assert "would run: lsof -nP -iTCP -sTCP:LISTEN" in planned.stdout
    assert len(ops.stubs.calls("lsof")) == 2


def test_harden_fix_permissions_modes(ops):
    stub_lsof(ops, [])
    before = private_paths(ops)

    reported = ops.run("harden_services.sh")
    assert reported.returncode == 1
    assert reported.stdout.count("FINDING:") == 4
    assert "the environment file has mode 644" in reported.stdout
    assert "--fix-permissions" in reported.stdout
    for path, mode in before.items():
        assert mode_of(path) == mode, path

    planned = ops.run("harden_services.sh", "--fix-permissions", "--dry-run")
    assert planned.returncode == 0
    assert planned.stdout.count("would run: chmod") == 4
    for path, mode in before.items():
        assert mode_of(path) == mode, path

    fixed = ops.run("harden_services.sh", "--fix-permissions")
    assert fixed.returncode == 0, fixed.stdout + fixed.stderr
    assert fixed.stdout.count("fixed:") == 4
    assert mode_of(ops.project / ENV_NAME) == 0o600
    assert mode_of(ops.project / "config" / "speakers.yaml") == 0o600
    assert mode_of(ops.project / "agent_space") == 0o700
    assert mode_of(ops.project / "backups") == 0o700
    # Already private: nothing to report, nothing changed.
    again = ops.run("harden_services.sh")
    assert again.returncode == 0
    assert "FINDING" not in again.stdout and "fixed:" not in again.stdout


def test_harden_prints_config_lines_never_edits_store_configs(ops):
    stub_lsof(
        ops,
        [
            lsof_line("postgres", "*", ops.ports["POSTGRES_PORT"]),
            lsof_line("mongod", "*", ops.ports["MONGODB_PORT"]),
            lsof_line("redis-ser", "*", ops.ports["REDIS_PORT"]),
            lsof_line("qdrant", "*", ops.ports["QDRANT_PORT"]),
            lsof_line("java", "*", ops.neo4j_port),
            lsof_line("python3", "*", ops.ports["LLM_PORT"]),
            lsof_line("python3", "*", ops.ports["API_PORT"]),
        ],
    )
    result = ops.run("harden_services.sh", "--fix-permissions")
    assert result.returncode == 1
    assert result.stdout.count("FINDING:") == 7
    for line in (
        "listen_addresses = '127.0.0.1'",
        "net.bindIp: 127.0.0.1",
        "bind 127.0.0.1 ::1",
        "protected-mode yes",
        "server.default_listen_address=127.0.0.1",
        "service.host: 127.0.0.1",
        "LLM_HOST",
        "MTC_API_HOST=127.0.0.1",
    ):
        assert line in result.stdout, line
    # It ran lsof and nothing else, and wrote nothing into the project.
    assert {call[0] for call in ops.stubs.calls()} == {"lsof"}
    assert sorted(path.name for path in ops.project.iterdir()) == []

    # The source cannot edit a configuration file: the only thing it changes
    # is the mode of the four private paths.
    source = (OPS_DIR / "harden_services.sh").read_text()
    code = [
        line
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    body = "\n".join(code)
    for forbidden in (r"\bsed\b", r"\btee\b", r"\bmv\b", r"\bcp\b", r"\brm\b", r">>"):
        assert re.search(forbidden, body) is None, forbidden
    writes = [line for line in code if re.search(r"(?<![<>&])>(?![>&])\s*\S", line)]
    writes = [line for line in writes if ">&2" not in line and "/dev/null" not in line]
    assert writes == [], writes
    assert len(re.findall(r"^\s*chmod ", body, re.MULTILINE)) == 1
    assert (
        len(re.findall(r"^check_mode ", body, re.MULTILINE))
        + len(re.findall(r"^\s+check_mode ", body, re.MULTILINE))
        == 4
    )


def _healthy_stubs(ops: Ops) -> None:
    install_store_stubs(ops)
    ops.stubs.add("pg_isready")


def test_health_check_alert_only_exit_codes(ops):
    _healthy_stubs(ops)
    ops.set(POSTGRES_ENABLED=True, QDRANT_ENABLED=True, REDIS_ENABLED=True)
    log = ops.project / "health.log"

    healthy = ops.run("health_check.sh", "--log-file", str(log))
    assert healthy.returncode == 0, healthy.stdout + healthy.stderr
    assert healthy.stdout.splitlines() == [
        "postgres: healthy",
        "qdrant: healthy",
        "redis: healthy",
        "api: healthy",
    ]
    # Quiet and healthy: silence.
    quiet = ops.run("health_check.sh", "--quiet", "--log-file", str(log))
    assert quiet.returncode == 0 and quiet.stdout == ""

    # One component down: exit 1, and the line says which.
    ops.stubs.add("pg_isready", exit_code=1)
    down = ops.run("health_check.sh", "--log-file", str(log))
    assert down.returncode == 1
    assert "postgres: DOWN" in down.stdout and "qdrant: healthy" in down.stdout
    loud = ops.run("health_check.sh", "--quiet", "--log-file", str(log))
    assert loud.returncode == 1
    assert loud.stdout.splitlines() == ["postgres: DOWN"]

    lines = log.read_text().splitlines()
    assert len(lines) == 4
    stamp = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
    assert re.fullmatch(
        stamp
        + r" health: ok postgres=healthy qdrant=healthy redis=healthy api=healthy",
        lines[0],
    )
    assert re.fullmatch(
        stamp + r" health: UNHEALTHY postgres=down qdrant=healthy redis=healthy "
        r"api=healthy",
        lines[2],
    )

    # A local brain is a component; a disabled store is not listed.
    ops.set(LLM_PROVIDER="local", POSTGRES_ENABLED=False)
    brain = ops.run("health_check.sh")
    assert brain.returncode == 0
    assert "brain: healthy" in brain.stdout and "postgres" not in brain.stdout
    planned = ops.run("health_check.sh", "--dry-run")
    assert planned.returncode == 0
    assert "would check: GET http://" in planned.stdout
    assert "nothing was contacted" in planned.stdout
    assert ops.run("health_check.sh", "--log-file").returncode == 2


def test_health_check_never_restarts(ops):
    # Everything down, a service manager configured: the check still only
    # reports, and calls no manager, no launcher and no signal.
    install_store_stubs(ops)
    for tool in ("pg_isready", "mongosh", "redis-cli", "curl"):
        ops.stubs.add(tool, exit_code=1)
    for manager in ("brew", "docker", "systemctl", "mtc", "llama-server"):
        ops.stubs.add(manager)
    ops.set(
        SERVICE_MANAGER="brew",
        POSTGRES_ENABLED=True,
        MONGODB_ENABLED=True,
        QDRANT_ENABLED=True,
        REDIS_ENABLED=True,
        NEO4J_ENABLED=True,
        LLM_PROVIDER="local",
        LLM_BACKEND="llama_cpp",
    )
    result = ops.run("health_check.sh")
    assert result.returncode == 1
    assert result.stdout.count("DOWN") == 7
    called = {call[0] for call in ops.stubs.calls()}
    assert called == {"pg_isready", "mongosh", "redis-cli", "curl"}
    assert not (ops.project / "data").exists() and not (ops.project / "logs").exists()

    # The source holds no command that starts, restarts or signals anything.
    source = (OPS_DIR / "health_check.sh").read_text()
    code = "\n".join(
        line
        for line in source.split("EOF\n}", 1)[1].splitlines()
        if not line.lstrip().startswith("#")
    )
    for forbidden in (
        r"\bbrew\b",
        r"\bsystemctl\b",
        r"\bdocker\b",
        r"\bkill\b",
        r"\bpkill\b",
        r"\bnohup\b",
        r"\bexec\s+[a-z]",
        r"start_llm",
        r"start_stack",
        r"stop_stack",
        r"mtc serve",
        r"\brestart\b",
        r"services (start|stop)",
        r"&\s*$",
    ):
        assert re.search(forbidden, code, re.MULTILINE) is None, forbidden


def test_harden_names_enabled_services_with_no_visible_listener(ops):
    """lsof without root lists only this user's sockets, so an enabled store
    with no visible listener was not measured; the report names it instead
    of calling every listener loopback."""
    pg_port = ops.ports["POSTGRES_PORT"]
    ops.set(POSTGRES_ENABLED=True, REDIS_ENABLED=True, LLM_PROVIDER="mock")
    stub_lsof(ops, [lsof_line("redis-ser", "127.0.0.1", ops.ports["REDIS_PORT"])])
    result = ops.run("harden_services.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FINDING" not in result.stdout
    assert f"no listener visible: postgres (port {pg_port})" in result.stdout
    assert "not measured" in result.stdout
    assert "every listed listener is on loopback" not in result.stdout
    # A disabled store, and the model server under the mock provider, are not
    # expected, so they are not named.
    assert "mongodb (port" not in result.stdout
    assert "model server (port" not in result.stdout
    assert "redis (port" not in result.stdout


def test_health_check_labels_a_bare_port_probe_not_healthy(ops):
    """Without the store's client tool (and always for Neo4j) the check is a
    TCP connect, and an open port is not a health answer: the line says
    "port open" and why, never "healthy"."""
    import socket

    from .conftest import LOOPBACK

    install_store_stubs(ops)
    ops.stubs.remove("pg_isready")
    ops.set(POSTGRES_ENABLED=True, NEO4J_ENABLED=True)
    log = ops.project / "health.log"
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as pg,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bolt,
    ):
        pg.bind((LOOPBACK, ops.ports["POSTGRES_PORT"]))
        pg.listen(8)
        bolt.bind((LOOPBACK, ops.neo4j_port))
        bolt.listen(8)
        result = ops.run("health_check.sh", "--log-file", str(log))
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert "postgres: healthy" not in lines and "neo4j: healthy" not in lines
    assert (
        "postgres: port open (pg_isready is not installed, so health was not "
        "measured)" in lines
    )
    assert any(line.startswith("neo4j: port open (") for line in lines), lines
    (entry,) = log.read_text().splitlines()
    assert entry.endswith(
        " health: unverified postgres=port_open neo4j=port_open api=healthy"
    )
