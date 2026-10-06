"""Stub-command helpers for the operations script tests.

The scripts under ``scripts/ops`` talk to stores, service managers and
servers through command-line tools. No test here runs one of those tools for
real: every script runs with a temporary ``PATH`` whose first entry holds
stub commands, each of which records its arguments (and the environment
variables a test asks for) and then does what the test scripted. A stub that
a test did not script still exists and refuses, so a forgotten stub can never
fall through to a real tool on the machine.

Every port in the scripts' environment is an unused ephemeral loopback port
chosen per test, every store is disabled unless the test enables it, and the
only processes a test ever signals are dummies it started itself.
"""

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPS_DIR = ROOT / "scripts" / "ops"
SHELL_SCRIPTS = (
    "start_stack.sh",
    "stop_stack.sh",
    "start_llm.sh",
    "backup.sh",
    "restore.sh",
    "restore_drill.sh",
    "harden_services.sh",
    "health_check.sh",
)
LOOPBACK = "127.0.0.1"
SEPARATOR = "\x1f"

# Tools the scripts may call. Each gets a refusing stub by default.
GUARDED_TOOLS = (
    "curl",
    "nc",
    "lsof",
    "pg_isready",
    "psql",
    "pg_dump",
    "pg_restore",
    "createdb",
    "dropdb",
    "mongosh",
    "mongodump",
    "mongorestore",
    "redis-cli",
    "neo4j",
    "neo4j-admin",
    "cypher-shell",
    "brew",
    "docker",
    "systemctl",
    "qdrant",
    "llama-server",
    "mtc",
)
PORT_FIELDS = (
    "POSTGRES_PORT",
    "MONGODB_PORT",
    "QDRANT_PORT",
    "REDIS_PORT",
    "LLM_PORT",
    "API_PORT",
)
ENABLE_FIELDS = (
    "POSTGRES_ENABLED",
    "MONGODB_ENABLED",
    "QDRANT_ENABLED",
    "REDIS_ENABLED",
    "NEO4J_ENABLED",
)


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1]


class Stubs:
    """A directory of stub commands that record how they were called."""

    def __init__(self, base: Path) -> None:
        self.bin = base / "stub-bin"
        self.state = base / "stub-state"
        self.log = base / "stub-calls.log"
        self.env_dir = base / "stub-env"
        for directory in (self.bin, self.state, self.env_dir):
            directory.mkdir(parents=True)
        for tool in GUARDED_TOOLS:
            self.add(tool, 'echo "stub: this tool was not scripted" >&2', exit_code=97)

    def add(
        self,
        name: str,
        body: str = "",
        *,
        exit_code: int = 0,
        record_env: Sequence[str] = (),
    ) -> Path:
        """Write a stub; ``body`` is bash run after the call is recorded."""
        env_file = self.env_dir / f"{name}.env"
        lines = [
            "#!/usr/bin/env bash",
            "{",
            f"  printf '%s' '{name}'",
            '  for argument in "$@"; do printf \'\\037%s\' "${argument}"; done',
            "  printf '\\n'",
            f'}} >>"{self.log}"',
        ]
        if record_env:
            names = " ".join(record_env)
            lines += [
                f"for variable in {names}; do",
                f'  printf \'%s=%s\\n\' "${{variable}}" "${{!variable-}}" >>"{env_file}"',
                "done",
                f"printf -- '--\\n' >>\"{env_file}\"",
            ]
        lines.append(f'STATE="{self.state}"')
        lines.append(textwrap.dedent(body).strip("\n"))
        lines.append(f"exit {int(exit_code)}")
        path = self.bin / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def remove(self, name: str) -> None:
        """Drop a stub so the real tool is used (only with test-owned ports)."""
        (self.bin / name).unlink()

    def calls(self, name: Optional[str] = None) -> List[List[str]]:
        """Recorded calls as argument lists, the tool name first."""
        if not self.log.exists():
            return []
        found = []
        for line in self.log.read_text(encoding="utf-8").splitlines():
            parts = line.split(SEPARATOR)
            if name is None or parts[0] == name:
                found.append(parts)
        return found

    def environments(self, name: str) -> List[Dict[str, str]]:
        """The recorded environment variables of each call of a stub."""
        path = self.env_dir / f"{name}.env"
        if not path.exists():
            return []
        records: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line == "--":
                records.append(current)
                current = {}
                continue
            key, _, value = line.partition("=")
            current[key] = value
        return records

    def mark(self, flag: str) -> None:
        (self.state / flag).write_text("", encoding="utf-8")

    def marked(self, flag: str) -> bool:
        return (self.state / flag).exists()


class Ops:
    """One test's project directory, stubs and script environment."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.project = base / "project"
        self.project.mkdir()
        self.stubs = Stubs(base)
        self.ports = {field: unused_port() for field in PORT_FIELDS}
        self.neo4j_port = unused_port()
        self.dummies: List[int] = []
        self.env: Dict[str, str] = {
            "PATH": os.pathsep.join(
                [
                    str(self.stubs.bin),
                    str(Path(sys.executable).parent),
                    "/usr/bin",
                    "/bin",
                    "/usr/sbin",
                    "/sbin",
                ]
            ),
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "HOME": str(base / "home"),
            "LANG": "C",
            "MTC_DEVICE": "cpu",
            "MTC_PROJECT_ROOT": str(self.project),
            "MTC_LLM_PROVIDER": "mock",
            "MTC_NEO4J_URI": f"bolt://{LOOPBACK}:{self.neo4j_port}",
        }
        (base / "home").mkdir()
        for field, port in self.ports.items():
            self.env[f"MTC_{field}"] = str(port)
        for field in ("POSTGRES_HOST", "MONGODB_HOST", "QDRANT_HOST", "REDIS_HOST"):
            self.env[f"MTC_{field}"] = LOOPBACK
        self.env["MTC_LLM_HOST"] = LOOPBACK
        self.env["MTC_API_HOST"] = LOOPBACK
        for field in ENABLE_FIELDS:
            self.env[f"MTC_{field}"] = "false"

    def set(self, **settings: object) -> None:
        """Set Settings fields for the scripts (``NAME=value`` without prefix)."""
        for field, value in settings.items():
            text = str(value).lower() if isinstance(value, bool) else str(value)
            self.env[f"MTC_{field}"] = text

    def run(
        self,
        script: str,
        *arguments: str,
        timeout: float = 180.0,
        stdin_text: Optional[str] = None,
        **extra_env: str,
    ) -> subprocess.CompletedProcess:
        """Run an operations script with the stub environment."""
        env = dict(self.env)
        env.update(extra_env)
        return subprocess.run(
            ["/bin/bash", str(OPS_DIR / script), *arguments],
            cwd=self.project,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=stdin_text if stdin_text is not None else "",
        )

    def call(
        self, script: str, snippet: str, timeout: float = 60.0
    ) -> subprocess.CompletedProcess:
        """Source a script (its main does not run) and run a bash snippet that
        calls its functions, with the stub environment."""
        return subprocess.run(
            [
                "/bin/bash",
                "-c",
                'source "$1"; DRY_RUN=0; FAILED=0; ' + snippet,
                "ops-test-call",
                str(OPS_DIR / script),
            ],
            cwd=self.project,
            env=dict(self.env),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def run_dir(self) -> Path:
        path = self.project / "data" / "run"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def spawn_dummy(self, name: str, body: str) -> int:
        """Start a dummy process this test owns and return its PID.

        Its command line holds ``name``. It is started detached, the way the
        stack script starts a server, so it is not this process's child and
        disappears from the process table as soon as it exits.
        """
        directory = self.base / "dummies"
        directory.mkdir(exist_ok=True)
        script = directory / name
        script.write_text(
            "#!/usr/bin/env bash\n" + textwrap.dedent(body).strip("\n") + "\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        launched = subprocess.run(
            [
                "/bin/bash",
                "-c",
                'nohup /bin/bash "$1" serve >/dev/null 2>&1 & echo $!',
                "launch",
                str(script),
            ],
            capture_output=True,
            text=True,
            cwd=self.project,
            timeout=30,
            check=True,
        )
        pid = int(launched.stdout.strip())
        self.dummies.append(pid)
        return pid

    def adopt_pid_file(self, name: str) -> Optional[int]:
        """Read a PID file a script wrote for a dummy this test owns."""
        path = self.run_dir() / name
        if not path.exists():
            return None
        return int(path.read_text(encoding="utf-8").strip())

    def cleanup(self) -> None:
        """Stop every dummy this test started.

        The PID files live in this test's own project directory, so a
        process they name was started from this test's stub ``PATH`` (or by
        the test itself): it is a dummy the test owns.
        """
        for name in ("api.pid", "llm.pid", "qdrant.pid"):
            try:
                pid = self.adopt_pid_file(name)
            except ValueError:
                pid = None
            if pid is not None:
                _stop_pid(pid)
        for pid in self.dummies:
            _stop_pid(pid)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _stop_pid(pid: int) -> None:
    """Stop a dummy process the test owns (a test may force its own dummy)."""
    if not _alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def pid_alive(pid: int) -> bool:
    return _alive(pid)


def stop_dummy(pid: int) -> None:
    """Stop a dummy the calling test owns."""
    _stop_pid(pid)


def wait_gone(pid: int, timeout: float = 5.0) -> bool:
    """True once the process has left the process table."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


# A stand-in server for stubs that must answer HTTP: it binds the port given
# as --port (or the first argument), answers GET with 503 for the first
# READY_AFTER requests and then 200 with a ready body.
FAKE_SERVICE = """
import http.server
import json
import sys

args = sys.argv[1:]
port = int(args[args.index("--port") + 1]) if "--port" in args else int(args[0])
ready_after = int(args[args.index("--ready-after") + 1])
seen = {"count": 0}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        seen["count"] += 1
        ready = seen["count"] > ready_after
        body = json.dumps({"status": "ok", "ready": ready, "data": []}).encode()
        self.send_response(200 if ready else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def write_fake_service(base: Path) -> Path:
    path = base / "fake_service.py"
    path.write_text(FAKE_SERVICE, encoding="utf-8")
    return path


# Stub bodies for the store tools: each writes what the real tool would, so
# a backup, a restore and a drill can run end to end without any store.
PG_DUMP = """
while [ "$#" -gt 0 ]; do
  case "$1" in -f) printf 'postgres-dump-bytes' >"$2"; shift ;; esac
  shift
done
"""
PSQL = """
case "$*" in
  *pg_tables*) printf 'messages\\nmemories\\n' ;;
  *pg_database*) if [ -f "${STATE}/database-exists" ]; then echo 1; fi ;;
  *'count(*)'*messages*) cat "${STATE}/rows-messages" 2>/dev/null || echo 30 ;;
  *'count(*)'*) echo 12 ;;
esac
"""
MONGODUMP = """
for argument in "$@"; do
  case "${argument}" in
    --archive=*) printf 'mongo-archive-bytes' >"${argument#--archive=}" ;;
  esac
done
previous=""
for argument in "$@"; do
  if [ "${previous}" = "--config" ]; then cp "${argument}" "${STATE}/mongo-options"; fi
  previous="${argument}"
done
"""
MONGOSH = """
case "$*" in
  *dropDatabase*) echo 1 ;;
  *countDocuments*) cat "${STATE}/documents" 2>/dev/null || echo 7 ;;
  *) echo 1 ;;
esac
"""
CURL = """
out=""; method="GET"; url=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) out="$2"; shift ;;
    -X) method="$2"; shift ;;
    --max-time | -H | -F) shift ;;
    --silent | --fail) ;;
    *) url="$1" ;;
  esac
  shift
done
case "${method}:${url}" in
  POST:*/snapshots/upload*) printf '{"result": true}' ;;
  POST:*/snapshots) printf '{"result": {"name": "snap-1.snapshot"}}' ;;
  DELETE:*) printf '{"result": true}' ;;
  GET:*/snapshots/*) printf 'qdrant-snapshot-bytes' >"${out}" ;;
  GET:*/collections/*)
    points="$(cat "${STATE}/points" 2>/dev/null || echo 5)"
    printf '{"result": {"points_count": %s}}' "${points}"
    ;;
  GET:*/health) printf '{"status": "ok", "ready": true}' ;;
  GET:*) if [ -n "${out}" ]; then printf '{}' >"${out}"; else printf '{}'; fi ;;
esac
"""
REDIS_CLI = """
previous=""
for argument in "$@"; do
  if [ "${previous}" = "--rdb" ]; then printf 'redis-rdb-bytes' >"${argument}"; fi
  previous="${argument}"
done
case "$*" in *ping*) echo PONG ;; esac
"""
NEO4J_ADMIN = """
case "$*" in
  *dump*) printf 'neo4j-dump-bytes' ;;
  *load*) cat >"${STATE}/neo4j-loaded" ;;
esac
"""
PASSWORD_VARIABLES = ("PGPASSWORD", "REDISCLI_AUTH")


def install_store_stubs(ops: Ops) -> None:
    """Script every store tool so the backup scripts run end to end."""
    stubs = ops.stubs
    stubs.add("pg_dump", PG_DUMP, record_env=PASSWORD_VARIABLES)
    stubs.add("psql", PSQL, record_env=PASSWORD_VARIABLES)
    stubs.add("pg_restore", record_env=PASSWORD_VARIABLES)
    stubs.add("createdb", record_env=PASSWORD_VARIABLES)
    stubs.add("dropdb", record_env=PASSWORD_VARIABLES)
    stubs.add("mongodump", MONGODUMP)
    stubs.add("mongorestore", MONGODUMP)
    stubs.add("mongosh", MONGOSH)
    stubs.add("curl", CURL)
    stubs.add("redis-cli", REDIS_CLI, record_env=PASSWORD_VARIABLES)
    stubs.add("neo4j-admin", NEO4J_ADMIN)


def enable_all_stores(ops: Ops) -> None:
    ops.set(
        POSTGRES_ENABLED=True,
        MONGODB_ENABLED=True,
        QDRANT_ENABLED=True,
        REDIS_ENABLED=True,
        DB_PREFIX="acme",
    )


def backup_dirs(ops: Ops, location: str = "backups") -> List[Path]:
    root = ops.project / location
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def read_manifest(directory: Path) -> Dict[str, Dict[str, List[str]]]:
    """A manifest's lines by kind: files, counts and statuses."""
    parsed: Dict[str, Dict[str, List[str]]] = {"file": {}, "count": {}, "status": {}}
    for line in (directory / "MANIFEST.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] in parsed:
            parsed[parts[0]][parts[1]] = parts[2:]
    return parsed


@pytest.fixture
def ops(tmp_path) -> Iterator[Ops]:
    environment = Ops(tmp_path)
    try:
        yield environment
    finally:
        environment.cleanup()
