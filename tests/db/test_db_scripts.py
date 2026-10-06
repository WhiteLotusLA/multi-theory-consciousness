"""The scripts/db wrappers delegate to the package, and init_postgres.sh is
safe: no password literal, no drop, identifier quoting through format and
gexec, bash and shellcheck clean, and a password that reaches psql only
through the environment. Every run uses a stub psql on a temporary PATH."""

import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.core.config import Settings
from mtc.db.migrations import MIGRATION_FILES

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "db" / "init_postgres.sh"
SECRET = "pw" + "x7" * 12

WRAPPERS = {
    "init_mongo.py": ("mtc.db.bootstrap", ["--only", "mongodb"]),
    "init_qdrant.py": ("mtc.db.bootstrap", ["--only", "qdrant"]),
    "init_neo4j.py": ("mtc.db.bootstrap", ["--only", "neo4j"]),
    "bootstrap.py": ("mtc.db.bootstrap", []),
    "verify.py": ("mtc.db.verify", []),
    "reembed_postgres_to_qdrant.py": ("mtc.db.reembed", []),
}

STUB_PSQL = """#!/bin/bash
input=""
for arg in "$@"; do
    if [ "$arg" = "-" ]; then
        input="$(cat)"
    fi
done
{
    printf 'CALL\\n'
    for arg in "$@"; do
        printf 'ARG %s\\n' "$arg"
    done
    printf 'PGPASSWORD %s\\n' "${PGPASSWORD:+set}"
    if [ -n "${PGPASSWORD:-}" ] && [ "${PGPASSWORD}" = "$(cat "$STUB_EXPECTED")" ]; then
        printf 'PGPASSWORD_MATCH yes\\n'
    fi
    if [ "$(printenv MTC_INIT_PASSWORD)" = "$(cat "$STUB_EXPECTED")" ]; then
        printf 'INIT_PASSWORD_MATCH yes\\n'
    fi
    printf 'PGUSER %s\\n' "${PGUSER:-}"
    printf 'PGDATABASE %s\\n' "${PGDATABASE:-}"
    printf 'PGHOST %s\\n' "${PGHOST:-}"
    printf 'STDIN %s\\n' "$input"
} >> "$STUB_LOG"
case "$input" in
    *"SELECT 1 FROM pg_catalog.pg_roles"*) printf '%s\\n' "${STUB_ROLE_EXISTS:-}" ;;
esac
exit 0
"""


def _child_env(tmp_path, stub_dir=None, **extra):
    path = "/usr/bin:/bin"
    if stub_dir is not None:
        path = f"{stub_dir}:{path}"
    env = {
        "PATH": path,
        "HOME": str(tmp_path),
        "LANG": "C",
        "MTC_DEVICE": "cpu",
        "MTC_LLM_PROVIDER": "mock",
    }
    env.update(extra)
    return env


def _env_file(tmp_path, password=SECRET):
    path = tmp_path / "project.env"
    path.write_text(
        "# test project\n"
        "MTC_DB_PREFIX=lab\n"
        f"MTC_POSTGRES_PASSWORD={password}\n"
        "MTC_POSTGRES_HOST=127.0.0.1\n"
        "MTC_OTHER_SECRET=ignored-value\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_wrappers_delegate_with_only_flag(name, monkeypatch):
    module_name, prefix = WRAPPERS[name]
    path = ROOT / "scripts" / "db" / name
    text = path.read_text(encoding="utf-8")
    assert f"from {module_name} import main" in text
    module = __import__(module_name, fromlist=["main"])
    seen = []

    def fake_main(argv=None):
        seen.append(list(argv))
        return 7

    monkeypatch.setattr(module, "main", fake_main)
    monkeypatch.setattr(sys, "argv", [str(path), "--json"])
    with pytest.raises(SystemExit) as info:
        runpy.run_path(str(path), run_name="__main__")
    assert info.value.code == 7
    assert seen == [prefix + ["--json"]]


def test_init_postgres_script_has_no_password_literal_or_drop():
    text = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"\bDROP\b", text)
    assert not re.search(r"(?i)\bALTER\s+(ROLE|USER)\b", text)
    assert not re.search(r"PASSWORD\s+'", text)
    assert not re.search(r"(?i)password\s*=\s*['\"][^'\"$]", text)
    # No psql variable carries the password on the command line, and every
    # PGPASSWORD assignment takes the role's password from its variable.
    assert not re.search(r"-v\s+pw\b", text)
    assert not re.search(r'PGPASSWORD=(?!"\$\{role_password\}")', text)
    assert not re.search(r"(^|\s)(eval|source)\s", text)
    assert not re.search(r"(^|\s)\.\s+\S", text)
    # The role's password reaches psql through the environment only.
    assert "printenv MTC_INIT_PASSWORD" in text
    assert 'PGPASSWORD="${role_password}"' in text


def test_init_postgres_uses_gexec_and_identifier_quoting():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "\\\\gexec" in text
    assert "format('${role_template}', :'role', :'pw')" in text
    assert "CREATE ROLE %I LOGIN PASSWORD %L" in text
    assert 'CREATE ROLE %I LOGIN"' in text
    assert "CREATE DATABASE %I OWNER %I" in text
    assert "WHERE NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles" in text
    assert "WHERE NOT EXISTS (SELECT 1 FROM pg_catalog.pg_database" in text
    assert text.count("ON_ERROR_STOP=1") >= 3
    assert "set -euo pipefail" in text
    # The default prefix mirrors Settings.
    default = Settings.model_fields["DB_PREFIX"].default
    assert f'DEFAULT_DB_PREFIX="{default}"' in text


def test_init_postgres_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_init_postgres_shellcheck():
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck is not installed")
    result = subprocess.run(
        ["shellcheck", str(SCRIPT)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_init_postgres_dry_run_prints_statements_without_password(tmp_path):
    env_file = _env_file(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--env-file", str(env_file)],
        cwd=tmp_path,
        env=_child_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert SECRET not in out + result.stderr
    assert "ignored-value" not in out
    assert "role lab, database lab_consciousness" in out
    assert "CREATE ROLE %I LOGIN PASSWORD %L" in out
    assert "CREATE DATABASE %I OWNER %I" in out
    assert "\\gexec" in out
    for name, _ in MIGRATION_FILES:
        assert name in out
    # An empty password plans LOGIN without one and says so.
    empty = _env_file(tmp_path, password="")
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--no-migrate", "--env-file", str(empty)],
        cwd=tmp_path,
        env=_child_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "CREATE ROLE %I LOGIN'" in result.stdout
    assert "trust authentication" in result.stdout
    assert "001_core_schema.sql" not in result.stdout


def _stub(tmp_path):
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    psql = stub_dir / "psql"
    psql.write_text(STUB_PSQL, encoding="utf-8")
    psql.chmod(0o755)
    expected = tmp_path / "expected.txt"
    expected.write_text(SECRET, encoding="utf-8")
    return stub_dir, tmp_path / "psql.log", expected


def _calls(log):
    calls = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line == "CALL":
            calls.append([])
        else:
            calls[-1].append(line)
    return calls


def test_init_postgres_passes_password_by_environment_only(tmp_path):
    stub_dir, log, expected = _stub(tmp_path)
    env_file = _env_file(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--env-file", str(env_file)],
        cwd=tmp_path,
        env=_child_env(
            tmp_path, stub_dir, STUB_LOG=str(log), STUB_EXPECTED=str(expected)
        ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert SECRET not in result.stdout + result.stderr
    assert SECRET not in log.read_text(encoding="utf-8")
    calls = _calls(log)
    assert len(calls) == 3 + len(MIGRATION_FILES)
    exists, role, database = calls[:3]
    assert any("pg_catalog.pg_roles WHERE rolname" in line for line in exists)
    role_text = "\n".join(role)
    assert "CREATE ROLE %I LOGIN PASSWORD %L" in role_text
    assert "INIT_PASSWORD_MATCH yes" in role
    assert "PGPASSWORD " in role  # the administrator's connection has none here
    assert "CREATE DATABASE %I OWNER %I" in "\n".join(database)
    for call, (name, _) in zip(calls[3:], MIGRATION_FILES):
        assert f"ARG {ROOT / 'migrations' / 'postgres' / name}" in call
        assert "PGPASSWORD set" in call and "PGPASSWORD_MATCH yes" in call
        assert "PGUSER lab" in call and "PGDATABASE lab_consciousness" in call
        assert "PGHOST 127.0.0.1" in call
        assert "INIT_PASSWORD_MATCH yes" not in call
    assert "already exists" not in result.stdout


def test_init_postgres_keeps_an_existing_role(tmp_path):
    stub_dir, log, expected = _stub(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--no-migrate", "--env-file", str(_env_file(tmp_path))],
        cwd=tmp_path,
        env=_child_env(
            tmp_path,
            stub_dir,
            STUB_LOG=str(log),
            STUB_EXPECTED=str(expected),
            STUB_ROLE_EXISTS="1",
        ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "role lab already exists; its password was not changed" in result.stdout
    assert len(_calls(log)) == 3


def test_init_postgres_without_psql_exits_2(tmp_path):
    if (Path("/usr/bin") / "psql").exists() or (Path("/bin") / "psql").exists():
        pytest.skip("a system psql is on the minimal PATH")
    result = subprocess.run(
        ["bash", str(SCRIPT), "--env-file", str(_env_file(tmp_path))],
        cwd=tmp_path,
        env=_child_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "psql not found" in result.stderr
    assert not os.path.exists(tmp_path / "psql.log")
