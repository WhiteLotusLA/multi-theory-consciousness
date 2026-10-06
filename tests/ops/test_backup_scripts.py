"""backup.sh against stub dump tools: layout, manifest, retention and secrets.

No dump tool and no store is touched: every tool is a stub that records its
arguments and the password variables it was given, and writes what the real
tool would.
"""

import hashlib
import os
import re
import socket
import stat
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from .conftest import (
    LOOPBACK,
    Ops,
    backup_dirs,
    enable_all_stores,
    install_store_stubs,
    read_manifest,
)

pytestmark = pytest.mark.slow

STAMP = re.compile(r"^\d{8}T\d{6}Z$")
ENV_NAME = "." + "env"


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def write_state(ops: Ops) -> None:
    """The private state a project holds after setup."""
    files = {
        "agent_space/identity_statement.md": "You are Testagent.\n",
        f"agent_space/{ENV_NAME}": "SHOULD_NOT_TRAVEL=1\n",
        "config/persona.yaml": "name: Testagent\n",
        "config/speakers.yaml": "speakers: []\n",
        "config/ctm/persona.yaml": 'name: ""\n',
        "config/ctm/seeds.d/10_growth.yaml": "questions: []\n",
        "config/ctm/ctm.yaml": "{}\n",
        ENV_NAME: "MTC_SYSTEM_NAME=Testagent\n",
    }
    for relative, text in files.items():
        path = ops.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def old_backup(ops: Ops, name: str, location: str = "backups") -> Path:
    path = ops.project / location / name
    path.mkdir(parents=True)
    (path / "MANIFEST.txt").write_text("mtc-backup-manifest 1\n", encoding="utf-8")
    return path


def test_layout_and_manifest_with_sha256(ops):
    install_store_stubs(ops)
    enable_all_stores(ops)
    write_state(ops)
    result = ops.run("backup.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    (backup,) = backup_dirs(ops)
    assert STAMP.match(backup.name)
    assert f"backup written: {backup.name}" in result.stdout
    expected = {
        "postgres.dump": b"postgres-dump-bytes",
        "mongo.archive.gz": b"mongo-archive-bytes",
        "qdrant-acme_memories.snapshot": b"qdrant-snapshot-bytes",
        "redis.rdb": b"redis-rdb-bytes",
    }
    assert sorted(path.name for path in backup.iterdir()) == sorted(
        list(expected) + ["state.tar.gz", "MANIFEST.txt"]
    )
    for name, content in expected.items():
        assert (backup / name).read_bytes() == content
    manifest = read_manifest(backup)
    text = (backup / "MANIFEST.txt").read_text()
    assert text.splitlines()[0] == "mtc-backup-manifest 1"
    assert text.splitlines()[1] == f"created_utc {backup.name}"
    # Every file is listed with its size and its SHA-256.
    assert sorted(manifest["file"]) == sorted(list(expected) + ["state.tar.gz"])
    for name, (size, digest) in manifest["file"].items():
        data = (backup / name).read_bytes()
        assert int(size) == len(data), name
        assert digest == hashlib.sha256(data).hexdigest(), name
    # Rows, documents and points, as the stores answered.
    assert manifest["count"] == {
        "postgres": ["rows", "42"],
        "mongodb": ["documents", "7"],
        "qdrant": ["points", "5"],
    }
    assert manifest["status"] == {
        "postgres": ["ok"],
        "mongodb": ["ok"],
        "qdrant": ["ok"],
        "redis": ["ok"],
        "neo4j": ["skipped"],
        "state": ["ok"],
    }
    # The dump tools were called with the configured names and no password.
    (dump,) = ops.stubs.calls("pg_dump")
    assert dump[1:] == [
        "-Fc",
        "-h",
        LOOPBACK,
        "-p",
        str(ops.ports["POSTGRES_PORT"]),
        "-U",
        "acme",
        "-d",
        "acme_consciousness",
        "-f",
        str(backup / "postgres.dump"),
    ]
    (mongo,) = ops.stubs.calls("mongodump")
    assert "--db" in mongo and mongo[mongo.index("--db") + 1] == "acme_neural"
    assert "--gzip" in mongo and "--config" not in mongo
    (rdb,) = [call for call in ops.stubs.calls("redis-cli") if "--rdb" in call]
    assert rdb[-2:] == ["--rdb", str(backup / "redis.rdb")]
    # The server-side snapshot the backup made is removed again.
    curl_calls = [" ".join(call) for call in ops.stubs.calls("curl")]
    assert any("-X DELETE" in call and "snap-1.snapshot" in call for call in curl_calls)
    assert ops.stubs.calls("neo4j-admin") == []

    # A dry run prints the plan and neither writes nor calls anything more.
    before = len(ops.stubs.calls())
    planned = ops.run("backup.sh", "--dry-run")
    assert planned.returncode == 0
    assert "postgres: would run: pg_dump -Fc" in planned.stdout
    assert "password in PGPASSWORD" in planned.stdout
    assert len(ops.stubs.calls()) == before
    assert len(backup_dirs(ops)) == 1


def test_only_enabled_stores_dumped(ops):
    install_store_stubs(ops)
    ops.set(POSTGRES_ENABLED=True)
    result = ops.run("backup.sh", "--no-state")
    assert result.returncode == 0, result.stdout + result.stderr
    (backup,) = backup_dirs(ops)
    assert sorted(path.name for path in backup.iterdir()) == [
        "MANIFEST.txt",
        "postgres.dump",
    ]
    called = {call[0] for call in ops.stubs.calls()}
    assert called == {"pg_dump", "psql"}
    status = read_manifest(backup)["status"]
    assert status["postgres"] == ["ok"]
    for store in ("mongodb", "qdrant", "redis", "neo4j", "state"):
        assert status[store] == ["skipped"], store

    # With nothing enabled there is only the manifest, and nothing is called.
    ops.set(POSTGRES_ENABLED=False)
    before = len(ops.stubs.calls())
    assert ops.run("backup.sh", "--no-state").returncode == 0
    assert len(ops.stubs.calls()) == before
    # Run back to back, the two backups still got a directory each.
    assert len(backup_dirs(ops)) == 2


def test_neo4j_skipped_unless_offline_flag_and_port_closed(ops):
    install_store_stubs(ops)
    ops.set(NEO4J_ENABLED=True)
    # Not asked for: skipped, and that is not a failure.
    plain = ops.run("backup.sh", "--no-state")
    assert plain.returncode == 0
    assert ops.stubs.calls("neo4j-admin") == []
    assert read_manifest(backup_dirs(ops)[-1])["status"]["neo4j"] == ["skipped"]

    # Asked for while Neo4j runs: nothing is dumped, and the run says so.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bolt:
        bolt.bind((LOOPBACK, ops.neo4j_port))
        bolt.listen(8)
        running = ops.run("backup.sh", "--no-state", "--include-neo4j-offline")
    assert running.returncode == 1
    assert "only be dumped while stopped" in running.stderr
    assert ops.stubs.calls("neo4j-admin") == []
    newest = sorted(backup_dirs(ops))[-1]
    assert not (newest / "neo4j.dump").exists()

    # Asked for with Neo4j stopped: the offline dump is taken.
    import time

    time.sleep(1.1)  # a new second, so a new backup directory
    stopped = ops.run("backup.sh", "--no-state", "--include-neo4j-offline")
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    assert ops.stubs.calls("neo4j-admin") == [
        ["neo4j-admin", "database", "dump", "--to-stdout", "neo4j"]
    ]
    newest = sorted(backup_dirs(ops))[-1]
    assert (newest / "neo4j.dump").read_bytes() == b"neo4j-dump-bytes"
    assert read_manifest(newest)["status"]["neo4j"] == ["ok"]


def test_state_archive_excludes_env_file(ops):
    install_store_stubs(ops)
    write_state(ops)
    assert ops.run("backup.sh").returncode == 0
    (backup,) = backup_dirs(ops)
    with tarfile.open(backup / "state.tar.gz") as archive:
        names = sorted(
            member.name for member in archive.getmembers() if member.isfile()
        )
    assert names == [
        "agent_space/identity_statement.md",
        "config/ctm/persona.yaml",
        "config/ctm/seeds.d/10_growth.yaml",
        "config/persona.yaml",
        "config/speakers.yaml",
    ]
    # The environment file is never part of a backup, wherever one lies.
    assert not any(Path(name).name.startswith(ENV_NAME) for name in names)
    raw = (backup / "state.tar.gz").read_bytes()
    assert b"SHOULD_NOT_TRAVEL" not in raw

    # --no-state leaves the archive out.
    import time

    time.sleep(1.1)
    assert ops.run("backup.sh", "--no-state").returncode == 0
    newest = sorted(backup_dirs(ops))[-1]
    assert not (newest / "state.tar.gz").exists()
    assert read_manifest(newest)["status"]["state"] == ["skipped"]


def test_retention_keeps_minimum_and_deletes_old(ops):
    install_store_stubs(ops)
    ops.set(BACKUP_RETENTION_DAYS=30, BACKUP_KEEP_MINIMUM=5)
    old = [f"200001{day:02d}T000000Z" for day in range(1, 9)]
    for name in old:
        old_backup(ops, name)
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y%m%dT%H%M%SZ")
    old_backup(ops, recent)
    keep_me = ops.project / "backups" / "notes"
    keep_me.mkdir()
    (ops.project / "backups" / "20000101T000000Z.txt").write_text("a file\n")

    planned = ops.run("backup.sh", "--no-state", "--dry-run")
    assert planned.returncode == 0
    assert "retention: would remove 20000101T000000Z" in planned.stdout
    assert len(backup_dirs(ops)) == 10

    result = ops.run("backup.sh", "--no-state")
    assert result.returncode == 0, result.stdout + result.stderr
    names = [path.name for path in backup_dirs(ops)]
    new = [n for n in names if STAMP.match(n) and n not in old and n != recent]
    assert len(new) == 1
    # The newest five are kept whatever their age: the new one, the recent
    # one and the three youngest old ones. Older than 30 days beyond that go.
    assert sorted(names) == sorted(
        [
            "notes",
            recent,
            new[0],
            "20000106T000000Z",
            "20000107T000000Z",
            "20000108T000000Z",
        ]
    )
    for removed in old[:5]:
        assert f"retention: removed {removed}" in result.stdout
    # What is not a backup directory is never touched.
    assert keep_me.is_dir()
    assert (ops.project / "backups" / "20000101T000000Z.txt").is_file()

    # A larger minimum keeps everything.
    ops.set(BACKUP_KEEP_MINIMUM=50)
    for name in old[:3]:
        old_backup(ops, name)
    import time

    time.sleep(1.1)
    assert ops.run("backup.sh", "--no-state").returncode == 0
    assert all((ops.project / "backups" / name).is_dir() for name in old[:3])


def test_retention_never_follows_a_link_out_of_the_location(ops):
    install_store_stubs(ops)
    ops.set(BACKUP_RETENTION_DAYS=1, BACKUP_KEEP_MINIMUM=1)
    outside = ops.base / "outside" / "20000101T000000Z"
    outside.mkdir(parents=True)
    (outside / "precious.txt").write_text("keep\n")
    (ops.project / "backups").mkdir()
    (ops.project / "backups" / "20000101T000000Z").symlink_to(outside)
    result = ops.run("backup.sh", "--no-state")
    assert result.returncode == 1
    assert "failed the containment check and was left alone" in result.stderr
    assert (outside / "precious.txt").read_text() == "keep\n"

    # The containment assertion itself, on real and on wrong candidates.
    location = (ops.project / "backups").resolve()
    real = old_backup(ops, "20000202T000000Z")
    ok = ops.call("backup.sh", f'assert_inside "{real.resolve()}" "{location}"')
    assert ok.returncode == 0, ok.stderr
    for candidate in (
        location / "20000101T000000Z",  # a symbolic link
        location / "notes",  # not a backup name
        location,  # the location itself
        ops.project,  # outside
        outside,  # a backup name, elsewhere
    ):
        refused = ops.call("backup.sh", f'assert_inside "{candidate}" "{location}"')
        assert refused.returncode != 0, candidate


def test_retention_refuses_unsafe_location(ops):
    install_store_stubs(ops)
    ops.set(POSTGRES_ENABLED=True, BACKUP_RETENTION_DAYS=1, BACKUP_KEEP_MINIMUM=1)
    victim = ops.project / "20000101T000000Z"
    victim.mkdir()
    (victim / "data.txt").write_text("not a backup\n")
    elsewhere = ops.base / "elsewhere"
    old_outside = elsewhere / "20000101T000000Z"
    old_outside.mkdir(parents=True)
    (ops.project / "linked").symlink_to(elsewhere)

    cases = {
        "/": "file system root",
        ".": "project directory itself",
        "..": "contains the project directory",
        "backups/../../elsewhere": "resolves outside",
        "linked": "resolves outside",
    }
    for location, words in cases.items():
        result = ops.run("backup.sh", MTC_BACKUP_LOCATION=location)
        assert result.returncode == 2, location
        assert words in result.stderr, location
        assert "refusing to back up or to apply retention there" in result.stderr
    # --dest is held to the same rule.
    assert ops.run("backup.sh", "--dest", "/").returncode == 2
    assert ops.run("backup.sh", "--dest", "~/backups").returncode == 2
    # Nothing was dumped, nothing was removed, anywhere.
    assert ops.stubs.calls() == []
    assert (victim / "data.txt").is_file()
    assert old_outside.is_dir()

    # The assertion itself: each unsafe shape is refused, a safe one passes.
    project = str(ops.project.resolve())
    refused = (
        ('""', '""'),
        ("/", "/"),
        (".", project),
        ("..", str(ops.project.resolve().parent)),
        ("backups/../../x", str(ops.base.resolve() / "x")),
    )
    for configured, resolved in refused:
        outcome = ops.call(
            "backup.sh", f'assert_safe_location {configured} "{resolved}" "{project}"'
        )
        assert outcome.returncode != 0, configured
    safe = ops.call(
        "backup.sh", f'assert_safe_location backups "{project}/backups" "{project}"'
    )
    assert safe.returncode == 0, safe.stderr
    absolute = ops.call(
        "backup.sh",
        f'assert_safe_location "{ops.base.resolve()}/vault" '
        f'"{ops.base.resolve()}/vault" "{project}"',
    )
    assert absolute.returncode == 0, absolute.stderr


def test_one_failed_dump_continues_and_exits_1(ops):
    install_store_stubs(ops)
    enable_all_stores(ops)
    ops.stubs.add("pg_dump", 'echo "pg_dump: connection refused" >&2', exit_code=1)
    result = ops.run("backup.sh", "--no-state")
    assert result.returncode == 1
    assert "postgres: the dump failed; the other stores are still backed up" in (
        result.stderr
    )
    (backup,) = backup_dirs(ops)
    present = sorted(path.name for path in backup.iterdir())
    assert present == [
        "MANIFEST.txt",
        "mongo.archive.gz",
        "qdrant-acme_memories.snapshot",
        "redis.rdb",
    ]
    manifest = read_manifest(backup)
    assert manifest["status"]["postgres"] == ["failed"]
    for store in ("mongodb", "qdrant", "redis"):
        assert manifest["status"][store] == ["ok"], store
    assert "postgres.dump" not in manifest["file"]
    assert "postgres" not in manifest["count"]

    # A tool that exits 0 without writing anything is a failure too.
    ops.stubs.add("pg_dump")
    import time

    time.sleep(1.1)
    empty = ops.run("backup.sh", "--no-state")
    assert empty.returncode == 1
    assert "wrote no data" in empty.stderr


def test_password_passed_by_env_not_argv(ops):
    install_store_stubs(ops)
    enable_all_stores(ops)
    secrets_held = {
        "postgres": "pg-password-value-1",
        "mongodb": "mongo-password-value-1",
        "redis": "redis-password-value-1",
    }
    ops.set(
        POSTGRES_PASSWORD=secrets_held["postgres"],
        MONGODB_USER="acme_app",
        MONGODB_PASSWORD=secrets_held["mongodb"],
        REDIS_PASSWORD=secrets_held["redis"],
    )
    result = ops.run("backup.sh", "--no-state")
    assert result.returncode == 0, result.stdout + result.stderr
    (backup,) = backup_dirs(ops)

    # PostgreSQL and Redis tools get the password in their own variable.
    for tool in ("pg_dump", "psql"):
        for seen in ops.stubs.environments(tool):
            assert (seen["PGPASSWORD"] == secrets_held["postgres"]) is True, tool
    rdb_env = ops.stubs.environments("redis-cli")
    assert rdb_env and all(
        (seen["REDISCLI_AUTH"] == secrets_held["redis"]) is True for seen in rdb_env
    )
    # The MongoDB tools read theirs from a private options file that is gone
    # afterwards.
    (mongo,) = ops.stubs.calls("mongodump")
    options = mongo[mongo.index("--config") + 1]
    assert mongo[mongo.index("--username") + 1] == "acme_app"
    assert not Path(options).exists()
    copied = (ops.stubs.state / "mongo-options").read_text()
    assert (secrets_held["mongodb"] in copied) is True

    # No password on any command line, in the output or in the backup.
    argv = "\n".join(" ".join(call) for call in ops.stubs.calls())
    output = result.stdout + result.stderr
    manifest = (backup / "MANIFEST.txt").read_text()
    for value in secrets_held.values():
        assert (value in argv) is False
        assert (value in output) is False
        assert (value in manifest) is False
    leftovers = [path.name for path in backup.iterdir() if path.name.startswith(".")]
    assert leftovers == []
    # With credentials configured mongosh is not used, so the count is unknown.
    assert read_manifest(backup)["count"]["mongodb"] == ["documents", "unknown"]
    assert ops.stubs.calls("mongosh") == []


def test_backup_dir_mode_0700(ops):
    install_store_stubs(ops)
    enable_all_stores(ops)
    write_state(ops)
    previous = os.umask(0o022)
    try:
        result = ops.run("backup.sh")
    finally:
        os.umask(previous)
    assert result.returncode == 0, result.stdout + result.stderr
    assert mode_of(ops.project / "backups") == 0o700
    (backup,) = backup_dirs(ops)
    assert mode_of(backup) == 0o700
    for path in backup.iterdir():
        assert mode_of(path) == 0o600, path.name
    # An existing, looser location is tightened.
    loose = ops.project / "vault"
    loose.mkdir(mode=0o755)
    loose.chmod(0o755)
    assert ops.run("backup.sh", "--dest", "vault", "--no-state").returncode == 0
    assert mode_of(loose) == 0o700


def failed_backup(ops: Ops, name: str) -> Path:
    path = ops.project / "backups" / name
    path.mkdir(parents=True)
    (path / "MANIFEST.txt").write_text(
        "mtc-backup-manifest 1\nstatus postgres failed\nstatus state ok\n",
        encoding="utf-8",
    )
    return path


def test_retention_skipped_when_this_run_failed(ops):
    install_store_stubs(ops)
    ops.set(POSTGRES_ENABLED=True, BACKUP_RETENTION_DAYS=1, BACKUP_KEEP_MINIMUM=1)
    old = ["20000101T000000Z", "20000102T000000Z", "20000103T000000Z"]
    for name in old:
        old_backup(ops, name)
    ops.stubs.add("pg_dump", 'echo "pg_dump: connection refused" >&2', exit_code=1)
    result = ops.run("backup.sh", "--no-state")
    assert result.returncode == 1
    assert "retention: skipped because this run had failures" in result.stderr
    assert "retention: removed" not in result.stdout
    # Every earlier backup is still there.
    assert all((ops.project / "backups" / name).is_dir() for name in old)


def test_failed_backups_do_not_count_toward_keep_minimum(ops):
    install_store_stubs(ops)
    ops.set(BACKUP_RETENTION_DAYS=30, BACKUP_KEEP_MINIMUM=2)
    good = ["20000101T000000Z", "20000102T000000Z"]
    bad = [f"200001{day:02d}T000000Z" for day in range(3, 7)]
    for name in good:
        old_backup(ops, name)
    for name in bad:
        failed_backup(ops, name)
    result = ops.run("backup.sh", "--no-state")
    assert result.returncode == 0, result.stdout + result.stderr
    names = [path.name for path in backup_dirs(ops)]
    new = [n for n in names if n not in good + bad]
    assert len(new) == 1
    # The new backup and the youngest good one make the minimum of two; the
    # failed runs in between neither count nor survive their age.
    assert sorted(names) == sorted([new[0], "20000102T000000Z"])
    for name in bad + ["20000101T000000Z"]:
        assert f"retention: removed {name}" in result.stdout


def test_interrupted_mongodump_leaves_no_password_file(ops):
    install_store_stubs(ops)
    ops.set(MONGODB_ENABLED=True, MONGODB_USER="acme_app")
    ops.set(MONGODB_PASSWORD="mongo-password-value-2")
    ops.stubs.add(
        "mongodump",
        MONGODUMP_THEN_INTERRUPT,
    )
    scratch = ops.base / "tmp"
    scratch.mkdir()
    result = ops.run("backup.sh", "--no-state", TMPDIR=str(scratch))
    assert result.returncode != 0
    (call,) = ops.stubs.calls("mongodump")
    options = Path(call[call.index("--config") + 1])
    # The stub saw the options file, so the interrupt came while it existed.
    assert (ops.stubs.state / "mongo-options").is_file()
    assert not options.exists()
    assert list(scratch.iterdir()) == []
    for backup in backup_dirs(ops):
        assert [p.name for p in backup.iterdir() if p.name.startswith(".")] == []


MONGODUMP_THEN_INTERRUPT = """
previous=""
for argument in "$@"; do
  if [ "${previous}" = "--config" ]; then cp "${argument}" "${STATE}/mongo-options"; fi
  previous="${argument}"
done
kill -TERM "${PPID}"
sleep 1
exit 1
"""


def test_state_path_outside_project_is_reported(ops):
    install_store_stubs(ops)
    write_state(ops)
    outside = ops.base / "elsewhere-agent-space"
    outside.mkdir()
    (outside / "notes.md").write_text("kept elsewhere\n", encoding="utf-8")
    ops.set(AGENT_SPACE_DIR=str(outside))
    result = ops.run("backup.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    (backup,) = backup_dirs(ops)
    text = (backup / "MANIFEST.txt").read_text(encoding="utf-8")
    assert "excluded state AGENT_SPACE_DIR outside_project" in text.splitlines()
    assert "AGENT_SPACE_DIR is outside the project" in result.stderr
    # The path itself is not echoed (it may hold a user name).
    assert str(outside) not in result.stdout + result.stderr
    with tarfile.open(backup / "state.tar.gz") as archive:
        names = [member.name for member in archive.getmembers()]
    assert not any("notes.md" in name for name in names)
