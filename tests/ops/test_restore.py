"""restore.sh against stub tools: confirmation, checksums and each store's
command. Nothing is restored anywhere: every tool is a recording stub."""

import hashlib
import socket
from pathlib import Path
from typing import Dict

import pytest

from .conftest import LOOPBACK, Ops, install_store_stubs

pytestmark = pytest.mark.slow

FILES: Dict[str, bytes] = {
    "postgres.dump": b"postgres-dump-bytes",
    "mongo.archive.gz": b"mongo-archive-bytes",
    "qdrant-acme_memories.snapshot": b"qdrant-snapshot-bytes",
    "redis.rdb": b"redis-rdb-bytes",
    "neo4j.dump": b"neo4j-dump-bytes",
}


def make_backup(
    ops: Ops, files: Dict[str, bytes] = FILES, name: str = "20000101T000000Z"
) -> Path:
    """A backup directory as backup.sh writes it, with a true manifest."""
    backup = ops.project / "backups" / name
    backup.mkdir(parents=True)
    lines = ["mtc-backup-manifest 1", f"created_utc {name}"]
    for name, data in files.items():
        (backup / name).write_bytes(data)
        lines.append(f"file {name} {len(data)} {hashlib.sha256(data).hexdigest()}")
    lines += [
        "count postgres rows 42",
        "count mongodb documents 7",
        "count qdrant points 5",
    ]
    (backup / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return backup


@pytest.fixture
def restore(ops):
    install_store_stubs(ops)
    ops.set(DB_PREFIX="acme", POSTGRES_PASSWORD="pg-password-value-1")
    return ops, make_backup(ops)


def run_restore(ops: Ops, backup: Path, store: str, *extra: str, typed: str = ""):
    return ops.run(
        "restore.sh", "--from", str(backup), "--store", store, *extra, stdin_text=typed
    )


def test_restore_requires_yes_or_typed_confirmation(restore):
    ops, backup = restore
    # Nothing typed, the wrong thing typed, or only "yes": nothing happens.
    for typed in ("", "\n", "yes\n", "acme\n", "ACME_CONSCIOUSNESS\n"):
        refused = run_restore(ops, backup, "postgres", typed=typed)
        assert refused.returncode == 2, typed
        assert "not confirmed; nothing was restored" in refused.stderr
        assert "Type the target name (acme_consciousness)" in refused.stderr
    assert ops.stubs.calls() == []

    # The target's name, typed: the restore runs.
    typed = run_restore(ops, backup, "postgres", typed="acme_consciousness\n")
    assert typed.returncode == 0, typed.stderr
    assert "postgres: restored into acme_consciousness" in typed.stdout
    assert len(ops.stubs.calls("pg_restore")) == 1
    # --yes: no question is asked.
    assert run_restore(ops, backup, "postgres", "--yes").returncode == 0
    assert len(ops.stubs.calls("pg_restore")) == 2
    # With a suffix, the name to type is the suffixed one.
    suffixed = run_restore(
        ops,
        backup,
        "postgres",
        "--target-suffix",
        "_copy",
        typed="acme_consciousness\n",
    )
    assert suffixed.returncode == 2
    assert len(ops.stubs.calls("pg_restore")) == 2

    for arguments in (
        ["--from", str(backup)],
        ["--store", "postgres"],
        ["--from", str(backup), "--store", "sqlite"],
        ["--from", str(backup), "--store", "postgres", "--target-suffix", "a;b"],
        ["--from", str(ops.project / "nowhere"), "--store", "postgres", "--yes"],
    ):
        assert ops.run("restore.sh", *arguments).returncode == 2, arguments


def test_restore_commands_per_store(restore):
    ops, backup = restore
    pg_port = str(ops.ports["POSTGRES_PORT"])
    connection = ["-h", LOOPBACK, "-p", pg_port, "-U", "acme"]

    assert run_restore(ops, backup, "postgres", "--yes").returncode == 0
    assert ops.stubs.calls("pg_restore") == [
        ["pg_restore", "--clean", "--if-exists", "--no-owner"]
        + connection
        + ["-d", "acme_consciousness", str(backup / "postgres.dump")]
    ]
    assert ops.stubs.calls("createdb") == []
    for seen in ops.stubs.environments("pg_restore"):
        assert (seen["PGPASSWORD"] == "pg-password-value-1") is True

    # A suffixed target is created first when it does not exist yet.
    copied = run_restore(ops, backup, "postgres", "--target-suffix", "_copy", "--yes")
    assert copied.returncode == 0, copied.stderr
    assert ops.stubs.calls("createdb") == [
        ["createdb"] + connection + ["acme_consciousness_copy"]
    ]
    assert ops.stubs.calls("pg_restore")[-1][-3:-1] == ["-d", "acme_consciousness_copy"]
    ops.stubs.mark("database-exists")
    assert (
        run_restore(
            ops, backup, "postgres", "--target-suffix", "_copy", "--yes"
        ).returncode
        == 0
    )
    assert len(ops.stubs.calls("createdb")) == 1

    assert run_restore(ops, backup, "mongodb", "--yes").returncode == 0
    (mongo,) = ops.stubs.calls("mongorestore")
    assert mongo[1:] == [
        "--host",
        LOOPBACK,
        "--port",
        str(ops.ports["MONGODB_PORT"]),
        f"--archive={backup / 'mongo.archive.gz'}",
        "--gzip",
        "--drop",
        "--nsFrom",
        "acme_neural.*",
        "--nsTo",
        "acme_neural.*",
    ]
    assert (
        run_restore(
            ops, backup, "mongodb", "--target-suffix", "_copy", "--yes"
        ).returncode
        == 0
    )
    assert ops.stubs.calls("mongorestore")[-1][-1] == "acme_neural_copy.*"

    assert run_restore(ops, backup, "qdrant", "--yes").returncode == 0
    (upload,) = ops.stubs.calls("curl")
    port = ops.ports["QDRANT_PORT"]
    assert (
        f"http://{LOOPBACK}:{port}/collections/acme_memories/snapshots/upload"
        "?priority=snapshot"
    ) in upload
    assert f"snapshot=@{backup / 'qdrant-acme_memories.snapshot'}" in upload
    assert upload[upload.index("-X") + 1] == "POST"

    # No password ever appears on a command line.
    argv = "\n".join(" ".join(call) for call in ops.stubs.calls())
    assert ("pg-password-value-1" in argv) is False

    # A dry run prints the command and calls nothing.
    before = len(ops.stubs.calls())
    planned = run_restore(ops, backup, "postgres", "--dry-run")
    assert planned.returncode == 0
    assert "would run: pg_restore --clean --if-exists --no-owner" in planned.stdout
    assert ("pg-password-value-1" in planned.stdout + planned.stderr) is False
    assert len(ops.stubs.calls()) == before


def test_restore_refuses_a_file_the_manifest_does_not_vouch_for(restore):
    ops, backup = restore
    (backup / "postgres.dump").write_bytes(b"tampered")
    changed = run_restore(ops, backup, "postgres", "--yes")
    assert changed.returncode == 1
    assert "does not match the checksum in the manifest" in changed.stderr
    (backup / "qdrant-acme_memories.snapshot").unlink()
    assert run_restore(ops, backup, "qdrant", "--yes").returncode == 2
    manifest = backup / "MANIFEST.txt"
    kept = [
        line
        for line in manifest.read_text().splitlines()
        if "mongo.archive.gz" not in line
    ]
    manifest.write_text("\n".join(kept) + "\n")
    unlisted = run_restore(ops, backup, "mongodb", "--yes")
    assert unlisted.returncode == 1
    assert "does not list mongo.archive.gz" in unlisted.stderr
    # A tool that fails is reported as a failed restore.
    restored = make_backup(ops, name="20000102T000000Z")
    ops.stubs.add("pg_restore", exit_code=1)
    failed = run_restore(ops, restored, "postgres", "--yes")
    assert failed.returncode == 1
    assert "the postgres restore into acme_consciousness failed" in failed.stderr
    assert [call[0] for call in ops.stubs.calls()] == ["pg_restore"]


def test_redis_restore_prints_manual_steps(restore):
    ops, backup = restore
    result = run_restore(ops, backup, "redis")
    assert result.returncode == 0
    assert "Redis is not restored by this script" in result.stdout
    assert str(backup / "redis.rdb") in result.stdout
    for step in ("stop Redis", "dump.rdb", "mtc db check --only redis"):
        assert step in result.stdout
    # Nothing was asked and nothing was run.
    assert "Type the target name" not in result.stderr
    assert ops.stubs.calls() == []


def test_neo4j_restore_requires_stopped_server(restore):
    ops, backup = restore
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bolt:
        bolt.bind((LOOPBACK, ops.neo4j_port))
        bolt.listen(8)
        running = run_restore(ops, backup, "neo4j", "--yes")
    assert running.returncode == 1
    assert "Neo4j is running. Stop it first" in running.stderr
    assert ops.stubs.calls("neo4j-admin") == []

    stopped = run_restore(ops, backup, "neo4j", "--yes")
    assert stopped.returncode == 0, stopped.stderr
    assert ops.stubs.calls("neo4j-admin") == [
        [
            "neo4j-admin",
            "database",
            "load",
            "--from-stdin",
            "--overwrite-destination=true",
            "neo4j",
        ]
    ]
    # The dump travelled on standard input, unchanged.
    assert (ops.stubs.state / "neo4j-loaded").read_bytes() == FILES["neo4j.dump"]
    # The community edition holds one database: there is no scratch target.
    suffixed = run_restore(ops, backup, "neo4j", "--target-suffix", "_copy", "--yes")
    assert suffixed.returncode == 2
    assert len(ops.stubs.calls("neo4j-admin")) == 1


MONGORESTORE_THEN_INTERRUPT = """
previous=""
for argument in "$@"; do
  if [ "${previous}" = "--config" ]; then cp "${argument}" "${STATE}/mongo-options"; fi
  previous="${argument}"
done
kill -TERM "${PPID}"
sleep 1
exit 1
"""


def test_interrupted_mongorestore_leaves_no_password_file(restore):
    ops, backup = restore
    ops.set(MONGODB_USER="acme_app", MONGODB_PASSWORD="mongo-password-value-3")
    ops.stubs.add("mongorestore", MONGORESTORE_THEN_INTERRUPT)
    scratch = ops.base / "tmp"
    scratch.mkdir()
    result = ops.run(
        "restore.sh",
        "--from",
        str(backup),
        "--store",
        "mongodb",
        "--yes",
        TMPDIR=str(scratch),
    )
    assert result.returncode != 0
    (call,) = ops.stubs.calls("mongorestore")
    options = Path(call[call.index("--config") + 1])
    assert (ops.stubs.state / "mongo-options").is_file()
    assert not options.exists()
    assert list(scratch.iterdir()) == []
