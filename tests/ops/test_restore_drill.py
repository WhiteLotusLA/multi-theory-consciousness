"""restore_drill.sh: it creates and drops scratch targets only.

The hermetic tests run it against recording stubs. The one real round trip is
marked ``db`` and runs only on the throwaway fixtures (never by default).
"""

import os
import shutil
import subprocess
import sys

import pytest

from .conftest import OPS_DIR, ROOT, Ops, enable_all_stores, install_store_stubs
from .test_restore import FILES, make_backup

pytestmark = pytest.mark.slow

SCRATCH = "_restore_test"
LIVE = {
    "postgres": "acme_consciousness",
    "mongodb": "acme_neural",
    "qdrant": "acme_memories",
}


async def drop_drill_scratch(connection, scratch: str) -> None:
    """Drop a drill's scratch database, guarded: the name must be a throwaway
    name (the conftest pattern) followed by the drill's suffix, or nothing is
    sent. The round trip calls it in a finally, so an interrupted drill does
    not leave its scratch database on the server."""
    from tests.conftest import assert_throwaway

    assert scratch.endswith(SCRATCH), scratch
    assert_throwaway(scratch[: -len(SCRATCH)])
    await connection.execute(f'DROP DATABASE IF EXISTS "{scratch}" WITH (FORCE)')


@pytest.fixture
def drill(ops):
    install_store_stubs(ops)
    enable_all_stores(ops)
    ops.set(POSTGRES_PASSWORD="pg-password-value-1")
    files = {name: data for name, data in FILES.items() if "neo4j" not in name}
    return ops, make_backup(ops, files)


def created_and_dropped(ops: Ops) -> dict:
    """Every name a tool was asked to create, restore into or drop."""
    names = {"created": [], "restored": [], "dropped": []}
    for call in ops.stubs.calls("createdb"):
        names["created"].append(call[-1])
    for call in ops.stubs.calls("dropdb"):
        names["dropped"].append(call[-1])
    for call in ops.stubs.calls("pg_restore"):
        names["restored"].append(call[call.index("-d") + 1])
    for call in ops.stubs.calls("mongorestore"):
        names["restored"].append(call[call.index("--nsTo") + 1].split(".")[0])
    for call in ops.stubs.calls("mongosh"):
        if any("dropDatabase" in part for part in call):
            names["dropped"].append(call[call.index("--eval") - 1])
    for call in ops.stubs.calls("curl"):
        url = next(part for part in call if part.startswith("http"))
        collection = url.split("/collections/")[1].split("/")[0].split("?")[0]
        method = call[call.index("-X") + 1] if "-X" in call else "GET"
        if method == "POST":
            names["restored"].append(collection)
        elif method == "DELETE":
            names["dropped"].append(collection)
    return names


def test_drill_targets_end_with_restore_test(drill):
    ops, backup = drill
    result = ops.run("restore_drill.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"drilling {backup.name}" in result.stdout
    assert "drill passed" in result.stdout
    names = created_and_dropped(ops)
    scratch = {f"{live}{SCRATCH}" for live in LIVE.values()}
    assert names["created"] == [LIVE["postgres"] + SCRATCH]
    assert set(names["restored"]) == scratch
    assert set(names["dropped"]) == scratch
    for group in names.values():
        for name in group:
            assert name.endswith(SCRATCH), name
    # The scratch database is dropped before it is created and again after.
    order = [call[0] for call in ops.stubs.calls() if call[0] in ("dropdb", "createdb")]
    assert order == ["dropdb", "createdb", "dropdb"]
    for line in (
        "postgres: verified, 42 rows match the manifest",
        "mongodb: verified, 7 documents match the manifest",
        "qdrant: verified, 5 points match the manifest",
    ):
        assert line in result.stdout
    # The password went to the PostgreSQL tools in their own variable.
    for tool in ("dropdb", "createdb", "pg_restore", "psql"):
        seen = ops.stubs.environments(tool)
        assert seen, tool
        assert all((env["PGPASSWORD"] == "pg-password-value-1") is True for env in seen)
    argv = "\n".join(" ".join(call) for call in ops.stubs.calls())
    assert ("pg-password-value-1" in argv) is False

    # --from picks a backup; a dry run touches nothing.
    before = len(ops.stubs.calls())
    planned = ops.run("restore_drill.sh", "--from", str(backup), "--dry-run")
    assert planned.returncode == 0
    assert "nothing was created or dropped" in planned.stdout
    assert f"createdb {LIVE['postgres']}{SCRATCH}" in planned.stdout
    assert len(ops.stubs.calls()) == before


def test_drill_refuses_non_scratch_names(drill):
    ops, _backup = drill
    live = LIVE["postgres"]
    refused = (
        (live, live),  # the live name itself
        ("", live),  # nothing
        (f"{live}_copy", live),  # another suffix
        (f"other{SCRATCH}", live),  # a scratch name of something else
        (SCRATCH, ""),  # no live name to derive from
        (f"{live}{SCRATCH}x", live),  # the suffix is not at the end
    )
    for name, base in refused:
        outcome = ops.call("restore_drill.sh", f'assert_scratch "{name}" "{base}"')
        assert outcome.returncode != 0, (name, base)
        assert "refusing" in outcome.stderr
    accepted = ops.call(
        "restore_drill.sh", f'assert_scratch "{live}{SCRATCH}" "{live}"'
    )
    assert accepted.returncode == 0, accepted.stderr

    # Each drop and each create asserts first: handed a live name, no tool runs.
    for function, base in (
        ("drop_postgres_scratch", LIVE["postgres"]),
        ("create_postgres_scratch", LIVE["postgres"]),
        ("drop_mongo_scratch", LIVE["mongodb"]),
        ("drop_qdrant_scratch", LIVE["qdrant"]),
    ):
        outcome = ops.call("restore_drill.sh", f'{function} "{base}" "{base}"')
        assert outcome.returncode != 0, function
    assert ops.stubs.calls() == []
    # Handed the scratch name, they do run.
    done = ops.call(
        "restore_drill.sh",
        # What main reads from Settings before it calls these functions.
        'MTC_POSTGRES_USER=acme; MTC_POSTGRES_PASSWORD=""; '
        f'drop_postgres_scratch "{live}{SCRATCH}" "{live}" '
        f'&& drop_qdrant_scratch "{LIVE["qdrant"]}{SCRATCH}" "{LIVE["qdrant"]}"',
    )
    assert done.returncode == 0, done.stderr
    assert [call[0] for call in ops.stubs.calls()] == ["dropdb", "curl"]


def test_drill_drops_only_scratch_targets(drill):
    ops, _backup = drill
    # A restore that fails still drops its scratch target, and only that.
    ops.stubs.add("pg_restore", exit_code=1)
    ops.stubs.add("mongorestore", exit_code=1)
    result = ops.run("restore_drill.sh")
    assert result.returncode == 1
    assert "postgres: the restore into acme_consciousness_restore_test failed" in (
        result.stderr
    )
    assert "drill failed" in result.stderr
    names = created_and_dropped(ops)
    assert names["dropped"], "the scratch targets must be cleaned up"
    live_names = set(LIVE.values())
    for name in names["created"] + names["dropped"] + names["restored"]:
        assert name not in live_names, name
        assert name.endswith(SCRATCH), name
    # No tool was ever pointed at a live database or collection to change it.
    for call in ops.stubs.calls():
        if call[0] in ("dropdb", "createdb"):
            assert call[-1] not in live_names
        if call[0] == "curl" and "-X" in call:
            url = next(part for part in call if part.startswith("http"))
            assert f"/collections/{LIVE['qdrant']}{SCRATCH}" in url
    # Reading from the live side happens nowhere: counts come from the scratch
    # copies and from the manifest.
    for call in ops.stubs.calls("psql"):
        assert call[call.index("-d") + 1].endswith(SCRATCH)


def test_counts_compared_with_manifest(drill):
    ops, backup = drill
    # The restored copy holds fewer rows than the manifest recorded.
    (ops.stubs.state / "rows-messages").write_text("29\n")
    short = ops.run("restore_drill.sh")
    assert short.returncode == 1
    assert (
        "postgres: MISMATCH: the manifest records 42 rows, the restored copy holds 41"
        in short.stderr
    )
    assert "mongodb: verified" in short.stdout and "qdrant: verified" in short.stdout
    (ops.stubs.state / "rows-messages").unlink()

    (ops.stubs.state / "points").write_text("6\n")
    more = ops.run("restore_drill.sh")
    assert more.returncode == 1
    assert "qdrant: MISMATCH" in more.stderr
    (ops.stubs.state / "points").unlink()

    # A count the manifest does not know cannot be called verified.
    manifest = backup / "MANIFEST.txt"
    manifest.write_text(
        manifest.read_text().replace(
            "count mongodb documents 7", "count mongodb documents unknown"
        )
    )
    unknown = ops.run("restore_drill.sh")
    assert unknown.returncode == 1
    assert "mongodb: restored, but not verified" in unknown.stderr
    # Every scratch target was still dropped on every run.
    names = created_and_dropped(ops)
    assert names["dropped"].count(LIVE["postgres"] + SCRATCH) == 6
    assert all(name.endswith(SCRATCH) for name in names["dropped"])

    # Nothing to drill is not a pass.
    empty = ops.project / "elsewhere"
    assert ops.run("restore_drill.sh", "--from", str(empty)).returncode == 2
    assert ops.run("restore_drill.sh", MTC_BACKUP_LOCATION="none-here").returncode == 2


def test_drill_with_mongo_credentials_says_what_it_cannot_do(drill):
    ops, _backup = drill
    ops.set(MONGODB_USER="acme_app", MONGODB_PASSWORD="mongo-password-value-1")
    result = ops.run("restore_drill.sh")
    assert result.returncode == 1
    assert "mongodb: not drilled: credentials are configured" in result.stderr
    assert ops.stubs.calls("mongorestore") == [] and ops.stubs.calls("mongosh") == []
    assert "postgres: verified" in result.stdout
    assert ("mongo-password-value-1" in result.stdout + result.stderr) is False


@pytest.mark.db
@pytest.mark.asyncio
async def test_backup_and_drill_round_trip_on_throwaway_postgres(
    throwaway_postgres, tmp_path
):
    """A real dump and a real drill, on a throwaway database only."""
    for tool in ("pg_dump", "pg_restore", "psql", "createdb", "dropdb"):
        if shutil.which(tool) is None:
            pytest.skip(
                f"{tool} is not installed; the round trip needs the client tools"
            )
    from mtc.core.config import get_settings

    pool = throwaway_postgres
    async with pool.acquire() as connection:
        server_major = (
            int(await connection.fetchval("SHOW server_version_num")) // 10000
        )
    reported = subprocess.run(
        ["pg_dump", "--version"], capture_output=True, text=True, timeout=30
    ).stdout
    digits = "".join(ch if ch.isdigit() else " " for ch in reported).split()
    client_major = int(digits[0]) if digits else 0
    if client_major < server_major:
        pytest.skip(
            f"pg_dump {client_major} cannot dump a PostgreSQL {server_major} server; "
            "install client tools at least as new as the server"
        )
    async with pool.acquire() as connection:
        database = await connection.fetchval("SELECT current_database()")
        await connection.execute("CREATE TABLE drill_rows (id int primary key)")
        await connection.execute("INSERT INTO drill_rows SELECT generate_series(1, 25)")
    prefix = os.environ.get("MTC_TEST_DB_PREFIX", "mtc_test")
    assert database.startswith(prefix + "_"), database
    scratch = f"{database}{SCRATCH}"
    settings = get_settings()
    project = tmp_path / "project"
    project.mkdir()
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PATH": os.pathsep.join([os.path.dirname(sys.executable), os.environ["PATH"]]),
        "MTC_DEVICE": "cpu",
        "MTC_PROJECT_ROOT": str(project),
        "MTC_POSTGRES_ENABLED": "true",
        "MTC_MONGODB_ENABLED": "false",
        "MTC_QDRANT_ENABLED": "false",
        "MTC_REDIS_ENABLED": "false",
        "MTC_NEO4J_ENABLED": "false",
        "MTC_POSTGRES_DB": database,
        "MTC_POSTGRES_USER": settings.POSTGRES_USER,
        "MTC_POSTGRES_PASSWORD": settings.POSTGRES_PASSWORD,
        "MTC_POSTGRES_HOST": settings.POSTGRES_HOST,
        "MTC_POSTGRES_PORT": str(settings.POSTGRES_PORT),
    }

    def run(script: str, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["/bin/bash", str(OPS_DIR / script), *arguments],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

    try:
        backed_up = run("backup.sh", "--no-state")
        assert backed_up.returncode == 0, backed_up.stderr
        drilled = run("restore_drill.sh")
        assert drilled.returncode == 0, drilled.stdout + drilled.stderr
        assert "postgres: verified, 25 rows match the manifest" in drilled.stdout
        async with pool.acquire() as connection:
            left = await connection.fetchval(
                "SELECT count(*) FROM pg_database WHERE datname = $1", scratch
            )
            rows = await connection.fetchval("SELECT count(*) FROM drill_rows")
        # The scratch database is gone and the source is untouched.
        assert left == 0
        assert rows == 25
    finally:
        # An interrupted or failed drill can leave its scratch database; the
        # guarded cleanup drops it (a no-op when the drill already did).
        async with pool.acquire() as connection:
            await drop_drill_scratch(connection, scratch)


class _RecordingConnection:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)


def test_scratch_cleanup_drops_only_throwaway_restore_test_names():
    """The round trip's guarded cleanup: it drops a drill's scratch database
    only when the name is a throwaway name plus the drill's suffix, and it
    refuses every other name before any statement is sent."""
    import asyncio

    from tests.conftest import throwaway_name

    good = throwaway_name() + SCRATCH
    connection = _RecordingConnection()
    asyncio.run(drop_drill_scratch(connection, good))
    assert connection.statements == [f'DROP DATABASE IF EXISTS "{good}" WITH (FORCE)']
    refused = (
        LIVE["postgres"] + SCRATCH,  # a live name with the suffix
        throwaway_name(),  # a throwaway name without the suffix
        throwaway_name() + SCRATCH + "x",
        "postgres",
        throwaway_name() + '"; DROP DATABASE postgres; --' + SCRATCH,
    )
    for name in refused:
        quiet = _RecordingConnection()
        with pytest.raises(AssertionError):
            asyncio.run(drop_drill_scratch(quiet, name))
        assert quiet.statements == [], name
