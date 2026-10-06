"""Bootstrap, verification, the role script and the re-embed backfill against
throwaway stores (marked db; skipped at collection without MTC_TEST_STORES=1).

Every name comes from the throwaway fixtures or ``throwaway_name`` and is
asserted before anything is created or removed. Neo4j is not exercised: the
community edition holds one graph and no test writes to it.
"""

import os
import secrets
import shutil
import subprocess
from pathlib import Path

import pytest

from mtc.core.checks import CheckStatus, exit_code_for
from mtc.core.config import get_settings
from mtc.core.embeddings import HashEmbedder
from mtc.db.bootstrap import bootstrap
from mtc.db.migrations import expected_names
from mtc.db.reembed import reembed
from mtc.db.verify import verify
from mtc.memory.protocols import point_id_for_memory
from tests.conftest import (
    PROBE_TIMEOUT_S,
    assert_throwaway,
    require_opt_in,
    require_package,
    throwaway_name,
)

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "db" / "init_postgres.sh"


async def _database_name(pool):
    return await pool.fetchval("SELECT current_database()")


async def _scoped(tmp_path, pg_pool, mongo_db, qdrant_target, redis_target):
    base = get_settings()
    return base.model_copy(
        update={
            "PROJECT_ROOT": str(tmp_path),
            "PERSONA_PATH": str(tmp_path / "absent-persona.yaml"),
            "POSTGRES_ENABLED": True,
            "MONGODB_ENABLED": True,
            "QDRANT_ENABLED": True,
            "REDIS_ENABLED": True,
            "NEO4J_ENABLED": False,
            "POSTGRES_DB": assert_throwaway(await _database_name(pg_pool)),
            "MONGODB_DB": assert_throwaway(mongo_db.name),
            "QDRANT_COLLECTION": assert_throwaway(qdrant_target.collection),
            "REDIS_KEY_PREFIX": redis_target.prefix,
        }
    )


@pytest.mark.asyncio
async def test_bootstrap_and_verify_on_throwaway_stores(
    tmp_path, throwaway_postgres, throwaway_mongo, throwaway_qdrant, throwaway_redis
):
    settings = await _scoped(
        tmp_path, throwaway_postgres, throwaway_mongo, throwaway_qdrant, throwaway_redis
    )
    outcome = await bootstrap(settings, env_path=tmp_path / "test.env")
    assert exit_code_for(outcome.results) == 0, [
        (r.name, r.status.value, r.detail) for r in outcome.results
    ]
    assert outcome.birth_date_written
    assert outcome.seed is not None and all(outcome.seed.core_blocks.values())
    results = await verify(settings)
    assert exit_code_for(results) == 0, [
        (r.name, r.status.value, r.detail) for r in results
    ]
    statuses = {r.name: r.status for r in results}
    assert statuses["postgres:tables"] is CheckStatus.OK
    assert statuses["postgres:indexes"] is CheckStatus.OK
    assert statuses["mongodb:indexes"] is CheckStatus.OK
    assert statuses["qdrant:collection"] is CheckStatus.OK
    assert statuses["redis:ping"] is CheckStatus.OK
    assert statuses["store:neo4j"] is CheckStatus.SKIPPED
    tables, _ = expected_names()
    found = {
        row[0]
        for row in await throwaway_postgres.fetch(
            "SELECT tablename FROM pg_catalog.pg_tables "
            "WHERE schemaname = current_schema()"
        )
    }
    assert tables <= found


@pytest.mark.asyncio
async def test_second_bootstrap_changes_nothing(
    tmp_path, throwaway_postgres, throwaway_mongo, throwaway_qdrant, throwaway_redis
):
    settings = await _scoped(
        tmp_path, throwaway_postgres, throwaway_mongo, throwaway_qdrant, throwaway_redis
    )
    env = tmp_path / "test.env"
    await bootstrap(settings, env_path=env)
    before = env.read_bytes()
    second = await bootstrap(settings, env_path=env)
    data = {r.name: r.data for r in second.results}
    assert data["store:postgres"]["created_tables"] == []
    assert data["store:postgres"]["created_indexes"] == []
    assert data["store:mongodb"]["created_indexes"] == []
    assert data["store:qdrant"]["created"] is False
    assert data["store:qdrant"]["payload_indexes_created"] == []
    assert not second.birth_date_written
    assert not any(second.seed.core_blocks.values())
    assert env.read_bytes() == before


@pytest.mark.asyncio
async def test_init_postgres_script_creates_throwaway_role_and_database(tmp_path):
    require_opt_in()
    asyncpg = require_package("asyncpg", "postgres")
    if shutil.which("psql") is None:
        pytest.skip("psql is not installed")
    settings = get_settings()
    try:
        admin = await asyncpg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD or None,
            database="postgres",
            timeout=PROBE_TIMEOUT_S,
        )
    except Exception as exc:
        pytest.skip(f"PostgreSQL did not answer within 2 s: {exc}")
    role = assert_throwaway(throwaway_name())
    database = assert_throwaway(throwaway_name())
    password = secrets.token_hex(16)
    env_file = tmp_path / "role.env"
    env_file.write_text(
        f"MTC_POSTGRES_USER={role}\nMTC_POSTGRES_DB={database}\n", encoding="utf-8"
    )
    child_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "MTC_DEVICE": "cpu",
        "MTC_LLM_PROVIDER": "mock",
        "MTC_POSTGRES_PASSWORD": password,
        "MTC_POSTGRES_HOST": settings.POSTGRES_HOST,
        "MTC_POSTGRES_PORT": str(settings.POSTGRES_PORT),
        "PGHOST": settings.POSTGRES_HOST,
        "PGPORT": str(settings.POSTGRES_PORT),
        "PGUSER": settings.POSTGRES_USER,
        "PGDATABASE": "postgres",
    }
    if settings.POSTGRES_PASSWORD:
        child_env["PGPASSWORD"] = settings.POSTGRES_PASSWORD
    try:
        result = subprocess.run(
            ["bash", str(SCRIPT), "--env-file", str(env_file)],
            cwd=tmp_path,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert password not in result.stdout + result.stderr
        owner = await admin.fetchval(
            "SELECT pg_catalog.pg_get_userbyid(datdba) FROM pg_catalog.pg_database "
            "WHERE datname = $1",
            database,
        )
        assert owner == role
        as_role = await asyncpg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=role,
            password=password,
            database=database,
            timeout=PROBE_TIMEOUT_S,
        )
        try:
            tables = {
                row[0]
                for row in await as_role.fetch(
                    "SELECT tablename FROM pg_catalog.pg_tables "
                    "WHERE tableowner = current_user"
                )
            }
        finally:
            await as_role.close()
        assert expected_names()[0] <= tables
        again = subprocess.run(
            ["bash", str(SCRIPT), "--env-file", str(env_file)],
            cwd=tmp_path,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert again.returncode == 0, again.stderr[-2000:]
        assert "already exists" in again.stdout
    finally:
        assert_throwaway(database)
        await admin.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
        assert_throwaway(role)
        await admin.execute(f'DROP ROLE IF EXISTS "{role}"')
        await admin.close()


@pytest.mark.asyncio
async def test_reembed_backfills_missing_points(
    tmp_path, throwaway_postgres, throwaway_qdrant
):
    from mtc.db.postgres_store import apply_migrations

    base = get_settings()
    settings = base.model_copy(
        update={
            "PROJECT_ROOT": str(tmp_path),
            "POSTGRES_ENABLED": True,
            "QDRANT_ENABLED": True,
            "POSTGRES_DB": assert_throwaway(await _database_name(throwaway_postgres)),
            "QDRANT_COLLECTION": assert_throwaway(throwaway_qdrant.collection),
        }
    )
    await apply_migrations(throwaway_postgres)
    ids = []
    for index in range(5):
        memory_type = "archived" if index == 2 else "episodic"
        ids.append(
            await throwaway_postgres.fetchval(
                "INSERT INTO episodic_memory (content, memory_type, importance) "
                "VALUES ($1, $2, $3) RETURNING id",
                f"synthetic memory {index}",
                memory_type,
                5.0,
            )
        )
    embedder = HashEmbedder(settings.VECTOR_DIMENSIONS)
    dry = await reembed(
        settings, dry_run=True, embedder=embedder, allow_non_semantic=True
    )
    assert (dry.scanned, dry.missing, dry.embedded) == (5, 4, 0)
    first = await reembed(settings, embedder=embedder, allow_non_semantic=True)
    assert first.errors == {} and first.embedded == 4 and first.skipped_archived == 1
    points = throwaway_qdrant.client.retrieve(
        collection_name=throwaway_qdrant.collection,
        ids=[point_id_for_memory(i) for i in ids],
        with_payload=True,
    )
    assert {p.payload["postgres_id"] for p in points} == {
        ids[0],
        ids[1],
        ids[3],
        ids[4],
    }
    second = await reembed(settings, embedder=embedder, allow_non_semantic=True)
    assert second.missing == 0 and second.embedded == 0
