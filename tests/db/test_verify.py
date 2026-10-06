"""The verifier (mtc db check): unreachable stays unreachable, the JSON shape,
the shared exit codes and the quick probe doctor uses."""

import asyncio
import json
import time

import pytest

from mtc.core.checks import CheckResult, CheckStatus
from mtc.core.config import Settings
from mtc.db import postgres_store, redis_store
from mtc.db import verify as verify_module
from mtc.db.verify import probe_store, verify
from tests.db.test_store_checks import FakeRedis


def make_settings(tmp_path, **values) -> Settings:
    return Settings(_env_file=None, PROJECT_ROOT=str(tmp_path), **values)


def test_verify_reports_unreachable_as_unreachable(tmp_path, monkeypatch, capsys):
    async def refused(*args, **kwargs):
        raise ConnectionRefusedError(61, "Connection refused")

    monkeypatch.setattr(postgres_store, "create_postgres_pool", refused)
    settings = make_settings(tmp_path, POSTGRES_ENABLED=True)
    results = asyncio.run(verify(settings))
    by_name = {r.name: r for r in results}
    assert by_name["store:postgres"].status is CheckStatus.UNREACHABLE
    assert by_name["store:postgres"].status is not CheckStatus.OK
    for store in ("mongodb", "qdrant", "neo4j", "redis"):
        assert by_name[f"store:{store}"].status is CheckStatus.SKIPPED
    monkeypatch.setattr(verify_module, "load_settings", lambda env_file: settings)
    assert verify_module.main([]) == 3
    out = capsys.readouterr().out
    assert "[unreachable]" in out and "store:postgres" in out


def test_verify_json_shape(tmp_path, monkeypatch, capsys):
    async def fake_verify(settings=None, *, only=None):
        return [
            CheckResult("postgres:tables", CheckStatus.OK, "all present"),
            CheckResult(
                "redis:maxmemory_policy",
                CheckStatus.INSUFFICIENT_DATA,
                "CONFIG GET was refused",
                required=False,
            ),
            CheckResult("store:neo4j", CheckStatus.SKIPPED, "off", required=False),
        ]

    monkeypatch.setattr(verify_module, "verify", fake_verify)
    monkeypatch.setattr(
        verify_module, "load_settings", lambda env_file: make_settings(tmp_path)
    )
    assert verify_module.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"checks", "summary", "exit_code"}
    assert payload["exit_code"] == 0
    assert [c["name"] for c in payload["checks"]] == [
        "postgres:tables",
        "redis:maxmemory_policy",
        "store:neo4j",
    ]
    for check in payload["checks"]:
        assert set(check) == {"name", "status", "detail", "required", "data"}
    assert payload["summary"]["insufficient_data"] == 1
    assert payload["summary"]["total"] == 3
    assert verify_module.main(["--json", "--strict"]) == 1
    assert json.loads(capsys.readouterr().out)["exit_code"] == 1


@pytest.mark.parametrize(
    "status, required, extra, plain, strict",
    [
        (CheckStatus.OK, True, None, 0, 0),
        (CheckStatus.FAILED, True, None, 1, 1),
        (CheckStatus.UNREACHABLE, True, None, 3, 3),
        (CheckStatus.WARNING, False, None, 0, 1),
        (CheckStatus.INSUFFICIENT_DATA, False, None, 0, 1),
        (CheckStatus.SKIPPED, False, None, 0, 0),
        (CheckStatus.FAILED, True, "postgres", 2, 2),
    ],
)
def test_db_check_exit_codes(
    tmp_path, monkeypatch, capsys, status, required, extra, plain, strict
):
    data = {"missing_extra": extra} if extra else {}

    async def fake_verify(settings=None, *, only=None):
        return [
            CheckResult("store:postgres", status, "x", required=required, data=data)
        ]

    monkeypatch.setattr(verify_module, "verify", fake_verify)
    monkeypatch.setattr(
        verify_module, "load_settings", lambda env_file: make_settings(tmp_path)
    )
    assert verify_module.main([]) == plain
    assert verify_module.main(["--strict"]) == strict
    out = capsys.readouterr().out
    if extra:
        assert "multi-theory-consciousness[postgres]" in out


def test_probe_store_quick_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(
        redis_store, "create_redis_client", lambda *a, **k: FakeRedis(ping_delay=30)
    )
    settings = make_settings(tmp_path, REDIS_ENABLED=True)
    started = time.monotonic()
    result = asyncio.run(probe_store("redis", settings, timeout_s=0.2))
    assert result.status is CheckStatus.UNREACHABLE
    assert time.monotonic() - started < 3.0
    quick = asyncio.run(probe_store("redis", make_settings(tmp_path)))
    assert quick.status is CheckStatus.SKIPPED and not quick.required
    monkeypatch.setattr(redis_store, "create_redis_client", lambda *a, **k: FakeRedis())
    assert asyncio.run(probe_store("redis", settings)).status is CheckStatus.OK
    with pytest.raises(ValueError):
        asyncio.run(probe_store("sqlite", settings))


def test_verify_and_probe_error_details_mask_store_passwords(tmp_path, monkeypatch):
    from mtc.db import mongo_store, neo4j_store, qdrant_store

    secrets = dict(
        POSTGRES_PASSWORD="pgsecretvalue1",
        MONGODB_USER="operator",
        MONGODB_PASSWORD="mongosecretvalue2",
        REDIS_PASSWORD="redissecretvalue3",
        NEO4J_PASSWORD="neosecretvalue4",
    )
    quoted = " ".join(v for k, v in sorted(secrets.items()) if k.endswith("PASSWORD"))

    async def failing(*args, **kwargs):
        raise RuntimeError(f"refused the connection string with {quoted}")

    stores = (postgres_store, mongo_store, qdrant_store, neo4j_store, redis_store)
    for module in stores:
        monkeypatch.setattr(module, "check", failing)
        monkeypatch.setattr(module, "probe", failing)
    settings = make_settings(
        tmp_path,
        POSTGRES_ENABLED=True,
        MONGODB_ENABLED=True,
        QDRANT_ENABLED=True,
        NEO4J_ENABLED=True,
        REDIS_ENABLED=True,
        **secrets,
    )
    results = asyncio.run(verify(settings))
    results += [
        asyncio.run(probe_store(name, settings))
        for name in ("postgres", "mongodb", "qdrant", "neo4j", "redis")
    ]
    assert len(results) == 10
    assert all("refused the connection string" in r.detail for r in results)
    text = json.dumps([r.detail for r in results])
    for key, value in secrets.items():
        if key.endswith("PASSWORD"):
            assert value not in text, key
