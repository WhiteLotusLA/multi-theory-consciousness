"""Schema registry: every artifact loads, none is missing, none is unsafe."""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.core import schema_registry
from mtc.core.schema_registry import (
    SCHEMA_ARTIFACTS,
    artifacts_for,
    load_artifact,
    registered_paths,
)

ROOT = Path(__file__).resolve().parents[2]
CLIENT_PACKAGES = ("asyncpg", "motor", "pymongo", "redis", "neo4j", "qdrant_client")
ARTIFACT_NAME = re.compile(
    r"^([A-Z0-9_]+_SCHEMA_SQL|[A-Z0-9_]+_INDEXES|COLLECTION_PARAMS|"
    r"PAYLOAD_INDEXES|CONSTRAINTS|KEY_FAMILIES)$"
)


def test_every_artifact_loads_without_clients():
    code = (
        "import sys\n"
        "from mtc.core.schema_registry import SCHEMA_ARTIFACTS, load_artifact\n"
        "count = 0\n"
        "for paths in SCHEMA_ARTIFACTS.values():\n"
        "    for path in paths:\n"
        "        assert load_artifact(path) is not None, path\n"
        "        count += 1\n"
        f"loaded = [n for n in {CLIENT_PACKAGES!r} if n in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print(count)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
        env={**__import__("os").environ, "PYPHI_WELCOME_OFF": "yes"},
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert int(result.stdout.strip().splitlines()[-1]) == len(registered_paths())


def _module_constants():
    found = set()
    for path in sorted((ROOT / "mtc").rglob("*.py")):
        dotted = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            else:
                continue
            for name in names:
                if ARTIFACT_NAME.match(name):
                    found.add(f"{dotted}.{name}")
    return found


def test_registry_covers_every_artifact():
    found = _module_constants()
    registered = set(registered_paths())
    assert found, "the scan found no artifact constants"
    assert found <= registered, sorted(found - registered)
    # the registered paths exist in the source (the neutral seed list is the
    # one registered constant outside the naming convention)
    assert registered - found == {"mtc.knowledge.graph_manager.SEED_CONCEPTS"}


def test_registry_lists_each_path_once_under_known_stores():
    assert set(SCHEMA_ARTIFACTS) == {"postgres", "mongodb", "qdrant", "neo4j", "redis"}
    paths = registered_paths()
    assert len(paths) == len(set(paths))
    assert all(path.startswith("mtc.") for path in paths)


def test_artifact_shapes():
    postgres = artifacts_for("postgres")
    assert all(
        isinstance(sql, tuple) and sql and all(isinstance(s, str) for s in sql)
        for sql in postgres.values()
    )
    for path, table in artifacts_for("mongodb").items():
        assert isinstance(table, dict) and table, path
        assert all(isinstance(name, str) and specs for name, specs in table.items())
    qdrant = artifacts_for("qdrant")
    params = qdrant["mtc.neural.vectors.coordinator.COLLECTION_PARAMS"]
    assert params["size"] == 384 and params["distance"] == "cosine"
    assert qdrant["mtc.neural.vectors.coordinator.PAYLOAD_INDEXES"]
    neo4j = artifacts_for("neo4j")
    assert all(
        isinstance(c, str) for c in neo4j["mtc.knowledge.graph_manager.CONSTRAINTS"]
    )
    assert artifacts_for("redis")["mtc.memory.redis_manager.KEY_FAMILIES"]


def test_postgres_artifacts_have_no_drop_password_or_seed():
    for path, statements in artifacts_for("postgres").items():
        for sql in statements:
            upper = " ".join(sql.split()).upper()
            assert upper.startswith(
                ("CREATE TABLE IF NOT EXISTS", "CREATE INDEX IF NOT EXISTS")
            ), (path, upper[:60])
            for forbidden in (
                "DROP",
                "PASSWORD",
                "INSERT",
                "EXTENSION",
                "PGVECTOR",
                "COPY ",
                "GRANT",
            ):
                assert forbidden not in upper, (path, forbidden)
            assert not re.search(r"\bVECTOR\s*\(", upper), path


def test_unregistered_paths_are_refused():
    with pytest.raises(ValueError, match="not a registered"):
        load_artifact("os.path")
    with pytest.raises(ValueError, match="not a registered"):
        load_artifact("mtc.core.config.Settings")
    with pytest.raises(ValueError, match="unknown store"):
        artifacts_for("sqlite")


def test_a_registered_path_with_a_missing_constant_names_it(monkeypatch):
    broken = {"postgres": ("mtc.safety.stores.NO_SUCH_SCHEMA_SQL",)}
    monkeypatch.setattr(schema_registry, "SCHEMA_ARTIFACTS", broken)
    with pytest.raises(AttributeError, match="NO_SUCH_SCHEMA_SQL"):
        load_artifact("mtc.safety.stores.NO_SUCH_SCHEMA_SQL")


def test_loaded_artifacts_are_the_module_constants():
    from mtc.monitoring import stores

    assert load_artifact("mtc.monitoring.stores.ALERT_INDEXES") is stores.ALERT_INDEXES
