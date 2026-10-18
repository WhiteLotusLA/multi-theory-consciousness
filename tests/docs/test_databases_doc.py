"""docs/DATABASES.md names every artifact the schema registry defines.

Every expected name comes from the registry at test time: the PostgreSQL
tables and indexes the rendered migrations create, every MongoDB collection
with its index keys and uniqueness, the Qdrant collection parameters and
payload indexes, the Neo4j constraints and seed concepts, and the Redis key
families with their types and expiries.
"""

import re
from pathlib import Path

from mtc.core.schema_registry import artifacts_for
from mtc.db.migrations import MIGRATION_FILES, expected_names

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "DATABASES.md"


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def _row(text: str, first: str):
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) > 1 and first in cells[:2]:
            return cells
    return None


def _render(spec) -> str:
    parts = []
    hashed = False
    for key, direction in spec.keys:
        parts.append(f"`{key}`")
        hashed = hashed or direction == "hashed"
    rendered = " + ".join(parts)
    if hashed:
        rendered += " (hashed)"
    if spec.unique:
        rendered += " (unique)"
    return rendered


def test_databases_doc_names_every_artifact():
    text = _text()

    # PostgreSQL: every migration file, table and index.
    tables, indexes = expected_names()
    assert tables and indexes
    for filename, _ in MIGRATION_FILES:
        assert f"`{filename}`" in text, filename
    for name in sorted(tables | indexes):
        assert f"`{name}`" in text, f"PostgreSQL artifact {name} not documented"

    # MongoDB: each collection's row lists exactly its indexes.
    mongo_section = text.split("\n## MongoDB", 1)[1].split("\n## ", 1)[0]
    for path, table in artifacts_for("mongodb").items():
        assert f"`{path}`" in mongo_section, path
        for collection, specs in table.items():
            row = _row(mongo_section, f"`{collection}`")
            assert row is not None, f"collection {collection} has no row"
            documented = {item.strip() for item in row[-1].split(";")}
            assert documented == {_render(spec) for spec in specs}, collection

    # Qdrant: the collection parameters and the payload indexes.
    qdrant = artifacts_for("qdrant")
    params = next(v for k, v in qdrant.items() if k.endswith("COLLECTION_PARAMS"))
    payload = next(v for k, v in qdrant.items() if k.endswith("PAYLOAD_INDEXES"))
    assert _row(text, "vector size")[1].startswith(str(params["size"]))
    assert _row(text, "distance")[1] == str(params["distance"])
    for key, value in params["hnsw_config"].items():
        row = _row(text, f"HNSW `{key}`")
        assert row is not None, key
        assert row[1] == str(value).lower(), key
    for field, kind in payload:
        assert f"`{field}` ({kind})" in text, field

    # Neo4j: every constraint name and seed concept.
    neo4j = artifacts_for("neo4j")
    for path, value in neo4j.items():
        if path.endswith("CONSTRAINTS"):
            for statement in value:
                name = re.match(r"CREATE (?:CONSTRAINT|INDEX) (\w+)", statement)
                assert f"`{name.group(1)}`" in text, statement
        else:
            for concept in value:
                assert f"`{concept}`" in text, concept

    # Redis: each key family's row with its type and expiry.
    for families in artifacts_for("redis").values():
        for family in families:
            row = _row(text, f"`{family.pattern}`")
            assert row is not None, family.pattern
            assert row[1] == family.redis_type, family.pattern
            if family.ttl_seconds is None:
                assert not re.match(r"^\d", row[2]), family.pattern
            else:
                assert row[2].startswith(f"{family.ttl_seconds} seconds"), family


def _flat() -> str:
    return re.sub(r"\s+", " ", _text().replace("**", "")).strip()


def test_databases_doc_states_what_the_simple_consolidation_pass_writes():
    """The simple pass writes the group-size strength of each semantic row,
    and leaves only the confidence and the consolidation strength unwritten."""
    text = _flat()
    assert "writes no strength and no confidence" not in text
    assert (
        "the simple consolidation pass writes the group-size strength, no "
        "confidence and no consolidation strength" in text
    )


def test_databases_doc_states_the_pass_level_delete_confirmation():
    """The pruner's index deletes wait for at least one confirmed row of the
    pass, not for a confirmation of each row."""
    text = _flat()
    assert "confirmed the row's delete" not in text
    assert (
        "the vector points and memory nodes of the pass's deleted rows are "
        "deleted only after the store confirmed at least one row of the pass "
        "deleted; a delete that confirms none keeps them and is counted as an "
        "error" in text
    )
