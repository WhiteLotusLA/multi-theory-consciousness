"""The PostgreSQL migrations are rendered from the schema registry: committed
files equal the rendering, every artifact lands once, statements are
idempotent creates in dependency order, and the check mode reports drift."""

import re
import shutil
from pathlib import Path

import pytest

from mtc.core.schema_registry import SCHEMA_ARTIFACTS, load_artifact
from mtc.db import migrations
from mtc.devtools.privacy_audit import PATTERNS
from mtc.db.migrations import (
    MIGRATION_FILES,
    MIGRATIONS_DIR,
    expected_names,
    render_all,
    render_migration,
    statements_in_order,
)

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / MIGRATIONS_DIR


def _statements(text):
    body = "\n".join(line for line in text.splitlines() if not line.startswith("--"))
    return [part.strip() for part in body.split(";") if part.strip()]


def test_committed_migrations_match_registry():
    rendered = render_all()
    assert list(rendered) == [
        "001_core_schema.sql",
        "002_research_schema.sql",
        "003_safety_schema.sql",
    ]
    committed = sorted(path.name for path in COMMITTED.glob("*.sql"))
    assert committed == sorted(rendered)
    for name, text in rendered.items():
        assert (COMMITTED / name).read_bytes() == text.encode("utf-8"), name


def test_every_postgres_artifact_in_exactly_one_file():
    assigned = [path for _, paths in MIGRATION_FILES for path in paths]
    assert len(assigned) == len(set(assigned))
    assert sorted(assigned) == sorted(SCHEMA_ARTIFACTS["postgres"])
    # Every statement of every artifact appears in exactly one rendered file.
    rendered = render_all()
    for dotted in SCHEMA_ARTIFACTS["postgres"]:
        for statement in load_artifact(dotted):
            normalized = migrations.normalize_statement(statement)
            holders = [name for name, text in rendered.items() if normalized in text]
            assert len(holders) == 1, (dotted, holders)
    assert statements_in_order() == [
        statement for text in rendered.values() for statement in _rendered(text)
    ]


def _rendered(text):
    return [statement + ";" for statement in _statements(text)]


def test_rendered_sql_is_idempotent_creates():
    allowed = re.compile(
        r"^CREATE (TABLE IF NOT EXISTS|INDEX IF NOT EXISTS|UNIQUE INDEX IF NOT "
        r"EXISTS) "
    )
    statements = statements_in_order()
    assert statements
    for statement in statements:
        assert allowed.match(statement), statement[:80]
        assert statement.endswith(";") and not statement.endswith(";;")
        assert statement.count(";") == 1


def test_rendered_sql_has_no_drop_password_seed_or_extension():
    forbidden = re.compile(
        r"\bDROP\b|\bPASSWORD\b|\bINSERT\b|CREATE\s+EXTENSION|pgvector|\bTRUNCATE\b"
        r"|\bALTER\s+ROLE\b|\bVECTOR\s*\(",
        re.IGNORECASE,
    )
    for name, text in render_all().items():
        body = "\n".join(
            line for line in text.splitlines() if not line.startswith("--")
        )
        assert not forbidden.search(body), name


def test_foreign_keys_follow_their_targets():
    statements = statements_in_order()
    created = []
    for statement in statements:
        for target in re.findall(r"REFERENCES\s+(\w+)", statement, re.IGNORECASE):
            assert target.lower() in created, (target, statement[:60])
        table = re.match(r"CREATE TABLE IF NOT EXISTS (\w+)", statement)
        if table:
            created.append(table.group(1).lower())
    assert created.index("conversations") < created.index("messages")
    assert created.index("consciousness_snapshots") < created.index(
        "consciousness_events"
    )


def test_header_names_sources_and_no_dates():
    for name, paths in MIGRATION_FILES:
        text = render_migration(name)
        header = [line for line in text.splitlines() if line.startswith("--")]
        joined = "\n".join(header)
        assert "python3 -m mtc.db.migrations --write" in joined
        for dotted in paths:
            assert dotted in joined
        assert "idempotent" in joined and "safe to" in joined
        assert "PostgreSQL 14 or later" in joined and "no extension" in joined
        assert not re.search(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", text)
        assert not re.search(r"(?i)author|generated on|hostname", text)
        assert not PATTERNS["homepath"].search(text)
        # Statements are separated by one blank line and end with a newline.
        assert text.endswith(";\n")
        assert "\n\n\n" not in text


def test_check_mode_reports_drift(tmp_path, capsys):
    target = tmp_path / "migrations"
    shutil.copytree(COMMITTED, target)
    assert migrations.main(["--check", "--dir", str(target)]) == 0
    edited = target / "002_research_schema.sql"
    edited.write_text(edited.read_text(encoding="utf-8") + "-- edit\n")
    (target / "004_extra.sql").write_text("-- not rendered\n", encoding="utf-8")
    (target / "003_safety_schema.sql").unlink()
    assert migrations.main(["--check", "--dir", str(target)]) == 1
    out = capsys.readouterr().out
    assert "002_research_schema.sql" in out
    assert "003_safety_schema.sql" in out
    assert "004_extra.sql" in out
    # --write repairs the rendered files and leaves the extra file alone.
    assert migrations.main(["--write", "--dir", str(target)]) == 0
    assert migrations.check_dir(target) == ["004_extra.sql"]
    with pytest.raises(SystemExit):
        migrations.main([])


def test_expected_names_cover_every_table_and_index():
    tables, indexes = expected_names()
    text = "\n".join(statements_in_order())
    assert tables == {
        name.lower() for name in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", text)
    }
    assert indexes == {
        name.lower()
        for name in re.findall(r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS (\w+)", text)
    }
    for table in (
        "core_memory",
        "episodic_memory",
        "semantic_memory",
        "conversations",
        "messages",
        "emotional_history",
        "consciousness_snapshots",
        "consciousness_events",
        "safety_audit",
    ):
        assert table in tables, table
    assert "episodic_memory_tsv_idx" in indexes
    assert len(indexes) == text.count("INDEX IF NOT EXISTS")
    assert expected_names(["CREATE UNIQUE INDEX IF NOT EXISTS u_idx ON t (a);"]) == (
        frozenset(),
        frozenset({"u_idx"}),
    )
