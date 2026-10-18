"""The release records say what was decided, and retired files stay gone.

- docs/MIGRATION_FROM_V04.md records results, not open questions: it holds
  neither the words "decision pending" nor "a report, not a result".
- experiments/program.md, the research-loop notes the release removed, does
  not come back.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "docs" / "MIGRATION_FROM_V04.md"


def test_migration_record_holds_no_pending_decision():
    text = " ".join(MIGRATION.read_text(encoding="utf-8").split()).lower()
    for phrase in ("decision pending", "a report, not a result"):
        assert phrase not in text, f"MIGRATION_FROM_V04.md still says {phrase!r}"


def test_experiments_program_file_does_not_exist():
    assert not (ROOT / "experiments" / "program.md").exists()
