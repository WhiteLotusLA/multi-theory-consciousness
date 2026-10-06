"""The setup documents are honest about what was run.

- Every user-facing document that mentions Docker Compose says the Compose
  file was reviewed and not run on the development machine, and names the
  path that was run there.
- docs/SETUP_MACOS.md gives each step exactly one of three statuses:
  ``tested``, ``reviewed-not-run`` or ``pending``. ``tested`` is accepted only
  for a step the fresh-machine verification ran, so until its report is in,
  the document says the verification is pending; GETTING_STARTED's
  verification note says the same.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# docs/CONFIGURATION.md is generated from the Settings field descriptions, a
# reference of keys rather than a setup guide, so it is not held to the note.
GENERATED = {ROOT / "docs" / "CONFIGURATION.md"}
DOCUMENTS = sorted(
    {ROOT / "README.md", ROOT / "docker" / "README.md"}
    | set((ROOT / "docs").glob("*.md")) - GENERATED
)
COMPOSE = re.compile(r"\bCompose\b|docker compose|docker-compose\.yml|make db-up")
REVIEWED = re.compile(
    r"reviewed\b.{0,200}?\bnot run on the development machine", re.IGNORECASE | re.S
)
TESTED_PATH = re.compile(
    r"path (?:run|tested) (?:on the development machine|there) is\s+the in-memory"
    r"\s+install with the mock provider",
    re.IGNORECASE | re.S,
)
STATUSES = ("tested", "reviewed-not-run", "pending")
REQUIRED_STEPS = (
    "mtc init",
    "mtc doctor",
    "the full pipeline demo",
    "mtc assess",
    "init_postgres.sh",
    "start_stack.sh",
)


def _mentions_compose(path: Path) -> bool:
    return bool(COMPOSE.search(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    "path",
    [p for p in DOCUMENTS if _mentions_compose(p)],
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_compose_docs_say_reviewed_not_run(path):
    text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))
    assert REVIEWED.search(text), f"{path.name} mentions Compose without the note"
    assert TESTED_PATH.search(text), f"{path.name} does not name the tested path"


def test_compose_is_mentioned_where_expected():
    names = {p.name for p in DOCUMENTS if _mentions_compose(p)}
    assert {"SETUP_DOCKER.md", "GETTING_STARTED.md"} <= names


def _step_rows(text: str):
    section = text.split("## What was tested", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] not in ("Step",) and set(cells[0]) - {"-"}:
            rows.append((cells[0], cells[1]))
    return section, rows


def test_setup_macos_names_what_was_tested():
    text = (ROOT / "docs" / "SETUP_MACOS.md").read_text(encoding="utf-8")
    section, rows = _step_rows(text)
    assert rows, "SETUP_MACOS.md has no step table"
    for step, status in rows:
        assert status in STATUSES, f"{step}: {status!r}"
    joined = " ".join(step for step, _ in rows)
    for required in REQUIRED_STEPS:
        assert required in joined, f"no status for {required}"
    statuses = {status for _, status in rows}
    getting_started = (ROOT / "docs" / "GETTING_STARTED.md").read_text(encoding="utf-8")
    note = getting_started.split("## Verification", 1)[1].split("\n## ", 1)[0]
    if "pending" in statuses:
        # Until the report is in, both documents say so plainly.
        assert re.search(r"verification[^.]*pending", section, re.I | re.S)
        assert re.search(r"pending", note, re.I)
    if "tested" in statuses:
        # A tested step names when and on what it was verified.
        assert re.search(r"\b\d{4}-\d{2}-\d{2}\b", section)
        assert re.search(r"Python 3\.\d+", section)


def test_generated_configuration_notes_compose_was_reviewed_not_run():
    """CONFIGURATION.md is generated from the Settings field descriptions, so
    the note lives in the description: one that names Docker or Compose says
    the Compose file was reviewed and not run on the development machine."""
    from mtc.core.config import Settings

    mentions = {
        name: re.sub(r"\s+", " ", info.description or "")
        for name, info in Settings.model_fields.items()
        if re.search(r"compose|docker", info.description or "", re.IGNORECASE)
    }
    assert "SERVICE_MANAGER" in mentions
    for name, text in mentions.items():
        assert REVIEWED.search(text), f"{name} names Compose without the note"
    generated = re.sub(
        r"\s+", " ", (ROOT / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")
    )
    for line in generated.split("| `MTC_")[1:]:
        if re.search(r"compose|docker", line, re.IGNORECASE):
            assert REVIEWED.search(line), line[:80]


def test_getting_started_tests_install_matches_ci():
    """The Tests section installs the extras the suite needs, the same ones the
    CI tests job installs: core plus dev cannot build the workspace (it needs
    fep) and many tests need neural, api and llm."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    tests_job = ci.split("\n  tests:", 1)[1].split("\n  install-smoke:", 1)[0]
    (ci_extras,) = re.findall(r'pip install -e "\.\[([a-z,]+)\]"', tests_job)
    text = (ROOT / "docs" / "GETTING_STARTED.md").read_text(encoding="utf-8")
    section = text.split("## Tests", 1)[1].split("\n## ", 1)[0]
    installs = re.findall(r'pip install -e "\.\[([a-z,]+)\]"', section)
    assert installs, "the Tests section installs nothing"
    for extras in installs:
        assert set(extras.split(",")) == set(ci_extras.split(",")), extras


def test_setup_macos_matches_what_the_wizard_writes():
    """The wizard generates MTC_NEO4J_PASSWORD, so the guide sets Neo4j's
    initial password to that value rather than to a new one; and a MongoDB
    user the wizard configured is created by nothing, so the guide says how
    to create it."""
    text = re.sub(
        r"\s+", " ", (ROOT / "docs" / "SETUP_MACOS.md").read_text(encoding="utf-8")
    )
    neo4j = text.split("**Neo4j.**", 1)[1].split("**", 1)[0]
    assert "the value the wizard generated" in neo4j
    assert "MTC_NEO4J_PASSWORD" in neo4j
    assert "nothing creates the MongoDB user" in text
    assert "db.createUser" in text
