"""The Makefile's targets: read, never run."""

import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = ROOT / "Makefile"

REQUIRED_TARGETS = (
    "db-up",
    "db-init",
    "db-verify",
    "db-down",
    "demo",
    "api",
    "test",
    "test-fast",
    "privacy-audit",
    "env-check",
    "generate",
    "lint",
    "hooks",
)
GENERATED_CHECKS = (
    "scripts/gen_env_example.py --check",
    "scripts/gen_schemas.py --check",
    "-m mtc.db.migrations --check",
    "scripts/sync_resources.py --check",
)


def _text() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _recipes() -> Dict[str, List[str]]:
    """Each target's recipe lines, with make variables expanded."""
    text = _text()
    variables = dict(
        (name, value.strip())
        for name, value in re.findall(r"^([A-Z_]+)\s*\??=\s*(.*)$", text, re.MULTILINE)
    )

    def expand(line: str) -> str:
        for _ in range(3):
            line = re.sub(
                r"\$\(([A-Z_]+)\)", lambda m: variables.get(m.group(1), ""), line
            )
        return line

    recipes: Dict[str, List[str]] = {}
    current = None
    for line in text.splitlines():
        header = re.match(r"^([a-z][a-z-]*):", line)
        if header:
            current = header.group(1)
            recipes[current] = []
        elif line.startswith("\t") and current is not None:
            recipes[current].append(expand(line.strip()))
        elif line.strip() and not line.startswith("#"):
            current = None
    return recipes


def test_required_targets_exist():
    recipes = _recipes()
    for target in REQUIRED_TARGETS:
        assert recipes.get(target), f"the {target} target is missing or empty"
    phony = re.search(r"^\.PHONY:(.*)$", _text(), re.MULTILINE).group(1).split()
    assert set(REQUIRED_TARGETS) <= set(phony)
    assert recipes["db-init"] == ["mtc db init"]
    assert recipes["db-verify"] == ["mtc db check"]
    assert recipes["api"] == ["mtc serve"]
    assert any("examples/full_pipeline_demo.py" in line for line in recipes["demo"])
    # env-check runs the four generated-artifact checks; generate rewrites them.
    checks = " ; ".join(recipes["env-check"])
    for check in GENERATED_CHECKS:
        assert check in checks, check
    writes = " ; ".join(recipes["generate"])
    assert "--check" not in writes
    for generator in (
        "scripts/gen_env_example.py",
        "scripts/gen_schemas.py",
        "-m mtc.db.migrations --write",
        "scripts/sync_resources.py",
    ):
        assert generator in writes, generator
    lint = " ; ".join(recipes["lint"])
    assert "black --check --line-length 88 mtc tests scripts examples" in lint
    assert "flake8 mtc tests scripts examples" in lint
    assert "bash -n" in lint and "shellcheck" in lint
    # The fast suite never selects the store tests or a live stack.
    assert any(
        '-m "not slow and not db and not live"' in line for line in recipes["test-fast"]
    )
    for target in ("test", "test-fast"):
        assert "MTC_TEST_STORES" not in " ".join(recipes[target])


def test_db_down_never_removes_volumes():
    recipes = _recipes()
    down = recipes["db-down"]
    assert len(down) == 1
    assert down[0].endswith("docker/docker-compose.yml down")
    whole = _text()
    for forbidden in (
        r"\bdown\b[^\n]*\s(-v|--volumes)\b",
        r"--volumes\b",
        r"\bvolume\s+(rm|prune)\b",
        r"\bsystem\s+prune\b",
        r"--remove-orphans\b",
        r"\brm\s+-[a-z]*r",
    ):
        assert re.search(forbidden, whole) is None, forbidden
    # The file says so where the operator will read it.
    assert "volumes" in whole.lower()


def test_compose_invocation_uses_env_file():
    recipes = _recipes()
    compose = "docker compose --env-file .env -f docker/docker-compose.yml"
    assert recipes["db-up"] == [f"{compose} up -d --wait postgres mongo qdrant"]
    assert recipes["db-down"] == [f"{compose} down"]
    text = _text()
    assert re.search(r"^DB_SERVICES \?= postgres mongo qdrant$", text, re.MULTILINE)
    assert re.search(r"^COMPOSE = " + re.escape(compose) + "$", text, re.MULTILINE)
    # Every Compose call goes through the one definition.
    assert len(re.findall(r"docker compose", text)) == 1
    assert (ROOT / "docker" / "docker-compose.yml").is_file()
