"""The CI workflows: read as data, never run."""

import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
DB_STORES = WORKFLOWS / "db-stores.yml"

REQUIRED_JOBS = (
    "privacy-audit",
    "env-sync",
    "lint",
    "tests",
    "install-smoke",
    "license-audit",
)


def _load(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(document: Dict[str, Any]) -> Dict[str, Any]:
    # YAML reads the bare key "on" as the boolean true.
    value = document.get("on", document.get(True))
    assert value is not None, "the workflow has no trigger"
    return value if isinstance(value, dict) else {name: None for name in value}


def _run_lines(job: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for step in job["steps"]:
        lines.extend(str(step.get("run", "")).splitlines())
    return [line.strip() for line in lines if line.strip()]


def test_jobs_present():
    document = _load(CI)
    jobs = document["jobs"]
    assert set(jobs) == set(REQUIRED_JOBS)
    assert document["permissions"] == {"contents": "read"}
    assert set(_triggers(document)) == {"push", "pull_request"}

    sync = " ; ".join(_run_lines(jobs["env-sync"]))
    for check in (
        "scripts/gen_env_example.py --check",
        "scripts/gen_schemas.py --check",
        "-m mtc.db.migrations --check",
        "scripts/sync_resources.py --check",
    ):
        assert check in sync, check

    lint = " ; ".join(_run_lines(jobs["lint"]))
    assert "black --check --line-length 88 mtc tests scripts examples" in lint
    assert "flake8 mtc tests scripts examples" in lint
    assert "shellcheck" in lint and "bash -n" in lint

    tests = " ; ".join(_run_lines(jobs["tests"]))
    assert 'pip install -e ".[neural,fep,llm,anthropic,api,databases,dev]"' in tests
    assert '-m "not slow and not db and not live"' in tests
    assert "MTC_TEST_STORES" not in tests
    assert jobs["tests"]["strategy"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
        "3.13",
    ]

    licences = _run_lines(jobs["license-audit"])
    joined = " ; ".join(licences)
    assert "scripts/license_audit.py --extras all" in joined
    assert "scripts/license_audit.py --extras iit" in joined
    # The GPL extra is installed and audited on its own, after the rest.
    assert joined.index('".[all]"') < joined.index("--extras all")
    assert joined.index("--extras all") < joined.index('".[iit]"')
    assert joined.index('".[iit]"') < joined.index("--extras iit")
    for name in ("install-smoke", "license-audit"):
        setup = [s for s in jobs[name]["steps"] if "setup-python" in str(s.get("uses"))]
        assert setup[0]["with"]["python-version"] == "3.12", name


def test_install_smoke_is_non_editable_and_audits_runtime_logs():
    job = _load(CI)["jobs"]["install-smoke"]
    lines = _run_lines(job)
    joined = " ; ".join(lines)
    installs = [line for line in lines if line.startswith("pip install")]
    assert len(installs) == 1
    assert '".[all]"' in installs[0]
    assert " -e " not in installs[0] and "--editable" not in installs[0]
    assert "download.pytorch.org/whl/cpu" in installs[0]
    # Everything after the install runs outside the checkout.
    assert 'cd "$RUNNER_TEMP/proj"' in joined
    ordered = [
        "mtc init --print-example-answers > answers.yaml",
        "mtc init --from answers.yaml --yes --no-probe",
        "mtc doctor --json",
        "examples/full_pipeline_demo.py",
        "examples/continuous_thought.py",
        "examples/basic_assessment.py",
        "examples/multi_theory_pipeline.py",
        "examples/neural_substrates.py",
        "examples/single_theory.py",
        "mtc assess --quick --json --json-path results/smoke.json",
        "scripts/privacy_audit.py",
    ]
    positions = [joined.index(command) for command in ordered]
    assert positions == sorted(positions)
    for example in ordered[3:9]:
        assert (ROOT / example).is_file(), example
    audit = [line for line in lines if "scripts/privacy_audit.py" in line]
    assert len(audit) == 1
    assert "--runtime" in audit[0]
    assert "results" in audit[0].split("--runtime", 1)[1]
    # The smoke never talks to a model, a store or an accelerator.
    environment = job["env"]
    assert environment["MTC_DEVICE"] == "cpu"
    assert "MTC_TEST_STORES" not in joined and "mtc serve" not in joined
    assert "mtc db init" not in joined


def test_db_workflow_manual_or_scheduled_only():
    document = _load(DB_STORES)
    triggers = _triggers(document)
    assert set(triggers) == {"workflow_dispatch", "schedule"}
    (schedule,) = triggers["schedule"]
    minute, hour, day, month, weekday = schedule["cron"].split()
    # Once a week, on one weekday.
    assert day == "*" and month == "*" and re.fullmatch(r"[0-6]", weekday)
    assert minute.isdigit() and hour.isdigit()
    assert document["permissions"] == {"contents": "read"}

    (job,) = document["jobs"].values()
    services = job["services"]
    assert set(services) == {"postgres", "mongo", "redis", "qdrant"}
    assert services["postgres"]["image"] == "postgres:17"
    assert services["mongo"]["image"] == "mongo:8"
    assert services["redis"]["image"] == "redis:7"
    assert services["qdrant"]["image"].startswith("qdrant/qdrant:")
    # Neo4j is not in the matrix: its fixtures skip on the community edition.
    assert "neo4j" not in services
    # The container trusts local connections, so no password is written here.
    assert services["postgres"]["env"]["POSTGRES_HOST_AUTH_METHOD"] == "trust"
    lines = _run_lines(job)
    joined = " ; ".join(lines)
    assert "MTC_TEST_STORES=1" in joined
    assert "-m db -rs" in joined
    # The runner waits for the stores before the tests start.
    wait = joined.index("/dev/tcp/")
    assert wait < joined.index("MTC_TEST_STORES=1")
    # The compose tag and the workflow tag for Qdrant are the same one.
    compose = (ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    assert f"image: {services['qdrant']['image']}" in compose


def test_no_password_literals_in_workflows():
    files = sorted(WORKFLOWS.glob("*.yml"))
    assert [path.name for path in files] == ["ci.yml", "db-stores.yml"]
    assignment = re.compile(
        r"(?i)(password|passwd|secret|token|api[_-]?key)[A-Z_a-z]*\s*[:=]\s*(\S+)"
    )
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for match in assignment.finditer(line):
                value = match.group(2)
                # A reference to a repository secret is not a literal.
                assert value.startswith("${{ secrets.") or value.startswith(
                    "${{"
                ), f"{path.name}:{number} assigns a secret-like value"
        text = path.read_text(encoding="utf-8")
        assert "POSTGRES_PASSWORD" not in text
        assert "MONGO_INITDB_ROOT_PASSWORD" not in text
        assert "requirepass" not in text
    # The detector is not vacuous.
    assert assignment.search("POSTGRES_PASSWORD: hunter2")
    assert assignment.search("api_key=abc")
