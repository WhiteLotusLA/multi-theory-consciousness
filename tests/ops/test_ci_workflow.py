"""The CI workflows: read as data, never run."""

import re
import subprocess
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
    "published-numbers",
)
ENV_FIXTURE = ROOT / "tests" / "fixtures" / "baseline_v050_env.txt"
PUBLISHED_JOB = "published-numbers"
PUBLISHED_RUNNER = "macos-14"
PINNED_PACKAGES = ("numpy", "scipy", "torch", "snntorch", "reservoirpy")
# The tests that hold the published numbers: the legacy preset, both v0.5.0
# fixtures, seeded reproducibility, the exact Phi path and the DCM benchmark.
PUBLISHED_TEST_FILES = (
    "tests/test_legacy_preset.py",
    "tests/test_v050_baseline.py",
    "tests/test_run_assessment.py",
    "tests/test_run_experiment.py",
    "tests/assessment/test_phi_exact_path.py",
    "tests/assessment/test_dcm_benchmark.py",
)
PUBLISHED_COMMAND = (
    "python -m pytest "
    + " ".join(PUBLISHED_TEST_FILES)
    + ' -q -m "not db and not gpu" --timeout=900'
)
PUBLISHED_CONDITION = (
    "github.event_name == 'schedule' || "
    "github.event_name == 'workflow_dispatch' || "
    "(github.event_name == 'pull_request' && github.base_ref == 'main')"
)
SKIP_ON_SCHEDULE = (
    "github.event_name != 'schedule' && github.event_name != 'workflow_dispatch'"
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
    assert set(_triggers(document)) == {
        "push",
        "pull_request",
        "schedule",
        "workflow_dispatch",
    }

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


def _recorded_environment() -> Dict[str, str]:
    """Package name to recorded version, plus the interpreter line."""
    lines = ENV_FIXTURE.read_text(encoding="utf-8").splitlines()
    recorded: Dict[str, str] = {"interpreter": lines[0]}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 2:
            recorded[parts[0]] = parts[1]
    return recorded


def _published_job() -> Dict[str, Any]:
    return _load(CI)["jobs"][PUBLISHED_JOB]


def _pin_step() -> str:
    """The run text of the step that writes the pins file."""
    texts = [
        str(step["run"])
        for step in _published_job()["steps"]
        if "run" in step
        and "pins.txt" in step["run"]
        and "pip install" not in step["run"]
    ]
    assert len(texts) == 1
    return texts[0]


def _run_pin_step(directory: Path, runner_temp: Path) -> "subprocess.CompletedProcess":
    # A shell snippet over a text file: no mtc command, no store, no network.
    return subprocess.run(
        ["bash", "-e", "-c", _pin_step()],
        cwd=directory,
        env={
            "PATH": "/usr/bin:/bin",
            "RUNNER_TEMP": str(runner_temp),
            "MTC_DEVICE": "cpu",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_published_numbers_job_runs_where_the_fixtures_were_recorded():
    job = _published_job()
    interpreter = _recorded_environment()["interpreter"].split()
    # The fixtures were recorded with this interpreter on arm64 macOS.
    assert interpreter[0] == "python" and interpreter[2:] == ["arm64", "Darwin"]
    assert job["runs-on"] == PUBLISHED_RUNNER
    setup = [s for s in job["steps"] if "setup-python" in str(s.get("uses"))]
    major_minor = ".".join(interpreter[1].split(".")[:2])
    assert [step["with"]["python-version"] for step in setup] == [major_minor]
    assert job["env"] == {"MTC_DEVICE": "cpu", "PYPHI_WELCOME_OFF": "yes"}


def test_published_numbers_job_runs_the_slow_and_gated_tests():
    job = _published_job()
    lines = _run_lines(job)
    joined = " ; ".join(lines)
    # Exactly the command the job is for: slow tests included, the stores and
    # the accelerator excluded, never the opt-in store switch.
    assert PUBLISHED_COMMAND in lines
    assert "not slow" not in PUBLISHED_COMMAND
    assert "MTC_TEST_STORES" not in joined and "mtc serve" not in joined
    for name in PUBLISHED_TEST_FILES:
        assert (ROOT / name).is_file(), name
    installs = [line for line in lines if "pip install" in line]
    assert len(installs) == 1
    assert '".[neural,fep,research,iit,dev]"' in installs[0]
    assert '-c "$RUNNER_TEMP/pins.txt"' in installs[0]
    # Only pull requests to main and the nightly run: not every push.
    assert job["if"] == PUBLISHED_CONDITION


def test_published_numbers_pins_are_read_from_the_recorded_environment(tmp_path):
    assert "tests/fixtures/baseline_v050_env.txt" in _pin_step()
    recorded = _recorded_environment()
    done = _run_pin_step(ROOT, tmp_path)
    assert done.returncode == 0, done.stderr
    pins = (tmp_path / "pins.txt").read_text(encoding="utf-8").splitlines()
    assert pins == [f"{name}=={recorded[name]}" for name in PINNED_PACKAGES]


def test_the_pin_step_fails_when_the_record_lacks_a_package(tmp_path):
    project = tmp_path / "project"
    (project / "tests" / "fixtures").mkdir(parents=True)
    kept = [
        line
        for line in ENV_FIXTURE.read_text(encoding="utf-8").splitlines()
        if not line.startswith("torch ")
    ]
    (project / "tests" / "fixtures" / "baseline_v050_env.txt").write_text(
        "\n".join(kept) + "\n", encoding="utf-8"
    )
    assert _run_pin_step(project, tmp_path).returncode != 0


def test_only_the_published_numbers_job_runs_on_the_schedule_or_by_hand():
    document = _load(CI)
    (schedule,) = _triggers(document)["schedule"]
    minute, hour, day, month, weekday = schedule["cron"].split()
    # Nightly: every day of every month.
    assert minute.isdigit() and hour.isdigit()
    assert (day, month, weekday) == ("*", "*", "*")
    for name, job in document["jobs"].items():
        if name == PUBLISHED_JOB:
            continue
        assert job.get("if") == SKIP_ON_SCHEDULE, name
