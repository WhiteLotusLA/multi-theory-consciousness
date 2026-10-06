"""The assessment runner lives in mtc.cli.assess: the script is a shim, the
default results directory is ASSESSMENT_OUTPUT_DIR under PROJECT_ROOT, presets
and the schema resolve through the shipped resources, and both entry points
give the same report for the same seed."""

import ast
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.cli import assess
from mtc.core import config as config_module
from mtc.core.config import Settings
from mtc.core.param_overrides import load_overrides
from mtc.core.resources import packaged_path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_assessment.py"


def test_runner_script_is_a_shim():
    text = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text)
    docstring = ast.get_docstring(tree)
    for flag in ("--quick", "--full", "--report", "--ablation", "--json", "--seed"):
        assert flag in docstring, flag
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert [f.name for f in functions] == ["__getattr__"]
    assert "from mtc.cli.assess import main" in text
    assert "raise SystemExit(main())" in text
    assert len(text.splitlines()) < 80
    # Names a caller reads from the script come from the package module.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_shim_run_assessment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main is assess.main
    assert module._seed_everything is assess._seed_everything
    assert module.warmup is assess.warmup


def test_results_default_under_assessment_output_dir(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        PROJECT_ROOT=str(tmp_path / "project"),
        ASSESSMENT_OUTPUT_DIR="out/assessments",
    )
    monkeypatch.setattr(config_module, "_settings", settings)
    monkeypatch.chdir(tmp_path)

    class Report:
        def to_dict(self):
            return {"overall_score": 0.5}

    async def fake_standard(*args, **kwargs):
        return Report()

    monkeypatch.setattr(assess, "run_standard", fake_standard)
    monkeypatch.setattr(assess, "print_report", lambda report, markdown=False: None)
    monkeypatch.setattr(
        assess, "_build_standalone_modules", lambda overrides: (None,) * 5
    )
    monkeypatch.setattr(assess, "build_cycle_augmenter", lambda *a, **k: None)
    assert assess.main(["--json"]) == 0
    written = list((tmp_path / "project" / "out" / "assessments").glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text()) == {"overall_score": 0.5}
    assert not (tmp_path / "results").exists()
    assert Settings.model_fields["ASSESSMENT_OUTPUT_DIR"].default == (
        "results/consciousness_assessments"
    )


def test_presets_and_schema_resolve_through_resources(tmp_path, monkeypatch, capsys):
    settings = Settings(_env_file=None, PROJECT_ROOT=str(tmp_path))
    monkeypatch.setattr(config_module, "_settings", settings)
    overrides = assess._load_overrides(None, "legacy_v04")
    expected = load_overrides(
        str(packaged_path("experiments/presets/legacy_v04.toml")),
        str(packaged_path("experiments/param_schema.toml")),
    )
    assert overrides == expected and overrides
    assert not hasattr(assess, "SCHEMA_PATH") and not hasattr(assess, "PRESETS_DIR")
    assert assess.SCHEMA_SOURCE == "experiments/param_schema.toml"
    assert assess.main(["--preset", "no_such_preset"]) == 1
    err = capsys.readouterr().err
    assert "PARAM ERROR" in err and "no_such_preset" in err


def _record_run(monkeypatch, tmp_path, argv):
    """Run main(argv) through the real parser with recording fakes."""
    settings = Settings(_env_file=None, PROJECT_ROOT=str(tmp_path))
    monkeypatch.setattr(config_module, "_settings", settings)
    calls = {"warmup": 0, "extra_modules": None}

    class Workspace:
        phi_tracker = object()

    class Report:
        def to_dict(self):
            return {"overall_score": 0.25}

    async def fake_warmup(*args, **kwargs):
        calls["warmup"] += 1

    async def fake_standard(*args, **kwargs):
        calls["extra_modules"] = dict(kwargs.get("extra_modules") or {})
        return Report()

    monkeypatch.setattr(assess, "warmup", fake_warmup)
    monkeypatch.setattr(assess, "run_standard", fake_standard)
    monkeypatch.setattr(assess, "print_report", lambda report, markdown=False: None)
    monkeypatch.setattr(
        assess,
        "_build_standalone_modules",
        lambda overrides: (Workspace(), None, None, None, None),
    )
    monkeypatch.setattr(assess, "build_cycle_augmenter", lambda *a, **k: None)
    out = tmp_path / "report.json"
    code = assess.main(list(argv) + ["--json-path", str(out)])
    return code, calls, json.loads(out.read_text())


def test_quick_flag_is_the_default_mode(tmp_path, monkeypatch, capsys):
    parsed = assess.build_parser().parse_args(["--quick"])
    assert parsed.quick is True and parsed.full is False
    assert assess.build_parser().parse_args([]).full is False

    quick = _record_run(monkeypatch, tmp_path / "quick", ["--quick"])
    default = _record_run(monkeypatch, tmp_path / "default", [])
    full = _record_run(monkeypatch, tmp_path / "full", ["--full"])
    assert quick[0] == default[0] == 0
    assert quick[1] == default[1]
    assert quick[1]["warmup"] == 0 and quick[1]["extra_modules"] == {}
    # --full warms up, so the recorder can tell the modes apart.
    assert full[1]["warmup"] == 1 and "phi_tracker" in full[1]["extra_modules"]

    with pytest.raises(SystemExit) as excinfo:
        assess.build_parser().parse_args(["--quick", "--full"])
    assert excinfo.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def _unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


STORE_KEYS = (
    "MTC_POSTGRES_ENABLED",
    "MTC_MONGODB_ENABLED",
    "MTC_REDIS_ENABLED",
    "MTC_QDRANT_ENABLED",
    "MTC_NEO4J_ENABLED",
)


def _child_env(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MTC_") and key != "ANTHROPIC_API_KEY"
    }
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
            "MTC_LLM_PROVIDER": "mock",
            "MTC_PROJECT_ROOT": str(tmp_path),
            "MTC_DATA_DIRECTORY": str(tmp_path / "data"),
            "MTC_API_PORT": str(_unused_port()),
        }
    )
    env.update({key: "false" for key in STORE_KEYS})
    return env


# Wall-clock and per-run identity fields: they differ between any two runs.
def _volatile(key):
    return (
        "timestamp" in key
        or key.endswith("_at")
        or key.endswith("_ms")
        or "duration" in key
        or key == "session_id"
    )


def _strip_times(value):
    if isinstance(value, dict):
        return {
            key: _strip_times(item) for key, item in value.items() if not _volatile(key)
        }
    if isinstance(value, list):
        return [_strip_times(item) for item in value]
    return value


@pytest.mark.slow
def test_mtc_assess_matches_runner_script(tmp_path):
    outputs = []
    commands = [
        [sys.executable, str(SCRIPT)],
        [sys.executable, "-m", "mtc.cli.assess"],
    ]
    for index, command in enumerate(commands):
        out = tmp_path / f"report{index}.json"
        result = subprocess.run(
            command + ["--seed", "11", "--json-path", str(out)],
            cwd=tmp_path,
            env=_child_env(tmp_path),
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        outputs.append(_strip_times(json.loads(out.read_text())))
    assert outputs[0] == outputs[1]
    assert "overall_score" in outputs[0]
