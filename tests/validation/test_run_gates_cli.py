"""Acceptance-gate CLI: spec and shadow directory in, artifacts and exit code out."""

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from mtc.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validation" / "run_gates.py"

SPEC_YAML = """signal_id: cli_signal
dimensions: [valence]
gates:
  movement:
    min_std: 0.01
    max_modal_fraction: 0.6
    min_distinct: 5
    min_samples: 50
"""


@pytest.fixture
def cli(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("run_gates_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = Settings(
        _env_file=None,
        PROJECT_ROOT=str(tmp_path),
        SHADOW_LOG_DIR="shadow",
        RESULTS_DIR="results",
    )
    monkeypatch.setattr(module, "get_settings", lambda: settings)
    return module


def _write_log(path: Path, values) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    with open(path, "w", encoding="utf-8") as handle:
        for i, value in enumerate(values):
            handle.write(
                json.dumps(
                    {
                        "ts": (start + timedelta(minutes=i)).isoformat(),
                        "signal_id": "cli_signal",
                        "values": {"valence": float(value)},
                    }
                )
                + "\n"
            )
    return path


def _moving(n=120):
    rng = np.random.default_rng(13)
    t = np.linspace(0, 6 * np.pi, n)
    return np.clip(np.sin(t) * 0.6 + rng.normal(0, 0.1, n), -1, 1)


def _spec(tmp_path) -> Path:
    path = tmp_path / "spec.yaml"
    path.write_text(SPEC_YAML, encoding="utf-8")
    return path


def test_cli_reads_spec_and_shadow_dir(cli, tmp_path, capsys):
    _write_log(tmp_path / "shadow" / "cli_signal.jsonl", _moving())
    assert cli.main(["--spec", str(_spec(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "cli_signal" in out and "overall: PASS" in out
    assert out.isascii()
    other = tmp_path / "elsewhere"
    _write_log(other / "cli_signal.jsonl", [0.29] * 120)
    assert cli.main(["--spec", str(_spec(tmp_path)), "--shadow-dir", str(other)]) == 1
    spec_json = tmp_path / "spec.json"
    spec_json.write_text(
        json.dumps(
            {
                "signal_id": "cli_signal",
                "dimensions": ["valence"],
                "gates": {
                    "movement": {
                        "min_std": 0.01,
                        "max_modal_fraction": 0.6,
                        "min_distinct": 5,
                    }
                },
            }
        )
    )
    assert cli.main(["--spec", str(spec_json)]) == 0


def test_artifacts_under_results_dir(cli, tmp_path, capsys):
    _write_log(tmp_path / "shadow" / "cli_signal.jsonl", _moving())
    assert cli.main(["--spec", str(_spec(tmp_path))]) == 0
    artifacts = sorted((tmp_path / "results" / "validation").iterdir())
    assert [p.suffix for p in artifacts] == [".json", ".md"]
    assert artifacts[0].name.startswith("cli_signal-gates-")
    assert json.loads(artifacts[0].read_text())["overall_pass"] is True
    explicit = tmp_path / "explicit_out"
    assert cli.main(["--spec", str(_spec(tmp_path)), "--out", str(explicit)]) == 0
    assert len(list(explicit.glob("cli_signal-gates-*.md"))) == 1


def test_exit_code_nonzero_on_failure(cli, tmp_path, capsys):
    log = _write_log(tmp_path / "shadow" / "cli_signal.jsonl", [0.29] * 120)
    assert cli.main(["--spec", str(_spec(tmp_path))]) == 1
    assert "FAIL" in capsys.readouterr().out
    # The script also runs standalone from the repository root.
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--spec",
            str(_spec(tmp_path)),
            "--log",
            str(log),
            "--out",
            str(tmp_path / "standalone"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert proc.returncode == 1, proc.stderr
    assert "FAIL" in proc.stdout
    assert list((tmp_path / "standalone").glob("cli_signal-gates-*.json"))


def test_missing_shadow_log_is_reported_not_passed(cli, tmp_path, capsys):
    code = cli.main(["--spec", str(_spec(tmp_path))])
    out = capsys.readouterr().out
    assert code == 2
    assert "cli_signal.jsonl" in out and "not passed" in out
    assert not (tmp_path / "results").exists()
    bad_spec = tmp_path / "missing.yaml"
    assert cli.main(["--spec", str(bad_spec)]) == 2
    broken = _write_log(tmp_path / "shadow" / "cli_signal.jsonl", _moving())
    broken.write_text(broken.read_text() + "not json\n")
    assert cli.main(["--spec", str(_spec(tmp_path))]) == 2
