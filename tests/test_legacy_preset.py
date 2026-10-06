"""The legacy_v04 preset must reproduce the recorded v0.4.0 numbers.

``tests/fixtures/baseline_v040_experiment.txt`` is the experiment runner's
output as v0.4.0 printed it; the preset must print every one of its metric
lines again, as the same string. ``tests/fixtures/baseline_legacy_assessment.json``
is the seeded assessment CLI's report under the preset (the numbers the
migration record publishes); the preset must reproduce every field of it.

Every row of the parameter schema must also say what it does under the
preset: it has a preset value, or it existed in v0.4.0 with the same default
(``tests/fixtures/param_schema_v040_defaults.json``), or it is listed in
``INERT_UNDER_LEGACY`` with the reason.
"""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

import pytest

from tests.baseline_compare import (
    WALL_CLOCK_EVIDENCE_KEYS,
    assessment_diff,
    current_environment,
    diff_blocks,
    explain,
    metric_block,
    recorded_environment,
    runner_environment,
    theory_lines,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
BASELINE = FIXTURES / "baseline_v040_experiment.txt"
ASSESSMENT_BASELINE = FIXTURES / "baseline_legacy_assessment.json"
ENVIRONMENT_RECORD = FIXTURES / "baseline_v050_env.txt"
V040_DEFAULTS = FIXTURES / "param_schema_v040_defaults.json"
SCHEMA = ROOT / "experiments" / "param_schema.toml"
PRESET = ROOT / "experiments" / "presets" / "legacy_v04.toml"

# The three lines the v0.4.0 runner did not print, as the preset prints them.
LINES_ADDED_SINCE_V040 = {
    "assessment_status": "measured",
    "coverage": "1.000",
    "phi_source_used": "calculator_spectral_proxy",
}

# Schema rows that are new since v0.4.0 and have no preset value, each with
# the reason the preset still reproduces v0.4.0 without one. The list only
# shrinks: a new row gets a preset value.
INERT_UNDER_LEGACY = {
    "phi.scale": (
        "multiplies the PhiTracker's proxy and leaves it unchanged at the "
        "default 1.0; the preset reads Phi from the calculator "
        "(assessment.phi_source)"
    ),
    "phi.warmup_reference": (
        "denominator of the PhiTracker's normalized readout during warm-up; "
        "the preset reads Phi from the calculator (assessment.phi_source)"
    ),
    "pad.coherence_window": (
        "read only by the temporal_structure coherence; the preset sets "
        "pad.coherence_method to v04_variance"
    ),
    "pad.coherence_min_history": (
        "read only by the temporal_structure coherence; the preset sets "
        "pad.coherence_method to v04_variance"
    ),
    "assessment.phi_normalization": (
        "new row whose default, fixed, is what v0.4.0 did: Phi divided by a " "constant"
    ),
    "assessment.phi_fixed_reference": (
        "new row whose default, 5.0, is the constant v0.4.0 divided Phi by"
    ),
    "cycle.drive_urgency_threshold": (
        "read only while cycle.drive_candidates is on; the preset sets it false"
    ),
}


def _failure_message(differences) -> str:
    return explain(
        differences,
        recorded_environment(ENVIRONMENT_RECORD.read_text()),
        current_environment(),
    )


@pytest.mark.slow
def test_legacy_v04_preset_reproduces_baseline(tmp_path):
    """Every metric line v0.4.0 printed comes back as the same string."""
    expected = metric_block(BASELINE.read_text())
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--preset", "legacy_v04"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=runner_environment(ROOT, tmp_path),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    actual = metric_block(result.stdout)
    # The whole block: the fixture's lines, the three lines added since, and
    # no other line
    differences = diff_blocks({**expected, **LINES_ADDED_SINCE_V040}, actual)
    assert differences == [], _failure_message(differences)
    assert len(theory_lines(expected)) == 8
    assert set(theory_lines(actual)) == set(theory_lines(expected))


@pytest.mark.slow
def test_legacy_v04_assessment_reproduces(tmp_path):
    """The seeded assessment under the preset lands on its recorded report."""
    pytest.importorskip("pymdp")
    baseline = json.loads(ASSESSMENT_BASELINE.read_text())
    json_path = tmp_path / "assessment.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_assessment.py",
            "--preset",
            "legacy_v04",
            "--full",
            "--json",
            "--seed",
            "42",
            "--json-path",
            str(json_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=runner_environment(ROOT, tmp_path),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(json_path.read_text())
    # The preset keeps the wall-clock introspection cooldown of v0.4.0, so
    # the three counter readouts that follow it are not compared
    differences = assessment_diff(
        baseline, report, ignore_evidence=WALL_CLOCK_EVIDENCE_KEYS
    )
    assert differences == [], _failure_message(differences)


def test_legacy_assessment_fixture_holds_the_published_numbers():
    """The fixture is the record of the published run, not of a later tree."""
    text = ASSESSMENT_BASELINE.read_text()
    report = json.loads(text)
    assert f"{report['overall_score']:.6f}" == "0.333844"
    assert report["passing_count"] == 13
    assert report["total_indicators"] == 30
    assert report["status"] == "partial"
    assert f"{report['coverage']:.3f}" == "0.833"
    assert f"{report['confidence']:.3f}" == "0.480"
    assert {
        theory: f"{score:.3f}" for theory, score in report["theory_scores"].items()
    } == {
        "GWT": "0.468",
        "AST": "0.621",
        "HOT": "0.656",
        "FEP": "0.330",
        "IIT": "0.000",
        "RPT": "0.155",
        "AE": "0.564",
        "BLT": "0.000",
    }
    assert len(report["indicator_results"]) == 30
    # A fixture names no machine: neither the home directory of the machine
    # running the test nor the directory that holds its home directories
    # (left out when that directory is the filesystem root)
    home = Path.home()
    markers = [str(home)]
    if home.parent != home.parent.parent:
        markers.append(str(home.parent) + os.sep)
    for marker in markers:
        assert marker not in text


# -- every schema row has a legacy value ---------------------------------


def _schema_rows(node: Mapping[str, Any], prefix: str = "") -> Dict[str, dict]:
    """Dotted row name to row: a row is a table that declares a type."""
    rows: Dict[str, dict] = {}
    for key, value in node.items():
        if isinstance(value, dict) and "type" in value:
            rows[prefix + key] = value
        elif isinstance(value, dict):
            rows.update(_schema_rows(value, prefix + key + "."))
    return rows


def _preset_keys(node: Mapping[str, Any], prefix: str = "") -> Set[str]:
    keys: Set[str] = set()
    for key, value in node.items():
        if isinstance(value, dict):
            keys |= _preset_keys(value, prefix + key + ".")
        else:
            keys.add(prefix + key)
    return keys


def _same_default(old: Any, new: Any) -> bool:
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool) and old == new
    return old == new


def _covered_by_v040(name: str, row: Mapping[str, Any], v040: Mapping) -> bool:
    return (
        name in v040 and "default" in row and _same_default(v040[name], row["default"])
    )


def _rows_without_legacy_value(
    rows: Mapping[str, Mapping[str, Any]],
    preset_keys: Set[str],
    v040_defaults: Mapping[str, Any],
    inert: Mapping[str, str],
) -> List[str]:
    """Schema rows whose behaviour under the preset nothing accounts for."""
    return [
        name
        for name, row in rows.items()
        if name not in preset_keys
        and not _covered_by_v040(name, row, v040_defaults)
        and name not in inert
    ]


def _load_schema_rows() -> Dict[str, dict]:
    return _schema_rows(tomllib.loads(SCHEMA.read_text()))


def _load_preset_keys() -> Set[str]:
    return _preset_keys(tomllib.loads(PRESET.read_text()))


def _load_v040_defaults() -> Dict[str, Any]:
    return json.loads(V040_DEFAULTS.read_text())


def test_every_schema_row_has_a_legacy_value():
    rows = _load_schema_rows()
    assert len(rows) >= 112
    missing = _rows_without_legacy_value(
        rows, _load_preset_keys(), _load_v040_defaults(), INERT_UNDER_LEGACY
    )
    assert missing == [], (
        "schema rows with no legacy_v04 value, no unchanged v0.4.0 default and "
        f"no recorded reason: {missing}"
    )


def test_a_new_schema_row_without_a_legacy_value_is_reported():
    rows = _load_schema_rows()
    preset_keys, v040 = _load_preset_keys(), _load_v040_defaults()
    # A row added with a default of its own and no preset value
    rows["gwt.probe_switch"] = {"type": "bool", "default": True}
    assert _rows_without_legacy_value(rows, preset_keys, v040, INERT_UNDER_LEGACY) == [
        "gwt.probe_switch"
    ]
    # The same row with a preset value is accounted for
    assert (
        _rows_without_legacy_value(
            rows, preset_keys | {"gwt.probe_switch"}, v040, INERT_UNDER_LEGACY
        )
        == []
    )
    # A v0.4.0 row whose default has changed needs a preset value too
    del rows["gwt.probe_switch"]
    name = "gwt.capacity"
    assert name in v040 and name not in preset_keys
    rows[name] = dict(rows[name], default=v040[name] + 1)
    assert _rows_without_legacy_value(rows, preset_keys, v040, INERT_UNDER_LEGACY) == [
        name
    ]


def test_exempt_rows_are_exactly_the_rows_that_need_the_exemption():
    without_exemptions = _rows_without_legacy_value(
        _load_schema_rows(), _load_preset_keys(), _load_v040_defaults(), {}
    )
    assert sorted(without_exemptions) == sorted(INERT_UNDER_LEGACY)


def test_inert_under_legacy_only_shrinks():
    rows = _load_schema_rows()
    preset_keys, v040 = _load_preset_keys(), _load_v040_defaults()
    # The seven rows exempt when the list was written: the list may lose any
    # of them, and no other row ever enters it, not even in place of one
    written_with = frozenset(
        {
            "phi.scale",
            "phi.warmup_reference",
            "pad.coherence_window",
            "pad.coherence_min_history",
            "assessment.phi_normalization",
            "assessment.phi_fixed_reference",
            "cycle.drive_urgency_threshold",
        }
    )
    entered = sorted(set(INERT_UNDER_LEGACY) - written_with)
    assert entered == [], f"rows exempted after the list was written: {entered}"
    for name, reason in INERT_UNDER_LEGACY.items():
        assert name in rows, f"{name} is no longer a schema row: remove it"
        assert name not in preset_keys, f"{name} has a preset value: remove it"
        assert not _covered_by_v040(
            name, rows[name], v040
        ), f"{name} is a v0.4.0 row with an unchanged default: remove it"
        assert len(reason.split()) >= 5, f"{name} needs its reason spelled out"


def test_v040_defaults_fixture_is_a_flat_map_of_row_to_default():
    v040 = _load_v040_defaults()
    assert len(v040) == 69
    assert v040["gwt.capacity"] == 7
    assert v040["blt.field_evidencing_threshold"] == 0.4
    for name, default in v040.items():
        assert "." in name
        assert isinstance(default, (bool, int, float, str)), name


def _run_script(tmp_path, *args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=runner_environment(ROOT, tmp_path),
    )


@pytest.mark.slow
def test_run_assessment_rejects_unknown_preset(tmp_path):
    result = _run_script(
        tmp_path, "scripts/run_assessment.py", "--preset", "no_such_preset"
    )
    assert result.returncode != 0
    assert "PARAM ERROR" in result.stderr
    assert "no_such_preset" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.slow
def test_run_experiment_reports_missing_params_file(tmp_path):
    result = _run_script(
        tmp_path, "scripts/run_experiment.py", "--params", "no_such_file.toml"
    )
    assert result.returncode != 0
    assert "PARAM ERROR" in result.stderr
    assert "Traceback" not in result.stderr
