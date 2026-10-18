"""Acceptance gates: synthetic signals whose verdicts are known by construction."""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from mtc.validation.gates import (
    GateReport,
    GateResult,
    content_coupling_gate,
    load_shadow_log,
    load_spec,
    movement_gate,
    null_beating_gate,
    range_honesty_gate,
    run_signal_gates,
    write_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SPEC = ROOT / "config" / "validation" / "affect_estimator_v1.yaml"
NOW = datetime(2000, 1, 2, tzinfo=timezone.utc)

MOVE = {"min_std": 0.01, "max_modal_fraction": 0.6, "min_distinct": 5}


def _moving(n=120, seed=13):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * np.pi, n)
    return np.clip(np.sin(t) * 0.6 + rng.normal(0, 0.1, n), -1, 1)


def _records(values, feature=None):
    records = []
    for i, value in enumerate(values):
        record = {
            "ts": (
                datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
            ).isoformat(),
            "signal_id": "synthetic_signal",
            "values": {"valence": float(value)},
            "confidence": 0.8,
            "computed_from": ["synthetic"],
            "context": {},
        }
        if feature is not None:
            record["context"]["feature"] = float(feature[i])
        records.append(record)
    return records


def test_movement_fails_constant_and_template_dominated():
    constant = movement_gate([0.29] * 100, signal_id="s", dimension="valence", **MOVE)
    assert not constant.passed and "constant" in constant.reason
    stamped = [0.29] * 80 + list(np.linspace(-0.5, 0.5, 20))
    template = movement_gate(stamped, signal_id="s", dimension="valence", **MOVE)
    assert not template.passed and "template-dominated" in template.reason
    assert template.metrics["modal_fraction"] == pytest.approx(0.8)
    jitter = [0.29 + 1e-9 * (i % 2) for i in range(80)] + list(
        np.linspace(-0.5, 0.5, 20)
    )
    assert not movement_gate(jitter, signal_id="s", dimension="v", **MOVE).passed
    few = movement_gate(
        [0.0, 1.0] * 50,
        signal_id="s",
        dimension="v",
        min_std=0.01,
        max_modal_fraction=0.6,
        min_distinct=5,
    )
    assert not few.passed and "distinct" in few.reason
    moving = movement_gate(_moving(), signal_id="s", dimension="valence", **MOVE)
    assert moving.passed and isinstance(moving, GateResult)
    assert moving.metrics["std"] > 0.01


def test_content_coupling_uses_spearman_and_permutation():
    signal = _moving()
    rng = np.random.default_rng(3)
    feature = np.exp(3 * signal) + rng.normal(0, 0.01, signal.size)  # monotonic
    result = content_coupling_gate(
        signal,
        feature,
        signal_id="s",
        dimension="valence",
        feature_name="synthetic feature",
        min_abs_corr=0.3,
        max_p=0.05,
        seed=1,
        n_permutations=200,
    )
    rho = stats.spearmanr(signal, feature).statistic
    assert result.passed
    assert result.metrics["correlation"] == pytest.approx(rho)
    assert result.metrics["p_permutation"] == pytest.approx(1 / 201)
    again = content_coupling_gate(
        signal,
        feature,
        signal_id="s",
        dimension="valence",
        feature_name="synthetic feature",
        min_abs_corr=0.3,
        max_p=0.05,
        seed=1,
        n_permutations=200,
    )
    assert again.metrics == result.metrics
    unrelated = rng.normal(0, 1, signal.size)
    uncoupled = content_coupling_gate(
        signal,
        unrelated,
        signal_id="s",
        dimension="valence",
        feature_name="noise",
        min_abs_corr=0.3,
        max_p=0.05,
        seed=1,
        n_permutations=200,
    )
    assert not uncoupled.passed
    with pytest.raises(ValueError):
        content_coupling_gate(
            [1.0, 2.0],
            [1.0],
            signal_id="s",
            dimension="v",
            feature_name="f",
            min_abs_corr=0.1,
            max_p=0.5,
            seed=0,
        )


def test_range_honesty_requires_visiting_claimed_regions():
    regions = [(-1.0, -0.1, "negative"), (-0.1, 0.1, "neutral"), (0.1, 1.0, "positive")]
    kwargs = dict(
        signal_id="s",
        dimension="valence",
        claimed_min=-1.0,
        claimed_max=1.0,
        required_regions=regions,
        min_hits_per_region=5,
    )
    never_negative = np.linspace(0.0, 1.0, 100)
    stuck = range_honesty_gate(never_negative, **kwargs)
    assert not stuck.passed and "never genuinely reached" in stuck.reason
    assert stuck.metrics["region_hits"]["negative"] == 0
    full = range_honesty_gate(np.linspace(-1.0, 1.0, 100), **kwargs)
    assert full.passed
    # The top region includes the claimed maximum itself.
    at_top = range_honesty_gate([-0.5] * 40 + [0.0] * 40 + [1.0] * 20, **kwargs)
    assert at_top.passed and at_top.metrics["region_hits"]["positive"] == 20
    outside = range_honesty_gate(list(np.linspace(-1.0, 1.0, 99)) + [1.5], **kwargs)
    assert not outside.passed and "outside the claimed range" in outside.reason


def test_null_beating_against_matched_baseline():
    rng = np.random.default_rng(5)
    nulls = rng.normal(0.2, 0.05, 500)
    beats = null_beating_gate(
        real_metric=0.6,
        null_metrics=nulls,
        signal_id="s",
        dimension=None,
        metric_name="binding",
    )
    assert beats.passed
    assert beats.metrics["null_p95"] == pytest.approx(np.percentile(nulls, 95))
    at_null = null_beating_gate(
        real_metric=float(nulls.mean()),
        null_metrics=nulls,
        signal_id="s",
        dimension=None,
        metric_name="binding",
    )
    assert not at_null.passed and "indistinguishable" in at_null.reason


def test_nan_inputs_fail_closed():
    nan = float("nan")
    moving = list(_moving())
    assert not movement_gate(
        moving[:-1] + [nan], signal_id="s", dimension="v", **MOVE
    ).passed
    assert not movement_gate(
        moving[:-1] + [math.inf], signal_id="s", dimension="v", **MOVE
    ).passed
    ranged = range_honesty_gate(
        list(np.linspace(-1, 1, 99)) + [nan],
        signal_id="s",
        dimension="v",
        claimed_min=-1.0,
        claimed_max=1.0,
        required_regions=[(-1.0, 0.0, "low"), (0.0, 1.0, "high")],
        min_hits_per_region=1,
    )
    assert not ranged.passed and "non-finite" in ranged.reason
    coupling = dict(
        signal_id="s",
        dimension="v",
        feature_name="f",
        min_abs_corr=0.1,
        max_p=0.5,
        seed=0,
        n_permutations=50,
    )
    signal = _moving()
    constant_feature = content_coupling_gate(signal, np.ones(signal.size), **coupling)
    assert not constant_feature.passed
    assert constant_feature.metrics["correlation"] is None
    feature = signal.copy()
    feature[3] = nan
    assert not content_coupling_gate(signal, feature, **coupling).passed
    for real, null in ((nan, [0.1, 0.2]), (0.5, [0.1, nan]), (0.5, [])):
        result = null_beating_gate(
            real_metric=real,
            null_metrics=null,
            signal_id="s",
            dimension=None,
            metric_name="m",
        )
        assert not result.passed and "failing closed" in result.reason
    # NaN parsed from a shadow line fails the runner's gates too.
    records = _records(_moving())
    records[0]["values"]["valence"] = nan
    spec = {"signal_id": "s", "dimensions": ["valence"], "gates": {"movement": MOVE}}
    assert not run_signal_gates(spec, records, now=NOW).overall_pass


def test_thin_data_fails_closed():
    short = list(_moving(n=20))
    move = movement_gate(short, signal_id="s", dimension="v", **MOVE)
    assert not move.passed and "insufficient" in move.reason
    ranged = range_honesty_gate(
        short,
        signal_id="s",
        dimension="v",
        claimed_min=-1.0,
        claimed_max=1.0,
        required_regions=[(-1.0, 0.0, "low")],
        min_hits_per_region=1,
    )
    assert not ranged.passed and "insufficient" in ranged.reason
    coupled = content_coupling_gate(
        short,
        short,
        signal_id="s",
        dimension="v",
        feature_name="f",
        min_abs_corr=0.1,
        max_p=0.5,
        seed=0,
    )
    assert not coupled.passed and "insufficient" in coupled.reason
    spec = {"signal_id": "s", "dimensions": ["valence"], "gates": {"movement": MOVE}}
    empty = run_signal_gates(spec, [], now=NOW)
    assert empty.n_samples == 0 and not empty.overall_pass
    no_gates = run_signal_gates(
        {"signal_id": "s", "dimensions": ["valence"], "gates": {}},
        _records(_moving()),
        now=NOW,
    )
    assert no_gates.results == [] and not no_gates.overall_pass


def test_runner_aggregates_and_reads_context_features():
    signal = _moving()
    rng = np.random.default_rng(13)
    feature = signal * 0.9 + rng.normal(0, 0.1, signal.size)
    records = _records(signal, feature)
    records.append({"ts": "2000-01-03T00:00:00+00:00", "values": {"valence": None}})
    spec = {
        "signal_id": "synthetic_signal",
        "dimensions": ["valence"],
        "gates": {
            "movement": MOVE,
            "range_honesty": {
                "claimed_min": -1.0,
                "claimed_max": 1.0,
                "required_regions": [[-1.0, 0.0, "negative"], [0.0, 1.0, "positive"]],
                "min_hits_per_region": 3,
            },
            "content_coupling": {
                "feature_key": "feature",
                "feature_name": "synthetic feature",
                "min_abs_corr": 0.3,
                "max_p": 0.05,
                "seed": 1,
                "n_permutations": 200,
            },
        },
    }
    report = run_signal_gates(spec, records, now=NOW)
    assert isinstance(report, GateReport)
    assert report.overall_pass is True
    assert [r.gate for r in report.results] == [
        "movement",
        "range_honesty",
        "content_coupling",
    ]
    assert report.n_samples == len(records)
    assert report.generated_at == NOW.isoformat()
    assert report.window_start.startswith("2000-01-01")
    limited = dict(spec)
    limited["dimensions"] = ["valence", "arousal"]
    limited["gates"] = {"movement": {**MOVE, "dimensions": ["valence"]}}
    only = run_signal_gates(limited, records, now=NOW)
    assert [r.dimension for r in only.results] == ["valence"]
    with pytest.raises(ValueError):
        run_signal_gates(
            {"signal_id": "s", "dimensions": ["v"], "gates": {"x": {}}}, []
        )
    with pytest.raises(ValueError):
        run_signal_gates({"dimensions": ["v"], "gates": {}}, [])


def test_artifacts_are_neutral(tmp_path):
    spec = {
        "signal_id": "synthetic_signal",
        "dimensions": ["valence"],
        "gates": {"movement": MOVE},
    }
    report = run_signal_gates(spec, _records(_moving()), now=NOW)
    paths = write_artifacts(report, tmp_path / "out")
    assert [p.suffix for p in paths] == [".json", ".md"]
    assert paths[0].name == "synthetic_signal-gates-2000-01-02.json"
    data = json.loads(paths[0].read_text())
    assert data["overall_pass"] is True
    assert data["results"][0]["gate"] == "movement"
    md = paths[1].read_text()
    assert md.isascii()
    assert "synthetic_signal" in md and "PASS" in md and "std" in md
    for word in ("reviewable by", "readable by", "dear", "you "):
        assert word not in md.lower()
    failing = run_signal_gates(spec, _records([0.29] * 120), now=NOW)
    md_fail = write_artifacts(failing, tmp_path / "fail")[1].read_text()
    assert "FAIL" in md_fail and md_fail.isascii()


def test_load_shadow_log_and_spec(tmp_path):
    records = _records(_moving(n=5))
    path = tmp_path / "synthetic_signal.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n\n")
    loaded = load_shadow_log(path)
    assert len(loaded) == 5
    assert loaded[0]["values"]["valence"] == records[0]["values"]["valence"]
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"ok": 1}\nnot json\n')
    with pytest.raises(ValueError, match="line 2"):
        load_shadow_log(broken)
    spec_json = tmp_path / "spec.json"
    spec_json.write_text(json.dumps({"signal_id": "s", "dimensions": [], "gates": {}}))
    assert load_spec(spec_json)["signal_id"] == "s"


def test_example_spec_is_runnable():
    spec = load_spec(EXAMPLE_SPEC)
    assert spec["signal_id"] == "affect_estimator_v1"
    assert spec["dimensions"] == ["valence", "arousal"]
    assert set(spec["gates"]) == {"movement", "range_honesty", "content_coupling"}
    report = run_signal_gates(spec, [], now=NOW)
    assert not report.overall_pass  # no evidence is not a pass
    assert all("insufficient" in r.reason for r in report.results)
    assert EXAMPLE_SPEC.read_text().isascii()


COUPLING = {
    "feature_key": "feature",
    "feature_name": "synthetic feature",
    "min_abs_corr": 0.3,
    "max_p": 0.05,
    "seed": 1,
    "n_permutations": 200,
}


def _uncoupled_records():
    rng = np.random.default_rng(29)
    signal = _moving()
    return _records(signal, rng.normal(0, 1, signal.size))


def _typo_spec():
    return {
        "signal_id": "synthetic_signal",
        "dimensions": ["valence"],
        "gates": {
            "movement": MOVE,
            "content_coupling": {**COUPLING, "dimensions": ["Valence"]},
        },
    }


def test_filter_naming_an_unknown_dimension_is_an_error():
    records = _uncoupled_records()
    with pytest.raises(ValueError) as caught:
        run_signal_gates(_typo_spec(), records, now=NOW)
    message = str(caught.value)
    assert "content_coupling" in message and "Valence" in message
    # With the right name the claim is tested, and it fails on this data.
    fixed = _typo_spec()
    fixed["gates"]["content_coupling"]["dimensions"] = ["valence"]
    report = run_signal_gates(fixed, records, now=NOW)
    assert [(r.gate, r.passed) for r in report.results] == [
        ("movement", True),
        ("content_coupling", False),
    ]
    assert report.overall_pass is False


def test_empty_filter_and_empty_dimensions_are_errors():
    records = _records(_moving())
    empty_filter = {
        "signal_id": "s",
        "dimensions": ["valence"],
        "gates": {"movement": {**MOVE, "dimensions": []}},
    }
    with pytest.raises(ValueError, match="movement"):
        run_signal_gates(empty_filter, records, now=NOW)
    no_dimensions = {"signal_id": "s", "dimensions": [], "gates": {"movement": MOVE}}
    with pytest.raises(ValueError, match="movement"):
        run_signal_gates(no_dimensions, records, now=NOW)
    # The null-beating gate needs no dimension and still runs once.
    null_only = {
        "signal_id": "s",
        "dimensions": [],
        "gates": {
            "null_beating": {
                "real_metric": 0.9,
                "null_metrics": [0.1, 0.2, 0.3],
                "metric_name": "m",
            }
        },
    }
    report = run_signal_gates(null_only, records, now=NOW)
    assert [r.gate for r in report.results] == ["null_beating"]


def test_every_configured_gate_has_a_result(monkeypatch):
    signal = _moving()
    rng = np.random.default_rng(13)
    records = _records(signal, signal * 0.9 + rng.normal(0, 0.1, signal.size))
    spec = {
        "signal_id": "synthetic_signal",
        "dimensions": ["valence", "arousal"],
        "gates": {
            "movement": {**MOVE, "dimensions": ["valence"]},
            "range_honesty": {
                "claimed_min": -1.0,
                "claimed_max": 1.0,
                "required_regions": [[-1.0, 0.0, "negative"], [0.0, 1.0, "positive"]],
                "min_hits_per_region": 3,
                "dimensions": ["valence"],
            },
            "content_coupling": {**COUPLING, "dimensions": ["valence"]},
            "null_beating": {
                "real_metric": 0.9,
                "null_metrics": [0.1, 0.2, 0.3],
                "metric_name": "m",
            },
        },
    }
    report = run_signal_gates(spec, records, now=NOW)
    assert {r.gate for r in report.results} == set(spec["gates"])
    # The second line of defence: with the filter check bypassed, a gate that
    # ran nowhere is still reported, as a failure.
    from mtc.validation import gates as gates_module

    monkeypatch.setattr(gates_module, "_check_dimension_filters", lambda *a: None)
    bypassed = run_signal_gates(_typo_spec(), _uncoupled_records(), now=NOW)
    assert {r.gate for r in bypassed.results} == {"movement", "content_coupling"}
    [missing] = [r for r in bypassed.results if r.gate == "content_coupling"]
    assert missing.passed is False and "configured but not run" in missing.reason
    assert bypassed.overall_pass is False
