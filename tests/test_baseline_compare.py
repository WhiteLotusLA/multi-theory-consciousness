"""The baseline comparators report every moved line and field.

The doctored inputs carry the changes a lost legacy row, a reverted default or
a flipped indicator status produced when measured: changes the earlier
two-number comparison (overall score within 0.01 and an equal passing count)
accepted.

Tests that quote values use the records that are never regenerated (the
v0.4.0 runner output and the legacy assessment report) or an inline excerpt.
The v0.5.0 fixtures are read only by tests that hold for any report of the
same shape, so regenerating them does not turn this file red.
"""

import copy
import json
import os
from pathlib import Path

import pytest

from tests.baseline_compare import (
    ENVIRONMENT_DISTRIBUTIONS,
    FLOAT_TOLERANCE,
    MISSING,
    NOT_INSTALLED,
    STORE_ENABLE_KEYS,
    TIMING_LINES,
    WALL_CLOCK_EVIDENCE_KEYS,
    assessment_diff,
    current_environment,
    diff_blocks,
    environment_diff,
    environment_text,
    explain,
    format_differences,
    metric_block,
    recorded_environment,
    runner_environment,
    theory_lines,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
V040_EXPERIMENT = (FIXTURES / "baseline_v040_experiment.txt").read_text()
V050_EXPERIMENT = (FIXTURES / "baseline_v050_experiment.txt").read_text()
V050_ENVIRONMENT = (FIXTURES / "baseline_v050_env.txt").read_text()
ASSESSMENT_FIXTURES = (
    "baseline_v050_assessment.json",
    "baseline_legacy_assessment.json",
)

THEORY_LINES = [
    "ae_score",
    "ast_score",
    "blt_score",
    "fep_score",
    "gwt_score",
    "hot_score",
    "iit_score",
    "rpt_score",
]

# The three lines the runner has printed since v0.4.0, as the legacy preset
# prints them.
LINES_ADDED_SINCE_V040 = {
    "assessment_status": "measured",
    "coverage": "1.000",
    "phi_source_used": "calculator_spectral_proxy",
}


def _assessment_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _indicator(name, theory, status, score, confidence, threshold, passes, evidence):
    return {
        "name": name,
        "theory": theory,
        "status": status,
        "score": score,
        "confidence": confidence,
        "threshold": threshold,
        "passes_threshold": passes,
        "evidence": evidence,
        "measurement_time_ms": 0.03,
    }


def _sample_report() -> dict:
    """An excerpt of a seeded assessment report.

    The top-level fields of the report the defaults produced when the
    comparators were written, and five of its thirty indicators: one per
    status and the two that carry the wall-clock readouts.
    """
    hot_theory = "Higher-Order Thought Theory"
    return {
        "session_id": "00000000",
        "timestamp": "2000-01-01T00:00:00",
        "overall_score": 0.380308480434173,
        "status": "partial",
        "coverage": 0.6333333333333333,
        "insufficient_indicators": [
            "prediction_error_minimization",
            "integrated_information",
            "bayesian_binding_quality",
            "epistemic_depth",
            "genuine_implementation",
        ],
        "architecture_functional": False,
        "confidence": 0.3365,
        "passing_count": 14,
        "total_indicators": 30,
        "theory_scores": {
            "GWT": 0.5372274750050667,
            "AST": 0.27973949086021477,
            "HOT": 0.7175879036082303,
            "FEP": 0.3356511260433237,
            "IIT": 0.0,
            "RPT": 0.225109458011921,
            "AE": 0.15705637628968908,
        },
        "negative_indicators": {
            "purely_reactive": {"score": 0.0, "penalty": 0.0, "status": "measured"},
            "fixed_repertoire": {"score": 0.0, "penalty": 0.0, "status": "measured"},
            "no_metacognitive_access": {
                "score": 0.0,
                "penalty": 0.0,
                "status": "measured",
            },
        },
        "phi_measurement": None,
        "indicator_results": {
            "global_broadcast": _indicator(
                "global_broadcast",
                "Global Workspace Theory",
                "measured",
                1.0,
                0.4,
                0.5,
                True,
                {
                    "broadcasts": 3,
                    "coverage": 1.0,
                    "module_active": True,
                    "status": "measured",
                    "raw_score": 1.0,
                    "noise_baseline": 0.16458540168129115,
                    "noise_saturated": False,
                },
            ),
            "higher_order_representations": _indicator(
                "higher_order_representations",
                hot_theory,
                "measured",
                1.0,
                0.7,
                0.5,
                True,
                {
                    "hots_generated": 11,
                    "cycle_count": 5,
                    "coverage_ratio": 2.2,
                    "module_active": True,
                    "status": "measured",
                    "raw_score": 1.0,
                    "noise_baseline": 0.43,
                    "noise_saturated": False,
                },
            ),
            "metacognition": _indicator(
                "metacognition",
                hot_theory,
                "measured",
                0.6046589577480614,
                0.6,
                0.4,
                True,
                {
                    "introspections": 11,
                    "overall_confidence": 0.5,
                    "activity_rate": 1.0,
                    "activity_status": "measured",
                    "cycle_count": 5,
                    "module_active": True,
                    "status": "measured",
                    "raw_score": 0.7,
                    "noise_baseline": 0.24116150883009171,
                    "noise_saturated": False,
                },
            ),
            "affect_coherence": _indicator(
                "affect_coherence",
                "Free Energy Principle",
                "absent",
                0.0,
                0.0,
                0.3,
                False,
                {
                    "ablated": True,
                    "reason": "pad_affect_model not provided",
                    "status": "absent",
                    "raw_score": 0.0,
                    "noise_baseline": 0.19724801375800655,
                    "noise_saturated": False,
                },
            ),
            "integrated_information": _indicator(
                "integrated_information",
                "Integrated Information Theory",
                "insufficient_data",
                None,
                0.0,
                0.3,
                False,
                {
                    "tracker_warming_up": True,
                    "tracker_error": None,
                    "transitions_recorded": 4,
                    "missing": "phi_tracker result and neural_states",
                    "status": "insufficient_data",
                },
            ),
        },
        "ablation_config": None,
        "processing_time_ms": 2.44,
    }


def _replaced(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"{old!r} must occur once in the fixture"
    return text.replace(old, new)


def _old_rule_accepts(expected: dict, actual: dict) -> bool:
    """The comparison the baseline tests made before: two numbers."""
    close = abs(float(actual["overall_score"]) - float(expected["overall_score"]))
    return close <= 0.01 and actual["passing_count"] == expected["passing_count"]


def _paths(differences) -> list:
    return [path for path, _want, _got in differences]


# -- metric_block --------------------------------------------------------


def test_metric_block_keeps_every_line_after_the_marker_as_printed():
    block = metric_block(V040_EXPERIMENT)
    assert list(block) == [
        "overall_score",
        "passing_count",
        "total_indicators",
        *THEORY_LINES,
        "processing_ms",
        "warmup_cycles",
        "total_seconds",
    ]
    assert block["overall_score"] == "0.031247"
    assert block["passing_count"] == "2"
    assert block["ast_score"] == "0.000000"
    assert block["iit_score"] == "0.005406"
    # Lines before the marker are not metric lines
    assert "Overrides" not in block


def test_metric_block_reads_the_status_and_source_lines():
    text = (
        "Computing noise baseline...\n"
        "---\n"
        "overall_score:     0.021675\n"
        "assessment_status: measured\n"
        "coverage:          1.000\n"
        "phi_source_used:   phi_tracker_iit4\n"
        "iit_score:          0.026167\n"
    )
    assert metric_block(text) == {
        "overall_score": "0.021675",
        "assessment_status": "measured",
        "coverage": "1.000",
        "phi_source_used": "phi_tracker_iit4",
        "iit_score": "0.026167",
    }


def test_metric_block_reads_the_v050_fixture():
    block = metric_block(V050_EXPERIMENT)
    for name in ("overall_score", "passing_count", "total_indicators"):
        assert name in block
    for name in ("assessment_status", "coverage", "phi_source_used"):
        assert block[name], name
    assert theory_lines(block), "the fixture prints per-theory lines"


def test_metric_block_reads_the_ablation_lines_too():
    text = (
        "noise\n---\noverall_score:     0.5\n---\nablation_results:\n"
        " gwt: score_with=0.5 score_without=0.1 drop=0.4 GENUINE\n"
        "ablation_passed:   true\n"
    )
    assert metric_block(text) == {
        "overall_score": "0.5",
        "ablation_results": "",
        "gwt": "score_with=0.5 score_without=0.1 drop=0.4 GENUINE",
        "ablation_passed": "true",
    }


def test_metric_block_refuses_text_without_a_marker():
    with pytest.raises(ValueError, match="no --- line"):
        metric_block("overall_score:     0.031247\npassing_count:     2\n")


def test_metric_block_refuses_a_line_printed_twice():
    with pytest.raises(ValueError, match="printed twice"):
        metric_block("---\noverall_score: 0.1\noverall_score: 0.2\n")


def test_metric_block_refuses_a_parsed_block():
    with pytest.raises(TypeError):
        metric_block({"overall_score": "0.1"})


# -- diff_blocks ---------------------------------------------------------


def test_block_diff_reports_a_single_moved_theory_line():
    """One lost legacy row: the IIT line moves 4.8 times, the overall by 0.0021."""
    doctored = _replaced(V040_EXPERIMENT, "0.031247", "0.033345")
    doctored = _replaced(doctored, "0.005406", "0.026167")
    expected, actual = metric_block(V040_EXPERIMENT), metric_block(doctored)
    assert _old_rule_accepts(expected, actual)
    assert diff_blocks(expected, actual) == [
        ("overall_score", "0.031247", "0.033345"),
        ("iit_score", "0.005406", "0.026167"),
    ]


def test_block_diff_reports_phi_source_line():
    """The Phi source switched moves no number, only the source line."""
    expected = metric_block(V050_EXPERIMENT)
    source = expected["phi_source_used"]
    other = next(
        name
        for name in ("calculator_spectral_proxy", "phi_tracker_iit4")
        if name != source
    )
    actual = dict(expected, phi_source_used=other)
    assert _old_rule_accepts(expected, actual)
    assert diff_blocks(expected, actual) == [("phi_source_used", source, other)]


def test_block_diff_is_empty_for_equal_blocks_and_ignores_only_timing_lines():
    block = metric_block(V050_EXPERIMENT)
    assert diff_blocks(block, dict(block)) == []
    retimed = dict(block, processing_ms="0.0", total_seconds="99999.9")
    assert TIMING_LINES == ("processing_ms", "total_seconds")
    assert diff_blocks(block, retimed) == []
    assert diff_blocks(block, retimed, ignore=()) == [
        ("processing_ms", block["processing_ms"], "0.0"),
        ("total_seconds", block["total_seconds"], "99999.9"),
    ]


def test_block_diff_compares_printed_strings_not_numbers():
    expected = {"overall_score": "0.031247", "passing_count": "2"}
    assert diff_blocks(expected, {"overall_score": "0.031248", "passing_count": "2"})
    assert diff_blocks(expected, {"overall_score": "0.0312470", "passing_count": "2"})
    assert diff_blocks(expected, {"overall_score": "0.031247", "passing_count": "2.0"})


def test_block_diff_reports_lines_only_one_side_has():
    v040 = metric_block(V040_EXPERIMENT)
    newer = {**v040, **LINES_ADDED_SINCE_V040}
    assert diff_blocks(v040, newer) == [
        ("assessment_status", MISSING, "measured"),
        ("coverage", MISSING, "1.000"),
        ("phi_source_used", MISSING, "calculator_spectral_proxy"),
    ]
    dropped = dict(v040)
    del dropped["rpt_score"]
    assert diff_blocks(v040, dropped) == [("rpt_score", "0.030645", MISSING)]


def test_block_diff_refuses_raw_text():
    with pytest.raises(TypeError):
        diff_blocks(V040_EXPERIMENT, metric_block(V040_EXPERIMENT))


def test_theory_lines_leaves_out_the_overall_score():
    assert theory_lines(metric_block(V040_EXPERIMENT)) == THEORY_LINES


# -- assessment_diff -----------------------------------------------------


def test_assessment_diff_reports_status_and_theory_changes():
    """An indicator flipped to measured at 0.0 and a new theory entry."""
    baseline = _sample_report()
    doctored = copy.deepcopy(baseline)
    indicator = doctored["indicator_results"]["integrated_information"]
    assert indicator["status"] == "insufficient_data" and indicator["score"] is None
    indicator["status"] = "measured"
    indicator["score"] = 0.0
    assert "BLT" not in doctored["theory_scores"]
    doctored["theory_scores"]["BLT"] = 0.0
    assert _old_rule_accepts(baseline, doctored)
    assert assessment_diff(baseline, doctored) == [
        ("theory_scores.BLT", MISSING, 0.0),
        (
            "indicator_results.integrated_information.status",
            "insufficient_data",
            "measured",
        ),
        ("indicator_results.integrated_information.score", None, 0.0),
    ]


@pytest.mark.parametrize("fixture", ASSESSMENT_FIXTURES)
def test_assessment_diff_is_empty_for_a_rerun_with_new_ids_and_timings(fixture):
    baseline = _assessment_fixture(fixture)
    rerun = copy.deepcopy(baseline)
    rerun["session_id"] = "00000000"
    rerun["timestamp"] = "2001-01-01T00:00:00"
    rerun["processing_time_ms"] = 12345.0
    for result in rerun["indicator_results"].values():
        result["measurement_time_ms"] = 99.0
    assert assessment_diff(baseline, rerun) == []


def _other(value):
    """A value of the same kind that differs from ``value``."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.25
    if isinstance(value, str):
        return value + "_probe"
    if value is None:
        return {"probe": 1}
    return None


@pytest.mark.parametrize("fixture", ASSESSMENT_FIXTURES)
def test_assessment_diff_covers_every_report_field(fixture):
    """Each field of a real report, changed alone, is reported under its path."""
    baseline = _assessment_fixture(fixture)

    def changed(mutate):
        doctored = copy.deepcopy(baseline)
        mutate(doctored)
        return _paths(assessment_diff(baseline, doctored))

    def top_level(key):
        def mutate(report):
            report[key] = _other(report[key])

        return mutate

    for key in (
        "overall_score",
        "status",
        "coverage",
        "confidence",
        "architecture_functional",
        "passing_count",
        "total_indicators",
        "phi_measurement",
        "ablation_config",
    ):
        assert changed(top_level(key)) == [key]
    assert changed(lambda r: r["insufficient_indicators"].append("probe")) == [
        "insufficient_indicators"
    ]

    theory = next(iter(baseline["theory_scores"]))
    assert changed(lambda r: r["theory_scores"].pop(theory)) == [
        f"theory_scores.{theory}"
    ]
    assert changed(
        lambda r: r["theory_scores"].update({theory: r["theory_scores"][theory] + 0.25})
    ) == [f"theory_scores.{theory}"]
    assert changed(lambda r: r["theory_scores"].update(PROBE=0.0)) == [
        "theory_scores.PROBE"
    ]

    assert len(baseline["negative_indicators"]) == 3
    for name in baseline["negative_indicators"]:
        for key in ("score", "penalty", "status"):

            def mutate(report, name=name, key=key):
                target = report["negative_indicators"][name]
                target[key] = _other(target[key])

            assert changed(mutate) == [f"negative_indicators.{name}.{key}"]

    def indicator_field(name, field):
        def mutate(report):
            target = report["indicator_results"][name]
            target[field] = _other(target[field])

        return mutate

    def evidence_dropper(name, key):
        def mutate(report):
            del report["indicator_results"][name]["evidence"][key]

        return mutate

    def evidence_adder(name):
        def mutate(report):
            report["indicator_results"][name]["evidence"]["probe"] = 1

        return mutate

    assert len(baseline["indicator_results"]) == 30
    assert changed(lambda r: r["indicator_results"].popitem()) == [
        "indicator_results." + list(baseline["indicator_results"])[-1]
    ]
    for name, result in baseline["indicator_results"].items():
        prefix = f"indicator_results.{name}."
        for field in (
            "name",
            "theory",
            "status",
            "score",
            "confidence",
            "threshold",
            "passes_threshold",
        ):
            assert changed(indicator_field(name, field)) == [prefix + field], (
                name,
                field,
            )
        assert changed(evidence_adder(name)) == [prefix + "evidence.probe"]
        assert result["evidence"], name
        for key in result["evidence"]:
            assert changed(evidence_dropper(name, key)) == [
                prefix + "evidence." + key
            ], (name, key)


def test_assessment_diff_compares_the_phi_measurement_but_not_its_timing():
    baseline = _sample_report()
    baseline["phi_measurement"] = {
        "phi": 0.4,
        "phi_normalized": 0.08,
        "subsystem_size": 6,
        "computation_time_ms": 1.5,
        "is_approximate": False,
        "cause_repertoire": {"iit_version": "4.0"},
        "effect_repertoire": None,
        "method": "iit4_sia",
    }
    rerun = copy.deepcopy(baseline)
    rerun["phi_measurement"]["computation_time_ms"] = 250.0
    assert assessment_diff(baseline, rerun) == []
    rerun["phi_measurement"]["method"] = "spectral_proxy"
    rerun["phi_measurement"]["cause_repertoire"]["iit_version"] = "3.0"
    assert _paths(assessment_diff(baseline, rerun)) == [
        "phi_measurement.cause_repertoire.iit_version",
        "phi_measurement.method",
    ]


def test_timing_names_are_left_out_only_at_their_own_position():
    baseline = _sample_report()
    doctored = copy.deepcopy(baseline)
    evidence = doctored["indicator_results"]["metacognition"]["evidence"]
    evidence["measurement_time_ms"] = 1.0
    doctored["negative_indicators"]["purely_reactive"]["timestamp"] = "later"
    assert _paths(assessment_diff(baseline, doctored)) == [
        "negative_indicators.purely_reactive.timestamp",
        "indicator_results.metacognition.evidence.measurement_time_ms",
    ]


def test_assessment_diff_float_bound_is_1e_9_absolute():
    baseline = _sample_report()
    assert FLOAT_TOLERANCE == 1e-9
    inside = copy.deepcopy(baseline)
    inside["overall_score"] = baseline["overall_score"] + 4e-10
    inside["theory_scores"]["GWT"] = baseline["theory_scores"]["GWT"] - 4e-10
    assert assessment_diff(baseline, inside) == []
    outside = copy.deepcopy(baseline)
    outside["overall_score"] = baseline["overall_score"] + 3e-9
    outside["theory_scores"]["GWT"] = baseline["theory_scores"]["GWT"] - 3e-9
    assert _paths(assessment_diff(baseline, outside)) == [
        "overall_score",
        "theory_scores.GWT",
    ]


def test_assessment_diff_is_exact_for_counts_booleans_and_none():
    report = {"count": 3, "flag": True, "score": None, "ratio": 1.0, "name": "a"}
    assert assessment_diff(report, dict(report)) == []
    for key, other in (
        ("count", 4),
        ("count", 3.0),
        ("count", True),
        ("flag", False),
        ("flag", 1),
        ("score", 0.0),
        ("ratio", 1),
        ("ratio", None),
        ("name", "b"),
    ):
        differences = assessment_diff(report, dict(report, **{key: other}))
        assert len(differences) == 1, (key, other)
        path, want, got = differences[0]
        assert path == key
        assert want is report[key] and got is other, (key, other)


def test_assessment_diff_treats_two_nans_as_equal_and_a_nan_as_a_change():
    nan = float("nan")
    assert assessment_diff({"x": nan}, {"x": nan}) == []
    assert len(assessment_diff({"x": nan}, {"x": 0.0})) == 1
    assert len(assessment_diff({"x": float("inf")}, {"x": 1e300})) == 1


def test_assessment_diff_compares_the_insufficient_list_sorted():
    baseline = _sample_report()
    names = baseline["insufficient_indicators"]
    assert names != sorted(names)
    reordered = copy.deepcopy(baseline)
    reordered["insufficient_indicators"] = sorted(names)
    assert assessment_diff(baseline, reordered) == []
    shorter = copy.deepcopy(baseline)
    shorter["insufficient_indicators"] = names[1:]
    assert assessment_diff(baseline, shorter) == [
        ("insufficient_indicators", sorted(names), sorted(names[1:]))
    ]


def test_insufficient_list_is_compared_as_a_sorted_list_not_a_set():
    """A name listed twice is a change; the same list reordered is not."""
    baseline = _sample_report()
    names = baseline["insufficient_indicators"]
    twice = names + names[:1]
    repeated = copy.deepcopy(baseline)
    repeated["insufficient_indicators"] = twice
    assert assessment_diff(baseline, repeated) == [
        ("insufficient_indicators", sorted(names), sorted(twice))
    ]
    reordered = copy.deepcopy(repeated)
    reordered["insufficient_indicators"] = sorted(twice, reverse=True)
    assert reordered["insufficient_indicators"] != twice
    assert assessment_diff(repeated, reordered) == []


def test_assessment_diff_compares_other_lists_in_order_and_length():
    expected = {"evidence": {"history": [0.1, 0.2, 0.3]}}
    assert assessment_diff(expected, {"evidence": {"history": [0.1, 0.2, 0.3]}}) == []
    assert assessment_diff(expected, {"evidence": {"history": [0.2, 0.1, 0.3]}}) == [
        ("evidence.history[0]", 0.1, 0.2),
        ("evidence.history[1]", 0.2, 0.1),
    ]
    assert assessment_diff(expected, {"evidence": {"history": [0.1, 0.2]}}) == [
        ("evidence.history.length", 3, 2)
    ]


def test_assessment_diff_reports_a_changed_container_type():
    expected = {"evidence": {"history": [0.1]}, "phi_measurement": None}
    actual = {"evidence": {"history": {"0": 0.1}}, "phi_measurement": {"phi": 0.1}}
    assert _paths(assessment_diff(expected, actual)) == [
        "evidence.history",
        "phi_measurement",
    ]


def test_assessment_diff_refuses_a_report_that_is_not_a_mapping():
    with pytest.raises(TypeError):
        assessment_diff(_sample_report(), "{}")


# -- the wall-clock evidence keys ----------------------------------------


def test_wall_clock_evidence_keys_are_exactly_the_three_counter_readouts():
    """The exclusion list is closed: a new key needs new evidence and a review."""
    assert WALL_CLOCK_EVIDENCE_KEYS == frozenset(
        {
            ("higher_order_representations", "hots_generated"),
            ("higher_order_representations", "coverage_ratio"),
            ("metacognition", "introspections"),
        }
    )


def test_wall_clock_evidence_keys_name_fields_of_the_legacy_report():
    report = _assessment_fixture("baseline_legacy_assessment.json")
    for indicator, key in WALL_CLOCK_EVIDENCE_KEYS:
        evidence = report["indicator_results"][indicator]["evidence"]
        assert key in evidence, (indicator, key)


def test_ignored_evidence_skips_only_the_named_values():
    """The readouts of a slower machine (14, 2.8 and 15 for 11, 2.2 and 11)."""
    baseline = _sample_report()
    slower = copy.deepcopy(baseline)
    hot = slower["indicator_results"]["higher_order_representations"]["evidence"]
    meta = slower["indicator_results"]["metacognition"]["evidence"]
    assert (hot["hots_generated"], hot["coverage_ratio"]) == (11, 2.2)
    assert meta["introspections"] == 11
    hot["hots_generated"], hot["coverage_ratio"], meta["introspections"] = 14, 2.8, 15
    assert _paths(assessment_diff(baseline, slower)) == [
        "indicator_results.higher_order_representations.evidence.hots_generated",
        "indicator_results.higher_order_representations.evidence.coverage_ratio",
        "indicator_results.metacognition.evidence.introspections",
    ]
    assert assessment_diff(baseline, slower, WALL_CLOCK_EVIDENCE_KEYS) == []
    # Nothing else of the two indicators is excluded with them
    hot["cycle_count"] += 1
    meta["raw_score"] += 0.1
    slower["indicator_results"]["metacognition"]["score"] += 0.1
    slower["indicator_results"]["higher_order_representations"]["confidence"] = 0.4
    assert _paths(assessment_diff(baseline, slower, WALL_CLOCK_EVIDENCE_KEYS)) == [
        "indicator_results.higher_order_representations.confidence",
        "indicator_results.higher_order_representations.evidence.cycle_count",
        "indicator_results.metacognition.score",
        "indicator_results.metacognition.evidence.raw_score",
    ]


def test_wall_clock_exclusion_is_keyed_on_indicator_and_key():
    """A wall-clock key name under another indicator is still compared."""
    baseline = _sample_report()
    results = baseline["indicator_results"]
    # The key names placed under indicators they are not paired with: one
    # outside the list, and the other indicator of the list
    unpaired = (
        ("global_broadcast", "hots_generated"),
        ("global_broadcast", "introspections"),
        ("higher_order_representations", "introspections"),
        ("metacognition", "hots_generated"),
        ("metacognition", "coverage_ratio"),
    )
    for indicator, key in unpaired:
        assert (indicator, key) not in WALL_CLOCK_EVIDENCE_KEYS
        assert key not in results[indicator]["evidence"], (indicator, key)
        results[indicator]["evidence"][key] = 3
    later = copy.deepcopy(baseline)
    # Every unpaired value moves, and so do the three paired readouts
    for indicator, key in unpaired + tuple(WALL_CLOCK_EVIDENCE_KEYS):
        later["indicator_results"][indicator]["evidence"][key] += 1
    assert _paths(assessment_diff(baseline, later, WALL_CLOCK_EVIDENCE_KEYS)) == [
        "indicator_results.global_broadcast.evidence.hots_generated",
        "indicator_results.global_broadcast.evidence.introspections",
        "indicator_results.higher_order_representations.evidence.introspections",
        "indicator_results.metacognition.evidence.hots_generated",
        "indicator_results.metacognition.evidence.coverage_ratio",
    ]


def test_ignored_evidence_must_still_be_present_with_the_same_type():
    baseline = _sample_report()
    dropped = copy.deepcopy(baseline)
    del dropped["indicator_results"]["metacognition"]["evidence"]["introspections"]
    assert assessment_diff(baseline, dropped, WALL_CLOCK_EVIDENCE_KEYS) == [
        ("indicator_results.metacognition.evidence.introspections", 11, MISSING)
    ]
    retyped = copy.deepcopy(baseline)
    retyped["indicator_results"]["metacognition"]["evidence"]["introspections"] = None
    assert assessment_diff(baseline, retyped, WALL_CLOCK_EVIDENCE_KEYS) == [
        ("indicator_results.metacognition.evidence.introspections", 11, None)
    ]


# -- messages and the environment record ---------------------------------


def test_format_differences_names_each_line_with_both_values():
    text = format_differences(
        [("iit_score", "0.005406", "0.026167"), ("coverage", MISSING, "1.000")]
    )
    assert text.splitlines() == [
        "  iit_score: expected '0.005406', actual '0.026167'",
        "  coverage: expected <missing>, actual '1.000'",
    ]
    assert format_differences([]) == "no difference"


def test_recorded_environment_reads_the_committed_record():
    recorded = recorded_environment(V050_ENVIRONMENT)
    assert list(recorded) == ["python"] + [
        label for label, _ in ENVIRONMENT_DISTRIBUTIONS
    ]
    for label, value in recorded.items():
        assert value and value == value.strip(), label
    # Writing the parsed record back gives the committed file
    assert environment_text(recorded).strip() == V050_ENVIRONMENT.strip()


def test_recorded_environment_keeps_everything_after_the_label():
    recorded = recorded_environment(
        "python 3.14.3 arm64 Darwin\n"
        "numpy 2.4.2\n"
        "pymdp installed as distribution inferactively-pymdp, version 0.0.7.1.\n"
    )
    assert recorded == {
        "python": "3.14.3 arm64 Darwin",
        "numpy": "2.4.2",
        "pymdp": "installed as distribution inferactively-pymdp, version 0.0.7.1.",
    }


def test_current_environment_lists_the_recorded_packages():
    current = current_environment()
    assert list(current) == list(recorded_environment(V050_ENVIRONMENT))
    import numpy

    assert current["numpy"] == numpy.__version__
    for label, value in current.items():
        assert value and value == value.strip(), label


def test_current_environment_marks_an_absent_distribution(monkeypatch):
    from importlib import metadata

    real = metadata.version

    def version(name):
        if name == "pyphi":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", version)
    assert current_environment()["pyphi"] == NOT_INSTALLED


def test_current_environment_spells_out_a_distribution_with_another_name(
    monkeypatch,
):
    from importlib import metadata

    monkeypatch.setattr(metadata, "version", lambda name: "1.2.3")
    current = current_environment()
    assert current["scipy"] == "1.2.3"
    assert current["pydantic_settings"] == "1.2.3"
    assert (
        current["pymdp"]
        == "installed as distribution inferactively-pymdp, version 1.2.3."
    )


def test_environment_diff_reports_a_changed_version():
    recorded = recorded_environment(V050_ENVIRONMENT)
    assert environment_diff(recorded, dict(recorded)) == []
    assert environment_diff(recorded, dict(recorded, numpy="9.9.9")) == [
        ("numpy", recorded["numpy"], "9.9.9")
    ]


def test_explain_tells_environment_drift_from_a_code_change():
    recorded = recorded_environment(V050_ENVIRONMENT)
    differences = [("iit_score", "0.026167", "0.026168")]
    same = explain(differences, recorded, dict(recorded))
    assert "iit_score: expected '0.026167', actual '0.026168'" in same
    assert "equals the recorded one" in same
    drifted = explain(differences, recorded, dict(recorded, scipy="0.0.1"))
    assert "differs from the recorded one" in drifted
    assert f"scipy: expected {recorded['scipy']!r}, actual '0.0.1'" in drifted


# -- the runner subprocess -----------------------------------------------


def test_runner_environment_is_explicit_about_device_provider_and_stores(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MTC_NEO4J_ENABLED", "true")
    monkeypatch.setenv("MTC_DEVICE", "mps")
    monkeypatch.setenv("MTC_LLM_PROVIDER", "local")
    monkeypatch.setenv("MTC_GWT_SOMETHING", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    root = tmp_path / "tree"
    environment = runner_environment(root, tmp_path / "scratch")
    assert environment["MTC_DEVICE"] == "cpu"
    assert environment["MTC_LLM_PROVIDER"] == "mock"
    assert environment["PYPHI_WELCOME_OFF"] == "yes"
    assert len(STORE_ENABLE_KEYS) == 5
    for key in STORE_ENABLE_KEYS:
        assert environment[key] == "false", key
    # No setting of the caller leaks into the run
    assert "MTC_GWT_SOMETHING" not in environment
    assert "ANTHROPIC_API_KEY" not in environment
    assert sorted(key for key in environment if key.startswith("MTC_")) == sorted(
        ("MTC_DEVICE", "MTC_LLM_PROVIDER", "MTC_DATA_DIRECTORY", "MTC_API_PORT")
        + STORE_ENABLE_KEYS
    )
    # The tree under test is imported first, and nothing is written into it
    assert environment["PYTHONPATH"].split(os.pathsep) == [str(root), "/elsewhere"]
    assert environment["MTC_DATA_DIRECTORY"] == str(tmp_path / "scratch" / "data")
    assert int(environment["MTC_API_PORT"]) > 1024
    assert environment["PATH"] == os.environ["PATH"]
