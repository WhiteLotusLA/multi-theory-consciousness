"""Statistics helper: effect size, tests, trend, ablation effects, export."""

import json

import numpy as np
import pytest
from scipy import stats as scipy_stats

from mtc.research import statistics as st


def test_cohen_d_known_value():
    assert st.cohen_d([1, 2, 3], [2, 3, 4]) == pytest.approx(-1.0)
    assert st.cohen_d([2, 3, 4], [1, 2, 3]) == pytest.approx(1.0)


def test_cohen_d_none_for_zero_variance_or_tiny_samples():
    assert st.cohen_d([1, 1, 1], [2, 2, 2]) is None
    assert st.cohen_d([1.0], [2.0, 3.0]) is None
    assert st.cohen_d([], []) is None
    assert st.cohen_d([1, 2, None, float("nan")], [2, 3, 4]) == pytest.approx(
        st.cohen_d([1, 2], [2, 3, 4])
    )


def test_reproducibility_none_for_zero_mean():
    assert st.reproducibility([1.0, -1.0, 0.0]) is None
    assert st.reproducibility([0.5]) is None
    assert st.reproducibility([0.8, 0.8, 0.8]) == pytest.approx(1.0)
    noisy = st.reproducibility([0.1, 0.9, 0.5])
    assert 0.0 <= noisy < 0.5
    assert st.is_reproducible([0.8, 0.8, 0.8]) is True
    assert st.is_reproducible([0.1, 0.9]) is False
    assert st.is_reproducible([1.0]) is None


def test_t_test_matches_scipy():
    a = [0.61, 0.72, 0.55, 0.68, 0.74, 0.66]
    b = [0.51, 0.49, 0.58, 0.45, 0.53, 0.50, 0.47]
    res = st.compare_samples(a, b)
    ref = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert res.status == "measured"
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.p_value == pytest.approx(ref.pvalue)
    assert res.difference == pytest.approx(np.mean(a) - np.mean(b))
    assert res.n_a == 6 and res.n_b == 7
    assert res.significant is True
    assert res.meaningful_effect is True


def test_t_test_insufficient_cases():
    assert st.compare_samples([1.0], [1.0, 2.0]).status == "insufficient_data"
    flat = st.compare_samples([1, 1, 1], [2, 2, 2])
    assert flat.status == "insufficient_data" and flat.reason == "zero variance"
    assert flat.p_value is None
    with pytest.raises(ValueError):
        st.compare_samples([1, 2], [3, 4], method="other")


def test_permutation_seeded_reproducible():
    a = [0.9, 0.8, 0.85, 0.95, 0.88]
    b = [0.5, 0.55, 0.6, 0.45, 0.52]
    r1 = st.compare_samples(a, b, method="permutation", n_permutations=2000, seed=7)
    r2 = st.compare_samples(a, b, method="permutation", n_permutations=2000, seed=7)
    assert r1.p_value == r2.p_value
    assert r1.p_value < 0.05
    same = st.compare_samples(
        [1, 2, 3, 4], [1, 2, 3, 4], method="permutation", n_permutations=500, seed=1
    )
    assert same.p_value > 0.5


def test_one_sample_test_against_chance():
    values = [0.7, 0.8, 0.75, 0.9, 0.65, 0.85]
    res = st.one_sample_test(values, chance=0.5)
    ref = scipy_stats.ttest_1samp(values, 0.5)
    assert res.p_value == pytest.approx(ref.pvalue)
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.ci_low < np.mean(values) < res.ci_high
    assert res.effect_size == pytest.approx(
        (np.mean(values) - 0.5) / np.std(values, ddof=1)
    )
    assert res.significant is True
    assert st.one_sample_test([0.5]).status == "insufficient_data"
    assert st.one_sample_test([0.5, 0.5, 0.5]).reason == "zero variance"


def test_trend_needs_three_points():
    assert st.trend([1.0, 2.0]) is None
    assert st.trend([]) is None
    out = st.trend([1.0, 2.0, 3.0, 4.0])
    assert out.slope == pytest.approx(1.0)
    assert out.n == 4 and out.r_squared == pytest.approx(1.0)
    timed = st.trend([1.0, 3.0, 5.0], times=[0, 2, 4])
    assert timed.slope == pytest.approx(1.0)
    assert st.trend([1.0, 2.0, 3.0], times=[5, 5, 5]) is None
    assert st.trend([1.0, None, 3.0, 4.0]).n == 3


def run(base, **others):
    configs = [{"name": "Baseline", "overall_score": base}]
    configs += [{"name": k, "overall_score": v} for k, v in others.items()]
    return {"configurations": configs}


def test_ablation_effects_skip_none_overall():
    runs = [
        run(0.80, No_GWT=0.60, No_AST=None),
        run(0.82, No_GWT=0.58, No_AST=0.70),
        run(0.78, No_GWT=0.62, No_AST=0.72),
        run(None, No_GWT=0.61, No_AST=0.71),
    ]
    out = st.ablation_effects(runs)
    assert set(out) == {"No_GWT", "No_AST"}
    gwt = out["No_GWT"]
    assert gwt.n_a == 3 and gwt.n_b == 4
    assert gwt.difference == pytest.approx(0.80 - np.mean([0.60, 0.58, 0.62, 0.61]))
    ast = out["No_AST"]
    assert ast.n_b == 3  # the None overall left out of four runs
    only_none = st.ablation_effects([run(None, No_X=None), run(None, No_X=None)])
    assert only_none["No_X"].status == "insufficient_data"


def test_export_is_plain_json_without_claims(tmp_path):
    result = st.compare_samples([1, 2, 3, 4], [2, 3, 4, 6])
    bad = st.compare_samples([1, 1, 1], [2, 2, 2])
    path = st.export_results(
        {"a": result, "b": bad, "x": float("nan"), "arr": np.arange(3)},
        tmp_path / "deep" / "out.json",
    )
    text = path.read_text()
    data = json.loads(text)
    assert data["results"]["x"] is None
    assert data["results"]["arr"] == [0, 1, 2]
    assert data["results"]["a"]["method"] == "t_test"
    assert data["thresholds"]["alpha"] == 0.05
    lowered = text.lower()
    for word in ("demonstrated", "validated", "proof", "funding", "conscious"):
        assert word not in lowered
    assert "NaN" not in text


def test_module_docstring_warns_about_the_stdlib_name():
    assert "standard library" in st.__doc__
    assert "scripts" in st.__doc__
