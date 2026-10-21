"""Tests for the DCM benchmark: the spec manager, the evidence tiers, the
evidence-only default, the experimental posterior and its failure handling,
the baselines and the command line."""

import ast
import importlib.util
import logging
import sys
from pathlib import Path
import json
import pytest
import tempfile
import os
from datetime import datetime

# Minimal mock spec matching the structure from
# https://dcm.rethinkpriorities.org/schemes/133/json
# Each stance has an evidencers tree: features -> subfeatures -> indicators
MOCK_SPEC = [
    {
        "name": "Test Stance A",
        "last_updated": "2025-05-22T15:34:33.446Z",
        "evidencers": [
            {
                "name": "Feature Alpha",
                "support": "strong",
                "demandingness": "moderately demanding",
                "evidencers": [
                    {
                        "name": "Subfeature A1",
                        "evidencers": [
                            {"name": "Indicator 1", "evidencers": []},
                            {"name": "Indicator 2", "evidencers": []},
                        ],
                    }
                ],
            },
            {
                "name": "Feature Beta",
                "support": "weak",
                "demandingness": "weakly demanding",
                "evidencers": [
                    {
                        "name": "Subfeature B1",
                        "evidencers": [
                            {"name": "Indicator 3", "evidencers": []},
                        ],
                    }
                ],
            },
        ],
    },
    {
        "name": "Test Stance B",
        "last_updated": "2025-06-02T12:25:34.961Z",
        "evidencers": [
            {
                "name": "Feature Alpha",
                "support": "moderate",
                "demandingness": "neutral",
                "evidencers": [
                    {
                        "name": "Subfeature A1",
                        "evidencers": [
                            {"name": "Indicator 1", "evidencers": []},
                        ],
                    }
                ],
            },
        ],
    },
]


class _StubEngine:
    """An engine with the BayesianEngine contract: fixed stance posteriors,
    the equal-weight mean as the overall, no pymc needed."""

    def __init__(self, posteriors):
        self.posteriors = dict(posteriors)
        self.overall = sum(self.posteriors.values()) / len(self.posteriors)
        self.failed_stances = {}
        self.failures = {}
        self.scores = None

    def run_all_stances(self, spec, indicator_scores, plausibility_weights=None):
        self.scores = dict(indicator_scores)
        values = list(self.posteriors.values())
        confidence = {name: 0.1 for name in self.posteriors}
        return (
            dict(self.posteriors),
            confidence,
            self.overall,
            (min(values), max(values)),
        )


def _write_spec(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    spec_data = {"version": "test_v1", "fetched_at": "", "raw": MOCK_SPEC}
    with open(os.path.join(cache_dir, "current_spec.json"), "w") as f:
        json.dump(spec_data, f)


def _hide_pymc(monkeypatch):
    """Make pymc unimportable whether or not it is installed."""
    monkeypatch.setitem(sys.modules, "pymc", None)


def _results_files(cache_dir):
    results = Path(cache_dir) / "results"
    return sorted(results.glob("*.json")) if results.exists() else []


class TestModelSpecManager:
    """Tests for fetching, caching, and loading DCM model specs."""

    def test_parse_spec_extracts_stances(self):
        from mtc.assessment.dcm_benchmark import ModelSpecManager

        mgr = ModelSpecManager(cache_dir=tempfile.mkdtemp())
        mgr._raw_spec = MOCK_SPEC
        parsed = mgr._parse_spec()

        assert len(parsed.stances) == 2
        assert parsed.stances[0].name == "Test Stance A"
        assert parsed.stances[1].name == "Test Stance B"

    def test_parse_spec_extracts_indicators(self):
        from mtc.assessment.dcm_benchmark import ModelSpecManager

        mgr = ModelSpecManager(cache_dir=tempfile.mkdtemp())
        mgr._raw_spec = MOCK_SPEC
        parsed = mgr._parse_spec()

        # Should find all unique indicators across all stances
        assert "Indicator 1" in parsed.all_indicator_names
        assert "Indicator 2" in parsed.all_indicator_names
        assert "Indicator 3" in parsed.all_indicator_names

    def test_cache_and_load(self):
        from mtc.assessment.dcm_benchmark import ModelSpecManager

        cache_dir = tempfile.mkdtemp()
        mgr = ModelSpecManager(cache_dir=cache_dir)
        mgr._raw_spec = MOCK_SPEC

        # Save to cache
        version = mgr._save_to_cache()
        assert version.startswith("v133_")

        # Load from cache
        mgr2 = ModelSpecManager(cache_dir=cache_dir)
        loaded = mgr2.load_cached()
        assert loaded is not None
        assert len(loaded.stances) == 2

    def test_bundled_fallback(self):
        from mtc.assessment.dcm_benchmark import ModelSpecManager

        # Empty cache dir, no network
        cache_dir = tempfile.mkdtemp()
        mgr = ModelSpecManager(cache_dir=cache_dir)

        # Should raise if no cache AND no bundled spec
        # (bundled spec is tested separately in integration)
        with pytest.raises(FileNotFoundError):
            mgr.load_cached()

    def test_indicator_count_from_spec(self):
        from mtc.assessment.dcm_benchmark import ModelSpecManager

        mgr = ModelSpecManager(cache_dir=tempfile.mkdtemp())
        mgr._raw_spec = MOCK_SPEC
        parsed = mgr._parse_spec()

        assert len(parsed.all_indicator_names) == 3


@pytest.mark.skipif(
    importlib.util.find_spec("pymc") is None,
    reason="pymc not installed (install the 'assessment' extra)",
)
class TestBayesianEngine:
    """Tests for PyMC Bayesian inference engine."""

    def test_run_stance_returns_posterior(self):
        from mtc.assessment.dcm_benchmark import BayesianEngine, StanceNode

        engine = BayesianEngine(samples=25, seed=42)
        stance = StanceNode(
            name="Test",
            evidencers=[
                {
                    "name": "Feature",
                    "support": "strong",
                    "demandingness": "moderately demanding",
                    "evidencers": [
                        {
                            "name": "Sub",
                            "evidencers": [
                                {"name": "Ind1", "evidencers": []},
                            ],
                        }
                    ],
                }
            ],
        )
        # Indicator present -> should increase posterior above prior (1/6)
        mean_val, ci_width = engine.run_stance(stance, {"Ind1": 1.0})
        assert 0.0 <= mean_val <= 1.0
        assert ci_width >= 0.0

    def test_run_stance_reads_the_posterior_from_the_sampler_default(self):
        """The sampling call passes no ``return_inferencedata`` and raises no
        deprecation warning attributed to the engine's own code. pymc 6.3.2
        warns (a FutureWarning at the ``pm.sample`` call) that
        ``return_inferencedata=False`` is deprecated and will be removed, which
        the v0.5.0 release-candidate matrices reported on 2026-10-19, so the
        engine reads the posterior from the InferenceData the sampler returns
        by default. A warning a library raises inside itself is not the
        engine's and is not counted."""
        import warnings

        from mtc.assessment import dcm_benchmark
        from mtc.assessment.dcm_benchmark import BayesianEngine, StanceNode

        engine = BayesianEngine(samples=25, seed=42)
        stance = StanceNode(
            name="Test",
            evidencers=[
                {
                    "name": "Feature",
                    "support": "strong",
                    "demandingness": "moderately demanding",
                    "evidencers": [
                        {
                            "name": "Sub",
                            "evidencers": [
                                {"name": "Ind1", "evidencers": []},
                            ],
                        }
                    ],
                }
            ],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mean_val, ci_width = engine.run_stance(stance, {"Ind1": 1.0})
        engine_file = Path(dcm_benchmark.__file__).resolve()
        attributed = [
            f"{item.category.__name__}: {item.message}"
            for item in caught
            if issubclass(item.category, (DeprecationWarning, FutureWarning))
            and Path(item.filename).resolve() == engine_file
        ]
        assert attributed == []
        assert 0.0 <= mean_val <= 1.0
        assert 0.0 <= ci_width <= 1.0

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "The DCM engine adds each indicator score as a constant term that "
            "no model variable depends on, and the scheme's sub-feature and "
            "indicator nodes carry no support or demandingness labels, so the "
            "posterior equals the prior mean for any evidence (the DCM "
            "benchmark limitation in docs/HONEST_LIMITATIONS.md). The strict "
            "mark turns red the day the engine responds to the evidence; the "
            "comparison carries a 1e-6 guard because the constant term changes "
            "only the rounding of the summed log density (a residue of 1e-14 at "
            "this seed and under 5e-12 across seeds, whose sign depends on the "
            "pymc version)."
        ),
    )
    def test_high_evidence_increases_posterior(self):
        from mtc.assessment.dcm_benchmark import BayesianEngine, StanceNode

        engine = BayesianEngine(samples=50, seed=42)
        stance = StanceNode(
            name="Test",
            evidencers=[
                {
                    "name": "Feature",
                    "support": "overwhelming",
                    "demandingness": "overwhelmingly demanding",
                    "evidencers": [
                        {
                            "name": "Sub",
                            "evidencers": [
                                {"name": "Ind1", "evidencers": []},
                            ],
                        }
                    ],
                }
            ],
        )
        # All evidence present
        high_mean, _ = engine.run_stance(stance, {"Ind1": 1.0})
        # No evidence
        low_mean, _ = engine.run_stance(stance, {"Ind1": 0.0})
        assert high_mean - low_mean > 1e-6

    def test_run_all_stances(self):
        from mtc.assessment.dcm_benchmark import (
            BayesianEngine,
            ParsedSpec,
            StanceNode,
        )

        engine = BayesianEngine(samples=25, seed=42)
        spec = ParsedSpec(
            stances=[
                StanceNode(
                    name="S1",
                    evidencers=[
                        {
                            "name": "F1",
                            "support": "moderate",
                            "demandingness": "neutral",
                            "evidencers": [
                                {
                                    "name": "Sub",
                                    "evidencers": [
                                        {"name": "Ind1", "evidencers": []},
                                    ],
                                }
                            ],
                        }
                    ],
                ),
                StanceNode(
                    name="S2",
                    evidencers=[
                        {
                            "name": "F2",
                            "support": "weak",
                            "demandingness": "neutral",
                            "evidencers": [
                                {
                                    "name": "Sub",
                                    "evidencers": [
                                        {"name": "Ind2", "evidencers": []},
                                    ],
                                }
                            ],
                        }
                    ],
                ),
            ],
            all_indicator_names={"Ind1", "Ind2"},
            raw=[],
        )
        posteriors, confidence, overall, ci = engine.run_all_stances(
            spec, {"Ind1": 0.8, "Ind2": 0.3}
        )
        assert len(posteriors) == 2
        assert "S1" in posteriors
        assert "S2" in posteriors
        assert 0.0 <= overall <= 1.0

    def test_support_map_keys(self):
        from mtc.assessment.dcm_benchmark import BayesianEngine

        engine = BayesianEngine()
        # Verify key labels from the spec are handled
        assert engine.SUPPORT_MAP["overwhelming"] == 45.0
        assert engine.SUPPORT_MAP["strong"] == 6.7
        assert engine.SUPPORT_MAP["moderate"] == 2.5
        assert engine.SUPPORT_MAP["weak"] == 1.2
        assert engine.SUPPORT_MAP["neutral"] == 1.0


class TestBenchmarkRunner:
    """Tests for the full benchmark orchestration."""

    def test_run_with_mock_adapter(self):
        from mtc.assessment.dcm_benchmark import BenchmarkRunner

        cache_dir = tempfile.mkdtemp()

        # Write mock spec to cache
        spec_data = {
            "version": "test_v1",
            "fetched_at": "2026-01-01",
            "raw": MOCK_SPEC,
        }
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "current_spec.json"), "w") as f:
            json.dump(spec_data, f)

        runner = BenchmarkRunner(cache_dir=cache_dir)

        # Mock adapter
        class MockAdapter:
            def collect_module_evidence(self):
                return {"Indicator 1": 0.8, "Indicator 2": 0.6}

            def get_system_description(self):
                return "Test system"

            def get_conversation_samples(self, n=20):
                return ["Hello", "World"]

            def get_known_zeros(self):
                return set()

        engine = _StubEngine({"Test Stance A": 0.3, "Test Stance B": 0.5})
        result = runner.run(
            adapter=MockAdapter(),
            samples=25,
            mode="local",
            compute_posterior=True,
            engine=engine,
        )

        assert result.overall_probability >= 0.0
        assert result.overall_probability <= 1.0
        assert result.overall_probability == pytest.approx(0.4)
        assert result.overall_probability == engine.overall
        assert result.posterior_status == "experimental"
        assert result.tier1_count == 2
        assert result.spec_version == "test_v1"
        assert engine.scores == {"Indicator 1": 0.8, "Indicator 2": 0.6}

    def test_known_zeros_applied(self):
        from mtc.assessment.dcm_benchmark import BenchmarkRunner

        cache_dir = tempfile.mkdtemp()
        spec_data = {
            "version": "test_v1",
            "fetched_at": "2026-01-01",
            "raw": MOCK_SPEC,
        }
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "current_spec.json"), "w") as f:
            json.dump(spec_data, f)

        runner = BenchmarkRunner(cache_dir=cache_dir)

        class MockAdapter:
            def collect_module_evidence(self):
                return {"Indicator 1": 0.8}

            def get_system_description(self):
                return "Test"

            def get_conversation_samples(self, n=20):
                return []

            def get_known_zeros(self):
                return {"Indicator 3"}  # This one is biological

        engine = _StubEngine({"Test Stance A": 0.25})
        result = runner.run(
            adapter=MockAdapter(), samples=25, compute_posterior=True, engine=engine
        )
        assert result.tier3_count == 1
        assert result.overall_probability == engine.overall
        assert engine.scores == {"Indicator 1": 0.8, "Indicator 3": 0.0}

    def test_result_serialization(self):
        from mtc.assessment.dcm_benchmark import DCMBenchmarkResult

        result = DCMBenchmarkResult(
            timestamp=datetime.now(),
            spec_version="v133_2026-03-18",
            stance_posteriors={"GWT": 0.5, "IIT": 0.3},
            stance_confidence={"GWT": 0.2, "IIT": 0.4},
            overall_probability=0.4,
            overall_ci=(0.2, 0.6),
            indicators_measured=150,
            tier1_count=60,
            tier2_count=50,
            tier3_count=40,
        )
        d = result.to_dict()
        assert d["overall_probability"] == 0.4
        assert d["overall_ci"] == [0.2, 0.6]
        assert d["tier1_count"] == 60


# ---------------------------------------------------------------------------
# History cap, reload of the newest saved result, and the CLI entry point
# ---------------------------------------------------------------------------


def _result(when: datetime, probability: float):
    from mtc.assessment.dcm_benchmark import DCMBenchmarkResult

    return DCMBenchmarkResult(
        timestamp=when,
        spec_version="test",
        stance_posteriors={"A": probability},
        stance_confidence={"A": 0.1},
        overall_probability=probability,
        overall_ci=(probability, probability),
        indicators_measured=3,
        tier1_count=1,
        tier2_count=1,
        tier3_count=1,
        comparison={"baseline": probability - 0.1},
    )


class _StubAdapter:
    def collect_module_evidence(self):
        return {"Indicator 1": 0.5}

    def get_system_description(self):
        return "stub"

    def get_conversation_samples(self, n=20):
        return []

    def get_known_zeros(self):
        return set()


_FACTORY_CALLS = []


def _stub_adapter_factory(state_dir=None):
    _FACTORY_CALLS.append(state_dir)
    return _StubAdapter()


def test_result_history_capped_at_50(monkeypatch, tmp_path):
    from mtc.assessment.dcm_benchmark import BenchmarkRunner, ModelSpecManager

    runner = BenchmarkRunner(cache_dir=str(tmp_path))
    assert runner.history_size == 50
    spec_data = {"version": "test_v1", "fetched_at": "", "raw": MOCK_SPEC}
    (tmp_path / "current_spec.json").write_text(json.dumps(spec_data))
    monkeypatch.setattr(ModelSpecManager, "check_drift", lambda self: None)

    from mtc.assessment import dcm_benchmark

    monkeypatch.setattr(
        dcm_benchmark.BayesianEngine,
        "run_all_stances",
        lambda self, spec, scores, plausibility_weights=None: (
            {"A": 0.2},
            {"A": 0.1},
            0.2,
            (0.2, 0.2),
        ),
    )
    for _ in range(55):
        runner.run(adapter=_StubAdapter(), samples=5, compute_posterior=True)
    assert len(runner.get_history()) == 50


def test_get_latest_reloads_newest_saved_result(tmp_path):
    from mtc.assessment.dcm_benchmark import BenchmarkRunner

    writer = BenchmarkRunner(cache_dir=str(tmp_path))
    writer._save_result(_result(datetime(2026, 1, 1, 12, 0), 0.25))
    newest = _result(datetime(2026, 3, 2, 9, 30), 0.4)
    writer._save_result(newest)

    fresh = BenchmarkRunner(cache_dir=str(tmp_path))
    latest = fresh.get_latest()
    assert latest is not None
    assert latest.overall_probability == pytest.approx(0.4)
    assert latest.timestamp == newest.timestamp
    assert latest.overall_ci == (0.4, 0.4)

    restored = BenchmarkRunner(cache_dir=str(tmp_path))
    restored.from_state_dict({})
    assert restored._latest_result is not None
    assert restored._latest_result.overall_probability == pytest.approx(0.4)

    empty = BenchmarkRunner(cache_dir=str(tmp_path / "nothing"))
    assert empty.get_latest() is None


def test_main_loads_adapter_from_dotted_path(monkeypatch, tmp_path):
    from mtc.assessment import dcm_benchmark

    captured = {}

    def fake_run(self, adapter, evaluator=None, mode="local", samples=100, **kw):
        captured["adapter"] = adapter
        captured["evaluator"] = evaluator
        return _result(datetime(2026, 4, 1), 0.3)

    monkeypatch.setattr(dcm_benchmark.BenchmarkRunner, "run", fake_run)
    _FACTORY_CALLS.clear()
    state_dir = tmp_path / "state"
    code = dcm_benchmark.main(
        [
            "--adapter",
            "tests.assessment.test_dcm_benchmark:_stub_adapter_factory",
            "--state-dir",
            str(state_dir),
            "--mode",
            "manual",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--samples",
            "5",
        ]
    )
    assert code == 0
    assert _FACTORY_CALLS == [str(state_dir)]
    assert isinstance(captured["adapter"], _StubAdapter)
    assert captured["evaluator"] is None

    assert dcm_benchmark.main(["--adapter", "no_such_module_here:factory"]) != 0
    assert dcm_benchmark.main(["--adapter", "missing-colon"]) != 0


def test_main_compare_reads_reloaded_result(tmp_path, capsys):
    from mtc.assessment import dcm_benchmark

    writer = dcm_benchmark.BenchmarkRunner(cache_dir=str(tmp_path))
    writer._save_result(_result(datetime(2026, 5, 1), 0.35))
    code = dcm_benchmark.main(["--compare", "--cache-dir", str(tmp_path)])
    assert code == 0
    assert "0.3500" in capsys.readouterr().out


def test_dcm_benchmark_can_import():
    import mtc.assessment.dcm_benchmark as module

    assert callable(module.main)
    assert module.DEFAULT_ADAPTER == (
        "mtc.assessment.dcm_reference_adapter:build_default_adapter"
    )
    source = Path(module.__file__).read_text()
    assert "No DCMEvidenceAdapter configured" not in source


# ---------------------------------------------------------------------------
# The evidence-only default and the experimental posterior
# ---------------------------------------------------------------------------


class _EvidenceAdapter(_StubAdapter):
    def collect_module_evidence(self):
        return {"Indicator 1": 0.8, "Indicator 2": 0.6}

    def get_known_zeros(self):
        return {"Indicator 3"}


def test_default_run_reports_evidence_only(tmp_path):
    from mtc.assessment.dcm_benchmark import BenchmarkRunner

    _write_spec(tmp_path)
    runner = BenchmarkRunner(cache_dir=str(tmp_path))
    result = runner.run(adapter=_EvidenceAdapter(), samples=5)
    assert result.overall_probability is None
    assert result.overall_ci is None
    assert result.posterior_status == "not_computed"
    assert result.stance_posteriors == {}
    assert result.stance_confidence == {}
    assert result.comparison == {}
    assert (result.tier1_count, result.tier2_count, result.tier3_count) == (2, 0, 1)
    assert result.indicators_measured == 3
    record = result.to_dict()
    assert record["overall_probability"] is None
    assert record["posterior_status"] == "not_computed"
    files = _results_files(tmp_path)
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["overall_probability"] is None
    assert saved["stance_posteriors"] == {}
    reloaded = BenchmarkRunner(cache_dir=str(tmp_path)).get_latest()
    assert reloaded.overall_probability is None
    assert reloaded.posterior_status == "not_computed"


def test_evidence_only_run_keeps_a_same_day_experimental_result(tmp_path):
    from mtc.assessment.dcm_benchmark import BenchmarkRunner

    _write_spec(tmp_path)
    runner = BenchmarkRunner(cache_dir=str(tmp_path))
    engine = _StubEngine({"Test Stance A": 0.25})
    experimental = runner.run(
        adapter=_EvidenceAdapter(), samples=5, compute_posterior=True, engine=engine
    )
    assert experimental.posterior_status == "experimental"
    evidence_only = runner.run(adapter=_EvidenceAdapter(), samples=5)
    assert evidence_only.posterior_status == "not_computed"
    files = _results_files(tmp_path)
    assert len(files) == 2
    saved = {
        record["posterior_status"]: record
        for record in (json.loads(path.read_text()) for path in files)
    }
    assert set(saved) == {"experimental", "not_computed"}
    assert saved["experimental"]["overall_probability"] == engine.overall
    assert saved["not_computed"]["overall_probability"] is None


def test_result_without_a_status_derives_it_from_the_probability():
    from mtc.assessment.dcm_benchmark import DCMBenchmarkResult

    common = dict(
        timestamp=datetime.now(),
        spec_version="test",
        stance_posteriors={},
        stance_confidence={},
        indicators_measured=1,
        tier1_count=1,
        tier2_count=0,
        tier3_count=0,
    )
    absent = DCMBenchmarkResult(overall_probability=None, overall_ci=None, **common)
    assert absent.posterior_status == "not_computed"
    present = DCMBenchmarkResult(
        overall_probability=0.3, overall_ci=(0.3, 0.3), **common
    )
    assert present.posterior_status == "experimental"
    assert DCMBenchmarkResult.from_dict(absent.to_dict()).posterior_status == (
        "not_computed"
    )


def test_default_run_needs_no_pymc(tmp_path, monkeypatch):
    from mtc.assessment.dcm_benchmark import BenchmarkRunner

    _hide_pymc(monkeypatch)
    _write_spec(tmp_path)
    result = BenchmarkRunner(cache_dir=str(tmp_path)).run(adapter=_StubAdapter())
    assert result.posterior_status == "not_computed"
    assert result.overall_probability is None
    assert result.tier1_count == 1


def _main_args(tmp_path, *extra):
    return [
        "--adapter",
        "tests.assessment.test_dcm_benchmark:_stub_adapter_factory",
        "--mode",
        "manual",
        "--cache-dir",
        str(tmp_path),
        "--samples",
        "5",
        *extra,
    ]


def test_cli_default_prints_not_computed(tmp_path, capsys):
    from mtc.assessment import dcm_benchmark

    _write_spec(tmp_path)
    code = dcm_benchmark.main(_main_args(tmp_path))
    out = capsys.readouterr().out
    assert code == 0
    assert "Posterior: not computed" in out
    assert "HONEST_LIMITATIONS" in out
    assert not [
        line for line in out.splitlines() if line.startswith("Overall P(conscious):")
    ]
    assert "Comparison" not in out
    assert " vs " not in out
    assert "Tier 1 (module): 1" in out


def test_experimental_flag_labels_the_number(tmp_path, capsys, monkeypatch):
    from mtc.assessment import dcm_benchmark

    _write_spec(tmp_path)
    (tmp_path / "baselines").mkdir()
    (tmp_path / "baselines" / "verified.json").write_text(
        json.dumps({"overall_probability": 0.9})
    )
    monkeypatch.setattr(
        dcm_benchmark.BayesianEngine,
        "run_all_stances",
        lambda self, spec, scores, plausibility_weights=None: (
            {"Test Stance A": 0.2},
            {"Test Stance A": 0.1},
            0.2,
            (0.2, 0.2),
        ),
    )
    code = dcm_benchmark.main(_main_args(tmp_path, "--experimental-posterior"))
    out = capsys.readouterr().out
    assert code == 0
    assert "Experimental P(conscious): 0.2000" in out
    assert "does not respond to the indicator evidence" in out
    assert "prior mean" in out
    assert not [
        line for line in out.splitlines() if line.startswith("Overall P(conscious):")
    ]
    # No comparison with a published value, even a verified one, is printed.
    assert "Comparison" not in out
    assert "verified" not in out
    latest = dcm_benchmark.BenchmarkRunner(cache_dir=str(tmp_path)).get_latest()
    assert latest.posterior_status == "experimental"
    assert latest.overall_probability == pytest.approx(0.2)


def test_no_full_bayesian_claim_in_package_docs():
    import mtc
    import mtc.assessment
    from mtc.assessment import dcm_benchmark

    for module in (mtc, mtc.assessment, dcm_benchmark):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "Full Bayesian" not in source, module.__name__
        assert "Bayesian, Shiller" not in source, module.__name__
        assert "proper PyMC" not in source, module.__name__
    engine_doc = " ".join(dcm_benchmark.BayesianEngine.__doc__.split())
    assert "constant term" in engine_doc
    assert "no support or demandingness labels" in engine_doc
    assert "thresholded at 0.5" in engine_doc
    assert "prior mean" in engine_doc


# ---------------------------------------------------------------------------
# Failure handling of the experimental posterior
# ---------------------------------------------------------------------------


def test_run_without_pymc_reports_engine_unavailable(tmp_path, monkeypatch):
    from mtc.assessment import dcm_benchmark

    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "pymc" or name.startswith("pymc."):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    _write_spec(tmp_path)
    runner = dcm_benchmark.BenchmarkRunner(cache_dir=str(tmp_path))
    with pytest.raises(dcm_benchmark.DCMEngineUnavailable) as caught:
        runner.run(adapter=_StubAdapter(), compute_posterior=True)
    assert isinstance(caught.value, ImportError)
    assert "multi-theory-consciousness[research]" in str(caught.value)
    assert _results_files(tmp_path) == []
    assert runner.get_latest() is None


def test_run_without_pymc_raises_import_error_naming_the_extra(tmp_path, monkeypatch):
    from mtc.assessment import dcm_benchmark

    _hide_pymc(monkeypatch)
    _write_spec(tmp_path)
    runner = dcm_benchmark.BenchmarkRunner(cache_dir=str(tmp_path))
    with pytest.raises(ImportError, match="research"):
        runner.run(adapter=_StubAdapter(), compute_posterior=True)
    assert _results_files(tmp_path) == []


def _failing_stance(fail):
    def run_stance(self, stance, indicator_scores):
        if stance.name in fail:
            raise RuntimeError("sampler failed")
        return 0.3 if stance.name == "Test Stance B" else 0.4, 0.1

    return run_stance


def test_failed_stance_is_excluded_and_reported(tmp_path, monkeypatch, caplog):
    from mtc.assessment import dcm_benchmark

    monkeypatch.setattr(dcm_benchmark, "pymc_available", lambda: True)
    monkeypatch.setattr(
        dcm_benchmark.BayesianEngine, "run_stance", _failing_stance({"Test Stance A"})
    )
    _write_spec(tmp_path)
    runner = dcm_benchmark.BenchmarkRunner(cache_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="mtc.assessment.dcm_benchmark"):
        result = runner.run(adapter=_StubAdapter(), compute_posterior=True)
    assert result.overall_probability == pytest.approx(0.3)
    assert result.overall_ci == (0.3, 0.3)
    assert result.stance_posteriors == {"Test Stance B": 0.3}
    assert "Test Stance A" not in result.stance_confidence
    assert result.failed_stances == {"Test Stance A": "RuntimeError"}
    assert result.failures == {"dcm_stance": 1}
    assert result.posterior_status == "experimental"
    record = result.to_dict()
    assert record["failed_stances"] == {"Test Stance A": "RuntimeError"}
    assert record["failures"] == {"dcm_stance": 1}
    assert any(
        "Test Stance A" in r.getMessage() and "RuntimeError" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
    assert len(_results_files(tmp_path)) == 1
    reloaded = dcm_benchmark.BenchmarkRunner(cache_dir=str(tmp_path)).get_latest()
    assert reloaded.failed_stances == {"Test Stance A": "RuntimeError"}
    assert reloaded.failures == {"dcm_stance": 1}


def test_all_stances_failed_gives_none_and_saves_nothing(tmp_path, monkeypatch):
    from mtc.assessment import dcm_benchmark

    monkeypatch.setattr(dcm_benchmark, "pymc_available", lambda: True)
    monkeypatch.setattr(
        dcm_benchmark.BayesianEngine,
        "run_stance",
        _failing_stance({"Test Stance A", "Test Stance B"}),
    )
    _write_spec(tmp_path)
    runner = dcm_benchmark.BenchmarkRunner(cache_dir=str(tmp_path))
    result = runner.run(adapter=_StubAdapter(), compute_posterior=True)
    assert result.overall_probability is None
    assert result.overall_ci is None
    assert result.stance_posteriors == {}
    assert set(result.failed_stances) == {"Test Stance A", "Test Stance B"}
    assert result.failures == {"dcm_stance": 2}
    assert _results_files(tmp_path) == []
    assert runner.get_latest() is None
    assert runner.get_history() == []


def test_failed_stance_counted_and_raised_under_debug(monkeypatch):
    from mtc.assessment import dcm_benchmark
    from mtc.core.config import Settings

    monkeypatch.setattr(dcm_benchmark, "pymc_available", lambda: True)
    monkeypatch.setattr(
        dcm_benchmark.BayesianEngine, "run_stance", _failing_stance({"Test Stance A"})
    )
    manager = dcm_benchmark.ModelSpecManager(cache_dir=tempfile.mkdtemp())
    manager._raw_spec = MOCK_SPEC
    spec = manager._parse_spec()
    engine = dcm_benchmark.BayesianEngine(samples=5)
    posteriors, confidence, overall, ci = engine.run_all_stances(
        spec, {"Indicator 1": 0.5}
    )
    assert "Test Stance A" not in posteriors
    assert posteriors["Test Stance B"] == pytest.approx(0.3)
    assert engine.failures == {"dcm_stance": 1}
    assert engine.failed_stances == {"Test Stance A": "RuntimeError"}
    assert overall == pytest.approx(0.3)

    monkeypatch.setattr(dcm_benchmark, "get_settings", lambda: Settings(DEBUG=True))
    with pytest.raises(RuntimeError, match="sampler failed"):
        dcm_benchmark.BayesianEngine(samples=5).run_all_stances(
            spec, {"Indicator 1": 0.5}
        )


def test_main_without_pymc_prints_plain_message(tmp_path, capsys, monkeypatch):
    from mtc.assessment import dcm_benchmark

    _hide_pymc(monkeypatch)
    _write_spec(tmp_path)
    code = dcm_benchmark.main(_main_args(tmp_path, "--experimental-posterior"))
    captured = capsys.readouterr()
    # A missing extra is a configuration error: exit 2, as everywhere else.
    assert code == 2
    assert "research" in captured.err
    assert captured.err.strip().count("\n") == 0
    assert "Traceback" not in captured.err + captured.out
    assert _results_files(tmp_path) == []


def test_main_reports_when_every_stance_failed(tmp_path, capsys, monkeypatch):
    from mtc.assessment import dcm_benchmark

    monkeypatch.setattr(dcm_benchmark, "pymc_available", lambda: True)
    monkeypatch.setattr(
        dcm_benchmark.BayesianEngine,
        "run_stance",
        _failing_stance({"Test Stance A", "Test Stance B"}),
    )
    _write_spec(tmp_path)
    code = dcm_benchmark.main(_main_args(tmp_path, "--experimental-posterior"))
    captured = capsys.readouterr()
    assert code == 1
    assert "every stance failed" in captured.err
    assert "Traceback" not in captured.err + captured.out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _runner_with_baselines(tmp_path, files):
    from mtc.assessment.dcm_benchmark import BenchmarkRunner

    folder = tmp_path / "baselines"
    folder.mkdir()
    for name, text in files.items():
        (folder / name).write_text(text)
    return BenchmarkRunner(cache_dir=str(tmp_path))


def test_malformed_baseline_is_skipped_with_warning(tmp_path, caplog):
    runner = _runner_with_baselines(
        tmp_path,
        {
            "broken.json": "{not json",
            "good.json": json.dumps({"overall_probability": 0.5}),
        },
    )
    with caplog.at_level(logging.WARNING, logger="mtc.assessment.dcm_benchmark"):
        comparison = runner._load_comparisons(0.7)
    assert comparison == {"good": pytest.approx(0.2)}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("broken.json" in w and "JSONDecodeError" in w for w in warnings)


def test_baseline_without_probability_is_skipped(tmp_path):
    runner = _runner_with_baselines(
        tmp_path,
        {
            "missing.json": json.dumps({"name": "missing"}),
            "null.json": json.dumps({"overall_probability": None}),
            "text.json": json.dumps({"overall_probability": "0.4"}),
            "flag.json": json.dumps({"overall_probability": True}),
            "nan.json": '{"overall_probability": NaN}',
            "list.json": json.dumps([0.4]),
            "good.json": json.dumps({"overall_probability": 0.25}),
        },
    )
    assert runner._load_comparisons(0.5) == {"good": pytest.approx(0.25)}
    assert runner._load_comparisons(None) == {}


def test_approximate_baseline_is_skipped(tmp_path):
    runner = _runner_with_baselines(
        tmp_path,
        {
            "rough.json": json.dumps({"overall_probability": 0.5, "approximate": True}),
            "exact.json": json.dumps(
                {"overall_probability": 0.1, "approximate": False}
            ),
        },
    )
    assert runner._load_comparisons(0.6) == {"exact": pytest.approx(0.5)}
    assert runner._load_comparisons(0.6, include_approximate=True) == {
        "exact": pytest.approx(0.5),
        "rough": pytest.approx(0.1),
    }


def test_shipped_baselines_are_marked_approximate():
    root = Path(__file__).resolve().parents[2] / "data" / "dcm" / "baselines"
    files = sorted(root.glob("*.json"))
    assert len(files) == 4
    for path in files:
        data = json.loads(path.read_text())
        assert data["approximate"] is True, path.name
    readme = (root / "README.txt").read_text()
    assert "approximate" in readme
    assert "skip" in readme


def test_compare_prints_no_approximate_baseline(tmp_path, capsys):
    from mtc.assessment import dcm_benchmark

    runner = _runner_with_baselines(
        tmp_path,
        {
            "rough.json": json.dumps({"overall_probability": 0.5, "approximate": True}),
            "exact.json": json.dumps({"overall_probability": 0.1}),
        },
    )
    stored = _result(datetime(2026, 6, 1), 0.35)
    stored.comparison = {"rough": -0.15}
    runner._save_result(stored)
    code = dcm_benchmark.main(["--compare", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Experimental P(conscious): 0.3500" in out
    assert "does not respond to the indicator evidence" in out
    assert "exact" in out
    assert "rough" not in out


def test_compare_on_an_evidence_only_result(tmp_path, capsys):
    from mtc.assessment import dcm_benchmark

    _write_spec(tmp_path)
    dcm_benchmark.BenchmarkRunner(cache_dir=str(tmp_path)).run(adapter=_StubAdapter())
    code = dcm_benchmark.main(["--compare", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Posterior: not computed" in out


def test_no_module_level_string_holds_a_url():
    from mtc.assessment import dcm_benchmark

    tree = ast.parse(Path(dcm_benchmark.__file__).read_text(encoding="utf-8"))
    offenders = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for constant in ast.walk(node):
                if (
                    isinstance(constant, ast.Constant)
                    and isinstance(constant.value, str)
                    and "://" in constant.value
                ):
                    offenders.append((node.lineno, constant.value))
    assert offenders == []
    assert not hasattr(dcm_benchmark, "DEFAULT_DCM_API_URL")
