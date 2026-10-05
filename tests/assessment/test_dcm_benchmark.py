"""Tests for the full DCM Benchmark implementation."""

import importlib.util
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
        assert high_mean > low_mean

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

        result = runner.run(
            adapter=MockAdapter(),
            samples=25,
            mode="local",
        )

        assert result.overall_probability >= 0.0
        assert result.overall_probability <= 1.0
        assert result.tier1_count == 2
        assert result.spec_version == "test_v1"

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

        result = runner.run(adapter=MockAdapter(), samples=25)
        assert result.tier3_count == 1

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
        runner.run(adapter=_StubAdapter(), samples=5)
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
