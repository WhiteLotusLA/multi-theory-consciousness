"""Tests for the full DCM Benchmark implementation."""

import json
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
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
