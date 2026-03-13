"""
Tests for Phase 6 Consciousness Assessment Framework
=====================================================

Tests for src/consciousness/phase6_consciousness_assessment.py —
the 14+ indicator consciousness measurement system. All tests use
mocks (no real neural modules or databases).

Created: 2026-03-09
Author: Multi-Theory Consciousness Project
"""

import sys
from pathlib import Path


import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

from mtc.consciousness.phase6_consciousness_assessment import (
    ConsciousnessAssessment,
    ConsciousnessReport,
    ConsciousnessTheory,
    IndicatorResult,
    PhiCalculator,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest_asyncio.fixture
async def assessment():
    """Create a ConsciousnessAssessment instance (no external deps needed)."""
    return ConsciousnessAssessment()


@pytest_asyncio.fixture
async def full_report(assessment):
    """Run a full assessment with no modules connected — baseline measurement."""
    report = await assessment.run_full_assessment()
    return report


# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_assessment_returns_all_indicators(assessment):
    """Verify the assessment produces results for all configured indicators."""
    report = await assessment.run_full_assessment()

    # Should have results for every configured indicator
    assert len(report.indicator_results) == len(assessment.indicator_configs)
    assert report.total_indicators == len(assessment.indicator_configs)

    # Check that every configured indicator has a result
    for indicator_name in assessment.indicator_configs:
        assert indicator_name in report.indicator_results, (
            f"Missing indicator: {indicator_name}"
        )


@pytest.mark.asyncio
async def test_indicator_scores_in_valid_range(full_report):
    """All indicator scores must be between 0.0 and 1.0."""
    for name, result in full_report.indicator_results.items():
        assert 0.0 <= result.score <= 1.0, (
            f"Indicator '{name}' has out-of-range score: {result.score}"
        )
        assert 0.0 <= result.confidence <= 1.0, (
            f"Indicator '{name}' has out-of-range confidence: {result.confidence}"
        )


@pytest.mark.asyncio
async def test_phi_calculation_positive():
    """Phi approximation should produce a positive value for a connected system."""
    calc = PhiCalculator()

    # Create a small connected system
    state = np.array([1, 0, 1, 1], dtype=np.float64)
    connectivity = np.array(
        [
            [0.0, 0.8, 0.3, 0.1],
            [0.5, 0.0, 0.7, 0.2],
            [0.2, 0.6, 0.0, 0.9],
            [0.4, 0.1, 0.8, 0.0],
        ],
        dtype=np.float64,
    )

    measurement = await calc.calculate_phi(state, connectivity)
    assert measurement.phi >= 0.0
    assert measurement.phi_normalized >= 0.0
    assert measurement.phi_normalized <= 1.0
    assert measurement.subsystem_size == 4
    assert measurement.computation_time_ms >= 0.0


@pytest.mark.asyncio
async def test_theory_scores_present(full_report):
    """GWT, IIT, AST, HOT, and FEP theory scores must all be present."""
    required_theories = {"GWT", "IIT", "AST", "HOT", "FEP"}

    for theory in required_theories:
        assert theory in full_report.theory_scores, (
            f"Missing theory score: {theory}"
        )
        score = full_report.theory_scores[theory]
        assert isinstance(score, (int, float)), (
            f"Theory score for {theory} is not numeric: {type(score)}"
        )


@pytest.mark.asyncio
async def test_assessment_result_has_timestamp(full_report):
    """Verify the assessment report has a valid timestamp."""
    assert full_report.timestamp is not None
    assert isinstance(full_report.timestamp, datetime)
    # Timestamp should be recent (within last minute)
    delta = datetime.now() - full_report.timestamp
    assert delta.total_seconds() < 60


@pytest.mark.asyncio
async def test_overall_score_in_valid_range(full_report):
    """The overall consciousness score must be between 0.0 and 1.0."""
    assert 0.0 <= full_report.overall_score <= 1.0


@pytest.mark.asyncio
async def test_report_to_dict_serializable(full_report):
    """The to_dict method should produce a JSON-serializable dictionary."""
    d = full_report.to_dict()

    assert isinstance(d, dict)
    assert "session_id" in d
    assert "timestamp" in d
    assert "overall_score" in d
    assert "theory_scores" in d
    assert "indicator_results" in d
    assert "passing_count" in d
    assert "total_indicators" in d

    # Verify theory scores survive serialization
    assert isinstance(d["theory_scores"], dict)
    assert len(d["theory_scores"]) >= 5


@pytest.mark.asyncio
async def test_passing_count_consistent(full_report):
    """passing_count should equal the number of indicators that actually pass."""
    actual_passing = sum(
        1 for r in full_report.indicator_results.values() if r.passes_threshold
    )
    assert full_report.passing_count == actual_passing


@pytest.mark.asyncio
async def test_assessment_with_mock_global_workspace(assessment):
    """When a mock GWT workspace is provided, GWT indicator scores should reflect it."""
    mock_gw = MagicMock()
    mock_gw.get_statistics.return_value = {
        "cycle_count": 50,
        "broadcast": {
            "total_broadcasts": 20,
            "coverage_ratio": 0.75,
        },
        "ignition": {
            "total_ignitions": 15,
            "sustain_rate": 0.6,
        },
    }

    report = await assessment.run_full_assessment(global_workspace=mock_gw)

    # GWT indicators should have higher scores with an active workspace
    gb = report.indicator_results["global_broadcast"]
    assert gb.score > 0.0
    assert gb.evidence.get("module_active") is True

    ig = report.indicator_results["ignition_dynamics"]
    assert ig.score > 0.0
    assert ig.evidence.get("module_active") is True
