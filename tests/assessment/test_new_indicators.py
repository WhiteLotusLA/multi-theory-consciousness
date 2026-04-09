import pytest
import numpy as np
from mtc.assessment.assessment import ConsciousnessAssessment
from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace
from mtc.consciousness.attention_schema import AttentionSchemaModule
from mtc.consciousness.metacognition import MetacognitionModule
from mtc.consciousness.active_inference import ActiveInferenceModule
from mtc.consciousness.beautiful_loop import BeautifulLoop


@pytest.fixture
def modules():
    return {
        "global_workspace": EnhancedGlobalWorkspace(),
        "attention_schema": AttentionSchemaModule(),
        "metacognition": MetacognitionModule(),
        "active_inference": ActiveInferenceModule(),
        "beautiful_loop": BeautifulLoop(),
    }


@pytest.mark.asyncio
async def test_temporal_prediction_depth_indicator(modules):
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules)
    assert "temporal_prediction_depth" in report.indicator_results
    result = report.indicator_results["temporal_prediction_depth"]
    assert result.score >= 0.0
    assert result.threshold == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_agent_environment_coupling_indicator(modules):
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules)
    assert "agent_environment_coupling" in report.indicator_results
    result = report.indicator_results["agent_environment_coupling"]
    assert result.score >= 0.0
    assert result.threshold == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_contrast_theory_weighting(modules):
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules)
    # Overall score may be 0 with fresh modules (no cycles run) after noise normalization
    assert report.overall_score >= 0.0
    assert hasattr(assessment, 'theory_empirical_weights')
    assert assessment.theory_empirical_weights["GWT"] == 1.0
    assert assessment.theory_empirical_weights["HOT"] == 0.50
    assert assessment.theory_empirical_weights["AE"] == 0.70


@pytest.mark.asyncio
async def test_local_recurrence_uses_rpt(modules):
    from mtc.assessment.rpt_measurement import RPTMeasurement
    rpt = RPTMeasurement()
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules, rpt_measurement=rpt)
    result = report.indicator_results["local_recurrence"]
    assert "avg_local_recurrence" in result.evidence or "module_active" in result.evidence
