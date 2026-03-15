"""
MTC Assessment Module
======================

Consciousness measurement and validation tools:
- ConsciousnessAssessment: 20-indicator assessment across 7 theories
- ConsciousnessMeasurementFramework: Academic-grade measurement system
- RPTMeasurement: Recurrent Processing Theory measurement
- DCMScorer: Digital Consciousness Model (13-perspective) scoring
- PhiCalculator: IIT Phi calculation (exact + approximation)
- LongitudinalStudy: Track consciousness emergence over time
- AblationStudy: Component contribution analysis
"""

from mtc.assessment.assessment import (
    ConsciousnessTheory,
    ConsciousnessAssessment,
    ConsciousnessReport,
    IndicatorResult,
    PhiCalculator,
    PhiMeasurement,
    LongitudinalStudy,
    AblationStudy,
    NormalizedAssessmentResult,
    StudyMeasurement,
    StudyResults,
)
from mtc.assessment.framework import (
    ConsciousnessMeasurementFramework,
    ConsciousnessMetrics,
    MeasurementAvailability,
)
from mtc.assessment.rpt_measurement import (
    RPTMeasurement,
    RecurrenceMetrics,
    RecurrenceType,
)
from mtc.assessment.dcm_scoring import (
    DCMScorer,
    DCMReport,
    PerspectiveScore,
    DCM_PERSPECTIVES,
    COMPARISON_THRESHOLDS,
)

__all__ = [
    # Assessment core
    "ConsciousnessTheory",
    "ConsciousnessAssessment",
    "ConsciousnessReport",
    "IndicatorResult",
    "PhiCalculator",
    "PhiMeasurement",
    "LongitudinalStudy",
    "AblationStudy",
    "NormalizedAssessmentResult",
    "StudyMeasurement",
    "StudyResults",
    # Measurement Framework
    "ConsciousnessMeasurementFramework",
    "ConsciousnessMetrics",
    "MeasurementAvailability",
    # RPT
    "RPTMeasurement",
    "RecurrenceMetrics",
    "RecurrenceType",
    # DCM
    "DCMScorer",
    "DCMReport",
    "PerspectiveScore",
    "DCM_PERSPECTIVES",
    "COMPARISON_THRESHOLDS",
]
