"""
MTC Assessment Module
======================

Consciousness measurement and validation tools:
- ConsciousnessAssessment: 20-indicator assessment across 7 theories
- ConsciousnessMeasurementFramework: Academic-grade measurement system
- RPTMeasurement: Recurrent Processing Theory measurement
- BenchmarkRunner: Full Bayesian DCM benchmark (Shiller & Duffy 2026)
- DCMEvaluator: LLM-based DCM indicator evaluator
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
from mtc.assessment.dcm_benchmark import (
    BenchmarkRunner,
    DCMBenchmarkResult,
    DCMEvidenceAdapter,
    ModelSpecManager,
    BayesianEngine,
    ParsedSpec,
    StanceNode,
    IndicatorScore,
)
from mtc.assessment.dcm_evaluator import (
    DCMEvaluator,
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
    # DCM Benchmark (Bayesian, Shiller & Duffy 2026)
    "BenchmarkRunner",
    "DCMBenchmarkResult",
    "DCMEvidenceAdapter",
    "ModelSpecManager",
    "BayesianEngine",
    "ParsedSpec",
    "StanceNode",
    "IndicatorScore",
    "DCMEvaluator",
]
