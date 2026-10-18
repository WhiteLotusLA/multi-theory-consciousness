"""
MTC Assessment Module
======================

Consciousness measurement and validation tools:
- ConsciousnessAssessment: 30 positive and 3 negative indicators across 8
  theories, with measured, absent and insufficient_data statuses
- ConsciousnessMeasurementFramework: the legacy 14-metric framework
- RPTMeasurement: Recurrent Processing Theory measurement
- BenchmarkRunner: DCM benchmark (Shiller & Duffy 2026); reports the
  collected indicator evidence, with an experimental posterior on request
- DCMEvaluator: LLM-based DCM indicator evaluator
- ReferenceDCMAdapter: DCM evidence adapter for MTC's own module set
- CausalEmergenceAnalyzer: Hoel Effective Information + CE 2.0
- PhiCalculator: exact IIT 4.0 Phi through the PyPhi bridge when a TPM is
  supplied, otherwise a labelled spectral integration proxy
- LongitudinalStudy: Track consciousness emergence over time
- AblationStudy: Component contribution analysis

Phi has two paths into the assessment: a PhiTracker passed as ``phi_tracker``
gives exact IIT 4.0 Phi of the module graph (a module-graph integration
proxy), and the PhiCalculator works on supplied neural states. Importing this
package never imports PyPhi.
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
from mtc.assessment.dcm_reference_adapter import (
    KNOWN_ZEROS,
    ReferenceDCMAdapter,
    build_default_adapter,
)
from mtc.assessment.causal_emergence import (
    CausalEmergenceAnalyzer,
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
    # DCM benchmark (Shiller & Duffy 2026): evidence; experimental posterior
    "BenchmarkRunner",
    "DCMBenchmarkResult",
    "DCMEvidenceAdapter",
    "ModelSpecManager",
    "BayesianEngine",
    "ParsedSpec",
    "StanceNode",
    "IndicatorScore",
    "DCMEvaluator",
    "ReferenceDCMAdapter",
    "build_default_adapter",
    "KNOWN_ZEROS",
    # Causal Emergence (v0.2.0)
    "CausalEmergenceAnalyzer",
]
