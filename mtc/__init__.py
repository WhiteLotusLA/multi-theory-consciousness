"""
Multi-Theory Consciousness Framework (MTC)
============================================

A research-grade framework for implementing and measuring consciousness
indicators across multiple theoretical perspectives.

Implements 7 consciousness theories:
- Global Workspace Theory (GWT) - Baars, 1988
- Integrated Information Theory (IIT) - Tononi, 2004
- Attention Schema Theory (AST) - Graziano, 2013
- Higher-Order Thought Theory (HOT) - Rosenthal, 2005
- Free Energy Principle (FEP) - Friston, 2010
- Recurrent Processing Theory (RPT) - Lamme, 2006
- Beautiful Loop Theory (BLT) - Laukkonen, Friston & Chandaria, 2025

Based on Butlin et al. (2023, 2025): "Consciousness in Artificial
Intelligence: Insights from the Science of Consciousness"
"""

__version__ = "0.1.0"
__project__ = "Multi-Theory Consciousness Framework"

from mtc.assessment.assessment import (
    ConsciousnessTheory,
    ConsciousnessAssessment,
    ConsciousnessReport,
    IndicatorResult,
    PhiCalculator,
    PhiMeasurement,
    LongitudinalStudy,
    Phase6AblationStudy,
    NormalizedAssessmentResult,
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
)
from mtc.core.config import Settings, get_settings

__all__ = [
    # Version info
    "__version__",
    "__project__",
    # Assessment
    "ConsciousnessTheory",
    "ConsciousnessAssessment",
    "ConsciousnessReport",
    "IndicatorResult",
    "PhiCalculator",
    "PhiMeasurement",
    "LongitudinalStudy",
    "Phase6AblationStudy",
    "NormalizedAssessmentResult",
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
    # Config
    "Settings",
    "get_settings",
]
