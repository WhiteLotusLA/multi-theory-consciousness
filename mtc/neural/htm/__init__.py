"""
HTM (Hierarchical Temporal Memory) Module for Consciousness Research
==========================================================================

This module provides research-grade HTM implementations for consciousness modeling,
including MPS GPU-accelerated, C++ Metal, and Python fallback versions.

Available Implementations (in order of preference):
- MPSHTM: PyTorch MPS GPU acceleration (fastest)
- CppHtmInterface: C++ HTM with Metal acceleration
- ProductionHTM: Pure Python research implementation (fallback)
"""

import logging

logger = logging.getLogger(__name__)

# Try MPS-accelerated HTM first (fastest on Apple Silicon)
try:
    from .mps_htm import MPSHTM, MPSHTMConfig, MPS_AVAILABLE

    HTM_MPS_AVAILABLE = MPS_AVAILABLE
    if HTM_MPS_AVAILABLE:
        logger.info("MPS HTM with Metal Performance Shaders available")
except ImportError as e:
    HTM_MPS_AVAILABLE = False
    MPSHTM = None
    MPSHTMConfig = None
    logger.debug(f"MPS HTM not available: {e}")

# Try C++ HTM extension
try:
    from .cpp_integration import CppHtmInterface, HTMBuildSystem

    HTM_CPP_AVAILABLE = True
except ImportError as e:
    HTM_CPP_AVAILABLE = False
    CppHtmInterface = None
    HTMBuildSystem = None

# Always available Python implementations
from .htm_core import HTM, SDR, SpatialPooler, TemporalMemory
from .production_htm import ProductionHTM


def get_best_htm():
    """
    Get the best available HTM implementation.
    Returns the fastest available: MPS > C++ > Python
    """
    if HTM_MPS_AVAILABLE:
        return MPSHTM()
    elif HTM_CPP_AVAILABLE:
        from .cpp_integration import HTMCppConfig

        return CppHtmInterface(HTMCppConfig())
    else:
        return ProductionHTM()


__all__ = [
    # Best implementation selector
    "get_best_htm",
    # MPS GPU accelerated (fastest)
    "MPSHTM",
    "MPSHTMConfig",
    "HTM_MPS_AVAILABLE",
    # C++ accelerated (if available)
    "CppHtmInterface",
    "HTMBuildSystem",
    "HTM_CPP_AVAILABLE",
    # Python implementations
    "HTM",
    "ProductionHTM",
    "SDR",
    "SpatialPooler",
    "TemporalMemory",
]

# Log availability summary
if HTM_MPS_AVAILABLE:
    logger.info(
        "HTM: Using MPS GPU acceleration"
    )
elif HTM_CPP_AVAILABLE:
    logger.info("HTM: Using C++ with Eigen SIMD")
else:
    logger.warning("HTM: Using Python fallback (slower)")
