"""
HTM (Hierarchical Temporal Memory) Module for Consciousness Research
=====================================================================

Pure NumPy hierarchical temporal memory for consciousness modeling:

- ``ProductionHTM``: spatial pooler plus temporal memory, 4,096 columns by
  default (``HTM_COLUMNS`` and ``HTM_CELLS_PER_COLUMN`` size the one the memory
  consolidator builds)
- ``HTM``, ``SDR``, ``SpatialPooler`` and ``TemporalMemory``: the compact
  reference implementation
- ``MemoryConsolidator``, ``EpisodicMemory`` and ``SemanticPattern``: the
  sleep-like replay that turns episodes into semantic patterns

There is no accelerated or native variant: ``get_best_htm`` returns
``ProductionHTM``.
"""

from typing import Optional

from .htm_core import HTM, SDR, SpatialPooler, TemporalMemory
from .memory_consolidator import (
    ConsolidatorConfig,
    EpisodicMemory,
    MemoryConsolidator,
    SemanticPattern,
)
from .production_htm import HTMConfig, ProductionHTM


def get_best_htm(config: Optional[HTMConfig] = None) -> ProductionHTM:
    """Return a ``ProductionHTM`` built from ``config`` (the default config when
    None)."""
    return ProductionHTM(config)


__all__ = [
    "get_best_htm",
    "HTM",
    "HTMConfig",
    "ProductionHTM",
    "SDR",
    "SpatialPooler",
    "TemporalMemory",
    "ConsolidatorConfig",
    "EpisodicMemory",
    "MemoryConsolidator",
    "SemanticPattern",
]
