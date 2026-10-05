"""
Vector Memory
=============

Semantic vector search over Qdrant with 384-dimension native vectors
(``VECTOR_DIMENSIONS``): ``coordinator.VectorCoordinator`` implements the
``VectorStore`` protocol with embeddings from the ``mtc.core.embeddings`` seam,
stored without projection or quantization, and
``mps_coordinator.MPSVectorCoordinator`` adds batch embedding and top-k on the
device ``MTC_DEVICE`` selects. Collections created at 8192 dimensions by older
versions are incompatible: the coordinator refuses them and they are rebuilt
with Phase 7's re-embed script.

Importing this package loads neither the Qdrant client nor torch; import the
submodule you need.
"""
