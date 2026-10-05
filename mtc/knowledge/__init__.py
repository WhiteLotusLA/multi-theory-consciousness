"""
Knowledge Graph
===============

The semantic knowledge network over Neo4j: ``Neo4jConfig`` (connection
settings read from ``NEO4J_*`` at call time), the node and relationship models,
``KnowledgeGraphManager`` (the ``GraphStore`` implementation with ``MERGE``
writes, the ``CONSTRAINTS`` schema artifact and the Settings-driven seeds),
``ConceptExtractor`` (keyword matching against known concepts) and
``LLMConceptExtractor`` (concept extraction through ``BrainProtocol``).

Importing this package loads no Neo4j client; the driver is imported when a
manager first connects. Person nodes, conversation mentions and extracted
person concepts hold real names at runtime and are private data.
"""

from mtc.knowledge.concept_extraction import ConceptExtractor, create_concept_extractor
from mtc.knowledge.config import Neo4jConfig
from mtc.knowledge.graph_manager import (
    CONSTRAINTS,
    SEED_CONCEPTS,
    KnowledgeGraphManager,
)
from mtc.knowledge.llm_concept_extraction import (
    ConceptExtraction,
    ExtractedConcept,
    LLMConceptExtractor,
)
from mtc.knowledge.models import (
    Concept,
    Emotion,
    Experience,
    Learned,
    Memory,
    MentionedIn,
    Person,
    ReferencesIn,
    RelatesTo,
    Taught,
    Topic,
)

__all__ = [
    "CONSTRAINTS",
    "SEED_CONCEPTS",
    "Concept",
    "ConceptExtraction",
    "ConceptExtractor",
    "Emotion",
    "Experience",
    "ExtractedConcept",
    "KnowledgeGraphManager",
    "LLMConceptExtractor",
    "Learned",
    "Memory",
    "MentionedIn",
    "Neo4jConfig",
    "Person",
    "ReferencesIn",
    "RelatesTo",
    "Taught",
    "Topic",
    "create_concept_extractor",
]
