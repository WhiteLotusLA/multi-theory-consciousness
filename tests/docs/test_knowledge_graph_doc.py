"""docs/KNOWLEDGE_GRAPH.md names every node label with its properties, every
relationship type and every constraint the graph code defines.

Labels and relationship types are read from the Cypher in
``mtc/knowledge/graph_manager.py``, properties from the models in
``mtc.knowledge.models`` and constraint names from ``CONSTRAINTS``.
"""

import re
from pathlib import Path

from mtc.knowledge import models
from mtc.knowledge.graph_manager import CONSTRAINTS, SEED_CONCEPTS

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "KNOWLEDGE_GRAPH.md"
SOURCE = ROOT / "mtc" / "knowledge" / "graph_manager.py"


def _row_cells(text: str, first_cell: str):
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == first_cell:
            return cells
    return None


def test_every_label_relationship_and_constraint_documented():
    text = DOC.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    labels = set(re.findall(r"\(\w*:([A-Z][A-Za-z]+)\b", source))
    relationships = set(re.findall(r"-\[\w*:([A-Z][A-Z_]+)\b", source))
    names = set(
        re.match(r"CREATE (?:CONSTRAINT|INDEX) (\w+)", stmt).group(1)
        for stmt in CONSTRAINTS
    )
    assert labels and relationships and names

    for label in labels:
        cells = _row_cells(text, f"`{label}`")
        assert cells is not None, f"label {label} has no row"
        model = getattr(models, label)
        documented = set(re.findall(r"`(\w+)`", cells[1]))
        assert documented == set(model.model_fields), label
    for relationship in relationships:
        assert _row_cells(text, f"`{relationship}`") is not None, relationship
    for name in names:
        assert _row_cells(text, f"`{name}`") is not None, name
    for concept in SEED_CONCEPTS:
        assert f"`{concept}`" in text, concept
