"""Knowledge graph manager over a recording fake driver (no Neo4j server)."""

import ast
import inspect
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

import pytest

from mtc.core.config import Settings
from mtc.knowledge import graph_manager as gm
from mtc.knowledge.config import Neo4jConfig
from mtc.knowledge.graph_manager import (
    CONSTRAINTS,
    SEED_CONCEPTS,
    KnowledgeGraphManager,
)
from mtc.knowledge.llm_concept_extraction import ConceptExtraction, ExtractedConcept
from mtc.knowledge.models import Concept, Person, Taught
from mtc.memory.protocols import GraphStore, StoreUnavailable

T0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
MODULE_PATH = Path(gm.__file__)

Handler = Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]


class ServiceUnavailable(Exception):
    """Named like the driver's connection error (matched by class name)."""


class CypherSyntaxError(Exception):
    """Named like a driver error that is not an outage."""


class FakeResult:
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = list(records)

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None


class FakeSession:
    def __init__(self, driver: "FakeDriver", database: Optional[str]) -> None:
        self.driver = driver
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, parameters: Optional[Dict[str, Any]] = None, **kw):
        params = dict(parameters or {})
        params.update(kw)
        self.driver.calls.append((query, params, self.database))
        if self.driver.fail is not None:
            raise self.driver.fail
        return FakeResult(self.driver.handler(query, params))


class FakeDriver:
    def __init__(self, handler: Optional[Handler] = None) -> None:
        self.handler: Handler = handler or (lambda query, params: [])
        self.calls: List[Tuple[str, Dict[str, Any], Optional[str]]] = []
        self.fail: Optional[BaseException] = None
        self.unreachable = False
        self.closed = False

    def session(self, database: Optional[str] = None) -> FakeSession:
        return FakeSession(self, database)

    def verify_connectivity(self) -> None:
        if self.unreachable:
            raise ServiceUnavailable("connection refused")

    def close(self) -> None:
        self.closed = True

    def queries(self) -> List[str]:
        return [query for query, _params, _db in self.calls]


class MergeGraph:
    """Answers the seed MERGE statements the way a database would."""

    def __init__(self) -> None:
        self.nodes: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def __call__(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        label = re.search(r"MERGE \(\w+:(\w+) \{name: \$name\}\)", query)
        if not label:
            return []
        key = (label.group(1), params["name"])
        created = key not in self.nodes
        if created:
            self.nodes[key] = dict(params["props"])
        return [{"created": created}]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def settings(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_CREATOR_NAME", "Alice")
    monkeypatch.setenv("MTC_CREATOR_RELATIONSHIP", "mentor")
    monkeypatch.setenv("MTC_SYSTEM_BIRTH_DATE", "2000-01-01")
    monkeypatch.setenv("MTC_NEO4J_PASSWORD", "placeholder")
    return Settings()


def _manager(settings, handler=None):
    driver = FakeDriver(handler)
    manager = KnowledgeGraphManager(
        config=Neo4jConfig("bolt://graph.example:1", "u", "p", "testdb"),
        driver=driver,
        settings=settings,
        clock=lambda: T0,
    )
    return manager, driver


def _all_string_constants() -> List[str]:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_satisfies_graph_store(settings):
    manager, _ = _manager(settings)
    assert isinstance(manager, GraphStore)


def test_bootstrap_constraints_statements(settings):
    statements = " ".join(CONSTRAINTS)
    assert all("IF NOT EXISTS" in s for s in CONSTRAINTS)
    assert all(not s.lstrip().upper().startswith("DROP") for s in CONSTRAINTS)
    assert re.search(r"\(\w+:Concept\) REQUIRE \w+\.name IS UNIQUE", statements)
    for label in ("Person", "Concept", "Memory", "Experience", "Emotion", "Topic"):
        assert re.search(
            rf"\(\w+:{label}\) REQUIRE \w+\.id IS UNIQUE", statements
        ), label
    assert re.search(
        r"\(\w+:Experience\) REQUIRE \w+\.conversation_id IS UNIQUE", statements
    )
    assert re.search(
        r"INDEX \w+ IF NOT EXISTS FOR \(\w+:Person\) ON \(\w+\.name\)", statements
    )
    manager, driver = _manager(settings)
    assert manager.bootstrap_constraints() == len(CONSTRAINTS)
    assert driver.queries() == list(CONSTRAINTS)
    assert {db for _q, _p, db in driver.calls} == {"testdb"}


def test_every_node_write_is_merge(settings):
    for text in _all_string_constants():
        assert not re.search(r"\bCREATE\s*\(", text), text
    manager, driver = _manager(
        settings, lambda q, p: [{"id": str(UUID(int=1)), "created": True}]
    )
    manager.create_person(Person(name="Alice", relationship="friend"))
    manager.create_concept(Concept(name="river", category="nature"))
    manager.merge_concept("bridge")
    manager.add_concept("stone")
    manager.link_concept_to_conversation("lantern", "conv-1", "Guest")
    manager.link_memory_to_concepts(3, "a bridge", ["bridge"])
    manager.seed_core_entities()
    writes = [q for q in driver.queries() if "SET" in q or "DELETE" in q]
    assert writes and all("MERGE" in q for q in writes)


def test_seed_from_settings(settings):
    manager, driver = _manager(settings, MergeGraph())
    counts = manager.seed_core_entities()
    assert counts == {"persons_created": 2, "concepts_created": len(SEED_CONCEPTS)}
    persons = [p for q, p, _ in driver.calls if ":Person" in q]
    by_name = {p["name"]: p["props"] for p in persons}
    assert set(by_name) == {settings.SYSTEM_NAME, "Alice"}
    creator = by_name["Alice"]
    assert creator["relationship"] == "mentor"
    assert creator["first_met"] == "2000-01-01T00:00:00+00:00"
    assert by_name[settings.SYSTEM_NAME]["relationship"] == "self"
    assert "first_met" not in by_name[settings.SYSTEM_NAME]
    for props in by_name.values():
        assert not any("bond" in k or "trust" in k or "comfort" in k for k in props)
    concepts = [p["name"] for q, p, _ in driver.calls if ":Concept" in q]
    assert concepts == list(SEED_CONCEPTS)
    assert SEED_CONCEPTS == ("consciousness", "emotions", "neural networks")


def test_seed_without_birth_date_or_creator(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_CREATOR_NAME", "Alice")
    manager, driver = _manager(Settings(), MergeGraph())
    counts = manager.seed_core_entities(include_creator=False)
    assert counts["persons_created"] == 1
    persons = [p for q, p, _ in driver.calls if ":Person" in q]
    assert [p["name"] for p in persons] == [Settings().SYSTEM_NAME]
    manager, driver = _manager(Settings(), MergeGraph())
    manager.seed_core_entities()
    creator = [p for q, p, _ in driver.calls if p.get("name") == "Alice"][0]
    assert "first_met" not in creator["props"]  # no birth date set, none assumed


def test_seed_twice_creates_nothing_new(settings):
    graph = MergeGraph()
    manager, driver = _manager(settings, graph)
    first = manager.seed_core_entities()
    nodes_after_first = dict(graph.nodes)
    second = manager.seed_core_entities()
    assert first["persons_created"] == 2
    assert second == {"persons_created": 0, "concepts_created": 0}
    assert graph.nodes == nodes_after_first
    for query in driver.queries():
        assert "MERGE" in query and "ON CREATE SET" in query


@pytest.mark.db
def test_seed_twice_creates_nothing_new_on_a_throwaway_graph(request):
    target = request.getfixturevalue("throwaway_neo4j")
    manager = KnowledgeGraphManager(
        config=Neo4jConfig("unused", "unused", "unused", target.database),
        driver=target.driver,
    )
    manager.bootstrap_constraints()
    first = manager.seed_core_entities()
    second = manager.seed_core_entities()
    assert first["persons_created"] >= 1 and first["concepts_created"] == 3
    assert second == {"persons_created": 0, "concepts_created": 0}


def test_who_taught_takes_learner_parameter(settings):
    rows = [{"teacher": "Alice", "when": "x", "method": "conversation"}]
    manager, driver = _manager(settings, lambda q, p: rows)
    assert manager.who_taught_concept("rivers") == rows
    query, params, _ = driver.calls[-1]
    assert params == {"concept_name": "rivers", "learner": settings.SYSTEM_NAME}
    assert "$learner" in query and settings.SYSTEM_NAME not in query
    manager.who_taught_concept("rivers", learner="Guest")
    assert driver.calls[-1][1]["learner"] == "Guest"
    signature = inspect.signature(KnowledgeGraphManager.who_taught_concept)
    assert signature.parameters["learner"].default is None


def test_link_memory_to_concepts_merges(settings):
    manager, driver = _manager(
        settings, lambda q, p: [{"linked": 2, "names": ["river", "bridge"]}]
    )
    linked = manager.link_memory_to_concepts(
        11, "a bridge over the river" * 20, ["river", "bridge", "unknown"], 0.6
    )
    assert linked == 2
    query, params, _ = driver.calls[-1]
    assert "MERGE (m:Memory {id: $memory_id})" in query
    assert "MERGE (m)-[r:REFERENCES_IN]->(c)" in query
    assert "MATCH (c:Concept {name: concept_name})" in query  # unknown skipped
    assert params["memory_id"] == str(UUID(int=11))
    assert params["memory_props"]["postgres_id"] == 11
    assert len(params["preview"]) == 200
    assert params["names"] == ["river", "bridge", "unknown"]
    assert params["link_props"]["relevance"] == 0.6
    assert params["link_props"]["extraction_method"] == "keyword"
    calls = len(driver.calls)
    manager.link_memory_to_concepts(12, "no concepts", [])
    assert len(driver.calls) == calls + 1  # the memory node is still refreshed


def test_mention_and_relation_writes(settings):
    manager, driver = _manager(settings, lambda q, p: [{"strength": 0.9}])
    manager.link_concept_to_conversation("lantern", "conv-1", "Alice")
    query, params, _ = driver.calls[-1]
    assert "MERGE (e:Experience {conversation_id: $conversation_id})" in query
    assert "MENTIONED_IN {speaker: $speaker}" in query
    assert params["speaker"] == "Alice" and params["conversation_id"] == "conv-1"
    assert "Alice" not in params["props"].get("description", "")
    assert manager.relate_concepts("a", "b", "similar_to", 0.9) is True
    assert manager.relate_concepts("a", "a") is False  # no call for a self-loop
    assert "RELATES_TO" in driver.calls[-1][0]
    manager_none, _ = _manager(settings, lambda q, p: [])
    assert manager_none.relate_concepts("a", "missing") is False


def test_process_llm_extraction_maps_categories(settings):
    seen: List[Dict[str, Any]] = []

    def handler(query, params):
        seen.append(params)
        if "MERGE (c:Concept {name: $name})" in query:
            return [{"created": params["name"] != "river"}]
        if "REFERENCES_IN" in query:
            return [{"linked": 1}]
        return []

    manager, driver = _manager(settings, handler)
    extraction = ConceptExtraction(
        concepts=[
            ExtractedConcept("river", "place", 0.9),
            ExtractedConcept("curiosity", "Emotion", 0.7),
            ExtractedConcept("odd", "spaceship", None),
        ],
        status="measured",
    )
    stats = manager.process_llm_extraction(extraction, 5, memory_content="text")
    merged = {p["name"]: p["props"] for p in seen if "props" in p and "name" in p}
    assert merged["river"]["category"] == "place"
    assert merged["curiosity"]["category"] == "emotion"
    assert merged["odd"]["category"] == "general"
    assert merged["river"]["importance"] == 0.9
    assert "importance" not in merged["odd"]
    assert stats["concepts_created"] == 2 and stats["concepts_found"] == 1
    assert stats["memory_linked"] is True and stats["status"] == "measured"
    links = [p for q, p, _ in driver.calls if "REFERENCES_IN" in q]
    assert {p["link_props"]["extraction_method"] for p in links} == {"llm"}


def test_llm_extraction_writes_no_importance_without_relevance(settings):
    merges: List[Dict[str, Any]] = []

    def handler(query, params):
        if "MERGE (c:Concept {name: $name})" in query:
            merges.append(params)
            return [{"created": True}]
        if "REFERENCES_IN" in query:
            return [{"linked": 1}]
        return []

    manager, _ = _manager(settings, handler)
    extraction = ConceptExtraction(
        concepts=[
            ExtractedConcept("lantern", "object", None),
            ExtractedConcept("harbour", "place", 0.8),
        ],
        status="measured",
    )
    manager.process_llm_extraction(extraction, 7, memory_content="text")
    props = {params["name"]: params["props"] for params in merges}
    assert "importance" not in props["lantern"]
    assert props["harbour"]["importance"] == pytest.approx(0.8)


def test_process_llm_extraction_relationships_need_existing_people(settings):
    def handler(query, params):
        if "TAUGHT" in query:
            return [{"kind": "TAUGHT"}] if params["teacher"] == "Alice" else []
        if "LEARNED" in query or "RELATES_TO" in query:
            return [{"kind": "x"}]
        return [{"created": True}]

    manager, driver = _manager(settings, handler)
    stats = manager.process_llm_extraction(
        {
            "concepts": [{"name": "rivers", "category": "topic", "relevance": 0.5}],
            "relationships": [
                {"from": "Alice", "to": "rivers", "type": "taught", "strength": 0.8},
                {"from": "Operator", "to": "rivers", "type": "teaches"},
                {"from": "x", "to": "rivers", "type": "learned", "strength": 0.6},
                {"from": "rivers", "to": "bridges", "type": "causes"},
                {"from": "rivers"},
            ],
        },
        9,
        learner="Guest",
    )
    assert stats["relationships_created"] == 3
    assert stats["relationships_skipped"] == 2
    learned = [p for q, p, _ in driver.calls if "LEARNED" in q and "TAUGHT" not in q]
    # one edge for the learned relationship, one for the learner of the taught one
    assert len(learned) == 2 and all(p["learner"] == "Guest" for p in learned)
    proficiencies = sorted(
        str(p["props"].get("proficiency")) for p in learned
    )  # the taught edge measured no proficiency, so it carries none
    assert proficiencies == ["0.6", "None"]
    taught = [p for q, p, _ in driver.calls if "TAUGHT" in q]
    assert [p["teacher"] for p in taught] == ["Alice", "Operator"]
    assert all(p["concept"] == "rivers" for p in taught)


def test_process_llm_extraction_missing_strength_stays_unset(settings):
    def handler(query, params):
        if "TAUGHT" in query or "LEARNED" in query or "RELATES_TO" in query:
            return [{"kind": "x"}]
        return [{"created": True}]

    manager, driver = _manager(settings, handler)
    stats = manager.process_llm_extraction(
        {
            "concepts": [{"name": "rivers", "category": "topic", "relevance": 0.5}],
            "relationships": [
                {"from": "Alice", "to": "rivers", "type": "taught"},
                {"from": "x", "to": "rivers", "type": "learned", "strength": None},
                {"from": "rivers", "to": "bridges", "type": "causes"},
            ],
        },
        4,
        learner="Guest",
    )
    assert stats["relationships_created"] == 3
    taught = [p for q, p, _ in driver.calls if "TAUGHT" in q]
    assert len(taught) == 1 and "effectiveness" not in taught[0]["props"]
    learned = [p for q, p, _ in driver.calls if "LEARNED" in q and "TAUGHT" not in q]
    assert len(learned) == 2
    for params in learned:
        for key in ("confidence", "proficiency", "depth"):
            assert key not in params["props"]
    related = [(q, p) for q, p, _ in driver.calls if "RELATES_TO" in q]
    assert len(related) == 1
    query, params = related[0]
    assert params["strength"] is None and params["relationship"] == "causes"
    # an existing edge keeps its strength; a new one gets no property
    assert "coalesce($strength, r.strength)" in query


def test_unmeasured_relations_sort_after_measured(settings):
    query = gm._RELATED
    assert "ORDER BY strength IS NULL, strength DESC" in query


def test_process_llm_extraction_writes_nothing_without_a_measurement(settings):
    manager, driver = _manager(settings)
    for status in ("no_answer", "unparseable"):
        stats = manager.process_llm_extraction(ConceptExtraction([], status), 3)
        assert stats["status"] == status and stats["concepts_created"] == 0
    assert driver.calls == []


def test_add_concept_is_sync_and_documented(settings):
    assert not inspect.iscoroutinefunction(KnowledgeGraphManager.add_concept)
    assert "asyncio.to_thread" in (KnowledgeGraphManager.add_concept.__doc__ or "")
    manager, driver = _manager(settings)
    assert manager.add_concept("stone", source="web:example") is None
    query, params, _ = driver.calls[-1]
    assert "ON CREATE SET" in query and "ON MATCH SET" in query
    assert params["props"]["category"] == "web:example"
    assert set(params["props"]) <= set(Concept.model_fields)


def test_merge_paths_write_the_concept_model(settings):
    manager, driver = _manager(settings, lambda q, p: [{"created": True}])
    manager.merge_concept("bridge", "a crossing", "structure")
    manager.add_concept("stone")
    manager.link_concept_to_conversation("lantern", "conv-1", "Guest")
    manager.process_llm_extraction(
        ConceptExtraction([ExtractedConcept("river", "place", 0.5)], "measured"), 1
    )
    expected = set(Concept.model_fields) - {"understanding_level", "last_mentioned"}
    for query, params, _ in driver.calls:
        if "MERGE (c:Concept" in query:
            assert expected <= set(params["props"]) <= set(Concept.model_fields)


def test_delete_memory_nodes(settings):
    manager, driver = _manager(settings, lambda q, p: [{"removed": 1}])
    assert manager.delete_memory_nodes([1, 2]) == 1
    query, params, _ = driver.calls[-1]
    assert "DETACH DELETE" in query
    assert params["ids"] == [str(UUID(int=1)), str(UUID(int=2))]
    calls = len(driver.calls)
    assert manager.delete_memory_nodes([]) == 0
    assert len(driver.calls) == calls


def test_reads_shape_rows(settings):
    rows = [{"concept": "b", "relationship": "similar_to", "strength": 0.9}]
    manager, driver = _manager(settings, lambda q, p: rows)
    assert manager.get_related_concepts("a", limit=3) == rows
    assert driver.calls[-1][1] == {"concept_name": "a", "limit": 3}
    manager, driver = _manager(settings, lambda q, p: [{"name": "a"}, {"name": "b"}])
    assert manager.list_concept_names() == ["a", "b"]
    manager.list_concept_names(limit=-4)
    assert driver.calls[-1][1] == {"limit": 0}
    manager, driver = _manager(settings, lambda q, p: [])
    assert manager.get_person_by_name("Alice") is None
    assert manager.get_concept_by_name("river") is None
    assert manager.get_memories_referencing_concept("river") == []
    assert manager.get_concepts_in_memory(4) == []
    assert driver.calls[-1][1] == {"memory_id": str(UUID(int=4))}
    manager.get_knowledge_network("river", depth=99)
    assert "[*1..5]" in driver.calls[-1][0]


def test_relationship_writers_take_ids(settings):
    manager, driver = _manager(settings, lambda q, p: [{"kind": "TAUGHT"}])
    teacher, concept = UUID(int=1), UUID(int=2)
    assert manager.create_taught_relationship(
        teacher, concept, Taught(when=T0, method="conversation")
    )
    query, params, _ = driver.calls[-1]
    assert params["teacher_id"] == str(teacher) and params["concept_id"] == str(concept)
    assert "MERGE (teacher)-[r:TAUGHT]->(concept)" in query


def test_status_when_unreachable(settings, monkeypatch):
    import sys
    import types

    driver = FakeDriver()
    driver.unreachable = True
    monkeypatch.setitem(
        sys.modules,
        "neo4j",
        types.SimpleNamespace(
            GraphDatabase=types.SimpleNamespace(driver=lambda uri, auth: driver)
        ),
    )
    manager = KnowledgeGraphManager(
        config=Neo4jConfig("bolt://graph.example:1", "u", "secret-value", "db"),
        settings=settings,
    )
    assert manager.status()["available"] is False  # never connected yet
    with pytest.raises(StoreUnavailable):
        manager.connect()
    status = manager.status()
    assert status["backend"] == "neo4j" and status["available"] is False
    assert "connection refused" in status["last_error"]
    assert status["uri"] == "bolt://graph.example:1"
    assert "secret-value" not in repr(status)
    assert driver.closed is True  # a driver it built that never connected is closed

    injected = FakeDriver()
    injected.unreachable = True
    owned_elsewhere = KnowledgeGraphManager(
        config=Neo4jConfig("bolt://graph.example:1", "u", "p", "db"),
        driver=injected,
        settings=settings,
    )
    with pytest.raises(StoreUnavailable):
        owned_elsewhere.connect()
    assert injected.closed is False  # the caller owns an injected driver

    live, fake = _manager(settings)
    fake.fail = ServiceUnavailable("dropped")
    with pytest.raises(StoreUnavailable):
        live.list_concept_names()
    assert live.status()["available"] is False
    fake.fail = CypherSyntaxError("bad query")
    with pytest.raises(CypherSyntaxError):
        live.list_concept_names()  # not an outage: the caller's problem
    fake.fail = None
    assert live.list_concept_names() == []
    assert live.status()["available"] is True and live.status()["last_error"] is None


def test_injected_driver_is_not_closed(settings):
    manager, driver = _manager(settings)
    manager.list_concept_names()
    manager.close()
    assert driver.closed is False
    assert manager.status()["available"] is True


def test_missing_neo4j_names_the_extra(settings, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "neo4j", None)
    manager = KnowledgeGraphManager(settings=settings)
    with pytest.raises(ImportError) as info:
        manager.list_concept_names()
    assert "[neo4j]" in str(info.value)


def test_queries_are_parameterized(settings):
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            assert not re.search(r"\b(MATCH|MERGE|SET|RETURN)\b", text), text
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            assert not isinstance(node.left, ast.Constant), ast.unparse(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
        ):
            assert not isinstance(node.func.value, ast.Constant), ast.unparse(node)

    tricky = "river' }) DETACH DELETE (n) //"
    manager, driver = _manager(settings, lambda q, p: [])
    manager.merge_concept(tricky)
    manager.add_concept(tricky)
    manager.link_concept_to_conversation(tricky, tricky, tricky)
    manager.link_memory_to_concepts(1, tricky, [tricky])
    manager.relate_concepts(tricky, "b")
    manager.get_related_concepts(tricky)
    manager.who_taught_concept(tricky, learner=tricky)
    manager.get_knowledge_network(tricky)
    for query, _params, _db in driver.calls:
        assert tricky not in query


def test_module_docstring_names_private_names():
    doc = gm.__doc__ or ""
    assert "Person" in doc and "MENTIONED_IN" in doc and "real" in doc
