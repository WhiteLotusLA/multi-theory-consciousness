"""Neo4j connection settings: read at call time, masked, lazily connected."""

import logging
import os
import sys

import pytest

from mtc.core.config import Settings
from mtc.knowledge import config as knowledge_config
from mtc.knowledge.config import Neo4jConfig


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(knowledge_config, "_WARNED", set())
    return tmp_path


def test_config_from_settings_at_call_time(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_NEO4J_URI", "bolt://graph.example:1234")
    monkeypatch.setenv("MTC_NEO4J_USER", "reader")
    monkeypatch.setenv("MTC_NEO4J_PASSWORD", "placeholder")
    monkeypatch.setenv("MTC_NEO4J_DATABASE", "knowledge")
    first = Neo4jConfig.from_settings(Settings())
    assert first.uri == "bolt://graph.example:1234"
    assert first.user == "reader"
    assert first.password == "placeholder"
    assert first.database == "knowledge"

    monkeypatch.setenv("MTC_NEO4J_DATABASE", "other")
    fresh = Settings()
    monkeypatch.setattr(knowledge_config, "get_settings", lambda: fresh)
    assert Neo4jConfig.from_settings().database == "other"
    assert first.database == "knowledge"  # an existing config is not mutated


def test_explicit_values_win():
    config = Neo4jConfig("bolt://x.example:1", "u", "p", "d")
    assert (config.uri, config.user, config.password, config.database) == (
        "bolt://x.example:1",
        "u",
        "p",
        "d",
    )


def test_masked_hides_password(clean_env):
    config = Neo4jConfig("bolt://x.example:1", "u", "secret-value", "d")
    masked = config.masked()
    assert masked["password"] == "***"
    assert "secret-value" not in repr(masked)
    assert "secret-value" not in repr(config)
    assert masked["uri"] == "bolt://x.example:1"
    assert masked["user"] == "u" and masked["database"] == "d"
    assert Neo4jConfig("bolt://x.example:1", "u", "", "d").masked()["password"] == ""


def test_empty_password_warns_once(clean_env, caplog):
    caplog.set_level(logging.WARNING, logger="mtc.knowledge.config")
    Neo4jConfig.from_settings(Settings())
    Neo4jConfig.from_settings(Settings())
    Neo4jConfig("bolt://x.example:1", "u", "", "d")
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "NEO4J_PASSWORD" in r.getMessage()
    ]
    # one warning per (uri, user) pair, not one per config built
    assert len(warnings) == 2
    caplog.clear()
    Neo4jConfig("bolt://x.example:1", "u", "set", "d")
    assert not caplog.records


def test_missing_neo4j_names_the_extra(clean_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "neo4j", None)
    config = Neo4jConfig("bolt://x.example:1", "u", "p", "d")
    with pytest.raises(ImportError) as info:
        config.create_driver()
    assert "[neo4j]" in str(info.value) and "'neo4j'" in str(info.value)


def test_create_driver_passes_uri_and_auth(clean_env, monkeypatch):
    import types

    captured = {}

    class GraphDatabase:
        @staticmethod
        def driver(uri, auth=None, **kwargs):
            captured.update(uri=uri, auth=auth, kwargs=kwargs)
            return "driver"

    monkeypatch.setitem(
        sys.modules, "neo4j", types.SimpleNamespace(GraphDatabase=GraphDatabase)
    )
    config = Neo4jConfig("bolt://x.example:1", "u", "p", "d")
    assert config.create_driver(connection_timeout=2.0) == "driver"
    assert captured == {
        "uri": "bolt://x.example:1",
        "auth": ("u", "p"),
        "kwargs": {"connection_timeout": 2.0},
    }
