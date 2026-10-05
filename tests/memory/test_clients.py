"""The MongoDB client factory builds clients that do not connect at construction.

A fake ``pymongo`` module in ``sys.modules`` records the arguments each client
receives, so no driver is imported and no socket is opened.
"""

import os
import sys
import types
from typing import Any, Dict, List

import pytest

from mtc.core.config import Settings
from mtc.memory import clients


@pytest.fixture
def settings(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env file here
    return Settings()


@pytest.fixture
def built(monkeypatch) -> List[Dict[str, Any]]:
    """Install a fake pymongo; return the argument dicts its clients receive."""
    calls: List[Dict[str, Any]] = []

    class MongoClient(dict):
        def __init__(self, **kwargs: Any) -> None:
            calls.append(dict(kwargs))
            super().__init__()

        def __getitem__(self, name: str) -> str:
            return f"database:{name}"

    module = types.ModuleType("pymongo")
    module.MongoClient = MongoClient
    monkeypatch.setitem(sys.modules, "pymongo", module)
    return calls


def test_client_does_not_connect_by_default(built, settings):
    clients.create_mongo_client(settings)
    assert built[-1]["connect"] is False
    # an unrelated override keeps the default
    clients.create_mongo_client(settings, serverSelectionTimeoutMS=50)
    assert built[-1]["connect"] is False
    assert built[-1]["serverSelectionTimeoutMS"] == 50


def test_explicit_connect_override_wins(built, settings):
    clients.create_mongo_client(settings, connect=True)
    assert built[-1]["connect"] is True
    clients.create_mongo_client(settings, connect=False)
    assert built[-1]["connect"] is False


def test_mongo_database_without_client_builds_an_unconnected_one(built, settings):
    assert clients.mongo_database(settings=settings) == (
        f"database:{settings.MONGODB_DB}"
    )
    assert clients.mongo_database(settings=settings, name="x") == "database:x"
    assert [call["connect"] for call in built] == [False, False]
    # a passed client is used as is: nothing new is built
    assert clients.mongo_database({"x": "given"}, settings=settings, name="x") == (
        "given"
    )
    assert len(built) == 2
