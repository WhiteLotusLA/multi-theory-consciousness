"""MONGODB_AUTH_SOURCE reaches the MongoDB client: an explicit authentication
database when set, and no separate one when it is empty (the client then
authenticates against MONGODB_DB, the database MONGODB_URL names)."""

import os
import sys
import types

import pytest

from mtc.core.config import Settings
from mtc.memory import clients


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    captured = {}

    class MongoClient:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "pymongo", types.SimpleNamespace(MongoClient=MongoClient)
    )
    return captured


def test_auth_source_passed_when_set(recorded):
    settings = Settings(
        MONGODB_USER="operator",
        MONGODB_PASSWORD="placeholder-value",
        MONGODB_AUTH_SOURCE="admin",
    )
    clients.create_mongo_client(settings)
    assert recorded["username"] == "operator"
    assert recorded["authSource"] == "admin"
    assert recorded["authSource"] != settings.MONGODB_DB


def test_no_auth_source_when_empty(recorded):
    """'No auth source' means no separate authentication database, not an
    absent keyword. With credentials and an empty MONGODB_AUTH_SOURCE the
    client authenticates against MONGODB_DB, the database MONGODB_URL names.
    The client is built from host and port rather than from the URL, where
    the driver would fall back to its own admin database, so that database is
    passed explicitly as authSource; tests/core/test_lazy_store_imports.py
    pins the same value. Without credentials no authentication keyword is
    passed at all."""
    # No credentials: no authentication keywords at all, whatever the key says.
    clients.create_mongo_client(Settings(MONGODB_AUTH_SOURCE="admin"))
    assert "authSource" not in recorded and "username" not in recorded
    # Credentials with an empty key: no separate authentication database; the
    # client authenticates against the database MONGODB_URL names.
    settings = Settings(MONGODB_USER="operator", MONGODB_PASSWORD="placeholder-value")
    assert settings.MONGODB_AUTH_SOURCE == ""
    clients.create_mongo_client(settings)
    assert recorded["authSource"] == settings.MONGODB_DB
    assert settings.MONGODB_URL.endswith(f"/{settings.MONGODB_DB}")


def test_auth_source_description_matches_client():
    """The documented meaning of an empty key is the value the client passes:
    MONGODB_DB. The driver's own default is not what the client uses (it is
    built from host and port, where that default is the admin database), so
    the description must not present the fallback as the driver default."""
    description = Settings.model_fields["MONGODB_AUTH_SOURCE"].description
    assert "MONGODB_DB" in description
    assert "driver default" not in description.lower()
