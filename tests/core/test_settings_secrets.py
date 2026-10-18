"""Secret Settings fields stay out of repr and dumps; store URLs quote their
credentials."""

import os
from urllib.parse import unquote, urlsplit

import pytest

from mtc.core.config import Settings
from mtc.core.env_example import SECRET_MARKERS
from mtc.core.env_export import SECRET_COMPUTED_FIELDS, export_lines


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


SYNTHETIC = {
    "POSTGRES_PASSWORD": "pgsecretvalue001",
    "MONGODB_PASSWORD": "mongosecretvalue002",
    "REDIS_PASSWORD": "redissecretvalue003",
    "NEO4J_PASSWORD": "graphsecretvalue004",
    "API_KEY": "apikeysecretvalue005",
    "ANTHROPIC_API_KEY": "vendorkeysecretvalue006",
    "SPEAKER_TOKENS": "Alice:speakertokenvalue007",
}


def _secret_named_fields():
    return [
        name
        for name, info in Settings.model_fields.items()
        if info.annotation is str and any(m in name for m in SECRET_MARKERS)
    ]


def test_repr_and_dumps_hold_no_secret(clean_env):
    settings = Settings(MONGODB_USER="Operator", **SYNTHETIC)
    # Attribute access still returns the plain value.
    for name, value in SYNTHETIC.items():
        assert getattr(settings, name) == value, name
    views = {
        "repr": repr(settings),
        "str": str(settings),
        "model_dump": repr(settings.model_dump()),
        "model_dump_json": settings.model_dump_json(),
    }
    for view, text in views.items():
        for name, value in SYNTHETIC.items():
            assert value not in text, (view, name)
            assert value.split(":")[-1] not in text, (view, name)
    dumped = settings.model_dump()
    for name in list(SYNTHETIC) + list(SECRET_COMPUTED_FIELDS):
        assert name not in dumped, name
    # Ordinary fields are still dumped.
    assert dumped["POSTGRES_DB"] == "mtc_consciousness"
    assert dumped["LLM_URL"] == settings.LLM_URL


def test_every_secret_named_field_is_hidden():
    names = _secret_named_fields()
    assert set(SYNTHETIC) <= set(names)
    for name in names:
        info = Settings.model_fields[name]
        assert info.repr is False, name
        assert info.exclude is True, name


def test_store_urls_quote_credentials(clean_env):
    password = "p@ss/w:1"
    user = "op:er@tor"
    settings = Settings(
        POSTGRES_USER=user,
        POSTGRES_PASSWORD=password,
        MONGODB_USER=user,
        MONGODB_PASSWORD=password,
        REDIS_PASSWORD=password,
    )
    # The configured hosts are the Settings defaults.
    expectations = {
        "POSTGRES_URL": (settings.POSTGRES_HOST, settings.POSTGRES_PORT, user),
        "MONGODB_URL": (settings.MONGODB_HOST, settings.MONGODB_PORT, user),
        "REDIS_URL": (settings.REDIS_HOST, settings.REDIS_PORT, ""),
    }
    for name, (host, port, expected_user) in expectations.items():
        parts = urlsplit(getattr(settings, name))
        assert parts.hostname == host, name
        assert parts.port == port, name
        assert unquote(parts.password) == password, name
        assert unquote(parts.username or "") == expected_user, name
    assert urlsplit(settings.POSTGRES_URL).path == f"/{settings.POSTGRES_DB}"
    assert urlsplit(settings.MONGODB_URL).path == f"/{settings.MONGODB_DB}"
    # The export tool still knows the three URLs and refuses them unless asked.
    lines = export_lines(list(SECRET_COMPUTED_FIELDS), settings=settings, secrets=True)
    assert len(lines) == 3
    assert lines[0].startswith("export MTC_POSTGRES_URL=")


def test_store_urls_without_credentials_are_unchanged(clean_env):
    s = Settings()
    tails = {
        "POSTGRES_URL": (
            "postgresql",
            f"{s.POSTGRES_USER}:@{s.POSTGRES_HOST}:{s.POSTGRES_PORT}/{s.POSTGRES_DB}",
        ),
        "MONGODB_URL": (
            "mongodb",
            f"{s.MONGODB_HOST}:{s.MONGODB_PORT}/{s.MONGODB_DB}",
        ),
        "REDIS_URL": ("redis", f"{s.REDIS_HOST}:{s.REDIS_PORT}"),
    }
    for name, (scheme, tail) in tails.items():
        url = getattr(s, name)
        assert urlsplit(url).scheme == scheme, name
        assert url.split("//", 1)[1] == tail, name
