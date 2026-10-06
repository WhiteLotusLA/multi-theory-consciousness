"""The Compose file (parsed with PyYAML; Docker is not run): five services
with healthchecks, loopback ports, named volumes, passwords from the
environment, defaults equal to Settings, tools recorded in the README, no
obsolete version key and a README that says the file was reviewed, not run."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

from mtc.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker" / "docker-compose.yml"
README = ROOT / "docker" / "README.md"
SERVICES = ("postgres", "mongo", "redis", "qdrant", "neo4j")
INTERPOLATION = re.compile(r"\$\{(MTC_[A-Z0-9_]+)(:-|:\?)([^}]*)\}")


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_five_services_with_healthchecks(compose):
    assert compose["name"] == "mtc"
    assert tuple(compose["services"]) == SERVICES
    for name, service in compose["services"].items():
        check = service.get("healthcheck")
        assert check and check.get("test"), name
        assert check["test"][0] in ("CMD", "CMD-SHELL"), name
        assert check.get("interval") and check.get("retries"), name
        assert ":" in service["image"], name
        assert not service["image"].endswith(":latest"), name


def test_ports_bind_loopback_only(compose):
    for name, service in compose["services"].items():
        ports = service.get("ports") or []
        assert ports, name
        for mapping in ports:
            assert isinstance(mapping, str), name
            assert mapping.startswith("127.0.0.1:"), (name, mapping)
        assert service.get("network_mode") != "host", name
    neo4j_port = urlsplit(Settings.model_fields["NEO4J_URI"].default).port
    assert compose["services"]["neo4j"]["ports"] == [
        f"127.0.0.1:{neo4j_port}:{neo4j_port}"
    ]


def test_named_volumes(compose):
    expected = {
        "postgres": "mtc_pgdata:/var/lib/postgresql/data",
        "mongo": "mtc_mongodata:/data/db",
        "redis": "mtc_redisdata:/data",
        "qdrant": "mtc_qdrantdata:/qdrant/storage",
        "neo4j": "mtc_neo4jdata:/data",
    }
    for name, volume in expected.items():
        assert compose["services"][name]["volumes"] == [volume], name
    assert set(compose["volumes"]) == {v.split(":")[0] for v in expected.values()}
    for name, service in compose["services"].items():
        for volume in service["volumes"]:
            source = volume.split(":")[0]
            assert not source.startswith((".", "/", "~")), (name, volume)


def test_passwords_from_env_not_literals(compose):
    text = COMPOSE.read_text(encoding="utf-8")
    postgres = compose["services"]["postgres"]["environment"]
    assert postgres["POSTGRES_PASSWORD"].startswith("${MTC_POSTGRES_PASSWORD:?")
    assert compose["services"]["neo4j"]["environment"]["NEO4J_AUTH"] == (
        "neo4j/${MTC_NEO4J_PASSWORD:-}"
    )
    for service in compose["services"].values():
        for key, value in (service.get("environment") or {}).items():
            if "PASSWORD" in key or key == "NEO4J_AUTH":
                assert "${MTC_" in str(value), key
        command = " ".join(str(part) for part in service.get("command") or [])
        assert "requirepass" not in command and "PASSWORD" not in command
        test = " ".join(str(part) for part in service["healthcheck"]["test"])
        assert "PASSWORD" not in test.upper()
        assert " -p " not in test and "--password" not in test
    assert not re.search(r"(?i)password[^\n]*:\s*['\"]?[A-Za-z0-9]{6,}['\"]?\s*$", text)


def test_compose_defaults_match_settings():
    defaults = Settings(_env_file=None, DB_PREFIX="mtc")
    found = INTERPOLATION.findall(COMPOSE.read_text(encoding="utf-8"))
    assert found
    names = set()
    for variable, operator, default in found:
        field = variable[len("MTC_") :]
        assert field in Settings.model_fields, variable
        names.add(field)
        if operator == ":-":
            assert default == str(getattr(defaults, field)), variable
        else:
            assert field == "POSTGRES_PASSWORD", variable
    assert names >= {
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "POSTGRES_PORT",
        "MONGODB_PORT",
        "REDIS_PORT",
        "QDRANT_PORT",
        "NEO4J_PASSWORD",
    }


def test_healthcheck_tools_recorded_in_readme(compose):
    readme = README.read_text(encoding="utf-8")
    for name, service in compose["services"].items():
        test = service["healthcheck"]["test"]
        tool = test[1].split()[0] if test[0] == "CMD-SHELL" else test[1]
        assert f"`{tool}`" in readme, (name, tool)
        image = service["image"]
        assert f"`{image}`" in readme, image
    for path in (
        "/var/lib/postgresql/data",
        "/data/db",
        "/qdrant/storage",
        "`/data`",
    ):
        assert path in readme, path
    assert "Checked on" in readme and "Source" in readme
    assert "--wait" in readme
    assert "interpolat" in readme.lower()


def test_no_obsolete_version_key(compose):
    assert "version" not in compose
    assert not re.search(r"(?m)^version\s*:", COMPOSE.read_text(encoding="utf-8"))


def test_readme_states_reviewed_not_run():
    readme = README.read_text(encoding="utf-8")
    assert "Reviewed, not run" in readme
    assert "not run on the development" in readme
    assert "path tested" in readme
    assert "make db-up DB_SERVICES=" in readme
    header = COMPOSE.read_text(encoding="utf-8").splitlines()[:8]
    assert any("not run on the development machine" in line for line in header)
