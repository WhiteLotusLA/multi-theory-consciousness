"""The host-service store configs: Qdrant on loopback with project-relative
storage and no telemetry; the Neo4j example on loopback with the configured
default database and no password; no home-directory path in either."""

import re
from pathlib import Path

import yaml

from mtc.core.config import Settings
from mtc.devtools.privacy_audit import PATTERNS

ROOT = Path(__file__).resolve().parents[2]
QDRANT = ROOT / "config" / "qdrant.yaml"
NEO4J = ROOT / "config" / "neo4j.conf.example"


def _neo4j_settings():
    values = {}
    for line in NEO4J.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_qdrant_config_loopback_relative_storage_telemetry_off():
    config = yaml.safe_load(QDRANT.read_text(encoding="utf-8"))
    assert config["service"]["host"] == "127.0.0.1"
    assert (
        config["service"]["http_port"] == Settings.model_fields["QDRANT_PORT"].default
    )
    assert config["service"]["grpc_port"] == 6334
    assert config["telemetry_disabled"] is True
    for key in ("storage_path", "snapshots_path"):
        path = config["storage"][key]
        assert path.startswith("./data/qdrant/"), path
        assert not Path(path).is_absolute()
    assert set(config) <= {"log_level", "storage", "service", "telemetry_disabled"}


def test_neo4j_conf_example_loopback_default_db_and_no_password():
    values = _neo4j_settings()
    assert values["server.default_listen_address"] == "127.0.0.1"
    assert values["initial.dbms.default_database"] == (
        Settings.model_fields["NEO4J_DATABASE"].default
    )
    assert values["dbms.security.auth_enabled"] == "true"
    for key in (
        "server.memory.heap.initial_size",
        "server.memory.heap.max_size",
        "server.memory.pagecache.size",
    ):
        assert re.fullmatch(r"\d+[mg]", values[key]), key
        size, unit = int(values[key][:-1]), values[key][-1]
        assert (size if unit == "m" else size * 1024) <= 2048, key
    text = NEO4J.read_text(encoding="utf-8")
    assert "password" not in text.lower()
    assert "neo4j/" not in text
    assert "NEO4J_CONF" in text and "directory" in text
    assert not any(key.startswith("server.directories.") for key in values)


def test_no_home_paths_in_configs():
    # The privacy audit's own home-directory and external-volume pattern, plus
    # the shell forms of a home directory.
    shell_home = re.compile(r"(~/|\$HOME)")
    for path in (
        QDRANT,
        NEO4J,
        ROOT / "config" / "pyphi_config.yml",
        ROOT / "docker" / "docker-compose.yml",
    ):
        text = path.read_text(encoding="utf-8")
        assert not PATTERNS["homepath"].search(text), path.name
        assert not shell_home.search(text), path.name
        assert not re.search(r"\b(?!127\.)\d{1,3}(\.\d{1,3}){3}\b", text), path.name
