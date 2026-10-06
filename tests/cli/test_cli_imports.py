"""Importing the command line, the database package and the onboarding package
loads no store client, no web framework and no HTTP client: every handler
imports its heavy modules when it runs.

The probe runs in a child process, which the in-process network guard does
not cover, so the child gets an explicit environment (no inherited project
variable, the CPU device, the mock provider, every store off, an unused API
port) and an empty working directory with no environment file to read."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence

from mtc.core.config import Settings
from tests.cli.test_http_client import unused_port

ROOT = Path(__file__).resolve().parents[2]
PREFIX = Settings.model_config["env_prefix"]
STORE_ENABLE_FIELDS = (
    "POSTGRES_ENABLED",
    "MONGODB_ENABLED",
    "QDRANT_ENABLED",
    "NEO4J_ENABLED",
    "REDIS_ENABLED",
)
PROVIDER_KEY_VARIABLE = "ANTHROPIC_API_KEY"

STORE_CLIENTS = ("asyncpg", "motor", "pymongo", "redis", "neo4j", "qdrant_client")
WEB_FRAMEWORK = ("fastapi", "starlette", "uvicorn")
HTTP_CLIENTS = ("httpx", "anthropic", "requests", "aiohttp")
HEAVY_EXTRAS = ("sentence_transformers", "huggingface_hub", "pyphi")
FORBIDDEN = STORE_CLIENTS + WEB_FRAMEWORK + HTTP_CLIENTS + HEAVY_EXTRAS

# Named modules, plus every module of these packages.
MODULES = (
    "mtc.cli",
    "mtc.cli.main",
    "mtc.cli.doctor",
    "mtc.cli.http",
    "mtc.onboarding",
    "mtc.onboarding.schema",
    "mtc.onboarding.wizard",
    "mtc.monitoring.backup_status",
)
PACKAGES = ("mtc.cli", "mtc.db", "mtc.onboarding")

_PROBE = r"""
import importlib, json, pkgutil, sys

modules = json.loads(sys.argv[1])
packages = json.loads(sys.argv[2])
forbidden = json.loads(sys.argv[3])
walked, failed = [], []
for name in modules:
    importlib.import_module(name)
    walked.append(name)
for name in packages:
    package = importlib.import_module(name)
    for info in pkgutil.walk_packages(package.__path__, name + "."):
        try:
            importlib.import_module(info.name)
            walked.append(info.name)
        except ImportError as exc:
            failed.append([info.name, repr(exc)])

# Building the parser and answering --version load nothing either.
from mtc.cli.main import build_parser, main

parser = build_parser()
code = main(["--version"])
import mtc.onboarding as onboarding
exported = sorted(onboarding.__all__)

loaded = sorted({m.split(".")[0] for m in sys.modules if m.split(".")[0] in forbidden})
print(json.dumps({
    "walked": sorted(set(walked)),
    "failed": failed,
    "loaded": loaded,
    "version_exit": code,
    "exported": exported,
}))
"""


def _child_env() -> Dict[str, str]:
    """The environment a probe child runs with, whatever the shell exports."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(PREFIX) and key != PROVIDER_KEY_VARIABLE
    }
    env["PYTHONPATH"] = str(ROOT)
    env["PYPHI_WELCOME_OFF"] = "yes"
    env[f"{PREFIX}DEVICE"] = "cpu"
    env[f"{PREFIX}LLM_PROVIDER"] = "mock"
    for field in STORE_ENABLE_FIELDS:
        env[f"{PREFIX}{field}"] = "false"
    env[f"{PREFIX}API_PORT"] = str(unused_port())
    return env


def _run_probe(code: str, arguments: Sequence[str], workdir: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", code, *arguments],
        cwd=workdir,
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_probe_child_gets_an_explicit_environment(monkeypatch):
    for field in STORE_ENABLE_FIELDS:
        assert field in Settings.model_fields
        monkeypatch.setenv(f"{PREFIX}{field}", "true")
    default_port = Settings.model_fields["API_PORT"].default
    monkeypatch.setenv(f"{PREFIX}API_PORT", str(default_port))
    monkeypatch.setenv(f"{PREFIX}LLM_PROVIDER", "local")
    monkeypatch.setenv(f"{PREFIX}DEVICE", "auto")
    monkeypatch.setenv(f"{PREFIX}SYSTEM_NAME", "Testagent")
    monkeypatch.setenv(PROVIDER_KEY_VARIABLE, "inherited")

    env = _child_env()

    project_keys = {key for key in env if key.startswith(PREFIX)}
    assert project_keys == {
        f"{PREFIX}DEVICE",
        f"{PREFIX}LLM_PROVIDER",
        f"{PREFIX}API_PORT",
        *(f"{PREFIX}{field}" for field in STORE_ENABLE_FIELDS),
    }
    assert env[f"{PREFIX}DEVICE"] == "cpu"
    assert env[f"{PREFIX}LLM_PROVIDER"] == "mock"
    for field in STORE_ENABLE_FIELDS:
        assert env[f"{PREFIX}{field}"] == "false"
    assert int(env[f"{PREFIX}API_PORT"]) != default_port
    assert PROVIDER_KEY_VARIABLE not in env
    assert env["PYTHONPATH"] == str(ROOT)


def test_importing_cli_loads_no_store_client_web_framework_or_http_client(tmp_path):
    report = _run_probe(
        _PROBE,
        [
            json.dumps(list(MODULES)),
            json.dumps(list(PACKAGES)),
            json.dumps(list(FORBIDDEN)),
        ],
        tmp_path,
    )
    assert report["failed"] == [], report["failed"]
    walked = set(report["walked"])
    for name in MODULES + (
        "mtc.db.bootstrap",
        "mtc.db.verify",
        "mtc.db.migrations",
        "mtc.onboarding.seed_db",
    ):
        assert name in walked, f"{name} was not imported by the probe"
    assert report["loaded"] == [], (
        f"importing the command line loaded {report['loaded']}; a handler must "
        "import its heavy modules when it runs"
    )
    assert report["version_exit"] == 0
    assert "run_init" in report["exported"]


def test_the_probe_detects_a_loaded_client(tmp_path):
    """The audit is not vacuous: a module that is loaded is reported."""
    report = _run_probe(
        "import sys, types\n"
        "sys.modules['uvicorn'] = types.ModuleType('uvicorn')\n" + _PROBE,
        [
            json.dumps(["mtc.cli.http"]),
            json.dumps([]),
            json.dumps(list(FORBIDDEN)),
        ],
        tmp_path,
    )
    assert report["loaded"] == ["uvicorn"]
