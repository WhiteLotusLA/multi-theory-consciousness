"""Importing the package root or any non-neural package loads no torch.

The legacy measurement framework is the only module on a non-neural import
path that asked for torch, and it never used it. These tests pin the
property: a fresh interpreter that imports ``mtc`` (or every module of the
non-neural packages) has no torch, snntorch or reservoirpy in
``sys.modules``. The absence is asserted after the import; torch is never
blocked with a ``None`` entry, because scipy reads ``sys.modules`` and then
fails inside the pymdp import.

The probes run in a child process, which the in-process network guard does
not cover, so the child gets an explicit environment: no inherited project
variable or provider key, the CPU device, the mock provider, every store off
and an unused API port.
"""

import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence

from mtc.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_SOURCE = ROOT / "mtc" / "assessment" / "framework.py"
PREFIX = Settings.model_config["env_prefix"]
STORE_ENABLE_FIELDS = (
    "POSTGRES_ENABLED",
    "MONGODB_ENABLED",
    "QDRANT_ENABLED",
    "NEO4J_ENABLED",
    "REDIS_ENABLED",
)
PROVIDER_KEY_VARIABLE = "ANTHROPIC_API_KEY"

NEURAL_LIBRARIES = ("torch", "snntorch", "reservoirpy")
NON_NEURAL_PACKAGES = (
    "mtc.pipeline",
    "mtc.brain",
    "mtc.cli",
    "mtc.core",
    "mtc.assessment",
    "mtc.research",
    "mtc.consciousness",
    "mtc.memory",
    "mtc.safety",
    "mtc.privacy",
    "mtc.monitoring",
    "mtc.validation",
    "mtc.devtools",
)
# What ``import mtc`` alone must leave out: the neural libraries, PyPhi,
# pymdp, scipy, the web framework and its server, and every store client.
ROOT_IMPORT_FORBIDDEN = (
    "torch",
    "snntorch",
    "reservoirpy",
    "pyphi",
    "pymdp",
    "scipy",
    "fastapi",
    "starlette",
    "uvicorn",
    "asyncpg",
    "motor",
    "pymongo",
    "redis",
    "neo4j",
    "qdrant_client",
)

_WALK_PROBE = r"""
import importlib, json, pkgutil, sys
packages = json.loads(sys.argv[1])
watched = json.loads(sys.argv[2])
walked, failed = [], []
for name in packages:
    try:
        package = importlib.import_module(name)
    except Exception as exc:
        failed.append([name, repr(exc)])
        continue
    walked.append(name)
    for info in pkgutil.walk_packages(package.__path__, name + "."):
        try:
            importlib.import_module(info.name)
            walked.append(info.name)
        except Exception as exc:
            failed.append([info.name, repr(exc)])
loaded = sorted({m.split(".")[0] for m in sys.modules if m.split(".")[0] in watched})
print(json.dumps({"walked": walked, "failed": failed, "loaded": loaded}))
"""

_ROOT_PROBE = r"""
import json, sys
watched = json.loads(sys.argv[1])
import mtc
loaded = sorted({m.split(".")[0] for m in sys.modules if m.split(".")[0] in watched})
print(json.dumps({"loaded": loaded, "version": mtc.__version__}))
"""


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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
    env[f"{PREFIX}API_PORT"] = str(_unused_port())
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


def test_non_neural_packages_load_no_torch(tmp_path):
    report = _run_probe(
        _WALK_PROBE,
        [json.dumps(list(NON_NEURAL_PACKAGES)), json.dumps(list(NEURAL_LIBRARIES))],
        tmp_path,
    )
    assert report["failed"] == [], report["failed"]
    walked = set(report["walked"])
    # The walk reached every package and the modules that used to pull the
    # framework (and so torch) in.
    for name in NON_NEURAL_PACKAGES + (
        "mtc.assessment.framework",
        "mtc.consciousness.consciousness_integration",
        "mtc.consciousness.self_assessment",
        "mtc.pipeline.runtime",
        "mtc.devtools.privacy_audit",
    ):
        assert name in walked, name
    assert report["loaded"] == [], (
        f"importing the non-neural packages loaded {report['loaded']}; "
        "import them inside the function that needs them"
    )


def test_import_mtc_is_light(tmp_path):
    report = _run_probe(
        _ROOT_PROBE, [json.dumps(list(ROOT_IMPORT_FORBIDDEN))], tmp_path
    )
    assert report["version"]
    assert (
        report["loaded"] == []
    ), f"import mtc loaded {report['loaded']}; the package root must stay light"


def test_framework_construction_logs_no_warning(caplog):
    from mtc.assessment.framework import ConsciousnessMeasurementFramework

    with caplog.at_level(logging.DEBUG):
        framework = ConsciousnessMeasurementFramework(
            snn_neurons=10, lsm_neurons=10, htm_columns=4
        )
    assert framework.htm_columns == 4
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], [r.getMessage() for r in warnings]
    assert not hasattr(framework, "htm_processor")


def test_framework_source_names_no_native_htm():
    source = FRAMEWORK_SOURCE.read_text(encoding="utf-8")
    for name in ("CppHtmInterface", "htm_processor", "process_sdr", "C++"):
        assert name not in source, name
    lines = [line.strip() for line in source.splitlines()]
    assert "import torch" not in lines
