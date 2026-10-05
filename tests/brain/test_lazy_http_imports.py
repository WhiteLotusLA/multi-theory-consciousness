"""Importing the brain and its providers loads no HTTP client or vendor SDK."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Clients the providers import only when a client is built.
HTTP_CLIENTS = ("httpx", "anthropic")

MODULES = (
    "mtc.brain",
    "mtc.brain.protocol",
    "mtc.brain.families",
    "mtc.brain.thinking",
    "mtc.brain.thinking_loop_detector",
    "mtc.brain.providers",
    "mtc.brain.providers.base",
    "mtc.brain.providers.local_chat",
    "mtc.brain.providers.anthropic_provider",
    "mtc.brain.providers.mock_provider",
    "mtc.brain.providers.registry",
    "mtc.core.narration",
    "mtc.assessment.dcm_evaluator",
)

_PROBE = r"""
import importlib, json, pkgutil, sys
names = json.loads(sys.argv[1])
clients = json.loads(sys.argv[2])
import mtc.brain
extra = [info.name for info in pkgutil.walk_packages(mtc.brain.__path__, "mtc.brain.")]
for name in names + extra:
    importlib.import_module(name)
print(json.dumps({
    "loaded": [c for c in clients if c in sys.modules],
    "walked": sorted(extra),
}))
"""


def test_importing_brain_loads_no_http_client():
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYPHI_WELCOME_OFF": "yes"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            json.dumps(list(MODULES)),
            json.dumps(list(HTTP_CLIENTS)),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["loaded"] == [], (
        f"importing the brain loaded {report['loaded']}; import them inside the "
        "function that builds a client"
    )
    # Every provider module is covered by the walk as well as the list.
    for name in MODULES:
        if name.startswith("mtc.brain.providers."):
            assert name in report["walked"], name
