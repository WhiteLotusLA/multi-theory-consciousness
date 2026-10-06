"""Only mtc.api imports the web framework: importing the pipeline, the brain
and the continuous thought packages (every module in each) loads no web
framework, server, HTTP client or vendor SDK."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The web framework and its server, the HTTP client and the vendor SDK.
FORBIDDEN = ("fastapi", "starlette", "uvicorn", "httpx", "anthropic")
PACKAGES = ("mtc.pipeline", "mtc.brain", "mtc.consciousness.continuous")

_PROBE = r"""
import importlib, json, pkgutil, sys
packages = json.loads(sys.argv[1])
forbidden = json.loads(sys.argv[2])
walked, failed = [], []
for name in packages:
    package = importlib.import_module(name)
    walked.append(name)
    for info in pkgutil.walk_packages(package.__path__, name + "."):
        try:
            importlib.import_module(info.name)
            walked.append(info.name)
        except ImportError as exc:
            failed.append([info.name, repr(exc)])
loaded = sorted(
    {m.split(".")[0] for m in sys.modules if m.split(".")[0] in forbidden}
)
print(json.dumps({"walked": walked, "failed": failed, "loaded": loaded}))
"""


def _probe():
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYPHI_WELCOME_OFF": "yes"}
    env.setdefault("MTC_DEVICE", "cpu")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            json.dumps(list(PACKAGES)),
            json.dumps(list(FORBIDDEN)),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_pipeline_brain_and_ctm_loads_no_web_framework():
    report = _probe()
    assert report["failed"] == [], report["failed"]
    # The walk reached the pipeline modules, the providers and the machine.
    walked = set(report["walked"])
    for name in (
        "mtc.pipeline.speaker_trust",
        "mtc.pipeline.consciousness_orchestrator",
        "mtc.brain.brain",
        "mtc.brain.providers.local_chat",
        "mtc.consciousness.continuous.machine",
    ):
        assert name in walked, name
    assert report["loaded"] == [], (
        f"importing the pipeline, brain or thought packages loaded "
        f"{report['loaded']}; import them inside the function that needs them"
    )


def test_no_module_outside_the_api_imports_the_web_framework():
    """A static check over every module: the framework names appear in an
    import statement only under mtc/api."""
    import ast

    offenders = []
    for path in sorted((ROOT / "mtc").rglob("*.py")):
        relative = path.relative_to(ROOT)
        if relative.parts[:2] == ("mtc", "api"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split(".")[0] in ("fastapi", "starlette", "uvicorn"):
                    offenders.append(f"{relative}:{node.lineno}:{name}")
                if name == "mtc.api" or name.startswith("mtc.api."):
                    offenders.append(f"{relative}:{node.lineno}:{name}")
    assert offenders == [], offenders
