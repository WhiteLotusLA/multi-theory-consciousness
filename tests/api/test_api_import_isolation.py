"""Only mtc.api imports the web framework: importing the pipeline, the brain
and the continuous thought packages (every module in each) loads no web
framework, server, HTTP client or vendor SDK. Importing every module of the
pipeline, the brain and the command line loads no torch either: only the
neural substrates need it."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The web framework and its server, the HTTP client and the vendor SDK.
FORBIDDEN = ("fastapi", "starlette", "uvicorn", "httpx", "anthropic")
PACKAGES = ("mtc.pipeline", "mtc.brain", "mtc.consciousness.continuous")
# The packages whose every module must import without torch, and the
# accelerator libraries that must stay out (a separate list, so the web
# framework message above stays exact).
TORCH_FREE_PACKAGES = ("mtc.pipeline", "mtc.brain", "mtc.cli")
ACCELERATOR_LIBRARIES = ("torch",)
STORE_KEYS = (
    "MTC_POSTGRES_ENABLED",
    "MTC_MONGODB_ENABLED",
    "MTC_REDIS_ENABLED",
    "MTC_QDRANT_ENABLED",
    "MTC_NEO4J_ENABLED",
)
# The callers of the API package from outside it: (file, module) -> the one
# function that may hold the import statement. The serve command starts the
# server, so its handler imports the server module when the command runs.
API_CALLERS = {("mtc/cli/main.py", "mtc.api.server"): "_cmd_serve"}

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


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _child_env() -> dict:
    """The probe's whole environment: no MTC_ variable of the caller, the CPU
    device, the mock provider, every store off and an unused API port."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTC_")}
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
            "MTC_LLM_PROVIDER": "mock",
            "MTC_API_PORT": str(_unused_port()),
        }
    )
    env.update({key: "false" for key in STORE_KEYS})
    return env


def _probe(packages=PACKAGES, forbidden=FORBIDDEN):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            json.dumps(list(packages)),
            json.dumps(list(forbidden)),
        ],
        cwd=ROOT,
        env=_child_env(),
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


def test_importing_pipeline_brain_and_cli_loads_no_torch():
    """Every module of the pipeline, the brain and the command line imports
    without loading torch, so a conversation, the server and every command
    that builds no neural substrate run on a machine without the neural
    extra."""
    report = _probe(TORCH_FREE_PACKAGES, ACCELERATOR_LIBRARIES)
    assert report["failed"] == [], report["failed"]
    walked = set(report["walked"])
    for name in (
        "mtc.pipeline.runtime",
        "mtc.pipeline.conversation_pipeline",
        "mtc.brain.brain",
        "mtc.cli.main",
        "mtc.cli.doctor",
    ):
        assert name in walked, name
    assert report["loaded"] == [], (
        f"importing the pipeline, brain or command line loaded "
        f"{report['loaded']}; import it inside the function that needs it"
    )


def _import_offenders(relative: str, text: str) -> list:
    """Import statements in one module that name the web framework or the
    API package, except an ``API_CALLERS`` import inside its function."""
    import ast

    tree = ast.parse(text)
    # Each import statement's outermost enclosing function, if it has one.
    enclosing = {}
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    enclosing.setdefault(id(node), function.name)
    offenders = []
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
                allowed_in = API_CALLERS.get((relative, name))
                if allowed_in is not None and enclosing.get(id(node)) == allowed_in:
                    continue
                offenders.append(f"{relative}:{node.lineno}:{name}")
    return offenders


def test_no_module_outside_the_api_imports_the_web_framework():
    """A static check over every module: the framework names appear in an
    import statement only under mtc/api, and the API package is imported
    from outside it only by the callers listed in ``API_CALLERS``."""
    offenders = []
    for path in sorted((ROOT / "mtc").rglob("*.py")):
        relative = path.relative_to(ROOT)
        if relative.parts[:2] == ("mtc", "api"):
            continue
        offenders += _import_offenders(
            relative.as_posix(), path.read_text(encoding="utf-8")
        )
    assert offenders == [], offenders


def test_the_api_caller_allowance_is_one_import_in_one_function():
    assert API_CALLERS == {("mtc/cli/main.py", "mtc.api.server"): "_cmd_serve"}
    handler = "def _cmd_serve():\n    import mtc.api.server as server\n"
    assert _import_offenders("mtc/cli/main.py", handler) == []
    # Module level, another function, another module of the package, the
    # package itself, another file and the framework are still offenders.
    for relative, text in (
        ("mtc/cli/main.py", "import mtc.api.server\n"),
        ("mtc/cli/main.py", "def other():\n    import mtc.api.server\n"),
        ("mtc/cli/main.py", "def _cmd_serve():\n    import mtc.api.auth\n"),
        ("mtc/cli/main.py", "def _cmd_serve():\n    from mtc.api import server\n"),
        ("mtc/cli/doctor.py", "def _cmd_serve():\n    import mtc.api.server\n"),
        ("mtc/cli/main.py", "def _cmd_serve():\n    import uvicorn\n"),
    ):
        assert len(_import_offenders(relative, text)) == 1, (relative, text)


def test_the_serve_handler_imports_the_server_with_a_statement():
    """The allowance is used, not bypassed: the handler holds the plain
    import, and nothing in the entry point loads a module by name."""
    import ast

    tree = ast.parse((ROOT / "mtc/cli/main.py").read_text(encoding="utf-8"))
    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_cmd_serve"
    )
    imported = [
        alias.name
        for node in ast.walk(handler)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert "mtc.api.server" in imported, imported
    by_name = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
        and getattr(node.func, "attr", getattr(node.func, "id", ""))
        in ("import_module", "__import__")
    ]
    assert by_name == [], by_name
