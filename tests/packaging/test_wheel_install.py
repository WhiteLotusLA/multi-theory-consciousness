"""A wheel works from anywhere: build it, install it into a fresh virtual
environment and, from a project directory outside the repository, check the
version, the packaged resources, the setup wizard, doctor and the full
pipeline demo. No network: the build uses the installed setuptools and the
venv sees the system site-packages for the dependencies.

The sources are copied to a temporary directory first, so the build writes
nothing into the repository. Every child process gets an explicit
environment: the CPU device, the mock provider, every store disabled, an API
port nobody listens on, offline model hubs and a temporary HOME."""

import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "MANIFEST.in",
    "THIRD_PARTY_LICENSES.md",
)
STORE_KEYS = (
    "MTC_POSTGRES_ENABLED",
    "MTC_MONGODB_ENABLED",
    "MTC_REDIS_ENABLED",
    "MTC_QDRANT_ENABLED",
    "MTC_NEO4J_ENABLED",
)

RESOURCE_CHECK = """
import json
import mtc
from mtc.core.resources import RESOURCE_MANIFEST, resolve_shipped
print(json.dumps({
    "file": mtc.__file__,
    "sources": {e.source: resolve_shipped(e.source)[1] for e in RESOURCE_MANIFEST},
}))
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _env(tmp_path: Path, venv: Path) -> dict:
    env = {
        "PATH": f"{venv / 'bin'}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "LANG": "C.UTF-8",
        "PYPHI_WELCOME_OFF": "yes",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MTC_DEVICE": "cpu",
        "MTC_LLM_PROVIDER": "mock",
        "MTC_API_PORT": str(_free_port()),
    }
    env.update({key: "false" for key in STORE_KEYS})
    (tmp_path / "home").mkdir(exist_ok=True)
    return env


def _run(command, *, cwd, env, timeout=900, **kwargs):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        **kwargs,
    )


@pytest.mark.slow
def test_wheel_installs_and_runs_demo(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(
        ROOT / "mtc",
        source / "mtc",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    venv = tmp_path / "venv"
    env = _env(tmp_path, venv)
    dist = tmp_path / "dist"
    build = _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(dist),
            str(source),
        ],
        cwd=tmp_path,
        env={**env, "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"},
    )
    assert build.returncode == 0, build.stdout[-2000:] + build.stderr[-2000:]
    [wheel] = list(dist.glob("multi_theory_consciousness-0.5.0-*.whl"))
    made = _run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        cwd=tmp_path,
        env=env,
    )
    assert made.returncode == 0, made.stderr[-2000:]
    python = venv / "bin" / "python"
    installed = _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        cwd=tmp_path,
        env=env,
    )
    assert installed.returncode == 0, (
        installed.stdout[-2000:] + installed.stderr[-2000:]
    )

    project = tmp_path / "proj"
    project.mkdir()
    checked = _run([str(python), "-c", RESOURCE_CHECK], cwd=project, env=env)
    assert checked.returncode == 0, checked.stderr[-2000:]
    report = json.loads(checked.stdout.strip().splitlines()[-1])
    assert Path(report["file"]).resolve().is_relative_to(venv.resolve())
    assert report["sources"] and set(report["sources"].values()) == {"package"}

    mtc = venv / "bin" / "mtc"
    version = _run([str(mtc), "--version"], cwd=project, env=env)
    assert version.returncode == 0, version.stderr[-2000:]
    assert "0.5.0" in version.stdout

    example = _run([str(mtc), "init", "--print-example-answers"], cwd=project, env=env)
    assert example.returncode == 0, example.stderr[-2000:]
    answers = project / "answers.yaml"
    answers.write_text(example.stdout, encoding="utf-8")
    init = _run(
        [str(mtc), "init", "--from", str(answers), "--yes", "--no-probe"],
        cwd=project,
        env=env,
    )
    assert init.returncode == 0, init.stdout[-2000:] + init.stderr[-2000:]
    assert (project / ".env").is_file()

    doctor = _run([str(mtc), "doctor", "--json"], cwd=project, env=env)
    assert doctor.returncode == 0, doctor.stdout[-3000:] + doctor.stderr[-2000:]
    payload = json.loads(doctor.stdout)
    assert payload["exit_code"] == 0

    demo = _run(
        [str(python), str(ROOT / "examples" / "full_pipeline_demo.py")],
        cwd=project,
        env=env,
    )
    assert demo.returncode == 0, demo.stdout[-3000:] + demo.stderr[-2000:]
    lines = [
        line for line in demo.stdout.splitlines() if line.startswith("DEMO_SUMMARY ")
    ]
    assert len(lines) == 1
    summary = json.loads(lines[0][len("DEMO_SUMMARY ") :])
    assert summary["ok"] is True
