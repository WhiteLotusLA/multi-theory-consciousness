"""The four older examples run offline and state the current counts.

Each example runs in a subprocess under a small runner that makes every
outgoing connection fail and counts the attempts, from a temporary working
directory, with the mock provider, the CPU device, every store disabled and an
unused API port, so a pass shows it needs no network, store, model server or
accelerator. The counts the examples state are compared with the assessment.
"""

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.assessment.assessment import ConsciousnessAssessment, ConsciousnessTheory

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "basic_assessment": ("pymdp",),
    "multi_theory_pipeline": ("pymdp",),
    "neural_substrates": ("torch", "snntorch"),
    "single_theory": (),
}
STORE_KEYS = (
    "MTC_POSTGRES_ENABLED",
    "MTC_MONGODB_ENABLED",
    "MTC_REDIS_ENABLED",
    "MTC_QDRANT_ENABLED",
    "MTC_NEO4J_ENABLED",
)
TIMEOUT_S = 300

RUNNER = """
import runpy
import socket
import sys

attempts = []


def refuse(*args, **kwargs):
    attempts.append(1)
    raise OSError("outgoing connections are disabled in this test")


socket.socket.connect = refuse
socket.socket.connect_ex = refuse
socket.create_connection = refuse
example = sys.argv[1]
sys.argv = [example]
code = 0
try:
    runpy.run_path(example, run_name="__main__")
except SystemExit as stop:
    code = stop.code if isinstance(stop.code, int) else 1
print("NETWORK_ATTEMPTS", len(attempts))
sys.exit(code)
"""

STALE = re.compile(
    r"\b(?:20|25) indicators\b|\b7 theories\b|\bseven theories\b", re.IGNORECASE
)
NUMBER_WORDS = {"three": 3, "eight": 8}


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _child_env(tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTC_")}
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
            "MTC_LLM_PROVIDER": "mock",
            "MTC_API_PORT": str(_unused_port()),
            "MTC_PROJECT_ROOT": str(tmp_path),
        }
    )
    env.update({key: "false" for key in STORE_KEYS})
    return env


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_runs_offline(name, tmp_path):
    for module in EXAMPLES[name]:
        pytest.importorskip(module, reason=f"{name} needs {module}")
    runner = tmp_path / "runner.py"
    runner.write_text(RUNNER, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    completed = subprocess.run(
        [sys.executable, str(runner), str(ROOT / "examples" / f"{name}.py")],
        cwd=work,
        env=_child_env(work),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "NETWORK_ATTEMPTS 0" in completed.stdout
    # The example wrote nothing into the repository's working tree.
    assert not (ROOT / "examples" / "results").exists()


def test_examples_state_current_counts():
    assessment = ConsciousnessAssessment()
    positive = len(assessment.indicator_configs)
    negative = len(assessment.negative_indicator_configs)
    theories = len(list(ConsciousnessTheory))
    stated = 0
    for name in EXAMPLES:
        text = (ROOT / "examples" / f"{name}.py").read_text(encoding="utf-8")
        assert not STALE.search(text), name
        for match in re.finditer(r"\b(\d+) positive\b", text):
            stated += 1
            assert int(match.group(1)) == positive, name
        for match in re.finditer(r"\b(\d+|three) negative\b", text):
            stated += 1
            value = match.group(1)
            assert (NUMBER_WORDS.get(value) or int(value)) == negative, name
        for match in re.finditer(r"\b(\d+|eight) (?:theories|theory modules)\b", text):
            stated += 1
            value = match.group(1)
            assert (NUMBER_WORDS.get(value) or int(value)) == theories, name
    # The assessment examples state the counts, so the loop above is not empty.
    assert stated >= 4
