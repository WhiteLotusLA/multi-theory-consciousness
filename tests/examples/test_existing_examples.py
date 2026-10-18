"""The four older examples run offline, state the current counts and print
only what was measured.

The slow test runs each example in a subprocess under a small runner that
makes every outgoing connection fail and counts the attempts, from a
temporary working directory, with the mock provider, the CPU device, every
store disabled and an unused API port, so a pass shows it needs no network,
store, model server or accelerator. The counts the examples state are
compared with the assessment. The other tests run an example's ``main`` in
process with captured output (CPU device, mock provider, stores off) and
check what it prints for an unmeasured value, what its docstring claims and
how it stops without the ``fep`` extra.
"""

import ast
import asyncio
import inspect
import os
import re
import runpy
import socket
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from mtc.assessment.assessment import ConsciousnessAssessment, ConsciousnessTheory
from mtc.core import config as config_module

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = {
    "basic_assessment": ("pymdp",),
    "multi_theory_pipeline": ("pymdp",),
    "neural_substrates": ("torch", "snntorch"),
    "single_theory": ("pymdp",),
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


# ---------------------------------------------------------------------------
# In-process runs: each example's main with captured output
# ---------------------------------------------------------------------------


@pytest.fixture
def example_env(monkeypatch, tmp_path):
    """No MTC_ variable but the CPU device, the mock provider and every store
    off; a temporary working directory; the Settings singleton rebuilt from
    this environment and restored afterwards."""
    for key in list(os.environ):
        if key.startswith("MTC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    monkeypatch.setenv("MTC_LLM_PROVIDER", "mock")
    for key in STORE_KEYS:
        monkeypatch.setenv(key, "false")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "_settings", None)
    yield tmp_path


def _example(name: str) -> Dict[str, Any]:
    """The example's module namespace, loaded without running ``main``."""
    return runpy.run_path(str(ROOT / "examples" / f"{name}.py"), run_name=name)


def _run_main(name: str) -> Any:
    outcome = _example(name)["main"]()
    if inspect.iscoroutine(outcome):
        return asyncio.run(outcome)
    return outcome


def _captured_reports(monkeypatch) -> List[Tuple[Any, Any]]:
    """Every (assessment, report) the run produces, recorded as it returns."""
    captured: List[Tuple[Any, Any]] = []
    real = ConsciousnessAssessment.run_full_assessment

    async def recording(self, *args, **kwargs):
        report = await real(self, *args, **kwargs)
        captured.append((self, report))
        return report

    monkeypatch.setattr(ConsciousnessAssessment, "run_full_assessment", recording)
    return captured


def _docstring(name: str) -> str:
    tree = ast.parse((ROOT / "examples" / f"{name}.py").read_text(encoding="utf-8"))
    return " ".join((ast.get_docstring(tree) or "").split())


NUMBER = re.compile(r"\d")


def test_multi_theory_prints_unmeasured_fields(example_env, capsys):
    """Binding quality, protoself stability and autobiographic continuity are
    None after one cycle (no numeric candidate vectors, no previous body
    state, no autobiographical input): the example prints "not measured" for
    each and raises nothing."""
    pytest.importorskip("pymdp", reason="the workspace needs pymdp")
    _run_main("multi_theory_pipeline")
    out = capsys.readouterr().out
    for label in (
        "Binding quality:",
        "Protoself stability:",
        "Autobiographic continuity:",
    ):
        (line,) = [text for text in out.splitlines() if label in text]
        assert line.split(label, 1)[1].strip() == "not measured", line


# Theory names as the example docstrings write them.
THEORY_NAMES = {
    "IIT": "IIT",
    "Integrated Information": "IIT",
    "Recurrent Processing": "RPT",
    "Agency and Embodiment": "AE",
    "Global Workspace": "GWT",
    "Attention Schema": "AST",
    "Higher-Order": "HOT",
    "Free Energy": "FEP",
    "Beautiful Loop": "BLT",
}


def _theories_named(text: str) -> List[str]:
    return sorted({key for name, key in THEORY_NAMES.items() if name in text})


def test_multi_theory_docstring_matches_the_absent_indicators(example_env, monkeypatch):
    """Every theory the docstring says the assessment reads from module
    statistics has at least one measured indicator in the report the example
    builds, and every theory whose indicators it calls absent has only absent
    ones."""
    pytest.importorskip("pymdp", reason="the workspace needs pymdp")
    doc = _docstring("multi_theory_pipeline")
    claimed = re.search(r"the assessment (.+?) from module statistics", doc)
    assert claimed, "the docstring names what the assessment reads"
    measured_claims = _theories_named(claimed.group(1))
    absent_claims = _theories_named(
        " ".join(re.findall(r"the (\S+) indicators are absent", doc))
    )
    assert measured_claims, "the clause names the theories it claims"
    captured = _captured_reports(monkeypatch)
    _run_main("multi_theory_pipeline")
    ((assessment, report),) = captured
    statuses: Dict[str, List[str]] = {}
    for name, result in report.indicator_results.items():
        theory = assessment.indicator_configs[name]["theory"].name
        statuses.setdefault(theory, []).append(result.status)
    for theory in measured_claims:
        assert "measured" in statuses[theory], (theory, statuses[theory])
    for theory in absent_claims:
        assert set(statuses[theory]) == {"absent"}, (theory, statuses[theory])


def _snn_state_reads(tree: ast.AST) -> Tuple[List[str], List[str]]:
    """The keys the example reads from ``snn_state`` (subscripts and ``get``
    calls) and the keys read with ``get`` and a default."""
    keys: List[str] = []
    defaulted: List[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "snn_state"
            and isinstance(node.slice, ast.Constant)
        ):
            keys.append(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "snn_state"
        ):
            key = node.args[0].value if node.args else None
            keys.append(key)
            if len(node.args) > 1 or node.keywords:
                defaulted.append(key)
    return keys, defaulted


def _snn_config_kwargs(tree: ast.AST) -> Dict[str, Any]:
    """The literal keyword arguments of the example's ``SNNConfig(...)``."""
    (call,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SNNConfig"
    ]
    return {item.arg: ast.literal_eval(item.value) for item in call.keywords}


def test_neural_substrates_reads_only_metrics_the_snn_returns():
    """Every value the example prints from the SNN's metrics is a key the SNN
    returns, read without a default, so a renamed metric fails loudly instead
    of printing a stand-in zero."""
    torch = pytest.importorskip("torch", reason="the SNN needs torch")
    pytest.importorskip("snntorch", reason="the SNN needs snntorch")
    from mtc.neural.spiking.production_snn import ProductionSNN, SNNConfig

    tree = ast.parse(
        (ROOT / "examples" / "neural_substrates.py").read_text(encoding="utf-8")
    )
    keys, defaulted = _snn_state_reads(tree)
    kwargs = _snn_config_kwargs(tree)
    assert kwargs["use_gpu"] is False, "the example runs on the CPU"
    torch.manual_seed(7)
    snn = ProductionSNN(config=SNNConfig(**kwargs))
    _output, metrics = snn.process_input(torch.randn(1, kwargs["input_dim"]))
    assert keys, "the example prints SNN metrics"
    assert defaulted == [], "a metric read with a default hides a missing key"
    assert set(keys) <= set(metrics), sorted(set(keys) - set(metrics))
    assert {"mean_firing_rate", "total_spikes"} <= set(keys)


# The module whose import the test table checks for each extra an example
# can name (the import names of the extra's distributions the example uses).
EXTRA_MODULES = {"fep": ("pymdp",), "neural": ("snntorch", "torch")}
WORKSPACE_MODULE = "mtc.consciousness.enhanced_global_workspace"


def _needs_line(doc: str) -> str:
    match = re.search(r"Needs [^.]*\.", doc)
    assert match, "the docstring states what the example needs"
    return match.group(0)


def _imports(name: str) -> List[str]:
    tree = ast.parse((ROOT / "examples" / f"{name}.py").read_text(encoding="utf-8"))
    found: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
    return found


def _core_dependencies() -> List[str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return [str(item).lower() for item in project["dependencies"]]


def _optional_extras() -> Dict[str, List[str]]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


def test_examples_state_the_extras_they_need():
    """The test table, each example's "Needs" line and pyproject.toml agree:
    the extras the line names exist, the table lists exactly their modules,
    "only the core install" lists none, and an example that builds the
    workspace needs pymdp unless pymdp is a core dependency."""
    pymdp_is_core = any("pymdp" in item for item in _core_dependencies())
    extras = _optional_extras()
    for name, modules in EXAMPLES.items():
        line = _needs_line(_docstring(name))
        named = re.findall(r"``(\w+)`` extra", line)
        for extra in named:
            assert extra in extras, (name, extra)
        if "only the core install" in line:
            assert named == [] and modules == (), (name, line)
        expected = sorted({m for extra in named for m in EXTRA_MODULES[extra]})
        assert sorted(modules) == expected, (name, line)
        if WORKSPACE_MODULE in _imports(name) and not pymdp_is_core:
            assert "pymdp" in modules, name
            assert "fep" in named, (name, line)


def _line(out: str, label: str) -> str:
    (line,) = [text for text in out.splitlines() if label in text]
    return line.split(label, 1)[1].strip()


def test_single_theory_prints_no_integration_number_without_signals(
    example_env, capsys
):
    """The example passes no neural signals, so the workspace measures no
    integration level and, with no receiver registered, no broadcast
    coverage: neither line prints a number."""
    pytest.importorskip("pymdp", reason="the workspace needs pymdp")
    _run_main("single_theory")
    out = capsys.readouterr().out
    integration = _line(out, "Integration level:")
    coverage = _line(out, "Broadcast coverage:")
    assert not NUMBER.search(integration), integration
    assert integration == "not measured (no neural signals were passed)"
    assert not NUMBER.search(coverage), coverage
    assert coverage == "unmeasured (no receiver is registered)"


def test_basic_assessment_prints_no_overall_number_when_nothing_was_measured(
    example_env, capsys
):
    """The modules of the basic example have run no cycle, so the report says
    insufficient_data and the overall line prints no number (never a 0.000
    built from absent indicators)."""
    pytest.importorskip("pymdp", reason="the workspace needs pymdp")
    _run_main("basic_assessment")
    out = capsys.readouterr().out
    assert _line(out, "Status:") == "insufficient_data"
    assert _line(out, "Overall score:") == "none"


def test_basic_assessment_formats_a_none_theory_score(capsys):
    """The report printer shows a theory score that is None as ``none`` and
    raises nothing."""
    from datetime import datetime, timezone

    from mtc.assessment.assessment import ConsciousnessReport

    report = ConsciousnessReport(
        session_id="example",
        timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),
        overall_score=None,
        architecture_functional=False,
        confidence=0.0,
        indicator_results={},
        passing_count=0,
        total_indicators=30,
        theory_scores={"GWT": None, "HOT": 0.25},
        phi_measurement=None,
        status="insufficient_data",
        coverage=0.0,
    )
    _example("basic_assessment")["print_report"](report, positive=30, negative=3)
    out = capsys.readouterr().out
    lines = {text.split()[0]: text.split()[1:] for text in out.splitlines() if text}
    assert lines["GWT"] == ["none"]
    assert lines["HOT"] == ["0.250"]
    assert "theories with no measured indicator are not listed" in out


WORKSPACE_EXAMPLES = (
    "basic_assessment",
    "multi_theory_pipeline",
    "single_theory",
    "full_pipeline_demo",
)


def _run_as_script(name: str, monkeypatch, *args: str) -> None:
    path = ROOT / "examples" / f"{name}.py"
    monkeypatch.setattr(sys, "argv", [str(path), *args])
    runpy.run_path(str(path), run_name="__main__")


def _without_pymdp(monkeypatch) -> None:
    """The active inference module as it loads on an install without pymdp:
    no agent class, so its constructor raises its own missing-extra error."""
    from mtc.consciousness import active_inference

    monkeypatch.setattr(active_inference, "Agent", None)


def _constructor_raises(monkeypatch, error: ImportError) -> None:
    """The active inference constructor raises ``error``."""
    from mtc.consciousness import active_inference

    def refuse(self, *args, **kwargs):
        raise error

    monkeypatch.setattr(active_inference.ActiveInferenceModule, "__init__", refuse)


def _script_args(name: str, tmp_path: Path) -> Tuple[str, ...]:
    if name == "full_pipeline_demo":
        return ("--state-dir", str(tmp_path / "demo"), "--ticks", "1")
    return ()


@pytest.mark.parametrize("name", WORKSPACE_EXAMPLES)
def test_examples_without_pymdp_print_one_line(name, example_env, monkeypatch, capsys):
    """On an install without the fep extra each example that builds the
    workspace prints one line on stderr that names the extra, no traceback,
    and exits 2 (a configuration error)."""
    from mtc.cli.missing_extra import LINE
    from mtc.consciousness.active_inference import (
        PYMDP_MISSING_MESSAGE,
        ActiveInferenceModule,
    )
    from mtc.core.logger import reset_logging

    _without_pymdp(monkeypatch)
    with pytest.raises(ImportError, match=re.escape(PYMDP_MISSING_MESSAGE)):
        ActiveInferenceModule()
    # The demo sets its token variable; restore the environment afterwards.
    monkeypatch.setenv("MTC_DEMO_OPERATOR_TOKEN", "unused")
    try:
        with pytest.raises(SystemExit) as stop:
            _run_as_script(name, monkeypatch, *_script_args(name, example_env))
    finally:
        reset_logging()
    err = capsys.readouterr().err
    assert stop.value.code == 2
    assert [line for line in err.splitlines() if line.strip()] == [LINE]
    assert "Traceback" not in err


@pytest.mark.parametrize("name", WORKSPACE_EXAMPLES)
def test_examples_raise_other_import_errors(name, example_env, monkeypatch):
    """Any other ImportError is not a missing extra and propagates."""
    from mtc.core.logger import reset_logging

    monkeypatch.setenv("MTC_DEMO_OPERATOR_TOKEN", "unused")
    other = ImportError("an unrelated module is broken", name="unrelatedlib")
    _constructor_raises(monkeypatch, other)
    try:
        with pytest.raises(ImportError) as raised:
            _run_as_script(name, monkeypatch, *_script_args(name, example_env))
    finally:
        reset_logging()
    assert raised.value is other
