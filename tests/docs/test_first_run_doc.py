"""Every number in docs/FIRST_RUN.md's tables equals its source in the code.

Each warmup row names its source as a dotted path (and, for a constructor or
function default, the parameter); the test resolves that source in the code
and compares. The cadence table is compared with the thought machine's
built-in defaults. The doctor tables (each install's summary and every line
that is not ok) are compared with doctor run on a wizard-made project with
each install's extras and model cache stood in.
"""

import dataclasses
import importlib
import inspect
import re
from pathlib import Path
from typing import List

from mtc.consciousness.continuous.config import Cadence

# The wizard-made project with stand-ins for the extras, the model hub and
# PyPhi, shared with doctor's own tests; requested by name below.
from tests.cli.test_doctor import environment, settings_for  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "FIRST_RUN.md"

# Rows whose number is a literal in a method or a call, read from the source.
SOURCE_LITERALS = {
    "mtc.assessment.assessment.ConsciousnessAssessment."
    "_v05_prediction_error_minimization": (
        "mtc/assessment/assessment.py",
        r"def _v05_prediction_error_minimization\(.*?len\(\w+\) < (\d+)",
    ),
    "mtc.pipeline.runtime": (
        "mtc/pipeline/runtime.py",
        r"build_phi_tracker\([^\n]*?,\s*(\d+)\)",
    ),
    "mtc.api.server.ASSESSMENT_CACHE_S": (
        "mtc/api/server.py",
        r"^ASSESSMENT_CACHE_S = ([0-9.]+)$",
    ),
}
REQUIRED = {
    "mtc.consciousness.phi_tracker.PhiTracker",
    "mtc.consciousness.phi_tracker.PhiTracker._PHI_NORM_MIN_HISTORY",
    "mtc.consciousness.active_inference.ActiveInferenceModule.BASELINE_MIN_SAMPLES",
    "mtc.consciousness.prediction_ledger.PredictionLedger",
    "mtc.consciousness.pad_affect.PADAffectModel",
    "mtc.pipeline.consciousness_orchestrator.OrchestratorConfig.rpt_every_n_cycles",
    "mtc.pipeline.consciousness_orchestrator.OrchestratorConfig."
    "causal_emergence_every_n_cycles",
    "mtc.pipeline.consciousness_orchestrator.OrchestratorConfig.heartbeat_s",
    "mtc.pipeline.consciousness_helpers.DEFAULT_LEVEL_INTERVAL_S",
    "mtc.assessment.assessment.ConsciousnessAssessment.run_noise_baseline",
} | set(SOURCE_LITERALS)


def _resolve(dotted: str):
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attribute in parts[cut:]:
            obj = getattr(obj, attribute)
        return obj
    raise ImportError(dotted)


def _code_value(dotted: str, parameter):
    if dotted in SOURCE_LITERALS:
        relative, pattern = SOURCE_LITERALS[dotted]
        source = (ROOT / relative).read_text(encoding="utf-8")
        match = re.search(pattern, source, re.S | re.M)
        assert match, f"{dotted}: the source no longer has the literal"
        return float(match.group(1))
    obj = _resolve(dotted)
    if parameter is None:
        if dataclasses.is_dataclass(obj) or inspect.isclass(obj):
            raise AssertionError(f"{dotted}: a class source needs a parameter")
        return float(obj)
    target = obj.__init__ if inspect.isclass(obj) else obj
    default = inspect.signature(target).parameters[parameter].default
    assert default is not inspect.Parameter.empty, f"{dotted} {parameter}"
    return float(default)


def _warmup_rows(text: str):
    section = text.split("## Warmup thresholds", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] in ("Threshold", "") or set(cells[0]) <= {"-"}:
            continue
        ticks = re.findall(r"`([^`]+)`", cells[3])
        assert ticks, f"row without a source: {cells[0]}"
        dotted = ticks[0]
        parameter = (
            ticks[1] if len(ticks) > 1 and dotted not in SOURCE_LITERALS else None
        )
        rows.append((cells[0], float(cells[1]), dotted, parameter))
    return rows


def test_first_run_doc_matches_code():
    text = DOC.read_text(encoding="utf-8")
    rows = _warmup_rows(text)
    seen = set()
    for label, stated, dotted, parameter in rows:
        assert _code_value(dotted, parameter) == stated, (label, dotted, parameter)
        seen.add(dotted)
    missing = REQUIRED - seen
    assert not missing, f"thresholds missing from FIRST_RUN.md: {sorted(missing)}"


def test_first_run_required_checks_match_doctor():
    """The doc's list of exit-deciding checks equals doctor's required set.

    Required means flagged required in ``CHECKS`` or, for ``extras``, made a
    required failure by a missing extra in ``BLOCKING_EXTRAS``.
    """
    from mtc.cli import doctor

    text = DOC.read_text(encoding="utf-8")
    match = re.search(
        r"The required checks \((.*?)\) decide the\s+exit code", text, re.S
    )
    assert match, "FIRST_RUN.md no longer lists the required checks"
    listed = match.group(1)
    store_checks = {f"store:{name}" for name in doctor.STORE_NAMES}
    names = {name for name, _ in doctor.CHECKS} - store_checks
    documented = set(re.findall(r"`([^`]+)`", listed)) & names
    required = {name for name, flag in doctor.CHECKS if flag} - store_checks
    if doctor.BLOCKING_EXTRAS:
        required.add("extras")
    assert documented == required
    assert store_checks <= {name for name, flag in doctor.CHECKS if flag}
    assert f"the {len(doctor.STORE_NAMES)} stores" in listed.replace(
        "five", "5"
    ), "the stores clause no longer matches STORE_NAMES"


def _doctor_section(text: str) -> str:
    return text.split("## A fresh `mtc doctor`", 1)[1].split("\n## ", 1)[0]


def _table(section: str, header: str) -> List[List[str]]:
    """The body rows of the table whose header row starts with ``header``."""
    lines = section.splitlines()
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] == header:
            rows = []
            for body in lines[index + 2 :]:
                if not body.startswith("|"):
                    break
                rows.append(
                    [
                        cell.strip().strip("`")
                        for cell in body.strip().strip("|").split("|")
                    ]
                )
            return rows
    raise AssertionError(f"FIRST_RUN.md has no table headed {header!r}")


# How each install the doctor tables name is stood in for: the modules its
# extras leave out and whether the embedding model is in the model cache. The
# iit extra is in none of them, so PyPhi does not import on any.
INSTALLS = {
    "[all]": {"absent": set(), "cached": False},
    "[all], fresh": {"absent": set(), "cached": False},
    "[all], after mtc doctor --fetch-models": {"absent": set(), "cached": True},
    "without the embeddings extra": {
        "absent": {"sentence_transformers"},
        "cached": False,
    },
    "without embeddings": {"absent": {"sentence_transformers"}, "cached": False},
}


def _doctor_on(state, install: str):
    from mtc.cli import doctor

    label = install.replace("`", "")
    assert label in INSTALLS, f"FIRST_RUN.md names an unknown install: {label}"
    stand_in = INSTALLS[label]
    state.absent.clear()
    state.absent.update(stand_in["absent"])
    state.hub.cached = stand_in["cached"]
    return doctor.run_doctor(settings_for(state), network=False)


def test_first_run_doctor_tables_match_doctor(request):
    """Each install's summary and every non-ok line the page states are what
    doctor reports on a wizard-made project with that install stood in."""
    from mtc.core.checks import CheckStatus, summarize

    state = request.getfixturevalue("environment")
    section = _doctor_section(DOC.read_text(encoding="utf-8"))

    summaries = _table(section, "Install")
    assert len(summaries) >= 3, summaries
    for install, summary in summaries:
        counts = summarize(_doctor_on(state, install))
        stated = dict(
            (key, int(value)) for key, value in re.findall(r"(\w+)=(\d+)", summary)
        )
        assert set(stated) == {status.value for status in CheckStatus}, summary
        assert stated == {key: counts[key] for key in stated}, (install, counts)

    header = None
    for line in section.splitlines():
        if line.startswith("| Check |"):
            header = [cell.strip() for cell in line.strip().strip("|").split("|")]
    assert header, "FIRST_RUN.md has no per-check doctor table"
    rows = _table(section, "Check")
    for column, install in enumerate(header[1:], start=1):
        results = _doctor_on(state, install)
        actual = {result.name: CheckStatus(result.status).value for result in results}
        stated = {row[0]: row[column] for row in rows}
        for name, status in stated.items():
            assert actual.get(name) == status, (install, name, actual.get(name))
        not_ok = {name for name, status in actual.items() if status != "ok"}
        missing = not_ok - set(stated)
        assert not missing, f"{install}: non-ok lines the page omits: {missing}"


def test_first_run_cadence_table_matches_defaults():
    text = DOC.read_text(encoding="utf-8")
    defaults = Cadence()
    found = dict(
        (state, float(rate))
        for state, rate in re.findall(
            r"^\| `?(awake|drowsy|dreaming|deep_sleep)`? \| ([0-9.]+) \|", text, re.M
        )
    )
    assert found == {
        state: float(getattr(defaults, state))
        for state in ("awake", "drowsy", "dreaming", "deep_sleep")
    }
