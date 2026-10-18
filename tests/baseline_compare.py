"""Comparators for the baseline and reproducibility tests.

Three pure comparators (no file, process or network access):

- ``metric_block`` reads the block the experiment runner prints after its
  ``---`` line into an ordered mapping of name to printed string.
- ``diff_blocks`` lists every metric line whose printed string differs between
  two blocks, or that only one of them has. Printed strings are compared
  exactly, so the comparison holds at the runner's print resolution.
- ``assessment_diff`` lists every field that differs between two assessment
  reports (the JSON the assessment CLI writes). Statuses, counts, keys,
  strings and booleans are compared exactly; floats within
  ``FLOAT_TOLERANCE``. Only the session id, the timestamp and the timing
  fields are left out, plus the evidence values a caller names (the three
  readouts of ``WALL_CLOCK_EVIDENCE_KEYS`` for a run that keeps a wall-clock
  cooldown).

Two functions read the running process and are kept apart from the
comparators: ``current_environment`` (package versions; the baseline tests
print its difference from the recorded environment when a comparison fails,
so that an environment change is told apart from a code change) and
``runner_environment`` (the environment a test gives a runner subprocess).
"""

import math
import re
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Sequence, Tuple, Union


class _Missing:
    """Marks the side of a difference that has no such line or field."""

    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()

Difference = Tuple[str, Any, Any]

BLOCK_MARKER = "---"

# The two runner lines that report durations: the only lines diff_blocks
# leaves out by default.
TIMING_LINES = ("processing_ms", "total_seconds")

# Floats of an assessment report are compared within this absolute bound.
FLOAT_TOLERANCE = 1e-9

# Report fields that identify or time a run and carry no measurement.
REPORT_RUN_FIELDS = ("session_id", "timestamp", "processing_time_ms")
INDICATOR_TIMING_FIELD = "measurement_time_ms"
PHI_TIMING_FIELD = "computation_time_ms"

# Top-level mappings the report gained after a fixture was captured: absent on
# the expected side they read as an empty mapping, so an older fixture equals
# a report that holds no entry under them, while any entry is still reported.
# ``failures`` holds the per-run failure counts by step.
LATER_TOP_LEVEL_MAPPINGS = ("failures",)

# The evidence keys of a seeded assessment report whose value depends on how
# much wall-clock time passes between warmup cycles, as (indicator, key).
# Reason: MetacognitionModule.introspect returns early while its wall-clock
# cooldown (introspection_cooldown, 1.0 s unless set) has not elapsed, and a
# completed pass generates up to three further higher-order thoughts. The
# number of thoughts generated and of introspections counted after N cycles
# therefore follows the speed and the load of the machine (measured with the
# clock scaled: 8 and 7, 11 and 11, or 14 and 15 for one seed). Both counters
# reach the report only as these three evidence keys: the scores they feed
# are capped once the counter passes the cycle count, so no score, status,
# confidence or other evidence key moves with them. A comparison that
# involves a run with a wall-clock cooldown passes this set as
# ``ignore_evidence``; each key must still be present on both sides with the
# same type. The set is pinned by a test and must not grow.
WALL_CLOCK_EVIDENCE_KEYS = frozenset(
    {
        ("higher_order_representations", "hots_generated"),
        ("higher_order_representations", "coverage_ratio"),
        ("metacognition", "introspections"),
    }
)

# The packages the environment record lists, as (label, distribution name).
ENVIRONMENT_DISTRIBUTIONS = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("torch", "torch"),
    ("snntorch", "snntorch"),
    ("reservoirpy", "reservoirpy"),
    ("pymdp", "inferactively-pymdp"),
    ("pyphi", "pyphi"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic-settings"),
)
NOT_INSTALLED = "not installed"

# Every store stays off in a runner subprocess.
STORE_ENABLE_KEYS = (
    "MTC_POSTGRES_ENABLED",
    "MTC_MONGODB_ENABLED",
    "MTC_REDIS_ENABLED",
    "MTC_QDRANT_ENABLED",
    "MTC_NEO4J_ENABLED",
)

_METRIC_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(.*)$")


# -- runner output blocks ------------------------------------------------


def metric_block(text: str) -> Dict[str, str]:
    """The ``name: value`` lines after the first ``---`` line, in print order.

    Each value is the printed string with the padding around it removed; no
    number is parsed. Lines that are not ``name: value`` (a second marker, a
    log line) are skipped. A text without a marker line, or one that prints a
    name twice, raises ValueError: neither is a block that can be compared.
    """
    if not isinstance(text, str):
        raise TypeError("metric_block reads the runner output as one string")
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == BLOCK_MARKER:
            start = index
            break
    if start is None:
        raise ValueError("no --- line: the text holds no runner metric block")
    block: Dict[str, str] = {}
    for line in lines[start + 1 :]:
        match = _METRIC_LINE.match(line.strip())
        if match is None:
            continue
        name = match.group(1)
        if name in block:
            raise ValueError(f"metric line {name!r} is printed twice")
        block[name] = match.group(2).strip()
    return block


def diff_blocks(
    expected: Mapping[str, str],
    actual: Mapping[str, str],
    ignore: Collection[str] = TIMING_LINES,
) -> List[Difference]:
    """Every metric line that differs between two blocks.

    Returns ``(name, expected, actual)`` for each name, outside ``ignore``,
    whose printed strings differ or that only one block has (the other side
    is ``MISSING``). Names follow the order of ``expected``, then the names
    only ``actual`` has.
    """
    for label, block in (("expected", expected), ("actual", actual)):
        if not isinstance(block, Mapping):
            raise TypeError(f"{label} must be a metric_block mapping")
    skipped = set(ignore)
    names = [name for name in expected if name not in skipped]
    names += [name for name in actual if name not in expected and name not in skipped]
    differences: List[Difference] = []
    for name in names:
        want = expected.get(name, MISSING)
        got = actual.get(name, MISSING)
        if want is MISSING or got is MISSING or want != got:
            differences.append((name, want, got))
    return differences


def theory_lines(block: Mapping[str, str]) -> List[str]:
    """The names of the per-theory score lines of a block, in print order."""
    return [
        name for name in block if name.endswith("_score") and name != "overall_score"
    ]


# -- assessment reports --------------------------------------------------


def assessment_diff(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    ignore_evidence: Collection[Tuple[str, str]] = (),
) -> List[Difference]:
    """Every field that differs between two assessment reports.

    Both arguments are reports as the assessment CLI writes them (JSON
    values). Returns ``(path, expected, actual)`` with a dotted path such as
    ``indicator_results.metacognition.evidence.raw_score``; a field only one
    report has carries ``MISSING`` on the other side.

    Compared exactly: every key set, strings (statuses, names, theories),
    booleans, integers, None and the sorted list of insufficient indicators.
    Floats are equal within ``FLOAT_TOLERANCE``; an integer never equals a
    float or a boolean. Left out: ``session_id``, ``timestamp``,
    ``processing_time_ms``, each indicator's ``measurement_time_ms`` and the
    Phi measurement's ``computation_time_ms``.

    ``ignore_evidence`` names evidence keys, as ``(indicator, key)``, whose
    values are not compared; each must still be present in both reports with
    the same type. A top-level mapping of ``LATER_TOP_LEVEL_MAPPINGS`` that
    the expected report lacks is compared as an empty mapping.
    """
    for label, report in (("expected", expected), ("actual", actual)):
        if not isinstance(report, Mapping):
            raise TypeError(f"{label} must be an assessment report mapping")
    differences: List[Difference] = []
    _walk((), expected, actual, frozenset(ignore_evidence), differences)
    return differences


def _is_timing(parts: Tuple[str, ...]) -> bool:
    if len(parts) == 1:
        return parts[0] in REPORT_RUN_FIELDS
    if len(parts) == 3 and parts[0] == "indicator_results":
        return parts[2] == INDICATOR_TIMING_FIELD
    return parts == ("phi_measurement", PHI_TIMING_FIELD)


def _is_ignored_evidence(parts: Tuple[str, ...], ignored: Collection) -> bool:
    return (
        len(parts) == 4
        and parts[0] == "indicator_results"
        and parts[2] == "evidence"
        and (parts[1], parts[3]) in ignored
    )


def _path(parts: Tuple[str, ...]) -> str:
    text = ""
    for part in parts:
        if part.startswith("["):
            text += part
        else:
            text += ("." if text else "") + part
    return text


def _same_scalar(want: Any, got: Any) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return isinstance(want, bool) and isinstance(got, bool) and want == got
    if isinstance(want, float) and isinstance(got, float):
        if math.isnan(want) or math.isnan(got):
            return math.isnan(want) and math.isnan(got)
        if math.isinf(want) or math.isinf(got):
            return want == got
        return abs(want - got) <= FLOAT_TOLERANCE
    if type(want) is not type(got):
        return False
    return want == got


def _walk(
    parts: Tuple[str, ...],
    want: Any,
    got: Any,
    ignored: Collection,
    out: List[Difference],
) -> None:
    if _is_timing(parts):
        return
    if _is_ignored_evidence(parts, ignored):
        if type(want) is not type(got):
            out.append((_path(parts), want, got))
        return
    if isinstance(want, Mapping) and isinstance(got, Mapping):
        keys = list(want) + [key for key in got if key not in want]
        for key in keys:
            child = parts + (str(key),)
            if _is_timing(child):
                continue
            if key not in want:
                if (
                    not parts
                    and key in LATER_TOP_LEVEL_MAPPINGS
                    and isinstance(got[key], Mapping)
                ):
                    _walk(child, {}, got[key], ignored, out)
                    continue
                out.append((_path(child), MISSING, got[key]))
            elif key not in got:
                out.append((_path(child), want[key], MISSING))
            else:
                _walk(child, want[key], got[key], ignored, out)
        return
    if isinstance(want, list) and isinstance(got, list):
        if parts == ("insufficient_indicators",):
            if sorted(want) != sorted(got):
                out.append((_path(parts), sorted(want), sorted(got)))
            return
        if len(want) != len(got):
            out.append((_path(parts) + ".length", len(want), len(got)))
        for index, (left, right) in enumerate(zip(want, got)):
            _walk(parts + (f"[{index}]",), left, right, ignored, out)
        return
    if not _same_scalar(want, got):
        out.append((_path(parts), want, got))


# -- the runner subprocess -----------------------------------------------


def runner_environment(
    root: Union[str, Path], scratch: Union[str, Path]
) -> Dict[str, str]:
    """The environment a baseline test gives a runner subprocess.

    The caller's environment without any ``MTC_`` setting or provider key,
    then: ``root`` first on ``PYTHONPATH`` (the tree under test is the one
    imported), the PyPhi welcome off, the CPU device, the mock provider,
    every store off, the data directory under ``scratch`` (so a run writes
    nothing into the tree) and an API port that is free at the time of the
    call (the runners never bind it).
    """
    import os
    import socket

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MTC_") and key != "ANTHROPIC_API_KEY"
    }
    search_path = [str(root)]
    if environment.get("PYTHONPATH"):
        search_path.append(environment["PYTHONPATH"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(search_path),
            "PYPHI_WELCOME_OFF": "yes",
            "MTC_DEVICE": "cpu",
            "MTC_LLM_PROVIDER": "mock",
            "MTC_DATA_DIRECTORY": str(Path(scratch) / "data"),
            "MTC_API_PORT": str(free_port),
        }
    )
    environment.update({key: "false" for key in STORE_ENABLE_KEYS})
    return environment


# -- messages and the environment record ---------------------------------


def format_differences(differences: Sequence[Difference]) -> str:
    """One line per difference, for an assertion message."""
    if not differences:
        return "no difference"
    return "\n".join(
        f"  {name}: expected {want!r}, actual {got!r}"
        for name, want, got in differences
    )


def current_environment() -> Dict[str, str]:
    """The running interpreter and package versions, as the record lists them.

    The labels and their order match ``tests/fixtures/baseline_v050_env.txt``:
    ``python`` with version, machine and system, then one entry per
    distribution of ``ENVIRONMENT_DISTRIBUTIONS`` (``NOT_INSTALLED`` when the
    distribution is absent).
    """
    import platform
    from importlib import metadata

    environment = {
        "python": " ".join(
            (platform.python_version(), platform.machine(), platform.system())
        )
    }
    for label, distribution in ENVIRONMENT_DISTRIBUTIONS:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            environment[label] = NOT_INSTALLED
            continue
        if label != distribution.replace("-", "_"):
            # The import name differs from the distribution name: the record
            # spells both out
            version = f"installed as distribution {distribution}, version {version}."
        environment[label] = version
    return environment


def environment_text(environment: Mapping[str, str]) -> str:
    """The environment record as file text: one ``label value`` line each."""
    return "".join(f"{label} {value}\n" for label, value in environment.items())


def recorded_environment(text: str) -> Dict[str, str]:
    """An environment record parsed back into label and value."""
    environment: Dict[str, str] = {}
    for line in text.splitlines():
        label, _, value = line.strip().partition(" ")
        if label:
            environment[label] = value.strip()
    return environment


def environment_diff(
    recorded: Mapping[str, str], current: Mapping[str, str]
) -> List[Difference]:
    """Every environment entry that differs between the record and now."""
    return diff_blocks(recorded, current, ignore=())


def explain(
    differences: Sequence[Difference],
    recorded: Mapping[str, str],
    current: Mapping[str, str],
) -> str:
    """The differences, then how the environment differs from the record."""
    drift = environment_diff(recorded, current)
    if drift:
        environment = (
            "the running environment differs from the recorded one:\n"
            + format_differences(drift)
        )
    else:
        environment = "the running environment equals the recorded one"
    return f"{len(differences)} difference(s):\n{format_differences(differences)}\n{environment}"
