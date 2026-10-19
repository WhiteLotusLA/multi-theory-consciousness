"""No provenance residue in the tree.

Three properties over the Python sources:

- no line of a Python file under mtc, tests, scripts or examples starts with a
  creation, author or authors header;
- no docstring or log string under mtc, scripts, examples or tests names a
  port phase (the word Phase followed by a number), and no Python file under
  mtc, tests, scripts or examples carries one in its name;
- no docstring or log string under mtc, scripts, examples or tests names the
  final review stage of the port by its stage name (the word release joined
  to the word gate);
- no docstring under mtc/pipeline uses a gendered pronoun (the pipeline's
  docstrings speak of a speaker, never of a he or a she).

Comments are outside the docstring and log-string properties, so algorithm
step comments are not matched.
"""

import ast
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[2]

HEADER = re.compile(
    r"^\s*(?:#\s*)?(?:"
    r"Created(?:\s+(?:by|on|at))?\s*:"
    r"|Created by\b"
    r"|Creation date\b"
    r"|Authors?\s*:"
    r"|@author\b"
    r"|__author__\s*="
    r")"
)
PORT_PHASE = re.compile(r"\bPhase\s+\d")
PORT_PHASE_NAME = re.compile(r"phase[_-]?\d", re.I)
STAGE_LABEL = re.compile(r"\brelease[\s-]+gates?\b", re.I)
GENDERED = re.compile(r"\b(?:he|she|him|her|hers|his|himself|herself)\b", re.I)
LOG_METHODS = {
    "debug",
    "info",
    "warning",
    "warn",
    "error",
    "exception",
    "critical",
    "log",
}


def _python_files(*tops: str) -> List[Path]:
    files: List[Path] = []
    for top in tops:
        files.extend(sorted((ROOT / top).rglob("*.py")))
    return files


def _parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return None


def _docstrings(tree: ast.AST) -> Iterator[Tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                yield body[0].value.lineno, body[0].value.value


def _log_strings(tree: ast.AST) -> Iterator[Tuple[int, str]]:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in LOG_METHODS
        ):
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        yield node.lineno, sub.value


def header_lines(paths: Iterable[Path], base: Path = ROOT) -> List[str]:
    found = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if HEADER.match(line):
                found.append(f"{path.relative_to(base).as_posix()}:{lineno}")
    return found


def _string_mentions(
    paths: Iterable[Path], pattern: "re.Pattern[str]", base: Path
) -> List[str]:
    found = set()
    for path in paths:
        tree = _parse(path)
        if tree is None:
            continue
        for lineno, text in list(_docstrings(tree)) + list(_log_strings(tree)):
            if pattern.search(text):
                found.add(f"{path.relative_to(base).as_posix()}:{lineno}")
    return sorted(found)


def phase_mentions(paths: Iterable[Path], base: Path = ROOT) -> List[str]:
    return _string_mentions(paths, PORT_PHASE, base)


def stage_label_mentions(paths: Iterable[Path], base: Path = ROOT) -> List[str]:
    return _string_mentions(paths, STAGE_LABEL, base)


def phase_file_names(paths: Iterable[Path], base: Path = ROOT) -> List[str]:
    return sorted(
        path.relative_to(base).as_posix()
        for path in paths
        if PORT_PHASE_NAME.search(path.stem)
    )


def gendered_docstrings(paths: Iterable[Path], base: Path = ROOT) -> List[str]:
    found = []
    for path in paths:
        tree = _parse(path)
        if tree is None:
            continue
        for lineno, text in _docstrings(tree):
            if GENDERED.search(text):
                found.append(f"{path.relative_to(base).as_posix()}:{lineno}")
    return found


def test_no_creation_or_author_header():
    hits = header_lines(_python_files("mtc", "tests", "scripts", "examples"))
    assert hits == [], hits


def test_no_port_phase_in_docstrings_or_log_strings():
    hits = phase_mentions(_python_files("mtc", "scripts", "examples"))
    assert hits == [], hits


def test_no_port_phase_in_test_docstrings_or_log_strings():
    hits = phase_mentions(_python_files("tests"))
    assert hits == [], hits


def test_no_port_phase_in_python_file_names():
    hits = phase_file_names(_python_files("mtc", "tests", "scripts", "examples"))
    assert hits == [], hits


def test_no_release_gate_label_in_docstrings_or_log_strings():
    hits = stage_label_mentions(_python_files("mtc", "tests", "scripts", "examples"))
    assert hits == [], hits


def test_no_gendered_pronoun_in_pipeline_docstrings():
    hits = gendered_docstrings(_python_files("mtc/pipeline"))
    assert hits == [], hits


def test_the_scans_find_planted_residue(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        '"""A module.\n\nAdded in Phase 4 of the port.\n"""\n'
        "# Author: Testagent\n"
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        "def f(speaker):\n"
        '    """Greet the speaker as she arrives."""\n'
        '    logger.info("built in Phase 7")\n'
        "    # Phase 3 of the algorithm: a step comment, outside the scan.\n"
        "    author: str = speaker\n"
        "    return author\n",
        encoding="utf-8",
    )
    assert header_lines([planted], base=tmp_path) == ["planted.py:5"]
    assert phase_mentions([planted], base=tmp_path) == [
        "planted.py:1",
        "planted.py:12",
    ]
    assert gendered_docstrings([planted], base=tmp_path) == ["planted.py:11"]
    clean = tmp_path / "clean.py"
    clean.write_text(
        '"""Each stage runs in order; the speaker gives their name."""\n'
        "# Phase 2 of the algorithm, a comment.\n"
        "author: str = 'x'\n",
        encoding="utf-8",
    )
    assert header_lines([clean], base=tmp_path) == []
    assert phase_mentions([clean], base=tmp_path) == []
    assert gendered_docstrings([clean], base=tmp_path) == []


def test_the_file_name_scan_finds_a_planted_phase_label(tmp_path):
    names = [
        "test_settings_phase4.py",
        "test_settings_Phase_6_brain.py",
        "phase-7.py",
        "test_settings_brain.py",
        "phase_offset.py",
        "two_phase_commit.py",
    ]
    for name in names:
        (tmp_path / name).write_text("", encoding="utf-8")
    found = phase_file_names([tmp_path / name for name in names], base=tmp_path)
    assert found == [
        "phase-7.py",
        "test_settings_Phase_6_brain.py",
        "test_settings_phase4.py",
    ]


def test_the_stage_label_scan_finds_a_planted_release_gate(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        '"""The keys that joined at the release gate."""\n'
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "\n"
        "def f():\n"
        '    """Read the Release-Gate safety keys."""\n'
        '    logger.warning("fixed by the release  gates")\n'
        "    # release gate: a comment, outside the scan.\n"
        "    return 1\n",
        encoding="utf-8",
    )
    assert stage_label_mentions([planted], base=tmp_path) == [
        "planted.py:1",
        "planted.py:7",
        "planted.py:8",
    ]
    clean = tmp_path / "clean.py"
    clean.write_text(
        '"""SAFETY_GATE_FAIL_CLOSED: the gate releases a held reply."""\n'
        "# release gate, a comment.\n"
        "release_gate_count = 0\n",
        encoding="utf-8",
    )
    assert stage_label_mentions([clean], base=tmp_path) == []
