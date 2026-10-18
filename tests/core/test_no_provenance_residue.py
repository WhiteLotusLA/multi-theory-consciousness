"""No provenance residue in the tree.

Three properties over the Python sources:

- no line of a Python file under mtc, tests, scripts or examples starts with a
  creation, author or authors header;
- no docstring or log string under mtc, scripts or examples names a port
  phase (the word Phase followed by a number);
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


def phase_mentions(paths: Iterable[Path], base: Path = ROOT) -> List[str]:
    found = set()
    for path in paths:
        tree = _parse(path)
        if tree is None:
            continue
        for lineno, text in list(_docstrings(tree)) + list(_log_strings(tree)):
            if PORT_PHASE.search(text):
                found.add(f"{path.relative_to(base).as_posix()}:{lineno}")
    return sorted(found)


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
