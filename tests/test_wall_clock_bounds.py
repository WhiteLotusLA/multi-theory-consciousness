"""A test whose pass depends on the machine's speed is marked ``slow``.

The per-push CI deselects ``slow`` and runs on the host's shared runners,
which are several times slower than the machine the bounds were written on:
the first run on the pushed v0.5.0 main (2026-10-20) failed the active
inference timing test at 92 and 79 seconds against its 60 second bound on two
of the three matrices while the other 4862 tests passed. A bound on a
wall-clock reading is a measurement of the machine, not of the code, so it
belongs to the ``slow`` set (with ``benchmark`` where it measures
performance), which the maintainer's wide gate runs.

The scan reads every test module as a syntax tree and finds the assertions
that bound a wall-clock reading from above: ``assert NAME < NUMBER`` (or
``<=``) where NAME is one of the readings below. Each such test must carry
``slow``, ``db`` or ``live`` on the function, its class or the module. A
reading under another name is outside the scan's reach; name it here when a
test adds one.
"""

import ast
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
READINGS = frozenset(
    {"elapsed", "elapsed_s", "elapsed_ms", "duration", "duration_ms", "took"}
)
DESELECTED = frozenset({"slow", "db", "live"})


def _mark_names(node: ast.AST) -> Set[str]:
    """The pytest mark names in a decorator or a ``pytestmark`` value."""
    names: Set[str] = set()
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Attribute)
            and isinstance(item.value, ast.Attribute)
            and item.value.attr == "mark"
            and isinstance(item.value.value, ast.Name)
            and item.value.value.id == "pytest"
        ):
            names.add(item.attr)
    return names


def _module_marks(tree: ast.Module) -> Set[str]:
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            names |= _mark_names(node.value)
    return names


def _bounds_a_reading(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        compare = node.test
        if (
            isinstance(compare.left, ast.Name)
            and compare.left.id in READINGS
            and len(compare.ops) == 1
            and isinstance(compare.ops[0], (ast.Lt, ast.LtE))
            and isinstance(compare.comparators[0], ast.Constant)
            and isinstance(compare.comparators[0].value, (int, float))
        ):
            return True
    return False


def _tests(tree: ast.Module):
    """``(name, decorator marks)`` for every test function, class marks included."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                yield node, _mark_names(
                    ast.Module(body=node.decorator_list, type_ignores=[])
                )
        elif isinstance(node, ast.ClassDef):
            class_marks = _mark_names(
                ast.Module(body=node.decorator_list, type_ignores=[])
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("test"):
                        marks = _mark_names(
                            ast.Module(body=item.decorator_list, type_ignores=[])
                        )
                        yield item, marks | class_marks


def bounded_tests_outside_the_deselected_marks() -> List[str]:
    """``path::test`` for every bounded wall-clock reading the per-push CI runs."""
    offenders: List[str] = []
    for path in sorted(TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_marks = _module_marks(tree)
        for function, marks in _tests(tree):
            if _bounds_a_reading(function) and not (
                (marks | module_marks) & DESELECTED
            ):
                offenders.append(
                    f"{path.relative_to(ROOT).as_posix()}::{function.name}"
                )
    return offenders


def test_the_scan_sees_a_bound_and_the_marks():
    source = (
        "import pytest\n"
        "pytestmark = pytest.mark.slow\n"
        "class TestSpeed:\n"
        "    @pytest.mark.benchmark\n"
        "    def test_fast(self):\n"
        "        elapsed = 0.1\n"
        "        assert elapsed < 1.0\n"
        "def test_unbounded():\n"
        "    elapsed = 0.1\n"
        "    assert elapsed >= 0.0\n"
    )
    tree = ast.parse(source)
    assert _module_marks(tree) == {"slow"}
    found = {function.name: marks for function, marks in _tests(tree)}
    assert found == {"test_fast": {"benchmark"}, "test_unbounded": set()}
    functions = {function.name: function for function, _ in _tests(tree)}
    assert _bounds_a_reading(functions["test_fast"]) is True
    assert _bounds_a_reading(functions["test_unbounded"]) is False


def test_bounded_wall_clock_readings_are_marked_slow():
    assert bounded_tests_outside_the_deselected_marks() == []
