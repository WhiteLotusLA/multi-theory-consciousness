"""A call that Python or a library has deprecated leaves the tree before the
removal lands, and this scan refuses it afterwards.

The v0.5.0 release-candidate matrices (2026-10-19, Python 3.14, pymc 6.3.2)
reported two such calls, and the host's first CI runs on the pushed v0.5.0
(2026-10-20) showed the first again as the suite's one warning. Python 3.14
deprecates ``asyncio.iscoroutinefunction`` ("slated for removal in Python
3.16; use inspect.iscoroutinefunction() instead"), which one test called.
pymc deprecates ``return_inferencedata=False`` on ``pm.sample`` ("will be
removed in a future release. Use the default ``return_inferencedata=True``
and work with the returned ``InferenceData`` object"), which the DCM
benchmark's experimental posterior path passed so it could read a plain
trace; that path would have stopped working the day pymc removed the
argument.

The scan reads every Python module under ``mtc``, ``tests``, ``scripts`` and
``examples`` as a syntax tree and names each call of ``<module>.<name>`` for a
listed module attribute, each ``from <module> import <name>`` of one, and
each call that passes a listed keyword argument. A deprecated call reached
another way (an alias of the module, a keyword passed through ``**kwargs``)
is outside the scan's reach; add its shape here when a deprecation report
names one.
"""

import ast
from pathlib import Path
from typing import Dict, FrozenSet, List

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("mtc", "tests", "scripts", "examples")
# Module attributes Python has deprecated, by module. The replacement for
# asyncio.iscoroutinefunction is inspect.iscoroutinefunction.
DEPRECATED_ATTRIBUTES: Dict[str, FrozenSet[str]] = {
    "asyncio": frozenset({"iscoroutinefunction"}),
}
# Keyword arguments a library has deprecated. pymc's pm.sample returns
# InferenceData by default, and the caller reads the posterior from it.
DEPRECATED_KEYWORDS: FrozenSet[str] = frozenset({"return_inferencedata"})


def deprecated_calls(tree: ast.Module) -> List[str]:
    """``line: shape`` for every deprecated call in one module, by line."""
    found: List[tuple] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in DEPRECATED_ATTRIBUTES:
            for alias in node.names:
                if alias.name in DEPRECATED_ATTRIBUTES[node.module]:
                    found.append(
                        (node.lineno, f"from {node.module} import {alias.name}")
                    )
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.attr in DEPRECATED_ATTRIBUTES.get(func.value.id, frozenset())
            ):
                found.append((node.lineno, f"{func.value.id}.{func.attr}()"))
            for keyword in node.keywords:
                if keyword.arg in DEPRECATED_KEYWORDS:
                    found.append((node.lineno, f"{keyword.arg}="))
    return [f"{line}: {shape}" for line, shape in sorted(found)]


def deprecated_calls_in_the_tree() -> List[str]:
    """``path:line: shape`` for every deprecated call under the scanned trees."""
    offenders: List[str] = []
    for directory in SCANNED:
        for path in sorted((ROOT / directory).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for item in deprecated_calls(tree):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{item}")
    return offenders


def test_the_scan_sees_each_shape_and_ignores_the_replacement():
    source = (
        "import asyncio\n"
        "import inspect\n"
        "from asyncio import iscoroutinefunction\n"
        "def check(function, pm):\n"
        "    old = asyncio.iscoroutinefunction(function)\n"
        "    new = inspect.iscoroutinefunction(function)\n"
        "    trace = pm.sample(draws=1, return_inferencedata=False)\n"
        "    return old, new, trace\n"
    )
    assert deprecated_calls(ast.parse(source)) == [
        "3: from asyncio import iscoroutinefunction",
        "5: asyncio.iscoroutinefunction()",
        "7: return_inferencedata=",
    ]


def test_the_tree_calls_nothing_deprecated():
    assert deprecated_calls_in_the_tree() == []
