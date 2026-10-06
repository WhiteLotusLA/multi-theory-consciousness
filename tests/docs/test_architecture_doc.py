"""The package tree in docs/ARCHITECTURE.md lists exactly the packages under mtc/.

The tree is parsed from its indentation (four characters per level, a
``|-- `` or a backquote and ``-- `` before each name), and the packages are
the directories under ``mtc`` that hold an ``__init__.py``.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "ARCHITECTURE.md"
ENTRY = re.compile(r"^((?:[| ]   )*)[|`]-- ([a-z_]+)/")


def _documented_packages(text: str):
    block = text.split("## Package tree", 1)[1].split("```", 2)[1]
    stack = []
    found = set()
    for line in block.splitlines():
        match = ENTRY.match(line)
        if not match:
            continue
        depth = len(match.group(1)) // 4
        del stack[depth:]
        stack.append(match.group(2))
        found.add("/".join(stack))
    return found


def _actual_packages():
    package = ROOT / "mtc"
    return {
        str(init.parent.relative_to(package))
        for init in package.rglob("__init__.py")
        if "__pycache__" not in init.parts and init.parent != package
    }


def test_every_subpackage_in_the_tree():
    documented = _documented_packages(DOC.read_text(encoding="utf-8"))
    actual = _actual_packages()
    assert actual, "no package found under mtc"
    assert documented == actual, (
        f"missing from the tree: {sorted(actual - documented)}; "
        f"not a package: {sorted(documented - actual)}"
    )
