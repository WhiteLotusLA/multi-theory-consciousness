"""No host, URI or default store port literal outside mtc/core/config.py."""

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = "mtc/core/config.py"

HOST_MARKERS = ("localhost", "127.0.0.1")
URI_SCHEMES = (
    "bolt://",
    "mongodb://",
    "redis://",
    "postgres://",
    "postgresql://",
)
STORE_PORTS = (5432, 6333, 6379, 7687, 27017)

# Known hits with their reasons; each is removed when its module takes the
# value from Settings.
KNOWN_ENDPOINT_LITERALS: Tuple[Tuple[str, str], ...] = (
    (
        "mtc/neural/vectors/coordinator.py",
        "host and port defaults of the vector coordinator signature; the "
        "knowledge and vector port reads them from Settings",
    ),
    (
        "mtc/neural/gpu_integration.py",
        "host and port defaults of the GPU factory signature; the vector port "
        "reads them from Settings",
    ),
)


def _docstring_nodes(tree: ast.AST) -> Set[int]:
    ids: Set[int] = set()
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
                ids.add(id(body[0].value))
    return ids


def find_endpoint_literals(source: str) -> List[Tuple[int, str]]:
    """Return ``(line, what)`` for each endpoint literal in a source text."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            text = node.value
            if any(marker in text for marker in HOST_MARKERS + URI_SCHEMES):
                hits.append((node.lineno, f"string {text[:40]!r}"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]
            for default in defaults:
                if (
                    isinstance(default, ast.Constant)
                    and isinstance(default.value, int)
                    and not isinstance(default.value, bool)
                    and default.value in STORE_PORTS
                ):
                    hits.append((default.lineno, f"default port {default.value}"))
    return sorted(hits)


def _scan() -> Dict[str, List[Tuple[int, str]]]:
    found: Dict[str, List[Tuple[int, str]]] = {}
    for path in sorted((ROOT / "mtc").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative == CONFIG_FILE:
            continue
        hits = find_endpoint_literals(path.read_text(encoding="utf-8"))
        if hits:
            found[relative] = hits
    return found


def test_no_hardcoded_endpoints():
    known = {name for name, _ in KNOWN_ENDPOINT_LITERALS}
    unexpected = [
        f"{name}:{line}: {what}"
        for name, hits in _scan().items()
        if name not in known
        for line, what in hits
    ]
    assert unexpected == [], "\n".join(unexpected)


def test_known_literals_still_exist_and_carry_a_reason():
    found = _scan()
    for name, reason in KNOWN_ENDPOINT_LITERALS:
        assert reason.strip(), name
        assert name in found, (
            f"{name} no longer holds an endpoint literal; remove it from "
            "KNOWN_ENDPOINT_LITERALS"
        )


def test_the_scanner_finds_what_it_claims_to():
    source = '''
"""A docstring may say localhost freely."""

URL = "redis://example"


def connect(host="x", port=6379, flag=True, retries=5432.0):
    return "plain text"


def other(timeout=30):
    return "bolt://somewhere"
'''
    hits = find_endpoint_literals(source)
    assert [line for line, _ in hits] == [4, 7, 12]
    assert find_endpoint_literals('"""localhost"""\nx = 1\n') == []
    assert find_endpoint_literals("def f(port=6379): pass") == [
        (1, "default port 6379")
    ]
