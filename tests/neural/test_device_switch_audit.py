"""MTC_DEVICE is the only device switch: no module picks an accelerator itself.

Every ``mtc/**/*.py`` is parsed. Outside ``mtc/core/device.py`` and
``mtc/neural/mps_utils.py`` (the two modules that may talk to the accelerator
backends) the audit reports ``file:line`` for:

- ``torch.device("mps")`` and ``torch.device("cuda")`` with a literal name
  (``"cuda:0"`` included);
- ``torch.backends.mps.is_available`` and ``torch.cuda.is_available``;
- any other call into ``torch.mps`` or ``torch.cuda``
  (``torch.mps.empty_cache()``, ``torch.cuda.synchronize()``, ...).

A module asks ``mtc.core.device`` (``resolve_device``) or the device manager in
``mtc.neural.mps_utils`` instead. The root of an attribute chain is matched when
its name contains ``torch`` (so an alias such as ``torch_module`` is covered).
``ALLOWED_OTHER_HITS`` lists any hit in a file this audit has not fixed, with
its reason; a guard test fails when an entry stops hitting, so the list only
ever shrinks.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "mtc"

DEVICE_MODULES = ("mtc/core/device.py", "mtc/neural/mps_utils.py")

# Hits that remain in files outside the device modules, with their reasons.
ALLOWED_OTHER_HITS: Dict[str, str] = {}

_ACCELERATOR_NAMES = ("mps", "cuda")


def _dotted(node: ast.AST) -> Optional[List[str]]:
    """The parts of ``a.b.c`` as ``["a", "b", "c"]``, or None for other shapes."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _is_torch_root(name: str) -> bool:
    return "torch" in name.lower()


def _literal_accelerator(call: ast.Call) -> Optional[str]:
    arguments = list(call.args) + [kw.value for kw in call.keywords]
    for argument in arguments:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            name = argument.value.split(":")[0]
            if name in _ACCELERATOR_NAMES:
                return argument.value
    return None


def find_device_hits(source: str) -> List[Tuple[int, str]]:
    """Return ``(line, what)`` for each device-switch bypass in a source text."""
    tree = ast.parse(source)
    hits: List[Tuple[int, str]] = []
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            parts = _dotted(node.func)
            if not parts or not _is_torch_root(parts[0]):
                continue
            tail = parts[1:]
            if tail == ["device"]:
                literal = _literal_accelerator(node)
                if literal is not None:
                    hits.append((node.lineno, f"torch.device({literal!r})"))
            elif tail in (
                ["backends", "mps", "is_available"],
                ["cuda", "is_available"],
            ):
                hits.append((node.lineno, ".".join(["torch"] + tail) + "()"))
            elif tail and tail[0] in _ACCELERATOR_NAMES:
                hits.append((node.lineno, ".".join(["torch"] + tail) + "()"))
        elif isinstance(node, ast.Attribute) and id(node) not in called:
            parts = _dotted(node)
            if (
                parts
                and _is_torch_root(parts[0])
                and parts[1:]
                in (["backends", "mps", "is_available"], ["cuda", "is_available"])
            ):
                # a reference without a call (``fn = torch.cuda.is_available``)
                hits.append((node.lineno, ".".join(["torch"] + parts[1:])))
    # one report per line and description
    return sorted(set(hits))


def _tree_hits() -> Dict[str, List[Tuple[int, str]]]:
    found: Dict[str, List[Tuple[int, str]]] = {}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative in DEVICE_MODULES or "__pycache__" in relative:
            continue
        hits = find_device_hits(path.read_text(encoding="utf-8"))
        if hits:
            found[relative] = hits
    return found


def test_device_switch_audit():
    unexpected = [
        f"{relative}:{line}: {what}"
        for relative, hits in _tree_hits().items()
        if relative not in ALLOWED_OTHER_HITS
        for line, what in hits
    ]
    assert not unexpected, (
        "modules pick a device without MTC_DEVICE; ask mtc.core.device or the "
        "device manager in mtc.neural.mps_utils instead:\n" + "\n".join(unexpected)
    )


def test_allowed_hits_each_have_a_reason_and_still_hit():
    found = _tree_hits()
    for relative, reason in ALLOWED_OTHER_HITS.items():
        assert (ROOT / relative).exists(), relative
        assert reason.strip(), f"{relative} has no reason"
        assert relative in found, f"{relative} no longer hits; remove its entry"


def test_the_device_modules_exist():
    for relative in DEVICE_MODULES:
        assert (ROOT / relative).is_file(), relative


def test_detector_flags_the_forbidden_shapes():
    cases = {
        'import torch\nx = torch.device("mps")\n': 1,
        'import torch\nx = torch.device(type="cuda")\n': 1,
        'import torch\nx = torch.device("cuda:0")\n': 1,
        "import torch\nok = torch.backends.mps.is_available()\n": 1,
        "import torch\nok = torch.cuda.is_available()\n": 1,
        "import torch\ntorch.mps.empty_cache()\n": 1,
        "import torch\ntorch.mps.synchronize()\n": 1,
        "import torch\ntorch.cuda.synchronize()\n": 1,
        "import torch\ntorch.cuda.empty_cache()\n": 1,
        "t = _import_torch()\nok = t.cuda.is_available()\n": 0,
        "import torch as torch_module\nok = torch_module.cuda.is_available()\n": 1,
        "import torch\nfn = torch.cuda.is_available\n": 1,
    }
    for source, expected in cases.items():
        assert len(find_device_hits(source)) == expected, source


def test_detector_leaves_the_allowed_shapes_alone():
    clean = (
        "import torch\n"
        'x = torch.device("cpu")\n'
        "y = torch.device(name)\n"
        "z = tensor.device.type == 'mps'\n"
        "w = torch.zeros(3)\n"
        "label = 'cuda'\n"
        "from mtc.core.device import resolve_device\n"
        "d = resolve_device('cpu')\n"
    )
    assert find_device_hits(clean) == []


def test_hits_are_reported_with_file_and_line():
    source = 'import torch\n\nx = torch.device("mps")\n'
    assert find_device_hits(source) == [(3, "torch.device('mps')")]
