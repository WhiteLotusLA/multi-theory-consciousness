"""Tree-wide audit: no module grants, scales or seeds trust by speaker name.

Every speaker, whatever it is called, gets the configured default trust unless
an injected lookup says otherwise. This test parses every module under ``mtc``
and fails on any comparison of a speaker-like variable with a literal name, and
on any read of the creator name outside the configuration module and the
explicit allowlist below. It runs at every gate of the port.
"""

import ast
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "mtc"

SPEAKER_NAMES = frozenset(
    {
        "speaker",
        "speaker_name",
        "sender",
        "sender_name",
        "user_name",
        "interlocutor",
    }
)
UNWRAP_METHODS = frozenset({"lower", "strip", "casefold", "upper"})
ROLE_LABELS = frozenset({"user", "assistant", "system", "unknown", ""})

# Modules allowed to read the creator name, each as (path, reason). A reader
# renders the name for display only; none grants trust from it.
ALLOWED_CREATOR_NAME_READERS: Tuple[Tuple[str, str], ...] = (
    ("mtc/memory/core_blocks.py", "renders the human block; never trust"),
    (
        "mtc/knowledge/graph_manager.py",
        "seeds the creator Person node through MERGE; never trust",
    ),
    (
        "mtc/brain/prompt.py",
        "renders the persona and the authenticated creator's speaker block; "
        "never trust",
    ),
    (
        "mtc/onboarding/wizard.py",
        "writes the configured creator name; never trust",
    ),
    (
        "mtc/cli/main.py",
        "default speaker name for the local chat client; trust still "
        "requires the token",
    ),
    (
        "mtc/cli/doctor.py",
        "compares the configured creator name with the persona file's and "
        "reports only whether they differ; never printed, never trust",
    ),
)
CONFIG_MODULE = PACKAGE / "core" / "config.py"


def _unwrap(node: ast.AST) -> ast.AST:
    """Strip trailing ``.lower()``, ``.strip()``, ``.casefold()``, ``.upper()``."""
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in UNWRAP_METHODS
    ):
        node = node.func.value
    return node


def _is_speaker_like(node: ast.AST) -> bool:
    node = _unwrap(node)
    if isinstance(node, ast.Name):
        return node.id in SPEAKER_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in SPEAKER_NAMES
    return False


def _string_literals(node: ast.AST) -> Optional[List[str]]:
    """Return the strings of a literal str or a literal collection of strings."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)) and node.elts:
        values: List[str] = []
        for element in node.elts:
            if not (
                isinstance(element, ast.Constant) and isinstance(element.value, str)
            ):
                return None
            values.append(element.value)
        return values
    return None


def _is_name_literal(node: ast.AST) -> bool:
    values = _string_literals(node)
    if values is None:
        return False
    return not all(value in ROLE_LABELS for value in values)


def find_name_comparisons(source: str, label: str = "<source>") -> List[str]:
    """Report ``label:line`` for each comparison of a speaker with a name literal."""
    tree = ast.parse(source)
    hits: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        sides = [node.left, *node.comparators]
        for left, right in zip(sides, sides[1:]):
            if (_is_speaker_like(left) and _is_name_literal(right)) or (
                _is_speaker_like(right) and _is_name_literal(left)
            ):
                hits.append(f"{label}:{node.lineno}")
                break
    return hits


def find_creator_name_reads(source: str, label: str = "<source>") -> List[str]:
    tree = ast.parse(source)
    return [
        f"{label}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "CREATOR_NAME"
    ]


def _module_files() -> Iterable[Path]:
    return sorted(
        path for path in PACKAGE.rglob("*.py") if "__pycache__" not in path.parts
    )


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_no_speaker_name_literals_in_modules():
    hits: List[str] = []
    for path in _module_files():
        hits.extend(
            find_name_comparisons(path.read_text(encoding="utf-8"), _relative(path))
        )
    assert not hits, "speaker compared with a name literal: " + ", ".join(hits)


def test_allowed_creator_name_readers_exist_and_give_reasons():
    for entry, reason in ALLOWED_CREATOR_NAME_READERS:
        assert (ROOT / entry).is_file(), entry
        assert reason.strip(), entry


def test_creator_name_read_only_by_allowed_modules():
    allowed = {ROOT / entry for entry, _reason in ALLOWED_CREATOR_NAME_READERS}
    hits: List[str] = []
    for path in _module_files():
        if path == CONFIG_MODULE or path in allowed:
            continue
        hits.extend(
            find_creator_name_reads(path.read_text(encoding="utf-8"), _relative(path))
        )
    assert not hits, "CREATOR_NAME read outside the allowed modules: " + ", ".join(hits)


def test_detector_flags_name_comparisons():
    """The audit is not vacuous: it catches the shapes it exists to forbid."""
    flagged = [
        'if speaker == "Alice":\n    pass',
        'if "Alice" == speaker:\n    pass',
        'if speaker.lower() == "alice":\n    pass',
        'if speaker.strip().casefold() != "alice":\n    pass',
        'if msg.speaker in {"Alice", "Guest"}:\n    pass',
        'if sender_name in ["Alice"]:\n    pass',
        'if user_name in ("Alice", "user"):\n    pass',
        'if interlocutor.upper() == "ALICE":\n    pass',
        'if 0 < 1 and speaker == "Operator":\n    pass',
    ]
    for source in flagged:
        assert find_name_comparisons(source), source


def test_detector_allows_role_labels_and_non_literals():
    allowed = [
        'if speaker == "user":\n    pass',
        'if speaker not in ["user", "assistant"]:\n    pass',
        'if speaker in {"system", "unknown", ""}:\n    pass',
        'if speaker.lower() == "assistant":\n    pass',
        "if speaker == other:\n    pass",
        'if speaker == trusted_names["x"]:\n    pass',
        'if label == "Alice":\n    pass',
        "if speaker is None:\n    pass",
    ]
    for source in allowed:
        assert not find_name_comparisons(source), source


def test_detector_flags_creator_name_reads():
    assert find_creator_name_reads("x = settings.CREATOR_NAME")
    assert not find_creator_name_reads("x = settings.SYSTEM_NAME")
