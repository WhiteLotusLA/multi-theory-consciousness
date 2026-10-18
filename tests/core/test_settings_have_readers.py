"""Every Settings field has a reader.

A field counts as read when code outside the settings module names it: an
attribute read or an exact string name in the Python sources of mtc, scripts,
examples and experiments, the bare name or its ``MTC_`` form in a shell script
(the operations scripts ask ``mtc.core.env_export`` for bare field names), or
the ``MTC_`` form in the Compose file, the Makefile, the workflows and the
config files. Inside the settings module only derivation counts: a field that
a property or a derived default reads is read when that property or derived
field is read. A validator that only checks a value is not a reader.
"""

import ast
import re
from pathlib import Path
from typing import Dict, Iterable, Set

from mtc.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]
CONFIG_MODULE = ROOT / "mtc" / "core" / "config.py"
# The generator renders every field and so reads none of them.
NOT_READERS = {CONFIG_MODULE, ROOT / "mtc" / "core" / "env_example.py"}
PYTHON_ROOTS = ("mtc", "scripts", "examples", "experiments")
SHELL_ROOTS = ("scripts", "docker")
PREFIXED_FILES = (
    "docker/docker-compose.yml",
    "Makefile",
)
PREFIXED_GLOBS = (".github/workflows/*.yml", "config/**/*")

# Fields read through a computed attribute name, with the reader named:
# mtc/memory/core_blocks.py reads the five personality seeds with
# getattr(cfg, trait.upper()) over BIG_FIVE_TRAITS.
DYNAMIC_READERS = {
    "OPENNESS",
    "CONSCIENTIOUSNESS",
    "EXTRAVERSION",
    "AGREEABLENESS",
    "NEUROTICISM",
}


def _python_files() -> Iterable[Path]:
    for top in PYTHON_ROOTS:
        for path in sorted((ROOT / top).rglob("*.py")):
            if path not in NOT_READERS:
                yield path


def _python_reads(path: Path, names: Set[str]) -> Set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in names:
            # logging.DEBUG and its kin are not Settings reads.
            base = node.value
            if isinstance(base, ast.Name) and base.id == "logging":
                continue
            if isinstance(node.ctx, ast.Load):
                found.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if text.startswith("MTC_"):
                text = text[len("MTC_") :]
            if text in names:
                found.add(text)
    return found


def _text_lines(path: Path) -> Iterable[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def _shell_reads(names: Set[str]) -> Set[str]:
    found: Set[str] = set()
    for top in SHELL_ROOTS:
        for path in sorted((ROOT / top).rglob("*.sh")):
            for line in _text_lines(path):
                for word in re.findall(r"[A-Z][A-Z0-9_]*", line):
                    bare = word[len("MTC_") :] if word.startswith("MTC_") else word
                    if bare in names:
                        found.add(bare)
    return found


def _prefixed_reads(names: Set[str]) -> Set[str]:
    paths = [ROOT / rel for rel in PREFIXED_FILES]
    for pattern in PREFIXED_GLOBS:
        paths.extend(p for p in sorted(ROOT.glob(pattern)) if p.is_file())
    found: Set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in _text_lines(path):
            for word in re.findall(r"\bMTC_([A-Z0-9_]+)\b", line):
                if word in names:
                    found.add(word)
    return found


def _self_reads(node: ast.AST) -> Set[str]:
    return {
        sub.attr
        for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute)
        and isinstance(sub.value, ast.Name)
        and sub.value.id == "self"
        and isinstance(sub.ctx, ast.Load)
    }


def _derivations() -> Dict[str, Set[str]]:
    """Map each property and derived field of Settings to the fields it reads."""
    tree = ast.parse(CONFIG_MODULE.read_text(encoding="utf-8"))
    edges: Dict[str, Set[str]] = {}
    for cls in tree.body:
        if not (isinstance(cls, ast.ClassDef) and cls.name == "Settings"):
            continue
        for item in cls.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            decorators = {
                getattr(d, "id", getattr(d, "attr", "")) for d in item.decorator_list
            }
            if "property" in decorators:
                edges.setdefault(item.name, set()).update(_self_reads(item))
                continue
            for sub in ast.walk(item):
                if not isinstance(sub, ast.Assign):
                    continue
                for target in sub.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        edges.setdefault(target.attr, set()).update(
                            _self_reads(sub.value)
                        )
    return edges


def _readerless() -> Set[str]:
    fields = set(Settings.model_fields)
    edges = _derivations()
    names = fields | set(edges)
    read: Set[str] = set(DYNAMIC_READERS)
    for path in _python_files():
        read |= _python_reads(path, names)
    read |= _shell_reads(names)
    read |= _prefixed_reads(names)
    changed = True
    while changed:
        changed = False
        for derived, inputs in edges.items():
            if derived in read and not inputs <= read:
                read |= inputs
                changed = True
    return fields - read


def test_every_settings_field_has_a_reader():
    missing = sorted(_readerless())
    assert missing == [], f"Settings fields without a reader: {missing}"


def test_scan_sees_attribute_string_and_derived_reads(tmp_path):
    names = {"ALPHA_KEY", "BETA_KEY", "GAMMA_KEY", "DEBUG"}
    source = tmp_path / "reader.py"
    source.write_text(
        "import logging\n"
        "level = logging.DEBUG\n"
        "value = settings.ALPHA_KEY\n"
        "other = getattr(settings, 'BETA_KEY')\n"
        "doc = 'GAMMA_KEY is named inside a longer sentence'\n",
        encoding="utf-8",
    )
    assert _python_reads(source, names) == {"ALPHA_KEY", "BETA_KEY"}
    edges = _derivations()
    # POSTGRES_URL reads the password; LLM_BASE_URL derives from LLM_URL.
    assert "POSTGRES_PASSWORD" in edges["POSTGRES_URL"]
    assert "LLM_URL" in edges["LLM_BASE_URL"]
    assert {"LLM_HOST", "LLM_PORT"} <= edges["LLM_URL"]
