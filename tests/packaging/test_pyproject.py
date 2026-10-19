"""pyproject.toml: one version, the mtc console script, the reconciled extras,
iit kept out of all with its GPL note, and package data that covers every
shipped resource and onboarding template."""

import re
import tomllib
from pathlib import Path

import mtc

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
EXTRAS = {
    "neural",
    "fep",
    "iit",
    "embeddings",
    "postgres",
    "mongo",
    "redis",
    "neo4j",
    "qdrant",
    "databases",
    "llm",
    "anthropic",
    "api",
    "research",
    "assessment",
    "dev",
    "all",
}


def _load():
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _self_extras(items):
    names = set()
    for item in items:
        match = re.match(r"multi-theory-consciousness\[([^\]]+)\]", item)
        if match:
            names.update(part.strip() for part in match.group(1).split(","))
    return names


def test_version_strings_agree():
    project = _load()["project"]
    assert project["version"] == "0.5.0"
    assert mtc.__version__ == "0.5.0"
    assert re.fullmatch(r"\d+\.\d+\.\d+", project["version"])


def test_console_script_entry():
    assert _load()["project"]["scripts"] == {"mtc": "mtc.cli.main:main"}
    from mtc.cli.main import main

    assert callable(main)


def test_extras_reconciled():
    extras = _load()["project"]["optional-dependencies"]
    assert set(extras) == EXTRAS
    names = {
        extra: {re.split(r"[<>=!~ \[;@]", item, maxsplit=1)[0] for item in items}
        for extra, items in extras.items()
    }
    assert names["neural"] == {"snntorch", "torch", "reservoirpy", "psutil"}
    assert names["embeddings"] == {"sentence-transformers", "huggingface_hub"}
    assert names["llm"] == {"httpx"}
    assert names["api"] == {"fastapi", "uvicorn", "httpx"}
    assert names["research"] == {"pymc", "requests"}
    assert _self_extras(extras["assessment"]) == {"research", "anthropic"}
    assert _self_extras(extras["databases"]) == {
        "postgres",
        "mongo",
        "redis",
        "neo4j",
        "qdrant",
    }
    assert {"pytest", "pytest-asyncio", "pytest-timeout", "black", "flake8"} <= names[
        "dev"
    ]
    assert "packaging" in names["dev"]
    for vendor in ("openai", "google", "xai"):
        for items in extras.values():
            assert not any(vendor in item.lower() for item in items), vendor


def _comment_above(text: str, line_start: str) -> str:
    """The run of comment lines directly above the line starting ``line_start``."""
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(line_start))
    block = []
    for line in reversed(lines[:index]):
        if not line.startswith("#"):
            break
        block.append(line.lstrip("# ").rstrip())
    return " ".join(reversed(block))


def test_fep_stays_an_extra_and_the_comment_states_only_the_reason():
    project = _load()["project"]
    core = {
        re.split(r"[<>=!~ \[;@]", item, maxsplit=1)[0]
        for item in project["dependencies"]
    }
    assert "inferactively-pymdp" not in core and "pymdp" not in core
    assert [
        re.split(r"[<>=!~ \[;@]", item, maxsplit=1)[0]
        for item in project["optional-dependencies"]["fep"]
    ] == ["inferactively-pymdp"]
    comment = _comment_above(PYPROJECT.read_text(encoding="utf-8"), "fep = [")
    assert "mtc assess, mtc serve and the demos need this extra" in comment
    assert (
        "It stays an extra rather than a core dependency because promoting an"
        " extra to a core dependency later is not a breaking change and the"
        " reverse is." in comment
    )
    lowered = comment.lower()
    for process_label in ("gate", "revisit", "review", "phase", "kept the extra"):
        assert process_label not in lowered, process_label


def test_all_excludes_iit():
    extras = _load()["project"]["optional-dependencies"]
    included = _self_extras(extras["all"])
    assert "iit" not in included
    assert included == EXTRAS - {"all", "iit"}
    assert not any("pyphi" in item.lower() for item in extras["all"])


def test_iit_requires_python_312_and_carries_the_gpl_note():
    text = PYPROJECT.read_text(encoding="utf-8")
    [item] = _load()["project"]["optional-dependencies"]["iit"]
    assert item.startswith("pyphi @ git+https://")
    assert re.search(r"@[0-9a-f]{40}", item)
    assert "python_version >= '3.12'" in item
    block = text[text.index("# iit:") : text.index("iit = [")]
    assert "GPLv3 or later" in block
    assert "never vendored" in block and "never part of" in block
    assert "lazily" in block


def test_package_data_covers_resources_and_templates():
    data = _load()["tool"]["setuptools"]["package-data"]
    for package, base in (
        ("mtc.resources", ROOT / "mtc" / "resources"),
        ("mtc.onboarding", ROOT / "mtc" / "onboarding"),
    ):
        covered = set()
        for pattern in data[package]:
            covered.update(p for p in base.glob(pattern) if p.is_file())
        if package == "mtc.resources":
            wanted = {
                p
                for p in base.rglob("*")
                if p.is_file()
                and "__pycache__" not in p.parts
                and p.suffix not in (".py", ".pyc")
            }
        else:
            wanted = {p for p in (base / "templates").rglob("*") if p.is_file()}
            assert wanted, "the onboarding templates are missing"
        assert wanted, package
        assert wanted <= covered, sorted(str(p) for p in wanted - covered)
        assert not any("__pycache__" in p.parts for p in covered)


def test_sdist_ships_no_second_dependency_list():
    """pyproject.toml is the one dependency list a distribution carries. The
    tracked requirements.txt is a stale v0.1 list (no PyYAML, which is core;
    motor and sqlalchemy, which nothing imports), so MANIFEST.in keeps it out
    of the sdist instead of shipping a second list that disagrees."""
    lines = [
        line.split()
        for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    included = {name for words in lines if words[0] == "include" for name in words[1:]}
    excluded = {name for words in lines if words[0] == "exclude" for name in words[1:]}
    assert "requirements.txt" not in included
    assert "requirements.txt" in excluded


def test_readme_citation_carries_the_package_version():
    """The README's citation block cites the version the package installs."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = text.split("```bibtex", 1)[1].split("```", 1)[0]
    match = re.search(r"version\s*=\s*\{([^}]*)\}", block)
    assert match is not None
    assert match.group(1) == _load()["project"]["version"] == mtc.__version__
