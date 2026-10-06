"""
Shipped Resources
=================

The runtime defaults the framework ships (the CTM configuration and seed pack,
the persona and speakers examples, the model-transition probes, the dream
configuration, the PyPhi presentation config, the DCM bundled spec and
baselines, and the experiment schema and presets) keep their source paths in
the repository, and ``scripts/sync_resources.py`` copies each into
``mtc/resources/`` so a non-editable install can read them from any working
directory. ``RESOURCE_MANIFEST`` lists every copied file or directory with the
function that reads it; nothing without a committed reader is listed.

``resolve_shipped`` returns the project copy (the source path joined to
``PROJECT_ROOT``) when it exists, else the packaged copy, and reports which
one it used (``project`` or ``package``). Loaders call it for their shipped
step only, so their order is: the configured file, the project copy, the
packaged copy, then any built-in default.

The packaged copies are generated and never edited by hand;
``sync_resources(root, check=True)`` reports drift, and a test fails on it.
Zipped installs are not supported: ``packaged_path`` raises when the package
does not live on a real file system.

Config key read: ``PROJECT_ROOT`` (through ``mtc.core.paths``). Nothing here
writes outside ``mtc/resources/``, and the resources hold no private data.
"""

from __future__ import annotations

import filecmp
import importlib.resources
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

from mtc.core.config import Settings
from mtc.core.paths import resolve_path

RESOURCE_PACKAGE = "mtc.resources"
RESOURCE_DIR = Path("mtc") / "resources"

# Files in the packaged tree that are not copies: the package marker.
_PACKAGE_OWN_FILES = frozenset({"__init__.py"})
_SKIPPED_PARTS = frozenset({"__pycache__", ".DS_Store"})


@dataclass(frozen=True)
class ResourceEntry:
    """One shipped resource.

    ``source`` is repository-relative (a file or a directory), ``packaged`` is
    relative to ``mtc/resources/``, and ``reader`` is the dotted path of the
    function that loads it.
    """

    source: str
    packaged: str
    reader: str


RESOURCE_MANIFEST: Tuple[ResourceEntry, ...] = (
    ResourceEntry(
        "config/ctm/ctm.yaml",
        "config/ctm/ctm.yaml",
        "mtc.consciousness.continuous.config.CTMConfig.load",
    ),
    ResourceEntry(
        "config/ctm/seeds/default.yaml",
        "config/ctm/seeds/default.yaml",
        "mtc.consciousness.continuous.seeds.SeedLibrary.from_config_dir",
    ),
    ResourceEntry(
        "config/ctm/persona.example.yaml",
        "config/ctm/persona.example.yaml",
        "mtc.consciousness.continuous.persona.CTMPersona.load",
    ),
    ResourceEntry(
        "config/persona.example.yaml",
        "config/persona.example.yaml",
        "mtc.brain.prompt.PromptAssembler._resolve_persona",
    ),
    ResourceEntry(
        "config/speakers.example.yaml",
        "config/speakers.example.yaml",
        "mtc.onboarding.wizard.plan_writes",
    ),
    ResourceEntry(
        "config/transition_prompts.json",
        "config/transition_prompts.json",
        "mtc.brain.model_transition.ModelTransitionProtocol.load_prompts",
    ),
    ResourceEntry(
        "config/dreams.yaml",
        "config/dreams.yaml",
        "mtc.consciousness.dream_engine.load_dream_config",
    ),
    ResourceEntry(
        "config/pyphi_config.yml",
        "config/pyphi_config.yml",
        "mtc.consciousness.pyphi_bridge.prepare_pyphi_home",
    ),
    ResourceEntry(
        "data/dcm/bundled_spec.json",
        "data/dcm/bundled_spec.json",
        "mtc.assessment.dcm_benchmark.ModelSpecManager.load_cached",
    ),
    ResourceEntry(
        "data/dcm/baselines",
        "data/dcm/baselines",
        "mtc.assessment.dcm_benchmark.BenchmarkRunner._load_comparisons",
    ),
    ResourceEntry(
        "experiments/param_schema.toml",
        "experiments/param_schema.toml",
        "mtc.pipeline.runtime.load_pipeline_overrides",
    ),
    ResourceEntry(
        "experiments/presets",
        "experiments/presets",
        "mtc.pipeline.runtime.load_pipeline_overrides",
    ),
)


def _relative_parts(value: str, what: str) -> Tuple[str, ...]:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"{what} {value!r} must be a relative path inside the tree")
    return path.parts


def packaged_path(packaged: str) -> Path:
    """Return the file-system path of a packaged copy (it may not exist).

    Zipped installs are not supported: the package must be a directory on a
    real file system, and anything else raises RuntimeError naming the reason.
    """
    parts = _relative_parts(packaged, "packaged path")
    base = importlib.resources.files(RESOURCE_PACKAGE)
    if not isinstance(base, Path):
        raise RuntimeError(
            f"{RESOURCE_PACKAGE} is not on a real file system (a zipped install); "
            "zipped installs are not supported"
        )
    return base.joinpath(*parts)


def _entry_for(source: str) -> Optional[Tuple[ResourceEntry, Tuple[str, ...]]]:
    parts = PurePosixPath(source).parts
    for entry in RESOURCE_MANIFEST:
        entry_parts = PurePosixPath(entry.source).parts
        if parts[: len(entry_parts)] == entry_parts:
            return entry, parts[len(entry_parts) :]
    return None


def resolve_shipped(
    source: str, *, settings: Optional[Settings] = None
) -> Tuple[Path, str]:
    """Return ``(path, where)`` for a shipped resource.

    ``where`` is ``project`` when the source path joined to ``PROJECT_ROOT``
    exists, else ``package`` when the packaged copy exists. ``source`` is a
    repository-relative path to a manifest entry or a file inside a manifest
    directory. A resource found in neither place raises FileNotFoundError
    naming both places.
    """
    _relative_parts(source, "resource source")
    project = resolve_path(source, settings=settings)
    if project.exists():
        return project, "project"
    found = _entry_for(source)
    if found is None:
        raise FileNotFoundError(
            f"{source} not found at {project}, and it is not a shipped resource "
            "(see mtc.core.resources.RESOURCE_MANIFEST)"
        )
    entry, rest = found
    packaged = packaged_path(entry.packaged)
    if rest:
        packaged = packaged.joinpath(*rest)
    if packaged.exists():
        return packaged, "package"
    raise FileNotFoundError(
        f"{source} not found at {project} or at the packaged copy {packaged}; "
        "run scripts/sync_resources.py in a checkout, or reinstall the package"
    )


def read_shipped_text(
    source: str, *, settings: Optional[Settings] = None
) -> Tuple[str, str]:
    """Return ``(text, where)`` for a shipped file (see ``resolve_shipped``)."""
    path, where = resolve_shipped(source, settings=settings)
    return path.read_text(encoding="utf-8"), where


def _files_under(directory: Path) -> List[str]:
    found = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part in _SKIPPED_PARTS for part in relative.parts):
            continue
        if path.is_file():
            found.append(relative.as_posix())
    return found


def _expected_copies(root: Path) -> List[Tuple[Path, Path, str]]:
    """``(source file, packaged file, packaged relative path)`` for every copy."""
    copies = []
    for entry in RESOURCE_MANIFEST:
        source = root / entry.source
        target = root / RESOURCE_DIR / entry.packaged
        if source.is_dir():
            for relative in _files_under(source):
                copies.append(
                    (
                        source / relative,
                        target / relative,
                        f"{entry.packaged}/{relative}",
                    )
                )
        else:
            copies.append((source, target, entry.packaged))
    return copies


def sync_resources(root: Path, *, check: bool) -> List[str]:
    """Copy every manifest source into ``mtc/resources/`` under ``root``.

    Returns the packaged paths (relative to ``mtc/resources/``) that were
    missing, different or stale. With ``check`` nothing is written and the
    list is the drift; without it the copies are written and stale packaged
    files (no longer produced by the manifest) are removed, only inside
    ``mtc/resources/``. A missing source raises FileNotFoundError.
    """
    root = Path(root)
    package_dir = root / RESOURCE_DIR
    copies = _expected_copies(root)
    for source, _, _ in copies:
        if not source.is_file():
            raise FileNotFoundError(f"resource source {source} is missing")
    for entry in RESOURCE_MANIFEST:
        if not (root / entry.source).exists():
            raise FileNotFoundError(f"resource source {entry.source} is missing")
    changed: List[str] = []
    expected = set()
    for source, target, relative in copies:
        expected.add(relative)
        same = target.is_file() and filecmp.cmp(source, target, shallow=False)
        if same:
            continue
        changed.append(relative)
        if not check:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    if package_dir.is_dir():
        for relative in _files_under(package_dir):
            if relative in expected or relative in _PACKAGE_OWN_FILES:
                continue
            changed.append(relative)
            if not check:
                stale = package_dir / relative
                if stale.resolve().is_relative_to(package_dir.resolve()):
                    stale.unlink()
    return sorted(changed)
