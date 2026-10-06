"""Shipped resources: the packaged copies equal their sources, every manifest
entry has a committed reader that names the file, drift is reported and every
source exists."""

import importlib
import inspect
import shutil
from pathlib import Path, PurePosixPath

import pytest

from mtc.core import resources
from mtc.core.resources import (
    RESOURCE_MANIFEST,
    RESOURCE_PACKAGE,
    packaged_path,
    read_shipped_text,
    resolve_shipped,
    sync_resources,
)

ROOT = Path(__file__).resolve().parents[2]


def _resolve_reader(dotted):
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        target = module
        for name in parts[cut:]:
            target = getattr(target, name)
        return module, target
    raise ImportError(dotted)


def test_packaged_copies_match_sources():
    assert sync_resources(ROOT, check=True) == []
    for entry in RESOURCE_MANIFEST:
        source = ROOT / entry.source
        copy = packaged_path(entry.packaged)
        assert copy.exists(), entry.packaged
        if source.is_dir():
            names = sorted(p.name for p in source.iterdir() if p.is_file())
            assert sorted(p.name for p in copy.iterdir() if p.is_file()) == names
            for name in names:
                assert (copy / name).read_bytes() == (source / name).read_bytes()
        else:
            assert copy.read_bytes() == source.read_bytes(), entry.source


def test_every_manifest_entry_has_a_reader():
    for entry in RESOURCE_MANIFEST:
        module, reader = _resolve_reader(entry.reader)
        assert callable(reader), entry.reader
        text = inspect.getsource(module)
        name = PurePosixPath(entry.source).name
        assert name in text, (entry.reader, name)


def test_sync_check_reports_drift(tmp_path):
    for entry in RESOURCE_MANIFEST:
        source = ROOT / entry.source
        target = tmp_path / entry.source
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)
    package_dir = tmp_path / "mtc" / "resources"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text('"""marker"""\n')
    missing = sync_resources(tmp_path, check=True)
    assert "config/dreams.yaml" in missing
    assert sync_resources(tmp_path, check=False) == missing
    assert sync_resources(tmp_path, check=True) == []
    # An edited copy, a stale file and a changed source are each reported.
    (package_dir / "config" / "dreams.yaml").write_text("edited: true\n")
    (package_dir / "config" / "stale.yaml").write_text("old: true\n")
    (tmp_path / "experiments" / "presets" / "legacy_v04.toml").write_text("# x\n")
    drift = sync_resources(tmp_path, check=True)
    assert drift == sorted(
        [
            "config/dreams.yaml",
            "config/stale.yaml",
            "experiments/presets/legacy_v04.toml",
        ]
    )
    sync_resources(tmp_path, check=False)
    assert not (package_dir / "config" / "stale.yaml").exists()
    assert (package_dir / "__init__.py").exists()
    assert sync_resources(tmp_path, check=True) == []


def test_manifest_sources_exist():
    assert RESOURCE_PACKAGE == "mtc.resources"
    seen = set()
    for entry in RESOURCE_MANIFEST:
        assert (ROOT / entry.source).exists(), entry.source
        assert entry.source not in seen
        seen.add(entry.source)
        assert not PurePosixPath(entry.source).is_absolute()
    for excluded in (
        "config/identity_statement.example.md",
        "config/rsi/canon.example.json",
        "config/validation/affect_estimator_v1.yaml",
    ):
        assert excluded not in seen, excluded
    with pytest.raises(ValueError):
        resolve_shipped("../outside.txt")
    with pytest.raises(ValueError):
        packaged_path("/absolute")


def test_resolution_reports_project_then_package(tmp_path):
    from mtc.core.config import Settings

    empty = Settings(_env_file=None, PROJECT_ROOT=str(tmp_path))
    path, where = resolve_shipped("config/dreams.yaml", settings=empty)
    assert where == "package" and path == packaged_path("config/dreams.yaml")
    inside, where = resolve_shipped("data/dcm/baselines/human.json", settings=empty)
    assert where == "package" and inside.is_file()
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "dreams.yaml").write_text("mine: true\n")
    text, where = read_shipped_text("config/dreams.yaml", settings=empty)
    assert (text, where) == ("mine: true\n", "project")
    with pytest.raises(FileNotFoundError) as info:
        resolve_shipped("config/not_shipped.yaml", settings=empty)
    assert "not a shipped resource" in str(info.value)


def test_zipped_install_is_refused(monkeypatch):
    class NotAPath:
        def joinpath(self, *parts):
            return self

    monkeypatch.setattr(
        resources.importlib.resources, "files", lambda package: NotAPath()
    )
    with pytest.raises(RuntimeError, match="zipped installs are not supported"):
        packaged_path("config/dreams.yaml")
