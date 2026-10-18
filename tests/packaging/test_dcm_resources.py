"""DCM file locations: the bundled spec and the baselines come from the shipped
copy when the cache is empty, and every write goes to DCM_CACHE_DIR."""

import json

import pytest

from mtc.assessment.dcm_benchmark import BenchmarkRunner, ModelSpecManager
from mtc.assessment.dcm_evaluator import DCMEvaluator
from mtc.core import config as config_module
from mtc.core.config import Settings
from mtc.core.resources import packaged_path


@pytest.fixture
def empty_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    settings = Settings(
        _env_file=None, PROJECT_ROOT=str(project), DCM_CACHE_DIR="cache/dcm"
    )
    monkeypatch.setattr(config_module, "_settings", settings)
    monkeypatch.chdir(tmp_path)
    return project


def test_bundled_spec_resolves_from_package_when_cache_empty(empty_project):
    manager = ModelSpecManager()
    assert manager.cache_dir == empty_project / "cache" / "dcm"
    assert manager.shipped_path("bundled_spec.json") == packaged_path(
        "data/dcm/bundled_spec.json"
    )
    spec = manager.load_cached()
    assert spec.version == "bundled"
    assert spec.stances
    # Reading wrote nothing.
    assert not (empty_project / "cache").exists()


def test_baselines_from_package(empty_project):
    runner = BenchmarkRunner()
    baselines = runner.spec_manager.shipped_path("baselines")
    assert baselines == packaged_path("data/dcm/baselines")
    comparison = runner._load_comparisons(0.5, include_approximate=True)
    names = {p.stem for p in baselines.glob("*.json")}
    assert set(comparison) == names and names
    expected = json.loads((baselines / "human.json").read_text())
    assert comparison["human"] == pytest.approx(
        0.5 - expected.get("overall_probability", 0.0)
    )


def test_cache_writes_go_to_dcm_cache_dir(empty_project, tmp_path):
    manager = ModelSpecManager()
    manager._raw_spec = json.loads(
        packaged_path("data/dcm/bundled_spec.json").read_text()
    )
    raw = manager._raw_spec
    manager._raw_spec = raw["raw"] if isinstance(raw, dict) and "raw" in raw else raw
    manager._save_to_cache()
    cache = empty_project / "cache" / "dcm"
    assert (cache / "current_spec.json").is_file()
    assert manager.load_cached().version.startswith("v133_")
    evaluator = DCMEvaluator()
    assert evaluator.cache_dir == cache / "eval_cache"
    assert not evaluator.cache_dir.exists()  # created on the first write
    from datetime import datetime, timezone

    evaluator._save_cache({"Curiosity": 0.5}, datetime.now(timezone.utc), "k" * 40)
    assert list(evaluator.cache_dir.glob("eval_*.json"))
    # Nothing was written to the shipped data or the working directory.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["project"]
    assert not (empty_project / "data").exists()
