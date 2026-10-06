"""CTMConfig: defaults, the shipped file, profiles and honest errors."""

import subprocess
import sys
from pathlib import Path

import pytest

from mtc.consciousness.continuous.config import (
    BUILTIN_PROFILES,
    CTMConfig,
    CTMConfigError,
    ctm_config_dir,
)
from mtc.core.config import Settings

ROOT = Path(__file__).resolve().parents[3]
SHIPPED = ROOT / "config" / "ctm"


def _settings(tmp_path, **extra) -> Settings:
    return Settings(PROJECT_ROOT=str(tmp_path), **extra)


def test_defaults_equal_documented_values():
    c = CTMConfig()
    assert c.cadence.model_dump() == {
        "awake": 0.008,
        "drowsy": 0.005,
        "dreaming": 0.003,
        "deep_sleep": 0.002,
    }
    assert c.jitter == 0.3 and c.min_sleep_s == 1.0
    assert c.max_thoughts_per_minute == 2
    m = c.consciousness_modulation
    assert (m.enabled, m.base, m.slope, m.refresh_every) == (True, 0.4, 1.2, 5)
    assert c.waking_weights() == {
        "memory": 0.40,
        "emotion": 0.25,
        "association": 0.20,
        "question": 0.10,
        "spontaneous": 0.05,
    }
    assert c.dream_weights() == {
        "consolidation": 0.50,
        "creative": 0.30,
        "emotion": 0.15,
        "spontaneous": 0.05,
    }
    assert (c.novelty.dedup, c.novelty.express, c.novelty.cache) == (0.85, 0.5, 50)
    assert c.contention_window_s == 120 and c.burst_gap_s == 300
    assert c.buffer_size == 100
    g = c.gist
    assert (g.every, g.cooldown_s, g.min_new_fraction, g.retention) == (
        50,
        60.0,
        0.5,
        0.8,
    )
    assert (c.llm.max_tokens, c.llm.temperature) == (150, 0.8)
    assert c.llm.placeholder_prefixes == []
    assert (c.resource.cpu_percent_limit, c.resource.check_every) == (5.0, 10)
    assert (c.association.top_k, c.association.min_score) == (3, 0.5)
    assert c.surfacing.excluded_type_prefixes == ["dream"]
    assert (c.surfacing.window_multiplier, c.surfacing.min_window) == (40, 200)
    assert c.lifecycle_noise_patterns == []
    assert len(c.theme_keywords) == 8


def test_shipped_file_equals_the_defaults(tmp_path):
    loaded = CTMConfig.load(SHIPPED, settings=_settings(tmp_path))
    assert loaded.source == "file"
    assert loaded.path == SHIPPED / "ctm.yaml"
    assert loaded.model_dump() == CTMConfig().model_dump()


def test_demo_profile_overlay(tmp_path):
    cfg = _settings(tmp_path)
    base = CTMConfig.load(SHIPPED, settings=cfg)
    demo = CTMConfig.load(SHIPPED, "demo", settings=cfg)
    assert demo.profile == "demo"
    assert demo.cadence.awake >= 0.2
    assert demo.max_thoughts_per_minute <= 30
    assert demo.gist.every == 5
    assert demo.contention_window_s == 0
    # Only the overlay moved; the rest is the base.
    assert demo.cadence.drowsy == base.cadence.drowsy
    assert demo.novelty == base.novelty
    assert demo.strategy_weights == base.strategy_weights


def test_builtin_demo_overlay_matches_the_file(tmp_path):
    cfg = _settings(tmp_path)
    from_file = CTMConfig.load(SHIPPED, "demo", settings=cfg)
    from_builtin = CTMConfig.load(tmp_path / "nothing", "demo", settings=cfg)
    assert from_builtin.source == "defaults"
    assert from_builtin.model_dump() == from_file.model_dump()
    assert set(BUILTIN_PROFILES) == {"demo"}


def test_missing_file_uses_defaults_and_says_so(tmp_path, caplog):
    cfg = _settings(tmp_path)
    with caplog.at_level("INFO"):
        loaded = CTMConfig.load(tmp_path / "absent", settings=cfg)
    assert loaded.source == "defaults" and loaded.path is None
    assert loaded.model_dump() == CTMConfig().model_dump()
    assert any("built-in CTM defaults" in r.getMessage() for r in caplog.records)


def test_unknown_key_rejected_naming_file(tmp_path):
    (tmp_path / "ctm.yaml").write_text("cadence:\n  awake: 0.1\nsurprise: 3\n")
    with pytest.raises(CTMConfigError) as err:
        CTMConfig.load(tmp_path, settings=_settings(tmp_path))
    assert str(tmp_path / "ctm.yaml") in str(err.value)
    assert "surprise" in str(err.value)
    (tmp_path / "ctm.yaml").write_text("cadence:\n  awake: 0.1\n  hasty: 1\n")
    with pytest.raises(CTMConfigError, match="cadence.hasty"):
        CTMConfig.load(tmp_path, settings=_settings(tmp_path))


def test_invalid_values_and_files_raise_naming_the_file(tmp_path):
    cfg = _settings(tmp_path)
    path = tmp_path / "ctm.yaml"
    for text in (
        "novelty:\n  dedup: 1.5\n",
        "cadence:\n  awake: 0\n",
        "strategy_weights:\n  memory: 0\n  emotion: 0\n  association: 0\n"
        "  question: 0\n  spontaneous: 0\n",
        "lifecycle_noise_patterns: ['(']\n",
        "- a list\n",
        "key: [unclosed\n",
        "profiles: 3\n",
    ):
        path.write_text(text)
        with pytest.raises(CTMConfigError) as err:
            CTMConfig.load(tmp_path, settings=cfg)
        assert str(path) in str(err.value), text
    path.write_text("jitter: 0.1\n")
    with pytest.raises(CTMConfigError, match="CTM_PROFILE"):
        CTMConfig.load(tmp_path, "nonesuch", settings=cfg)


def test_a_profile_in_the_file_overrides_the_builtin(tmp_path):
    (tmp_path / "ctm.yaml").write_text(
        "profiles:\n  demo:\n    max_thoughts_per_minute: 7\n"
    )
    demo = CTMConfig.load(tmp_path, "demo", settings=_settings(tmp_path))
    assert demo.max_thoughts_per_minute == 7
    # The file's overlay replaces the built-in one, it does not stack on it.
    assert demo.gist.every == 50


def test_config_dir_from_settings_at_call_time(tmp_path, monkeypatch):
    first = tmp_path / "one"
    second = tmp_path / "two"
    for folder, value in ((first, "0.011"), (second, "0.022")):
        folder.mkdir()
        (folder / "ctm.yaml").write_text(f"cadence:\n  awake: {value}\n")
    cfg = _settings(tmp_path, CTM_CONFIG_DIR="one")
    assert ctm_config_dir(settings=cfg) == first
    assert CTMConfig.load(settings=cfg).cadence.awake == 0.011
    cfg.CTM_CONFIG_DIR = "two"
    assert CTMConfig.load(settings=cfg).cadence.awake == 0.022
    cfg.CTM_PROFILE = "demo"
    assert CTMConfig.load(settings=cfg).profile == "demo"
    # An explicit argument beats the setting.
    assert CTMConfig.load(first, "default", settings=cfg).cadence.awake == 0.011


def test_cadence_for_state():
    c = CTMConfig()
    assert c.cadence.for_state("deep_sleep") == 0.002
    with pytest.raises(ValueError):
        c.cadence.for_state("napping")


def test_yaml_imported_lazily():
    code = (
        "import sys\n"
        "import mtc.consciousness.continuous.config\n"
        "assert 'yaml' not in sys.modules, 'yaml imported at module import'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr


def test_theme_keywords_are_generic():
    for bucket, words in CTMConfig().theme_keywords.items():
        assert bucket.isalpha() and bucket.islower()
        assert words and all(w.isalpha() and w.islower() for w in words)
