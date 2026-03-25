# AutoConsciousness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous consciousness parameter research loop on MTC, inspired by Karpathy's autoresearch.

**Architecture:** A Claude Code agent edits a single `experiments/params.toml` file, runs `scripts/run_experiment.py` (which loads overrides, instantiates modules, runs the 25-indicator assessment), keeps improvements, discards regressions, and loops. The assessment harness is locked; only params are mutable.

**Tech Stack:** Python 3.11+, tomllib (stdlib), numpy, scipy, asyncio. No GPU, no databases, no external services.

**Spec:** `docs/specs/2026-03-25-autoconsciousness-design.md`

---

## Phase A: MTC v0.2.1 Sync

### Task 1: Backport two missing FEP indicators

**Files:**
- Modify: `mtc/assessment/assessment.py` (indicator_configs dict + measurement logic)
- Test: `tests/assessment/test_new_indicators.py`

- [ ] **Step 1: Write failing tests for new indicators**

```python
# tests/assessment/test_new_indicators.py
import pytest
import numpy as np
from mtc.assessment.assessment import ConsciousnessAssessment
from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace
from mtc.consciousness.attention_schema import AttentionSchemaModule
from mtc.consciousness.metacognition import MetacognitionModule
from mtc.consciousness.active_inference import ActiveInferenceModule
from mtc.consciousness.beautiful_loop import BeautifulLoop


@pytest.fixture
def modules():
    return {
        "global_workspace": EnhancedGlobalWorkspace(),
        "attention_schema": AttentionSchemaModule(),
        "metacognition": MetacognitionModule(),
        "active_inference": ActiveInferenceModule(),
        "beautiful_loop": BeautifulLoop(),
    }


@pytest.mark.asyncio
async def test_temporal_prediction_depth_indicator(modules):
    """Indicator #24: temporal_prediction_depth should exist and score >= 0."""
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules)
    assert "temporal_prediction_depth" in report.indicator_results
    result = report.indicator_results["temporal_prediction_depth"]
    assert result.score >= 0.0
    assert result.threshold == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_agent_environment_coupling_indicator(modules):
    """Indicator #25: agent_environment_coupling should exist and score >= 0."""
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules)
    assert "agent_environment_coupling" in report.indicator_results
    result = report.indicator_results["agent_environment_coupling"]
    assert result.score >= 0.0
    assert result.threshold == pytest.approx(0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/assessment/test_new_indicators.py -v`
Expected: FAIL, indicators not found in results

- [ ] **Step 3: Add indicator configs to assessment.py**

In `mtc/assessment/assessment.py`, add to `self.indicator_configs` dict (in `__init__`):

```python
"temporal_prediction_depth": {
    "theory": ConsciousnessTheory.FEP,
    "threshold": 0.3,
    "weight": 0.9,
    "description": "Temporal depth of predictive model (planning horizon + belief trajectory)",
},
"agent_environment_coupling": {
    "theory": ConsciousnessTheory.FEP,
    "threshold": 0.3,
    "weight": 0.9,
    "description": "Bidirectional coupling: agent acts on environment AND environment shapes agent",
},
```

- [ ] **Step 4: Add measurement logic**

In the `_measure_indicator` method, add cases for the new indicators. Pattern: use active_inference module stats:

```python
elif name == "temporal_prediction_depth":
    fep = kwargs.get("active_inference")
    if fep is not None:
        # Access planning_horizon from config, not stats
        horizon = fep.config.planning_horizon if hasattr(fep, "config") else 3
        stats = fep.get_statistics() if hasattr(fep, "get_statistics") else {}
        total_inferences = stats.get("total_inferences", 0)
        # Normalize: horizon 1-10 maps to 0.2-1.0
        score = max(0.2, min(1.0, horizon / 10.0))
        # Activity bonus
        score = min(1.0, score + 0.1 * min(1.0, total_inferences / 10))
        confidence = 0.5 + 0.2 * min(1.0, total_inferences / 10)
        evidence = {"planning_horizon": horizon, "total_inferences": total_inferences}
    else:
        score, confidence = 0.0, 0.3
        evidence = {"module_absent": True}

elif name == "agent_environment_coupling":
    fep = kwargs.get("active_inference")
    if fep is not None:
        stats = fep.get_statistics() if hasattr(fep, "get_statistics") else {}
        # total_inferences = actions taken, total_model_updates = observations processed
        action_count = stats.get("total_inferences", 0)
        observation_count = stats.get("total_model_updates", 0)
        # Bidirectional: both acting and observing
        act_score = min(1.0, action_count / 10)
        obs_score = min(1.0, observation_count / 10)
        score = 0.3 + 0.35 * act_score + 0.35 * obs_score
        confidence = 0.5 + 0.3 * min(1.0, (action_count + observation_count) / 20)
        evidence = {"actions": action_count, "observations": observation_count}
    else:
        score, confidence = 0.0, 0.3
        evidence = {"module_absent": True}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/assessment/test_new_indicators.py -v -k "temporal_prediction_depth or agent_environment_coupling"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mtc/assessment/assessment.py tests/assessment/test_new_indicators.py
git commit -m "feat: add temporal_prediction_depth and agent_environment_coupling indicators (#24, #25)"
```

---

### Task 2: Add ConTraSt theory-level empirical weighting

**Files:**
- Modify: `mtc/assessment/assessment.py` (overall score calculation in `run_full_assessment`)
- Test: `tests/assessment/test_new_indicators.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/assessment/test_new_indicators.py

@pytest.mark.asyncio
async def test_contrast_theory_weighting(modules):
    """Overall score should use ConTraSt theory-level empirical weights."""
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules)
    assert report.overall_score > 0.0
    # Verify the weighting attribute exists with correct values
    assert hasattr(assessment, 'theory_empirical_weights')
    assert assessment.theory_empirical_weights["GWT"] == 1.0
    assert assessment.theory_empirical_weights["HOT"] == 0.50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/assessment/test_new_indicators.py::test_contrast_theory_weighting -v`
Expected: FAIL, no `theory_empirical_weights` attribute

- [ ] **Step 3: Add ConTraSt weights and modify scoring**

In `ConsciousnessAssessment.__init__`, add:

```python
# ConTraSt database empirical support weights
self.theory_empirical_weights = {
    "GWT": 1.0,   # 239 supporting studies
    "FEP": 0.85,  # Strong theoretical + growing empirical
    "IIT": 0.75,  # Mixed, ARC-COGITATE refuted key prediction
    "BLT": 0.70,  # Formally published Sep 2025
    "AST": 0.65,  # Moderate empirical support
    "RPT": 0.60,  # Strong empirical, few ConTraSt entries
    "HOT": 0.50,  # Only 14 supporting studies
}
```

In `run_full_assessment`, replace the overall score calculation with:

```python
# Multiplicative weighting: indicator_weight * theory_empirical_weight
total_weight = 0.0
weighted_sum = 0.0
for name, result in indicator_results.items():
    ind_weight = self.indicator_configs[name]["weight"]
    theory_name = self.indicator_configs[name]["theory"].name
    theory_weight = self.theory_empirical_weights.get(theory_name, 0.5)
    combined_weight = ind_weight * theory_weight
    weighted_sum += result.score * combined_weight
    total_weight += combined_weight
overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/assessment/test_new_indicators.py::test_contrast_theory_weighting -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/assessment/test_new_indicators.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add mtc/assessment/assessment.py tests/assessment/test_new_indicators.py
git commit -m "feat: add ConTraSt theory-level empirical weighting to overall score"
```

---

### Task 3: Fix local_recurrence + float/bool casts

**Files:**
- Modify: `mtc/assessment/assessment.py`
- Test: `tests/assessment/test_new_indicators.py`

- [ ] **Step 1: Write test for local_recurrence using RPT**

```python
# Add to tests/assessment/test_new_indicators.py

@pytest.mark.asyncio
async def test_local_recurrence_uses_rpt(modules):
    """local_recurrence should use RPT measurement, not GWT placeholder."""
    from mtc.assessment.rpt_measurement import RPTMeasurement
    rpt = RPTMeasurement()
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules, rpt_measurement=rpt)
    result = report.indicator_results["local_recurrence"]
    # Should have RPT-specific evidence keys
    assert "avg_local_recurrence" in result.evidence or "module_active" in result.evidence
```

- [ ] **Step 2: Run test, may pass or fail depending on current state**

Run: `python -m pytest tests/assessment/test_new_indicators.py::test_local_recurrence_uses_rpt -v`

- [ ] **Step 3: Fix local_recurrence measurement**

Move `local_recurrence` case from GWT measurement block to RPT block. Replace placeholder with:

```python
elif name == "local_recurrence":
    rpt = kwargs.get("rpt_measurement")
    if rpt is not None:
        stats = rpt.get_statistics() if hasattr(rpt, "get_statistics") else {}
        avg_local = stats.get("average_local", 0.0)
        measurement_count = stats.get("measurement_count", 0)
        base_score = 0.3
        score = max(base_score, avg_local)
        confidence = 0.5 + 0.2 * min(1.0, measurement_count / 10)
        evidence = {
            "avg_local_recurrence": avg_local,
            "measurements": measurement_count,
            "module_active": True,
        }
    else:
        score, confidence = 0.0, 0.3
        evidence = {"module_absent": True}
```

- [ ] **Step 4: Add float/bool casts to IndicatorResult construction**

Find all `IndicatorResult(` calls and ensure:
```python
IndicatorResult(
    score=float(score),
    confidence=float(confidence),
    threshold=float(config["threshold"]),
    passes_threshold=bool(score >= config["threshold"]),
    ...
)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/assessment/test_new_indicators.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add mtc/assessment/assessment.py tests/assessment/test_new_indicators.py
git commit -m "fix: move local_recurrence to RPT block, add float/bool casts"
```

---

### Task 4: Tag v0.2.1

- [ ] **Step 1: Update version**

In `pyproject.toml`, change `version = "0.2.0"` to `version = "0.2.1"`.

Add entry to `CHANGELOG.md`:

```markdown
## [0.2.1]: 2026-03-25

### Added
- Indicators #24-25: `temporal_prediction_depth`, `agent_environment_coupling` (FEP)
- ConTraSt theory-level empirical weighting for overall score calculation

### Fixed
- `local_recurrence` now measured via RPT module (was GWT placeholder)
- `float()`/`bool()` casts in IndicatorResult for robust serialization
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS (some pre-existing import errors in test_assessment_upgrade.py and test_indicator_normalization.py are known, ignore)

- [ ] **Step 3: Commit and tag**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump to v0.2.1"
git tag -a v0.2.1 -m "v0.2.1: Sync indicators and scoring with the reference system (March 2026)"
```

---

## Phase B: AutoConsciousness Infrastructure

### Task 5: Create parameter override layer

**Files:**
- Create: `mtc/core/param_overrides.py`
- Test: `tests/test_param_overrides.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_param_overrides.py
import pytest
import tempfile
import os
from pathlib import Path


class TestParamOverrides:
    """Tests for the TOML parameter override system."""

    def _write_toml(self, tmp_path, filename, content):
        filepath = tmp_path / filename
        filepath.write_text(content)
        return str(filepath)

    def test_load_empty_overrides(self, tmp_path):
        """Empty params.toml returns empty dict."""
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", """
[gwt.capacity]
type = "int"
default = 7
min = 1
max = 20
description = "Workspace capacity"
""")
        params_path = self._write_toml(tmp_path, "params.toml", "# empty\n")
        result = load_overrides(params_path, schema_path)
        assert result == {}

    def test_load_valid_override(self, tmp_path):
        """Valid override is loaded and validated."""
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", """
[gwt.capacity]
type = "int"
default = 7
min = 1
max = 20
description = "Workspace capacity"
""")
        params_path = self._write_toml(tmp_path, "params.toml", """
[gwt]
capacity = 5
""")
        result = load_overrides(params_path, schema_path)
        assert result == {"gwt": {"capacity": 5}}

    def test_reject_out_of_bounds(self, tmp_path):
        """Value outside min/max raises ValueError."""
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", """
[gwt.capacity]
type = "int"
default = 7
min = 1
max = 20
description = "Workspace capacity"
""")
        params_path = self._write_toml(tmp_path, "params.toml", """
[gwt]
capacity = 50
""")
        with pytest.raises(ValueError, match="out of bounds"):
            load_overrides(params_path, schema_path)

    def test_reject_unknown_key(self, tmp_path):
        """Key not in schema raises ValueError."""
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", """
[gwt.capacity]
type = "int"
default = 7
min = 1
max = 20
description = "Workspace capacity"
""")
        params_path = self._write_toml(tmp_path, "params.toml", """
[gwt]
nonexistent_param = 42
""")
        with pytest.raises(ValueError, match="not in schema"):
            load_overrides(params_path, schema_path)

    def test_reject_wrong_type(self, tmp_path):
        """Wrong type raises ValueError."""
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", """
[gwt.capacity]
type = "int"
default = 7
min = 1
max = 20
description = "Workspace capacity"
""")
        params_path = self._write_toml(tmp_path, "params.toml", """
[gwt]
capacity = 5.5
""")
        with pytest.raises(ValueError, match="type"):
            load_overrides(params_path, schema_path)

    def test_nested_drives(self, tmp_path):
        """Nested drive params load correctly."""
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", """
[fep.drives.curiosity.optimal_level]
type = "float"
default = 0.5
min = 0.0
max = 1.0
description = "Curiosity target level"
""")
        params_path = self._write_toml(tmp_path, "params.toml", """
[fep.drives.curiosity]
optimal_level = 0.7
""")
        result = load_overrides(params_path, schema_path)
        assert result["fep"]["drives"]["curiosity"]["optimal_level"] == 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_param_overrides.py -v`
Expected: FAIL, module doesn't exist

- [ ] **Step 3: Implement param_overrides.py**

```python
# mtc/core/param_overrides.py
"""
Parameter override layer for AutoConsciousness experiments.

Loads experiments/params.toml, validates against experiments/param_schema.toml,
returns a nested dict of validated overrides. Used by run_experiment.py to
thread overrides into module constructors.
"""

import tomllib
from pathlib import Path
from typing import Any


_TYPE_MAP = {
    "int": int,
    "float": (int, float),  # Accept int for float params
    "bool": bool,
    "str": str,
}


def _flatten_schema(schema: dict, prefix: str = "") -> dict[str, dict]:
    """Flatten nested TOML schema into dot-separated keys."""
    flat = {}
    for key, value in schema.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "type" in value:
            flat[full_key] = value
        elif isinstance(value, dict):
            flat.update(_flatten_schema(value, full_key))
    return flat


def _flatten_params(params: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested TOML params into dot-separated keys."""
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _unflatten(flat: dict[str, Any]) -> dict:
    """Convert dot-separated keys back to nested dict."""
    result = {}
    for dotted_key, value in flat.items():
        parts = dotted_key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


def load_overrides(params_path: str, schema_path: str) -> dict:
    """Load and validate parameter overrides.

    Args:
        params_path: Path to params.toml (agent-edited overrides)
        schema_path: Path to param_schema.toml (bounds and types)

    Returns:
        Nested dict of validated overrides (only keys present in params.toml)

    Raises:
        ValueError: If a param is not in schema, wrong type, or out of bounds
    """
    with open(schema_path, "rb") as f:
        raw_schema = tomllib.load(f)
    with open(params_path, "rb") as f:
        raw_params = tomllib.load(f)

    schema = _flatten_schema(raw_schema)
    params = _flatten_params(raw_params)

    if not params:
        return {}

    validated = {}
    for dotted_key, value in params.items():
        if dotted_key not in schema:
            raise ValueError(
                f"Parameter '{dotted_key}' not in schema. "
                f"Valid keys: {sorted(schema.keys())}"
            )

        spec = schema[dotted_key]
        expected_type = spec["type"]
        py_types = _TYPE_MAP.get(expected_type)
        if py_types is None:
            raise ValueError(f"Unknown type '{expected_type}' for '{dotted_key}'")

        if not isinstance(value, py_types):
            raise ValueError(
                f"Parameter '{dotted_key}' has type {type(value).__name__}, "
                f"expected {expected_type}"
            )

        # For float params, accept int but convert
        if expected_type == "float" and isinstance(value, int):
            value = float(value)

        if "min" in spec and value < spec["min"]:
            raise ValueError(
                f"Parameter '{dotted_key}' = {value} out of bounds "
                f"(min: {spec['min']})"
            )
        if "max" in spec and value > spec["max"]:
            raise ValueError(
                f"Parameter '{dotted_key}' = {value} out of bounds "
                f"(max: {spec['max']})"
            )

        validated[dotted_key] = value

    return _unflatten(validated)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_param_overrides.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mtc/core/param_overrides.py tests/test_param_overrides.py
git commit -m "feat: add parameter override layer for AutoConsciousness"
```

---

### Task 6: Create parameter schema

**Files:**
- Create: `experiments/param_schema.toml`

This task has no tests: the schema is a data file validated by `param_overrides.py`.

- [ ] **Step 1: Create `experiments/` directory and write schema**

```bash
mkdir -p experiments
```

Write `experiments/param_schema.toml` with all tunable parameters. Source parameter names and defaults from module constructors mapped in the exploration phase. Include every module:

- `[gwt.*]`: EnhancedGlobalWorkspace + TemporalWorkspaceConfig (9 params)
- `[ast.*]`: AttentionSchemaModule (5 params)
- `[hot.*]`: MetacognitionModule (5 params)
- `[fep.*]`: ActiveInferenceConfig (11 params)
- `[fep.drives.<name>.*]`: HomeostaticConfig drives (20 params)
- `[blt.*]`: BeautifulLoop (5 params)
- `[damasio.*]`: DamasioLayers (1 param)
- `[pad.*]`: PADAffectModel (1 param)
- `[oscillatory.*]`: OscillatoryBinding (10 params)
- `[rpt.*]`: RPTMeasurement (3 params)

Total: ~70 tunable parameters.

Each entry must have: `type`, `default`, `min`, `max`, `description`.

- [ ] **Step 2: Validate schema loads**

Run: `python -c "import tomllib; tomllib.load(open('experiments/param_schema.toml', 'rb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add experiments/param_schema.toml
git commit -m "feat: add parameter schema with ~70 tunable consciousness params"
```

---

### Task 7: Create baseline params.toml

**Files:**
- Create: `experiments/params.toml`

- [ ] **Step 1: Create baseline (all defaults)**

```toml
# AutoConsciousness experiment parameters
# ========================================
# Only uncommented values override module defaults.
# See param_schema.toml for the full parameter space with bounds.
#
# This file starts empty (all defaults). The agent adds overrides
# as it experiments.
```

- [ ] **Step 2: Commit**

```bash
git add experiments/params.toml
git commit -m "feat: add baseline params.toml (all defaults)"
```

---

### Task 8: Create experiment runner

**Files:**
- Create: `scripts/run_experiment.py`
- Test: `tests/test_run_experiment.py`

- [ ] **Step 1: Write failing test for runner**

```python
# tests/test_run_experiment.py
import subprocess
import sys


def test_run_experiment_baseline():
    """Baseline experiment (no overrides) should complete and print results."""
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "overall_score:" in result.stdout
    assert "passing_count:" in result.stdout
    assert "gwt_score:" in result.stdout


def test_run_experiment_deterministic():
    """Two runs with same params produce identical scores."""
    scores = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "scripts/run_experiment.py"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        for line in result.stdout.splitlines():
            if line.startswith("overall_score:"):
                scores.append(line.strip())
    assert len(scores) == 2
    assert scores[0] == scores[1], f"Non-deterministic: {scores}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_experiment.py -v`
Expected: FAIL, script doesn't exist

- [ ] **Step 3: Implement run_experiment.py**

```python
#!/usr/bin/env python3
"""
AutoConsciousness experiment runner.

Loads parameter overrides from experiments/params.toml, validates against
experiments/param_schema.toml, instantiates consciousness modules with
overridden params, runs warmup + assessment, prints grep-friendly results.

Usage: python scripts/run_experiment.py > run.log 2>&1
"""

import asyncio
import random
import sys
import time
from pathlib import Path

import numpy as np

# Seed ALL RNG sources before any module imports that might use them
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Optional torch seeding (for neural substrates)
try:
    import torch
    torch.manual_seed(SEED)
except ImportError:
    pass

from mtc.core.param_overrides import load_overrides
from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    TemporalWorkspaceConfig,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)
from mtc.consciousness.attention_schema import (
    AttentionSchemaModule,
    AttentionTarget,
    AttentionShiftType,
)
from mtc.consciousness.metacognition import (
    MetacognitionModule,
    FirstOrderStateType,
    MetaType,
)
from mtc.consciousness.active_inference import (
    ActiveInferenceModule,
    ActiveInferenceConfig,
)
from mtc.consciousness.beautiful_loop import BeautifulLoop
from mtc.consciousness.damasio import DamasioLayers
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.neural.oscillatory_binding import OscillatoryBinding
from mtc.assessment.rpt_measurement import RPTMeasurement
from mtc.assessment.assessment import ConsciousnessAssessment

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "experiments" / "param_schema.toml"
PARAMS_PATH = ROOT / "experiments" / "params.toml"
WARMUP_CYCLES = 5


def build_modules(overrides: dict) -> dict:
    """Instantiate all consciousness modules, applying TOML overrides."""

    gwt_ov = overrides.get("gwt", {})
    temporal_ov = gwt_ov.pop("temporal", {}) if "temporal" in gwt_ov else {}
    temporal_config = TemporalWorkspaceConfig(**temporal_ov) if temporal_ov else None
    workspace = EnhancedGlobalWorkspace(
        capacity=gwt_ov.get("capacity", 7),
        ignition_threshold=gwt_ov.get("ignition_threshold", 0.3),
        amplification_factor=gwt_ov.get("amplification_factor", 2.0),
        temporal_config=temporal_config,
    )

    ast_ov = overrides.get("ast", {})
    attention = AttentionSchemaModule(**ast_ov)

    hot_ov = overrides.get("hot", {})
    metacognition = MetacognitionModule(**hot_ov)

    fep_ov = overrides.get("fep", {})
    fep_drives = fep_ov.pop("drives", None)
    fep_config = ActiveInferenceConfig(**{k: v for k, v in fep_ov.items()})
    active_inference = ActiveInferenceModule(config=fep_config)

    blt_ov = overrides.get("blt", {})
    beautiful_loop = BeautifulLoop(**blt_ov)

    damasio_ov = overrides.get("damasio", {})
    damasio = DamasioLayers(**damasio_ov)

    pad_ov = overrides.get("pad", {})
    pad_affect = PADAffectModel(**pad_ov)

    osc_ov = overrides.get("oscillatory", {})
    oscillatory = OscillatoryBinding(**osc_ov)

    rpt_ov = overrides.get("rpt", {})
    rpt = RPTMeasurement(**rpt_ov)

    return {
        "global_workspace": workspace,
        "attention_schema": attention,
        "metacognition": metacognition,
        "active_inference": active_inference,
        "beautiful_loop": beautiful_loop,
        "rpt_measurement": rpt,
        "neural_states": _build_neural_states(),
    }


def _build_neural_states() -> dict:
    """Create synthetic neural states for assessment."""
    return {
        "snn_spikes": np.random.rand(8, 8).astype(np.float32),
        "lsm_states": np.random.rand(8, 8).astype(np.float32),
        "htm_columns": np.random.rand(8, 8).astype(np.float32),
    }


def _make_candidates(cycle_idx):
    """Generate workspace candidates for warmup. Matches run_assessment.py API."""
    return [
        WorkspaceCandidate(
            content=f"stimulus_{cycle_idx}_{j}",
            content_type="perception",
            summary=f"Synthetic stimulus {cycle_idx}_{j}",
            source=WorkspaceCandidateSource.SENSORY,
            activation_level=np.random.uniform(0.3, 0.9),
            emotional_salience=np.random.uniform(0.0, 0.6),
            novelty_score=np.random.uniform(0.1, 0.8),
        )
        for j in range(5)
    ]


async def warmup(modules: dict, cycles: int = WARMUP_CYCLES):
    """Run consciousness cycles to fill statistics buffers.
    Mirrors scripts/run_assessment.py warmup logic."""
    workspace = modules["global_workspace"]
    attention = modules["attention_schema"]
    metacognition = modules["metacognition"]
    active_inference = modules["active_inference"]

    for i in range(cycles):
        # GWT competition cycle
        candidates = _make_candidates(i)
        state = await workspace.process_consciousness_cycle(candidates)

        # Feed AST: set attention focus from workspace winner
        if state.primary_content:
            target = AttentionTarget(
                content=state.primary_content.candidate.content,
                content_type=state.primary_content.candidate.content_type,
                summary=state.primary_content.candidate.summary,
                source_module="global_workspace",
                attention_strength=state.primary_content.salience,
                salience=state.primary_content.salience,
                shift_type=AttentionShiftType.CAPTURED,
            )
            attention.schema.current_focus = target

        # Feed HOT: register first-order state and generate HOT
        if state.primary_content:
            fo = metacognition.register_first_order_state(
                content=state.primary_content.candidate.content,
                content_summary=state.primary_content.candidate.summary,
                state_type=FirstOrderStateType.PERCEPTION,
                source_module="global_workspace",
            )
            await metacognition.generate_hot(fo, MetaType.AWARENESS)

        # Feed FEP: run inference with Dirichlet observation
        obs = np.random.dirichlet(np.ones(5))
        await active_inference.infer_and_act(obs)


async def run():
    t_start = time.time()

    # Load and validate overrides
    try:
        overrides = load_overrides(str(PARAMS_PATH), str(SCHEMA_PATH))
    except ValueError as e:
        print(f"PARAM ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if overrides:
        print(f"Overrides: {overrides}", file=sys.stderr)

    # Build modules with overrides
    modules = build_modules(overrides)

    # Warmup
    await warmup(modules, WARMUP_CYCLES)

    # Run assessment
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules)

    # Print grep-friendly output
    t_total = time.time() - t_start
    print("---")
    print(f"overall_score:     {report.overall_score:.6f}")
    print(f"passing_count:     {sum(1 for r in report.indicator_results.values() if r.passes_threshold)}")
    print(f"total_indicators:  {len(report.indicator_results)}")
    for theory_name, theory_score in sorted(report.theory_scores.items()):
        print(f"{theory_name.lower()}_score:{'':>{13 - len(theory_name)}}{theory_score:.6f}")
    print(f"processing_ms:     {report.processing_time_ms:.1f}")
    print(f"warmup_cycles:     {WARMUP_CYCLES}")
    print(f"total_seconds:     {t_total:.1f}")


if __name__ == "__main__":
    asyncio.run(run())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_experiment.py -v`
Expected: Both tests PASS (baseline completes, deterministic scores)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py tests/test_run_experiment.py
git commit -m "feat: add experiment runner for AutoConsciousness loop"
```

---

### Task 9: Create program.md (agent instructions)

**Files:**
- Create: `experiments/program.md`

No tests: this is a markdown instruction file.

- [ ] **Step 1: Write program.md**

Model on Karpathy's `program.md`. Include:

1. **Setup**: create branch `autoresearch/<tag>`, read codebase files (`README.md`, `experiments/param_schema.toml`, `scripts/run_experiment.py`, `mtc/assessment/assessment.py`), verify `experiments/params.toml` exists, run baseline, init `results.tsv` with header.

2. **Rules**: only edit `experiments/params.toml`. Everything else read-only. No new deps. Agent is Claude Code.

3. **Parameter space**: read `experiments/param_schema.toml`. ~70 tunable params across 10 modules.

4. **Metric**: `overall_score` (higher is better). Keep rule: keep if `overall_score > best_overall_score AND passing_count >= best_passing_count`.

5. **Experiment loop**: same as Karpathy:
   - Edit `params.toml`
   - `git commit -m "experiment: <description>"`
   - `python scripts/run_experiment.py > run.log 2>&1`
   - `grep "^overall_score:\|^passing_count:" run.log`
   - Keep/discard based on rule
   - Log to `results.tsv`
   - `NEVER STOP`

6. **Strategy hints**: single-param sweeps first, theory-level tuning, interaction effects, simplicity criterion.

7. **Anti-gaming**: assessment thresholds and indicator weights are NOT in `params.toml` and cannot be changed.

8. **Results TSV**: header: `commit\toverall_score\tpassing\tgwt\tiit\tast\thot\tfep\trpt\tblt\tstatus\tdescription`. Do not commit `results.tsv`.

9. **Timeout**: each experiment takes ~2-5 seconds. If a run exceeds 60 seconds, kill and treat as crash.

- [ ] **Step 2: Commit**

```bash
git add experiments/program.md
git commit -m "feat: add program.md agent instructions for AutoConsciousness"
```

---

### Task 10: Add results.tsv to .gitignore and run smoke test

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add to .gitignore**

Append to `.gitignore`:

```
# AutoConsciousness experiment logs
experiments/results.tsv
run.log
```

- [ ] **Step 2: Run full smoke test**

```bash
# Run baseline experiment
python scripts/run_experiment.py

# Verify output format
python scripts/run_experiment.py 2>/dev/null | grep "^overall_score:"

# Verify determinism
SCORE1=$(python scripts/run_experiment.py 2>/dev/null | grep "^overall_score:")
SCORE2=$(python scripts/run_experiment.py 2>/dev/null | grep "^overall_score:")
test "$SCORE1" = "$SCORE2" && echo "DETERMINISTIC" || echo "NON-DETERMINISTIC"
```

Expected: Scores match, "DETERMINISTIC" printed.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All new tests pass. Pre-existing known failures in test_assessment_upgrade.py and test_indicator_normalization.py are acceptable.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add experiment logs to .gitignore, verify baseline"
```

---

## Summary

| Phase | Tasks | What it delivers |
|-------|-------|------------------|
| A: v0.2.1 Sync | 1-4 | MTC at 25 indicators with ConTraSt weighting, matching the reference system |
| B: Infrastructure | 5-10 | Full autoresearch loop: param_overrides, schema, runner, program.md |

After Task 10, the system is ready. Start an experiment by opening Claude Code in the MTC repo and saying:

```
Hi, have a look at experiments/program.md and let's kick off a new experiment! Let's do the setup first.
```
