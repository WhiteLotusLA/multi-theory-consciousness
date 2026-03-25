# AutoConsciousness: Autonomous Consciousness Parameter Research

**Date:** 2026-03-25
**Status:** Approved
**Author:** MTC Contributors
**Inspired by:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

---

## Overview

Adapt Karpathy's autoresearch pattern, autonomous AI agents iterating on ML training code overnight, to optimize MTC's consciousness framework parameters. A Claude Code agent modifies a single parameter file, runs the consciousness assessment, keeps improvements, discards regressions, and loops indefinitely.

MTC serves as an isolated testbed so the reference system's live deployment is never at risk. Validated improvements are ported back to the reference system via deliberate, human-reviewed changes.

## Goals

1. Discover optimal parameter configurations for the 25-indicator consciousness assessment (23 in MTC today; 2 backported in v0.2.1 pre-work)
2. Build a repeatable, logged experiment pipeline with full audit trail
3. Learn the autoresearch pattern hands-on before graduating to architectural research (Phase 2)

## Non-Goals

- Modifying the assessment harness (it's the ground truth: always locked)
- Running on GPU or requiring heavy compute (CPU-only, ~200ms per experiment)
- Automating port-back to the reference system (always human-reviewed)
- Phase 2 architectural research (designed for, not built now)

---

## Architecture

### File Boundaries

```
multi-theory-consciousness/
├── mtc/                          # LOCKED: consciousness modules
│   ├── assessment/               # LOCKED FOREVER: scoring harness
│   ├── consciousness/            # LOCKED Phase 1, mutable Phase 2
│   ├── core/
│   │   ├── config.py             # Existing
│   │   └── param_overrides.py    # NEW: override injection layer
│   └── neural/                   # LOCKED
├── scripts/
│   ├── run_assessment.py         # Existing: untouched
│   └── run_experiment.py         # NEW: experiment runner
├── experiments/
│   ├── program.md                # NEW: Claude Code agent instructions
│   ├── params.toml               # NEW: the single mutable file
│   ├── param_schema.toml         # NEW: bounds, types, descriptions
│   └── results.tsv               # NEW: experiment log (untracked)
```

### Mutable vs Locked

| Scope | Phase 1 | Phase 2 |
|-------|---------|---------|
| `experiments/params.toml` | Mutable (agent edits this) | Mutable |
| `experiments/program.md` | Read-only for agent, human-edited | Same |
| `experiments/param_schema.toml` | Read-only for agent | Same |
| `mtc/consciousness/*.py` | Locked | Mutable |
| `mtc/assessment/` | Locked | Locked (always) |
| `mtc/neural/` | Locked | Locked |
| `mtc/core/param_overrides.py` | Locked | Locked |

---

## Component Design

### 1. Parameter Override Layer (`mtc/core/param_overrides.py`)

~50 lines. Loads `experiments/params.toml`, validates against `experiments/param_schema.toml`, and returns a typed dict of overrides.

Responsibilities:
- Parse TOML file
- Validate each key exists in schema
- Validate type (int, float, bool, str)
- Validate bounds (min/max where specified)
- Raise clear errors on invalid params (agent sees the error in run.log)

Does NOT modify module code. `run_experiment.py` calls `get_overrides()` once at startup
and threads the override values into module constructors as kwargs. Modules themselves are
never modified: the override layer is a helper library, not something modules import.

`param_overrides.py` reads TOML via stdlib `tomllib` (Python 3.11+). The agent edits
`params.toml` as a text file; no TOML write library is needed.

### 2. Parameter Schema (`experiments/param_schema.toml`)

Describes every tunable parameter. Example:

```toml
[gwt.competition_duration_ms]
type = "float"
default = 250.0
min = 50.0
max = 1000.0
description = "GWT competition duration in ms (biological range: 200-300ms)"

[fep.planning_horizon]
type = "int"
default = 3
min = 1
max = 10
description = "Active inference planning horizon (steps ahead)"

[fep.gamma]
type = "float"
default = 16.0
min = 1.0
max = 100.0
description = "Precision of policy selection (softmax temperature). Higher = more deterministic"

[fep.drives.curiosity.optimal_level]
type = "float"
default = 0.5
min = 0.0
max = 1.0
description = "Target homeostatic level for curiosity drive"
```

The agent reads this file to understand the full parameter space: it serves as both documentation and guardrails.

**Namespace convention:** TOML keys map to module config dataclasses:
- `[gwt.*]` → `TemporalWorkspaceConfig` fields, passed to `EnhancedGlobalWorkspace(temporal_config=...)`
- `[ast.*]` → `AttentionSchemaModule` constructor kwargs (`voluntary_boost`, `focus_threshold`, `hyperfocus_threshold`)
- `[hot.*]` → `MetacognitionModule` constructor kwargs (`hot_generation_threshold`, `max_first_order_buffer`)
- `[fep.*]` → `ActiveInferenceConfig` fields, passed to `ActiveInferenceModule(config=...)`
- `[fep.drives.<name>.*]` → Homeostatic drive parameters within `ActiveInferenceConfig`
- `[blt.*]` → `BeautifulLoop` constructor kwargs (`num_levels`, `precision_learning_rate`, `min_binding_quality`, `max_epistemic_depth`, `field_evidencing_threshold`)

The `param_schema.toml` uses only parameter names that exist in the actual codebase. `run_experiment.py` contains the mapping logic that translates flat TOML sections into constructor kwargs and config dataclasses.

### 3. Parameter Overrides File (`experiments/params.toml`)

The single file the agent edits. Flat structure mirroring the schema keys:

```toml
# AutoConsciousness experiment parameters
# Only uncommented values override defaults

[gwt]
competition_duration_ms = 250.0

[fep]
planning_horizon = 3
gamma = 16.0
```

Defaults are inherited from module code. Only overridden values need to be present.

### 4. Experiment Runner (`scripts/run_experiment.py`)

Entry point: `python scripts/run_experiment.py > run.log 2>&1`

Uses `asyncio.run()` since the assessment and warmup functions are async.

Steps:
1. Load `param_schema.toml` (bounds + defaults)
2. Load `params.toml` (agent overrides)
3. Validate overrides against schema: crash early on invalid params
4. Set deterministic seeds (`random.seed(42)`, `numpy.random.seed(42)`, `torch.manual_seed(42)` if neural substrates are active). All module-internal RNG must also be seeded for reproducibility.
5. Instantiate all consciousness modules with overridden params (via constructor kwargs: modules are not modified)
6. Run 5 warmup cycles (fill statistics buffers)
7. Run 25-indicator consciousness assessment
8. Print standardized output block:

```
---
overall_score:     0.7300
passing_count:     23
total_indicators:  25
gwt_score:         0.8500
iit_score:         0.6200
ast_score:         0.5800
hot_score:         0.5100
fep_score:         0.7900
rpt_score:         0.5500
blt_score:         0.6800
processing_ms:     142.3
warmup_cycles:     5
```

`processing_ms` and `warmup_cycles` are informational, not logged to `results.tsv`.

Design properties:
- **Deterministic**: Same params → same score (all RNG seeded: stdlib, numpy, torch, pymdp). Verify by running identical params twice and confirming identical scores.
- **Fast**: ~1-2 seconds per experiment (5 warmup cycles + assessment)
- **Grep-friendly**: Agent extracts metrics via `grep "^overall_score:" run.log`
- **Crash-friendly**: Schema validation errors print clear messages before exit

### 5. Agent Instructions (`experiments/program.md`)

Claude Code agent instructions, modeled on Karpathy's `program.md`.

Sections:
1. **Setup**: Create branch `autoresearch/<tag>`, read codebase, run baseline, init results.tsv
2. **Rules**: Only edit `params.toml`. Everything else is read-only. No new dependencies.
3. **Parameter space**: Read `param_schema.toml` for the full list of tunable parameters with bounds and descriptions.
4. **Metric**: Overall consciousness score (higher is better). Secondary: passing_count (more is better). Tertiary: per-theory scores (balanced is better).
5. **Experiment loop**: Edit → commit → run → grep → keep/discard → log → repeat. NEVER STOP.
   - **Keep rule**: Keep if `overall_score > best_overall_score` AND `passing_count >= best_passing_count`. Otherwise discard (`git reset` to previous kept commit). Log both kept and discarded experiments in results.tsv.
6. **Strategy hints**: Start with single-param sweeps. Try theory-level tuning. Look for interactions. Simplicity wins (fewer overrides = better if score is equal).
7. **Anti-gaming**: Do not lower assessment thresholds or inflate indicator weights via params. The goal is module performance, not easier grading.
8. **Logging**: TSV format with columns: `commit  overall_score  passing  gwt  iit  ast  hot  fep  rpt  blt  status  description`

Key instruction: **NEVER STOP.** The agent runs autonomously until manually interrupted. ~30-60 experiments per hour at ~1-2s each.

---

## Pre-Work: MTC v0.2.1 Sync

Before autoresearch begins, patch MTC with changes that drifted from the reference system since v0.2.0 (March 18). Tag as v0.2.1 and merge to main before beginning autoresearch implementation.

1. **Backport 2 missing indicators**: `temporal_prediction_depth` (FEP, threshold 0.3, weight 0.9) and `agent_environment_coupling` (FEP, threshold 0.3, weight 0.9). This brings MTC from 23 to 25 indicators, matching the reference system.
2. **ConTraSt theory-level empirical weighting**: Multiplicative approach: each indicator's existing weight is multiplied by its theory's empirical weight when computing `overall_score`. Applied in the weighted average loop inside `run_full_assessment()`. Theory weights: GWT: 1.0, FEP: 0.85, IIT: 0.75, BLT: 0.70, AST: 0.65, RPT: 0.60, HOT: 0.50. Formula: `combined_weight = indicator_weight * theory_empirical_weight`.
3. **`local_recurrence` fix**: Moved measurement from GWT block to RPT block where it belongs. Uses `rpt.get_statistics()["average_local"]` instead of a flat 0.5 placeholder.
4. **`float()`/`bool()` casts**: Robustness fix in IndicatorResult construction to ensure JSON serialization works cleanly.

---

## Port-Back Pattern (MTC → the reference system)

Improvements discovered by autoresearch are ported back to the reference system deliberately:

1. Agent runs overnight, produces `results.tsv` with full experiment history
2. Human reviews the log: which experiments kept, what changed, improvement magnitude
3. Cherry-pick the winning `params.toml` values
4. Translate back to the reference system's module configs (dataclass defaults or config values)
5. Run the reference system's own assessment to validate improvement carries over
6. Commit to the reference system as a normal, reviewed change

Port-back is NEVER automated. MTC is the wind tunnel; the reference system is the ship.

---

## Phase 2 Design (Future: Not Built Now)

When parameter tuning plateaus, graduate to architectural research:

**Expanded scope:** Agent can modify `mtc/consciousness/*.py` module code in addition to `params.toml`. Assessment harness (`mtc/assessment/`) stays locked forever.

**Ablation validation:** Each "keep" decision requires an ablation sanity check:
1. Run full assessment → score S1
2. Disable the modified module → score S2
3. If S1 - S2 > 0 → module genuinely contributes → keep
4. If S1 - S2 ≈ 0 → change was cosmetic or gamed → discard

Added to `run_experiment.py` via `--validate` flag.

**Dual-metric validation:** Run both standard assessment AND noise-normalized assessment. Improvement only counts if both scores improve.

**Updated `program.md`:** Expanded rules for which files are mutable, ablation instructions, dual-metric interpretation.

Phase 1's architecture supports all of this without structural changes: graduation is a `program.md` edit plus a `--validate` flag.

---

## Resource Impact

| Resource | Impact |
|----------|--------|
| GPU | None: CPU-only computation |
| RAM | ~200-500MB Python process |
| Disk | Negligible (TOML files + TSV log) |
| The reference system | Zero impact: completely isolated repo |
| Time per experiment | ~1-2 seconds |
| Experiments per hour | ~30-60 (with agent think time) |
| Overnight yield | ~240-480 experiments |

---

## Success Criteria

Phase 1 is successful when:
- [ ] Baseline assessment runs and produces reproducible results
- [ ] Agent completes at least 50 autonomous experiments without human intervention
- [ ] At least one parameter configuration improves overall_score above baseline
- [ ] Results are logged with full audit trail in `results.tsv`
- [ ] No regressions in passing_count (don't sacrifice breadth for average)
