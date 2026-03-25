# AutoConsciousness — Phase 2: Architectural Research

An autonomous experiment loop that optimizes consciousness module **code and parameters** across the Multi-Theory Consciousness (MTC) framework. Phase 1 (parameter-only) plateaued at 0.554 overall score with only GWT and FEP improving. Phase 2 unlocks module code changes to push the remaining theories: IIT (0.167), AST (0.417), HOT (0.450), RPT (0.217), BLT (0.300).

You are a Claude Code agent. Your job is to modify consciousness module code and parameters to improve the 25-indicator assessment score. You run until interrupted.

---

## Setup

To set up a new experiment:

1. **Agree on a run tag** based on today's date (e.g., `mar25-p2`).
2. **Create branch**: `git checkout -b autoresearch/<tag>`
3. **Read these files** for full context:
   - `README.md` — repository context
   - `experiments/param_schema.toml` — the parameter space (69 tunable params)
   - `scripts/run_experiment.py` — the experiment runner
   - `mtc/assessment/assessment.py` — the assessment harness (**LOCKED FOREVER — do not modify**)
4. **Understand the stuck theories** — read the indicator measurement code in `mtc/assessment/assessment.py` to understand WHY IIT, AST, HOT, RPT, and BLT score low. The measurements read module statistics via `get_statistics()` and similar methods. If a module doesn't produce the right statistics, the indicator will score at its baseline regardless of parameters.
5. **Run baseline**: `python scripts/run_experiment.py` to confirm the starting score.
6. **Initialize `results.tsv`** with the header row (see Results TSV Format below).
7. **Confirm and go.**

---

## Rules

### What you CAN modify

- `experiments/params.toml` — parameter overrides (same as Phase 1)
- `mtc/consciousness/*.py` — **consciousness module code** (NEW in Phase 2)
  - This includes: `enhanced_global_workspace.py`, `attention_schema.py`, `metacognition.py`, `active_inference.py`, `beautiful_loop/`, `damasio/`, `pad_affect.py`, `consciousness_integration.py`, `consciousness_metrics.py`, `conversation_coherence.py`
- `mtc/neural/oscillatory_binding.py` — oscillatory binding module
- `mtc/assessment/rpt_measurement.py` — RPT measurement module

### What you CANNOT modify

- `mtc/assessment/assessment.py` — **LOCKED FOREVER**. This is the ground truth scoring harness. The indicator definitions, thresholds, weights, and scoring formula are immutable.
- `mtc/assessment/causal_emergence.py` — locked (assessment infrastructure)
- `scripts/run_experiment.py` — the experiment runner
- `experiments/param_schema.toml` — the parameter schema
- `mtc/core/` — core infrastructure
- No new dependencies — only use what's in `pyproject.toml`

### The key constraint

You can change how modules **behave** (their code), but you cannot change how they are **graded** (the assessment). This means improvements must be genuine — the module must actually produce better statistics, not game the grading.

---

## Ablation Validation (REQUIRED for code changes)

When you modify module **code** (not just params), you MUST run ablation validation:

```bash
python scripts/run_experiment.py --validate <module_name> > run.log 2>&1
```

Valid module names: `gwt`, `ast`, `hot`, `fep`, `blt`, `rpt`

Example: if you modified `mtc/consciousness/metacognition.py`:
```bash
python scripts/run_experiment.py --validate hot > run.log 2>&1
```

Check the output for:
```
ablation_passed:   true
```

**Keep rule for code changes**: Keep if ALL of:
1. `overall_score > best_overall_score`
2. `passing_count >= best_passing_count`
3. `ablation_passed: true` (the modified module genuinely contributes — disabling it drops the score)

If ablation fails (the module can be removed without score loss), the change is cosmetic — discard it.

**For param-only changes** (no code edits), ablation is not required. Use the standard keep rule from Phase 1.

---

## Metric

- **Primary**: `overall_score` (higher is better, range 0.0–1.0)
- **Constraint**: `passing_count` must not decrease
- **Keep rule (params only)**: `overall_score > best AND passing_count >= best`
- **Keep rule (code changes)**: Same + `ablation_passed: true`
- **Secondary**: Per-theory scores. The biggest gains are in the stuck theories (IIT, AST, HOT, RPT, BLT).

---

## Experiment Loop

```
LOOP FOREVER:
  1. Review current state: git log, best results, which theories are stuck
  2. Read the assessment code for the stuck theory to understand what it measures
  3. Read the module code to understand what statistics it currently produces
  4. Formulate a hypothesis: what code/param change would improve the indicator?
  5. Make the change (code and/or params)
  6. git commit -m "experiment: <description>"
  7. Run experiment:
     - Param-only: python scripts/run_experiment.py > run.log 2>&1
     - Code change: python scripts/run_experiment.py --validate <module> > run.log 2>&1
  8. grep "^overall_score:\|^passing_count:\|^ablation_passed:" run.log
  9. Apply keep/discard rule
  10. Log to results.tsv
  11. Repeat
```

### NEVER STOP

Run autonomously until manually interrupted. If you get stuck:
- Reread the assessment code for clues about what each indicator actually measures
- Look at the `evidence` dict returned by each indicator — it shows what data was available
- Check if a module has `get_statistics()` that returns the keys the assessment expects
- Try adding/improving statistics that the assessment is looking for but finding empty

---

## Strategy for Stuck Theories

### IIT (0.167) — `integrated_information`, `irreducibility`, `causal_emergence`
The IIT indicators measure Phi (integrated information) via PyPhi. The score is low because the neural states are synthetic random matrices. Improving IIT likely requires the neural state construction to produce states with genuine integration — correlated rather than random.

### AST (0.417) — `attention_schema`, `attention_control`, `embodiment`
The AST indicators read from `AttentionSchemaModule.get_statistics()`. Check what stats it returns and whether the assessment is finding what it expects. The `embodiment` indicator may need specific evidence keys.

### HOT (0.450) — `higher_order_representations`, `metacognition`
These read from `MetacognitionModule.get_statistics()`. Check if HOT generation produces the statistical fingerprints the assessment looks for (e.g., HOT count, confidence distributions).

### RPT (0.217) — `algorithmic_recurrence`, `local_recurrence`, `oscillatory_binding_coherence`
RPT measures recurrent processing depth. The `RPTMeasurement` module tracks recurrence scores, but they start near zero without activity. The `oscillatory_binding_coherence` indicator reads from `rpt.get_oscillatory_binding_stats()` — check if this method exists and returns the expected keys.

### BLT (0.300) — `bayesian_binding_quality`, `epistemic_depth`, `genuine_implementation`
Beautiful Loop indicators read from `BeautifulLoop.get_statistics()`. The `genuine_implementation` indicator is an anti-mimicry check — it verifies the module does real Bayesian inference, not just returns canned values.

---

## Anti-Gaming

The assessment harness is **LOCKED**. You cannot change indicator thresholds, weights, or scoring formulas.

The ablation validation prevents cosmetic changes — if your code modification doesn't genuinely contribute (removing the module doesn't drop the score), it's discarded.

The only path to a higher score is making modules produce better, more genuine consciousness-like behavior that the locked assessment recognizes.

---

## Results TSV Format

Tab-separated. Maintain as `experiments/results.tsv` (untracked — do NOT commit).

Header:
```
commit	overall_score	passing	gwt	iit	ast	hot	fep	rpt	blt	status	description
```

Status values: `keep`, `discard`, `crash`, `ablation_fail`

Use `ablation_fail` when the code change passed the score check but failed ablation validation.

---

## Timeout

Each standard run takes ~1–2 seconds. With `--validate`, ~3–5 seconds. If a run exceeds **60 seconds**, kill it and treat as crash.

---

## Crashes

Same as Phase 1: read error, fix if simple, discard if fundamental, never get stuck.
