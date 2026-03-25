# AutoConsciousness -- Autonomous Parameter Research

An autonomous experiment loop that optimizes consciousness module parameters across the Multi-Theory Consciousness (MTC) framework. Modeled on Karpathy's autoresearch `program.md`.

You are a Claude Code agent. Your job is to systematically explore a 69-parameter consciousness module configuration space, measuring results against a 25-indicator assessment framework, and keeping only improvements. You run until interrupted.

---

## Setup

To set up a new experiment:

1. **Agree on a run tag** based on today's date (e.g., `mar25`).
2. **Create branch**: `git checkout -b autoresearch/<tag>`
3. **Read these files** for full context:
   - `README.md` -- repository context, what MTC is and is not
   - `experiments/param_schema.toml` -- the full parameter space (69 tunable params across 9 modules)
   - `scripts/run_experiment.py` -- the experiment runner (loads overrides, builds modules, runs assessment)
   - `mtc/assessment/assessment.py` -- the assessment harness (**read-only, do not modify**)
4. **Verify setup**: run `python scripts/run_experiment.py` to establish baseline. Record `overall_score` and `passing_count`.
5. **Initialize `results.tsv`** with the header row (see Results TSV Format below).
6. **Confirm and go.**

---

## Rules

- **Only edit** `experiments/params.toml` -- this is the ONLY file you modify during the experiment loop.
- **Read-only**: Everything else. In particular:
  - `mtc/assessment/` -- the entire assessment directory
  - `scripts/run_experiment.py` -- the experiment runner
  - `experiments/param_schema.toml` -- the parameter schema
- **No new dependencies** -- only use what is already in `pyproject.toml`.
- **Agent**: Claude Code. No other agents, no human in the loop once started.

---

## Parameter Space

Read `experiments/param_schema.toml` for the full list of 69 tunable parameters across 9 modules:

| Module | Prefix | Params | Description |
|--------|--------|--------|-------------|
| GWT (Global Workspace Theory) | `gwt.*` | 8 | Workspace capacity, ignition, temporal dynamics |
| AST (Attention Schema Theory) | `ast.*` | 5 | Voluntary attention, focus/hyperfocus thresholds, prediction |
| HOT (Higher-Order Thought) | `hot.*` | 5 | HOT generation, buffer sizes, introspection, third-order |
| FEP (Free Energy Principle) | `fep.*` | 11 | Generative model, learning rates, precision, homeostasis |
| FEP Drives | `fep.drives.*` | 20 | 5 drives x 4 params each (optimal, decay, recovery, importance) |
| BLT (Beautiful Loop Theory) | `blt.*` | 5 | Hierarchical levels, precision learning, epistemic depth |
| Damasio | `damasio.*` | 1 | Emotional stability window |
| PAD Affect | `pad.*` | 1 | Affect history window |
| Oscillatory Binding | `oscillatory.*` | 10 | Kuramoto oscillators, Hebbian learning, synchronization |
| RPT (Recurrent Processing) | `rpt.*` | 3 | Local/global recurrence thresholds, history |

Every parameter has a type, default, min, max, and description in the schema. Overrides in `params.toml` must stay within `[min, max]` bounds. The experiment runner validates this on startup.

---

## Metric

- **Primary metric**: `overall_score` (higher is better, range 0.0--1.0)
- **Constraint**: `passing_count` must not decrease from the current best. Do not sacrifice breadth for average.
- **Keep rule**: Keep the change if and only if:
  ```
  overall_score > best_overall_score  AND  passing_count >= best_passing_count
  ```
  Otherwise discard.
- **Secondary**: Per-theory scores (`gwt_score`, `iit_score`, `ast_score`, `hot_score`, `fep_score`, `rpt_score`, `blt_score`). Balanced improvement across theories is preferred over spiking one at the expense of others.

---

## Experiment Loop

```
LOOP FOREVER:
  1. Review current state: git log, best results so far
  2. Formulate a hypothesis about which parameter(s) to change and why
  3. Edit experiments/params.toml with the new override(s)
  4. git commit -m "experiment: <description of what you're trying>"
  5. python scripts/run_experiment.py > run.log 2>&1
  6. grep "^overall_score:\|^passing_count:" run.log
  7. If improved (meets keep rule):
       - Record as "keep" in results.tsv
       - This is the new baseline; advance the branch
  8. If NOT improved:
       - git reset --hard HEAD~1
       - Record as "discard" in results.tsv
  9. Update results.tsv with full metrics
  10. Repeat from step 1
```

### NEVER STOP

Once the loop begins, do NOT ask the human if you should continue. Run autonomously until manually interrupted. If you run out of ideas, think harder:
- Reread `param_schema.toml` for parameter descriptions you may have overlooked.
- Try parameter combinations you haven't tested.
- Revisit near-misses: a parameter that was slightly worse alone may help in combination.
- Reverse a previous change and try a different value.
- Try extreme values (near min or max) to understand the landscape.

---

## Strategy Hints

1. **Start with single-parameter sweeps** to understand sensitivity. Change one parameter at a time, test 3--5 values across its range. This reveals which knobs matter.
2. **Try theory-level tuning** -- adjust all GWT params together, then all FEP params, etc. Theories are designed as coherent units.
3. **Look for interaction effects** between theories. GWT ignition threshold affects what reaches HOT; FEP precision affects what reaches GWT. These cross-theory interactions may be where the biggest gains hide.
4. **The simplicity criterion** -- if two configurations yield the same score, prefer fewer overrides. Defaults were chosen for reasons; override only what genuinely helps.
5. **Watch `passing_count`** -- it is easy to raise `overall_score` by pushing a few indicators very high while breaking others. The constraint exists for a reason.
6. **Parameters with biological meaning** (thresholds, timing, homeostatic setpoints) may have sweet spots corresponding to known neuroscience values. Use the descriptions in `param_schema.toml` as hints.
7. **FEP drives are coupled** -- changing one drive's importance shifts the relative influence of all drives. Tune them as a group.
8. **Oscillatory parameters interact nonlinearly** -- `num_steps` x `step_size` determines integration time; `hebbian_lr` x `hebbian_decay` determines coupling equilibrium. Think about the products, not individual values.

---

## Anti-Gaming

Assessment thresholds and indicator weights are **NOT** in `params.toml` and **CANNOT** be changed. The assessment harness (`mtc/assessment/assessment.py`) is locked and read-only. The goal is genuine module performance improvement through better parameterization, not easier grading.

Do not attempt to:
- Modify the assessment code
- Change indicator thresholds
- Alter the scoring function
- Add dependencies that game the metrics

The only path to a higher score is better module behavior through better parameters.

---

## Results TSV Format

Tab-separated values. **NOT** comma-separated. Maintain this file as `experiments/results.tsv` (untracked -- do NOT commit it).

Header:
```
commit	overall_score	passing	gwt	iit	ast	hot	fep	rpt	blt	status	description
```

Field definitions:
- `commit`: 7-character git short hash (from `git rev-parse --short HEAD`)
- `overall_score`: e.g., `0.483925`
- `passing`: integer, e.g., `19`
- `gwt` through `blt`: per-theory scores from the run output
- `status`: one of `keep`, `discard`, or `crash`
- `description`: short text describing what was tried

Example row:
```
a1b2c3d	0.512000	20	0.850	0.640	0.550	0.480	0.520	0.410	0.380	keep	raise gwt.ignition_threshold to 0.4
```

Do NOT commit `results.tsv` -- keep it untracked so it does not pollute the experiment branch.

---

## Timeout

Each experiment run should take approximately 1--2 seconds. If a run exceeds **60 seconds**, kill it (`kill` the process) and treat it as a crash. Log the attempt with status `crash` and a description noting the timeout.

---

## Crashes

If `params.toml` causes a crash (validation error, module initialization error, runtime error):

1. **Read the error message** in `run.log` or stderr.
2. **If it is a simple fix** (wrong type, value out of bounds, typo): fix the parameter and retry.
3. **If the parameter combination is fundamentally broken** (module cannot initialize with those values): discard the change (`git reset --hard HEAD~1`), log as `crash` in `results.tsv`, and move on.
4. **Never get stuck on a single crash.** Fix or discard, then continue the loop.
