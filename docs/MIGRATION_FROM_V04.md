# Migrating from v0.4.0

## Why the numbers changed

v0.5.0 resyncs the active inference, metacognition, beautiful loop, Damasio and
PAD modules with the reference implementation and fixes several defects that
v0.4.0 measured around rather than fixed. Each behaviour change that can move an
indicator score sits behind a constructor config field whose default is the new
behaviour. Scores from v0.5.0 defaults are therefore not comparable with the
published v0.4.0 numbers. They are not worse or better by construction; they
measure a different (corrected) system.

## Reproducing v0.4.0

Run the experiment runner with the legacy preset:

```bash
PYPHI_WELCOME_OFF=yes python3 scripts/run_experiment.py --preset legacy_v04
```

The preset (`experiments/presets/legacy_v04.toml`) sets every switch below to its
v0.4.0 value. It layers under `--params`, and
`tests/test_legacy_preset.py` checks that it still lands on the recorded
baseline (`tests/fixtures/baseline_v040_experiment.txt`). The assessment CLI
accepts the same two flags (`--preset`, `--params`).

The workspace builds its own inference module internally and has no config
argument, so the runners forward the `fep` behaviour switches to that instance as
well. Without that, the preset would switch only the runner's module and the
v0.4.0 numbers would not reproduce.

## Switches

| Switch | v0.4.0 | v0.5.0 default | What it fixes | What it can move |
| --- | --- | --- | --- | --- |
| `fep.hpp_hidden_dim_multiplier` | 2 | 1 | The hierarchical predictor's belief vector now lives in the same space as the error that updates it. At 2 every update skipped, so the hierarchy never learned. | The observation the workspace passes to inference, so FEP and AE indicators, and through the shared random stream the GWT and RPT indicators. The only switch that moves the runner scores. |
| `fep.meta_accuracy_method` | `median` | `ledger` | The meta-state learns from scored forecasts of the next observation instead of a running-median verdict that is about 50 percent true by construction. | Meta-inference statistics and the recursive confidence the beautiful loop reads. |
| `fep.drive_model` | `v04` | `f4` | Homeostatic drives respond to the activity names callers actually send. | Homeostatic urgency and valence (AE agency inputs). |
| `fep.drive_decay_floor` | 0.0 | 0.1 | Natural decay stops at a fraction of each drive's optimal level instead of running to zero. | Homeostatic urgency and valence. |
| `fep.legacy_double_decay` | true | false | Idle curiosity and social drives no longer decay twice per cycle. | Homeostatic urgency and valence. |
| `assessment.saturated_baseline_min_range` | 0.0 | 0.1 | An indicator whose noise baseline leaves under ten percent headroom scores 0.0 as uninformative instead of having a sliver of range rescaled to the full scale. Found through `affect_coherence`, whose coherence formula rewards variance and therefore reads about 0.97 to 1.0 on pure noise; that measurement is redesigned in the assessment resync. | `affect_coherence` and any other indicator with a near-saturated noise floor. |
| `blt.loop_quality_formula` | `v04` | `recursive` | Loop quality adds measured recursive confidence and a momentum boost. | Beautiful loop loop quality. |
| `blt.precision_floor_enabled` | false | true | Uniformly high precision across levels earns a 0.4 precision-balance floor instead of scoring near zero. | Beautiful loop precision balance. |
| `blt.require_recursive_confidence` | false | true | Field evidencing also needs the hyper-model's recursive confidence above 0.3. | Beautiful loop field evidencing. |
| `blt.field_evidencing_threshold` | 0.3 (the value the baseline run applied) | 0.3 | The constructor default was 0.4; the library default now matches the value the v0.4.0 runs used. | Beautiful loop field evidencing for callers that relied on the old default. |
| `damasio.circadian` | `v04_sawtooth` | `sinusoid` | A smooth day cycle instead of a sawtooth with a hard reset; `wall_clock` is also available. | Protoself arousal. Not read by the experiment runner's assessment. |
| `fep.world_model` | (none) | `compact_8x5x4` | New in v0.5.0. Selects the generative model; `conversational_16x10x8` is the labelled larger one. The default is 8 hidden states, 5 observation channels and 4 actions. | Nothing by itself at the default. |

Changes with no switch (history trims, bounded dictionaries, the encoder's
treatment of negative or NaN entries, and similar defects) give identical results
for v0.4.0 inputs and are not listed here. Later phases
append rows to this table.

## Measured effect

The experiment runner, seed 42, 25 warmup cycles, with `experiments/params.toml`
applied in every run (`overall_score` and `passing_count` lines):

| Configuration | overall_score | passing_count |
| --- | --- | --- |
| v0.4.0 baseline (`tests/fixtures/baseline_v040_experiment.txt`) | 0.031247 | 2 |
| `--preset legacy_v04` | 0.031247 | 2 |
| v0.5.0 defaults (no preset) | 0.048668 | 2 |

Per theory, baseline and `legacy_v04` give FEP 0.072995, GWT 0.052174, RPT
0.030645, AE 0.004150, IIT 0.005406 and zero for AST, HOT and BLT. The v0.5.0
defaults give FEP 0.171983, GWT 0.036364, RPT 0.033479, HOT 0.007544, IIT 0.005406,
and zero for AE, AST and BLT.

One-at-a-time runs (the v0.5.0 defaults with a single switch set to its v0.4.0
value) show that `fep.hpp_hidden_dim_multiplier = 2` alone reproduces the
baseline exactly. Every other switch, set alone to its v0.4.0 value, left every theory score
unchanged in this runner. A flat runner result is a statement about this
synthetic warmup only, not evidence that a switch is inert elsewhere.
