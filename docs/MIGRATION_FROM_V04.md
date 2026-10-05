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

The workspace builds its own inference, beautiful loop, damasio and metacognition
modules. The runners hand the matching overrides to its constructor
(`active_inference_config` for the `fep` behaviour switches, `beautiful_loop_kwargs`,
`damasio_kwargs`, `metacognition_kwargs`), so one preset switches every instance
in the run. Without that, the preset would switch only the runner's own modules
and the v0.4.0 numbers would not reproduce. The `ast.*` keys reach the runner's
attention module; the workspace's own attention module has no constructor
argument yet and runs on the module defaults.

## Switches

| Switch | v0.4.0 | v0.5.0 default | What it fixes | What it can move |
| --- | --- | --- | --- | --- |
| `fep.hpp_hidden_dim_multiplier` | 2 | 1 | The hierarchical predictor's belief vector now lives in the same space as the error that updates it. At 2 every update skipped, so the hierarchy never learned. | The observation the workspace passes to inference, so FEP and AE indicators, and through the shared random stream the GWT and RPT indicators. Of the Phase 1 switches, the only one that moves the runner scores. |
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
| `fep.hierarchy_observation` | `hpp_top_level` | `content` | The discrete agent observes the content-derived input instead of the encoded top-level hierarchy beliefs, which are signed and gave it a constant or invalid input; the hierarchy keeps learning in shadow. | Workspace inference statistics, and through the shared random stream every runner theory score. Without this key in the preset the legacy run drifted from the baseline (FEP 0.167 instead of 0.073, GWT 0.0 instead of 0.052). |
| `fep.planning_horizon` (schema default) | 3 | 5 | The `fep.*` schema defaults describe the compact world model (8 hidden states, 5 observations, 4 actions, horizon 5); the schema's 3 matched neither the module nor the workspace's own model. | Nothing at run time: the schema default is descriptive, and the module takes its horizon from the world model unless the key is set. The legacy preset sets its own value. |
| `gwt.competition_mode` | `flat` | `hierarchical` | Per-level capacity with promotion of strong lower-level candidates, instead of one flat pool. | Which candidates win and ignite, so GWT indicators and the shared random stream. |
| `gwt.thalamic_gate_enabled` | false | true | Candidates pass an arousal-, habituation- and goal-aware gate before competition. | Which candidates reach competition. |
| `gwt.theory_interactions_enabled` | false | true | The somatic and Phi signals set the next cycle's salience bias. | Composite salience from the second cycle on. |
| `gwt.legacy_coverage_ratio` | true | false | A cycle with no broadcast reports a coverage ratio of 0.0 instead of 1.0. | Any reader of broadcast coverage before the first broadcast. |
| `gwt.body_metrics_enabled` | false | true | Real system metrics and the previous cycle's duration reach the protoself once per cycle. | Protoself readings. It reads machine metrics and the previous cycle's wall-clock duration, so it is the one default whose effect depends on the machine; the experiment runner pins it off unless `gwt.body_metrics_enabled` is set, so seeded runs stay reproducible. |
| `gwt.precision_feedback_enabled` | false | true | Workspace winners raise the inference precision bias of their channels. | Inference statistics and the shared random stream. |
| `gwt.self_prediction_enabled` | false | true | The self-model predicts each cycle and scores that prediction afterwards, so its accuracy history fills. | Readings of self-model accuracy (higher-order and beautiful loop inputs). |
| `ast.provenance_mode` | `legacy_strength` | `typed` | A shift counts as chosen or pulled by its type, not by its strength; habitual shifts are counted as uncertain and excluded from the ratio. | The voluntary ratio and the readers of it. |
| `ast.focus_threshold` | 0.6 | 0.9 | Rescaled to the workspace's 0 to 1.5 salience scale (0.6 times 1.5). | Attention state labels. |
| `ast.hyperfocus_threshold` | 0.9 | 1.35 | Rescaled to the same scale (0.9 times 1.5). | Hyperfocus detection. |
| `ast.capture_salience_threshold` | 0.8 | 1.2 | Rescaled to the same scale (0.8 times 1.5); no keyword-driven goal shifts. In v0.4.0 every non-emotional, non-keyword shift was captured regardless of the threshold. | The shift type inferred for shifts the module classifies itself. |
| `runner.warmup_profile` | `v04` | `v05` | The warmup feeds the attention schema alternating voluntary and captured shifts, with goal-driven ones, instead of captured shifts only, so typed provenance has chosen shifts to count. | Voluntary ratio in the runner. A runner setting; it changes no module default. |
| `phi.scale` | (none) | 1.0 | New. The PhiTracker's integration proxy is reported in raw bits; this multiplies it. The runner's own lightweight tracker is gone: the real tracker is attached to the workspace. | Nothing at 1.0. |
| `phi.warmup_reference` | (none) | 5.0 | New. Denominator of the normalized readout until enough values exist for a percentile; a display calibration, never part of the measurement. | The normalized Phi readout during warm-up only. |

A bare `EnhancedGlobalWorkspace()` now defaults to `ignition_threshold` 0.3
(it was 0.1, an artifact; every caller already passed 0.3), so that default needs
no switch.

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
| v0.5.0 defaults after Phase 1 (no preset) | 0.048668 | 2 |
| v0.5.0 defaults after Phase 2 (no preset) | 0.000969 | 1 |

Per theory, baseline and `legacy_v04` give FEP 0.072995, GWT 0.052174, RPT
0.030645, AE 0.004150, IIT 0.005406 and zero for AST, HOT and BLT. The v0.5.0
defaults give FEP 0.171983, GWT 0.036364, RPT 0.033479, HOT 0.007544, IIT 0.005406,
and zero for AE, AST and BLT.

After Phase 2 the legacy preset still lands on the baseline at every theory score
(FEP 0.072995, GWT 0.052174, RPT 0.030645, AE 0.004150, IIT 0.005406, zero for
AST, HOT and BLT), and the v0.5.0 defaults give IIT 0.005406 and RPT 0.003695 with
every other theory at zero. The scores are normalized against a noise baseline
from the same synthetic warmup, so a low score means the runner's activity is not
separable from that noise here; it is a statement about this warmup, not a
verdict on the system.

One-at-a-time runs of the Phase 1 switches (the v0.5.0 defaults with a single
switch set to its v0.4.0 value) show that `fep.hpp_hidden_dim_multiplier = 2` alone reproduces the
baseline exactly. Every other switch, set alone to its v0.4.0 value, left every theory score
unchanged in this runner. A flat runner result is a statement about this
synthetic warmup only, not evidence that a switch is inert elsewhere.
