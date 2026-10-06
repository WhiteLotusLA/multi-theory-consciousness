# Migrating from v0.4.0

## Why the numbers changed

v0.5.0 resyncs the active inference, metacognition, beautiful loop, Damasio and
PAD modules, the global workspace, the attention schema and the PhiTracker, the
30 plus 3 indicator assessment, the measurement framework, RPT, causal
emergence, the DCM benchmark and the SNN, HTM, LSM and oscillator substrates
with the reference implementation, and fixes several defects that v0.4.0
measured around rather than fixed. Each behaviour change that can move an
indicator score sits behind a constructor config field whose default is the new
behaviour. Scores from v0.5.0 defaults are therefore not comparable with the
published v0.4.0 numbers. They are not worse or better by construction; they
measure a different (corrected) system, and the assessment now says what it
could not measure (`insufficient_data`) instead of substituting a constant.

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
and the v0.4.0 numbers would not reproduce. The `ast.*` keys reach both the
runner's attention module and the workspace's own one (through
`attention_schema_kwargs`). Both runners build the workspace through
`mtc.core.runner_support.build_workspace`, so the two cannot drift apart.

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
| `assessment.indicator_formulas` | `v04` | `v05` | The indicators read the statistics their producers emit: the bottleneck's `total_competitions`, `avg_candidates` and `source_distribution` instead of a `competition` key the workspace never emits; `peak_depth` instead of `max_depth_reached`; `total_hots_generated` instead of `total_generated`; a numeric prediction-error trend from the inference history instead of a text label that never compared equal to a number; the agent's real policy count; zero floors where v0.4.0 granted presence credit. Under v05 the causal_emergence pass rule is also strict (`score > threshold`, because sigmoid(0) is exactly 0.5); the v04 rule is inclusive. | sequential_module_querying, organized_percepts, flexible_responsiveness, genuine_implementation, epistemic_depth, no_metacognitive_access, prediction_error_minimization, complex_agency, temporal_prediction_depth, recurrent_processing, online_reportable_learning, agency and embodiment. In the runner it moves FEP from 0.009010 to 0.081411 and RPT from 0.003695 to 0.005109. Through the pass rule it also acts on `passing_count` together with `assessment.causal_emergence_threshold` (see that row). |
| `assessment.missing_signal_policy` | `v04_constants` | `insufficient_data` | A module the caller did not supply scores 0.0 as absent; a supplied module that has not produced its evidence yet is reported by name and left out of the averages and the passing count, instead of a 0.1, 0.3 or 0.5 stand-in. | Any indicator whose module has no data, plus the report's `status`, `coverage` and confidence. Nothing in the experiment runner, where every module runs 25 cycles; in the seeded assessment CLI five indicators are named and coverage reads 0.633. |
| `assessment.irreducibility_method` | `spectral_entropy` | `weighted_fiedler` | The algebraic connectivity of the symmetrized weighted graph through its normalized Laplacian, instead of a spectral entropy that a 1e-9 binarization turned into a complete graph for any dense matrix. | irreducibility. In the runner IIT moves from 0.005406 to 0.026167: both methods read the random connectivity close to their own noise mean (0.857 against 0.845, and 0.754 against 0.750), so the move is seed position under a different statistic, not signal. |
| `assessment.causal_emergence_threshold` | 0.0 | 0.5 | The causal emergence indicator passes only when the macro scale carries more causal power than the micro scale (a score strictly above sigmoid(0) = 0.5 under the v05 pass rule). At 0.0 with the inclusive v04 rule every causal_emergence result passed, including an absent analyzer scored 0.0: the v0.4.0 assessment fixture passes it that way (score 0.0, threshold 0.0, `analyzer not provided`). | The pass decision of causal_emergence, so `passing_count`. Under the legacy preset it is one of the two runner passes (normalized 0.000 at threshold 0.0, inclusive; the other is affect_coherence), and in the seeded assessment CLI one of the legacy run's 13. Set alone on top of the defaults it is inert (0.021675, 1 passing) because `indicator_formulas = "v05"` makes the comparison strict and 0.0 > 0.0 is false; the threshold moves a pass only together with `indicator_formulas = "v04"`. |
| `assessment.phi_source` | `calculator` | `tracker_preferred` | integrated_information reads the exact IIT 4.0 Phi of the supplied PhiTracker's module graph when it has a result (labelled a module-graph integration proxy) and falls back to the PhiCalculator on the neural states while the tracker warms up; `calculator` never reads the tracker. | integrated_information and the `phi_source_used` line. In the runner the tracker's exact Phi is 0.0 (the five-node module graph is reducible: the partition separates AST from the rest) against a noise mean of 0.0, and the calculator's proxy on the random neural states reads 0.518 against 0.556 under the legacy preset, so both normalize to 0.0. |
| `assessment.phi_normalization` | `fixed` | `fixed` | Unchanged default. `percentile` divides by the rolling 95th percentile of the recent Phi values once 20 exist, so a steady Phi reads near 1.0. | Nothing at the default; no preset value. |
| `assessment.phi_fixed_reference` | 5.0 | 5.0 | Unchanged default: the Phi that maps to 1.0 under fixed normalization, a display calibration. | Nothing at the default; no preset value. |
| `pad.coherence_method` | `v04_variance` | `temporal_structure` | Coherence is the mean lag-1 autocorrelation of the three PAD dimensions over the last `coherence_window` states (a flat dimension scores 0, fewer than `coherence_min_history` states reports none), instead of a variance formula that pure noise maxed out, which is what the Phase 1 headroom guard was added for. | affect_coherence. On the runner's uniformly random sources the old formula reads 0.991 against a noise floor of 0.985 and the legacy preset's `saturated_baseline_min_range = 0.0` rescales that sliver to 0.438; the new method reads 0.093 against 0.042. In the runner FEP moves from 0.072401 to 0.081411. |
| `pad.coherence_window` | 20 | 20 | New tuning key of the default method: the number of recent states the autocorrelation reads. | Nothing at the default; inert under `v04_variance`, so no preset value. |
| `pad.coherence_min_history` | 10 | 10 | New tuning key of the default method: states needed before a coherence is reported. | Nothing at the default; inert under `v04_variance`, so no preset value. |
| `causal_emergence.tpm_row_order` | `little_endian` | `big_endian` | The analyzer reorders the PhiTracker's big-endian rows before coarse-graining, so the macro partition groups the right source states; `little_endian` reads the rows as given, which matched the v0.4.0 warmup tracker but not the rebuilt one. | causal_emergence. In the runner the normalized score is 0.0 either way; the raw CE reads 0.725 under the default (the same value for every seed and candidate feed, because the recorded module-state transitions do not depend on the random stream) and 0.652 under the legacy order. |
| `rpt.stdp_plasticity_enabled` | false | true | RPT adds a plasticity component (min(1, 2 x mean weight standard deviation + weight diversity)) when a caller passes the SNN's STDP statistics with `plasticity_active`; phenomenal recurrence stays equal to local, so plasticity is counted once. | local_recurrence and algorithmic_recurrence once an SNN's STDP statistics are supplied. Inert in both runners, which pass synthetic spike counts without them. |

| `cycle.drive_candidates` | false | true | Urgent homeostatic drives become HOMEOSTATIC workspace candidates each cycle. | Workspace competition in the live pipeline. Both runners pin it off unless the key is set, so no runner number moves. |
| `cycle.drive_urgency_threshold` | (no drive candidates) | 0.5 | Urgency above which a drive becomes a candidate. | Nothing while `cycle.drive_candidates` is off. |
| `cycle.prospective_candidates` | false | true | The prospective simulation generated after a cycle is injected as an INTERNAL candidate of the next. | Workspace competition in the live pipeline; pinned off in the runners unless set. |
| `cycle.crossmodal_binding` | false | true | Cross-modal binding runs over the candidates before competition. | Workspace competition in the live pipeline; pinned off in the runners unless set. |
| `cycle.gate_arousal` | `workspace` | `protoself` | The thalamic gate's arousal comes from the protoself (`pad` leaves the PAD arousal recorded on the state for the next cycle; `workspace` leaves the gate to its own derivation). | Which candidates pass the gate in the live pipeline; the runners pin `workspace` unless set. |
| `cycle.pad_from_state` | false | true | The PAD coordinate is computed from the cycle's state and recorded on it. | PAD readings in the live pipeline; pinned off in the runners unless set, and the experiment runner then keeps its random PAD feed. |
| `cycle.candidate_cap` | 0 | 20 | A pre-attentive cap keeps the candidates with the highest activation plus emotional salience (0 means no cap). | Which candidates compete when more than 20 arrive; the runners pin 0 unless set. |

A bare `EnhancedGlobalWorkspace()` now defaults to `ignition_threshold` 0.3
(it was 0.1, an artifact; every caller already passed 0.3), so that default needs
no switch.

The `cycle` rows are the `CycleAugmenterConfig` switches of
`mtc/pipeline/cycle_augmenter.py`, mirrored by the `cycle` group of
`experiments/param_schema.toml`. Their defaults describe the live pipeline;
`mtc.core.runner_support.build_cycle_augmenter` pins every one of them to its
v0.4.0 value in both runners unless the layered overrides set it, and the
`legacy_v04` preset sets all seven explicitly. With every switch off the
augmenter is an identity (the candidates come back unchanged and the state is
untouched), so the runner numbers below are unaffected.

## Fixes without a switch

Changes that give identical results for v0.4.0 inputs (history trims, bounded
dictionaries, the encoder's treatment of negative or NaN entries, and similar
defects) are not listed. The fixes below changed behaviour but need no switch
because no published v0.4.0 number depended on the path:

- The PhiCalculator's exact path passed a keyword PyPhi rejects, so it raised and fell back to the spectral proxy on every call; it now runs through the PyPhi bridge when a TPM is supplied. Every reported v0.4.0 Phi came from the proxy, which is unchanged.
- `run_normalized_assessment` normalized an already normalized report; it now requests raw scores. No published number used that method.
- The noise stubs emit the producers' schema (bottleneck keys, `peak_depth`, numeric prediction errors). Only the stub-based baseline reads them; both runners compute their baselines from real modules.
- The legacy 14-indicator integration never awaited its indicators (every score was a coroutine, and the mean raised) and its workspace-integration metric read a NaN correlation on constant vectors as 1.0; it now awaits each indicator and reports `insufficient_data` with zero confidence when nothing was measured. It produced no value in v0.4.0, so there is no number to reproduce.
- The measurement framework matches self-reference and identity words as tokens ("this is fine" is not a self-reference), its Phi metric is documented as a spectral proxy, and `measure_consciousness` no longer imports PyPhi. No published number used it.
- DCM: the benchmark's result history is capped at 50 and `get_latest()` reloads the newest saved result; the evaluator's cache is keyed by its inputs and its model id comes from `Settings.ANTHROPIC_MODEL`; the reference adapter reads module statistics instead of constants. No runner reads any of it.
- Substrates: the SNN's membrane and trace decay are computed in seconds per step (`SNNConfig.dt`, default 1e-3; dt = 1.0 reproduces the v0.4.0 network, which was silent), the SNN reports population synchrony and STDP statistics, the HTM scores anomaly against the predictions made before the step (v0.4.0 read about 1.0 every step) and reports None before any input, the oscillator restore rejects non-finite phases and couplings, the LSM readout persists, and protocol timestamps are timezone-aware UTC. Neither runner runs a substrate, so there is no runner delta, and `SNNConfig.dt` gets no schema row because no runner reads it.
- The neural orchestrator imports the measurement framework from `mtc.assessment.framework`, reports its optional components as missing instead of silently absent, awaits the continuous-thought stop on shutdown and keeps the latest substrate signals. No runner delta.

## Measured effect

The experiment runner, seed 42, 25 warmup cycles, with `experiments/params.toml`
applied in every run (`overall_score` and `passing_count` lines):

| Configuration | overall_score | passing_count |
| --- | --- | --- |
| v0.4.0 baseline (`tests/fixtures/baseline_v040_experiment.txt`) | 0.031247 | 2 |
| `--preset legacy_v04` | 0.031247 | 2 |
| v0.5.0 defaults after Phase 1 (no preset) | 0.048668 | 2 |
| v0.5.0 defaults after Phase 2 (no preset) | 0.000969 | 1 |
| v0.5.0 defaults after Phase 3 (no preset) | 0.021675 | 1 |

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

## Phase 3 measurements

The experiment runner (seed 42, 25 warmup cycles, `experiments/params.toml`
applied in every run), every printed line:

| Line | v0.4.0 baseline | `--preset legacy_v04` | v0.5.0 defaults |
| --- | --- | --- | --- |
| overall_score | 0.031247 | 0.031247 | 0.021675 |
| passing_count | 2 | 2 | 1 |
| assessment_status | (not printed in v0.4.0) | measured | measured |
| coverage | (not printed in v0.4.0) | 1.000 | 1.000 |
| phi_source_used | (not printed in v0.4.0) | calculator_spectral_proxy | phi_tracker_iit4 |
| ae_score | 0.004150 | 0.004150 | 0.000000 |
| ast_score | 0.000000 | 0.000000 | 0.000000 |
| blt_score | 0.000000 | 0.000000 | 0.000000 |
| fep_score | 0.072995 | 0.072995 | 0.081411 |
| gwt_score | 0.052174 | 0.052174 | 0.000000 |
| hot_score | 0.000000 | 0.000000 | 0.000000 |
| iit_score | 0.005406 | 0.005406 | 0.026167 |
| rpt_score | 0.030645 | 0.030645 | 0.005109 |

The legacy preset reproduces every v0.4.0 line exactly; `tests/test_legacy_preset.py`
holds it to 0.01 on the overall score and an equal passing count. The defaults
are recorded in `tests/fixtures/baseline_v050_experiment.txt` and held by
`tests/test_v050_baseline.py` at the same tolerance. With PyPhi absent the
tracker path falls back to the calculator (`phi_source_used` reads
`calculator_spectral_proxy`), so the default numbers depend on the environment
recorded in `tests/fixtures/baseline_v050_env.txt`.

The seeded assessment CLI (`scripts/run_assessment.py --full --json --seed 42`,
five warmup cycles, the standalone modules without RPT, PAD, the analyzer or
neural states):

| Line | `--preset legacy_v04` | defaults |
| --- | --- | --- |
| overall_score | 0.333844 | 0.380308 |
| passing_count | 13 | 14 |
| status | partial | partial |
| coverage | 0.833 | 0.633 |
| confidence | 0.480 | 0.337 |
| insufficient indicators | none (the v0.4.0 constants stand in) | prediction_error_minimization, integrated_information, bayesian_binding_quality, epistemic_depth, genuine_implementation |
| GWT | 0.468 | 0.537 |
| AST | 0.621 | 0.280 |
| HOT | 0.656 | 0.718 |
| FEP | 0.330 | 0.336 |
| IIT | 0.000 | 0.000 |
| RPT | 0.155 | 0.225 |
| AE | 0.564 | 0.157 |
| BLT | 0.000 | omitted (no included indicator) |

The v0.4.0 assessment fixture (`tests/fixtures/baseline_v040_assessment.json`)
was captured unseeded at 0.306801 with 13 passing (GWT 0.487, AST 0.620, HOT
0.586, FEP 0.202, RPT 0.160, AE 0.571, IIT and BLT 0.000), so the legacy run is
compared by range, not equality: it lands within 0.07 of each theory except FEP
(0.330 against 0.202), whose inference statistics depend on the unseeded draw.
The defaults are recorded in `tests/fixtures/baseline_v050_assessment.json`.
Five indicators report `insufficient_data` after five cycles: the error trend
needs ten prediction errors, the PhiTracker needs more transitions than five
cycles record and there are no neural states to fall back on, and the three BLT
indicators read a beautiful loop that the standalone CLI never runs a conscious
moment through (its `cycle_count` is 0). Under `v04_constants` the same five
scored 0.0 (or 0.229 for the error trend) and were averaged in. The legacy
run's 13 passes include causal_emergence, which passes as an absent analyzer
(score 0.0 at threshold 0.0 under the inclusive v04 rule), exactly as the
v0.4.0 fixture does; none of the defaults' 14 passes is an absent indicator. Under the
defaults AE drops from 0.564 to 0.157 because `agency` and `embodiment` lose
their presence floors (raw 0.384 and 0.198 against 0.687 and 0.790) and
`embodiment` reads the typed voluntary ratio of 0.5 instead of the
strength-based 1.0, and AST drops from 0.621 to 0.280 because
`attention_control` reads that same typed ratio (0.449 instead of 1.000); both
of those ratios are Phase 2 switches (`ast.provenance_mode`,
`runner.warmup_profile`) at work, not Phase 3 ones.

### One switch at a time

Each row sets one switch to its v0.4.0 value on top of the defaults (a params
file copied from `experiments/params.toml` with the one key added) and runs the
experiment runner:

| Switch at its v0.4.0 value | overall_score | passing_count | Theory lines that moved |
| --- | --- | --- | --- |
| `assessment.indicator_formulas = "v04"` | 0.004923 | 0 | FEP 0.081411 to 0.009010, RPT 0.005109 to 0.003695 |
| `assessment.missing_signal_policy = "v04_constants"` | 0.021675 | 1 | none |
| `assessment.irreducibility_method = "spectral_entropy"` | 0.019578 | 1 | IIT 0.026167 to 0.005406 |
| `assessment.causal_emergence_threshold = 0.0` | 0.021675 | 1 | none (causal_emergence stays failed: the v05 pass rule is strict, and 0.000 > 0.0 is false) |
| `assessment.phi_source = "calculator"` | 0.021675 | 1 | none (`phi_source_used` reads calculator_spectral_proxy) |
| `pad.coherence_method = "v04_variance"` | 0.019818 | 1 | FEP 0.081411 to 0.072401 |
| `causal_emergence.tpm_row_order = "little_endian"` | 0.021675 | 1 | none |
| `rpt.stdp_plasticity_enabled = false` | 0.021675 | 1 | none |
| `assessment.saturated_baseline_min_range = 0.0` (Phase 1, re-checked) | 0.021675 | 1 | none |

`assessment.phi_normalization`, `assessment.phi_fixed_reference`,
`pad.coherence_window` and `pad.coherence_min_history` have their v0.4.0 value
as the default, so their one-at-a-time run is the default run. Three Phase 3
switches move a runner line (`indicator_formulas`, `irreducibility_method`,
`pad.coherence_method`); the rest are inert in this runner because every module
is warm (nothing is missing), the normalized causal emergence and Phi scores are
0.0 under either reading, the v05 pass rule keeps causal_emergence failed at a
threshold of 0.0, and no STDP statistics are passed. The remaining
distance to the legacy preset is the Phase 2 competition switch (next sections).

### Indicators that moved by more than 0.01

Normalized scores of the experiment runner, `--preset legacy_v04` against the
v0.5.0 defaults, with the raw score and the five-seed noise baseline:

| Indicator | legacy_v04 | v0.5.0 defaults | Switch and mechanism |
| --- | --- | --- | --- |
| prediction_error_minimization | 0.000 (raw 0.254, noise 0.258) | 0.434 (raw 0.684, noise 0.441) | `assessment.indicator_formulas`. The v05 body scores the least-squares trend of the last 20 prediction errors in the inference history (weight 0.7) plus the error level (0.3); v0.4.0 compared a text label to a number and never scored a trend. The trend of 20 random errors is the most seed-sensitive statistic in the runner (noise standard deviation 0.128, range 0.305 to 0.664 over the five noise seeds); seed 42 reads 0.684, 1.9 standard deviations above the noise mean, which is why this is the one indicator that passes under the defaults. It passes by seed position, not by signal: the input is the same Dirichlet noise as the baseline. |
| affect_coherence | 0.438 (raw 0.991, noise 0.985) | 0.054 (raw 0.093, noise 0.042) | `pad.coherence_method` together with `assessment.saturated_baseline_min_range`. The variance formula reads 0.99 on uniformly random sources and the legacy preset's zero headroom rescales a 0.007 sliver to 0.438; the lag-1 autocorrelation of random sources is near zero on both sides. |
| irreducibility | 0.016 (raw 0.754, noise 0.750) | 0.079 (raw 0.857, noise 0.845) | `assessment.irreducibility_method`. Both statistics read the random symmetric connectivity close to their noise mean (the weighted Fiedler value's noise standard deviation is 0.007); the move is seed position under a different statistic. |
| ignition_dynamics | 0.130 (raw 0.904, noise 0.890) | 0.000 (raw 0.784, noise 0.808) | `gwt.competition_mode` (Phase 2), not a Phase 3 switch: the formula is the same under v04 and v05. Hierarchical competition admits three level-0 winners per cycle instead of seven, and the seed-42 run ignites 16 times in 25 cycles against a five-seed noise mean of 17.0 (range 15 to 18); under the legacy preset the flat competition ignites 21 times against a noise mean of 20.4. In both cases the score is the position of one seed against five others of the same noise. |
| global_ignition_nuanced | 0.130 (raw 0.920, noise 0.908) | 0.000 (raw 0.820, noise 0.840) | Same mechanism as ignition_dynamics. The v05 body drops the decay term the detector never emitted and reads the sustain rate alone, which was 1.0 on both sides. |
| recurrent_processing | 0.130 (raw 0.920, noise 0.908) | 0.000 (raw 0.855, noise 0.866) | `gwt.competition_mode` through the GWT term (a broadcast rate of 16/25 against 21/25), and `assessment.indicator_formulas`, whose v05 body blends 0.5 x RPT average global recurrence with the GWT term when RPT measurements exist. |

Every other indicator moved by less than 0.01 (organized_percepts 0.000 to
0.009, local_recurrence 0.013 to 0.009, algorithmic_recurrence 0.010 to 0.007,
agency 0.008 to 0.000). The theory lines follow: GWT 0.052174 to 0.000000 is the
two ignition indicators, RPT 0.030645 to 0.005109 is recurrent_processing, FEP
0.072995 to 0.081411 is prediction_error_minimization up and affect_coherence
down, IIT 0.005406 to 0.026167 is irreducibility, AE 0.004150 to 0.000000 is
agency.

`passing_count` 2 to 1: the legacy preset passes affect_coherence (0.438
against its 0.3 threshold) and causal_emergence (normalized 0.000 against a
threshold of 0.0 under the inclusive v04 pass rule); the defaults pass
prediction_error_minimization alone (0.434 against 0.3). affect_coherence drops
below its threshold through `pad.coherence_method`, and causal_emergence stops
passing through `assessment.indicator_formulas` (the strict v05 rule) together
with `assessment.causal_emergence_threshold` (0.5). Either of those two switches
alone at its v0.5.0 value is enough to fail it, which is why neither
one-at-a-time run above brings `passing_count` back to 2.

### Noise saturation under the defaults (gate fired, decision pending)

The re-baseline plan required confirming that no indicator is `noise_saturated`
under the defaults, and stopping to report if a default inflated a noise score.
Both conditions fired, so this section records the facts for a decision, not a
result. Eight indicators carry `noise_saturated: true` under the defaults in
this runner: their five-seed noise baseline leaves less than the 0.1 headroom of
`assessment.saturated_baseline_min_range`, so the guard scores them 0.0 as
uninformative.

| Indicator | legacy_v04 noise | v0.5.0 noise | Why the floor is where it is |
| --- | --- | --- | --- |
| global_broadcast | 1.000 | 1.000 | Same floor in v0.4.0, which already normalized to 0.0. |
| higher_order_representations | 1.000 | 1.000 | Same floor in v0.4.0. |
| hierarchical_prediction | 1.000 | 1.000 | Same floor in v0.4.0. |
| bayesian_binding_quality | 0.999 | 1.000 | Same floor within 0.001; the legacy preset's zero headroom leaves it unflagged. |
| oscillatory_binding_coherence | 0.997 | 0.997 | Same floor; unflagged under the legacy preset for the same reason. |
| agent_environment_coupling | 0.980 | 0.980 | Same floor; unflagged under the legacy preset for the same reason. |
| genuine_implementation | 0.667 | 1.000 | Raised by `assessment.indicator_formulas`. The v05 body reads the loop's real `peak_depth` where the v04 body read a dead key that always gave 0 (two of three conditions, hence 0.667). The runner's warmup hands the loop one AWARENESS higher-order thought per cycle, which the metacognition module phrases as "I notice that i am perceiving ..." and the epistemic depth tracker counts as a self-referential thought, so `peak_depth` is 3 on every seed and the strange-loop condition is met on any input. |
| temporal_prediction_depth | 0.860 | 0.920 | Raised by `assessment.indicator_formulas`. 0.4 of the v05 formula is belief and free-energy history depth (0.2 x min(1, length / 20) each), which any 20-cycle run fills, and `experiments/params.toml` pins the planning horizon at 10, so the horizon term reads 0.6 x (1 - average error). The v04 body read (1 - average error) alone. |

The two raised floors are the gate's trigger. In both cases the v05 formula
reads a true statistic that a dead key or a narrower formula hid before, and in
both cases the raw seed-42 score sits at or below the noise mean, so neither
indicator gains a normalized score: all eight read 0.0, and setting the guard to
its v0.4.0 value of 0.0 leaves every runner line unchanged (0.021675, 1
passing). `tests/assessment/test_recalibration.py` passes under the defaults
(at most 3 indicators pass on noise, no perfect score, overall below 0.15).
What changed is that two indicators which could register in this runner under
v0.4.0 can no longer register in it under the defaults: genuine_implementation
because the warmup's own higher-order thought satisfies the strange-loop
condition on any input, and temporal_prediction_depth because 0.4 of its range
is filled by run length alone.

A decision is needed before the v0.5.0 fixture is committed, in one of two
forms:

1. Make the two stop saturating on this noise feed. For genuine_implementation
   the saturation comes from the runner's warmup feed, not the formula, so the
   candidate change is to the higher-order thought the warmup hands the loop
   (`scripts/run_experiment.py`), not to the indicator. For
   temporal_prediction_depth the two history-depth terms are run-length credit,
   and the candidate change is to drop or shrink them in
   `mtc/assessment/assessment.py`. Either change is outside the re-baseline
   task's files, regenerates `tests/fixtures/baseline_v050_experiment.txt` and
   may move the overall score.
2. Waive the check for this runner and keep the fixture as recorded. The guard
   already scores the eight as uninformative, the recalibration test passes, and
   the saturation is a statement about a synthetic warmup whose noise baseline
   is the feed itself. The cost is that these two indicators stay unmeasurable
   in the experiment runner until a later phase feeds it structured content.

The fixture in this tree records the eight at 0.0 normalized. Until the decision
is recorded here, the numbers above are a report, not a result.

### Candidate mix

The Phase 2 drop (0.048668 after Phase 1 to 0.000969 after Phase 2, now
0.021675 after Phase 3) was attributed to the runner's sensory-only candidate
feed starving the hierarchical competition. The raw counts confirm the
starvation: with seven SENSORY candidates per cycle the hierarchical bottleneck
admits 3.0 winners on average (level-0 capacity 3; promotion needs a salience of
0.7 against a mean top salience of 0.59) where the flat competition admitted
7.0, the workspace ignites 16 times in 25 cycles instead of 21,
`source_distribution` holds one source and flexible_responsiveness reads a
diversity ratio of 0.333. A mixed feed (the same seven candidates and the same
random draws, with sources SENSORY x3, MEMORY, CONCEPT, GIST and HOMEOSTATIC)
exercises all three levels: 7.0 winners per cycle, 44 ignitions, five sources,
diversity ratio 1.0. It does not move the normalized GWT score, which stays
0.000 (ignition_dynamics raw 1.000 against a noise mean of 1.000,
flexible_responsiveness 0.700 against 0.700, sequential_module_querying 0.850
against 0.850), because the noise baseline is computed from the same synthetic
feed and rises with it. The overall score moves from 0.021675 to 0.006075
(sources only; 0 passing) or 0.027715 (sources with matching content types; 1
passing), and all of that movement is in the seed-sensitive FEP trend term and
in HOT's sparse_smooth_coding through the shared random stream, none of it in
GWT. One-at-a-time runs of the Phase 2 switches agree: `gwt.competition_mode =
"flat"` alone gives 0.041626 with GWT still 0.000 (FEP 0.169885 through the
random stream); `gwt.thalamic_gate_enabled = false` is identical to the defaults
(the gate filtered 0 of 175 candidates: the runner's activation levels sit above
its threshold, no goals are set and the content is new every cycle);
`gwt.theory_interactions_enabled = false` gives 0.021807; and
`legacy_coverage_ratio`, `precision_feedback_enabled` and
`self_prediction_enabled` at their v0.4.0 values are identical to the defaults.

Decision: no `runner.candidate_mix` switch. The starvation is real in the raw
counts but is not the dominant cause of the low default score. The dominant
cause is that the runner's synthetic warmup is noise by construction, and
normalizing against a same-feed noise baseline zeroes whatever the feed itself
produces, under either competition mode and either candidate mix; a mixed feed
lifts the raw GWT counts and the noise baseline together. The preset is
unchanged. The hierarchy becomes measurable when a later phase feeds structured
content (real memories, concepts and drives) whose noise baseline is not the
content itself.

### Body metrics in seeded runs

`gwt.body_metrics_enabled` must stay pinned off in seeded runs. Two runs of the
experiment runner with the same seed and `gwt.body_metrics_enabled = true`
printed different lines: overall_score 0.036963 against 0.027516, gwt_score
0.047059 against 0.021622, iit_score 0.051853 against 0.032229, rpt_score
0.015747 against 0.007200 and fep_score 0.082771 against 0.081159. The
protoself reads machine metrics and the previous cycle's wall-clock duration,
and they reach the next cycle's salience bias through the theory interactions,
so the competition outcome, and everything downstream of it, changes from run
to run. Both runners
keep the pin (`body_metrics_default=False` in `build_workspace`) unless the key
is set explicitly.

## Changes outside the scores

v0.4.0 contained no database, API or onboarding code, so these changes move no
assessment number. They change what an install looks like.

**Store names come from `DB_PREFIX`.** With `MTC_DB_PREFIX` at its default
`mtc`, an empty store name derives from it: PostgreSQL database
`mtc_consciousness` and role `mtc`, MongoDB database `mtc_neural`, Qdrant
collection `mtc_memories`, Redis key prefix `mtc:`. The v0.4.0 `Settings`
defaults were `consciousness_db` and `consciousness` for PostgreSQL and
`consciousness_memories` for the MongoDB database and the Qdrant collection.
To keep the old names, set `MTC_POSTGRES_DB`, `MTC_POSTGRES_USER`,
`MTC_MONGODB_DB` and `MTC_QDRANT_COLLECTION` explicitly; an explicit value is
never replaced by a derived one.

**Renamed environment keys.** The v0.4.0 `.env.example` used names `Settings`
never read. They are now the field names with the `MTC_` prefix:
`MTC_PG_HOST`, `MTC_PG_PORT`, `MTC_PG_USER`, `MTC_PG_PASSWORD` and
`MTC_PG_DATABASE` became `MTC_POSTGRES_HOST`, `MTC_POSTGRES_PORT`,
`MTC_POSTGRES_USER`, `MTC_POSTGRES_PASSWORD` and `MTC_POSTGRES_DB`;
`MTC_MONGO_HOST`, `MTC_MONGO_PORT` and `MTC_MONGO_DATABASE` became
`MTC_MONGODB_HOST`, `MTC_MONGODB_PORT` and `MTC_MONGODB_DB`; the unprefixed
`DCM_API_URL`, `DCM_CACHE_DIR` and `DCM_EVALUATOR_MODE` became
`MTC_DCM_API_URL`, `MTC_DCM_CACHE_DIR` and `MTC_DCM_EVALUATOR_MODE`. The
Anthropic key is read from `ANTHROPIC_API_KEY` or `MTC_ANTHROPIC_API_KEY`.
`.env.example` and [CONFIGURATION.md](CONFIGURATION.md) are generated from
`Settings`, so they cannot drift again.

**Removed keys.** `MODEL_TYPE` (`LLM_MODEL` replaces it); `API_WORKERS` (the
runtime is in-process state, so the server runs one process); `API_TIMEOUT`
(the generation bounds `GENERATION_TIMEOUT_S` and
`GENERATION_STALL_TIMEOUT_S` replace it); `ENABLE_GRAPH_DATABASE` and
`ENABLE_VECTOR_SEARCH` (the per-store enable keys replace them);
`ENABLE_CONSCIOUSNESS_METRICS`, `ENABLE_DREAM_PROCESSING`,
`ENABLE_MULTI_MODAL` and `ENABLE_SAFETY_MONITOR` (nothing read them);
`NEO4J_HOST`, `NEO4J_PORT` and `NEO4J_BOLT_PORT` (`NEO4J_URI` carries the
address); and the key of a provider MTC does not support.

**Everything runs in memory by default.** The per-store enable keys
`POSTGRES_ENABLED`, `MONGODB_ENABLED`, `REDIS_ENABLED`, `QDRANT_ENABLED` and
`NEO4J_ENABLED` all default false, so a fresh install keeps every store in
process memory until `mtc init` (or an edit of `.env`) turns one on. See
[DATABASES.md](DATABASES.md).

**The API.** `API_PORT` moved from 8000 to 8200, the server binds
`127.0.0.1` by default, and authentication fails closed: `API_AUTH_ENABLED`
defaults true, and with an empty `API_KEY` every authenticated route answers
503. Authentication off is refused on any address outside loopback. See
[API.md](API.md) and [SECURITY.md](SECURITY.md).

**Vectors are 384 dimensions.** The memory layer embeds with
`EMBEDDING_MODEL` (`sentence-transformers/all-MiniLM-L6-v2`) into a
384-dimension Qdrant collection. A collection that exists with another width
(an 8192-dimension collection from an older deployment, for example) is
refused with `VectorDimensionMismatch` and never recreated in place: set
`MTC_QDRANT_COLLECTION` to a new name and run the re-embed backfill
(`python -m mtc.db.reembed`, or `scripts/db/reembed_postgres_to_qdrant.py`),
which embeds every episodic row that has no point yet.

**The `mtc` command.** One console script replaces the loose scripts as the
entry point: `mtc init`, `mtc doctor`, `mtc db init`, `mtc db check`,
`mtc assess`, `mtc serve`, `mtc chat`, `mtc ctm` and `mtc privacy-audit`.
`scripts/run_assessment.py` still works: it is a shim over `mtc assess`, with
the same flags and the same results.

**Five defaults changed with their new readers.** `GPU_LAYERS` 35 to -1 (-1
leaves the model server's own default), `THREADS` 8 to 0 (0 leaves the
server's own default), `USE_MLOCK` true to false (locking a whole model in
memory fails without raised memory-lock limits), `CONTEXT_SIZE` 131072 to
32768 (a 128k context allocates a key-value cache most machines cannot hold)
and `DCM_CACHE_DIR` `data/dcm` to `data/dcm_cache` (the tracked `data/dcm`
holds shipped data only; caches go to an ignored directory). Set the old value
in `.env` to keep it.

**Extras.** The `research` extra replaces `assessment`, which stays as an
alias, so `pip install ".[assessment]"` keeps working. The `iit` extra (PyPhi)
needs Python 3.12 or later and is never part of `all`; see
[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) for its licence.
