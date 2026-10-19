# Migrating from v0.4.0

The four headline numbers, seed 42, each a committed fixture that a test holds:

| Run | `--preset legacy_v04` (the v0.4.0 behaviour) | v0.5.0 defaults |
| --- | --- | --- |
| Experiment runner (`scripts/run_experiment.py`, 25 warmup cycles) | 0.031247, 2 of 30 passing: every metric line of the v0.4.0 run (`tests/fixtures/baseline_v040_experiment.txt`) | 0.017317, 0 of 30 passing, status partial, coverage 0.800 (`tests/fixtures/baseline_v050_experiment.txt`) |
| Seeded assessment (`scripts/run_assessment.py --full --json --seed 42`) | 0.333844, 13 of 30 passing, status partial, coverage 0.833 (`tests/fixtures/baseline_legacy_assessment.json`) | 0.275203, 11 of 30 passing, status partial, coverage 0.533 (`tests/fixtures/baseline_v050_assessment.json`) |

The legacy column does not move: the preset reproduces every printed metric
line of the v0.4.0 experiment run and every field of the legacy seeded
assessment except three wall-clock readouts (see Reproducing v0.4.0). The
v0.5.0 column measures the corrected system; the sections below name the
switch behind every line that moved. The maintainer will publish an erratum
note for the paper's v0.4.0 numbers; until then this record is the reference
for the difference.

## Why the numbers changed

v0.5.0 resyncs the active inference, metacognition, beautiful loop, Damasio and
PAD modules, the global workspace, the attention schema and the PhiTracker, the
30 plus 3 indicator assessment, the measurement framework, RPT, causal
emergence, the DCM benchmark and the SNN, HTM, LSM and oscillator substrates
with the reference implementation, and fixes several defects that v0.4.0
measured around rather than fixed. Each behaviour change that can move a
published number (either runner, under either configuration) sits behind a
switch whose default is the new behaviour: a constructor config field, or for
`runner.assess_blt_source` a runner key the assessment CLI reads. Scores from
v0.5.0 defaults are therefore not comparable with the published v0.4.0
numbers. They are not worse or better by construction; they measure a
different (corrected) system, and the assessment now says what it could not
measure (`insufficient_data`) instead of substituting a constant. A fix
whose output no published number depends on carries no switch; those are
listed under Fixes without a switch.

A whole-branch review before the release found more defects of the same kind.
Their fixes add the eleven Phase 9 rows of the Switches section and move the
overall score and the passing count of both v0.5.0 runs: the workspace
registers no acknowledgement-only broadcast receivers and reports no ignition sustain rate until a sustain check runs, so
global_broadcast, recurrent_processing, ignition_dynamics and
global_ignition_nuanced report insufficient_data; attention_schema reports
insufficient_data until a focus prediction has been scored; the beautiful
loop binds only winners that carry their own numeric vector; and the seeded
assessment reads the loop the warmup drives. The Phase 9 measurements section
attributes every moved line; the legacy column was proven again after those
fixes.

Not changed in this release, and expected to move numbers when it is: the
small-sample correction of the module-graph integration proxy. On module
signals that never change the proxy reads about 0.5 bits when its
128-transition warm-up ends and falls roughly as one over the number of
transitions ([HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md), section 13); a
correction would move the salience bias of every default run, so it is
left as an open follow-up.

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

The seeded assessment CLI with the preset lands on
`tests/fixtures/baseline_legacy_assessment.json`, the record of the published
legacy run (`scripts/run_assessment.py --preset legacy_v04 --full --json --seed 42`:
0.333844, 13 of 30 passing, partial, coverage 0.833, confidence 0.480).
`tests/test_legacy_preset.py` compares every field of that report (statuses,
counts, keys and booleans exactly, floats within 1e-9) and every metric line
of the experiment runner as printed. Three evidence readouts of the legacy
seeded assessment follow the wall clock, through the introspection cooldown:
`higher_order_representations.evidence.hots_generated`,
`higher_order_representations.evidence.coverage_ratio` and
`metacognition.evidence.introspections` (11, 2.2 and 11 on the recorded
machine; 8, 1.6 and 7 on a faster one; 14, 2.8 and 15 on a slower one); no
score, status, confidence, count or printed line depends on them, and the
legacy assessment test leaves exactly those three out. They follow the clock
because the preset keeps `hot.introspection_cooldown = 1.0`, the v0.4.0
wall-clock gate; without a preset both runners pin the cooldown at 0.0, so
the v0.5.0 fixtures are compared in full. The cooldown is not the only clock
read on the legacy path: under the preset's `gwt.recency_clock = "v04_wall"`
the attention bottleneck's recency bonus (0.1 x exp(-0.1 x the wall-clock
age in seconds of a candidate) in every composite salience) and, under the
v0.4.0 binding path, the binding's temporal coherence term reach the
workspace's loop state, but in the legacy seeded assessment neither reaches
a field of the report: three runs in one process at clock speeds 1, 1 and
25 differ in the three cooldown readouts (11, 2.2, 11 against 14, 2.8, 15
at speed 25) and in nothing else. Under the defaults the bonus is measured
on the workspace's simulated clock (`gwt.recency_clock = "simulated"`), so
no wall-clock read enters the salience and the default seeded assessment is
bit identical at clock speeds 1 and 25 and between runs (see "What the
cooldown pin removes" in the Phase 9 measurements section).

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
| `blt.field_evidencing_threshold` | 0.3 (the value the baseline run applied) | 0.3 for `BeautifulLoop()`; 0.4 for the workspace-owned loop | `BeautifulLoop()` defaults to 0.3, the value the v0.4.0 runs applied (the constructor default was 0.4). The workspace-owned loop (`_WORKSPACE_BEAUTIFUL_LOOP_DEFAULTS` in `enhanced_global_workspace.py`), and so the live runtime without a preset, keep 0.4; the experiment runner applies 0.3 from `experiments/params.toml`. | Beautiful loop field evidencing for callers that relied on the old default. |
| `damasio.circadian` | `v04_sawtooth` | `sinusoid` | A smooth day cycle instead of a sawtooth with a hard reset; `wall_clock` is also available. | Protoself arousal and, through the body delta, the protoself stability that feeds the PhiTracker's AST node, so under the defaults (`phi_source = tracker_preferred`) it can reach `iit_score`; not under the legacy preset, whose Phi comes from the calculator. Measured at the Phase 9 regeneration: `v04_sawtooth` alone leaves every printed line of the seed-42 experiment unchanged, the IIT line (0.034115) included, and every field of the seeded assessment. |
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
| `phi.scale` | (none) | 1.0 | New. The PhiTracker's integration proxy is reported in raw bits; this multiplies it. The runner's own lightweight tracker is gone: the real tracker is attached to the workspace. The tracker's module graph has five nodes named after theories, but two of them read other layers: the AST node is the protoself stability and the HOT node the core layer's feeling of knowing (the FEP node the most urgent drive's urgency, the BLT node the binding quality or, while binding is unmeasured, the loop quality, the GWT node the integration level or a composite of the conscious flag and the ignition count). | Nothing at 1.0. |
| `phi.warmup_reference` | (none) | 5.0 | New. Denominator of the normalized readout until enough values exist for a percentile; a display calibration, never part of the measurement. The readout is over the same five module signals, so a partition that separates the AST or HOT node separates the protoself and core-layer signals, not the attention schema or the higher-order thought module. | The normalized Phi readout during warm-up only. |
| `assessment.indicator_formulas` | `v04` | `v05` | The indicators read the statistics their producers emit: the bottleneck's `total_competitions`, `avg_candidates` and `source_distribution` instead of a `competition` key the workspace never emits; `peak_depth` instead of `max_depth_reached`; `total_hots_generated` instead of `total_generated`; a numeric prediction-error trend from the inference history instead of a text label that never compared equal to a number; the agent's real policy count; zero floors where v0.4.0 granted presence credit. Under v05 the causal_emergence pass rule is also strict (`score > threshold`, because sigmoid(0) is exactly 0.5); the v04 rule is inclusive. The noise stubs of the assessment CLI's baseline pass ride on this switch too: `v04` keeps the v0.4.0 stubs (see Fixes without a switch). | sequential_module_querying, organized_percepts, flexible_responsiveness, genuine_implementation, epistemic_depth, no_metacognitive_access, prediction_error_minimization, complex_agency, temporal_prediction_depth, recurrent_processing, online_reportable_learning, agency and embodiment. In the runner, under the Phase 3 defaults, it moved FEP from 0.009010 to 0.081411 and RPT from 0.003695 to 0.005109 (the Phase 3 record). Under the Phase 9 defaults, `v04` set alone prints 0.010680, 0 passing, measured, coverage 1.000 against the default 0.017317, 0 passing, partial, 0.800: FEP 0.017706 to 0.042338, RPT 0.004442 to 0.008566, AE 0.011778 to 0.002917 and GWT 0.009726 to 0.000000 (v04 to v05; the v04 bodies read the six unmeasured indicators as their stand-ins, the v0.4.0 constant where v0.4.0 had one and 0.0 for the broadcast coverage and the sustain rate, where v0.4.0 read 1.0, so they count in the averages and the coverage again). Through the pass rule it also acts on `passing_count` together with `assessment.causal_emergence_threshold` (see that row). |
| `assessment.missing_signal_policy` | `v04_constants` | `insufficient_data` | A module the caller did not supply scores 0.0 as absent; a supplied module that has not produced its evidence yet is reported by name and left out of the averages and the passing count, instead of a 0.1, 0.3 or 0.5 stand-in. | Any indicator whose module has no data, plus the report's `status`, `coverage` and confidence. In the experiment runner nothing under the Phase 3 defaults, where every indicator was measured after 25 cycles (the Phase 3 record); under the Phase 9 defaults six indicators are unmeasured there as well, so `v04_constants` set alone moves the runner from 0.017317, 0 passing, partial, coverage 0.800 to 0.015245, 0 passing, measured, coverage 1.000 (default to legacy value: gwt 0.000000 to 0.008626, rpt 0.008566 to 0.006852, the other theory lines unchanged). In the seeded assessment CLI eight indicators are named under the Phase 9 defaults and coverage reads 0.533 (five and 0.633 in the Phase 3 record). |
| `assessment.irreducibility_method` | `spectral_entropy` | `weighted_fiedler` | The algebraic connectivity of the symmetrized weighted graph through its normalized Laplacian, instead of a spectral entropy that a 1e-9 binarization turned into a complete graph for any dense matrix. | irreducibility. In the runner IIT moved from 0.005406 to 0.026167 under the Phase 3 defaults (the Phase 3 record) and moves from 0.013354 to 0.034115 under the Phase 9 defaults (`spectral_entropy` set alone; overall 0.014588 to 0.017317, no other theory line moves): both methods read the random connectivity close to their own noise mean (0.857 against 0.845 for `weighted_fiedler`, 0.754 against 0.750 for `spectral_entropy`, unchanged since Phase 3), so the move is seed position under a different statistic, not signal. |
| `assessment.causal_emergence_threshold` | 0.0 | 0.5 | The causal emergence indicator passes only when the macro scale carries more causal power than the micro scale (a score strictly above sigmoid(0) = 0.5 under the v05 pass rule). At 0.0 with the inclusive v04 rule every causal_emergence result passed, including an absent analyzer scored 0.0: the v0.4.0 assessment fixture passes it that way (score 0.0, threshold 0.0, `analyzer not provided`). | The pass decision of causal_emergence, so `passing_count`. Under the legacy preset it is one of the two runner passes (normalized 0.000 at threshold 0.0, inclusive; the other is affect_coherence), and in the seeded assessment CLI one of the legacy run's 13. Set alone on top of the Phase 3 defaults it was inert (0.021675, 1 passing: the Phase 3 record), because causal_emergence normalized to 0.000 there and 0.0 > 0.0 is false under the strict v05 rule. Under the Phase 9 defaults it is not inert: causal_emergence normalizes to 0.024 (raw 0.708 against 0.701, see the moved-indicator table), which is above a threshold of 0.0 under the strict rule, so the threshold at 0.0 set alone prints 0.017317, 1 passing, partial, 0.800 against the default's 0 passing, with no theory line moved (the one-switch table of the Phase 9 measurements). |
| `assessment.phi_source` | `calculator` | `tracker_preferred` | integrated_information reads the exact IIT 4.0 Phi of the supplied PhiTracker's module graph when it has a result (labelled a module-graph integration proxy) and falls back to the PhiCalculator on the neural states while the tracker warms up; `calculator` never reads the tracker. | integrated_information and the `phi_source_used` line. In the runner the tracker's exact Phi is 0.0 (the five-node module graph is reducible: the partition separates AST from the rest) against a noise mean of 0.0, and the calculator's proxy on the random neural states reads 0.518 against 0.556 under the legacy preset, so both normalize to 0.0. |
| `assessment.phi_normalization` | `fixed` | `fixed` | Unchanged default. `percentile` divides by the rolling 95th percentile of the recent Phi values once 20 exist, so a steady Phi reads near 1.0. | Nothing at the default; no preset value. |
| `assessment.phi_fixed_reference` | 5.0 | 5.0 | Unchanged default: the Phi that maps to 1.0 under fixed normalization, a display calibration. | Nothing at the default; no preset value. |
| `pad.coherence_method` | `v04_variance` | `temporal_structure` | Coherence is the mean lag-1 autocorrelation of the three PAD dimensions over the last `coherence_window` states (a flat dimension scores 0, fewer than `coherence_min_history` states reports none), instead of a variance formula that pure noise maxed out, which is what the Phase 1 headroom guard was added for. | affect_coherence. On the runner's uniformly random sources the old formula reads 0.991 against a noise floor of 0.985 under the legacy preset, whose `saturated_baseline_min_range = 0.0` rescales that sliver to 0.438 (the legacy run, unchanged in Phase 9). The new method read 0.093 against 0.042 under the Phase 3 defaults and FEP moved from 0.072401 to 0.081411 (the Phase 3 record). Under the Phase 9 defaults the new method reads 0.155 against 0.054 (normalized 0.106; `blt.binding_evidence` moved the shared random stream under it), and `v04_variance` set alone reads 1.000 against 0.994, flagged noise-saturated by the 0.1 headroom guard and scored 0.0, so FEP prints 0.024633 against the default 0.042338 (overall 0.012569; no other theory line moves). |
| `pad.coherence_window` | 20 | 20 | New tuning key of the default method: the number of recent states the autocorrelation reads. | Nothing at the default; inert under `v04_variance`, so no preset value. |
| `pad.coherence_min_history` | 10 | 10 | New tuning key of the default method: states needed before a coherence is reported. | Nothing at the default; inert under `v04_variance`, so no preset value. |
| `causal_emergence.tpm_row_order` | `little_endian` | `big_endian` | The analyzer reorders the PhiTracker's big-endian rows before coarse-graining, so the macro partition groups the right source states; `little_endian` reads the rows as given, which matched the v0.4.0 warmup tracker but not the rebuilt one. | causal_emergence. Under the Phase 3 defaults the normalized score was 0.0 either way and the analyzer's causal_emergence value read 0.725 under the default order (its raw score 0.810 against a noise baseline of 0.810; the same values for every seed and candidate feed there, because the recorded module-state transitions did not depend on the random stream) and 0.652 under the legacy order (raw score 0.787; the Phase 3 record). Under the Phase 9 defaults the tracker's HOT and BLT nodes carry measured signals, so the transitions follow the other switches (see the moved-indicator table): the default order's raw score reads 0.708 against a noise baseline of 0.701 (normalized 0.024; the analyzer's value 0.443), and `little_endian` set alone reads a raw score of 0.842 against 0.834 (normalized 0.046; the analyzer's value 0.837), so iit_score prints 0.041616 against the default 0.034115 (overall 0.018500; irreducibility and integrated_information unchanged, no other theory line moves). |
| `rpt.stdp_plasticity_enabled` | false | true | RPT adds a plasticity component (min(1, 2 x mean weight standard deviation + weight diversity)) when a caller passes the SNN's STDP statistics with `plasticity_active`; the local recurrence index stays equal to the local score, so plasticity is counted once. | local_recurrence and algorithmic_recurrence once an SNN's STDP statistics are supplied. Inert in both runners, which pass synthetic spike counts without them. |
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

### Phase 9 rows

The v0.5.0 fix round (Phase 9) put eleven more corrections behind switches:
ten at the first regeneration of the v0.5.0 fixtures and `gwt.recency_clock`
at the second (see the Phase 9 measurements section). Each
row has its v0.4.0 value in `legacy_v04.toml`; the "What it can move" column is
measured in the Phase 9 measurements section (one switch at a time from the
defaults, seed 42; the numbers read legacy value to default).

| Switch | v0.4.0 | v0.5.0 default | What it fixes | What it can move |
| --- | --- | --- | --- | --- |
| `assessment.noise_baseline_matching` | `v04_any` | `producer` | A Phi value is left un-normalised when the noise baseline was made by a different producer (the stub pass) instead of being normalised against whatever baseline exists. | integrated_information for `mtc assess --state-dir` with a reporting tracker and for API callers with normalisation on. Nothing in either published run: both normalise against baselines their own producer made. |
| `runner.assess_blt_source` | `standalone` | `workspace` | The assessment CLI's standalone path hands the assessment the workspace's own beautiful loop, which the warmup drives, instead of a separate loop nothing drives. A runner setting; it changes no module default. | The seeded assessment: epistemic_depth and genuine_implementation become measured (0.385 and 0.615, both passing), `theory_scores.BLT` appears (0.500), coverage 0.467 to 0.533, 9 to 11 passing, overall 0.248 to 0.275. Nothing in the experiment runner. |
| `gwt.legacy_broadcast_receivers` | `true` | `false` | No acknowledgement-only receivers: broadcast coverage is reported only for receivers a caller registers, and is none until then, instead of 1.0 by construction. | global_broadcast and recurrent_processing read insufficient_data in both runners. Experiment: coverage 0.867 to 0.800, rpt_score 0.016029 to 0.008566, overall 0.017033 to 0.017317. Seeded assessment: GWT 0.459 to 0.189, RPT 0.225 to 0.098, 13 to 11 passing, overall 0.341 to 0.275. |
| `gwt.legacy_sustain_rate` | `true` | `false` | The ignition sustain rate is none until a sustain check has run, instead of a constant 1.0 while no check runs. | ignition_dynamics and global_ignition_nuanced read insufficient_data in both runners. Experiment: gwt_score 0.039474 to 0.000000, coverage 0.867 to 0.800, overall 0.023765 to 0.017317. Seeded assessment: GWT 0.432 to 0.189, 13 to 11 passing, overall 0.321 to 0.275. |
| `gwt.refractory_clock` | `per_cycle` | `per_cycle` | New row, the same value on both sides. `per_cycle` advances the simulated refractory clock by the competition duration per cycle, machine independent; `elapsed` also adds the real time between cycles. | Nothing in either runner (both keep the default; measured: identical runs). The live runtime sets `elapsed`, so the pipeline no longer discards the winners of turns that arrive within one simulated refractory period. |
| `gwt.recency_clock` | `v04_wall` | `simulated` | The attention bottleneck ages a candidate's recency bonus (0.1 x exp(-0.1 x age) in every composite salience) on the workspace's simulated clock since it first saw the candidate, instead of on the wall clock since the candidate was created, so no wall-clock read enters the salience of a seeded run. Added at the second regeneration. | One evidence value of the default seeded assessment, `genuine_implementation.evidence.avg_loop_quality`, 0.5300177410248634 (the first regeneration, under the wall-clock bonus) to 0.5300177409697049 (5.5e-11, under the 1e-9 comparator); no score, status or count, and no printed line of the experiment runner. Under `v04_wall` that value follows the machine's speed (1.2e-9 at a clock 25 times faster); under `simulated` the report is bit identical at clock speeds 1 and 25 and between runs. |
| `hot.overall_confidence_source` | `v04_constant` | `hot_mean` | The metacognition module reports the mean confidence of its buffered higher-order thoughts (none before the first) instead of the constants 0.5 and doubt 0.0. | Experiment: iit_score 0.026167 to 0.034115 and overall 0.016063 to 0.017317 (causal_emergence reads the PhiTracker's transitions, and the tracker's HOT node reads the feeling of knowing the self-model derives from this confidence). Seeded assessment: metacognition 0.394 to 0.473 (the confidence term 0.5 to 0.6), HOT 0.609 to 0.628, 10 to 11 passing, overall 0.272 to 0.275. |
| `hot.introspection_counting` | `v04_per_hot` | `episodes` | Introspections count completed episodes, not one more per higher-order thought after the third. The legacy preset also sets `hot.introspection_cooldown = 1.0`; both runners otherwise pin the cooldown at 0.0. | Seeded assessment: metacognition 0.684 to 0.473 (the activity term 1.0 to 0.6), online_reportable_learning 0.451 to 0.319, HOT 0.714 to 0.628, overall 0.287 to 0.275. No printed line of the experiment runner moves. |
| `blt.binding_evidence` | `v04_salience_posterior` | `candidate_vectors` | Binding is measured only between winners that carry their own numeric vector and reports no quality otherwise, instead of belief vectors built from salience and a shared posterior, which bound at about 1 for any input and drew from the shared random stream on every inference. | The largest mover. Experiment: bayesian_binding_quality insufficient_data (coverage 0.833 to 0.800) and, through the shared random stream (the default path draws nothing), fep_score 0.081411 to 0.042338, iit_score 0.026167 to 0.034115, ae_score 0.000000 to 0.002917, rpt_score 0.005452 to 0.008566, passing_count 1 to 0, overall 0.026982 to 0.017317. Seeded assessment: bayesian_binding_quality 0.999 to insufficient_data, sparse_smooth_coding 0.815 to 0.722, complex_agency 0.524 to 0.096, sequential_module_querying 0.335 to 0.378, temporal_prediction_depth 0.332 to 0.318, agency 0.219 to 0.209; FEP 0.336 to 0.247, BLT 0.666 to 0.500, HOT 0.652 to 0.628, GWT 0.168 to 0.189, AE 0.157 to 0.152; 13 to 11 passing, overall 0.328 to 0.275. |
| `blt.strange_loop_test` | `v04_confidence_match` | `independent_calibration` | The strange-loop test compares the self-model's scored self-prediction accuracy with the inference ledger's forecast accuracy instead of comparing the self-model's confidence with the value it was copied from. | Nothing in either published run (measured: identical runs). The experiment runner's standalone loop reads `self_calibration_score` from the runner's stand-in self-model (`_WarmupSelfModel` in `scripts/run_experiment.py`, a constant 0.5), so the verdict is true on any input there under either test (genuine_implementation raw 1.000 against a noise baseline of 1.000). The verdict reaches genuine_implementation only through `field_evidencing_count` (the loop counts a cycle as field evidencing only when the verdict is true): the indicator's third condition is the evidence key `has_meta_prediction_depth`, the depth tracker's `peak_depth` >= 2 read from module state, not the verdict. In the default seeded run the verdict is false with basis insufficient_data on all five cycles (`meta_state.total_predictions` 4, minimum 5), so `field_evidencing_count` is 0 under either test while `has_meta_prediction_depth` is true. |
| `damasio.first_delta` | `v04_zero` | `unmeasured` | The protoself reports no body delta, stability or arousal on the first update, which has no previous body state, instead of recording 0.0, 1.0 and 0.0. Under the default the first delta also stays out of the stability history, so the stability differs on every following cycle, not only the first. | Experiment: iit_score 0.037811 to 0.034115, fep_score 0.040530 to 0.042338, rpt_score 0.009301 to 0.008566, ae_score 0.002660 to 0.002917, overall 0.017488 to 0.017317 (the stability feeds the PhiTracker's AST node and, through the theory interactions, the next cycle's salience bias). Seeded assessment: one evidence value (genuine_implementation `avg_loop_quality`), no score or status. |

One dependency: `blt.loop_quality_formula = "v04"` requires
`blt.binding_evidence = "v04_salience_posterior"` (the v04 formula reads a
binding quality only that evidence measures). The legacy preset sets both, and
`BeautifulLoop` refuses the formula without the evidence at construction with a
message that names both keys, so the formula cannot be set alone on top of the
defaults. Measured together (the formula with its companion), the runs equal the
`blt.binding_evidence` run alone in every printed line and every score.

A workspace built directly keeps the simulated per-cycle clock for its refractory
period; the runtime runs its workspace on the elapsed clock. A caller that drives
a workspace in real time sets `gwt.refractory_clock = elapsed`.

### Existing switches whose reach widened in Phase 9

No new row and no change of legacy value; one line per cause.

- `assessment.missing_signal_policy` (`v04_constants` keeps every v0.4.0 constant):
  - the attention schema's prediction accuracy is None until a prediction has been scored, so `attention_schema` reads insufficient_data in both published runs: on its own this moves the experiment's coverage 1.000 to 0.967 and overall 0.021675 to 0.022394 (the same weighted mean without that indicator), and the seeded AST 0.280 to 0.449;
  - the voluntary ratio is None with nothing classifiable;
  - the local recurrence index is None without an SNN or LSM state, so the seeded assessment's `organized_percepts` records `avg_local_recurrence` None where it recorded 0.0 (no score moves);
  - the v05 bodies read a missing statistic as missing instead of its v0.4.0 stand-in (0.0 for each of these, 0.5 for the embodiment's voluntary ratio) and report insufficient_data naming it (the bottleneck's average candidate count, the embodiment's capacity used and voluntary ratio, the free energy of agency, the local and global recurrence averages, the binding order parameter, the average binding quality, the average epistemic depth and the average loop quality);
  - the average binding quality is None when nothing was bound (the consumer side of `blt.binding_evidence`);
  - the overall is None when no included indicator is measured, and a theory with no measured member is omitted from `theory_scores`: the seeded assessment loses its IIT entry of 0.0.
- `assessment.indicator_formulas` (`v04`): the v04 bodies read each None as a stand-in: the v0.4.0 constant where v0.4.0 had one, and 0.0 for broadcast.coverage_ratio and ignition.sustain_rate, where v0.4.0 read 1.0 by construction (the acknowledgement-only receivers and the sustained flag that defaulted True); the legacy preset restores both producers, which is what keeps it exact, while `v04` alone reads those two as 0.0.
- `assessment.saturated_baseline_min_range` (`0.0`): the oscillatory binding layer measures its own zero-stimulus order parameter on every bind and the v05 formula applies the same headroom guard to that floor. In the experiment runner `oscillatory_binding_coherence` reads raw 0.0 flagged `floor_saturated` (zero-stimulus order parameter 0.9987, noise baseline 0.0) where it read 0.997 flagged noise-saturated; its normalized score is 0.0 either way and no printed line moves.
- `ast.provenance_mode` (`legacy_strength`): the ratio is None with nothing classifiable and a request matches a shift only by its own type; only a run with no chosen and no pulled shift sees either, and neither published run does.

## Fixes without a switch

Changes that give identical results for v0.4.0 inputs (history trims, bounded
dictionaries, the encoder's treatment of negative or NaN entries, and similar
defects) are not listed. The fixes below changed behaviour but need no switch
because no published v0.4.0 number depended on the path:

- The PhiCalculator's exact path passed a keyword PyPhi rejects, so it raised and fell back to the spectral proxy on every call; it now runs through the PyPhi bridge when a TPM is supplied. Every reported v0.4.0 Phi came from the proxy, which is unchanged.
- `run_normalized_assessment` normalized an already normalized report; it now requests raw scores. No published number used that method.
- The noise stubs emit the producers' schema (bottleneck keys, `peak_depth`, numeric prediction errors). Only the stub-based baseline reads them. This change has no switch of its own: it rides on `assessment.indicator_formulas`, whose `v04` value keeps the v0.4.0 stubs in the noise pass, so no v0.4.0 number moves. The experiment runner computes its baselines from real modules; the assessment CLI passes none, so it normalises against this stub pass and its v0.5.0 numbers (the seeded assessment fixture) include the change under the default `v05`.
- The legacy 14-indicator integration never awaited its indicators (every score was a coroutine, and the mean raised) and its workspace-integration metric read a NaN correlation on constant vectors as 1.0; it now awaits each indicator and reports `insufficient_data` with zero confidence when nothing was measured. It produced no value in v0.4.0, so there is no number to reproduce.
- The measurement framework matches self-reference and identity words as tokens ("this is fine" is not a self-reference), its Phi metric is documented as a spectral proxy, and `measure_consciousness` no longer imports PyPhi. No published number used it.
- DCM: the benchmark's result history is capped at 50 and `get_latest()` reloads the newest saved result; the evaluator's cache is keyed by its inputs and its model id comes from `Settings.ANTHROPIC_MODEL`; the reference adapter reads module statistics instead of constants. No runner reads any of it.
- Substrates: the SNN's membrane and trace decay are computed in seconds per step (`SNNConfig.dt`, default 1e-3; dt = 1.0 reproduces the v0.4.0 network, which was silent), the SNN reports population synchrony and STDP statistics, the HTM scores anomaly against the predictions made before the step (v0.4.0 read about 1.0 every step) and reports None before any input, the oscillator restore rejects non-finite phases and couplings, the LSM readout persists, and protocol timestamps are timezone-aware UTC. Neither runner runs a substrate, so there is no runner delta, and `SNNConfig.dt` gets no schema row because no runner reads it.
- The neural orchestrator imports the measurement framework from `mtc.assessment.framework`, reports its optional components as missing instead of silently absent, awaits the continuous-thought stop on shutdown and keeps the latest substrate signals. No runner delta.

The release gate added these, none of which a published number reads:

- The legacy 14-metric measurement framework (`mtc.assessment.framework`) is repaired and kept. It returned Phi 0.0 for every input, NaN-derived values, a constant confidence of 0.5 and a fixed ten percent interval; its metrics are now Optional, the Phi entry is a spectral integration proxy over (time, units) arrays with at least three rows (None for a one-dimensional snapshot, and exported as `spectral_integration_proxy`), the four metrics that need an HTM array are None without one, a confidence interval exists only with more than 30 historical values, and the analysis averages only the measured metrics and labels the rest `not_measured`. `measure_consciousness` loses its `input_data` parameter and the native HTM path. No runner or assessment number comes from this framework.
- The legacy 14-indicator panel's global workspace indicator reports a missing input, because its token-attention vector has no producer, and is left out of the legacy aggregate and its coverage; the panel's level and coverage move accordingly.
- The DCM reference adapter no longer reports four indicators derived from the configured planning horizon, reports the informational bottleneck only when there were broadcasts and the coverage lies strictly between 0 and 1, and omits the binding indicators when no binding was measured (and when the binding layer's zero-stimulus floor leaves less than 0.1 headroom) instead of copying the workspace coverage into them.
- The spectral Phi proxy rejects a connectivity matrix that is not square, does not match the state or holds non-finite values: integrated_information then reports insufficient_data with the error in `phi_error`, where an integration term of 0.5 or a density fallback was substituted. Both runners pass square, finite connectivity.
- The production liquid-state machine reports no Lyapunov estimate or edge-of-chaos score for fewer than ten states (it read a perfect 1.0), reports no valence or arousal until its readout was trained or restored (the raw `readout_mean` and `readout_std` are new keys), and seeds the reservoir library at construction instead of at import.
- `ProductionLSM.train_ridge` called a method the reservoir library does not have and raised; it now fits the ridge readout, which then drives the live output and the valence and arousal fields until `train_rls` is called. Both readouts and the name of the live one persist, and a restored readout that never saw a training sample does not count as trained.
- The active inference report is empty on a fresh module: the trend sentence needs more than ten prediction errors, the urgent-need sentence a positive urgency and the free-energy sentence an inference that ran, and the low branch reads "Low free energy." with no accuracy claim.
- The workspace's integration level is None without two usable signals (a constant vector is dropped, not fatal), where it read 0.0 without signals and 0.5 for one; the observation channel and the homeostasis activity still receive 0.0 for it. No runner passes neural signals.
- PAD: the label reads Neutral inside a pleasure band of 0.1 around zero, and `PADAffectModel.compute(sources, measured=...)` with a list of measured inputs renormalises each dimension over them and returns None when a dimension has none. Called without the list the arithmetic is unchanged, and both runners keep `cycle.pad_from_state` off, so only the live pipeline's coordinate changes.
- The perceptual reality monitor's discrimination uses only attribution confidences a caller supplied and is None otherwise; the 0.9 class prior (`DEFAULT_ATTRIBUTION`) is gone, so the pipeline's `prm_discrimination` is None until an attribution source exists.
- Memory: the simple consolidation pass writes the group-size strength with no confidence and no consolidation strength, a missing semantic strength or confidence reads as None instead of 0.5, and both copies of the episodic DDL drop the defaults of 0.5 on semantic strength and confidence and of 0 on consolidation strength (this is the first public release of the schema, so no installed base needs a migration); an unmeasured concept importance is not written.
- Gists and vectors: a gist's average salience, importance and retention are Optional, a vector point's emotional weight is the measured emotion intensity or None (it was the importance divided by ten), and a gist whose retention is unknown is never expired, never returned under a `min_retention` floor and computed by the next retention refresh, which now walks every stored gist.
- The core layer's somatic valence is None when no somatic marker exists, so the PAD extractor no longer lists it as a measured input with the value 0.0.
- The extended layer reports narrative coherence, temporal horizon, episodic resonance and identity continuity as None when nothing measures them (no 0.3 constant and no 0.3 base); `autobiographic_continuity` is None in that case.
- An indicator that raises is counted under `failures` (`indicator:<name>`), carries `failed: true` in its evidence and is re-raised when `DEBUG` is on; the report and `mtc assess --json` carry the new top-level `failures` mapping, empty on a clean run. The `v04_constants` path is unchanged, and no indicator raises in either published run.
- Four fixes reach `mtc assess` paths that no fixture covers; each is described under "Fixes without a switch outside the published runs" in the Phase 9 measurements section: the restored causal emergence result under `--state-dir`; the restored PhiTracker that keeps its constructor's settings under `--state-dir`; quick mode handing the assessment the workspace's PhiTracker, so integrated_information reads insufficient_data instead of absent; and the hyper-model's recursive confidence, None until a meta-error has been observed.

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
| v0.5.0 defaults after Phase 9 (no preset; the committed fixture) | 0.017317 | 0 |

Per theory, baseline and `legacy_v04` give FEP 0.072995, GWT 0.052174, RPT
0.030645, AE 0.004150, IIT 0.005406 and zero for AST, HOT and BLT. The v0.5.0
defaults after Phase 9 give FEP 0.042338, IIT 0.034115, RPT 0.008566, AE
0.002917 and zero for AST, HOT, GWT and BLT, with status partial and coverage
0.800 (six indicators report insufficient_data instead of a stand-in). The
Phase 1 defaults gave FEP 0.171983, GWT 0.036364, RPT 0.033479, HOT 0.007544,
IIT 0.005406 and zero for AE, AST and BLT.

The seeded assessment CLI (`scripts/run_assessment.py --full --json --seed 42`),
before and after this release's fix round:

| Configuration | overall_score | passing_count | status | coverage | confidence |
| --- | --- | --- | --- | --- | --- |
| `--preset legacy_v04` (`tests/fixtures/baseline_legacy_assessment.json`, unchanged) | 0.333844 | 13 | partial | 0.833 | 0.480 |
| v0.5.0 defaults after Phase 3 | 0.380308 | 14 | partial | 0.633 | 0.337 |
| v0.5.0 defaults after Phase 9 (the committed fixture) | 0.275203 | 11 | partial | 0.533 | 0.295 |

After Phase 2 the legacy preset still lands on the baseline at every theory score
(FEP 0.072995, GWT 0.052174, RPT 0.030645, AE 0.004150, IIT 0.005406, zero for
AST, HOT and BLT), and the v0.5.0 defaults give IIT 0.005406 and RPT 0.003695 with
every other theory at zero. The scores are normalized against a noise baseline
from the same synthetic warmup, so a low score means the runner's activity is not
separable from that noise here; it is a statement about this warmup, not a
verdict on the system.

One-at-a-time runs of the Phase 1 switches (the Phase 1 defaults with a single
switch set to its v0.4.0 value) show that `fep.hpp_hidden_dim_multiplier = 2` alone reproduces the
baseline exactly. Every other switch, set alone to its v0.4.0 value, left every theory score
unchanged in this runner. A flat runner result is a statement about this
synthetic warmup only, not evidence that a switch is inert elsewhere.

## Phase 3 measurements

The two tables of this section are the Phase 3 record: their v0.5.0 columns
are the defaults of that phase, superseded by the regenerated numbers in the
Phase 9 measurements section below. The experiment runner (seed 42, 25 warmup
cycles, `experiments/params.toml` applied in every run), every printed line:

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
holds every metric line of the runner at print resolution (every `name: value`
line after the `---` marker as a string; only the two timing lines are left
out) and every field of the legacy seeded assessment report against
`tests/fixtures/baseline_legacy_assessment.json` (statuses, counts, keys and
booleans exactly, floats within 1e-9; the three wall-clock readouts named
under Reproducing v0.4.0 left out). The defaults are recorded in
`tests/fixtures/baseline_v050_experiment.txt` and
`tests/fixtures/baseline_v050_assessment.json` and held by
`tests/test_v050_baseline.py` at the same bounds with nothing left out;
`tests/test_run_experiment.py` and `tests/test_run_assessment.py` hold two-run
determinism over every line and field. With PyPhi absent the tracker path
falls back to the calculator (`phi_source_used` reads
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
The Phase 3 defaults of this table are superseded: the fixture
`tests/fixtures/baseline_v050_assessment.json` now holds the Phase 9
regeneration (eight `insufficient_data` indicators and six absent, coverage
0.533, in the Phase 9 measurements section). In the Phase 3 run five
indicators reported `insufficient_data` after five cycles: the error trend
needs ten prediction errors, the PhiTracker needs more transitions than five
cycles record and there are no neural states to fall back on, and the three BLT
indicators read a beautiful loop that the standalone CLI never runs a conscious
moment through (its `cycle_count` is 0). Under `v04_constants` the same five
scored 0.0 (or 0.229 for the error trend) and were averaged in. The legacy
run's 13 passes include causal_emergence, which passes as an absent analyzer
(score 0.0 at threshold 0.0 under the inclusive v04 rule), exactly as the
v0.4.0 fixture does; none of the Phase 3 defaults' 14 passes is an absent
indicator. Under the defaults AE drops from 0.564 to 0.157 because `agency` and `embodiment` lose
their presence floors (raw 0.384 and 0.198 against 0.687 and 0.790) and
`embodiment` reads the typed voluntary ratio of 0.5 instead of the
strength-based 1.0, and AST drops from 0.621 to 0.280 because
`attention_control` reads that same typed ratio (0.449 instead of 1.000); both
of those ratios are Phase 2 switches (`ast.provenance_mode`,
`runner.warmup_profile`) at work, not Phase 3 ones.

### One switch at a time

The one-switch table now lives in the Phase 9 measurements section: it was
re-measured from the Phase 9 defaults for the nine Phase 3 switches, for the
ten Phase 9 rows of the first regeneration (the eleventh, `gwt.recency_clock`,
was measured at the second), for `damasio.circadian`, for the
`blt.loop_quality_formula` pair and for `hot.introspection_cooldown`, and for
nothing else: the Phase 1 and Phase 2 switches of the Switches table were not
run one at a time in Phase 9. Of the Phase 3 rows,
`assessment.causal_emergence_threshold = 0.0` is no longer inert on its own:
under the Phase 9 defaults causal_emergence normalizes to 0.024, which is above
a threshold of 0.0 under the strict rule, so that run prints `passing_count` 1.

### Indicators that moved by more than 0.01 (regenerated in Phase 9)

Normalized scores of the experiment runner, `--preset legacy_v04` against the
v0.5.0 defaults of the committed fixture, with the raw score and the five-seed
noise baseline. Six indicators that v0.4.0 scored as numbers are reported as
unmeasured under the defaults and leave the averages (coverage 0.800, 24 of
30); they are listed first.

| Indicator | legacy_v04 | v0.5.0 defaults | Switch and mechanism |
| --- | --- | --- | --- |
| global_broadcast | 0.000 (raw 1.000, noise 1.000) | insufficient_data (`global_workspace.broadcast.coverage_ratio`) | `gwt.legacy_broadcast_receivers`: with no acknowledgement-only receivers nothing measures the coverage. |
| ignition_dynamics | 0.130 (raw 0.904, noise 0.890) | insufficient_data (`global_workspace.ignition.sustain_rate`) | `gwt.legacy_sustain_rate`: no sustain check runs in the runner, so the rate is none instead of 1.0. |
| global_ignition_nuanced | 0.130 (raw 0.920, noise 0.908) | insufficient_data (the same statistic) | `gwt.legacy_sustain_rate`. |
| recurrent_processing | 0.130 (raw 0.920, noise 0.908) | insufficient_data (`global_workspace.broadcast.coverage_ratio`) | `gwt.legacy_broadcast_receivers`, through the GWT term of the v05 body. |
| attention_schema | 0.000 (raw 0.500, noise 0.500) | insufficient_data (`attention_schema.prediction_accuracy`) | `assessment.missing_signal_policy`: the accuracy is None until a prediction has been scored, and the runner's warmup scores none. |
| bayesian_binding_quality | 0.000 (raw 0.999, noise 0.999) | insufficient_data (`beautiful_loop.average_binding_quality`) | `blt.binding_evidence`: the warmup's winners carry no numeric vector, so nothing is bound. |
| affect_coherence | 0.438 (raw 0.991, noise 0.985) | 0.106 (raw 0.155, noise 0.054) | `pad.coherence_method` together with `assessment.saturated_baseline_min_range`, as in Phase 3; the autocorrelation reads 0.155 against 0.054 here where the Phase 3 defaults read 0.093 against 0.042, because `blt.binding_evidence` moved the shared random stream. |
| irreducibility | 0.016 (raw 0.754, noise 0.750) | 0.079 (raw 0.857, noise 0.845) | `assessment.irreducibility_method`, unchanged since Phase 3: seed position under a different statistic. |
| causal_emergence | 0.000 (raw 0.787, noise 0.793) | 0.024 (raw 0.708, noise 0.701) | `causal_emergence.tpm_row_order` (the legacy order reads 0.787 against 0.793) plus the Phase 9 node signals of the PhiTracker's module graph: with `hot.overall_confidence_source` or `blt.binding_evidence` alone at its v0.4.0 value the defaults read 0.000 again (0.731 against 0.731, or 0.717 against 0.719; the Phase 3 record read 0.810 against 0.810). Seed position either way. |
| organized_percepts | 0.000 (raw 0.305, noise 0.306) | 0.015 (raw 0.804, noise 0.801) | `assessment.indicator_formulas` (the v05 body reads the bottleneck's real keys), the margin moved from 0.009 to 0.015 by the shared random stream. |
| complex_agency | 0.000 (raw 0.250, noise 0.250) | 0.148 (raw 0.571, noise 0.497) | `assessment.indicator_formulas` (the v05 body reads the agent's real policy count and variances) with `blt.binding_evidence`: under the Phase 3 defaults the same body read 0.496 against 0.512 and scored 0.000; the shared random stream moved it. |

Every other indicator moved by less than 0.01 (prediction_error_minimization
0.000 to 0.000 with the raw score 0.254 to 0.364 against 0.258 to 0.472,
local_recurrence 0.013 to 0.015, algorithmic_recurrence 0.010 to 0.005, agency
0.008 to 0.006). The theory lines follow: GWT 0.052174 to 0.000000 is the two
ignition indicators leaving the average while the rest read 0.0; RPT 0.030645 to
0.008566 is recurrent_processing leaving and organized_percepts entering; FEP
0.072995 to 0.042338 is affect_coherence down and complex_agency up; IIT 0.005406
to 0.034115 is irreducibility and causal_emergence; AE 0.004150 to 0.002917 is
agency.

`passing_count` 2 to 0: the legacy preset passes affect_coherence (0.438 against
its 0.3 threshold) and causal_emergence (normalized 0.000 against a threshold of
0.0 under the inclusive v04 pass rule); the defaults pass nothing.
affect_coherence drops below its threshold through `pad.coherence_method`,
causal_emergence stops passing through `assessment.indicator_formulas` (the
strict v05 rule) together with `assessment.causal_emergence_threshold` (0.5),
and prediction_error_minimization, the one pass of the Phase 3 defaults (0.434),
sits inside its noise baseline once `blt.binding_evidence` stops drawing from
the shared stream (raw 0.364 against 0.472).

### Noise saturation under the defaults (the check waived, provisional)

The re-baseline plan required confirming that no indicator is `noise_saturated`
under the defaults, and stopping to report if a default inflated a noise score.
Both conditions fired in Phase 3. The decision of 2026-10-06, provisional until
the maintainer confirms it: the check is waived for the experiment runner in v0.5.0 and the fixtures are
kept as regenerated. The guard already scores the saturated indicators 0.0 as
uninformative, `tests/assessment/test_recalibration.py` passes under the
defaults (at most 3 indicators pass on noise, no perfect score, overall below
0.15), and the alternative (changing the higher-order thought the warmup hands
the loop in `scripts/run_experiment.py` and dropping the two run-length terms of
temporal_prediction_depth in `mtc/assessment/assessment.py`) is a research
change outside a release gate. The cost is that these indicators stay
unmeasurable in the experiment runner until a later phase feeds it structured
content whose noise baseline is not the content itself.

The list, rewritten from the regenerated run (Phase 9): five indicators carry
`noise_saturated: true` under the defaults in this runner (their five-seed noise
baseline leaves less than the 0.1 headroom of
`assessment.saturated_baseline_min_range`), and one carries `floor_saturated`
(the oscillatory binding layer's own zero-stimulus order parameter leaves no
headroom). global_broadcast and bayesian_binding_quality, which Phase 3
listed, now report insufficient_data instead (unmeasured, not saturated).

| Indicator | legacy_v04 noise | v0.5.0 noise | Why the floor is where it is |
| --- | --- | --- | --- |
| higher_order_representations | 1.000 | 1.000 | Same floor in v0.4.0, which already normalized to 0.0. |
| hierarchical_prediction | 1.000 | 1.000 | Same floor in v0.4.0. |
| agent_environment_coupling | 0.980 | 0.980 | Same floor; unflagged under the legacy preset only because its headroom guard is 0.0. |
| genuine_implementation | 0.667 | 1.000 | Raised by `assessment.indicator_formulas`. The v05 body reads the loop's real `peak_depth` where the v04 body read a dead key that always gave 0 (two of three conditions, hence 0.667). The runner's warmup hands the loop one AWARENESS higher-order thought per cycle, which the metacognition module phrases as "I notice that i am perceiving ..." and the epistemic depth tracker counts as a self-referential thought, so `peak_depth` is 3 on every seed (the third condition, `has_meta_prediction_depth`); and the depth tracker's strange-loop verdict, which reaches the indicator through `field_evidencing_count`, reads the runner's stand-in self-model, a constant 0.5, so it is true on any input under either `blt.strange_loop_test`. |
| temporal_prediction_depth | 0.860 | 0.917 | Raised by `assessment.indicator_formulas`. 0.4 of the v05 formula is belief and free-energy history depth (0.2 x min(1, length / 20) each), which any 20-cycle run fills, and `experiments/params.toml` pins the planning horizon at 10, so the horizon term reads 0.6 x (1 - average error). The v04 body read (1 - average error) alone. |
| oscillatory_binding_coherence (floor) | 0.997 | 0.000 (zero-stimulus order parameter 0.9987) | The static coupling synchronizes the 30 oscillators with no input; the v05 formula reads 0.0 flagged `floor_saturated` and the legacy preset's zero headroom reads the old 0.997 unflagged. |

The two raised floors (genuine_implementation and temporal_prediction_depth)
were the gate's trigger in Phase 3 and are unchanged in Phase 9. In both cases
the v05 formula reads a true statistic that a dead key or a narrower formula hid
before, and in both cases the raw seed-42 score sits at or below the noise mean,
so neither indicator gains a normalized score: all six read 0.0, and setting the
guard to its v0.4.0 value of 0.0 leaves every printed line of the regenerated
run unchanged (0.017317, 0 passing). What changed is that two indicators which
could register in this runner under v0.4.0 can no longer register in it under
the defaults: genuine_implementation because the warmup's own higher-order
thought satisfies the depth condition and the stand-in self-model makes the
strange-loop verdict, and so the field evidencing, true on any input, and
temporal_prediction_depth because 0.4 of its range is filled by run
length alone.

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

## Phase 9 measurements

The fix round of v0.5.0 (Phase 9) put eleven more corrections behind switches
(the Phase 9 rows of the Switches section) and widened the reach of four
existing ones. Every v0.5.0 number was regenerated from the tree twice, both
times in the environment of `tests/fixtures/baseline_v050_env.txt` (python
3.14.3 arm64 Darwin, numpy 2.4.2, scipy 1.17.1, torch 2.10.0, PyPhi
1.2.1.dev1562, the pymdp 0.0.7 release as distribution inferactively-pymdp:
unchanged since Phase 3): once after the fix round, and a second time after
two corrections of the fix round's own review (see "The second
regeneration"). Two runs of each default runner agree on every printed line
and on every report field with nothing left out; that determinism rests on
the two runner pins (`hot.introspection_cooldown` 0.0 and
`gwt.body_metrics_enabled` off) and, since the second regeneration, on the
simulated recency clock (`gwt.recency_clock`), which leaves no wall-clock
read on the default path (see "What the cooldown pin removes"). The legacy
numbers did not move: the legacy experiment prints every v0.4.0 line and the
legacy seeded assessment equals its fixture on every field. "Before" below is
the Phase 3 record (the fixtures committed before the fix round); "after" is
the committed fixture. Where the two regenerations differ (one evidence value
under 1e-9 and one evidence key name, nothing printed or scored) the second
is the one quoted.

### The second regeneration

The v0.5.0 fixtures were regenerated a second time on 2026-10-18, after the
review of the fix round itself, for two causes and nothing else:

- `gwt.recency_clock`: the attention bottleneck's recency bonus is measured on
  the workspace's simulated clock (the default, `simulated`) instead of on the
  wall clock since the candidate was created (`v04_wall`, which the legacy
  preset sets); the clock read behind the first regeneration's one residual
  term (see "What the cooldown pin removes").
- the evidence key `has_strange_loop` of genuine_implementation became
  `has_meta_prediction_depth` under the v05 formula, because the condition it
  reports is the depth tracker's `peak_depth` >= 2, not the tracker's
  strange-loop verdict (value true, unchanged; the v04 body and the legacy
  fixture keep `has_strange_loop`).

The procedure was the one of the first regeneration: two default experiment
runs and two default seeded assessments, compared with nothing excluded, agree
(every printed line; the two reports are bit identical apart from the run id,
the timestamp and the timing fields), the fixtures were written from the first
run, and the environment record did not change. Against the first
regeneration's fixtures: every printed line of the experiment runner is
unchanged; in the seeded assessment one evidence value moved,
`genuine_implementation.evidence.avg_loop_quality` 0.5300177410248634 to
0.5300177409697049 (5.5e-11, under the 1e-9 comparator), and one key was
renamed; no score, status, count or confidence moved (0.275203, 11 of 30,
partial, coverage 0.533, confidence 0.295). With every Python clock 25 times
faster the default seeded assessment is bit identical to the regenerated
fixture apart from the run id, the timestamp and the timing fields. The legacy
proof was repeated with `gwt.recency_clock = "v04_wall"` in the preset: the
legacy experiment prints every line of
`tests/fixtures/baseline_v040_experiment.txt` (0.031247, 2 passing; the three
added lines measured, 1.000 and calculator_spectral_proxy) and the legacy
seeded assessment equals `tests/fixtures/baseline_legacy_assessment.json` on
every field, the three wall-clock readouts included on that run (11, 2.2 and
11). The one-switch and cumulative proofs below are the first regeneration's
and were not rerun: the eleventh row moves the one evidence value measured
here and nothing else.

### Headline numbers

| Run | Before Phase 9 | After Phase 9 |
| --- | --- | --- |
| experiment runner, `--preset legacy_v04` | 0.031247, 2 passing, measured, coverage 1.000, calculator_spectral_proxy | unchanged |
| experiment runner, defaults | 0.021675, 1 passing, measured, coverage 1.000, phi_tracker_iit4 | 0.017317, 0 passing, partial, coverage 0.800, phi_tracker_iit4 |
| seeded assessment, `--preset legacy_v04` | 0.333844, 13 of 30, partial, coverage 0.833, confidence 0.480 | unchanged |
| seeded assessment, defaults | 0.380308, 14 of 30, partial, coverage 0.633, confidence 0.337 | 0.275203, 11 of 30, partial, coverage 0.533, confidence 0.295 |

### The experiment runner, every printed line

Seed 42, 25 warmup cycles, `experiments/params.toml` applied in every run. The
last column names the one switch which, set alone to its v0.4.0 value on top of
the defaults, prints the "before" string again.

| Line | Before Phase 9 | After Phase 9 | Restored by |
| --- | --- | --- | --- |
| overall_score | 0.021675 | 0.017317 | no single switch: it is the weighted mean over the lines below. With the ten Phase 9 rows of the first regeneration at their legacy values together it prints 0.022394, which is the "before" mean without attention_schema (see the cumulative proof). |
| passing_count | 1 | 0 | `blt.binding_evidence`: prediction_error_minimization passes again (0.434 against its threshold of 0.4). `assessment.causal_emergence_threshold = 0.0` also prints 1, through a different pass: causal_emergence reads 0.024 under the defaults, above a threshold of 0.0 under the strict rule. |
| assessment_status | measured | partial | `assessment.missing_signal_policy` or `assessment.indicator_formulas` (see coverage). |
| coverage | 1.000 | 0.800 | six indicators read insufficient_data: global_broadcast and recurrent_processing (`gwt.legacy_broadcast_receivers` alone: 0.867), ignition_dynamics and global_ignition_nuanced (`gwt.legacy_sustain_rate` alone: 0.867), bayesian_binding_quality (`blt.binding_evidence` alone: 0.833) and attention_schema (its prediction accuracy is None until a prediction has been scored, reached only through `assessment.missing_signal_policy`). Either reused assessment switch alone reads all six as numbers again (1.000, measured). |
| phi_source_used | phi_tracker_iit4 | phi_tracker_iit4 | unchanged (`assessment.phi_source = "calculator"` prints calculator_spectral_proxy and moves no other line). |
| ae_score | 0.000000 | 0.002917 | `blt.binding_evidence` (agency 0.006 through the shared random stream). |
| ast_score | 0.000000 | 0.000000 | unchanged: attention_control reads 0.0 and attention_schema left the average. |
| blt_score | 0.000000 | 0.000000 | unchanged: every BLT indicator sits inside its noise baseline or is unmeasured. |
| fep_score | 0.081411 | 0.042338 | `blt.binding_evidence`: prediction_error_minimization 0.434 to 0.000 (raw 0.684 to 0.364 against a noise mean of 0.441 to 0.472), affect_coherence 0.054 to 0.106 and complex_agency 0.000 to 0.148, all through the shared random stream (the v0.4.0 binding path drew from it on every inference; the default path draws nothing). |
| gwt_score | 0.000000 | 0.000000 | unchanged: two GWT indicators left the average and the others read 0.0. `gwt.legacy_sustain_rate` alone prints 0.039474, not the Phase 3 0.000000: the two ignition indicators are measured again at 0.079 each (18 ignitions in 25 cycles, raw 0.832 against a noise baseline of 0.818, where the Phase 3 run ignited 16 times against 0.808), because `blt.binding_evidence` moved the shared random stream under them. |
| hot_score | 0.000000 | 0.000000 | unchanged. |
| iit_score | 0.026167 | 0.034115 | `hot.overall_confidence_source` or `blt.binding_evidence`, either alone: causal_emergence normalizes to 0.024 under the defaults (raw 0.708 against 0.701) and to 0.000 with either switch alone at its v0.4.0 value (raw 0.731 against 0.731 under the constant confidence, 0.717 against 0.719 under the legacy binding; the Phase 3 record read 0.810 against 0.810), because the analyzer reads the PhiTracker's module-state transitions and under the defaults the tracker's HOT node reads a measured feeling of knowing and its BLT node the loop quality (binding unmeasured). Seed position either way; irreducibility is unchanged at 0.079. |
| rpt_score | 0.005109 | 0.008566 | no single switch: recurrent_processing left the average (`gwt.legacy_broadcast_receivers` alone prints 0.016029) and the three measured RPT indicators moved through the shared random stream (`blt.binding_evidence` alone prints 0.005452). The two together print 0.010725 (overall 0.025460, coverage 0.900); with `damasio.first_delta = "v04_zero"` added, the three together print 0.005109 (overall 0.025691), which is what the cumulative run prints. `damasio.first_delta` alone prints 0.009301 (0.008566 under the defaults): the first delta reaches the RPT indicators through the protoself stability and the competition (see the note at the end of this section). |

### The seeded assessment, every indicator that moved

`scripts/run_assessment.py --full --json --seed 42`, five warmup cycles, the
standalone modules without RPT, PAD, the analyzer or neural states.

| Indicator | Before Phase 9 | After Phase 9 | Switch |
| --- | --- | --- | --- |
| global_broadcast | measured 1.000, pass | insufficient_data (`global_workspace.broadcast.coverage_ratio`) | `gwt.legacy_broadcast_receivers` (no acknowledgement-only receivers) |
| recurrent_processing | measured 0.732, pass | insufficient_data (the same statistic) | `gwt.legacy_broadcast_receivers` (no acknowledgement-only receivers) |
| ignition_dynamics | measured 0.626, pass | insufficient_data (`global_workspace.ignition.sustain_rate`) | `gwt.legacy_sustain_rate` (no sustain check ran) |
| global_ignition_nuanced | measured 0.724, pass | insufficient_data (the same statistic) | `gwt.legacy_sustain_rate` (no sustain check ran) |
| attention_schema | measured 0.110 | insufficient_data (`attention_schema.prediction_accuracy`) | `assessment.missing_signal_policy` (no new row: the accuracy is None until a prediction has been scored) |
| epistemic_depth | insufficient_data (`beautiful_loop.cycle_count`) | measured 0.385, pass (raw 0.45, noise 0.106) | `runner.assess_blt_source`: the assessment reads the workspace's loop, which the warmup drives |
| genuine_implementation | insufficient_data | measured 0.615, pass (raw 0.667: loop quality and the meta-prediction depth, `has_meta_prediction_depth`; no field evidencing, so the strange-loop verdict, false with basis insufficient_data on every cycle, does not reach it) | `runner.assess_blt_source` |
| bayesian_binding_quality | insufficient_data (`beautiful_loop.cycle_count`) | insufficient_data (`beautiful_loop.average_binding_quality`: the winners carry no vector) | the missing statistic changes through `runner.assess_blt_source`; `blt.binding_evidence` measures it (0.999) |
| metacognition | 0.605 (raw 0.700) | 0.473 (raw 0.600) | `hot.overall_confidence_source` and `hot.introspection_counting` together: the raw score is 0.6 x confidence + 0.4 x activity; the constant 0.5 alone gives 0.394, the per-thought count alone 0.684, both together 0.605 |
| online_reportable_learning | 0.451 (raw 0.50) | 0.319 (raw 0.38) | `hot.introspection_counting` (the body reads the introspection count) |
| sparse_smooth_coding | 0.815 | 0.722 | `blt.binding_evidence` (the shared random stream: the inference error history) |
| temporal_prediction_depth | 0.332 | 0.318 | `blt.binding_evidence` (the same) |
| agency | 0.219 | 0.209 | `blt.binding_evidence` (the same: the free energy) |
| sequential_module_querying | 0.335 (3.8 candidates) | 0.378 (4.2 candidates) | `blt.binding_evidence` (the same: the candidates the warmup draws) |
| complex_agency | 0.524, pass | 0.096 | `blt.binding_evidence` (the same: the free-energy and prediction variances) |
| organized_percepts | 0.393, `avg_local_recurrence` 0.0, `perceptual_candidates` 13 | 0.393, `avg_local_recurrence` None, `perceptual_candidates` 14 | evidence only: `avg_local_recurrence` is None without an SNN or LSM state, under `assessment.missing_signal_policy` (`mtc assess` builds no RPT); the candidate count is `blt.binding_evidence` (the shared random stream: the candidates the warmup draws) |
| irreducibility | absent, `noise_baseline` 0.8198 | absent, `noise_baseline` 0.7990 | evidence only: `blt.binding_evidence` (the shared random stream reaches the five-seed baseline of an indicator that is itself absent) |

Report level: `theory_scores.BLT` appears (0.500) through
`runner.assess_blt_source`; `theory_scores.IIT` (0.0) is omitted
through `assessment.missing_signal_policy` (a theory with no measured member:
integrated_information insufficient, irreducibility and causal_emergence
absent); AST rises from 0.280 to 0.449 because attention_schema left its
average; GWT 0.537 to 0.189 needs both `gwt` switches and the binding evidence;
HOT 0.718 to 0.628 needs the two `hot` switches and the binding evidence; RPT
0.225 to 0.098 is recurrent_processing; FEP 0.336 to 0.247 and AE 0.157 to
0.152 are the binding evidence. `passing_count` 14 to 11: four passes left the
averages (the two coverage and the two sustain indicators), complex_agency fails, epistemic_depth and
genuine_implementation pass. The three counter readouts
(`higher_order_representations.evidence.hots_generated` and `coverage_ratio`,
`metacognition.evidence.introspections`) read 14, 2.8 and 3 on every machine
now: the cooldown pin (3 is the `episodes` count of the defaults; the ten-row
run below, which counts per thought, reads 15).

### One switch at a time

Each row sets one switch to its v0.4.0 value on top of the defaults: for the
experiment runner a params file copied from `experiments/params.toml` with the
one key added, for the seeded assessment a params file holding the one key.
The experiment columns give overall_score, passing_count, assessment_status and
coverage, then the theory lines that moved; the assessment columns give
overall_score, passing_count and coverage, then the indicators whose status or
score moved. "identical" means no printed line, or no field of the report,
differs from the default run.

| Switch at its v0.4.0 value | Experiment runner | Seeded assessment |
| --- | --- | --- |
| (defaults) | 0.017317, 0, partial, 0.800, phi_tracker_iit4; ae 0.002917, fep 0.042338, iit 0.034115, rpt 0.008566, the rest 0.000000 | 0.275203, 11, 0.533; AST 0.449, HOT 0.628, FEP 0.247, RPT 0.098, AE 0.152, BLT 0.500, GWT 0.189 |
| `assessment.noise_baseline_matching = "v04_any"` | identical | identical |
| `runner.assess_blt_source = "standalone"` | identical (the runner does not read it) | 0.248145, 9, 0.467; epistemic_depth and genuine_implementation insufficient_data, BLT omitted |
| `gwt.legacy_broadcast_receivers = true` | 0.017033, 0, partial, 0.867; rpt 0.008566 to 0.016029 | 0.340767, 13, 0.600; global_broadcast 1.000 and recurrent_processing 0.732 measured again; GWT 0.459, RPT 0.225 |
| `gwt.legacy_sustain_rate = true` | 0.023765, 0, partial, 0.867; gwt 0.000000 to 0.039474 | 0.321491, 13, 0.600; ignition_dynamics 0.626 and global_ignition_nuanced 0.724 measured again; GWT 0.432 |
| `gwt.refractory_clock = "per_cycle"` (the default) | identical | identical |
| `gwt.recency_clock = "v04_wall"` (added at the second regeneration; measured as the first regeneration's fixtures, which were produced under the wall-clock bonus, against the second) | identical | identical scores and statuses (one evidence value: genuine_implementation `avg_loop_quality`, 0.5300177409697049 to 0.5300177410248634, 5.5e-11) |
| `hot.overall_confidence_source = "v04_constant"` | 0.016063, 0, partial, 0.800; iit 0.034115 to 0.026167 | 0.272359, 10, 0.533; metacognition 0.473 to 0.394; HOT 0.609 |
| `hot.introspection_counting = "v04_per_hot"` | identical | 0.287100, 11, 0.533; metacognition 0.473 to 0.684, online_reportable_learning 0.319 to 0.451; HOT 0.714 |
| `blt.binding_evidence = "v04_salience_posterior"` | 0.026982, 1, partial, 0.833; ae 0.002917 to 0.000000, fep 0.042338 to 0.081411, iit 0.034115 to 0.026167, rpt 0.008566 to 0.005452 | 0.328068, 13, 0.567; bayesian_binding_quality 0.999 measured, sparse_smooth_coding 0.815, complex_agency 0.524, sequential_module_querying 0.335, temporal_prediction_depth 0.332, agency 0.219; FEP 0.336, BLT 0.666, HOT 0.652, GWT 0.168, AE 0.157 |
| `blt.strange_loop_test = "v04_confidence_match"` | identical | identical |
| `damasio.first_delta = "v04_zero"` | 0.017488, 0, partial, 0.800; ae 0.002917 to 0.002660, fep 0.042338 to 0.040530, iit 0.034115 to 0.037811, rpt 0.008566 to 0.009301 | identical scores and statuses (one evidence value: genuine_implementation `avg_loop_quality`) |
| `assessment.indicator_formulas = "v04"` | 0.010680, 0, measured, 1.000; ae 0.002917 to 0.011778, fep 0.042338 to 0.017706, gwt 0.000000 to 0.009726, rpt 0.008566 to 0.004442 | 0.202811, 7, 0.767; the seven unmeasured indicators read as numbers again (the v04 bodies read None as their stand-ins: 0.0 for the broadcast coverage and the sustain rate, where v0.4.0 read 1.0, and the v0.4.0 constant elsewhere) and 21 scores move; GWT 0.135, AST 0.288, HOT 0.607, FEP 0.254, RPT 0.000, AE 0.240, BLT 0.173 |
| `assessment.missing_signal_policy = "v04_constants"` | 0.015245, 0, measured, 1.000; gwt 0.000000 to 0.008626, rpt 0.008566 to 0.006852 | 0.196840, 11, 0.800; the eight insufficient indicators read their stand-ins (attention_schema 0.110, prediction_error_minimization 0.068, global_ignition_nuanced 0.035, ignition_dynamics 0.004, the rest 0.0), IIT 0.0 is listed again; GWT 0.083, AST 0.280, FEP 0.217, RPT 0.079, BLT 0.333 |
| `assessment.irreducibility_method = "spectral_entropy"` | 0.014588, 0, partial, 0.800; iit 0.034115 to 0.013354 | identical scores and statuses (irreducibility is absent; only its noise baseline evidence moves) |
| `assessment.causal_emergence_threshold = 0.0` | 0.017317, 1, partial, 0.800; passing_count 0 to 1 (causal_emergence 0.024 passes at 0.0 under the strict rule); no theory line moves | identical scores and statuses (the threshold field of the absent indicator) |
| `assessment.phi_source = "calculator"` | identical but `phi_source_used` calculator_spectral_proxy | 0.258706, 11, 0.533; integrated_information insufficient_data to absent (0.0: no neural states to fall back on) |
| `pad.coherence_method = "v04_variance"` | 0.012569, 0, partial, 0.800; fep 0.042338 to 0.024633 | identical (PAD is not built) |
| `causal_emergence.tpm_row_order = "little_endian"` | 0.018500, 0, partial, 0.800; iit 0.034115 to 0.041616 | identical (the analyzer is not built) |
| `rpt.stdp_plasticity_enabled = false` | identical | identical |
| `assessment.saturated_baseline_min_range = 0.0` | identical (oscillatory_binding_coherence reads raw 0.997 noise-saturated instead of 0.0 floor-saturated; normalized 0.0 either way) | identical |
| `damasio.circadian = "v04_sawtooth"` | identical (the seed-42 IIT line stays 0.034115) | identical |
| `blt.loop_quality_formula = "v04"` with `blt.binding_evidence = "v04_salience_posterior"` | equal to the `blt.binding_evidence` run on every printed line | equal to the `blt.binding_evidence` run on every score and status |
| `hot.introspection_cooldown = 1.0` alone (the gate the runners pin at 0.0) | 0.016633, 0, partial, 0.800; iit 0.034115 to 0.029783 | 0.269255, 9, 0.533; metacognition 0.473 to 0.367, online_reportable_learning 0.319 to 0.253; HOT 0.585; the three counter readouts 14, 2.8, 3 to 11, 2.2, 2 on this machine |

`assessment.phi_normalization`, `assessment.phi_fixed_reference`,
`pad.coherence_window` and `pad.coherence_min_history` have their v0.4.0 value
as the default, so their one-at-a-time run is the default run.

Interactions, lines that need more than one switch: the experiment's
overall_score (every mover) and rpt_score (`gwt.legacy_broadcast_receivers`,
`blt.binding_evidence` and `damasio.first_delta` together: the first two alone
print 0.010725, the three 0.005109); the seeded assessment's metacognition and HOT (the two
`hot` switches, HOT also `blt.binding_evidence`), GWT (both `gwt` switches and
`blt.binding_evidence`), overall_score, passing_count, coverage and confidence.
One line is restored by either of two switches: the experiment's iit_score
(`hot.overall_confidence_source` or `blt.binding_evidence`, each alone, through
causal_emergence).

### The cumulative reversion proof

A params file equal to `experiments/params.toml` plus the ten Phase 9 rows of
the first regeneration (`gwt.recency_clock` was not yet a row) at
their legacy values (and nothing else) run against the Phase 3 record:

- Experiment runner: every theory line, `passing_count` 1, `total_indicators`
  and `phi_source_used` print the Phase 3 strings. Three lines differ:
  overall_score 0.021675 to 0.022394, assessment_status measured to partial,
  coverage 1.000 to 0.967. All three are attention_schema reading
  insufficient_data (the prediction accuracy that is None until scored, under
  `assessment.missing_signal_policy`, no new row): recomputing the Phase 3 weighted mean without that indicator gives
  0.022394 to nine decimals, and no other indicator score, status or noise
  baseline moved.
- Seeded assessment (the ten rows alone): the same attention_schema (AST 0.280
  to 0.449, overall 0.380308 to 0.391395, coverage 0.633 to 0.600, confidence
  0.337 to 0.316), `theory_scores.IIT` omitted (no measured member) and
  `organized_percepts.evidence.avg_local_recurrence` 0.0 to None (no SNN or
  LSM state), all
  under `assessment.missing_signal_policy`; and the three counter readouts 11,
  2.2, 11 to 14, 2.8, 15, which is the runner's cooldown pin (0.0) against the
  1.0 s wall-clock gate of the Phase 3 run. No score moves with them.
- At the indicator level of the experiment (an indicator-by-indicator record
  of the run), two more residues, neither reaching a printed line:
  oscillatory_binding_coherence reads raw 0.0 and noise baseline 0.0 flagged
  `floor_saturated` with a zero-stimulus order parameter of 0.9987 where the
  Phase 3 run read 0.997 noise-saturated (the zero-stimulus floor guard under
  `assessment.saturated_baseline_min_range`; normalized 0.0 either way); and the
  three counter readouts (61, 2.44, 71 to 94, 3.76, 115), the same pin. The
  v0.4.0 binding path's `temporal_coherence` term reads the inference
  timestamps, so under `blt.binding_evidence = "v04_salience_posterior"` four
  evidence values of bayesian_binding_quality and genuine_implementation move
  in the seventh decimal between any two runs. In the experiment runner and
  under the legacy preset no score or printed line depends on them (the
  legacy seeded assessment's standalone loop never binds). With
  `blt.binding_evidence = "v04_salience_posterior"` alone on the seeded
  assessment, which reads the workspace's loop, the jitter reaches three
  scores: between two runs on this machine `bayesian_binding_quality.score`
  moved 1.9e-7 (0.9985956951 against 0.9985958843), `theory_scores.BLT`
  6.3e-8 and `overall_score` 8.3e-9, above the 1e-9 bound of the fixture
  tests, which is why no test pins that configuration.

No number moved without a switch.

### What the cooldown pin removes

Both runners pin `hot.introspection_cooldown` at 0.0 unless the overrides set
it. Set to its module default of 1.0 alone on top of the Phase 9 defaults, the
wall-clock gate reaches the scores: in the experiment runner iit_score 0.034115
to 0.029783 (overall 0.016633), and in the seeded assessment metacognition 0.473
to 0.367, online_reportable_learning 0.319 to 0.253, HOT 0.628 to 0.585 and 11
to 9 passing, besides the three counter readouts. Under `episodes` counting the
activity term of metacognition is no longer capped, so how many cycle
boundaries fall at least one second after the previous introspection, which
depends on the machine, would decide a score. The pin is what makes the v0.5.0
fixtures comparable with nothing left out. Under the legacy preset (per-thought
counting, both counters past their caps) only the three readouts follow the
clock, which is why the legacy assessment test leaves exactly those three out.

The pin removes the gate; the one other clock read on the default path was
found after the first regeneration and removed at the second. The default
seeded assessment of the first regeneration was run seven times with every
Python clock scaled (speeds 1, 1, 1, 5, 0.2, 25 and 0.04, the first run
discarded): against that fixture, with nothing excluded, every speed but 25
gave no difference at 1e-9, and speed 25 gave one,
`genuine_implementation.evidence.avg_loop_quality` 0.5300177422 against
0.5300177410 (1.2e-9); no score, status or count moved at any speed, and the
counters read 14 and 3 at every speed. The read behind it was the attention
bottleneck's recency bonus: every candidate's composite salience carries 0.1
x exp(-0.1 x age), and the age was the wall-clock time in seconds since the
candidate was created (`AttentionBottleneck._calculate_recency_bonus` in
`mtc/consciousness/enhanced_global_workspace.py`), so the winners' salience,
the self-model states the epistemic depth tracker records (their confidence
differed by 8e-6 between clock speeds 1 and 25), the hyper-model's precisions
and the loop quality followed machine speed at the scale of the microseconds
a candidate waits within a cycle: three seeded assessments in one process at
clock speeds 1, 1 and 25 read that value as 0.5300177410282785,
0.5300177410223317 and 0.5300177422498947, with 84 entries of the loop's
state differing between speeds 1 and 25. Since the second regeneration the
bonus is measured on the workspace's simulated clock since the bottleneck
first saw the candidate (`gwt.recency_clock = "simulated"`, the default): the
default seeded assessment is bit identical at clock speeds 1 and 25 and
between runs, the value reads 0.5300177409697049 (5.5e-11 below the first
regeneration's 0.5300177410248634, under the 1e-9 comparator), and no score,
status or count moved with it. `gwt.recency_clock = "v04_wall"` (the legacy
preset) keeps the v0.4.0 wall-clock arithmetic; under `v04_constant` the HOT
confidence is a constant, and the legacy seeded assessment moves no field of
its report between speeds 1 and 25 but the three cooldown readouts. The 1e-9
bound was not widened.

### Fixes without a switch outside the published runs

Four Phase 9 corrections change behaviour that no fixture covers and the legacy
preset never reads, so they carry no switch and are listed here as fixes
without a switch:

- `mtc assess --state-dir` restores the causal emergence analyzer's last
  result (partition, label and computation time, None when none was saved) and
  marks it restored, instead of dropping it.
- A restored PhiTracker keeps the constructor's `min_transitions`,
  `history_size` and `laplace_alpha`, and `is_exact` says the exact analysis ran
  on the empirical matrix, not how much of that matrix was observed
  (`tpm_coverage` reports that).
- `mtc assess` without `--full` (quick mode) hands the
  assessment the workspace's PhiTracker, so integrated_information reads
  insufficient_data (the tracker is warming up) instead of absent.
- The hyper-model's recursive confidence is None until a meta-error has
  been observed; under the recursive loop-quality formula a None drops its 0.15
  term and renormalises the rest. The experiment runner's default loop observes
  a meta-error before the first loop quality it averages, so its record is
  unchanged.

### Strange loop and first delta, where the attribution needs a note

Under the default `blt.strange_loop_test` the experiment runner's standalone
loop compares the self-model's scored self-prediction accuracy with the
ledger's forecast accuracy, but the runner's self-model is its own stand-in
(`_WarmupSelfModel` in `scripts/run_experiment.py`), whose
`self_calibration_score` is a constant 0.5. The verdict is therefore true on
any input in this runner under either test, which is part of why
genuine_implementation is noise-saturated there. The verdict reaches that
indicator only through `field_evidencing_count` (the loop counts a cycle as
field evidencing only when the verdict is true); the indicator's third
condition is `has_meta_prediction_depth`, the depth tracker's `peak_depth`
>= 2, not the verdict. In the seeded assessment the verdict is false with
basis insufficient_data on all five cycles (`meta_state.total_predictions` 4,
minimum 5), so `field_evidencing_count` is 0 there and the indicator's 0.667
is the loop quality and the depth.

`damasio.first_delta` moves four theory lines of the experiment on its own,
more than the first cycle's three values suggest: under the default the first
delta also stays out of the stability history, so the protoself stability
differs on every following cycle, and the stability feeds the PhiTracker's AST
node and, through the theory interactions, the next cycle's salience bias and
so the competition.

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
`mtc assess`, `mtc serve`, `mtc chat`, `mtc ctm`, `mtc safety` and
`mtc privacy-audit`.
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

**Settings keys that nothing read are removed.** 51 keys had no reader in the
code; each is removed, because a reader for the threshold or size families
would be a second route, outside the switches, for moving a number
(`GWT_IGNITION_THRESHOLD`, for example, was 0.4 where the runner uses 0.3).
`ABLATION_OUTPUT_DIR` stays and is now read by the ablation study. By family:

- service and paths: `SERVICE_HOST`, `ENVIRONMENT`, `MODEL_DIRECTORY`,
  `TEMP_DIRECTORY`, `DB_POOL_RECYCLE_TIME`;
- the model: `MODEL_QUANTIZATION`, `VISION_ENABLED`;
- substrate sizes: `SNN_NEURONS`, `SNN_HIDDEN_NEURONS`, `SNN_INPUT_NEURONS`,
  `SNN_OUTPUT_NEURONS`, `LSM_NEURONS`, `LSM_SPECTRAL_RADIUS`,
  `LSM_INPUT_SCALING`;
- consciousness toggles: `CONSCIOUSNESS_CYCLE_MS`, `CONSCIOUSNESS_ENABLED`,
  `META_COGNITION_ENABLED`, `SELF_REFLECTION_ENABLED`;
- indicator thresholds and the noise baseline: `GWT_BROADCAST_THRESHOLD`,
  `GWT_IGNITION_THRESHOLD`, `GWT_WORKSPACE_CAPACITY`,
  `AST_ATTENTION_THRESHOLD`, `AST_CONTROL_THRESHOLD`,
  `HOT_REPRESENTATION_THRESHOLD`, `HOT_METACOGNITION_THRESHOLD`,
  `FEP_PREDICTION_ERROR_THRESHOLD`, `FEP_HIERARCHICAL_THRESHOLD`,
  `FEP_AGENCY_THRESHOLD`, `IIT_PHI_THRESHOLD`, `IIT_IRREDUCIBILITY_THRESHOLD`,
  `RPT_LOCAL_THRESHOLD`, `RPT_GLOBAL_THRESHOLD`, `BLT_BINDING_THRESHOLD`,
  `BLT_DEPTH_THRESHOLD`, `BLT_GENUINE_THRESHOLD`,
  `CONSCIOUSNESS_SCORE_THRESHOLD`, `CONSCIOUSNESS_PASSING_RATIO`,
  `NOISE_BASELINE_ITERATIONS`, `NOISE_FLAG_THRESHOLD`, `TRACK_INDICATORS`;
- memory: `WORKING_MEMORY_DURATION`, `EPISODIC_MEMORY_RETENTION`,
  `SEMANTIC_MEMORY_PERMANENT`;
- emotion: `EMOTIONAL_MODEL`, `EMOTION_DECAY_RATE`,
  `EMOTION_AMPLIFICATION_MAX`, `EMOTIONAL_MEMORY_WEIGHT`,
  `PATTERN_RECOGNITION_ENABLED`;
- monitoring export: `PROMETHEUS_ENABLED`, `PROMETHEUS_PORT`,
  `METRICS_EXPORT_INTERVAL`.

`Settings` ignores keys it does not know, so an env file that still sets one
of these (with the `MTC_` prefix) is read without error and the value has no
effect. `tests/core/test_settings_have_readers.py` keeps every remaining field
read.

**New keys and a changed default.** `SLEEP_PRUNE_DRY_RUN` (false: night sleep
deletes the memories the pruner scores as weak; true logs what would be
removed), `SAFETY_GATE_FAIL_CLOSED` (true: an untrusted speaker's turn whose
safety check could not run ends with `safety_unavailable` and stores nothing)
and `TRUST_COMPASS_DEEP_MODE` (true: the trust compass may ask the brain for a
reflection when its keyword screen or the safety monitor's scores call for
one). `EMERGENCY_SHUTDOWN_ENABLED`
moved from true to false: an emergency-level score denies the message and
alerts the escalation role, and suspending every untrusted conversation is
opt-in, lifted with `mtc safety resume` or `POST /api/safety/resume`. See
[SECURITY.md](SECURITY.md) and [OPERATIONS.md](OPERATIONS.md).

**Secrets and the store URLs.** `repr(Settings)` and `model_dump()` leave out
the seven secret fields (`API_KEY`, `ANTHROPIC_API_KEY`, `SPEAKER_TOKENS` and
the four store passwords). `POSTGRES_URL`, `MONGODB_URL` and `REDIS_URL` are
plain properties instead of pydantic computed fields, so they leave the repr,
the dumps and `model_computed_fields`, and they percent-quote the user and
the password (`REDIS_URL` carries a password only); reading them as attributes
is unchanged.

**Removed modules.** `mtc/consciousness/global_workspace.py`, the v0.1.0
simple workspace, is removed with its `global_workspace` component row in the
neural orchestrator: nothing consulted it, and its salience used 0.5
stand-ins. `EnhancedGlobalWorkspace` is the workspace. The unused
`mtc/neural/spiking/snn_core.py` is removed (the spiking package keeps
`production_snn`, `snn_core_optimized` and `emotion_encoder`), and so is
`experiments/program.md`, stale notes for a parameter-research loop.

**Renamed recurrence fields.** `RecurrenceMetrics.phenomenal_consciousness`
and `access_consciousness` are `local_recurrence_index` and
`combined_recurrence_index`, and the `latest` entry of the RPT statistics
names them `local_index` and `combined_index` (it said `phenomenal` and
`access`). The values are unchanged; they are recurrence statistics, which the
theory associates with those notions. A state saved under the old names still
restores.

**Runner pins and schema bounds.** Both runners pin `hot.introspection_cooldown`
at 0.0 unless the overrides set it, because the cooldown is a wall-clock gate
and two runs of one seed must agree; the live runtime keeps the module's
1.0 s. The schema minimum of `hot.introspection_cooldown` is 0.0 (it was 0.1)
and that of `blt.num_levels` is 2 (it was 1; `BeautifulLoop(num_levels=1)`
raises). The refractory clock has no pin: a workspace built directly keeps the
simulated per-cycle clock and the live runtime sets `gwt.refractory_clock =
elapsed`. What the cooldown pin does to the numbers is recorded under "What
the cooldown pin removes".

**The package root stays eager.** `mtc/__init__.py` keeps its public names,
and importing `mtc` loads no torch, PyPhi, pymdp, scipy, web framework or
store client (`tests/assessment/test_torch_import_isolation.py`). Every
import of an `mtc` module still loads the assessment package.

**The `fep` extra is the minimum that runs the assessment.** The workspace
builds active inference unconditionally, and that module needs pymdp. On a
bare install `mtc assess`, `mtc serve`, `mtc chat` when it builds a runtime,
`mtc ctm`, the demo (`examples/full_pipeline_demo.py`) and the three other
examples that build a workspace each stop
with one `MISSING EXTRA` line on stderr and exit code 2; `mtc doctor` reports
the missing extra as a required failure and names the indicators that read
it.
