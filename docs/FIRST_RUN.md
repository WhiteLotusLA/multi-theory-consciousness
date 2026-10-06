# What a first run shows

A fresh install measures very little, and says so. Several measurements need
history before they read anything, and until then they report `warming_up`,
`None` or `insufficient_data` instead of a number. This page lists every such
threshold with the constant it comes from, then what a fresh
`mtc assess` and `mtc doctor` print and why, then the continuous thought
machine's cadence and quiet reasons.

A test (`tests/docs/test_first_run_doc.py`) compares every number in the table
below with its source constant in the code.

## Warmup thresholds

One cycle is one consciousness cycle of the workspace. In the live pipeline
each conversation turn runs one cycle; with `PHI_IDLE_SAMPLE_S` above 0 the
runtime also records the last cycle's module states into the PhiTracker
between turns (off by default).

| Threshold | Value | Reads before it | Source |
|---|---|---|---|
| PhiTracker transitions before any Phi | 128 | `phi.warming_up` true; `integrated_information` from the tracker is `insufficient_data` and the integration proxy is None. The first cycle records a state, each later cycle one transition, so 129 cycles | `mtc.consciousness.phi_tracker.PhiTracker` `min_transitions` |
| PhiTracker transitions the live runtime passes | 128 | same as above | `mtc.pipeline.runtime` `build_phi_tracker(..., 128)` |
| Phi values before percentile normalization | 20 | the normalized Phi readout divides by `phi_warmup_reference` instead of the rolling 95th percentile | `mtc.consciousness.phi_tracker.PhiTracker._PHI_NORM_MIN_HISTORY` |
| Phi warmup reference | 5.0 | the display denominator until 20 Phi values exist; never part of the measurement | `mtc.consciousness.phi_tracker.PhiTracker` `phi_warmup_reference` |
| Prediction errors before a prediction-error rank | 100 | `prediction_error_rank` is None and the first-run report shows `ranked: false` | `mtc.consciousness.active_inference.ActiveInferenceModule.BASELINE_MIN_SAMPLES` |
| Prediction errors before the error trend | 10 | `prediction_error_minimization` is `insufficient_data` | `mtc.assessment.assessment.ConsciousnessAssessment._v05_prediction_error_minimization` |
| Scored outcomes before a forecast accuracy | 20 | the prediction ledger's accuracy is None | `mtc.consciousness.prediction_ledger.PredictionLedger` `min_scored_for_accuracy` |
| PAD states before an affect coherence | 10 | `affect_coherence` is `insufficient_data` | `mtc.consciousness.pad_affect.PADAffectModel` `coherence_min_history` |
| PAD states in the coherence window | 20 | the coherence reads the most recent states only | `mtc.consciousness.pad_affect.PADAffectModel` `coherence_window` |
| Cycles between RPT measurements | 50 | RPT fields are named in `absent_signals` with `cadence_not_reached` | `mtc.pipeline.consciousness_orchestrator.OrchestratorConfig.rpt_every_n_cycles` |
| Cycles between causal emergence analyses | 50 | causal emergence is absent until the cadence is reached and the PhiTracker is warm (the analyzer returns None while the tracker warms up) | `mtc.pipeline.consciousness_orchestrator.OrchestratorConfig.causal_emergence_every_n_cycles` |
| Seconds between baseline assessments (heartbeat) | 900 | the baseline reads `unmeasured` until the first one runs | `mtc.pipeline.consciousness_orchestrator.OrchestratorConfig.heartbeat_s` |
| Seconds an orchestrator assessment is reused | 10.0 | a repeated request within the window returns the cached report | `mtc.pipeline.consciousness_orchestrator.OrchestratorConfig.assessment_cache_s` |
| Seconds the API reuses an assessment | 10.0 | `GET /api/consciousness/assessment?refresh=true` returns the cached report inside the window | `mtc.api.server.ASSESSMENT_CACHE_S` |
| Warmup cycles the orchestrator's assessment scales by | 25 | bounded histories are read against this scale, never against the running cycle count | `mtc.pipeline.consciousness_orchestrator.OrchestratorConfig.assessment_warmup_cycles` |
| Seconds between consciousness level reads | 300.0 | the thought machine's cadence is unmodulated until the first level exists | `mtc.pipeline.consciousness_helpers.DEFAULT_LEVEL_INTERVAL_S` |
| Noise baseline iterations | 10 | `run_noise_baseline` averages this many random runs per indicator | `mtc.assessment.assessment.ConsciousnessAssessment.run_noise_baseline` `iterations` |

When `run_full_assessment` normalizes and no baseline was handed in, it
computes its own with 5 iterations once and keeps it for later calls.

The orchestrator's `first_run_report()` reports the live state of these
thresholds (cycles, Phi transitions recorded against the minimum, whether the
prediction-error rank exists, the cycles until the next RPT and causal
emergence run, and the baseline's age and due reason). It is served under
`first_run` by `GET /api/consciousness/status`.

## A fresh `mtc assess`

`mtc assess` with no flags is the quick mode: it builds fresh modules and
assesses them without a warmup. A run of the assessment runner in this mode on
the development machine printed:

- `Status: insufficient_data`, `Coverage: 0.000`, `Passing indicators: 0/30`;
- 23 indicators `insufficient_data` (shown as `[NODATA]`): the workspace,
  attention schema, metacognition, active inference and beautiful loop are
  supplied but have not run a cycle, so every indicator that reads them has no
  evidence yet;
- 7 indicators `absent`, scored 0.0 and shown as `[FAIL]`:
  `affect_coherence`, `integrated_information`, `irreducibility`,
  `causal_emergence`, `algorithmic_recurrence`, `local_recurrence` and
  `oscillatory_binding_coherence`. Quick mode builds no PAD model, no RPT
  measurement, no causal emergence analyzer and no neural states, so those
  inputs are not supplied;
- `Overall score: 0.000`. Absent indicators are included in the average at
  0.0, so the command prints 0.000; the report's status is
  `insufficient_data`, and the API reports such a record as `overall_score`
  null with `overall_withheld` true, never as a measured zero.

`mtc assess --full` runs `--warmup-cycles` cycles (default 5) first. With the
default seed of 42 and five cycles, [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md)
records the result: 14 of 30 passing, status `partial`, coverage 0.633, and
five indicators still `insufficient_data` (the error trend needs 10 prediction
errors, the PhiTracker needs 128 transitions, and the beautiful loop has run
no conscious moment in the standalone runner). A run with a restored state
directory (`--state-dir`) is the path for a long-lived system.

## A fresh `mtc doctor`

`mtc doctor` (`mtc/cli/doctor.py`) runs 22 checks and reports each with one
status. Run with `--no-network` right after `mtc init` from the example
answers (the mock provider, every store disabled), in a temporary project on
the development machine, it exited 0 with this summary:
`ok=13 warning=1 failed=0 unreachable=0 skipped=6 insufficient_data=2`. Why
each non-`ok` line reads as it does:

- `store:postgres`, `store:mongodb`, `store:qdrant`, `store:neo4j` and
  `store:redis` are `skipped`: every store is disabled and its data stays in
  process memory. A disabled store is never reported `ok`.
- `provider_state` is `skipped`: no provider choice has been saved yet, so
  `LLM_PROVIDER` applies.
- `birth_date` is `insufficient_data`: `SYSTEM_BIRTH_DATE` is written by the
  first `mtc db init` (no store is needed for it), and no age is computed
  until then.
- `embedding_model` is `insufficient_data` when the `embeddings` extra is not
  installed (on that machine it was not): semantic recall and thought
  novelty are unmeasured without it. With the extra and no cached model it is
  a `warning` with the `--fetch-models` hint.
- `ctm` is a `warning`: with no semantic embedder the thought machine's
  thoughts stay quiet (`no_embedder`), and the line says so.

The required checks (`python`, `settings`, `api_key`, the five stores,
`timezone`, `brain` for the `local` and `anthropic` providers, and `extras`
when the `fep` extra is missing) decide the exit code. A missing `fep` extra
is a required `failed` and exits 1: the workspace builds the active inference
module unconditionally, so nothing measures without pymdp. A missing `neural`
extra is advisory (`insufficient_data`). On the machine above the `fep` extra
was installed, so `extras` read `ok`. The advisory checks change the exit code
only under `--strict`, where this fresh install would exit 1 on the warning
and the two `insufficient_data` lines. The count of 22 is for this run; the check list is in the module's
docstring.

## The continuous thought machine

The machine (`mtc/consciousness/continuous/machine.py`) runs with
`CTM_ENABLED` (true by default). Its cadence is in thoughts per second for each
consciousness state, from `config/ctm/ctm.yaml` (the built-in defaults are the
same):

| State | Thoughts per second | About one thought every |
|---|---|---|
| awake | 0.008 | 125 seconds |
| drowsy | 0.005 | 200 seconds |
| dreaming | 0.003 | 333 seconds |
| deep_sleep | 0.002 | 500 seconds |

Each delay is jittered by 0.3, at most 2 thoughts are generated per minute,
and the measured consciousness level scales the cadence by
`0.4 + 1.2 * level`, refreshed every 5 thoughts. Until a level exists the
cadence is unmodulated and `get_status()["modulation"]` says why. A gist is
written every 50 stored thoughts.

The `demo` profile (`CTM_PROFILE=demo`) raises the awake rate to 0.5 per
second, allows 30 thoughts per minute, sets the contention window to 0 and
writes a gist every 5 thoughts, for a short visible run.

A fresh machine is often quiet, and names why in `quiet_reason`:

- `no_embedder`: no embedder is configured, so novelty cannot be measured and
  no thought may surface (install the `embeddings` extra and fetch the model).
- `non_semantic_embedder`: only the hash embedder is present and the
  non-semantic opt-in was not given.
- `contended`: a conversation holds the brain or ended within the contention
  window (120 seconds by default), so no model-written form was made.
- `paused`: the autonomous loop or a caller paused the machine.
- `rate_limited`: the per-minute limit was reached.
- `no_seeds`: the generator had no seed text to work with.

See [CONTINUOUS_THOUGHT.md](CONTINUOUS_THOUGHT.md) for the full loop.
