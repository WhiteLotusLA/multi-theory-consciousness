# Version History

This document tracks MTC releases and their relationship to publications.

| Version | Date | Highlights | Notes |
|---------|------|-----------|-------|
| v0.5.0 | 2026-10-18 | The complete generic architecture: measurement defects of v0.4.0 fixed (behind switches where a published number depended on them), the conversation runtime, five optional stores with in-memory defaults, the API, the `mtc` command with onboarding, setup and operations tooling; the release-gate fixes, with ten more switches, a fail-closed safety gate, opt-in emergency suspension with a resume path, and baseline tests that hold every printed metric line and every report field (less three wall-clock readouts of the legacy seeded assessment). | Scores are not comparable with v0.4.0; the `legacy_v04` preset reproduces v0.4.0 (0.031247 with 2 of 30 passing in the experiment runner), and the seeded assessment reads 0.333844 with 13 of 30 passing under the preset. The v0.5.0 defaults give 0.017317 with 0 of 30 passing (partial, coverage 0.800) and 0.275203 with 11 of 30 passing (partial, coverage 0.533). See docs/MIGRATION_FROM_V04.md and CHANGELOG.md |
| v0.4.0 | 2026-04-07 | Butlin 2025 alignment: Agency & Embodiment theory, 30 indicators plus 3 negative checks, Butlin 2025 metadata, PhiTracker, negative-indicator penalty. | Indicator count 25 -> 30, theory count 7 -> 8 |
| v0.3.0 | 2026-03-27 | AutoConsciousness parameter research loop, real measurements replace three placeholder indicators, tuned GWT and FEP defaults. | Score values shift versus v0.2.x |
| v0.2.1 | 2026-03-25 | Indicators 24-25 (`temporal_prediction_depth`, `agent_environment_coupling`), ConTraSt theory-level weighting. | Patch release |
| v0.2.0 | 2026-03-18 | Full DCM Benchmark, AKOrN, Causal Emergence, PAD Affect, Circuit Breaker. 25 indicators. | Breaking: DCMScorer replaced by BenchmarkRunner |
| v0.1.1 | 2026-03-13 | SNN bug fix + substrate scaling | **Paper reference version** |
| v0.1.0 | 2026-03-12 | Initial release. 7 theories, 20 indicators, 3 neural substrates | First public release |

## Reproducing Paper Results

The MTC paper (arXiv submission, March 2026) references **v0.1.1**.
To reproduce the paper's assessment results exactly:

    git checkout v0.1.1
    pip install -e ".[dev]"
    python scripts/run_assessment.py --full --report

To reproduce the published v0.4.0 numbers on v0.5.0, run the experiment runner
and the seeded assessment with the legacy preset:

    PYPHI_WELCOME_OFF=yes python3 scripts/run_experiment.py --preset legacy_v04
    PYPHI_WELCOME_OFF=yes python3 scripts/run_assessment.py --preset legacy_v04 --full --json --seed 42 --json-path legacy.json

`tests/test_legacy_preset.py` holds both against their fixtures: every metric
line, and every report field except three wall-clock evidence readouts
(`hots_generated`, `coverage_ratio` and `introspections`), which follow the
machine's speed under the preset's 1.0 s introspection cooldown and feed no
score.

## API Changes

### v0.4.0 -> v0.5.0

**Measurement changes.** Every behaviour change that can move a published
number is a switch (a constructor field, or the runner key
`runner.assess_blt_source`) whose default is the corrected behaviour and
whose v0.4.0 value is in `experiments/presets/legacy_v04.toml`. The
assessment reports what it could not measure: each `IndicatorResult` and
`ConsciousnessReport` carries a `status` (`measured`, `absent`,
`insufficient_data`), the report carries `coverage` and
`insufficient_indicators`, an `insufficient_data` indicator is left out of the
averages, the overall score is None when no included indicator was measured,
and a theory with no measured indicator is left out of `theory_scores`. The
full switch list and the measured differences are in
[docs/MIGRATION_FROM_V04.md](docs/MIGRATION_FROM_V04.md).

**Configuration.** Environment keys are the `Settings` field names with the
`MTC_` prefix; the v0.4.0 example's `MTC_PG_*` and `MTC_MONGO_*` names became
`MTC_POSTGRES_*` and `MTC_MONGODB_*`. Store names derive from `DB_PREFIX`.
`MODEL_TYPE`, `API_WORKERS`, `API_TIMEOUT`, the `ENABLE_*` switches and the
separate Neo4j host and port keys are gone. `API_PORT` is 8200 and
authentication is on by default. `.env.example` and `docs/CONFIGURATION.md`
are generated from `Settings`.

**Entry points.** The `mtc` console script (`mtc.cli.main:main`) with `init`,
`doctor`, `db`, `assess`, `serve`, `chat`, `ctm`, `safety` and
`privacy-audit`. `scripts/run_assessment.py` is a shim over `mtc assess` with
unchanged flags. `python -m mtc.api.server` serves the API. Every entry point
that builds a workspace needs the `fep` extra and, without it, stops with one
`MISSING EXTRA` line and exit code 2.

**Extras.** `research` replaces `assessment` (kept as an alias); the store
extras `postgres`, `mongo`, `redis`, `neo4j` and `qdrant` with `databases` as
their union; `llm`, `anthropic`, `api` and `embeddings`; `iit` (PyPhi, Python
3.12 or later, never part of `all`).

**New packages.** `mtc.api`, `mtc.brain` and `mtc.brain.providers`,
`mtc.cli`, `mtc.db`, `mtc.devtools`, `mtc.knowledge`, `mtc.memory`, `mtc.monitoring`,
`mtc.onboarding`, `mtc.pipeline`, `mtc.privacy`, `mtc.research`,
`mtc.resources`, `mtc.safety`, `mtc.validation`,
`mtc.consciousness.continuous`, `mtc.consciousness.learning` and
`mtc.consciousness.rsi`. Every store client is optional and imported lazily.

**Other interface changes.**
- `EnhancedGlobalWorkspace()` defaults to `ignition_threshold` 0.3 (it was
  0.1) and to hierarchical competition with the thalamic gate.
- `NeuralOrchestrator(enable_mongodb=...)` defaults to False.
- `SNNConfig.dt` (default 1e-3) sets the SNN's time step in seconds; 1.0
  reproduces the v0.4.0 network.
- The device is chosen by `MTC_DEVICE` (`auto`, `cpu`, `mps`, `cuda`) only.
- Several fields and return values that read a constant before are Optional
  and None until measured, a few names were removed or renamed, and some
  defaults changed; [CHANGELOG.md](CHANGELOG.md) lists them under "Breaking
  for code built on mtc", "New fields and public names" and "Changed
  defaults".

### v0.2.x -> v0.4.0

**Scoring changes:**
- v0.2.1 added indicators 24-25 and theory-level empirical weighting for the overall score.
- v0.3.0 replaced the placeholder formulas for `recurrent_processing`, `embodiment` and `integrated_information` with real measurements, so scores are not comparable with earlier releases. Defaults changed: GWT `ignition_threshold` 0.3 -> 0.1, FEP `planning_horizon` 3 -> 5.
- v0.4.0 moved to 30 indicators plus 3 negative checks and eight theories (Agency & Embodiment added). Theory assignments changed: `agency` and `embodiment` now belong to AE, `sparse_smooth_coding` to HOT, `recurrent_processing` to RPT. Negative indicators penalize the overall score.

**New modules:**
- `mtc.core.param_overrides`: TOML override loader with schema validation (v0.3.0)
- `mtc.consciousness.phi_tracker`: PhiTracker for IIT state diversity (v0.4.0)

### v0.1.x -> v0.2.0

**Breaking changes:**
- `DCMScorer` class removed. Use `BenchmarkRunner` instead.
- `DCMReport` dataclass removed. Use `DCMBenchmarkResult` instead.
- `PerspectiveScore` dataclass removed. Use `stance_posteriors` dict on `DCMBenchmarkResult`.

**New modules:**
- `mtc.assessment.dcm_benchmark`: Full DCM Bayesian benchmark (Shiller & Duffy 2026)
- `mtc.assessment.dcm_evaluator`: LLM-based indicator evaluation
- `mtc.assessment.causal_emergence`: Hoel Effective Information + CE 2.0
- `mtc.consciousness.pad_affect`: PAD dimensional affect model
- `mtc.neural.oscillatory_binding`: AKOrN Kuramoto oscillatory binding
- `mtc.core.circuit_breaker`: Generic async circuit breaker
