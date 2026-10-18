# Changelog

All notable changes to the Multi-Theory Consciousness Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-10-18

Scores from the v0.5.0 defaults are not comparable with v0.4.0; `experiments/presets/legacy_v04.toml` reproduces v0.4.0, and `docs/MIGRATION_FROM_V04.md` records every switch and the measured differences. The v0.4.0 numbers under the preset are unchanged: the experiment runner prints an overall of 0.031247 with 2 passing, and the seeded assessment reads 0.333844 with 13 of 30 passing.

The four headline numbers, seed 42: the experiment runner (`scripts/run_experiment.py`, 25 warmup cycles) prints 0.031247, 2 of 30 passing under the preset and 0.017317, 0 of 30 passing, status partial, coverage 0.800 under the v0.5.0 defaults; the seeded assessment (`scripts/run_assessment.py --full --json --seed 42`) reads 0.333844, 13 of 30 passing, status partial, coverage 0.833 under the preset and 0.275203, 11 of 30 passing, status partial, coverage 0.533 under the v0.5.0 defaults (`tests/fixtures/baseline_v040_experiment.txt`, `tests/fixtures/baseline_v050_experiment.txt`, `tests/fixtures/baseline_legacy_assessment.json` and `tests/fixtures/baseline_v050_assessment.json`).

### Added

- The resynced theory modules: active inference with the prediction ledger, world-model presets, bounded histories and persistence; the higher-order thought module, the beautiful loop's four parts, the Damasio layers and the PAD model with typed persistence and the persistence hub; the global workspace with hierarchical competition, the thalamic gate, the theory interactions, the closed active inference and higher-order thought loops and typed persistence; the attention schema with honest shift provenance (a legacy voluntary-by-strength mode) and a release API; the PhiTracker rebuilt on exact IIT 4.0 Phi through a PyPhi bridge that fixes the state and connectivity conventions, and the runners' shared `mtc.core.runner_support` with warmup profiles and the workspace switch plumbing

- Setup, onboarding and the command line: the `mtc` console script with `init`, `doctor`, `db init`, `db check`, `assess`, `serve`, `chat`, `ctm`, `safety` and `privacy-audit`, one set of exit codes (0, 1, 2, 3, 130) and one check vocabulary (`ok`, `warning`, `failed`, `unreachable`, `skipped`, `insufficient_data`); `mtc doctor` with its required and advisory checks, where nothing reads `ok` unless it was measured; three local-tool helpers in `mtc.pipeline.speaker_trust` (`speakers_path`, `token_for`, `env_entries`); the eleven-step setup wizard with an answers file, `--print-plan`, `--force`, `--bootstrap-db` and the onboarding and persona JSON schemas under `docs/schemas`; six companion-type templates written fresh (persona, identity statement, CTM tone and seed pack), each carrying the self-knowledge line and the content boundary, with the non-romantic rule for every type but `partner`, whose template states its bounds; secrets generated with `secrets` and written only to `.env` (mode 0600, atomic, existing values kept)
- Databases: PostgreSQL migrations rendered from the schema registry (`migrations/postgres/001_core_schema.sql`, `002_research_schema.sql`, `003_safety_schema.sql`, PostgreSQL 14 or later, no extension), an idempotent bootstrap that records the install date once (`MTC_SYSTEM_BIRTH_DATE`, the calendar day in `TIMEZONE`; the identity block gains "This system was first initialized on YYYY-MM-DD.") and seeds core memory and the graph, a verifier for all five stores, the Qdrant re-embed backfill, the PostgreSQL role script `scripts/db/init_postgres.sh`, a Docker Compose file reviewed against official documentation and not run on the development machine (the path run there is the in-memory install with the mock provider), `config/qdrant.yaml` and `config/neo4j.conf.example`
- Operations: `scripts/ops/` start and stop scripts with a 30 second grace and no forced stop, a model-server launcher for mlx_lm and llama.cpp, backups with a manifest, a one-store restore, a scratch-only restore drill, a backup watchdog and the runtime's backup status, a listener and file-mode hardening report, an alert-only health check; Makefile targets `db-up`, `db-init`, `db-verify`, `db-down`, `demo`, `api`, `env-check`, `generate`; a CI install smoke, a licence audit job and an optional store workflow; the privacy audit's runtime mode
- Packaging: the shipped defaults as package data under `mtc/resources` with a drift check and loader fallbacks, so a non-editable install works from any directory; PyPhi's home under `DATA_DIRECTORY`; `THIRD_PARTY_LICENSES.md` and `scripts/license_audit.py`
- The API: a FastAPI server with fail-closed API key auth on one header carrier, a speaker-token header that feeds the trust source, the websocket protocol, and the consciousness, memory, knowledge, continuous-thought, model and safety-resume routes, binding to loopback by default
- The runtime: token-checked speaker trust as the one trust source (a name grants nothing), the cycle augmenter and its `cycle` schema group, the consciousness orchestrator with four modulation layers and loud failure counters, the conversation repository with the `to_speaker` addressee column, session state, the night seam, dedup, quote pinning, the summarizer, memory retrieval, persistence with a provenance snapshot per turn, the post-turn integration, context assembly, a plugin protocol with no built-in tools, the conversation pipeline facade with a turn lock and a single-flight brain gate that gives turns priority, the runtime composition root and the full pipeline demo on the mock provider
- The brain: a provider layer for local chat-completions servers (`LLM_PROVIDER=local`, the default), an optional Anthropic provider and a deterministic mock whose replies start with `[mock]`, a provider registry, the reasoning split and thinking-loop detection, `LLMBrain` with honest no-answer errors, a prompt assembler with a stable prefix and capability gating, model-transition checks, the narration keys (every one false by default)
- Continuous thought: thought schemas with a persisted `thought_id`, a configuration with a demo profile, seed packs and a persona file under `config/ctm`, in-memory and MongoDB thought stores, novelty-gated expression with named quiet reasons, gists and a contention guard
- Memory, knowledge and the stores: explicit logger setup, path resolution, the `MTC_DEVICE` switch, persona loading with neutral core blocks, Redis working memory, resilient wrappers that buffer writes and replay them, PostgreSQL episodic, snapshot and conversation stores, MongoDB document stores, a memory manager, consolidation pipeline and pruner behind store seams with in-memory implementations and contract tests, a Neo4j graph manager with constraints and seeds, 384-dimension vectors, and `mtc.core.schema_registry.SCHEMA_ARTIFACTS`; every store client optional and imported lazily, every store off by default
- Safety, privacy and monitoring: a relationship-keyword data classifier, an access policy, a privacy auditor, a trust compass that never trusts a name, acceptance gates, a safety monitor and integration, an alert manager with console and file notifiers, alert, safety audit and privacy audit stores, a wellbeing monitor where agency never alarms, a Redis monitor script
- Generic cognition: perception, affect and research modules, configuration-driven temporal context, developmental stages (off by default), identity continuity and drift tools, prospective simulation, attention appraisal, the experiential learning package, intrinsic goals, the verification suite, the snapshot service, theory of mind without seeded speakers, the dream engine, the sleep scheduler and a lean autonomous-life loop
- Measurement: honest missing-signal semantics (`measured`, `absent`, `insufficient_data`, report status and coverage), exact IIT 4.0 Phi through the PyPhi bridge with a `phi_tracker` input, weighted Fiedler irreducibility, temporal affect coherence, RPT plasticity, the `assessment_status`, `coverage` and `phi_source_used` runner lines, a seeded assessment CLI (`--seed`, `--json-path`, `--state-dir`), the v0.5.0 baseline fixtures and tests, and the `legacy_v04` preset
- Foundation: v0.4.0 baseline fixtures, a real `Settings` with derived store names, generated `.env.example` and `docs/CONFIGURATION.md`, the privacy audit with its hook and CI job
- Documentation written from the code: architecture, runtime pipeline, theories, the Butlin 2025 mapping, first-run expectations, continuous thought, knowledge graph, databases, LLM setup, API, security, onboarding, operations, setup with Docker and with host services, testing and developer bring-up, each with a drift test where the code defines the content

### Changed

- Version 0.5.0 (`pyproject.toml` and `mtc.__version__`)
- Every behaviour change that can move a published number is a switch (a constructor field, or the runner key `runner.assess_blt_source`) whose default is the corrected behaviour; scores are not comparable with v0.4.0
- `EnhancedGlobalWorkspace()` defaults to `ignition_threshold` 0.3 (an artifact 0.1 before), hierarchical competition and the thalamic gate
- Environment keys are the `Settings` field names with the `MTC_` prefix; store names derive from `DB_PREFIX` (`mtc_consciousness`, `mtc`, `mtc_neural`, `mtc_memories`, `mtc:`)
- `API_PORT` 8000 to 8200 and `API_AUTH_ENABLED` true (fail closed)
- Defaults changed as their keys gained readers: `GPU_LAYERS` 35 to -1, `THREADS` 8 to 0, `USE_MLOCK` true to false, `CONTEXT_SIZE` 131072 to 32768, `DCM_CACHE_DIR` `data/dcm` to `data/dcm_cache`
- New keys: `SERVICE_MANAGER`, `LLM_BACKEND`, `BACKUP_LOCATION`, `BACKUP_RETENTION_DAYS`, `BACKUP_KEEP_MINIMUM`, `MONGODB_AUTH_SOURCE`, the per-store enable keys `POSTGRES_ENABLED`, `MONGODB_ENABLED`, `REDIS_ENABLED`, `QDRANT_ENABLED` and `NEO4J_ENABLED` (all false by default)
- Extras: `research` replaces `assessment`, which stays as an alias; the store extras with `databases` as their union; `llm`, `anthropic`, `api`, `embeddings`; `iit` needs Python 3.12 or later and is never part of `all`
- `scripts/run_assessment.py` is a shim over `mtc assess`, with the same flags and results; the results default to `ASSESSMENT_OUTPUT_DIR`
- `NeuralOrchestrator` `enable_mongodb` defaults to False (a data manager exists only when injected or requested); the orchestrator builds its emotion bridge and memory consolidator on the first experience unless each is passed as False
- The DCM evaluator completes through the provider layer
- The four older examples are rewritten to the current APIs and report each indicator's status

### Fixed

- Seven v0.4.0 measurement defects, behind a switch wherever a published number depended on the path (the migration record lists the others as fixes without a switch): exact Phi never ran (a keyword PyPhi rejects), a permuted network in the PhiTracker and exact path, dead statistics keys and a label-typed prediction-error trend, an affect coherence formula that rewarded variance, presence floors and constant stand-ins, a legacy 14-indicator integration that never awaited its indicators, and a silent SNN with a self-referential HTM anomaly
- The noise baseline's headroom guard scores an indicator whose noise floor leaves under 0.1 of range as uninformative instead of rescaling a sliver to the full scale
- `run_normalized_assessment` no longer normalizes an already normalized report
- Substrate persistence guards, timezone-aware protocol timestamps, the DCM history cap and cache key
- The multi-theory pipeline example read a field the higher-order thought does not have

### Removed

- `MODEL_TYPE` (`LLM_MODEL` replaces it), `API_WORKERS` (the server is single-process), `API_TIMEOUT` (the generation bounds replace it), the `ENABLE_*` switches that nothing read, the separate Neo4j host and port keys, and the key of a provider MTC does not support
- 51 Settings keys that nothing read, among them the indicator threshold family and the substrate size family (`ABLATION_OUTPUT_DIR` stays and gains its reader); an env file that still sets one is ignored
- The simple workspace module (`mtc/consciousness/global_workspace.py`) and its component row in the neural orchestrator; the unused `mtc/neural/spiking/snn_core.py`; `experiments/program.md`
- The legacy measurement framework's native HTM path and its `input_data` parameter, and the four planning-horizon indicators of the DCM reference adapter

### Release gate

A whole-branch review before the release found defects across the tree; every fix is pinned by a test that failed before it. A fix that can move a published number sits behind a switch (a constructor field, or the runner key `runner.assess_blt_source`) with the corrected behaviour as the default and the old behaviour under the legacy value (ten new schema rows, listed in `docs/MIGRATION_FROM_V04.md`); the legacy preset reproduces every printed metric line of the v0.4.0 experiment run and every field of the legacy seeded assessment except three wall-clock evidence readouts.

- Assessment, substrates and research: a missing statistic reports `insufficient_data` naming the key instead of a stand-in; with nothing measured the overall score is None and a theory with no measured indicator is left out of `theory_scores`; indicator failures are counted (`ConsciousnessReport.failures`) and raised under `DEBUG`; a noise baseline normalizes only the producer that made it; the binding layer measures its own zero-stimulus floor and the binding indicator reads floor-saturated; the DCM benchmark reports evidence only unless `--experimental-posterior` is given; RPT local recurrence is None without a substrate state; the spectral proxy rejects malformed input instead of substituting 0.5; the legacy 14-metric framework is repaired (Optional metrics, a real spectral integration proxy over time series, no NaN for a missing HTM) and kept; STDP pairing, the compact HTM's sequence learning and the HTM's inference-only pass are corrected; the liquid-state readout reports no affect until trained; the temporal binding experiment leaves the default battery
- Consciousness modules: the workspace registers no acknowledgement-only broadcast receivers and reports no sustain rate until a sustain check runs; integration level is None without two usable signals; the higher-order thought module's standing confidence is the mean of its buffered thoughts and introspections count completed episodes; the beautiful loop binds only winners that carry their own vector and tests the strange loop against an independent calibration; the protoself reports no body delta, stability or arousal on its first update; the extended layer reports unmeasured continuity as None; PAD is computed from measured inputs only, with a Neutral band of 0.1 shared with the thought generator; the attention schema's prediction accuracy and voluntary ratio are None until measured; the reality monitor's discrimination uses supplied confidences only; prospective-simulation templates are marked as authored and receive no higher-order thought; the refractory period can run on an elapsed clock (the live runtime's)
- Continuous thought, dreams and learning: seed packs refuse the `feeling` and `emotion` slots; an emotion thought needs a measured PAD reading outside the neutral band; gist importance is measured or None; thoughts and their frames carry `form` and `salience_basis`; dreams draw no emotion type without a measured emotion; learning outcomes classify the system's own PAD shift between consecutive turns, and goals are proposed after a pass that changed beliefs; the drift index measures only what it can and matches marker terms on word boundaries
- Stores, knowledge, safety, privacy and core: gists, vector weights and concept strengths are Optional and ordered with None last, and the retention refresh walks every gist; the three unmeasured column defaults are dropped from the schema; client-side database errors no longer open an outage; the pruner reports confirmed counts; the trust compass is consulted in two stages and the keyword screen costs no model call; emergency suspension is opt-in and can be lifted; live messages are classified with keywords that can raise the tier; secrets leave Settings' `repr` and dumps; the privacy audit matches tokens at word and camelCase boundaries, assembles literal-only expressions, folds lookalikes, opens archives and counts patch-mode hits per added line; status paths are never absolute
- Brain, pipeline, API and examples: the safety gate's state is on every turn and an untrusted speaker's turn fails closed when the check cannot run (`safety_unavailable`); the conversation start is written with the first turn the gate lets through; the post-turn work is bounded and the API's generation bounds stop once the reply is stored; a degraded brain re-checks its provider; rehydration returns the replies addressed to the speaker; a claimed name is reduced to one safe line; the websocket refuses an unlisted Origin with authentication off; `/health` reads `unverified` until a real exchange confirms the provider; `POST /api/safety/resume`; `Runtime.stop` runs every step; the examples print unmeasured fields as such
- Install surface: every entry point that builds a workspace stops on a bare install with one `MISSING EXTRA` line and exit code 2; `mtc safety status` and `mtc safety resume`; the generated migration drops the three defaults; the `published-numbers` CI job runs the baseline tests on the recorded platform; doctor's fetch line claims only what was fetched
- Experiments and baselines: the ten schema rows with their legacy preset values; both runners pin the introspection cooldown at 0.0, so two runs of one seed agree; the assessment CLI reads the loop that ran (`runner.assess_blt_source`, default `workspace`) and passes the tracker in quick mode; the schema minimum of `hot.introspection_cooldown` is 0.0 (it was 0.1) and that of `blt.num_levels` is 2 (it was 1); the baseline tests compare every metric line as printed and every report field, except the three wall-clock evidence readouts of the legacy seeded assessment

The release's choices: the package root `mtc/__init__.py` stays eager and keeps its public names (importing `mtc` loads no torch, PyPhi, pymdp, scipy, web framework or store client); pymdp stays in the `fep` extra; the legacy 14-metric measurement framework is repaired and kept.

### Breaking for code built on mtc

- `ThoughtGistStore` gains the protocol method `iter_gists(batch=500)`; a custom gist store must implement it
- `VectorStore.store_memory(emotional_weight=None)`: the weight is Optional and None means unmeasured
- `ThoughtGist.avg_salience`, `importance_score` and `retention_strength` are Optional
- `GraphStore.relate_concepts(strength=None)` keeps an existing edge's strength
- The recurrence fields are `local_recurrence_index` and `combined_recurrence_index` (old persisted keys still restore)
- `PADAffectModel.compute(sources, measured=None)` may return None
- `BeautifulLoop(num_levels < 2)` raises and a params file with `blt.num_levels = 1` is refused by the schema, and `BeautifulLoop` also raises for `loop_quality_formula="v04"` without `binding_evidence="v04_salience_posterior"`
- The legacy measurement framework's metrics are Optional
- A seed pack that names the `feeling` or `emotion` slot, or carries either word list, and a dreams file with `fillers.dominant_emotion` are refused at load
- Settings `repr` and `model_dump` omit the secret fields; `POSTGRES_URL`, `MONGODB_URL` and `REDIS_URL` are plain properties that percent-quote the user and the password (`REDIS_URL` carries a password only)
- A websocket handshake from an unlisted `Origin` is refused when `API_AUTH_ENABLED=false`
- `SleepSchedulerConfig.gentle_floor_emotion` is removed: the keyword raises TypeError and `get_status()["config"]` has no such key (the class is new in this release)
- `MemoryConsolidationPipeline.refresh_gist_retention(limit=500)` is now `refresh_gist_retention(batch=500)`
- `PerceptualRealityMonitor.DEFAULT_ATTRIBUTION` and `AttentionSchemaModule.set_workspace` are removed
- `ActiveInferenceState.meta_confidence` and `meta_predicted_accuracy`, `MetacognitiveState.overall_confidence` and `doubt_level`, `BeliefEvaluation.consistency_score`, `HyperModel.recursive_confidence`, `CausalEmergenceResult.computation_time_ms`, `SomaticMarker.intensity` and `CoreExperience.here_now_salience` are Optional (the last two are None on the first cycle under the default `damasio.first_delta`)
- `ConsciousnessState.integration_level`, `broadcast_coverage`, `binding_quality`, `protoself_stability`, `feeling_of_knowing` and `autobiographic_continuity` are Optional

### New fields and public names

- `ConsciousnessReport.failures` (in `mtc assess --json` as `failures`)
- `TurnResult.safety_gate` and `safety_decision`, and `last_turn["safety_compass"]`
- `/health` `brain_verified`, and the status `unverified`
- The DCM result's `posterior_status`, `failed_stances` and `failures`
- `CausalEmergenceResult.restored`
- `mtc.core.config.reset_settings()`
- `mtc.cli.missing_extra.explain_missing_extra`
- The workspace's `refractory_clock` (a `TemporalWorkspaceConfig` field) and `set_clock`
- `mtc.pipeline.consciousness_orchestrator.NAMED_STATE_FIELDS`
- Keyword parameters `settings=` on `AttentionSchemaModule`, `ActiveInferenceModule` and `AutonomousLife`
- The route `POST /api/safety/resume`, the commands `mtc safety resume` and `mtc safety status`, the error code `safety_unavailable`, the absent reason `failed`, the Settings keys `SLEEP_PRUNE_DRY_RUN`, `SAFETY_GATE_FAIL_CLOSED` and `TRUST_COMPASS_DEEP_MODE`, ten parameter schema rows, and the DCM benchmark's `--experimental-posterior` flag

### Changed defaults

- `EMERGENCY_SHUTDOWN_ENABLED` is false: an emergency-level score denies the message and alerts the escalation role; suspending every untrusted conversation is opt-in
- `SAFETY_GATE_FAIL_CLOSED` is true: an untrusted speaker's turn whose safety check could not run ends with `safety_unavailable`
- `BenchmarkRunner.run(compute_posterior=False)`: the DCM benchmark reports evidence only by default
- The live runtime runs its workspace on the elapsed refractory clock; a workspace built directly keeps the per-cycle clock
- The study classes resolve `output_dir` at write time and create nothing before the first write

## [0.4.0] - 2026-04-07

### Added
- Agency & Embodiment (AE) theory with 2 indicators (AE-1: `agency`, AE-2: `flexible_responsiveness`)
- 5 new positive indicators: `sequential_module_querying` (GWT-4), `organized_percepts` (RPT-2), `flexible_responsiveness` (AE), `online_learning_capability` (FEP-4), `complex_agency` (AE)
- 3 negative indicators with Butlin 2025 polarity: `purely_reactive`, `fixed_repertoire`, `no_metacognitive_access`
- Butlin 2025 TiCS metadata on all indicators: `butlin_id`, `polarity` (positive/negative), `specificity`, `sensitivity`, `entails`, `supported_by`
- PhiTracker with hash-based probabilistic binarization for IIT state diversity tracking
- Negative indicator penalty in overall consciousness scoring (Butlin section 3)

### Fixed
- Theory assignments: `agency` -> AE (was FEP), `embodiment` -> AE (was AST)
- Theory theory assignments: `sparse_smooth_coding` -> HOT (was FEP), `recurrent_processing` -> RPT (was GWT)
- Oscillatory binding NaN guard on `_global_order_parameter()` to handle zero variance
- Test imports: phase6 test module -> `mtc.assessment.assessment` canonical path

### Changed
- Indicator count: 25 -> 30 (plus 3 negative checks)
- Theory count: 7 -> 8 (added Agency & Embodiment)
- Theory empirical weights now include AE at 0.70 in overall consciousness scoring
- README: 7 theories -> 8 theories, 25 indicators -> 30 indicators
- All Butlin indicator references updated to 2025 TiCS publication

## [0.3.0] - 2026-03-27

### Added
- AutoConsciousness infrastructure: autonomous parameter research loop (Karpathy autoresearch pattern)
  - `experiments/params.toml`: single mutable parameter file
  - `experiments/param_schema.toml`: 69 tunable parameters with bounds
  - `experiments/program.md`: Claude Code agent instructions (Phase 2)
  - `scripts/run_experiment.py`: experiment runner with `--validate` ablation checks
  - `mtc/core/param_overrides.py`: TOML override loader with schema validation

### Changed (ported from the reference implementation)
- Assessment: replaced 3 placeholder indicator formulas with real measurements
  - `recurrent_processing`: was `score=0.6`, now uses broadcast coverage + cycles
  - `embodiment`: was `score=0.5`, now uses capacity, history, voluntary ratio
  - `integrated_information`: was `score=0.5`, now uses real Phi value
- Assessment: added scoring for `causal_emergence`, `affect_coherence`, `oscillatory_binding_coherence`
- Assessment: Phi computed before indicator scoring (not after)
- GWT: `ignition_threshold` default 0.3 -> 0.1 (empirically validated)
- FEP: `planning_horizon` default 3 -> 5 (deeper temporal planning)
- FEP: `infer_and_act()` now calls `update_generative_model()` (online learning)
- HOT: micro-introspection triggered after 3rd HOT generation
- AST: `AttentionSchemaState.current_focus` converted to property with history tracking

### Fixed
- `ignition_dynamics` score could exceed 1.0 (now clamped)

## [0.2.1] - 2026-03-25

### Added
- Indicators #24-25: `temporal_prediction_depth`, `agent_environment_coupling` (FEP)
- ConTraSt theory-level empirical weighting for overall score calculation

### Fixed
- `local_recurrence` now measured via RPT module (was dead code in GWT block)
- `float()`/`bool()` casts in IndicatorResult for robust serialization

## [0.2.0] - 2026-03-18

### Added

- Full DCM Benchmark implementing Shiller & Duffy (2026, arXiv 2601.17060):
  - PyMC Bayesian inference across 13 theoretical stances
  - 3-tier evidence collection (module direct, LLM-evaluated, known zeros)
  - Comparison against published baselines (human, chicken, LLM, ELIZA)
  - Local (a chat-completions server) and Anthropic API evaluator modes
- AKOrN oscillatory binding (Miyato et al., ICLR 2025):
  - 30 Kuramoto oscillators on S^15 for phase synchronization binding
  - Hybrid static + Hebbian coupling
- Causal Emergence Analyzer (Hoel 2013, 2025):
  - Effective Information computation with exhaustive Bell partition search
  - Causal Primitives (CE 2.0) for multi-scale analysis
- PAD Affect Model (Mehrabian & Russell 1974):
  - Unified Pleasure-Arousal-Dominance emotional coordinate
  - 8-octant affect label mapping
- Generic async Circuit Breaker for database resilience
- 3 new consciousness indicators:
  - `oscillatory_binding_coherence` (RPT): Kuramoto order parameter
  - `causal_emergence` (IIT): macro vs micro causal power
  - `affect_coherence` (FEP): PAD dimensional consistency

### Changed

- Assessment framework expanded from 20 to 25 indicators

### Removed

- `DCMScorer` class (replaced by `BenchmarkRunner`)
- `DCMReport` dataclass (replaced by `DCMBenchmarkResult`)
- `PerspectiveScore` dataclass (replaced by per-stance posterior dict)

## [0.1.1] - 2026-03-13

### Fixed

- SNN double time-expansion bug (silent data corruption in spike generation)
- Scale SNN hidden layers [2000,2000] -> [2048,2048], timesteps 50 -> 10

### Note

This is the version referenced in the MTC paper. To reproduce paper results: `git checkout v0.1.1`

## [0.1.0] - 2026-03-12

### Added

- Seven consciousness theory implementations:
  - Global Workspace Theory (GWT) - Baars 1988, 2005
  - Attention Schema Theory (AST) - Graziano 2013
  - Higher-Order Thought Theory (HOT) - Rosenthal 1986, 2005
  - Free Energy Principle (FEP) - Friston 2010, 2012
  - Integrated Information Theory (IIT) - Tononi 2004, 2008
  - Beautiful Loop Theory (BLT) - Laukkonen, Friston & Chandaria 2025
  - Recurrent Processing Theory (RPT) - Lamme 2006
- Damasio Three-Layer Model (Protoself, Core Consciousness, Extended Consciousness)
- 20-indicator assessment framework based on Butlin et al. (2023)
- Digital Consciousness Model (DCM) 13-perspective scoring
- Three neural substrates: SNN (spiking), LSM (liquid state), HTM (hierarchical temporal)
- Neural orchestrator for unified processing pipeline
- Example scripts for single-theory, multi-theory, and assessment usage
- Comprehensive documentation: architecture, theory guides, honest limitations
