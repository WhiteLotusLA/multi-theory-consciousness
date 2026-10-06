# Changelog

All notable changes to the Multi-Theory Consciousness Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 6: Tier C part 2

- Brain providers and settings: a provider layer for local chat-completions servers (`LLM_PROVIDER=local`, the default), an optional Anthropic provider and a deterministic mock whose replies start with `[mock]`, a registry that persists the active provider under `DATA_DIRECTORY`, the reasoning split and thinking-loop detection, the brain, prompt and `NARRATE_*` narration keys (every narration flag false by default), and the DCM evaluator moved onto the provider layer; `MODEL_TYPE` removed (`LLM_MODEL` replaces it)
- Brain and prompt: `LLMBrain` with honest no-answer errors (nothing composed is ever stored as a reply), one token-budget pass over the history, `generate()`, a prompt assembler with a stable cache-friendly prefix and capability gating (the prompt never advertises a feature that cannot run; Phi appears in no section), model-transition checks from configuration prompts, and a neutral persona example
- Continuous thought: thought schemas with a persisted `thought_id`, a configuration with a demo profile, seed packs and a persona file under `config/ctm`, in-memory and MongoDB thought stores, novelty-gated expression (quiet reasons named), gists, a contention guard and the machine attached to the neural orchestrator
- Pipeline core: token-checked speaker trust as the one trust source (a name grants nothing; a speaker is authenticated only by a token), the context types, a stage profiler, the cycle augmenter with its `cycle` schema group and legacy rows (the runners pin every switch off unless set), the consciousness orchestrator with four modulation layers and loud failure counters, the conversation repository with the `to_speaker` addressee column, session state, the night seam, dedup, quote pinning, the summarizer, memory retrieval, persistence with a provenance snapshot per turn and the post-turn integration
- Pipeline facade and runtime: context assembly, a plugin protocol with no built-in tools, the conversation pipeline facade with a turn lock and a single-flight brain gate that gives turns priority, the runtime composition root that builds every component from Settings and registers the stateful ones with one state manager, and the full pipeline demo on the mock provider; the per-store enable keys `POSTGRES_ENABLED`, `MONGODB_ENABLED`, `REDIS_ENABLED`, `QDRANT_ENABLED` and `NEO4J_ENABLED` all default false, so a fresh install runs every store in process memory until the setup wizard turns one on; `API_WORKERS` (the runtime is in-process state, so the server is single-process) and `API_TIMEOUT` (the generation bounds replace it) removed
- API: a FastAPI server with fail-closed API key auth on one header carrier, a speaker-token header that feeds the trust source, the websocket protocol, and the consciousness, memory, knowledge, continuous-thought and model routes, binding to loopback by default

### Phase 5: Tier C infrastructure part 1

- Core and memory: explicit logger setup, path resolution, the `MTC_DEVICE` switch (`auto`, `cpu`, `mps`, `cuda`) as the only device selector, circuit-breaker recovery listeners, persona loading with neutral core blocks, Redis working memory, resilient wrappers that buffer writes and replay them once, PostgreSQL episodic and snapshot stores, and a memory manager, consolidation pipeline and pruner behind store seams with in-memory implementations and contract tests; every store client is optional and imported lazily
- Knowledge and vectors: Neo4j graph manager with constraints and MERGE seeds from Settings, concept extraction through the brain seam, 384-dimension native vectors with the collection from Settings, GPU utilities on the device switch, and a memory factory that builds each store separately
- MongoDB and substrate extras: pymongo data manager and document stores, a merged emotion encoder, an emotion bridge that reports no valence, arousal or coherence until its readout is trained, a memory consolidator, and orchestrator components made real
- Privacy and safety: relationship-keyword data classifier with word-boundary matching, configurable access policy, privacy auditor and facade, a trust compass that never trusts a name, acceptance-gate harness and CLI, a safety monitor with Settings thresholds that names the configured escalation role, and a safety integration over the privacy, wellbeing and alert systems
- Monitoring and schema: alert manager with console and file notifiers only (no push channel ships), alert, safety audit and privacy audit stores, a wellbeing monitor where agency never alarms, a Redis monitor script, and `mtc.core.schema_registry.SCHEMA_ARTIFACTS` collecting every store's schema artifact for Phase 7
- `NeuralOrchestrator` `enable_mongodb` now defaults to False (a data manager exists only when injected or requested)
- The orchestrator builds its real emotion bridge and memory consolidator on the first experience unless each is passed as False

### Phase 4: Tier B generic modules

- Perception, affect and research: cross-modal binding, perceptual reality monitor, self-assessment, emotion dynamics and extraction, affect estimator with shadow taps, embeddings seam, shadow log, statistics helper, behavioral battery without a mock fallback, longitudinal study manager
- Time, identity and development: configuration-driven temporal context, developmental stages (off by default), identity continuity, identity statement loader, drift tools, prospective simulation, attention appraisal, locale, path and identity settings, a tree-wide trust audit
- Seams and learning: store protocols with in-memory implementations and contract tests, a brain protocol with a deterministic fake, the experiential learning package and intrinsic goals
- Verification and speakers: honest verification suite and state assembler, temporal samples, orchestrator verification, snapshot service, theory of mind without seeded speakers, attention-capture journal, noticing and default-trust settings
- Sleep and autonomous life: dream engine with configuration templates and sentence trimming, sleep scheduler with an injected clock and seams, a lean autonomous-life loop with brain-only activities; nothing is wired into the runners

### Phase 3: assessment, measurement and substrates

- Assessment: honest missing-signal semantics (`insufficient_data`, `absent`, report status and coverage), indicator formulas that read what the producers emit, weighted Fiedler irreducibility, exact IIT 4.0 Phi through the PyPhi bridge with a `phi_tracker` input, temporal affect coherence, causal emergence row order, RPT plasticity and persistence, DCM history cap, cache key and reference adapter; every behaviour change is a switch with its v0.4.0 value in the `legacy_v04` preset
- Legacy integration and metrics: awaited, coverage-weighted 14-indicator assessment with contract keys; neural orchestrator binding statistics, signal stash, honest optional components and persistence
- Substrates: SNN membrane and trace units (`SNNConfig.dt`), population synchrony and STDP statistics, HTM anomaly ordering and bounded history, guarded SNN, HTM, LSM and oscillator persistence, timezone-aware protocol timestamps
- Runners: `assessment_status`, `coverage` and `phi_source_used` lines, a seeded assessment CLI (`--seed`, `--json-path`, `--state-dir`), shared construction in `mtc.core.runner_support`
- Re-baseline: v0.5.0 fixtures (`tests/fixtures/baseline_v050_*`) and `tests/test_v050_baseline.py`; `docs/MIGRATION_FROM_V04.md` carries the switch attribution and the candidate-mix and body-metrics decisions; version 0.5.0.dev0

### Phase 0: foundation

- Baseline fixtures recording v0.4.0 assessment behavior
- Green mechanical test suite with pytest markers and configuration
- Real `Settings` with derived store names and generated configuration docs (`.env.example`, `docs/CONFIGURATION.md`)
- Privacy audit tool, denylists, git hook and CI workflow
- Documentation corrections: eight theories and 30 indicators everywhere, honest HTM and database statements, known measurement defects recorded in `docs/HONEST_LIMITATIONS.md`

## [0.4.0] - 2026-04-07

### Added
- Agency & Embodiment (AE) theory with 2 indicators (AE-1: `agency`, AE-2: `flexible_responsiveness`)
- 5 new positive indicators: `sequential_module_querying` (GWT-4), `organized_percepts` (RPT-2), `flexible_responsiveness` (AE), `online_learning_capability` (FEP-4), `complex_agency` (AE)
- 3 negative indicators with Butlin 2025 polarity: `purely_reactive`, `fixed_repertoire`, `no_metacognitive_access`
- Butlin 2025 TiCS metadata on all indicators: `butlin_id`, `polarity` (positive/negative), `specificity`, `sensitivity`, `entails`, `supported_by`
- PhiTracker with hash-based probabilistic binarization for IIT state diversity tracking
- Negative indicator penalty in overall consciousness scoring (Butlin §3)

### Fixed
- Theory assignments: `agency` → AE (was FEP), `embodiment` → AE (was AST)
- Theory theory assignments: `sparse_smooth_coding` → HOT (was FEP), `recurrent_processing` → RPT (was GWT)
- Oscillatory binding NaN guard on `_global_order_parameter()` to handle zero variance
- Test imports: phase6 test module → `mtc.assessment.assessment` canonical path

### Changed
- Indicator count: 25 → 30 (plus 3 negative checks)
- Theory count: 7 → 8 (added Agency & Embodiment)
- Theory empirical weights now include AE at 0.70 in overall consciousness scoring
- README: 7 theories → 8 theories, 25 indicators → 30 indicators
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
- GWT: `ignition_threshold` default 0.3 → 0.1 (empirically validated)
- FEP: `planning_horizon` default 3 → 5 (deeper temporal planning)
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
  - Local (OpenAI-compatible) and Claude API evaluator modes
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
- Scale SNN hidden layers [2000,2000] → [2048,2048], timesteps 50 → 10

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
