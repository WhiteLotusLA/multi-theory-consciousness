# Changelog

All notable changes to the Multi-Theory Consciousness Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-03-27

### Added
- AutoConsciousness infrastructure: autonomous parameter research loop (Karpathy autoresearch pattern)
  - `experiments/params.toml` — single mutable parameter file
  - `experiments/param_schema.toml` — 69 tunable parameters with bounds
  - `experiments/program.md` — Claude Code agent instructions (Phase 2)
  - `scripts/run_experiment.py` — experiment runner with `--validate` ablation checks
  - `mtc/core/param_overrides.py` — TOML override loader with schema validation

### Changed (Backported from the reference implementation)
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

## [0.2.1] — 2026-03-25

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
  - `oscillatory_binding_coherence` (RPT) — Kuramoto order parameter
  - `causal_emergence` (IIT) — macro vs micro causal power
  - `affect_coherence` (FEP) — PAD dimensional consistency

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
