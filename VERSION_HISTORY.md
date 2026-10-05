# Version History

This document tracks MTC releases and their relationship to publications.

| Version | Date | Highlights | Notes |
|---------|------|-----------|-------|
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

## API Changes

### v0.2.x -> v0.4.0

**Scoring changes:**
- v0.2.1 added indicators 24-25 and theory-level empirical weighting for the overall score.
- v0.3.0 replaced the placeholder formulas for `recurrent_processing`, `embodiment` and `integrated_information` with real measurements, so scores are not comparable with earlier releases. Defaults changed: GWT `ignition_threshold` 0.3 -> 0.1, FEP `planning_horizon` 3 -> 5.
- v0.4.0 moved to 30 indicators plus 3 negative checks and eight theories (Agency & Embodiment added). Theory assignments changed: `agency` and `embodiment` now belong to AE, `sparse_smooth_coding` to HOT, `recurrent_processing` to RPT. Negative indicators penalize the overall score.

**New modules:**
- `mtc.core.param_overrides`: TOML override loader with schema validation (v0.3.0)
- `mtc.consciousness.phi_tracker`: PhiTracker for IIT state diversity (v0.4.0)

### v0.1.x → v0.2.0

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

**New indicators (#23-25):**
- `oscillatory_binding_coherence` (RPT)
- `causal_emergence` (IIT)
- `affect_coherence` (FEP)
