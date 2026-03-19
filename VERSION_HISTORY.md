# Version History

This document tracks MTC releases and their relationship to publications.

| Version | Date | Highlights | Notes |
|---------|------|-----------|-------|
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

### v0.1.x → v0.2.0

**Breaking changes:**
- `DCMScorer` class removed. Use `BenchmarkRunner` instead.
- `DCMReport` dataclass removed. Use `DCMBenchmarkResult` instead.
- `PerspectiveScore` dataclass removed. Use `stance_posteriors` dict on `DCMBenchmarkResult`.

**New modules:**
- `mtc.assessment.dcm_benchmark` — Full DCM Bayesian benchmark (Shiller & Duffy 2026)
- `mtc.assessment.dcm_evaluator` — LLM-based indicator evaluation
- `mtc.assessment.causal_emergence` — Hoel Effective Information + CE 2.0
- `mtc.consciousness.pad_affect` — PAD dimensional affect model
- `mtc.neural.oscillatory_binding` — AKOrN Kuramoto oscillatory binding
- `mtc.core.circuit_breaker` — Generic async circuit breaker

**New indicators (#23-25):**
- `oscillatory_binding_coherence` (RPT)
- `causal_emergence` (IIT)
- `affect_coherence` (FEP)
