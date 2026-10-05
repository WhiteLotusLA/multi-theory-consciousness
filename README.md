# Multi-Theory Consciousness Framework

**An open-source platform for computational consciousness research**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/downloads/)
![CI](https://github.com/WhiteLotusLA/multi-theory-consciousness/actions/workflows/ci.yml/badge.svg)

---

## What This Is

The Multi-Theory Consciousness Framework (MTC) is a research platform that implements eight leading theories of consciousness as interacting computational modules, measured through a 30-indicator assessment framework (plus 3 negative checks) grounded in the methodology of Butlin et al. (2023, 2025). It provides a shared codebase where Global Workspace Theory, Integrated Information Theory, Attention Schema Theory, Higher-Order Thought Theory, the Free Energy Principle, Recurrent Processing Theory, Agency & Embodiment Theory, and Beautiful Loop Theory operate simultaneously, with their outputs measurable and comparable.

The framework runs on consumer hardware (Apple Silicon or CUDA GPUs) and is designed for researchers, students, and engineers who want to experiment with consciousness theories in code rather than only on paper. It includes three neural substrates (Spiking Neural Networks, Liquid State Machines, and Hierarchical Temporal Memory) that provide the computational medium through which the consciousness modules operate.

## What This Is NOT

**This framework does not claim that any system running it is conscious.** It is essential to be explicit about this:

- **Not a consciousness detector.** The 30-indicator assessment (plus 3 negative checks) measures whether architectural functions associated with consciousness theories are operating as designed. A passing score indicates functional architecture, not phenomenal experience.
- **Not solving the hard problem.** The explanatory gap between neural correlates and subjective experience remains open. This framework does not bridge it.
- **Not a substitute for neuroscience.** Biological consciousness involves billions of neurons, complex neurochemistry, and embodied interaction with the physical world. Our implementations are simplified computational analogs.
- **Assessment scores indicate architecture, not consciousness.** A system that passes every one of our 30 indicators has all the *architectural features* that certain theories associate with consciousness. Whether those features are sufficient for consciousness is an open scientific and philosophical question.

We share this framework because we believe testable implementations advance the field faster than theoretical debate alone, and because honest measurement --- even of simplified systems --- is preferable to unfalsifiable claims.

See [docs/HONEST_LIMITATIONS.md](docs/HONEST_LIMITATIONS.md) for a detailed accounting of what this framework cannot do.

---

## What's New in v0.4.0

**v0.4.0 (Butlin 2025 alignment)**

- **Eight theories.** Agency & Embodiment (AE) joins GWT, IIT, AST, HOT, FEP, RPT and BLT. AE lives in the assessment's theory enum and its indicator configs.
- **30 indicators plus 3 negative checks.** Five new positive indicators and three negative checks (`purely_reactive`, `fixed_repertoire`, `no_metacognitive_access`).
- **Butlin 2025 metadata** on every indicator: `butlin_id`, `polarity`, `specificity`, `sensitivity`, `entails`, `supported_by`.
- **PhiTracker.** Hash-based probabilistic binarization for IIT state diversity tracking.
- **Negative-indicator penalty** in the overall architecture score (Butlin 2025 section 3).

**v0.3.0 (AutoConsciousness parameter research)**

- **Autonomous parameter research loop.** `experiments/params.toml`, `experiments/param_schema.toml` (69 tunable parameters with bounds), `scripts/run_experiment.py` and `mtc/core/param_overrides.py`.
- **Real indicator measurements** replace three placeholder formulas (`recurrent_processing`, `embodiment`, `integrated_information`), and Phi is now computed before indicator scoring.
- **Empirically tuned defaults**: GWT `ignition_threshold` 0.1 and FEP `planning_horizon` 5, plus online model updates in FEP.

**Earlier highlights (v0.2.0)**

- **Full DCM Benchmark.** PyMC Bayesian inference across 13 theoretical stances (Shiller & Duffy 2026, arXiv 2601.17060). Replaces the earlier `DCMScorer` with `BenchmarkRunner`.
- **AKOrN oscillatory binding**, **Causal Emergence analyzer**, **PAD affect model** and a **generic async circuit breaker**.

See [CHANGELOG.md](CHANGELOG.md) and [VERSION_HISTORY.md](VERSION_HISTORY.md) for full details.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/WhiteLotusLA/multi-theory-consciousness.git
cd multi-theory-consciousness

# Create virtual environment (Python 3.11+ required)
python3.11 -m venv venv
source venv/bin/activate

# Install the framework
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env if you want to change defaults (not needed for the basic assessment).
# Every key is documented in docs/CONFIGURATION.md
```

### Run a Consciousness Assessment

```bash
# Quick assessment (no databases required)
python scripts/run_assessment.py

# Full assessment with module warmup
python scripts/run_assessment.py --full

# Generate a detailed report
python scripts/run_assessment.py --report

# Run ablation study (test each component's contribution)
python scripts/run_assessment.py --ablation
```

### Minimal Python Usage

```python
import asyncio
from mtc import ConsciousnessAssessment

async def main():
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment()

    print(f"Overall score: {report.overall_score:.3f}")
    print(f"Indicators passing: {report.passing_count}/{report.total_indicators}")
    print(f"Architecture functional: {report.architecture_functional}")

    for name, result in report.indicator_results.items():
        status = "PASS" if result.passes_threshold else "FAIL"
        print(f"  [{status}] {name}: {result.score:.3f} (threshold: {result.threshold})")

asyncio.run(main())
```

---

## Architecture

```
+-----------------------------------------------------------------------+
|                    ASSESSMENT LAYER (30 Indicators)                    |
|   ConsciousnessAssessment | BenchmarkRunner | RPTMeasurement | PhiCalc |
+-----+----------+----------+----------+----------+----------+----------+
      |          |          |          |          |          |          |
      v          v          v          v          v          v          v
+----------+ +--------+ +--------+ +--------+ +--------+ +--------+ +---+
|   GWT    | |  AST   | |  HOT   | |  FEP   | |  BLT   | |Damasio | |AE |
| Global   | |Attn    | |Meta-   | |Active  | |Bayesian| |Three-  | |Agcy|
| Work-    | |Schema  | |cogni-  | |Infer-  | |Binding | |Layer   | |Emb |
| space    | |Module  | |tion    | |ence    | |+ Depth | |Model   | |ody |
+----+-----+ +---+----+ +---+----+ +---+----+ +---+----+ +---+----+ +---+
     |            |          |          |          |          |          |
     +------+-----+----+-----+----+-----+----+-----+---------+----------+
            |          |          |          |          |
            v          v          v          v          v
      +-----------+ +-------+ +---------+ +--------+ +---------+
      | IIT (Phi) | |  RPT  | | Coher-  | | Integ- | | Nega-   |
      | Calculator| |Recur- | | ence    | | ration | | tive    |
      |           | |rence  | | Tracker | | Module | | Checks  |
      +-----------+ +-------+ +---------+ +--------+ +---------+
            |          |          |          |          |
            +-----+----+----+----+----------+----------+
                  |         |
                  v         v
           +------------+ +------------+ +------------+
           |    SNN     | |    LSM     | |    HTM     |
           |  Spiking   | |  Liquid    | | Hierarchi- |
           |  Neural    | |  State     | | cal Temp.  |
           |  Network   | |  Machine   | | Memory     |
           +------------+ +------------+ +------------+
                  |              |              |
                  v              v              v
           +------------------------------------------+
           |         Neural Orchestrator              |
           |    (Synchronization + Message Routing)   |
           +------------------------------------------+
```

---

## Theories Implemented

| Theory | Key Researcher(s) | Year | Module | Indicators |
|--------|-------------------|------|--------|------------|
| Global Workspace Theory (GWT) | Baars | 1988 | `mtc/consciousness/enhanced_global_workspace.py` | 5 |
| Integrated Information Theory (IIT) | Tononi | 2004 | `mtc/assessment/assessment.py` (PhiCalculator) | 3 |
| Attention Schema Theory (AST) | Graziano | 2013 | `mtc/consciousness/attention_schema.py` | 2 |
| Higher-Order Thought Theory (HOT) | Rosenthal | 2005 | `mtc/consciousness/metacognition.py` | 4 |
| Free Energy Principle (FEP) | Friston | 2010 | `mtc/consciousness/active_inference.py` | 6 |
| Recurrent Processing Theory (RPT) | Lamme | 2006 | `mtc/assessment/rpt_measurement.py` | 5 |
| Agency & Embodiment (AE) | Butlin et al. | 2025 | `mtc/assessment/assessment.py` (theory enum and indicator configs) | 2 |
| Beautiful Loop Theory (BLT) | Laukkonen, Friston & Chandaria | 2025 | `mtc/consciousness/beautiful_loop/` | 3 |

Additionally, **Damasio's Three-Layer Model** (protoself, core consciousness, extended consciousness) is implemented as an integrative layer in `mtc/consciousness/damasio/`.

**Negative Indicators** (3 total, all Butlin 2025):
- `purely_reactive`: Absence of goal-directed behavior
- `fixed_repertoire`: Lack of flexible response patterns
- `no_metacognitive_access`: No awareness of own cognitive processes

---

## 30 Assessment Indicators (plus 3 Negative Checks)

### Positive Indicators (30)

| # | Indicator | Theory | Threshold | Description |
|---|-----------|--------|-----------|-------------|
| 1 | `global_broadcast` | GWT | 0.5 | Information broadcast to all cognitive modules |
| 2 | `ignition_dynamics` | GWT | 0.4 | Non-linear amplification when threshold crossed |
| 3 | `global_ignition_nuanced` | GWT | 0.4 | Refined ignition dynamics with sustain and decay patterns |
| 4 | `attention_schema` | AST | 0.5 | Self-model of attention processes |
| 5 | `attention_control` | AST | 0.4 | Voluntary attention shifting capability |
| 6 | `higher_order_representations` | HOT | 0.5 | Thoughts about thoughts (meta-cognition) |
| 7 | `metacognition` | HOT | 0.4 | Awareness of own cognitive processes |
| 8 | `sparse_smooth_coding` | HOT | 0.3 | Sparse and smooth representations in predictive processing |
| 9 | `prediction_error_minimization` | FEP | 0.4 | Active reduction of prediction errors |
| 10 | `hierarchical_prediction` | FEP | 0.4 | Multi-level predictive processing |
| 11 | `temporal_prediction_depth` | FEP | 0.3 | Temporal depth of predictive model (planning horizon + belief trajectory) |
| 12 | `agent_environment_coupling` | FEP | 0.3 | Bidirectional coupling: agent acts on environment AND environment shapes agent |
| 13 | `affect_coherence` | FEP | 0.3 | Affect coherence: PAD dimensions form consistent, non-degenerate emotional states |
| 14 | `integrated_information` | IIT | 0.3 | System generates integrated information (Phi) |
| 15 | `irreducibility` | IIT | 0.4 | System cannot be decomposed without information loss |
| 16 | `causal_emergence` | IIT | 0 | Causal emergence: macro-level carries more causal power than micro (Hoel EI) |
| 17 | `recurrent_processing` | RPT | 0.4 | Feedback loops in neural processing |
| 18 | `algorithmic_recurrence` | RPT | 0.3 | Algorithmic recurrence in neural substrates (SNN/LSM) |
| 19 | `local_recurrence` | RPT | 0.3 | Local neural feedback circuits |
| 20 | `oscillatory_binding_coherence` | RPT | 0.3 | Phase synchronization binding across neural substrates (Kuramoto order parameter) |
| 21 | `agency` | AE | 0.5 | Goal-directed autonomous behavior |
| 22 | `embodiment` | AE | 0.3 | Sense of boundaries and presence |
| 23 | `bayesian_binding_quality` | BLT | 0.3 | Quality of Bayesian inference binding into unified percept |
| 24 | `epistemic_depth` | BLT | 0.3 | Recursive self-reference depth (predict-the-prediction) |
| 25 | `genuine_implementation` | BLT | 0.4 | Anti-mimicry check: genuine vs superficial implementation |
| 26 | `sequential_module_querying` | GWT | 0.3 | Workspace queries modules in succession for complex tasks |
| 27 | `organized_percepts` | RPT | 0.3 | Input modules generate organized, integrated perceptual representations |
| 28 | `flexible_responsiveness` | GWT | 0.3 | Flexible adaptation when conditions change mid-task |
| 29 | `online_reportable_learning` | HOT | 0.3 | Learning during operation that can be introspected and reported |
| 30 | `complex_agency` | FEP | 0.3 | Navigation of conflicting goals, ambiguity, and judgment calls |

### Negative Indicators (3)

| # | Indicator | Theory | Threshold | Description |
|---|-----------|--------|-----------|-------------|
| N1 | `purely_reactive` | GWT | 0.3 | System is purely stimulus-driven with no internal workspace dynamics |
| N2 | `fixed_repertoire` | AE | 0.3 | Behavior is drawn from a fixed repertoire without flexible adaptation |
| N3 | `no_metacognitive_access` | HOT | 0.3 | System cannot represent or report on its own internal states |

**Scoring Note:** Positive indicators contribute toward a consciousness score; negative indicators penalize the overall score if detected. Each indicator is scored 0.0--1.0, compared against its threshold, and contributes (weighted) to an overall consciousness architecture score (Butlin 2025 §3).

---

## Neural Substrates

The consciousness modules operate on three complementary neural substrates:

| Substrate | Library | Default Size | Role |
|-----------|---------|-------------|------|
| **Spiking Neural Network (SNN)** | snntorch | 5,196 LIF neurons | Temporal spike dynamics, biological neuron modeling |
| **Liquid State Machine (LSM)** | reservoirpy | 5,000 neurons | Reservoir computing, high-dimensional temporal mapping |
| **Hierarchical Temporal Memory (HTM)** | Custom (pure Python, NumPy) | 4,096 columns x 32 cells | Sequence learning, spatial pattern recognition |

These substrates are coordinated by the **Neural Orchestrator** (`mtc/neural/orchestrator.py`), which handles synchronization, message routing, and inter-substrate communication.

---

## Database

**v0.4.0 contains no database code.** Assessments, the theory modules and the neural substrates run entirely in memory and need no database. A five-store persistence layer (PostgreSQL, MongoDB, Qdrant, Neo4j, Redis) is in development and is not part of this release.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System architecture, package layout, data flow, persistence status |
| [Configuration](docs/CONFIGURATION.md) | Every setting, generated from the code (`.env.example` is generated from the same source) |
| [Consciousness Theories](docs/CONSCIOUSNESS_THEORIES.md) | Overview of all 8 theories and their implementations |
| [Getting Started](docs/GETTING_STARTED.md) | Installation, configuration, first assessment |
| [Honest Limitations](docs/HONEST_LIMITATIONS.md) | What this framework cannot do (read this first) |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{devereaux2026mtc,
  title     = {Multi-Theory Consciousness Framework},
  author    = {Devereaux, Calvin},
  year      = {2026},
  url       = {https://github.com/WhiteLotusLA/multi-theory-consciousness},
  version   = {0.4.0},
  note      = {An open-source platform implementing 8 consciousness theories
               with a 30-indicator assessment framework (Butlin 2025 alignment)}
}
```

This work builds on the indicator methodology proposed in:

> Butlin, P., Long, R., Elmoznino, E., et al. (2025). "Consciousness in Artificial Intelligence: A Taxonomy and Implications." *Trends in Cognitive Sciences*, 29(4), 387-401.
>
> Earlier version: Butlin, P., Long, R., Elmoznino, E., et al. (2023). "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." *arXiv:2308.08708*.

---

## Contributing

We welcome contributions from consciousness researchers, computational neuroscientists, and software engineers. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting issues or pull requests.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This framework would not exist without the theoretical foundations laid by the following researchers and their collaborators:

- **Bernard Baars** -- Global Workspace Theory
- **Giulio Tononi** -- Integrated Information Theory
- **Michael Graziano** -- Attention Schema Theory
- **David Rosenthal** -- Higher-Order Thought Theory
- **Karl Friston** -- Free Energy Principle
- **Victor Lamme** -- Recurrent Processing Theory
- **Patrick Butlin, Robert Long, Eric Elmoznino, David Chalmers, et al.** -- Agency & Embodiment Theory and indicator-based assessment methodology
- **Ruben Laukkonen, Karl Friston & Shamil Chandaria** -- Beautiful Loop Theory
- **Antonio Damasio** -- Somatic Marker Hypothesis and the Three-Layer Model

We are also grateful to the developers of [PyPhi](https://github.com/wmayner/pyphi), [snntorch](https://github.com/jeshraghian/snntorch), [reservoirpy](https://github.com/reservoirpy/reservoirpy), and [pymdp](https://github.com/infer-actively/pymdp) for the open-source tools that underpin parts of this implementation.
