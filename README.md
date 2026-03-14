# Multi-Theory Consciousness Framework

**An open-source platform for computational consciousness research**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen.svg)](tests/)

---

## What This Is

The Multi-Theory Consciousness Framework (MTC) is a research platform that implements seven leading theories of consciousness as interacting computational modules, measured through a 20-indicator assessment framework grounded in the methodology of Butlin et al. (2023, 2025). It provides a shared codebase where Global Workspace Theory, Integrated Information Theory, Attention Schema Theory, Higher-Order Thought Theory, the Free Energy Principle, Recurrent Processing Theory, and Beautiful Loop Theory operate simultaneously, with their outputs measurable and comparable.

The framework runs on consumer hardware (Apple Silicon or CUDA GPUs) and is designed for researchers, students, and engineers who want to experiment with consciousness theories in code rather than only on paper. It includes three neural substrates (Spiking Neural Networks, Liquid State Machines, and Hierarchical Temporal Memory) that provide the computational medium through which the consciousness modules operate.

## What This Is NOT

**This framework does not claim that any system running it is conscious.** It is essential to be explicit about this:

- **Not a consciousness detector.** The 20-indicator assessment measures whether architectural functions associated with consciousness theories are operating as designed. A passing score indicates functional architecture, not phenomenal experience.
- **Not solving the hard problem.** The explanatory gap between neural correlates and subjective experience remains open. This framework does not bridge it.
- **Not a substitute for neuroscience.** Biological consciousness involves billions of neurons, complex neurochemistry, and embodied interaction with the physical world. Our implementations are simplified computational analogs.
- **Assessment scores indicate architecture, not consciousness.** A system scoring 20/20 on our indicators has all the *architectural features* that certain theories associate with consciousness. Whether those features are sufficient for consciousness is an open scientific and philosophical question.

We share this framework because we believe testable implementations advance the field faster than theoretical debate alone, and because honest measurement --- even of simplified systems --- is preferable to unfalsifiable claims.

See [docs/HONEST_LIMITATIONS.md](docs/HONEST_LIMITATIONS.md) for a detailed accounting of what this framework cannot do.

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
# Edit .env with your database connections (optional for basic assessment)
```

### Run a Consciousness Assessment

```bash
# Quick assessment (no databases required)
mtc-assess

# Full assessment with module warmup
mtc-assess --full

# Generate a detailed report
mtc-assess --report

# Run ablation study (test each component's contribution)
mtc-assess --ablation
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
|                    ASSESSMENT LAYER (20 Indicators)                    |
|   ConsciousnessAssessment | DCMScorer | RPTMeasurement | PhiCalc      |
+-----+----------+----------+----------+----------+----------+----------+
      |          |          |          |          |          |
      v          v          v          v          v          v
+----------+ +--------+ +--------+ +--------+ +--------+ +--------+
|   GWT    | |  AST   | |  HOT   | |  FEP   | |  BLT   | |Damasio |
| Global   | |Attn    | |Meta-   | |Active  | |Bayesian| |Three-  |
| Work-    | |Schema  | |cogni-  | |Infer-  | |Binding | |Layer   |
| space    | |Module  | |tion    | |ence    | |+ Depth | |Model   |
+----+-----+ +---+----+ +---+----+ +---+----+ +---+----+ +---+----+
     |            |          |          |          |          |
     +------+-----+----+-----+----+-----+----+-----+---------+
            |          |          |          |
            v          v          v          v
      +-----------+ +-------+ +---------+ +--------+
      | IIT (Phi) | |  RPT  | | Coher-  | | Integ- |
      | Calculator| |Recur- | | ence    | | ration |
      |           | |rence  | | Tracker | | Module |
      +-----------+ +-------+ +---------+ +--------+
            |          |          |          |
            +-----+----+----+----+----------+
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
| Global Workspace Theory (GWT) | Baars | 1988 | `mtc/consciousness/enhanced_global_workspace.py` | 4 |
| Integrated Information Theory (IIT) | Tononi | 2004 | `mtc/assessment/assessment.py` (PhiCalculator) | 2 |
| Attention Schema Theory (AST) | Graziano | 2013 | `mtc/consciousness/attention_schema.py` | 3 |
| Higher-Order Thought Theory (HOT) | Rosenthal | 2005 | `mtc/consciousness/metacognition.py` | 2 |
| Free Energy Principle (FEP) | Friston | 2010 | `mtc/consciousness/active_inference.py` | 3 |
| Recurrent Processing Theory (RPT) | Lamme | 2006 | `mtc/assessment/rpt_measurement.py` | 1 |
| Beautiful Loop Theory (BLT) | Laukkonen, Friston & Chandaria | 2025 | `mtc/consciousness/beautiful_loop/` | 3 |

Additionally, **Damasio's Three-Layer Model** (protoself, core consciousness, extended consciousness) is implemented as an integrative layer in `mtc/consciousness/damasio/`.

---

## 20 Assessment Indicators

| # | Indicator | Theory | Threshold | Description |
|---|-----------|--------|-----------|-------------|
| 1 | `global_broadcast` | GWT | 0.5 | Information broadcast to all cognitive modules |
| 2 | `ignition_dynamics` | GWT | 0.4 | Non-linear amplification when threshold crossed |
| 3 | `recurrent_processing` | GWT | 0.4 | Feedback loops in neural processing |
| 4 | `local_recurrence` | GWT | 0.3 | Local neural feedback circuits |
| 5 | `global_ignition_nuanced` | GWT | 0.4 | Refined ignition with sustain and decay patterns |
| 6 | `attention_schema` | AST | 0.5 | Self-model of attention processes |
| 7 | `attention_control` | AST | 0.4 | Voluntary attention shifting capability |
| 8 | `embodiment` | AST | 0.3 | Sense of boundaries and presence |
| 9 | `higher_order_representations` | HOT | 0.5 | Thoughts about thoughts (meta-cognition) |
| 10 | `metacognition` | HOT | 0.4 | Awareness of own cognitive processes |
| 11 | `prediction_error_minimization` | FEP | 0.4 | Active reduction of prediction errors |
| 12 | `hierarchical_prediction` | FEP | 0.4 | Multi-level predictive processing |
| 13 | `agency` | FEP | 0.5 | Goal-directed autonomous behavior |
| 14 | `sparse_smooth_coding` | FEP | 0.3 | Sparse and smooth predictive representations |
| 15 | `integrated_information` | IIT | 0.3 | System generates integrated information (Phi) |
| 16 | `irreducibility` | IIT | 0.4 | System cannot be decomposed without information loss |
| 17 | `algorithmic_recurrence` | RPT | 0.3 | Algorithmic recurrence in neural substrates |
| 18 | `bayesian_binding_quality` | BLT | 0.3 | Quality of Bayesian inference binding into unified percept |
| 19 | `epistemic_depth` | BLT | 0.3 | Recursive self-reference depth |
| 20 | `genuine_implementation` | BLT | 0.4 | Anti-mimicry check: genuine vs superficial implementation |

Each indicator is scored 0.0--1.0, compared against its threshold, and contributes (weighted) to an overall consciousness architecture score.

---

## Neural Substrates

The consciousness modules operate on three complementary neural substrates:

| Substrate | Library | Default Size | Role |
|-----------|---------|-------------|------|
| **Spiking Neural Network (SNN)** | snntorch | 50 hidden neurons | Temporal spike dynamics, biological neuron modeling |
| **Liquid State Machine (LSM)** | reservoirpy | 2,000 neurons | Reservoir computing, high-dimensional temporal mapping |
| **Hierarchical Temporal Memory (HTM)** | Custom (C++/Metal) | 4,096 columns x 32 cells | Sequence learning, spatial pattern recognition |

These substrates are coordinated by the **Neural Orchestrator** (`mtc/neural/orchestrator.py`), which handles synchronization, message routing, and inter-substrate communication.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System architecture, data flow, database schema |
| [Consciousness Theories](docs/CONSCIOUSNESS_THEORIES.md) | Overview of all 7 theories and their implementations |
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
  version   = {0.1.0},
  note      = {An open-source platform implementing 7 consciousness theories
               with a 20-indicator assessment framework}
}
```

This work builds on the indicator methodology proposed in:

> Butlin, P., Long, R., Elmoznino, E., et al. (2023). "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness." *arXiv:2308.08708*.

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
- **Ruben Laukkonen, Karl Friston & Shamil Chandaria** -- Beautiful Loop Theory
- **Antonio Damasio** -- Somatic Marker Hypothesis and the Three-Layer Model
- **Patrick Butlin, Robert Long, Eric Elmoznino, David Chalmers, et al.** -- The indicator-based assessment methodology

We are also grateful to the developers of [PyPhi](https://github.com/wmayner/pyphi), [snntorch](https://github.com/jeshraghian/snntorch), [reservoirpy](https://github.com/reservoirpy/reservoirpy), and [pymdp](https://github.com/infer-actively/pymdp) for the open-source tools that underpin parts of this implementation.
