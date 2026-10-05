# Getting Started

## Prerequisites

- **Python 3.11+** (3.12 also supported)
- **pip** (or your preferred package manager)
- **Git**

### Databases

- None for v0.4.0. This release contains no database code; a five-store persistence layer (PostgreSQL, MongoDB, Qdrant, Neo4j, Redis) is in development.

You can run assessments and experiment with individual theory modules without any databases.

---

## Installation

```bash
# Clone
git clone https://github.com/WhiteLotusLA/multi-theory-consciousness.git
cd multi-theory-consciousness

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core framework
pip install -e .

# Install with neural substrates (SNN, LSM)
pip install -e ".[neural]"

# Install with active inference (FEP)
pip install -e ".[fep]"

# Install everything
pip install -e ".[all]"
```

## Configuration

```bash
cp .env.example .env
# Edit .env only if you want to change a default (optional for basic usage)
```

Every setting is read through the `Settings` class in `mtc/core/config.py`. Environment keys carry the `MTC_` prefix and are named after the `Settings` fields. The full list, with types and defaults, is in [CONFIGURATION.md](CONFIGURATION.md), and `.env.example` is generated from the same source, so the two always agree. A machine with no `.env` file at all works: every setting has a default.

---

## First Assessment

Run the built-in assessment to verify installation:

```bash
# Quick assessment (no databases needed)
python scripts/run_assessment.py

# Full assessment
python scripts/run_assessment.py --full

# With detailed report
python scripts/run_assessment.py --report
```

Expected output has this shape (values elided):

```
================================================================
  CONSCIOUSNESS ASSESSMENT REPORT
================================================================
  Session:            ...
  Overall score:      ...
  Architecture OK:    ...
  Passing indicators: .../30

  Theory Scores:
    ...

  Indicators (.../30):
    [PASS] irreducibility  ... / 0.40
    [FAIL] ...
```

The assessment scores 30 indicators across eight theories. The passing count and the overall score depend on how much the modules have been warmed up (`--full` runs several cycles first), so no fixed numbers are given here.

Scores on a fresh system will be lower than a system that has been running and accumulating activity data. This is expected.

---

## Using Individual Theories

Each theory module can be used independently:

```python
from mtc.consciousness.enhanced_global_workspace import EnhancedGlobalWorkspace
from mtc.consciousness.attention_schema import AttentionSchemaModule
from mtc.consciousness.metacognition import MetacognitionModule
from mtc.consciousness.active_inference import ActiveInferenceModule

# Initialize a single module
gwt = EnhancedGlobalWorkspace()
await gwt.initialize()

# Create workspace candidates and run competition
# See examples/single_theory.py for full usage
```

---

## Using Neural Substrates

```python
from mtc.neural.spiking.production_snn import ProductionSNN
from mtc.neural.liquid.production_lsm import ProductionLSM
from mtc.neural.htm.production_htm import ProductionHTM

# Initialize neural layers
snn = ProductionSNN()
lsm = ProductionLSM()
htm = ProductionHTM()

# Process input through the pipeline
# See examples/neural_substrates.py for full usage
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Consciousness theory tests only
pytest tests/consciousness/ -v

# Neural substrate tests
pytest tests/neural/ -v

# With coverage
pytest tests/ -v --cov=mtc
```

---

## Next Steps

- Read [HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md) to understand what this framework can and cannot do
- Read [CONSCIOUSNESS_THEORIES.md](CONSCIOUSNESS_THEORIES.md) for theory backgrounds
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
- Explore the `examples/` directory for usage patterns
