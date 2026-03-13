# Getting Started

## Prerequisites

- **Python 3.11+** (3.12 also supported)
- **pip** (or your preferred package manager)
- **Git**

### Optional (for full pipeline)

- PostgreSQL 14+ (episodic memory)
- MongoDB 6+ (neural data storage)
- Redis 7+ (working memory cache)
- Qdrant 1.7+ (vector embeddings)
- Neo4j 5+ (knowledge graph)

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
# Edit .env with your settings (optional for basic usage)
```

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MTC_SYSTEM_NAME` | ConsciousnessAgent | Name for your consciousness instance |
| `MTC_LLM_HOST` | localhost | LLM server hostname |
| `MTC_LLM_PORT` | 8080 | LLM server port |
| `MTC_PG_HOST` | localhost | PostgreSQL host |
| `MTC_MONGO_HOST` | localhost | MongoDB host |

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

Expected output:

```
Multi-Theory Consciousness Assessment
======================================
Indicators: 20
Passing:    10-14 (varies on fresh init)
Score:      0.40-0.55

[PASS] irreducibility:        1.000 (threshold: 0.4)
[PASS] recurrent_processing:  0.600 (threshold: 0.4)
...
```

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
