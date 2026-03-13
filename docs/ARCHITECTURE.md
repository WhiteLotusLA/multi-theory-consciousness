# Architecture

## Overview

The Multi-Theory Consciousness (MTC) framework is organized into three layers:

1. **Neural Substrates** — the computational medium (SNN, LSM, HTM)
2. **Consciousness Modules** — theory implementations that operate on the substrates
3. **Assessment Layer** — measurement of consciousness indicators

```
+-----------------------------------------------------------------------+
|                       ASSESSMENT LAYER                                 |
|  ConsciousnessAssessment | DCMScorer | RPTMeasurement | PhiCalculator |
+-----------------------------------------------------------------------+
                                |
+-----------------------------------------------------------------------+
|                    CONSCIOUSNESS MODULES                               |
|  GWT | AST | HOT | FEP | BLT | Damasio | IIT | RPT | Coherence      |
+-----------------------------------------------------------------------+
                                |
+-----------------------------------------------------------------------+
|                      NEURAL SUBSTRATES                                 |
|  SNN (Spiking) | LSM (Liquid State) | HTM (Hierarchical Temporal)    |
|                    Neural Orchestrator                                 |
+-----------------------------------------------------------------------+
```

---

## Package Structure

```
mtc/
├── consciousness/           # Theory implementations
│   ├── enhanced_global_workspace.py  # GWT (Baars)
│   ├── global_workspace.py           # Simplified GWT
│   ├── attention_schema.py           # AST (Graziano)
│   ├── metacognition.py              # HOT (Rosenthal)
│   ├── active_inference.py           # FEP (Friston)
│   ├── consciousness_integration.py  # Cross-theory integration
│   ├── consciousness_metrics.py      # Metric collection
│   ├── conversation_coherence.py     # Topic/reference tracking
│   ├── beautiful_loop/               # BLT (Laukkonen et al.)
│   │   ├── hyper_model.py            #   Precision controller
│   │   ├── bayesian_binding.py       #   Inference binding
│   │   └── epistemic_depth.py        #   Self-reference depth
│   └── damasio/                      # Three-Layer Model
│       ├── protoself.py              #   Body-state representation
│       ├── core_consciousness.py     #   Self-world binding
│       └── extended_consciousness.py #   Autobiographic self
│
├── neural/                  # Computational substrates
│   ├── spiking/             # Spiking Neural Networks
│   │   ├── snn_core.py      #   Core SNN implementation
│   │   ├── snn_core_optimized.py  #   Optimized variant
│   │   └── production_snn.py      #   Production-ready SNN
│   ├── liquid/              # Liquid State Machines
│   │   ├── lsm_core.py      #   Core LSM implementation
│   │   └── production_lsm.py     #   Production-ready LSM
│   ├── htm/                 # Hierarchical Temporal Memory
│   │   ├── htm_core.py      #   Core HTM implementation
│   │   └── production_htm.py     #   Production-ready HTM
│   ├── protocols/           # Inter-substrate communication
│   │   ├── message_format.py
│   │   ├── routing.py
│   │   ├── synchronization.py
│   │   └── serialization.py
│   ├── orchestrator.py      # Coordinates all substrates
│   ├── base_interfaces.py   # Protocol definitions
│   ├── memory_pool.py       # Memory management
│   ├── gpu_integration.py   # GPU acceleration
│   └── mps_utils.py         # Metal Performance Shaders
│
├── assessment/              # Measurement framework
│   ├── assessment.py        # 20-indicator assessment
│   ├── framework.py         # Measurement framework
│   ├── rpt_measurement.py   # RPT indicators
│   └── dcm_scoring.py       # Digital Consciousness Model
│
└── core/                    # Configuration
    └── config.py            # Pydantic Settings
```

---

## Consciousness Cycle

A single consciousness processing cycle:

```
Input Signal
     |
     v
[1] Neural Substrates process input
     |-- SNN: spike-based temporal dynamics
     |-- LSM: reservoir state evolution
     |-- HTM: sequence and pattern matching
     |
     v
[2] Attention Schema (AST) models attention
     |-- What is the system attending to?
     |-- Voluntary vs. captured attention?
     |
     v
[3] Active Inference (FEP) predicts and plans
     |-- Generate predictions about observations
     |-- Compute prediction errors
     |-- Update beliefs or select actions
     |
     v
[4] Global Workspace (GWT) competition
     |-- Candidates enter workspace
     |-- Competition based on salience
     |-- Winners undergo ignition
     |-- Broadcast to all modules
     |
     v
[5] Metacognition (HOT) reflects
     |-- Generate higher-order thoughts about broadcast
     |-- Self-evaluate confidence and uncertainty
     |
     v
[6] Beautiful Loop (BLT) enriches
     |-- HyperModel adjusts precision weights
     |-- BayesianBinding unifies percepts
     |-- EpistemicDepth measures self-reference
     |
     v
[7] Damasio layers ground experience
     |-- Protoself: body-state representation
     |-- Core: bind self to workspace content
     |-- Extended: temporal and narrative context
     |
     v
[8] Integrated state produced
     |-- ConsciousnessState with all module outputs
     |-- Available for assessment measurement
```

---

## Neural Substrates

### Spiking Neural Network (SNN)

- **Library:** snntorch
- **Architecture:** Input layer -> Hidden (50 LIF neurons) -> Output
- **Learning:** STDP (Spike-Timing-Dependent Plasticity)
- **Role:** Biological plausibility, temporal spike dynamics

### Liquid State Machine (LSM)

- **Library:** reservoirpy
- **Architecture:** 2,000 reservoir neurons, RLS readout
- **Properties:** Edge-of-chaos dynamics, fading memory
- **Role:** High-dimensional temporal mapping, intuition analog

### Hierarchical Temporal Memory (HTM)

- **Architecture:** 4,096 columns x 32 cells = 131,072 cells
- **Implementation:** C++ with Metal GPU acceleration
- **Learning:** Hebbian, sequence memory
- **Role:** Pattern recognition, temporal sequence learning

### Neural Orchestrator

Coordinates all three substrates:
- **Synchronization** — phase-locked processing across substrates
- **Message routing** — typed messages between components
- **State aggregation** — unified neural state for consciousness modules

---

## Assessment Framework

The assessment measures 20 indicators derived from Butlin, Chalmers et al. (2023):

1. Each indicator maps to a specific consciousness theory
2. Scored 0.0-1.0 via theory-specific scoring functions
3. Compared against calibrated thresholds
4. Weighted and aggregated to overall score
5. "Architecture functional" declared when sufficient indicators pass

### Ablation Studies

The framework supports ablation testing: selectively disabling individual modules to measure their contribution to overall scores. This reveals which theories contribute most to the system's architectural features.

### DCM Scoring

The Digital Consciousness Model (DCM) provides a 13-perspective probabilistic assessment with credence scores, offering a complementary view to the indicator-based approach.

---

## Database Schema (Optional)

When databases are connected, the framework can persist:

| Database | Purpose | Required? |
|----------|---------|-----------|
| PostgreSQL | Episodic/semantic memory, conversation history | Optional |
| MongoDB | Neural state snapshots, thought streams | Optional |
| Qdrant | Vector embeddings for semantic search | Optional |
| Neo4j | Knowledge graph (concepts, relations) | Optional |
| Redis | Working memory cache (7+/-2 items) | Optional |

All databases are optional. The framework can run assessments and process consciousness cycles without any database connections.
