# Multi-Theory Consciousness Framework

**An open-source platform for computational consciousness research**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/downloads/)
![CI](https://github.com/WhiteLotusLA/multi-theory-consciousness/actions/workflows/ci.yml/badge.svg)

---

## What This Is

The Multi-Theory Consciousness Framework (MTC) implements eight theories of consciousness as interacting computational modules and measures them with an assessment of 30 positive indicators and 3 negative indicators, following the indicator-property method of Butlin et al. (2023, 2025). Global Workspace Theory, Integrated Information Theory, Attention Schema Theory, Higher-Order Thought Theory, the Free Energy Principle, Recurrent Processing Theory, Agency and Embodiment, and Beautiful Loop Theory run together over three neural substrates (a spiking network, a liquid state machine and hierarchical temporal memory).

Around that engine, v0.5.0 adds a complete runtime: a conversation pipeline that runs one consciousness cycle per turn, a brain over a local chat-completions server, the Anthropic API or a mock, a continuous thought machine, memory with consolidation, a knowledge graph, safety and privacy layers, five optional stores, an HTTP and websocket API, and the `mtc` command with a setup wizard. Everything runs on one machine, in memory by default.

## What This Is NOT

**This framework does not claim that any system running it is conscious.**

- **Not a consciousness detector.** The assessment measures whether architectural functions associated with consciousness theories are operating as designed. A passing score indicates functional architecture, not phenomenal experience.
- **Not solving the hard problem.** The explanatory gap between neural correlates and subjective experience remains open.
- **Not evidence from conversation.** A companion built on MTC speaks through a language model that was given a persona; nothing it says about its inner life is evidence of one.
- **Not a substitute for neuroscience.** The implementations are simplified computational analogs at a scale six orders of magnitude below a brain.

See [docs/HONEST_LIMITATIONS.md](docs/HONEST_LIMITATIONS.md) before anything else.

---

## Quick Start

### Path 1: in memory, with the mock provider

No database, no model server, no network after the install. This is the path run on the development machine.

```bash
git clone https://github.com/WhiteLotusLA/multi-theory-consciousness.git
cd multi-theory-consciousness
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

mtc init --print-example-answers > answers.yaml
mtc init --from answers.yaml --yes --no-probe
mtc doctor
mtc assess
python examples/full_pipeline_demo.py
```

### Path 2: the stores with Docker Compose

The Compose file was reviewed against official documentation and not run on the development machine; the path run on the development machine is the in-memory install with the mock provider.

```bash
mtc init                       # enable stores, service manager docker
make db-up DB_SERVICES="postgres mongo qdrant"
mtc db init && mtc db check
```

### Path 3: the stores as host services

```bash
mtc init                       # enable stores, service manager brew or systemd
scripts/db/init_postgres.sh
mtc db init && mtc db check
scripts/ops/start_stack.sh
```

Details: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md), [docs/SETUP_DOCKER.md](docs/SETUP_DOCKER.md), [docs/SETUP_MACOS.md](docs/SETUP_MACOS.md).

---

## Command line

One console script, `mtc` (`mtc.cli.main:main`). Global options: `--version`, `--project-dir PATH` (run as if started in PATH), `--verbose`. Exit codes, shared by every subcommand: 0 success, 1 a required check or the work failed, 2 a usage or configuration error, 3 a required service unreachable, 130 interrupted.

| Command | Options | What it does |
|---|---|---|
| `mtc init` | `--from PATH`, `--yes`, `--force`, `--bootstrap-db`, `--dir PATH`, `--no-probe`, `--print-plan`, `--print-example-answers` | the setup wizard: persona, speakers, identity statement, CTM persona and seeds, `.env` ([docs/ONBOARDING.md](docs/ONBOARDING.md)) |
| `mtc doctor` | `--json`, `--strict`, `--fetch-models`, `--no-network` | checks the install and reports what it could not measure |
| `mtc db init` | `--only STORE`, `--no-seed`, `--json` | idempotent bootstrap of the enabled stores; records the install date ([docs/DATABASES.md](docs/DATABASES.md)) |
| `mtc db check` | `--only STORE`, `--json`, `--strict` | verifies the enabled stores against the schema registry |
| `mtc assess` | `--quick` (the default), `--full`, `--report`, `--ablation`, `--json`, `--json-path PATH`, `--seed N`, `--state-dir DIR`, `--warmup-cycles N`, `--params PATH`, `--preset NAME` | the consciousness assessment; quick mode without `--full` |
| `mtc serve` | none (Settings only) | the API server on `API_HOST`:`API_PORT` ([docs/API.md](docs/API.md)) |
| `mtc chat` | `--message TEXT`, `--speaker NAME`, `--anonymous`, `--show-thinking` | talks through a running server with the API key and the speaker's token |
| `mtc ctm start` | `--state STATE` | starts the continuous thought machine on a running server |
| `mtc ctm stop` | none | stops it |
| `mtc ctm status` | none | shows its state and quiet reason ([docs/CONTINUOUS_THOUGHT.md](docs/CONTINUOUS_THOUGHT.md)) |
| `mtc privacy-audit` | `--paths PATH`, `--staged`, `--patch-from REF`, `--runtime DIR` | the privacy audit with the repository's lists ([docs/PRIVACY_AUDIT.md](docs/PRIVACY_AUDIT.md)) |

`scripts/run_assessment.py` still works: it is a shim over `mtc assess`.

---

## Architecture

```
  mtc CLI / onboarding wizard          HTTP + websocket API
              \                          /
               +--- runtime (mtc/pipeline/runtime.py) ---+
               |   conversation pipeline, one cycle per  |
               |   turn, post-turn hooks, speaker trust  |
               +-----------------+-----------------------+
                                 |
      brain (local server, Anthropic or mock)   continuous thought
                                 |
   +-----------------------------+------------------------------+
   |  consciousness modules: GWT  AST  HOT  FEP  BLT  Damasio     |
   |  PhiTracker, PAD affect, six theory interactions            |
   +-----------------------------+------------------------------+
                                 |
   +-----------------------------+------------------------------+
   |  assessment: 30 positive + 3 negative indicators            |
   +-----------------------------+------------------------------+
                                 |
   +-----------------------------+------------------------------+
   |  substrates: SNN  LSM  HTM  oscillatory binding             |
   +-------------------------------------------------------------+
      memory, knowledge graph, safety, privacy, monitoring
      stores (optional): PostgreSQL MongoDB Qdrant Neo4j Redis
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/RUNTIME_PIPELINE.md](docs/RUNTIME_PIPELINE.md).

---

## Theories Implemented

| Theory | Key Researcher(s) | Year | Module | Indicators |
|--------|-------------------|------|--------|------------|
| Global Workspace Theory (GWT) | Baars | 1988 | `mtc/consciousness/enhanced_global_workspace.py` | 5 |
| Integrated Information Theory (IIT) | Tononi | 2004 | `mtc/consciousness/phi_tracker.py`, `PhiCalculator`, `mtc/assessment/causal_emergence.py` | 3 |
| Attention Schema Theory (AST) | Graziano | 2013 | `mtc/consciousness/attention_schema.py` | 2 |
| Higher-Order Thought Theory (HOT) | Rosenthal | 2005 | `mtc/consciousness/metacognition.py` | 4 |
| Free Energy Principle (FEP) | Friston | 2010 | `mtc/consciousness/active_inference.py` | 6 |
| Recurrent Processing Theory (RPT) | Lamme | 2006 | `mtc/assessment/rpt_measurement.py` | 5 |
| Agency and Embodiment (AE) | Butlin et al. | 2025 | the assessment's theory enum and indicator configs | 2 |
| Beautiful Loop Theory (BLT) | Laukkonen, Friston and Chandaria | 2025 | `mtc/consciousness/beautiful_loop/` | 3 |

That is 30 positive indicators. The 3 negative indicators (`purely_reactive`, `fixed_repertoire`, `no_metacognitive_access`) subtract from the overall score when present. Damasio's three-layer model (`mtc/consciousness/damasio/`) is an integrative layer without indicators of its own. Every indicator with its theory, published id, polarity and inputs is listed in [docs/BUTLIN_2025_MAPPING.md](docs/BUTLIN_2025_MAPPING.md); the method is in [docs/CONSCIOUSNESS_THEORIES.md](docs/CONSCIOUSNESS_THEORIES.md). An indicator whose module has not produced evidence yet reports `insufficient_data` and is left out of the score; [docs/FIRST_RUN.md](docs/FIRST_RUN.md) lists the warmup thresholds.

---

## Databases

Every store is optional and off by default; each has an in-memory implementation behind the same protocol.

| Store | Enable key | Holds |
|---|---|---|
| PostgreSQL | `MTC_POSTGRES_ENABLED` | episodic, semantic and core memory; conversations and messages; snapshots; the safety audit |
| MongoDB | `MTC_MONGODB_ENABLED` | conversation turns, thoughts and gists, neural telemetry, learning, speaker models, alerts |
| Qdrant | `MTC_QDRANT_ENABLED` | 384-dimension memory vectors |
| Neo4j | `MTC_NEO4J_ENABLED` | the concept graph |
| Redis | `MTC_REDIS_ENABLED` | working memory and short-lived state |

`mtc db init` prepares the enabled stores and `mtc db check` verifies them; see [docs/DATABASES.md](docs/DATABASES.md).

---

## What's New in v0.5.0

- **Honest measurement.** Seven v0.4.0 measurement defects are fixed behind switches; every indicator reports `measured`, `absent` or `insufficient_data`, and the `legacy_v04` preset reproduces the v0.4.0 numbers ([docs/MIGRATION_FROM_V04.md](docs/MIGRATION_FROM_V04.md)).
- **The runtime.** A conversation pipeline with one consciousness cycle per turn, four modulation layers, a provenance snapshot per reply and honest failures (no reply is stored when the model gives none); speaker trust by token, never by name.
- **The brain.** A local chat-completions provider with three template families, an optional Anthropic provider and a labelled mock ([docs/LLM_SETUP.md](docs/LLM_SETUP.md)).
- **Continuous thought, memory and knowledge.** A novelty-gated thought machine, memory consolidation and pruning, a knowledge graph ([docs/KNOWLEDGE_GRAPH.md](docs/KNOWLEDGE_GRAPH.md)).
- **Five optional stores** with migrations rendered from one schema registry, an idempotent bootstrap and a verifier.
- **The API** with fail-closed key authentication on loopback ([docs/SECURITY.md](docs/SECURITY.md)).
- **The `mtc` command** with a setup wizard, a doctor, operations scripts for backups, restores and health checks ([docs/OPERATIONS.md](docs/OPERATIONS.md)).

See [CHANGELOG.md](CHANGELOG.md) and [VERSION_HISTORY.md](VERSION_HISTORY.md).

---

## Documentation

| Document | What it covers |
|---|---|
| [Honest Limitations](docs/HONEST_LIMITATIONS.md) | What this framework cannot do (read this first) |
| [Getting Started](docs/GETTING_STARTED.md) | Install, the wizard and the three paths |
| [First Run](docs/FIRST_RUN.md) | What a fresh install measures and why |
| [Onboarding](docs/ONBOARDING.md) | The wizard, the companion types and their bounds |
| [Configuration](docs/CONFIGURATION.md) | Every setting, generated from the code |
| [Architecture](docs/ARCHITECTURE.md) | Layers, packages, stores and where state lives |
| [Runtime Pipeline](docs/RUNTIME_PIPELINE.md) | One turn and one cycle, step by step |
| [Consciousness Theories](docs/CONSCIOUSNESS_THEORIES.md) | The eight theories and the assessment method |
| [Butlin 2025 Mapping](docs/BUTLIN_2025_MAPPING.md) | Every indicator with its published id |
| [Continuous Thought](docs/CONTINUOUS_THOUGHT.md) | The background thought machine |
| [Knowledge Graph](docs/KNOWLEDGE_GRAPH.md) | Labels, relationships, constraints and seeds |
| [Databases](docs/DATABASES.md) | The five stores, their artifacts and statuses |
| [LLM Setup](docs/LLM_SETUP.md) | Model servers, providers and changing the model |
| [API](docs/API.md) | Routes, headers, the websocket protocol |
| [Security](docs/SECURITY.md) | Authentication, trust, secrets and data flows |
| [Setup with Docker](docs/SETUP_DOCKER.md) | The Compose path (reviewed, not run) |
| [Setup with host services](docs/SETUP_MACOS.md) | Homebrew and apt |
| [Operations](docs/OPERATIONS.md) | Start, stop, backups, restores, hardening |
| [Testing](docs/TESTING.md) | Markers, hermetic runs and the drift tests |
| [Developer Bring-up](docs/DEVELOPER_BRINGUP.md) | A development checkout |
| [Privacy Audit](docs/PRIVACY_AUDIT.md) | What the audit checks and how to fix a hit |
| [Migrating from v0.4.0](docs/MIGRATION_FROM_V04.md) | Switches, measured differences, other changes |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{devereaux2026mtc,
  title     = {Multi-Theory Consciousness Framework},
  author    = {Devereaux, Calvin},
  year      = {2026},
  url       = {https://github.com/WhiteLotusLA/multi-theory-consciousness},
  version   = {0.5.0},
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

Contributions from consciousness researchers, computational neuroscientists and software engineers are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## License

MTC is licensed under the Apache License 2.0; see [LICENSE](LICENSE). It vendors no third-party code. The optional `iit` extra installs PyPhi, which is GPLv3 or later: it is never part of `all`, it is imported only when installed, and a user who installs it runs a GPLv3 combination. Every direct dependency's licence is listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

---

## Acknowledgments

This framework would not exist without the theoretical foundations laid by the following researchers and their collaborators:

- **Bernard Baars**: Global Workspace Theory
- **Giulio Tononi**: Integrated Information Theory
- **Michael Graziano**: Attention Schema Theory
- **David Rosenthal**: Higher-Order Thought Theory
- **Karl Friston**: Free Energy Principle
- **Victor Lamme**: Recurrent Processing Theory
- **Patrick Butlin, Robert Long, Eric Elmoznino, David Chalmers, et al.**: Agency and Embodiment and the indicator-based assessment methodology
- **Ruben Laukkonen, Karl Friston and Shamil Chandaria**: Beautiful Loop Theory
- **Antonio Damasio**: the somatic marker hypothesis and the three-layer model

We are also grateful to the developers of [PyPhi](https://github.com/wmayner/pyphi), [snntorch](https://github.com/jeshraghian/snntorch), [reservoirpy](https://github.com/reservoirpy/reservoirpy) and [pymdp](https://github.com/infer-actively/pymdp) for the open-source tools that underpin parts of this implementation.
