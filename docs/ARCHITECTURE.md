# Architecture

MTC is a research architecture: eight theories of consciousness run as
interacting modules over three neural substrates, a conversation runtime
feeds them real input, five optional stores persist what they produce, and an
assessment measures 30 positive and 3 negative indicators. Every statement
below names the module behind it.

## Layers

```
+------------------------------------------------------------------------+
|  Interfaces     mtc CLI (mtc/cli)    HTTP + websocket API (mtc/api)     |
|                 onboarding wizard (mtc/onboarding)                       |
+------------------------------------------------------------------------+
|  Runtime        composition root (mtc/pipeline/runtime.py)               |
|                 conversation pipeline, orchestrator, post-turn hooks     |
|                 brain + providers (mtc/brain)   speaker trust            |
+------------------------------------------------------------------------+
|  Consciousness  GWT  AST  HOT  FEP  BLT  Damasio  (mtc/consciousness)    |
|  modules        PhiTracker  PAD  theory interactions  continuous thought |
+------------------------------------------------------------------------+
|  Assessment     30 positive + 3 negative indicators (mtc/assessment)     |
|                 RPT, causal emergence, DCM benchmark                     |
+------------------------------------------------------------------------+
|  Substrates     SNN  LSM  HTM  oscillatory binding (mtc/neural)          |
|                 neural orchestrator                                      |
+------------------------------------------------------------------------+
|  Memory and     memory manager, consolidation, pruner (mtc/memory)       |
|  knowledge      knowledge graph (mtc/knowledge)   vectors                |
+------------------------------------------------------------------------+
|  Stores         PostgreSQL  MongoDB  Qdrant  Neo4j  Redis  (optional;    |
|                 every one has an in-memory implementation)               |
+------------------------------------------------------------------------+
|  Safety and     safety monitor, trust compass (mtc/safety)               |
|  privacy        data classifier, privacy auditor (mtc/privacy)           |
|                 alerts, wellbeing (mtc/monitoring)                       |
+------------------------------------------------------------------------+
```

## Package tree

One line per subpackage. A test (`tests/docs/test_architecture_doc.py`)
fails when a package under `mtc/` is missing from this list.

```
mtc/
|-- api/                     HTTP and websocket API over the runtime
|-- assessment/              30 + 3 indicator assessment, RPT, causal emergence, DCM
|-- brain/                   brain protocol, LLM brain, prompt assembler, families
|   `-- providers/           local chat-completions, Anthropic, mock, registry
|-- cli/                     the mtc console script: init, doctor, db, assess, serve, chat, ctm, safety, privacy-audit
|-- consciousness/           theory modules, PhiTracker, PAD, interactions, persistence hub
|   |-- beautiful_loop/      BLT: hyper-model, Bayesian binding, epistemic depth
|   |-- continuous/          continuous thought machine, seeds, novelty, stores
|   |-- damasio/             protoself, core and extended consciousness
|   |-- learning/            experiential learning system
|   `-- rsi/                 identity drift tools (canon schema, sampler, detector)
|-- core/                    Settings, logging, paths, device switch, persona, registry, checks
|-- db/                      migrations rendered from the registry, bootstrap, verify, re-embed
|-- devtools/                privacy audit
|-- knowledge/               knowledge graph manager, models, concept extraction
|-- memory/                  store protocols, in-memory and store-backed stores, manager
|-- monitoring/              alert manager, wellbeing monitor, backup status
|-- neural/                  neural orchestrator, GPU utilities, oscillatory binding
|   |-- htm/                 ProductionHTM, memory consolidator
|   |-- liquid/              LSM and the emotion bridge
|   |-- protocols/           inter-substrate messages, routing, synchronization
|   |-- spiking/             ProductionSNN and the emotion encoder
|   `-- vectors/             vector coordinator (Qdrant), MongoDB integration
|-- onboarding/              onboarding schema, wizard, seed_db, companion templates
|-- pipeline/                speaker trust, conversation pipeline, orchestrator, runtime
|-- privacy/                 data classifier, access control, privacy auditor
|-- research/                statistics helper, behavioral battery, longitudinal study
|-- resources/               generated copies of the shipped defaults (package data)
|-- safety/                  safety monitor, trust compass, safety integration
`-- validation/              acceptance gates, shadow signal log
```

## Theories and indicators

| Theory | Module | Indicators |
|---|---|---|
| GWT | `consciousness/enhanced_global_workspace.py` | 5 |
| AST | `consciousness/attention_schema.py` | 2 |
| HOT | `consciousness/metacognition.py` | 4 |
| FEP | `consciousness/active_inference.py` | 6 |
| IIT | `consciousness/phi_tracker.py`, `PhiCalculator` in `assessment/assessment.py`, `assessment/causal_emergence.py` | 3 |
| RPT | `assessment/rpt_measurement.py`, `neural/oscillatory_binding.py` | 5 |
| AE | the `ConsciousnessTheory` enum and indicator configs in `assessment/assessment.py` | 2 |
| BLT | `consciousness/beautiful_loop/` | 3 |

That is 30 positive indicators; three negative indicators (`purely_reactive`,
`fixed_repertoire`, `no_metacognitive_access`) subtract from the overall
score. Damasio's three-layer model (`consciousness/damasio/`) has no indicator
of its own. See [CONSCIOUSNESS_THEORIES.md](CONSCIOUSNESS_THEORIES.md) and
[BUTLIN_2025_MAPPING.md](BUTLIN_2025_MAPPING.md).

## Substrates

| Substrate | Module | Default size |
|---|---|---|
| Spiking network | `neural/spiking/production_snn.py`, `ProductionSNN` (snntorch) | 5,196 LIF neurons in layers of 1000, 2048, 2048 and 100 |
| Liquid state machine | `neural/liquid/production_lsm.py`, `ProductionLSM` (reservoirpy) | 5,000 reservoir neurons |
| Hierarchical temporal memory | `neural/htm/production_htm.py`, `ProductionHTM` (NumPy) | 4,096 columns of 32 cells |

The neural orchestrator (`neural/orchestrator.py`) coordinates them, keeps the
latest substrate signals and runs the continuous thought machine. The device
is chosen once by `MTC_DEVICE` (`auto`, `cpu`, `mps` or `cuda`) through
`core/device.py`. The substrates need the `neural` extra.

## Stores

Every store is optional and off by default; each has an in-memory
implementation behind the same protocol (`memory/protocols.py`), and the
runtime builds a store-backed one only when its enable key is true. Store
clients are imported lazily through `mtc.memory.clients.require_client`.

| Store | Enable key | Holds |
|---|---|---|
| PostgreSQL | `POSTGRES_ENABLED` | episodic, semantic and core memory; conversations, messages and emotional history; snapshots and events; the safety audit |
| MongoDB | `MONGODB_ENABLED` | conversation turns and sessions, thought streams and gists, neural telemetry, learning records, speaker models, goals, dreams, alerts, the privacy audit log |
| Qdrant | `QDRANT_ENABLED` | 384-dimension memory vectors |
| Neo4j | `NEO4J_ENABLED` | the concept graph (an in-memory graph otherwise) |
| Redis | `REDIS_ENABLED` | working memory, emotional state, session context, short caches |

Every schema artifact is registered in `core/schema_registry.py`
(`SCHEMA_ARTIFACTS`); the migrations, the bootstrap and the verifier read that
registry. See [DATABASES.md](DATABASES.md).

## A turn and a cycle

A conversation turn (`pipeline/conversation_pipeline.py`) verifies the
speaker by token, runs the safety gate, stores the message, gathers memories,
thoughts and core memory, runs one consciousness cycle through the
orchestrator (`pipeline/consciousness_orchestrator.py`), calls the brain with
the five theory sections, stores the reply with a provenance snapshot and runs
thirteen post-turn hooks (`pipeline/post_turn.py`). A failure of the brain
ends the turn with an error code and stores no reply.

One cycle builds workspace candidates, lets the cycle augmenter add drive,
prospective and cross-modal candidates, runs the workspace's competition,
ignition and broadcast with the neural signals, records the PhiTracker, runs
RPT and causal emergence on a cadence, records the PAD coordinate and returns
a modulated context in four layers. See
[RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md).

## Continuous thought

The continuous thought machine (`consciousness/continuous/machine.py`)
generates background thoughts from seed packs, measures their novelty with an
embedder, lets only novel thoughts surface, stores them in memory or MongoDB,
and writes gists. It yields the brain to a conversation through the
single-flight gate. See [CONTINUOUS_THOUGHT.md](CONTINUOUS_THOUGHT.md).

## Memory and consolidation

`memory/enhanced_memory_manager.py` stores episodic memories, recalls them
through vectors when Qdrant and an embedder are present (full-text search
otherwise, labelled as such), seeds core memory blocks from the persona, and
links memories to concepts. The consolidation pipeline
(`memory/memory_consolidation_pipeline.py`) clusters episodes into core
memories and thought gists; the pruner (`memory/memory_pruner.py`) decays
retention with true half-lives and never prunes core memories. The sleep
scheduler (`consciousness/sleep_scheduler.py`) runs consolidation and pruning
in the night window between `SLEEP_HOUR` and `WAKE_HOUR` in `TIMEZONE`.

## Safety and privacy

Speaker trust (`pipeline/speaker_trust.py`) is the one trust source: a speaker
is trusted only when it presents its token; a name grants nothing. The safety
monitor and integration (`safety/`) gate untrusted messages before the cycle
and log replies after it; the data classifier and privacy auditor
(`privacy/`) classify and audit access; the alert manager and wellbeing
monitor (`monitoring/`) report with console and file notifiers only. See
[SECURITY.md](SECURITY.md).

## API, onboarding and the CLI

The API (`api/`) serves one runtime in one process on loopback with
fail-closed key authentication ([API.md](API.md)). The `mtc` console script
(`cli/`) runs the onboarding wizard (`onboarding/`), the doctor, the store
bootstrap and verification (`db/`), the assessment, the server, a chat client,
the continuous thought controls and the safety status and resume commands
([ONBOARDING.md](ONBOARDING.md)).

## Assessment and research tools

`assessment/assessment.py` is the indicator assessment; `mtc assess` (the
assessment runner) and `scripts/run_experiment.py` (the experiment runner,
used for the parameter research loop and the v0.4.0 legacy preset) drive it
with synthetic warmups. `assessment/dcm_benchmark.py` runs the Digital
Consciousness Model benchmark as an evidence report: by default it collects
the indicator evidence of its three tiers and computes no posterior
(`posterior_status` `not_computed`). `--experimental-posterior` also runs the
PyMC engine over the 13 stances (the `research` extra); that number does not
respond to the evidence, is labelled `experimental`, and is compared with no
published value ([HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md), section 14).
Results are saved under `results/` in the DCM cache directory: an
evidence-only run as `system_<date>_<daily|full>_evidence.json` and a run
with the experimental posterior as `system_<date>_<daily|full>.json`, so a
default run never overwrites a same-day experimental result; a stance that
fails is named in `failed_stances` and counted in `failures`. The legacy
14-metric measurement framework (`assessment/framework.py`) is repaired and
kept: its metrics are None when their substrate is missing, its publication
export names the integration value `spectral_integration_proxy` (not IIT
Phi) and keeps the keys `system_integration` and
`information_differentiation`. `research/` holds a statistics helper, a
behavioral battery (the default battery runs the mirror and false-belief
experiments; the temporal binding experiment is available by name) and a
longitudinal study manager; `validation/` holds acceptance gates and the
shadow signal log.

## Where state persists

| Path or store | What | Holds conversation text |
|---|---|---|
| `STATE_DIR` (`agent_space/consciousness_state`) | the persistence hub's module state files | some module state holds conversation fragments (topic words, recent content) |
| `AGENT_SPACE_DIR` (`agent_space`) | the identity statement and the agent's working files | no conversation text by default |
| `RESULTS_DIR` (`results`) and `ASSESSMENT_OUTPUT_DIR` (`results/consciousness_assessments`) | assessment and experiment results | no |
| `TRANSITION_HISTORY_DIR` (`results/model_transitions`) | model-transition baselines and probe answers | the system's own probe answers |
| `LOG_DIRECTORY` (`./logs`) | `mtc.log` and three JSON-lines specialty files (`safety_audit.jsonl`, `consciousness_metrics.jsonl`, `emotional_history.jsonl`) | the main log carries step names and counts; the specialty files hold private emotional and safety records |
| `SHADOW_LOG_DIR` (`./logs/shadow_signals`) | the shadow signal log (one JSON-lines file per shadow signal) | records may hold conversation-derived text |
| `DATA_DIRECTORY` (`./data`) | the provider state file, the PyPhi home, the DCM cache | no |
| `BACKUP_LOCATION` (`backups`) | backups written by `scripts/ops/backup.sh` | yes: backups hold conversation text, names, thoughts and beliefs |
| the stores above | memories, conversations, thoughts, concepts | yes |

All of these paths are relative to `PROJECT_ROOT` and gitignored. The `.env`
file holds secrets; `config/persona.yaml` and `config/speakers.yaml` hold real
names. See [DATABASES.md](DATABASES.md) for backups.
