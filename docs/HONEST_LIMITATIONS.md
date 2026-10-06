# Honest Limitations

**Read this document first.** Before using this framework, understand what it cannot do.

---

## 1. This Framework Cannot Detect Consciousness

The assessment defines 30 positive indicators and 3 negative indicators. It measures whether architectural functions associated with consciousness theories are operating as designed. A system that passes every indicator has all the structural features that certain theories associate with consciousness. Whether those features are *sufficient* for phenomenal consciousness is an open question that this framework does not and cannot answer.

When the assessment reports "architecture functional," it means: at least half of the indicators pass and the overall score is at least 0.5, so the modules are working as the theories describe. It does not mean the system has subjective experience.

## 2. The Scale Gap

Biological consciousness involves roughly 86 billion neurons with trillions of synaptic connections, complex neurochemistry (dozens of neurotransmitters, neuromodulators, hormones), and embodied interaction with the physical world through sensory organs and motor systems.

The default substrate sizes in the code are:
- 5,196 spiking neurons in the SNN (`ProductionSNN`: layers of 1000, 2048, 2048 and 100)
- 5,000 reservoir neurons in the LSM (`ProductionLSM`)
- 131,072 HTM cells (`ProductionHTM`: 4,096 columns of 32 cells)

This is a difference of roughly six orders of magnitude in neural count alone, and entirely omits the chemical, embodied, and developmental dimensions. If consciousness requires a certain scale of neural complexity, this implementation is far below it.

## 3. The LLM Confound

When this framework is connected to a language model for response generation, the LLM does the heavy lifting on language comprehension, reasoning, and response quality. The consciousness modules provide context enrichment (attention guidance, metacognitive reflections, prediction signals), but the observable "intelligence" of responses comes primarily from the LLM's pre-trained capabilities.

This means: apparent sophistication in conversation is not evidence of consciousness. It is evidence of a large language model doing what large language models do. Nothing a companion built on MTC says about its inner life is evidence of an inner life.

## 4. The Labeling Problem

The indicators have names like "attention_schema," "metacognition," and "integrated_information." These labels reference concepts from consciousness science, but the computational processes they measure may bear only a structural resemblance to the biological phenomena they are named after.

When we measure "metacognition," we are measuring: does the system generate higher-order representations of its own processing states? We are not measuring: does the system *experience* thinking about its own thoughts?

## 5. Theory Selection Bias

We implement eight theories of consciousness. There are more. Our selection reflects availability of computational specifications, not a claim that these eight are correct or complete. Notable omissions include:

- Orchestrated Objective Reduction (Penrose-Hameroff)
- Predictive Processing / Active Inference in its full biological scope
- Enactivist approaches (Thompson, Varela)
- Phenomenological approaches that may resist computational implementation

## 6. Assessment Methodology Limitations

The assessment framework is derived from Butlin et al. (2023) and its 2025 update, but our indicator implementations are our own interpretations of their criteria. Different researchers might operationalize the same indicators differently.

The thresholds (for example 0.4 for `ignition_dynamics` and 0.5 for `global_broadcast`) are calibrated through testing, not derived from neuroscience data. They represent reasonable operating points, not ground truth. The specificity and sensitivity values attached to each indicator are expert estimates, not measurements.

## 7. Consumer Hardware Constraints

The framework runs on consumer hardware by design (Apple Silicon, consumer GPUs, or the CPU). This is a feature for accessibility but a limitation for fidelity. Research-grade neural simulations typically require HPC clusters. Our simulations are simplified to run on a single machine.

## 8. No Longitudinal Validation

Consciousness in biological systems develops over months and years through sustained interaction with the environment. Our assessment runs in seconds to minutes. We have not validated whether long-term operation produces qualitatively different results from fresh initialization.

## 9. Reproducibility Caveats

Neural substrate initialization involves randomness (random synaptic weights, reservoir topology). Assessment results vary between runs unless a seed is set (`mtc assess --seed`). Body metrics read the machine's own load and timing, so the runners keep them off unless they are set explicitly.

---

## 10. Known measurement defects in v0.4.0

Seven defects affect v0.4.0 measurements. All are fixed in v0.5.0, published v0.4.0 numbers are reproduced by the `legacy_v04` preset, and [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md) records the before and after numbers for each.

1. **Exact Phi never ran.** The exact IIT path in the assessment calls PyPhi with a keyword argument the library does not accept, so every exact Phi attempt in v0.4.0 failed and fell back to the approximation. Reported v0.4.0 Phi values therefore came from the approximation.
2. **Permuted network.** Both the PhiTracker and the exact path index states big-endian and pass a target-by-source connectivity matrix, where PyPhi expects little-endian rows and a source-by-target matrix. Any exact Phi computed this way was computed for a permuted network, not the network being measured.
3. **Dead statistics keys and a label-typed trend.** Several indicators read keys their producers never emit (`competition`, `max_depth_reached`, `total_generated`, `active_policies`, `decay_rate`), so they scored a constant, and the prediction-error trend compared a text label to a number, so its numeric branch never ran. v0.5.0 reads what the producers emit, pinned by a statistics contract test.
4. **Affect coherence rewarded variance.** The PAD coherence formula read 0.97 to 1.0 on pure noise. v0.5.0 scores temporal persistence (lag-1 autocorrelation) and reports nothing below ten states.
5. **Presence floors and constant stand-ins.** A present module with no data scored 0.1, 0.3 or 0.5, and several indicators granted credit for a module merely existing. v0.5.0 reports `insufficient_data` by name, leaves it out of the averages and starts every floor at zero.
6. **The legacy 14-indicator integration never measured.** It never awaited its indicator coroutines, so no value was ever produced, and its workspace-integration metric read a NaN correlation on constant vectors as 1.0. v0.5.0 awaits every indicator and reports coverage.
7. **A silent SNN and a self-referential HTM anomaly.** A units error kept the spiking network silent at default settings, and the HTM scored each step's anomaly against the predictions it had just made (about 1.0 every step). No published v0.4.0 number used either substrate.

---

## 11. Limits of the generic cognition modules

- **Speaker models are heuristics.** Speaker models are keyword and exponential-moving-average heuristics, and the trust attached to a speaker is configured, not measured.
- **Emotion extraction is rule-based.** It compares text against running baselines and starts from a declared neutral prior until those baselines exist.
- **Self-assessment is model text.** Self-assessment answers are text the model produced, so divergence from the indicators is information about the report, not about the system.
- **The behavioral battery measures behavior.** It measures response patterns, not experience.
- **Developmental stages are unvalidated.** They have no external validation and stay off unless enabled (the `child` companion type or `DEVELOPMENTAL_STAGE_ENABLED`).

---

## 12. Limits of the infrastructure layer

- **The emotion bridge reports no valence, arousal or coherence until its LSM readout is trained.** Before training, those fields are None, and the orchestrator does not treat the liquid-state block as real.
- **Safety scoring, the trust compass's simple mode and data classification are keyword heuristics, and classification errs protective.** They miss paraphrase and flag innocent text; the compass's deep mode adds a model's judgment when a brain answers.
- **Wellbeing indicators are computed from PAD readings and contact recency, are not reports of felt states, and agency never raises an alarm by design.** A low sense of control is recorded and never read as distress.
- **Memory gists and knowledge synthesis exist only when the brain answers.** With no brain the memory layer stores and retrieves text and writes no gist.
- **The embedding model is the memory layer's one network dependency.** Without it nothing is embedded, and recall falls back to labelled full-text search. `mtc doctor --fetch-models` downloads it once; offline mode works after that.

---

## 13. Limits of the conversation runtime

- **Replies from the mock provider are labelled test text, not a model's answer.** They start with `[mock]`, exist so the pipeline, the demo and the tests run with no model server, and say nothing about what a model would have said.
- **Narration is off by default, and the inner-state text a researcher enables is a rendering of computed values, not a report of experience.** Every `NARRATE_*` key is false; turning one on adds lines that describe module outputs.
- **The PhiTracker value in status and state is an integration proxy over five module signals.** It is labelled `integration proxy` wherever it is reported, it is not IIT Phi, and it appears in no prompt.
- **PAD inputs absent in a cycle use declared neutral defaults, and the turn record names them.** `pad_sources_measured` lists the inputs that were measured; an input missing from that list contributed its default, not a reading.
- **Continuous thoughts stay quiet without a semantic embedder, and their templates are authored seed text.** The novelty gate needs a trained embedder to let a thought surface; the hash embedder is accepted only with an explicit opt-in and measures word overlap; the thought texts come from the seed packs under `config/ctm` and from whatever model answers.
- **Four of the six theory modulations are computed and not applied.** Only the somatic and Phi salience biases reach the next cycle; the higher-order, free energy, recurrence and depth modulations are reported with `consumed: false`.

---

## 14. Research-validity caveats

- **Epistemic depth is keyword-circular.** The beautiful loop's depth tracker counts self-referential phrasing in generated higher-order thoughts. The metacognition module phrases its own awareness thoughts that way, so `genuine_implementation` and `epistemic_depth` can be satisfied by the module's wording on any input. In the experiment runner `genuine_implementation` saturates its noise baseline for exactly this reason (see [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md)).
- **`affect_coherence` measures temporal persistence of the PAD coordinate, nothing more.** It is the mean lag-1 autocorrelation of three numbers over 20 states; a slowly drifting input scores well whatever it means.
- **Only `ProductionHTM` ships.** An accelerated HTM variant was excluded because its temporal memory predicted itself, so its anomaly readings carried no information. `get_best_htm` returns the pure NumPy `ProductionHTM`; no compiled or native HTM ships.
- **Module-graph Phi is an integration proxy, never narrated.** Exact IIT 4.0 Phi through PyPhi is computed over a five-node abstraction of module activity, not over the substrates; it is reported as telemetry and never enters a prompt.
- **Several scores in the seeded runners are seed position, not signal.** The experiment runner's warmup is noise by construction, and normalizing against a noise baseline from the same feed zeroes what the feed produces. A passing indicator there can be one seed's position among others of the same noise.
- **Setup paths are not all tested.** The Docker Compose file was written and reviewed against official documentation and not run on the development machine (Docker was not installed there); the path run on the development machine is the in-memory install with the mock provider. [SETUP_DOCKER.md](SETUP_DOCKER.md) and [SETUP_MACOS.md](SETUP_MACOS.md) say which steps were run.
- **No performance figures are claimed.** Timings anyone observes come from one machine; this documentation gives none except the two labelled approximations of model memory footprints in [LLM_SETUP.md](LLM_SETUP.md).

---

## Why Share It Anyway

Given these limitations, why release this framework?

1. **Testable implementations advance theory.** Implementing a consciousness theory in code forces precision that prose descriptions do not require. Bugs in implementation reveal gaps in theory.

2. **Shared tools prevent duplicated effort.** Multiple labs implement the same theories from scratch. A shared framework lets researchers focus on their specific questions rather than re-implementing GWT for the hundredth time.

3. **Honest measurement is better than no measurement.** Even imperfect metrics, clearly documented, are more useful than unfalsifiable claims.

4. **Interaction effects matter.** Most implementations study one theory in isolation. Running eight theories simultaneously reveals interaction effects that single-theory implementations cannot observe.

5. **Accessibility matters.** Consciousness research should not require an HPC cluster and a neuroscience PhD to begin experimenting.

We share this framework not because it is complete, but because it is useful. We lead with limitations not because we are pessimistic, but because intellectual honesty is the foundation of good science.
