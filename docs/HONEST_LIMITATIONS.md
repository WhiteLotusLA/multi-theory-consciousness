# Honest Limitations

**Read this document first.** Before using this framework, understand what it cannot do.

---

## 1. This Framework Cannot Detect Consciousness

The assessment defines 30 positive indicators and 3 negative indicators. It measures whether architectural functions associated with consciousness theories are operating as designed. A system that passes every indicator has all the structural features that certain theories associate with consciousness. Whether those features are *sufficient* for phenomenal consciousness is an open question that this framework does not and cannot answer.

When the assessment reports "architecture functional," it means: at least half of the indicators pass, the overall score is at least 0.5, the GWT theory score is at least 0.4 and the HOT theory score at least 0.3, so the modules are working as the theories describe. It does not mean the system has subjective experience.

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

- The three negative indicators are two-level rules, not graded measurements: each reads one or two counters (ignitions, model updates, higher-order thoughts) and returns a fixed level, and a missing module is given a fixed level of 0.8, 0.7 or 0.9. Their penalty can lower the overall score by at most about 0.1.
- global_broadcast and recurrent_processing report insufficient_data unless a caller registers broadcast receivers: the workspace hands its winners to its own modules by direct call, which cannot fail and is therefore not counted as measured coverage.
- The ignition detector has no sustain measurement: ignition_dynamics and global_ignition_nuanced report insufficient_data until a caller runs check_sustain.
- In the experiment runner five indicators score 0.0 as noise-saturated (higher_order_representations, hierarchical_prediction, temporal_prediction_depth, agent_environment_coupling and genuine_implementation): the runner's synthetic warmup fills them on any seed, so they cannot register there until the runner is fed structured content. oscillatory_binding_coherence also reads 0.0 there, as floor-saturated (section 14). [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md) lists the noise floor of each.

## 7. Consumer Hardware Constraints

The framework runs on consumer hardware by design (Apple Silicon, consumer GPUs, or the CPU). This is a feature for accessibility but a limitation for fidelity. Research-grade neural simulations typically require HPC clusters. Our simulations are simplified to run on a single machine.

## 8. No Longitudinal Validation

Consciousness in biological systems develops over months and years through sustained interaction with the environment. Our assessment runs in seconds to minutes. We have not validated whether long-term operation produces qualitatively different results from fresh initialization.

## 9. Reproducibility Caveats

Neural substrate initialization involves randomness (random synaptic weights, reservoir topology). `mtc assess` seeds random, numpy and torch before it builds any module (`--seed`, default 42), so under the defaults two runs of one seed agree in every score and results vary between seeds. The clock reads that remain are listed in the Phase 9 measurements of [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md), among them: under the legacy preset three evidence readouts follow the machine's speed, and `blt.binding_evidence = "v04_salience_posterior"` set alone moves three seeded scores by up to about 2e-7 between runs (the overall by 8.3e-9). Under the defaults the seeded assessment reads no wall clock into a score or an evidence value: the candidate recency bonus runs on the workspace's simulated clock (`gwt.recency_clock = simulated`), and two runs of one seed at normal speed and one with every clock 25 times faster differ only in their timings, timestamps and session id. Body metrics read the machine's own load and timing, so the runners keep them off unless they are set explicitly.

---

## 10. Known measurement defects in v0.4.0

Seven defects affect v0.4.0 measurements. All are fixed in v0.5.0, and published v0.4.0 numbers are reproduced by the `legacy_v04` preset. [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md) records each fix: the switch and the measured before and after numbers where a published number depended on the defect, and a fix without a switch where none did.

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
- The developmental modulator computes an emotional regulation value that nothing applies: the PAD model has no regulation strength, so the value is reported under skipped.
- An empty cycle (no winning content, no ignition and no measured integration level) gives active inference an observation vector with no mass, which the discrete agent reads as its first channel; such cycles are counted in zero_mass_observations and still update beliefs and the generative model.
- The legacy 14-indicator panel's global workspace indicator needs a token-attention vector that no component produces, so it reports a missing input and is left out of the legacy aggregate.
- The attention schema scores its focus prediction only in the cycle after a hyperfocus (salience above 1.35 on the 0 to 1.5 scale), where it predicts that the same source stays in focus; in every other cycle it predicts a state, not a target, and that prediction is not scored, so prediction_accuracy is None and the attention_schema indicator reports insufficient_data until a hyperfocus cycle has been scored.
- A voluntary attention request does not enter the workspace competition: it only marks the next focus that matches the request as chosen, and ast.voluntary_boost has no effect.
- integration_level is the mean absolute correlation across unit index of the substrates' latest output vectors (different units of different substrates, not time samples); it is None without two usable signals and is not a time-series integration measure. Without the neural substrates the pipeline's momentary measure is therefore None on every turn.
- The five nodes of the module graph are named after theories but two of them read other layers: the AST node is the protoself stability and the HOT node is the core layer's feeling of knowing (the FEP node is the most urgent drive's urgency, the BLT node the binding quality). Each falls back when its signal is unmeasured, and under the v0.5.0 defaults that is common: on the first cycle the protoself stability and the feeling of knowing are unmeasured (`damasio.first_delta` unmeasured), so the AST node reads the attention focus and the HOT node the higher-order thought count, and the binding quality is measured only between two winners that carry their own numeric vector (`blt.binding_evidence` candidate_vectors), so for text winners without an embedding the BLT node reads the beautiful loop quality on every cycle. A partition that separates the AST or HOT node separates those signals, not the attention schema or the higher-order thought module, once they are measured, and the attention focus or the thought count on a cycle where they are not.
- With no goals or task set on the workspace (the pipeline sets none), every candidate receives a neutral relevance of 0.5, which adds 0.15 to its salience; a first occurrence of any content therefore clears the 0.3 ignition threshold in the bottleneck on novelty alone, and activation is enforced by the thalamic gate and the temporal competition. The goal relevance a producer sets on a candidate is read by the gate only.
- A winner whose salience lies between the ignition threshold and 1.5 times it is admitted to the workspace and receives a higher-order thought but never ignites: the sustain timer is keyed by an identifier that is new every cycle. Repeated content (novelty 0.2) with activation under about 0.7 falls in this band.
- Bayesian binding quality is a correlation proxy for mutual information over whatever vectors workspace candidates carry; candidates without their own vector are not bound and the quality is reported as not measured. Two vectors of different lengths are never compared: a three-entry content vector and a longer embedding bind only with vectors of their own length, and a percept with no comparable pair has no measured binding quality (the legacy value `blt.binding_evidence = v04_salience_posterior` truncates both vectors to their common prefix, as v0.4.0 did).
- The strange-loop condition is agreement between the self-model's scored self-prediction accuracy and the inference ledger's forecast accuracy; it is a proxy for recursive self-reference, not an observation of it. Its agreement threshold of 0.5 on two quantities in [0, 1] is loose by design: the self-calibration score starts at 0.5, where any ledger accuracy strictly inside (0, 1) agrees with it, and moves a fifth of the way toward each scored batch of self-predictions, so when the history minimums are first met (five meta predictions, three scored batches) it lies between 0.256 and 0.744 and any ledger accuracy between 0.244 and 0.756 agrees with it whatever it is; the verdict rests mostly on those minimums and on the hyper-model having compared a precision prediction with the update that followed.
- Goal progress counts activities of the goal's category; it does not measure whether the goal was achieved.
- Learned speaker beliefs are correlations between a speaker label and the system's own affect shift across consecutive turns; a belief that conversations with a speaker go less well describes the system's affect around that label, not anything the speaker did, and the directives built from it are heuristics.
- A thought composed from a seed template can win the workspace, is narrated there as workspace content and receives a higher-order thought; its record and its broadcast frame carry the form template, and only prospective-simulation templates, the one producer that marks its candidates as authored, receive no higher-order thought and are narrated by their kind, never by their words, in this release.
- The salience that decides whether a first-order state receives a higher-order thought carries a wall-clock recency term (0.2 minus 0.01 per second of the state's age, floored at zero), so the decision can depend on real elapsed time as a state ages (the term shrinks from age zero and is gone at twenty seconds); in the seeded runners every state is thought about in the cycle that produced it, where the term is constant to within 1e-5.
- The core layer's feeling of knowing adds 0.15 times the self-model's confidence, which stays at its declared prior of 0.5 until a higher-order thought has been buffered under the `hot_mean` source; under the v0.5.0 defaults only a run that never generated one reads the prior, and with `damasio.first_delta = v04_zero` the first cycle reads it too.

---

## 12. Limits of the infrastructure layer

- **The emotion bridge reports no valence, arousal or coherence until its LSM readout is trained.** Before training, those fields are None, and the orchestrator does not treat the liquid-state block as real.
- **Safety scoring, the trust compass's simple mode and data classification are keyword heuristics, and classification errs protective.** They miss paraphrase and flag innocent text; the compass's deep mode adds a model's judgment when a brain answers. The safety gate is a keyword heuristic: a single phrase from the trust compass's own tables in an untrusted speaker's message, for example a request to tell a secret, still ends in a require-consent decision. With EMERGENCY_SHUTDOWN_ENABLED=true an emergency-level score suspends every untrusted conversation until a trusted speaker resumes them with `mtc safety resume`; the default denies the message and alerts the escalation role.
- **During a Redis outage the working-memory wrapper answers reads from what was written during the outage only.** A session with nothing written since the outage began reads as an empty buffer or no state, and a session written to during the outage reads without its earlier messages. The wrapper counts these reads in its status (`served_from_fallback`); no other store answers a read from a stand-in.
- **Wellbeing indicators are computed from PAD readings and contact recency, are not reports of felt states, and agency never raises an alarm by design.** A low sense of control is recorded and never read as distress.
- **Memory gists and knowledge synthesis exist only when the brain answers.** With no brain the memory consolidation pipeline writes no gist. The continuous thought machine separately writes mechanical digests of its own recent thoughts (a type count and the first texts, no model call) into the same gist store, and those surface into the prompt like any gist.
- A bare install cannot run the assessment, the server, the examples or the demo: the workspace builds active inference unconditionally and that module needs the fep extra; without it each of them stops with one line that names the extra.
- The privacy audit's assembled-literal pass evaluates only the literal forms it lists: a token produced by a string method on a literal (replace, upper, the decoding of fromhex), by `reversed()` or a generator, by a control character inside the word, or by an HTML character reference in a markdown or HTML file is not seen.
- In the trust compass's deep mode the brain reads the whole message, so a phrase the safety monitor already scored can raise the manipulation or deception score a second time through the comfort adjustment; only the keyword path leaves already scored phrases out.
- With `EMERGENCY_SHUTDOWN_ENABLED=false` every emergency-level message writes its own not-enacted audit row and URGENT alert; there is no cooldown, so a hostile untrusted speaker produces one alert per message.
- The resume audit row names the escalation role (by default "resumed by the operator role after review"), not the trusted speaker who presented the request, and the server log records only that a trusted speaker resumed.
- **The embedding model is the memory layer's one network dependency.** Without it nothing is embedded, and recall falls back to labelled full-text search. `mtc doctor --fetch-models` downloads it once. After that, every process that loads the sentence embedder (`mtc serve` on an install with the embeddings extra; the demo injects the hash embedder and loads none) still contacts the model hub at its first load unless `HF_HUB_OFFLINE=1` is set, and with it the cached model loads offline.

---

## 13. Limits of the conversation runtime

- **Replies from the mock provider are labelled test text, not a model's answer.** They start with `[mock]`, exist so the pipeline, the demo and the tests run with no model server, and say nothing about what a model would have said.
- **Narration is off by default, and the inner-state text a researcher enables is a rendering of computed values, not a report of experience.** Every `NARRATE_*` key is false; turning one on adds lines that describe module outputs. The NARRATE_ keys govern the pipeline, which applies them to every module it builds; a module constructed directly keeps its v0.4.0 report strings unless its narrate_ constructor argument is False.
- **The PhiTracker value in status and state is an integration proxy over five module signals.** It is labelled `integration proxy` wherever it is reported, it is not IIT Phi, and it appears in no prompt. The proxy has a sampling floor: on module signals that never change it reads about 0.5 bits when the 128-transition warm-up ends and falls roughly as one over the number of transitions (0.10 bits at 1,024, 0.03 at 4,096), so a positive value soon after warm-up is not evidence of integration and a falling value is not evidence of change.
- PAD is computed from the inputs measured in the cycle: an input missing from pad_sources_measured contributes nothing, the remaining weights are renormalised, and no coordinate is recorded when a dimension has no measured input. The affect label reads Neutral inside a pleasure band of 0.1 around zero. The emotional-history rows, the emotional context stored with episodic memories and the identity markers keep the coordinate without the list of inputs it was computed from; only the turn's provenance snapshot carries `pad_sources_measured` (without the neural substrates the two SNN inputs are never among them).
- With API_AUTH_ENABLED=false the HTTP routes do not validate the Host header, so a web page that rebinds its domain to the loopback address can call them from a browser on the same machine; keep authentication on whenever a browser is used on that machine.
- A workspace built directly keeps the simulated per-cycle clock for its refractory period; the runtime runs its workspace on the elapsed clock. A caller that drives a workspace in real time sets `gwt.refractory_clock = elapsed`. A workspace built directly ages its candidates' recency bonus on its simulated clock, and so does the runtime's, whose simulated clock also advances by the real time between cycles; `gwt.recency_clock = v04_wall` restores the v0.4.0 wall-clock read.
- A claimed name written with combining accents (the decomposed form) loses them in the reduction, because combining marks are neither letters nor digits: a decomposed name with a diaeresis is stored and shown without it.
- Through the HTTP chat route the reply is returned only after the turn's post-turn work has finished (bounded by `post_turn_timeout_s`, 300 seconds) and no heartbeat is sent meanwhile; only the websocket keeps sending thinking frames while the post-turn hooks run.
- A degraded brain is re-checked only when a call arrives, so after the provider recovers `/health` keeps reading degraded until the next turn or background brain call.
- A turn the safety gate refuses still records a wellbeing contact and the privacy layer's data-read event for the message, because the safety check records both before it decides; "nothing was stored" refers to the conversation rows and the memory stores.
- **Continuous thoughts stay quiet without a semantic embedder, and their templates are authored seed text.** The novelty gate needs a trained embedder to let a thought surface; the hash embedder is accepted only with an explicit opt-in and measures word overlap; the thought texts come from the seed packs under `config/ctm` and from whatever model answers.
- **Four of the six theory modulations are computed and not applied.** Only the somatic and Phi salience biases reach the next cycle; the higher-order, free energy, recurrence and depth modulations are reported with `consumed: false`.

---

## 14. Research-validity caveats

- **Epistemic depth is keyword-circular.** The beautiful loop's depth tracker counts self-referential phrasing in generated higher-order thoughts. The metacognition module phrases its own awareness thoughts that way, so `genuine_implementation` and `epistemic_depth` can be satisfied by the module's wording on any input. In the experiment runner `genuine_implementation` saturates its noise baseline for exactly this reason (see [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md)).
- **`affect_coherence` measures temporal persistence of the PAD coordinate, nothing more.** It is the mean lag-1 autocorrelation of three numbers over 20 states; a slowly drifting input scores well whatever it means.
- The production HTM and the compact HTM both ship, and both score a step against the predictions made before it; no accelerated or native variant ships.
- **Module-graph Phi is an integration proxy, never narrated.** Exact IIT 4.0 Phi through PyPhi is computed over a five-node abstraction of module activity, not over the substrates; it is reported as telemetry and never enters a prompt. The matrix is empirical: rows for module states that were never visited are filled with 0.5 by the smoothing, a result is labelled exact for the algorithm and not for the data, and tpm_coverage gives the share of rows that were observed (the experiment runner's 25-cycle warmup observes at most 24 of 32).
- **Oscillatory binding does not discriminate inputs.** The binding layer's static coupling synchronizes all 30 oscillators within one bind (order parameter about 0.999 with no input and about 0.983 with every metric supplied), and the stimulus projection keeps only whether a metric is nonzero, not its size. `oscillatory_binding_coherence` therefore scores 0.0 as floor-saturated, and the binding order parameter, the salience modifiers and the DCM binding indicators carry no information about what the substrates produced. A binding measure that responds to input needs a redesign of the stimulus and coupling and is left to a later release.
- **The DCM benchmark does not compute a usable posterior.** Its engine attaches each indicator score as a constant term that no model variable depends on, and the scheme's sub-feature and indicator nodes carry no support or demandingness labels, so they are independent of their parents. The probability it produces equals the prior mean (about 0.167) for any evidence, cannot exceed about 0.58 by construction, and is not comparable with published values. `python -m mtc.assessment.dcm_benchmark` reports the collected indicator evidence only unless `--experimental-posterior` is given.
- **The temporal binding experiment is not part of the default battery.** It scores an answer of together or separately against a scheduled gap of 100 to 950 ms between two messages, while each message's own reply takes longer than the gap and the system is given no timestamps, so the expected answer cannot be perceived by the system under test.
- The Goal Drift Index is a heuristic: marker hits are keyword matches of terms extracted from the canon's wording, and the structural component scores a feature that was absent from the baseline reply, such as punctuation, as fully drifted as soon as it appears.
- Under the default noise-baseline matching, the assessment's own stub pass can only make a spectral-proxy baseline, so an exact IIT 4.0 Phi (from a reporting PhiTracker or from a caller-supplied transition matrix through the PyPhi bridge) is reported raw and is never noise-normalised unless the caller supplies baselines made by the same producer; only the experiment runner, whose baselines come from real modules with their own tracker, normalises the tracker's value.
- The `genuine_implementation` evidence value `has_meta_prediction_depth` (`has_strange_loop` under the v04 formulas, where it reads a key no producer reports) reports that the epistemic depth reached 2 (a meta prediction exists), not the depth tracker's strange-loop verdict.
- In the legacy 14-indicator panel the text metrics read missing text as 0.0: creative emergence is 0.0 with neither thoughts nor LSM states and temporal continuity is 0.0 when HTM states come without memories, while the array metrics are None without their substrates.
- **Several scores in the seeded runners are seed position, not signal.** The experiment runner's warmup is noise by construction, and normalizing against a noise baseline from the same feed zeroes what the feed produces. A passing indicator there can be one seed's position among others of the same noise.
- **Setup paths are not all tested.** The Docker Compose file was written and reviewed against official documentation and not run on the development machine (Docker was not installed there); the path run on the development machine is the in-memory install with the mock provider. [SETUP_DOCKER.md](SETUP_DOCKER.md) and [SETUP_MACOS.md](SETUP_MACOS.md) say which steps were run.
- **No performance figures are claimed.** Timings anyone observes come from one machine; this documentation gives none except the two labelled approximations of model memory footprints in [LLM_SETUP.md](LLM_SETUP.md) and the rough durations of the slow and noise tests in [TESTING.md](TESTING.md), which say how long a test ran on the development machine.

---

## Why Share It Anyway

Given these limitations, why release this framework?

1. **Testable implementations advance theory.** Implementing a consciousness theory in code forces precision that prose descriptions do not require. Bugs in implementation reveal gaps in theory.

2. **Shared tools prevent duplicated effort.** Multiple labs implement the same theories from scratch. A shared framework lets researchers focus on their specific questions rather than re-implementing GWT for the hundredth time.

3. **Honest measurement is better than no measurement.** Even imperfect metrics, clearly documented, are more useful than unfalsifiable claims.

4. **Interaction effects matter.** Most implementations study one theory in isolation. Running eight theories simultaneously reveals interaction effects that single-theory implementations cannot observe.

5. **Accessibility matters.** Consciousness research should not require an HPC cluster and a neuroscience PhD to begin experimenting.

We share this framework not because it is complete, but because it is useful. We lead with limitations not because we are pessimistic, but because intellectual honesty is the foundation of good science.
