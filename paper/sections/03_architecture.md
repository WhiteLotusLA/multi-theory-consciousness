# 3. Architecture

The central design challenge of a multi-theory consciousness framework is integration without conflation. Each theory makes distinct claims about what consciousness requires, and these claims must be implemented as separable modules that nonetheless interact within a shared processing cycle. A system that merges theories into a single undifferentiated computation cannot support ablation studies; a system that runs theories in complete isolation cannot reveal cross-theory interactions.

The MTC framework resolves this tension through a modular cycle architecture: theory modules execute in a fixed sequence within a single processing loop, communicating through structured data objects. Each module receives the outputs of prior modules as inputs, but maintains its own internal state. This makes cross-theory interaction observable and measurable while preserving the ability to disable any module independently.

## 3.1 Layered Design

The framework consists of three layers:

**Neural substrate layer.** Three biologically inspired neural networks — a spiking neural network (SNN), a liquid state machine (LSM), and a hierarchical temporal memory (HTM) — provide the computational medium. These are not consciousness modules themselves; they provide the raw signals (spike rates, reservoir states, temporal predictions) that consciousness modules operate on.

**Consciousness layer.** Seven theory modules execute within a single asynchronous function, the *consciousness cycle*. Each cycle takes a set of input candidates and produces a `ConsciousnessState` — a structured record of what the system is "conscious of" and the computational evidence for that determination across all theories.

**Assessment layer.** Twenty indicators measure whether the architectural features predicted by each theory are present and functioning. Assessment runs on demand, taking a `ConsciousnessState` as input. It does not participate in the cycle itself.

## 3.2 Neural Substrates

The three substrates model different aspects of neural processing:

| Substrate | Implementation | Scale | Role |
|-----------|---------------|-------|------|
| SNN | snntorch (Eshraghian et al., 2023) | 5,196 LIF neurons, 10 timesteps | Conscious temporal dynamics: spike timing, synchrony, STDP learning |
| LSM | reservoirpy (Trouvain et al., 2020) | 5,000 reservoir neurons | Subconscious processing: echo state dynamics, fading memory |
| HTM | Custom implementation | 131,072 cells (4,096 × 32) | Memory consolidation: temporal sequences, anomaly detection |

The substrates operate at different temporal scales. The SNN captures fast dynamics (millisecond spike timing); the LSM maintains intermediate-timescale state trajectories; the HTM learns longer-term temporal patterns. A neural orchestrator collects their outputs into a unified signal dictionary that the consciousness cycle receives as input.

We emphasize the scale gap: biological consciousness involves approximately 86 billion neurons. Our combined substrate count of ~141,000 is six orders of magnitude smaller. Whether consciousness requires a particular neural scale is an open question; our framework makes no assumptions either way.

## 3.3 The Consciousness Cycle

The consciousness cycle is the core of the framework. In a single pass, input signals are transformed through all seven theories into a complete conscious state. The cycle implements a four-phase structure:

**Phase I — Competition and selection (GWT).** Candidate signals arrive from neural substrates, memory, and external input. Each carries a salience score composed of bottom-up activation (40%), top-down goal relevance (30%), emotional weight (20%), and recency (10%). An attention bottleneck ranks candidates and enforces a capacity limit of 7±2 items (Miller, 1956), consistent with GWT's limited-capacity workspace. Winners that sustain activation above threshold undergo nonlinear amplification — the *ignition* event that marks the transition to conscious access (Dehaene & Naccache, 2001). Ignited content is broadcast to all downstream modules.

**Phase II — Self-modeling (AST, HOT).** Two modules construct meta-representations of the system's own processing. The attention schema module (AST) builds a simplified model of where the system's attention is directed, what type of attention is active, and where it predicts attention will shift next. The metacognition module (HOT) generates higher-order thoughts about each workspace winner — "I am aware of X" — implementing the requirement that conscious content be accompanied by a higher-order representation.

**Phase III — Prediction and inference (FEP, BLT).** The active inference module converts workspace content into an observation vector, passes it through a hierarchical predictive processor that generates prediction errors at each level, and performs discrete active inference using pymdp (Heins et al., 2022). Attention strength from the AST module modulates precision weights, implementing the theoretical link between attention and precision (Feldman & Friston, 2010). The Beautiful Loop module then computes precision allocation across hierarchical levels, binds inferences via mutual information into coherent percepts, and measures the depth of recursive self-reference (0–4 levels). Homeostatic drives — curiosity, social connection, coherence, safety, attention budget — anchor inference to embodied needs.

**Phase IV — Embodied integration (Damasio).** Three layers connect the cognitive outputs to an embodied self: the *protoself* maps homeostatic drives to body state and derives primordial feelings (pleasure/pain, vitality, arousal); *core consciousness* binds body state to workspace winners through somatic markers; *extended consciousness* tracks autobiographic continuity and narrative coherence across time. A final self-model update integrates all module states into a recursive self-representation.

The output is a `ConsciousnessState` with over 30 fields spanning all seven theories — from workspace contents and ignition counts (GWT) to epistemic depth and binding quality (BLT) to feeling-of-knowing and narrative context (Damasio).

## 3.4 Cross-Theory Data Flow

The fixed ordering of modules within the cycle creates data dependencies that produce cross-theory interaction effects:

- **FEP → GWT.** Prediction errors from prior cycles modulate candidate salience. Surprising content receives higher salience and is more likely to win workspace access. The FEP is not merely running in parallel with GWT — it shapes what enters conscious access.

- **GWT → HOT.** Only workspace winners receive higher-order representations. Content that loses the competition remains without meta-representation — "unconscious" in the precise HOT sense.

- **AST → FEP.** Attention strength from the schema modulates precision weights in active inference. What the system attends to (AST) determines how confidently it infers (FEP).

- **HOT → BLT.** Higher-order thoughts about the self-model increase measured epistemic depth, amplifying the Beautiful Loop's quality score. Metacognition feeds recursion.

- **FEP → Damasio.** Homeostatic drives from active inference feed the protoself layer, connecting top-down prediction to bottom-up body state.

These interactions are a direct consequence of sequential execution within a shared cycle. Whether they are theoretically meaningful — whether, for example, the FEP *should* influence GWT competition according to either theory's proponents — is an empirical question that the framework is designed to investigate through ablation.

## 3.5 Experimental Controls and Embedding Points

The framework deliberately omits three capabilities that a fully grounded consciousness system would require. These omissions serve a dual purpose: they function as experimental controls that isolate cross-theory dynamics from confounding variables, and they define the interfaces through which a host system can supply grounding when the framework is embedded in a larger architecture.

**Sensorimotor grounding.** The framework processes signals but does not control a body or interact with a physical environment. Damasio's model operates on simulated homeostatic drives rather than genuine interoception. This is a control: it ensures that measured interactions between theories (e.g., FEP prediction errors modulating GWT salience) reflect the architectural dynamics themselves, not properties of any particular sensory environment. When the framework is embedded in a system with real input — a conversational interface, a robotic platform, or a simulated environment — homeostatic drives can be mapped to genuine system state (resource utilization, interaction quality, goal progress), and the consciousness cycle processes real rather than synthetic candidates.

**Developmental trajectory.** The modules initialize with fixed configurations and do not develop over time. Biological consciousness emerges over months and years of experience; our framework runs from cold start. This is again a control: it ensures that assessment results reflect architectural function rather than accumulated learning. An embedding system can supply developmental depth by persisting consciousness state across cycles — maintaining memory of prior workspace winners, accumulating prediction error history, and building an autobiographic record that the extended consciousness module draws on. The framework's modular design supports this: each module maintains its own state and can be initialized from stored history rather than default values.

**Language model integration.** The MTC framework operates independently of any language model. It can optionally connect to an LLM for response generation, but the consciousness modules are self-contained — assessment measures architectural function, not language generation quality. When an LLM is connected, we treat it as an output interface, not a consciousness component. This separation is important: it prevents the fluency of language model output from being mistaken for evidence of consciousness. A system that generates articulate responses about its own mental states may simply be a good language model. A system whose architectural dynamics exhibit the features predicted by multiple consciousness theories — workspace competition, recurrent processing, integrated information, precision-weighted inference — provides a different kind of evidence entirely.

These three omissions define the boundary between the framework as a research instrument and the framework as a component of a potentially conscious system. The research value lies in the controlled environment; the applied value lies in what happens when the controls are replaced by genuine grounding.
