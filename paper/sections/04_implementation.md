# 4. Implementation

Translating a consciousness theory into code forces interpretive choices that the theory itself leaves open. A theory may describe "competition for workspace access" without specifying the scoring function. It may require "higher-order representations" without defining the data structure. Each implementation decision is an interpretation, and we document ours explicitly so that they can be challenged, refined, or replaced.

This section describes the implementation of each consciousness module, organized by the four phases of the consciousness cycle introduced in Section 3. For each module, we describe what it implements, how it connects to other modules, and what it simplifies. Section 4.6 collects the simplifications into a single table for reference.

## 4.1 Phase I: Competition and Selection

**Global Workspace Theory** is implemented in `EnhancedGlobalWorkspace`. The module manages a population of `WorkspaceCandidate` objects — signals from neural substrates, memory retrieval, external input, and internal thought — that compete for access to a limited-capacity workspace.

Each candidate carries a composite salience score — a weighted blend of four factors: how strongly activated it is (40%), how relevant it is to current goals (30%), how emotionally charged it is (20%), and how recent it is (10%). Formally:

$$s = 0.4 \cdot a_{\text{activation}} + 0.3 \cdot a_{\text{goal}} + 0.2 \cdot a_{\text{emotion}} + 0.1 \cdot a_{\text{recency}}$$

These weights are an interpretive choice — not derived from GWT itself — and are configurable. The decision to weight bottom-up activation most heavily reflects the priority biological systems give to strong sensory signals, but a researcher could justifiably reweight these for a different experimental context.

An `AttentionBottleneck` enforces a capacity limit inspired by Miller's (1956) "magical number seven, plus or minus two" — the well-established finding that human short-term memory holds roughly 5 to 9 items simultaneously. The bottleneck ranks all candidates by salience and admits only the top 5 to 9, discarding the rest to an unconscious buffer. This is a simplification on two fronts: GWT describes a capacity limit but does not specify a fixed number, and the relationship between Miller's short-term memory findings and Baars' workspace capacity is assumed rather than established.

Candidates that sustain activation above a configurable threshold undergo *ignition* — a nonlinear amplification modeled as a step function that marks the transition from unconscious processing to conscious access (Dehaene & Naccache, 2001). An `IgnitionDetector` tracks these events. In biological systems, ignition involves recurrent amplification across prefrontal-parietal networks; we model it as a threshold crossing, capturing the all-or-nothing quality but not the neural dynamics.

Ignited content is broadcast via a `GlobalBroadcast` object to all registered downstream modules. The broadcast is synchronous within the cycle — all modules receive the same content in the same pass. This differs from biological broadcasting, which involves propagation delays across cortical networks.

**Cross-theory input.** Prediction errors from the FEP module in prior cycles modulate candidate salience: content with high prediction error receives a salience boost proportional to the surprise signal. This implements the hypothesis that unexpected content is more likely to reach conscious access — a cross-theory interaction that emerges from the architecture rather than being stipulated by either theory alone.

## 4.2 Phase II: Self-Modeling

### Attention Schema Theory

AST is implemented in `AttentionSchemaModule`. Following Graziano (2013), the module constructs a simplified model of the system's own attention — not the attention mechanism itself, but a *report* about attention. This distinction is important: the actual filtering and selection happens in the workspace bottleneck. The attention schema is a meta-representation of that process.

The module tracks `AttentionTarget` objects and classifies the system's attention state into one of six categories: focused, divided, scanning, absent, hyperfocused, or shifting. It maintains a history of the last 50 targets and uses this history to predict where attention will shift next, implementing AST's claim that the schema is a predictive model.

The prediction mechanism is shallow: it uses pattern matching over a 3-item history window. Biological attention prediction likely involves far richer contextual modeling. We chose simplicity over sophistication here because AST's core claim is that the schema *exists* as a predictive model, not that it predicts with high accuracy.

The module includes a Theory of Mind capability — modeling where *another agent's* attention is directed — implemented through keyword matching when no language model is available. This is a minimal implementation of a feature Graziano considers central to AST's account of social cognition.

**Output.** The module produces an `AttentionSchemaState` that is added to the `ConsciousnessState` and passed as context to downstream modules. The attention strength field feeds directly into the FEP module's precision weighting (Section 4.3).

### Higher-Order Thought Theory

HOT is implemented in `MetacognitionModule`. The module generates explicit `HigherOrderThought` objects — data structures representing thoughts about the system's own mental states — implementing Rosenthal's (2005) requirement that conscious content be accompanied by a higher-order representation.

This is perhaps our most significant interpretive choice. In Rosenthal's theory, higher-order thoughts are properties of cognitive processing, not discrete data structures. We reify them as objects because computational implementation requires something concrete to pass between modules. A HOT in our system is a structured record containing the target mental state, the type of higher-order operation (awareness, evaluation, doubt, reflection, monitoring, control, or attribution), and a confidence score.

The module supports recursive introspection up to depth 3–4: a thought about a thought about a thought. This is capped for computational reasons; human metacognition may be unbounded in principle, though in practice it rarely exceeds a few levels. Each workspace winner from Phase I receives at least one HOT, implementing the theory's claim that workspace access without meta-representation yields unconscious processing.

A `SelfModel` subcomponent (added in Phase 3.4) maintains an aggregate self-representation drawing from the attention schema, active inference beliefs, and homeostatic state. It generates self-predictions — forecasts of the system's own future behavior — and tracks their accuracy over time. This self-calibration data feeds into the epistemic depth tracker in Phase III.

## 4.3 Phase III: Prediction and Inference

### Free Energy Principle

Active inference is implemented in `ActiveInferenceModule`, built on the pymdp library (Heins et al., 2022) for discrete state-space active inference. The module maintains a hierarchical generative model and minimizes variational free energy through belief updating.

The implementation uses discrete state spaces rather than the continuous variational distributions described in Friston's (2010) formulation. The system's knowledge is encoded in three matrices: an observation likelihood matrix **A** ("given the true state, what do I expect to observe?"), a state transition matrix **B** ("given the current state, what state comes next?"), and a preference matrix **C** ("what observations do I prefer?"). These are finite tables of probabilities rather than continuous distributions — a common simplification in computational active inference (Heins et al., 2022), but one that means our free energy values are not directly comparable to the continuous-domain quantities discussed in the theoretical literature.

A `HierarchicalPredictiveProcessor` implements multi-level prediction with 3–4 levels. Each level generates predictions about the level below and receives prediction errors from it. The total prediction error across levels constitutes the system's "surprise" signal. The hierarchy depth is fixed; biological predictive processing likely involves dynamic level creation and pruning.

`HomeostaticDrives` represent the system's internal needs: curiosity, social connection, novelty, and an attention budget. In a standalone framework run, these are initialized to default values and evolve according to internal dynamics. When the framework is embedded in a host system, these drives can be mapped to genuine system state — for example, mapping social connection to interaction frequency, or curiosity to information-seeking behavior. This is the primary interface through which an embedding system grounds the FEP module in real needs.

The observation likelihood matrix **A** learns from experience: when the system observes something unexpected, it adjusts **A** to make that observation more likely given the current believed state. The update is proportional to the gap between what was observed and what was predicted, scaled by a learning rate and by how strongly the system believes it is in each state. Formally: $\mathbf{A} \leftarrow \mathbf{A} + \eta \cdot (\mathbf{t} - \mathbf{A}) \cdot \mathbf{b}^\top$. This is simpler than full Bayesian evidence accumulation but avoids the overshoot problems we encountered with an earlier approach that only updated toward the single most-believed state.

**Cross-theory input.** Attention strength from the AST module modulates precision weights in the predictive processor. Higher attention to a particular level increases the precision (inverse variance) assigned to prediction errors at that level, implementing the theoretical link between attention and precision proposed by Feldman and Friston (2010). This means what the system attends to (AST) directly determines how confidently it infers (FEP).

### Beautiful Loop Theory

BLT is implemented across four components, following Laukkonen, Friston, and Chandaria (2025).

The `HyperModel` is a meta-Bayesian precision controller that operates *above* the hierarchical predictive processor. It learns per-level precision weights from the running variance of prediction errors, adjusting how much confidence the system places in predictions at each hierarchical level. When prediction errors at a given level are consistently low, precision increases; when they are volatile, precision decreases. The hyper-model writes updated precisions back to the processor, closing a loop between meta-level monitoring and object-level inference.

`BayesianBinding` implements the binding of separate inferences into a coherent unified percept. The core question binding answers is: which of the workspace winners "belong together" as parts of a single conscious experience? The module measures this by computing how much information each pair of inferences shares — their mutual information. Two inferences that are highly correlated (knowing one tells you a lot about the other) share high mutual information and are candidates for binding. We approximate mutual information using the correlation between belief vectors: $I(X;Y) \approx -\frac{1}{2}\ln(1 - \rho^2)$, where $\rho$ is the Pearson correlation. A greedy algorithm then builds a coherent set by starting with the strongest pair and iteratively adding the inference that shares the most information with the existing set. The result is a `BoundPercept` — a unified representation with a binding quality score.

This is a simplification on multiple fronts: the binding uses pairwise rather than higher-order mutual information, assumes Gaussian distributions, and selects greedily rather than optimally. These choices make the computation tractable at the cost of theoretical completeness.

The `EpistemicDepthTracker` measures the depth of recursive self-reference on a scale from 0 to 4+:

| Depth | Description | Detected By |
|-------|-------------|-------------|
| 0 | No self-reference | No self-model present |
| 1 | Self-model exists | SelfModel component active |
| 2 | Meta-predictions about self | MetaState tracking prediction accuracy |
| 3 | HOTs about self-model | Metacognition reflecting on self-representation |
| 4+ | Higher-order reflection | Recursive self-reference chains |

The tracker also detects *strange loops* — circular self-reference patterns in the spirit of Hofstadter (1979) — though the detection is heuristic rather than formally verified.

The `BeautifulLoop` integration class combines all three components in a single `process_conscious_moment()` call. Its output, a `ConsciousMoment`, includes a `loop_quality` score and an `is_field_evidencing` flag. Field-evidencing — the system recursively evidencing its own predictions — requires loop quality ≥ 0.4, epistemic depth ≥ 2, and a detected strange loop. This threshold is our interpretation of Laukkonen et al.'s criterion; the theory does not specify numerical cutoffs.

## 4.4 Phase IV: Embodied Integration

Damasio's (1999, 2010) three-layer model is implemented across three classes. Unlike the other modules, which implement formal theories of consciousness, Damasio's framework provides an integrative account of how embodiment connects to cognition. We include it because it addresses a gap the other theories leave open: the relationship between body state and conscious experience.

The `Protoself` maps homeostatic drives and optional system metrics to a six-dimensional `BodyState` vector: metabolic state, energy level, pain signals, circadian phase, temperature, and interoceptive state. From this vector, it derives three *primordial feelings* — pleasure/pain, vitality, and arousal — computed from body state *trends* rather than snapshots. A body that is becoming more satisfied produces positive valence; a body in decline produces negative valence, regardless of current absolute state.

In the standalone framework, the body state is driven by the homeostatic module's internal dynamics. When embedded in a host system, each dimension can be mapped to genuine system state: pain signals to error rates, energy level to resource availability, metabolic state to computational throughput. This mapping is the mechanism through which the framework's embodiment becomes real rather than simulated.

`CoreConsciousness` binds the protoself to workspace winners. It computes a `self_world_binding` score from the perturbation signal (how much the body state changed) multiplied by the salience of workspace content, implementing Damasio's claim that core consciousness arises from the interaction between self and object. It generates `SomaticMarker` objects — emotional tags on workspace content derived from the body state — implementing Damasio's (1994) somatic marker hypothesis. Each marker carries a valence (approach/avoid) and an intensity.

`ExtendedConsciousness` adds temporal depth. It maintains an `AutobiographicSelf` that tracks identity continuity, narrative coherence, and growth trajectory. It computes `episodic_resonance` — how strongly the current moment connects to past experiences — and generates a `narrative_context` string that provides temporal grounding. In the standalone framework, this draws on a limited internal history. When embedded, it can draw on persistent memory systems that accumulate over weeks or months, providing the developmental trajectory the standalone framework deliberately omits.

## 4.5 Neural Substrates

The three neural substrates (described architecturally in Section 3.2) are coordinated by a `NeuralOrchestrator`.

The **spiking neural network** uses snntorch (Eshraghian et al., 2023) with 4,116 leaky integrate-and-fire (LIF) neurons organized in two hidden layers of 2,048 neurons each, running for 10 timesteps per processing step. It implements spike-timing-dependent plasticity (STDP) for online learning. The SNN provides the conscious temporal dynamics layer — fast spike timing and synchrony patterns.

The **liquid state machine** uses reservoirpy (Trouvain et al., 2020) with 5,000 reservoir neurons. It operates at the edge of chaos (spectral radius near 1.0) with recursive least squares (RLS) learning. The LSM provides the subconscious processing layer — echo state dynamics and fading memory that influence cognition without reaching workspace access.

The **hierarchical temporal memory** is a custom implementation with 131,072 cells arranged as 4,096 columns of 32 cells each. It learns temporal sequences and detects anomalies in input patterns. The HTM provides the memory consolidation layer — converting episodic patterns into learned temporal expectations.

The orchestrator runs the substrates sequentially (SNN → LSM → HTM) and collects their outputs into a unified signal dictionary. This sequential execution is a simplification: biological neural processing involves massive parallelism. However, it ensures reproducible results and allows each substrate's output to be inspected independently.

## 4.6 Interpretive Choices and Simplifications

Every implementation simplifies its target theory. The following table collects the most significant simplifications. We include this not as a concession but as a resource: each entry is an opportunity for future work.

| Module | Simplification | Theoretical Consequence |
|--------|---------------|------------------------|
| GWT | Fixed 7±2 capacity with deterministic ranking | May miss dynamic capacity effects that biological workspaces exhibit |
| GWT | Instantaneous broadcast, no propagation delay | Cannot study temporal dynamics of conscious access |
| AST | Schema is a reportable model, not an attention mechanism | Cannot distinguish "having attention" from "modeling attention" within the module |
| AST | Shallow 3-item prediction window | Underestimates biological attention prediction capabilities |
| HOT | Higher-order thoughts as discrete data structures | Reifies what the theory describes as a property of processing |
| HOT | Recursion capped at depth 3–4 | May undercount metacognitive depth in systems with richer self-models |
| FEP | Discrete state spaces, not continuous variational inference | Free energy values not directly comparable to theoretical quantities |
| FEP | 5 homeostatic drives, not full biological homeostasis | Embodiment is schematic rather than comprehensive |
| FEP | Fixed hierarchy depth (3–4 levels) | Cannot study dynamic hierarchy formation |
| BLT | Pairwise mutual information with Gaussian proxy | Misses higher-order statistical dependencies in binding |
| BLT | Greedy binding selection | Binding quality may be suboptimal |
| BLT | Heuristic strange loop detection | May miss subtle recursive patterns |
| Damasio | Six-dimensional body state | Reduces biological homeostasis to a low-dimensional proxy |
| Damasio | Linear primordial feeling derivation | Real body-to-feeling mapping involves nonlinear integration |
| Damasio | Limited internal history for extended consciousness | Autobiographic depth depends on embedding system providing persistent memory |
| IIT | Φ approximation, not exact computation | Integrated information values are estimates, not ground truth |
| RPT | Binary recurrence classification (superficial/deep) | Misses the continuous spectrum of recurrence in biological systems |
| Neural | Sequential substrate execution | Does not model biological parallelism |

These simplifications share a common character: they reduce continuous, high-dimensional, or computationally intractable aspects of the theories to discrete, low-dimensional, tractable approximations. In each case, we preserve the *structural claim* of the theory (there is a workspace, there are higher-order thoughts, there is free energy to minimize) while simplifying the *mechanism* that implements it. Whether the structural claim holds without the original mechanism is precisely the kind of question a computational testbed is designed to explore.
