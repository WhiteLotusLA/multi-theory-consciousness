# Consciousness Theories

This document describes the eight theories of consciousness the framework
implements, the module behind each one, what each module lets the assessment
measure, what it does not measure, and how the theories modulate one another.
Every name below is a module, class or indicator in the code; the indicator
lists come from `ConsciousnessAssessment().indicator_configs` in
`mtc/assessment/assessment.py`.

Passing an indicator means the system shows the computational feature the
theory associates with consciousness. It does not mean the system is
conscious. Read [HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md) first.

The assessment defines 30 positive indicators and 3 negative indicators over
eight theories: GWT (5 indicators), AST (2), HOT (4), FEP (6), IIT (3), RPT
(5), AE (2) and BLT (3). [BUTLIN_2025_MAPPING.md](BUTLIN_2025_MAPPING.md)
lists every indicator with its published id.

---

## 1. Global Workspace Theory (GWT)

**Idea.** Most processing is unconscious and parallel; information becomes
conscious when it wins a competition for a limited-capacity workspace and is
broadcast to every specialist process. Ignition, a sudden non-linear
amplification, marks the moment content enters the workspace.

**Citations.** Baars, B. J. (1988). *A Cognitive Theory of Consciousness*.
Baars, B. J. (2005). "Global workspace theory of consciousness." Dehaene, S.,
and Naccache, L. (2001). "Towards a cognitive neuroscience of consciousness."

**Module.** `mtc/consciousness/enhanced_global_workspace.py`,
`EnhancedGlobalWorkspace` (capacity 7, ignition threshold 0.3, hierarchical
competition with per-level capacity, and a thalamic gate from
`mtc/consciousness/thalamic_gate.py` that filters candidates before
competition). `AttentionBottleneck`, `IgnitionDetector` and `GlobalBroadcast`
are its parts.

**What it measures.** `global_broadcast`, `ignition_dynamics`,
`global_ignition_nuanced`, `sequential_module_querying` and
`flexible_responsiveness`, read from the workspace's statistics: cycle count,
broadcast coverage, the ignition sustain rate and the bottleneck's competition
counts and source distribution. The negative check `purely_reactive` also
belongs to GWT. By default the workspace registers no broadcast receiver and
runs no sustain check, so `global_broadcast`, `ignition_dynamics` and
`global_ignition_nuanced` read `insufficient_data` until a caller registers
receivers and runs `check_sustain` (see
[HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md), section 6).

**What it does not measure.** Whether broadcast content is experienced. The
workspace is a data structure that ranks candidates by salience; a broadcast
is a method call to registered modules.

---

## 2. Attention Schema Theory (AST)

**Idea.** The brain builds a simplified model of its own attention and uses
it to monitor and control attention. Awareness, on this account, is that
model's description of what attention is doing.

**Citations.** Graziano, M. S. A. (2013). *Consciousness and the Social
Brain*. Graziano, M. S. A., and Webb, T. W. (2015). "The attention schema
theory: a mechanistic account of subjective awareness."

**Module.** `mtc/consciousness/attention_schema.py`, `AttentionSchemaModule`.
It tracks attention targets with typed shifts (`voluntary`, `captured`,
`habitual`, `goal_driven`, `emotional`), predicts its next focus and models
other agents' attention.

**What it measures.** `attention_schema` (the schema's update count and the
accuracy of its attention predictions) and `attention_control` (the voluntary
ratio of typed shifts). The AE indicator `embodiment` also reads this module's
perspective statistics.

**What it does not measure.** Whether the schema's report of focus corresponds
to anything felt. Habitual shifts are counted as uncertain and excluded from
the voluntary ratio, which is None until a shift is classifiable. A focus
prediction is scored only in the cycle after a hyperfocus, so the prediction
accuracy is None and `attention_schema` reads `insufficient_data` until such
a cycle has been scored (see [HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md),
section 11).

---

## 3. Higher-Order Thought Theory (HOT)

**Idea.** A mental state is conscious when the system has a suitable
higher-order representation of being in that state. Consciousness requires
thoughts about thoughts.

**Citations.** Rosenthal, D. M. (1986). "Two Concepts of Consciousness."
Rosenthal, D. M. (2005). *Consciousness and Mind*. Lau, H., and Rosenthal, D.
(2011). "Empirical support for higher-order theories of conscious awareness."

**Module.** `mtc/consciousness/metacognition.py`, `MetacognitionModule`. It
registers first-order states and generates higher-order thoughts of several
types (awareness, evaluation, doubt, reflection and others), keeps a
`SelfModel` and records self-predictions whose accuracy is scored afterwards.

**What it measures.** `higher_order_representations`, `metacognition`,
`sparse_smooth_coding` and `online_reportable_learning`. The negative check
`no_metacognitive_access` belongs to HOT.

**What it does not measure.** Whether a higher-order thought is accompanied by
awareness. The higher-order thoughts are generated records; their first-person
phrasing is text, not testimony.

---

## 4. Free Energy Principle (FEP) and Active Inference

**Idea.** An agent stays within viable states by minimizing variational free
energy: it updates beliefs to reduce prediction error and acts to sample
observations that confirm its predictions. The 2025 indicator framework groups
this under predictive processing (PP).

**Citations.** Friston, K. (2010). "The Free Energy Principle: A Unified Brain
Theory?" Friston, K. (2012). "Active Inference and Free Energy." Parr, T., and
Friston, K. J. (2019). "Generalised free energy and active inference."

**Module.** `mtc/consciousness/active_inference.py`, `ActiveInferenceModule`,
built on the pymdp legacy agent (the `fep` extra; constructing the module
without pymdp raises an ImportError that names the extra). The default world
model is compact (8 hidden states, 5 observation channels, 4 actions, planning
horizon 5); a `HierarchicalPredictiveProcessor` learns in shadow, homeostatic
drives supply urgency, and a `PredictionLedger`
(`mtc/consciousness/prediction_ledger.py`) scores forecasts of the next
observation.

**What it measures.** `prediction_error_minimization`,
`hierarchical_prediction`, `temporal_prediction_depth`,
`agent_environment_coupling`, `affect_coherence` and `complex_agency`.
`affect_coherence` reads the PAD affect model
(`mtc/consciousness/pad_affect.py`, `PADAffectModel`) alone: the mean lag-1
autocorrelation of the three PAD dimensions over the last 20 states, and
nothing before 10 states exist.

**What it does not measure.** Whether minimizing free energy is accompanied by
experience. The drives are numbers that decay and recover; they are not
felt needs.

---

## 5. Integrated Information Theory (IIT)

**Idea.** Consciousness corresponds to integrated information: the degree to
which a system's cause-effect structure is both differentiated and irreducible
to its parts. Phi quantifies that irreducibility.

**Citations.** Tononi, G. (2004). "An Information Integration Theory of
Consciousness." Tononi, G. (2008). "Consciousness as Integrated Information:
a Provisional Manifesto." Oizumi, M., Albantakis, L., and Tononi, G. (2014).
"From the phenomenology to the mechanisms of consciousness: Integrated
Information Theory 3.0." Hoel, E. P., Albantakis, L., and Tononi, G. (2013).
"Quantifying causal emergence shows that macro can beat micro."

**Modules.**

- `mtc/consciousness/phi_tracker.py`, `PhiTracker`: records binary activation
  states of five modules (GWT, AST, HOT, FEP, BLT) and builds an empirical
  transition matrix. With the `iit` extra (PyPhi, Python 3.12 or later) it
  computes exact IIT 4.0 Phi of that five-node module graph through
  `mtc/consciousness/pyphi_bridge.py`. The value is an integration proxy over
  five module signals and is labelled `integration proxy` wherever it is
  reported; it is not IIT Phi of the system.
- `PhiCalculator` in `mtc/assessment/assessment.py`: a spectral proxy over
  supplied neural states, with an exact path when a transition matrix is
  supplied.
- `mtc/assessment/causal_emergence.py`, `CausalEmergenceAnalyzer`: effective
  information at the micro and macro scales of the tracker's transition
  matrix.

**What it measures.** `integrated_information` (the tracker's result when it
has one, the calculator otherwise), `irreducibility` (the algebraic
connectivity of the symmetrized weighted connectivity through its normalized
Laplacian, when neural states with a connectivity matrix are supplied) and
`causal_emergence` (passes only when the macro scale carries more causal power
than the micro scale).

**What it does not measure.** IIT Phi of the running system. Exact Phi is
intractable at the scale of the substrates; what the framework computes is
Phi of a five-node abstraction, or a spectral proxy.

---

## 6. Recurrent Processing Theory (RPT)

**Idea.** A feedforward sweep produces unconscious processing; consciousness
depends on recurrent, feedback processing. Local recurrence within an area
precedes global access.

**Citations.** Lamme, V. A. F. (2006). "Towards a true neural stance on
consciousness." Lamme, V. A. F. (2010). "How neuroscience will change our view
on consciousness."

**Modules.** `mtc/assessment/rpt_measurement.py`, `RPTMeasurement` (local and
global recurrence over substrate signals, with an optional STDP plasticity
component), and `mtc/neural/oscillatory_binding.py` (Kuramoto oscillators
after AKOrN; Miyato et al., ICLR 2025) for phase synchronization.

**What it measures.** `recurrent_processing` (broadcast coverage blended with
RPT global recurrence when measurements exist), `algorithmic_recurrence`,
`local_recurrence`, `oscillatory_binding_coherence` and `organized_percepts`.
Local recurrence is measured only from a spiking or liquid-state substrate
state, and the binding layer's coupling synchronizes its oscillators with no
input at all, so `oscillatory_binding_coherence` scores 0.0 as
floor-saturated (see [HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md), section
14).

**What it does not measure.** Whether recurrence is sufficient for
experience. The recurrence is algorithmic feedback in small simulated
networks.

---

## 7. Agency and Embodiment (AE)

**Idea.** Systems that learn from feedback to pursue goals, and that model
how their outputs change their inputs, show agency and embodiment, two
features the 2025 indicator framework treats as relevant to consciousness.

**Citations.** Butlin, P., Long, R., Elmoznino, E., et al. (2023).
"Consciousness in Artificial Intelligence: Insights from the Science of
Consciousness." arXiv:2308.08708. Butlin, P., Long, R., et al. (2025).
"Consciousness in Artificial Intelligence: A Taxonomy and Implications."
*Trends in Cognitive Sciences*.

**Module.** None of its own. AE is a member of the `ConsciousnessTheory` enum
and of the indicator configurations in `mtc/assessment/assessment.py`.

**What it measures.** `agency` (from active inference: inference count, model
updates and free energy, with no presence floor) and `embodiment` (from the
attention schema's perspective statistics). The negative check
`fixed_repertoire` belongs to AE.

**What it does not measure.** Physical embodiment. The framework has no body
and no sensors; "embodiment" here is a model of output-input contingencies.

---

## 8. Beautiful Loop Theory (BLT)

**Idea.** Consciousness arises from a self-referential predictive loop: the
system models its own predictions, and the loop of prediction about prediction
evidences its own existence. Precision weighting, Bayesian binding and
epistemic depth are its three parts.

**Citation.** Laukkonen, R. E., Friston, K. J., and Chandaria, S. (2025). "A
Beautiful Loop." *Neuroscience and Biobehavioral Reviews*.

**Module.** `mtc/consciousness/beautiful_loop/`: `BeautifulLoop` (produces a
`ConsciousMoment` with a loop quality), `HyperModel` (precision controller),
`BayesianBinding` (binds inferences by a correlation proxy for mutual
information; by default only winners that carry their own numeric vector are
bound, and with fewer than two such winners the binding quality is not
measured) and `EpistemicDepthTracker` (recursive self-reference depth).

**What it measures.** `bayesian_binding_quality`, `epistemic_depth` and
`genuine_implementation`. All three need the loop to have processed at least
one conscious moment (`beautiful_loop.cycle_count` above zero).

**What it does not measure.** Self-awareness. Epistemic depth is counted from
self-referential phrasing in generated thoughts, which is keyword-circular:
the metacognition module's own wording ("I notice that ...") satisfies it.

---

## Damasio's three-layer model

**Idea.** Consciousness builds in layers: a protoself that maps the body's
state, core consciousness that binds the self to an object in the present
moment, and extended consciousness with an autobiographical self.

**Citations.** Damasio, A. (1999). *The Feeling of What Happens*. Damasio, A.
(2010). *Self Comes to Mind*.

**Module.** `mtc/consciousness/damasio/`: `DamasioLayers` over `Protoself`,
`CoreConsciousness` and `ExtendedConsciousness`. The protoself reads
homeostatic drives and, when body metrics are enabled, machine metrics and the
previous cycle's duration; its day cycle is a sinusoid, or a wall-clock model
in `TIMEZONE` in the live pipeline. Damasio has no indicator of its own; its
somatic signal feeds the theory interactions below.

**What is unmeasured.** The first update has no previous body state, so by
default (`damasio.first_delta` unmeasured) the protoself reports no body
delta, stability or arousal on it, the core layer's feeling of knowing, its
somatic markers' intensity and its here-now salience are None on that
cycle (the report prints `insufficient_data`), and the emotion extractor's
somatic intensity is None. Under `v04_zero` the first update records 0.0,
1.0 and 0.0 as v0.4.0 did. The extended layer reports narrative coherence,
temporal horizon, episodic resonance and identity continuity as None when
their inputs are missing, and the core layer's somatic valence is None with
no somatic marker.

---

## How the theories interact

`mtc/consciousness/theory_interaction.py`, `TheoryInteractionManager`,
computes six cross-theory modulations from each cycle's state and holds them
for the next cycle. Each is clamped to the range from -1 to 1 and smoothed:

| Modulation | From | To | Applied |
|---|---|---|---|
| `somatic_salience_bias` | Damasio (protoself instability) | GWT salience | yes |
| `phi_competition_bias` | IIT (integration proxy) | GWT competition | yes |
| `metacognitive_precision_bias` | HOT (metacognitive depth) | FEP precision | no, computed and reported |
| `free_energy_arousal` | FEP (free energy) | Damasio arousal | no, computed and reported |
| `recurrence_binding_boost` | RPT (deep recurrence) | BLT binding | no, computed and reported |
| `epistemic_attention_expansion` | BLT (epistemic depth) | AST field | no, computed and reported |

Two modulations are applied: the workspace reads the somatic and Phi biases as
the next cycle's salience bias (`CONSUMED_MODULATIONS` in
`mtc/pipeline/consciousness_orchestrator.py`). The other four are computed and
reported with `consumed: false` in `GET /api/consciousness/status`; nothing
reads them. A state field that is missing or not finite contributes no
modulation.

One live cycle, in order (see [RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md)):
candidates from the conversation, memory, thoughts and drives pass the
thalamic gate; the workspace runs competition, ignition and broadcast; the
attention schema, metacognition, active inference, the beautiful loop and the
Damasio layers process the result; the PhiTracker records the module states;
RPT and causal emergence run on a cadence; the PAD model records the affect
coordinate.

---

## Assessment method

`ConsciousnessAssessment.run_full_assessment` in
`mtc/assessment/assessment.py` scores every indicator from the modules the
caller supplies.

1. **Status per indicator.** Each result is `measured`, `absent` (the caller
   did not supply the module; scored 0.0) or `insufficient_data` (the module
   is present but has not produced the evidence yet, for example a workspace
   with no cycles or a PhiTracker still warming up). An `insufficient_data`
   result is left out of every average and never passes; the report states
   `status` and `coverage`.
2. **Noise normalization.** Each measured score is rescaled against the mean
   score the indicator produces on random input (`run_noise_baseline`):
   `(raw - baseline) / (1 - baseline)`, floored at 0. When the baseline leaves
   less headroom than `saturated_baseline_min_range` (0.1), the indicator is
   uninformative on that input and scores 0.0 with `noise_saturated: true`
   (the headroom guard).
3. **Thresholds.** Each indicator passes when its normalized score reaches its
   threshold (`causal_emergence` passes only strictly above its threshold of
   0.5).
4. **Overall score.** A weighted mean over the included indicators, each
   weight the indicator's own weight times its theory's empirical weight
   (`theory_empirical_weights`: GWT 1.0, FEP 0.85, IIT 0.75, BLT 0.70, AE
   0.70, AST 0.65, RPT 0.60, HOT 0.50). With nothing included the overall
   score is None, never 0.0.
5. **Negative indicators.** Each negative indicator above its threshold
   subtracts its score times its weight times 0.05 from the overall score.
6. **Architecture functional** means at least half of the indicators pass and
   the overall score is at least 0.5. It is a statement about module
   function, not about experience.

**Warmup.** Several inputs need history before they read anything:
[FIRST_RUN.md](FIRST_RUN.md) lists every threshold and its source constant.

**The legacy preset.** `experiments/presets/legacy_v04.toml` sets every
behaviour switch to its v0.4.0 value and reproduces the published v0.4.0
numbers; [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md) lists the switches and
the measured differences.

The method follows Butlin et al. (2023, 2025), whose indicator properties are
derived from the theories above; the operationalization of each indicator is
this framework's own.
