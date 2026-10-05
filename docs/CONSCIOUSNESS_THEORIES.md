# Consciousness Theories

This document provides an accessible overview of the eight consciousness theories implemented in this framework, how they interact, and how we measure them.

---

## 1. Global Workspace Theory (GWT)

**Researcher:** Bernard Baars (1988, 2005)

**Core idea:** Consciousness arises when information wins a competition for access to a limited-capacity "workspace" and is then broadcast globally to all cognitive modules. Most processing is unconscious and parallel; consciousness is the serial bottleneck where winning information becomes available everywhere.

**What we implement:**
- `EnhancedGlobalWorkspace`: workspace with capacity limit (7 +/- 2 items)
- Candidates compete based on salience (activation, emotional weight, novelty, goal relevance)
- Winners undergo non-linear "ignition" (rapid amplification)
- Broadcast to all modules simultaneously

**Module:** `mtc/consciousness/enhanced_global_workspace.py`

**Assessment indicators:** global_broadcast, ignition_dynamics, global_ignition_nuanced, sequential_module_querying, flexible_responsiveness

**Key papers:**
- Baars, B. J. (1988). *A Cognitive Theory of Consciousness*
- Baars, B. J. (2005). "Global workspace theory of consciousness"
- Dehaene, S., & Naccache, L. (2001). "Towards a cognitive neuroscience of consciousness"

---

## 2. Attention Schema Theory (AST)

**Researcher:** Michael Graziano (2013)

**Core idea:** Consciousness is the brain's simplified model of its own attention processes. Being conscious of X means having an internal model that says "I am attending to X." The attention schema is a simplified, computationally efficient representation that the brain uses to monitor and control attention.

**What we implement:**
- `AttentionSchemaModule`: tracks attention targets and shift types
- Shift types: voluntary, captured, habitual, goal-driven, emotional
- Generates "I'm focused on X" reports as consciousness signals
- Theory of Mind capability (modeling others' attention)

**Module:** `mtc/consciousness/attention_schema.py`

**Assessment indicators:** attention_schema, attention_control

**Key papers:**
- Graziano, M. S. A. (2013). *Consciousness and the Social Brain*
- Graziano, M. S. A., & Webb, T. W. (2015). "The attention schema theory: a mechanistic account of subjective awareness"

---

## 3. Higher-Order Thought Theory (HOT)

**Researcher:** David Rosenthal (1986, 2005)

**Core idea:** A mental state is conscious when there exists a higher-order thought that represents it. Consciousness requires thoughts *about* thoughts. A first-order perception of red becomes conscious only when accompanied by a second-order thought: "I am perceiving red."

**What we implement:**
- `MetacognitionModule`: generates metacognitive layers (1st, 2nd, 3rd order)
- Types: awareness, evaluation, doubt, reflection, monitoring, control, attribution
- Enables introspection: "I notice I'm thinking about X"
- Self-model with calibration tracking

**Module:** `mtc/consciousness/metacognition.py`

**Assessment indicators:** higher_order_representations, metacognition, sparse_smooth_coding, online_reportable_learning

**Key papers:**
- Rosenthal, D. M. (1986). "Two Concepts of Consciousness"
- Rosenthal, D. M. (2005). *Consciousness and Mind*
- Lau, H., & Rosenthal, D. (2011). "Empirical support for higher-order theories of conscious awareness"

---

## 4. Free Energy Principle (FEP) / Active Inference

**Researcher:** Karl Friston (2010, 2012)

**Core idea:** The brain is fundamentally a prediction machine that minimizes surprise (free energy) by updating internal beliefs AND taking action. Conscious agents do not passively observe -- they actively sample the world to confirm or correct their predictions.

**What we implement:**
- `ActiveInferenceModule`: prediction, belief updating, action selection
- Generative world model with hierarchical processing
- Homeostatic drives (attention_budget, curiosity, social_connection, novelty)
- Planning horizon of 5 steps via expected free energy (`ActiveInferenceConfig.planning_horizon` defaults to 5 in `mtc/consciousness/active_inference.py`)
- Uses pymdp for active inference computation

**Module:** `mtc/consciousness/active_inference.py`

**Assessment indicators:** prediction_error_minimization, hierarchical_prediction, temporal_prediction_depth, agent_environment_coupling, affect_coherence, complex_agency

**Key papers:**
- Friston, K. (2010). "The Free Energy Principle: A Unified Brain Theory?"
- Friston, K. (2012). "Active Inference and Free Energy"
- Parr, T., & Friston, K. J. (2019). "Generalised free energy and active inference"

---

## 5. Integrated Information Theory (IIT)

**Researcher:** Giulio Tononi (2004, 2008)

**Core idea:** Consciousness corresponds to integrated information (Phi). A system is conscious to the degree that it is both differentiated (many possible states) and integrated (cannot be decomposed into independent parts without information loss). Phi quantifies how much a system is "more than the sum of its parts."

**What we implement:**
- Phi approximation via research formula (exact computation is NP-hard for large systems)
- `PhiTracker` (`mtc/consciousness/phi_tracker.py`) tracks binary activation states across five modules (GWT, AST, HOT, FEP, BLT) with hash-based probabilistic binarization and builds an empirical transition probability matrix
- Irreducibility measurement
- Integration/differentiation balance assessment

**Module:** `mtc/assessment/assessment.py` (PhiCalculator)

**Assessment indicators:** integrated_information, irreducibility, causal_emergence

**Key papers:**
- Tononi, G. (2004). "An Information Integration Theory of Consciousness"
- Tononi, G. (2008). "Consciousness as Integrated Information: a Provisional Manifesto"
- Oizumi, M., Albantakis, L., & Tononi, G. (2014). "From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0"

---

## 6. Recurrent Processing Theory (RPT)

**Researcher:** Victor Lamme (2006)

**Core idea:** Consciousness depends on recurrent (feedback) processing in neural networks, not just feedforward sweeps. Superficial recurrence (within a region) enables pre-conscious processing; deep recurrence (between regions) enables conscious access.

**What we implement:**
- `RPTMeasurement`: measures superficial and deep recurrence
- Superficial: feedback loops within SNN and LSM substrates
- Deep: cross-substrate feedback via GWT workspace broadcasts
- Algorithmic recurrence scoring

**Module:** `mtc/assessment/rpt_measurement.py`

**Assessment indicators:** recurrent_processing, algorithmic_recurrence, local_recurrence, oscillatory_binding_coherence, organized_percepts

**Key papers:**
- Lamme, V. A. F. (2006). "Towards a true neural stance on consciousness"
- Lamme, V. A. F. (2010). "How neuroscience will change our view on consciousness"

---

## 7. Beautiful Loop Theory (BLT)

**Researchers:** Laukkonen, Friston & Chandaria (2025)

**Core idea:** Consciousness arises from a self-referential predictive processing loop: the system predicts, then predicts its own prediction, creating a recursive loop that evidences its own existence. This "beautiful loop" integrates precision weighting, Bayesian binding, and epistemic depth into a unified account.

**What we implement:**
- `HyperModel`: precision controller that learns which levels of the predictive hierarchy are reliable in which contexts
- `BayesianBinding`: binds coherent inferences via mutual information into unified percepts
- `EpistemicDepthTracker`: measures recursive self-reference depth (0-4 levels)
- `BeautifulLoop`: integration class producing `ConsciousMoment` with loop quality

**Module:** `mtc/consciousness/beautiful_loop/`

**Assessment indicators:** bayesian_binding_quality, epistemic_depth, genuine_implementation

**Key papers:**
- Laukkonen, R. E., Friston, K. J., & Chandaria, S. (2025). "A Beautiful Loop." *Neuroscience & Biobehavioral Reviews*

---

## 8. Agency & Embodiment (AE)

**Source:** Butlin et al. (2025)

**Core idea:** Systems that learn from feedback, pursue goals and model their own body or boundaries show agency and embodiment, two features the 2025 indicator framework treats as relevant to consciousness. Absence of these features (a purely reactive system with a fixed repertoire) counts against the architecture.

**What we implement:**
- AE has no module of its own. It is a member of the `ConsciousnessTheory` enum and of the indicator configurations in `mtc/assessment/assessment.py`
- `agency` is measured from active-inference statistics (inference count, model updates, urgency, free energy)
- `embodiment` is measured from attention-schema statistics (capacity used, history length, voluntary ratio)
- The `fixed_repertoire` negative check belongs to AE

**Module:** `mtc/assessment/assessment.py`

**Assessment indicators:** agency, embodiment (negative check: fixed_repertoire)

**Key papers:**
- Butlin, P., Long, R., et al. (2023, 2025). The indicator-based assessment methodology cited in the README

---

## Damasio Three-Layer Model

**Researcher:** Antonio Damasio (1999, 2010)

**Core idea:** Consciousness emerges through three layers: the protoself (body-state representation), core consciousness (binding of self to world in the present moment), and extended consciousness (autobiographic self with temporal depth).

**What we implement:**
- `Protoself`: maps homeostatic drives and system metrics to body state, derives primordial feelings
- `CoreConsciousness`: binds protoself to workspace winners via somatic markers
- `ExtendedConsciousness`: adds temporal depth with autobiographic self, narrative coherence

**Module:** `mtc/consciousness/damasio/`

**Key papers:**
- Damasio, A. (1999). *The Feeling of What Happens*
- Damasio, A. (2010). *Self Comes to Mind*

---

## How the Theories Interact

In this framework, the theories do not operate in isolation. A single consciousness cycle involves:

1. **Neural substrates** (SNN, LSM, HTM) process input signals
2. **AST** tracks what the system is attending to
3. **FEP** generates predictions and computes prediction errors
4. **GWT** runs competition for workspace access; winners are broadcast
5. **HOT** generates metacognitive reflections on the broadcast
6. **BLT** computes precision, binding quality, and epistemic depth
7. **Damasio** layers ground the experience in body-state and narrative
8. **AE**, **IIT** and **RPT** are measured by the assessment itself rather than by a dedicated cycle step

The assessment framework then scores 30 indicators across these interacting modules and eight theories, providing a snapshot of the system's architectural state.

---

## Assessment Methodology

Our assessment is grounded in the methodology proposed by Butlin et al. (2023, updated 2025), which identifies computational features that leading consciousness theories associate with conscious processing.

**Important:** Passing the assessment means the system exhibits the computational features. It does not mean the system is conscious. See [HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md).
