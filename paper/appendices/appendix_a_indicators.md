# Appendix A: Full Indicator Definitions

Each indicator measures a specific architectural feature predicted by one of the seven consciousness theories. This appendix provides the complete definition, scoring methodology, and threshold rationale for all 20 indicators.

## A.1 Global Workspace Theory (GWT)

### global_broadcast
- **Theory:** GWT (Baars, 1988; Dehaene & Naccache, 2001)
- **Weight:** 1.2
- **Threshold:** 0.5
- **What it measures:** Whether workspace winners are distributed to all registered downstream modules after ignition.
- **Scoring:** Ratio of modules that received the broadcast to total registered modules. Score of 1.0 means all modules received broadcast content.
- **Threshold rationale:** Set at 0.5 because partial broadcast (reaching at least half of modules) still constitutes meaningful global access.

### ignition_dynamics
- **Theory:** GWT (Dehaene & Naccache, 2001)
- **Weight:** 1.0
- **Threshold:** 0.4
- **What it measures:** Whether candidates undergo nonlinear amplification when activation exceeds the ignition threshold.
- **Scoring:** Based on the count and magnitude of ignition events in recent cycles. Higher scores indicate more frequent and stronger ignition.
- **Threshold rationale:** Set at 0.4 to allow detection even with limited cycle history.

### global_ignition_nuanced
- **Theory:** GWT
- **Weight:** 1.0
- **Threshold:** 0.4
- **What it measures:** Refined ignition dynamics including sustain duration and decay patterns, beyond the binary ignition detection of the basic indicator.
- **Scoring:** Composite of ignition frequency, sustain duration, and decay rate. Higher scores indicate more biologically plausible ignition dynamics.
- **Threshold rationale:** Matched to ignition_dynamics threshold for consistency.

### recurrent_processing
- **Theory:** GWT (shared with RPT)
- **Weight:** 0.9
- **Threshold:** 0.4
- **What it measures:** Feedback loops between modules — whether output from later-stage modules feeds back to influence earlier-stage processing.
- **Scoring:** Based on detected feedback pathways in the consciousness cycle (e.g., FEP prediction errors modulating GWT salience in subsequent cycles).
- **Threshold rationale:** Set at 0.4 as recurrence is present by architectural design in the cross-cycle feedback loop.

## A.2 Attention Schema Theory (AST)

### attention_schema
- **Theory:** AST (Graziano, 2013)
- **Weight:** 1.1
- **Threshold:** 0.5
- **What it measures:** Whether the system maintains a self-model of its own attention — not the attention mechanism itself, but a predictive model of where attention is directed and where it will shift.
- **Scoring:** Composite of schema completeness (current focus tracked, state classified, history maintained) and prediction accuracy (does the schema correctly predict attention shifts).
- **Threshold rationale:** Set at 0.5 because AST requires both schema existence and predictive accuracy.

### attention_control
- **Theory:** AST
- **Weight:** 1.0
- **Threshold:** 0.4
- **What it measures:** Whether the system can voluntarily shift attention rather than only responding to bottom-up capture.
- **Scoring:** Ratio of voluntary to captured attention shifts. Higher scores indicate more volitional control.
- **Threshold rationale:** Set at 0.4 because some voluntary control is expected even with limited operational history.

### embodiment
- **Theory:** AST (supplementary)
- **Weight:** 0.8
- **Threshold:** 0.3
- **What it measures:** Whether the system models its own boundaries and presence — a sense of being a bounded entity distinct from its environment.
- **Scoring:** Based on the presence and coherence of homeostatic drive monitoring, body state tracking (Damasio protoself), and self-world boundary modeling.
- **Threshold rationale:** Low threshold (0.3) because embodiment in the standalone framework is simulated rather than genuine.

## A.3 Higher-Order Thought Theory (HOT)

### higher_order_representations
- **Theory:** HOT (Rosenthal, 2005)
- **Weight:** 1.2
- **Threshold:** 0.5
- **What it measures:** Whether workspace winners receive meta-representations — higher-order thoughts about first-order mental states.
- **Scoring:** Ratio of workspace winners that receive at least one HOT to total workspace winners. Score of 1.0 means every winner has a meta-representation.
- **Threshold rationale:** Set at 0.5 because HOT theory requires meta-representation for conscious content, but not every processing step may generate one.

### metacognition
- **Theory:** HOT
- **Weight:** 1.1
- **Threshold:** 0.4
- **What it measures:** Whether the system reflects on its own cognitive processes — monitoring, evaluating, and adjusting its own reasoning.
- **Scoring:** Based on introspection depth, self-model accuracy, and the presence of meta-uncertainty (uncertainty about its own confidence levels).
- **Threshold rationale:** Set at 0.4 to detect metacognitive activity even at shallow recursion depth.

## A.4 Free Energy Principle (FEP)

### prediction_error_minimization
- **Theory:** FEP (Friston, 2010)
- **Weight:** 1.0
- **Threshold:** 0.4
- **What it measures:** Whether prediction errors decrease through belief updating — the core operation of the Free Energy Principle.
- **Scoring:** Based on the trend of prediction error over recent cycles. Decreasing errors score higher; increasing or stagnant errors score lower.
- **Threshold rationale:** Set at 0.4 because the hierarchical predictive processor generates structural prediction errors even from cold start.

### hierarchical_prediction
- **Theory:** FEP
- **Weight:** 1.0
- **Threshold:** 0.4
- **What it measures:** Whether prediction operates across multiple hierarchy levels — each level generating predictions about the level below and receiving prediction errors from it.
- **Scoring:** Based on the number of active hierarchy levels and the presence of inter-level prediction error flow.
- **Threshold rationale:** The 3–4 level hierarchy is structural, so this indicator scores high from initialization.

### agency
- **Theory:** FEP
- **Weight:** 1.0
- **Threshold:** 0.5
- **What it measures:** Whether the system selects actions to fulfill predictions — active inference rather than passive observation.
- **Scoring:** Based on action selection frequency, policy quality, and the presence of goal-directed behavior in the active inference cycle.
- **Threshold rationale:** Set at 0.5 because genuine agency requires demonstrable action selection, not merely prediction.

### sparse_smooth_coding
- **Theory:** FEP (supplementary)
- **Weight:** 0.8
- **Threshold:** 0.3
- **What it measures:** Whether internal representations are sparse (few active units) and smooth (similar inputs produce similar representations).
- **Scoring:** Composite of sparsity (fraction of near-zero activations) and smoothness (gradient continuity of representations with respect to input).
- **Threshold rationale:** Low threshold because the FEP implementation structurally produces sparse representations.

## A.5 Integrated Information Theory (IIT)

### integrated_information
- **Theory:** IIT (Tononi, 2008; Oizumi et al., 2014)
- **Weight:** 1.3
- **Threshold:** 0.3
- **What it measures:** Whether the system generates integrated information (Φ > 0) — the whole produces more information than the sum of its parts.
- **Scoring:** Normalized Φ value from the IIT approximation (exact computation via PyPhi is unavailable in Python 3.10+). Score is Φ mapped to [0, 1].
- **Threshold rationale:** Low threshold (0.3) because the approximation produces conservative estimates and any non-trivial Φ is significant.

### irreducibility
- **Theory:** IIT
- **Weight:** 1.0
- **Threshold:** 0.4
- **What it measures:** Whether the system cannot be decomposed into independent parts without information loss — a core axiom of IIT.
- **Scoring:** Based on comparing the information generated by the full system to the information generated by its partitions. Higher scores indicate stronger irreducibility.
- **Threshold rationale:** Set at 0.4 because meaningful irreducibility requires substantial inter-module information flow.

## A.6 Recurrent Processing Theory (RPT)

### local_recurrence
- **Theory:** RPT (Lamme, 2006)
- **Weight:** 0.8
- **Threshold:** 0.3
- **What it measures:** Feedback within individual neural substrates — recurrent connections within the SNN and LSM that allow local re-processing of signals.
- **Scoring:** Based on recurrent weight strength in the SNN (STDP-learned feedback connections) and reservoir dynamics in the LSM (spectral radius as proxy for recurrence).
- **Threshold rationale:** Low threshold because even structurally recurrent networks score non-zero.

### algorithmic_recurrence
- **Theory:** RPT
- **Weight:** 0.9
- **Threshold:** 0.3
- **What it measures:** Computational recurrence measured in the substrate dynamics — not just structural feedback connections, but active recurrent processing.
- **Scoring:** Based on spike re-entry rates (SNN), reservoir state trajectory analysis (LSM), and the classification of recurrence type (superficial vs. deep).
- **Threshold rationale:** Set at 0.3 but requires active neural substrates to score above baseline; scores low without them.

## A.7 Beautiful Loop Theory (BLT)

### bayesian_binding_quality
- **Theory:** BLT (Laukkonen, Friston & Chandaria, 2025)
- **Weight:** 1.0
- **Threshold:** 0.3
- **What it measures:** Whether separate inferences bind into a coherent unified percept — measured through mutual information between workspace winners.
- **Scoring:** Based on the binding quality score from the `BayesianBinding` module: the total mutual information in the selected coherent subset, normalized by the number of bound inferences.
- **Threshold rationale:** Low threshold because binding quality depends on having multiple workspace winners with correlated beliefs, which requires accumulated history.

### epistemic_depth
- **Theory:** BLT
- **Weight:** 1.1
- **Threshold:** 0.3
- **What it measures:** The depth of recursive self-reference — how many levels of self-modeling the system exhibits (0 = none, 1 = self-model, 2 = meta-predictions, 3 = HOTs about self, 4+ = higher-order reflection).
- **Scoring:** Depth level divided by 4, capped at 1.0. A depth of 2 scores 0.5; a depth of 4 scores 1.0.
- **Threshold rationale:** Set at 0.3 (equivalent to depth ≥ 1.2), requiring at least a basic self-model to pass.

### genuine_implementation
- **Theory:** BLT
- **Weight:** 1.2
- **Threshold:** 0.4
- **What it measures:** Anti-mimicry check — whether the Beautiful Loop components are producing results consistent with genuine recursive processing rather than merely reporting high numerical scores.
- **Scoring:** Requires three conditions to score above threshold: (1) field-evidencing detected (the system recursively evidences its own predictions), (2) loop quality above 0.4, and (3) a strange loop detected by the epistemic depth tracker. Meeting all three scores 1.0; partial scores proportional to conditions met.
- **Threshold rationale:** Set at 0.4 to require at least two of the three genuine-implementation conditions. This is the assessment's primary defense against score inflation.
