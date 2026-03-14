# 2. Background

We briefly review the seven theories implemented in the MTC framework, followed by a survey of prior computational implementations and the gap our work addresses. For comprehensive treatments of each theory, we refer readers to the primary sources cited below and to the recent comparative analyses by Doerig, Schurger & Herzog (2020) and Seth & Bayne (2022).

## 2.1 Theories of Consciousness

**Global Workspace Theory (GWT).** Baars (1988) proposed that consciousness arises when information wins a competition for access to a limited-capacity "global workspace" and is then broadcast to a wide network of specialized processors. Dehaene and Naccache (2001) formalized this as "neuronal global workspace theory," identifying prefrontal-parietal networks as the neural substrate and proposing an "ignition" threshold — a nonlinear amplification event that distinguishes conscious from unconscious processing. GWT predicts that conscious content is globally available, while unconscious content remains locally processed.

**Attention Schema Theory (AST).** Graziano (2013; Webb & Graziano, 2015) proposed that consciousness is the brain's simplified, self-directed model of its own attention. The brain constructs an "attention schema" — an internal description of the process of attention — just as it constructs body schemas for motor control. Under AST, subjective awareness is not a byproduct of attention but is the schema itself: a predictive model the brain uses to monitor and control its attentional state. AST predicts that a system should be able to report on its attention and predict its own attentional shifts.

**Higher-Order Thought Theory (HOT).** Rosenthal (2005; Lau & Rosenthal, 2011) argued that a mental state becomes conscious when there exists a higher-order representation of that state. A first-order perception of red becomes conscious when accompanied by a second-order thought: "I am perceiving red." HOT predicts a hierarchy of meta-representations, with deeper levels of self-reflection corresponding to richer conscious experience. The theory draws a sharp distinction between first-order content and the higher-order representations that make that content conscious.

**Free Energy Principle (FEP).** Friston (2010) proposed that biological systems minimize variational free energy — an upper bound on surprisal — through a combination of perceptual inference (updating internal beliefs) and active inference (acting on the world to fulfill predictions). Applied to consciousness, the FEP suggests that conscious experience corresponds to the system's best explanation of sensory data: a posterior belief that minimizes prediction error across a hierarchical generative model. Homeostatic drives anchor this process to embodied needs (Pezzulo, Rigoli & Friston, 2015), connecting prediction minimization to survival.

**Integrated Information Theory (IIT).** Tononi (2004, 2008; Oizumi, Albantakis & Tononi, 2014) proposed that consciousness is identical to integrated information, denoted Φ (phi). A system is conscious to the degree that it is both differentiated (many possible states) and integrated (the whole generates more information than the sum of its parts). IIT is notable for being formulated as a mathematical framework with specific axioms (intrinsicality, composition, information, integration, exclusion) and postulates connecting them to physical mechanisms. Computing Φ exactly is computationally intractable, growing super-exponentially with system size (Tegmark, 2016), requiring approximations for systems beyond trivial size.

**Recurrent Processing Theory (RPT).** Lamme (2006, 2010) proposed that consciousness depends on recurrent (feedback) processing in the brain, distinguishing between feedforward sweeps (which occur unconsciously) and recurrent loops (which give rise to phenomenal experience). RPT further distinguishes superficial recurrence (local feedback within sensory areas, associated with phenomenal consciousness) from deep recurrence (involving frontal areas and global feedback, associated with access consciousness and reportability). RPT makes specific predictions about the neural dynamics required for different aspects of conscious experience.

**Beautiful Loop Theory (BLT).** Laukkonen, Friston, and Chandaria (2025) proposed that consciousness arises from a recursive loop of precision-weighted inference over a hierarchical generative model. The theory introduces a "hypermodel" — a meta-Bayesian controller that allocates precision (confidence weights) across levels of a predictive hierarchy — and proposes that phenomenal experience corresponds to "field-evidencing": the recursive process of the model evidencing its own predictions. BLT integrates elements of FEP and IIT, adding explicit mechanisms for Bayesian binding of inferences into unified percepts and epistemic depth (recursive self-reference).

**Damasio's Three-Layer Model.** While not a formal theory of consciousness in the same sense as the above, Damasio's (1999, 2010) framework provides an integrative account linking embodiment to higher cognition through three layers: (1) the *protoself*, mapping the body's internal state through primordial feelings (pleasure/pain, vitality, arousal); (2) *core consciousness*, binding the protoself to objects of attention through somatic markers; and (3) *extended consciousness*, adding autobiographic memory and temporal continuity. We include Damasio's model as an integrative layer connecting homeostatic drives to the cognitive processes described by the other theories.

## 2.2 Prior Computational Implementations

Computational implementations of individual consciousness theories have provided valuable tools for their respective communities, but each addresses a single theory:

- **IIT:** PyPhi (Mayner et al., 2018) provides exact Φ computation for small networks. It is the standard tool for IIT research but does not implement any other theory or measure cross-theory interactions.

- **FEP/Active Inference:** Multiple implementations of active inference agents exist (Fountas et al., 2020; Tschantz et al., 2020; the pymdp library by Heins et al., 2022), providing prediction error minimization and action selection. These implement the FEP as an agent architecture but do not integrate workspace competition or higher-order meta-representation.

- **GWT:** Shanahan (2008) implemented a global workspace architecture with spiking neural networks. VanRullen and Kanai (2021) proposed computational principles for GWT-based architectures. Neither integrates IIT measurement, active inference, or attention schema modeling.

- **Neural substrates:** Large-scale neural simulations (e.g., the Human Brain Project's NEST simulator; Gewaltig & Diesmann, 2007) model neural dynamics at scale but are not organized around consciousness theories and do not include theory-specific assessment.

To our knowledge, no existing framework implements multiple consciousness theories as interacting modules within a shared architecture, nor provides a standardized assessment that measures indicators across theories simultaneously.

## 2.3 The Multi-Theory Gap

The single-theory approach has a structural limitation: it cannot reveal how theories interact. Consider two examples from our framework that illustrate this:

First, prediction error signals from the FEP module modulate salience scores in the global workspace competition. Content with high prediction error is more surprising, receives higher salience, and is more likely to win workspace access and be broadcast. This means the FEP is not merely describing a parallel process — it is actively shaping what enters conscious access as defined by GWT. This interaction effect is invisible when either theory is implemented alone.

Second, higher-order thoughts from the HOT module feed into the epistemic depth tracker of the Beautiful Loop module. The presence of meta-representations about self-predictions increases the measured depth of recursive self-reference, which in turn affects the loop quality score. HOT and BLT are not merely coexisting — they amplify each other's metrics in ways that their respective theorists may or may not predict.

Whether these interaction effects are theoretically meaningful or artifacts of our implementation choices is an open question — one that a shared computational testbed is well positioned to investigate. Crucially, studying these interactions requires a controlled environment. A system embedded in real-world interaction introduces confounds — conversational context, accumulated memory, personality development — that make it difficult to attribute effects to the theories themselves. The MTC framework provides this controlled environment while remaining architecturally ready to be embedded in richer systems where the theories operate on genuine input.
