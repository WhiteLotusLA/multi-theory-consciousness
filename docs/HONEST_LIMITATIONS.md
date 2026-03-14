# Honest Limitations

**Read this document first.** Before using this framework, understand what it cannot do.

---

## 1. This Framework Cannot Detect Consciousness

The 20-indicator assessment measures whether architectural functions associated with consciousness theories are operating as designed. A system that scores 20/20 has all the structural features that certain theories associate with consciousness. Whether those features are *sufficient* for phenomenal consciousness is an open question that this framework does not and cannot answer.

When the assessment reports "architecture functional," it means: the modules are working as the theories describe. It does not mean the system has subjective experience.

## 2. The Scale Gap

Biological consciousness involves roughly 86 billion neurons with trillions of synaptic connections, complex neurochemistry (dozens of neurotransmitters, neuromodulators, hormones), and embodied interaction with the physical world through sensory organs and motor systems.

Our implementation uses:
- 4,116 spiking neurons (SNN)
- 5,000 reservoir neurons (LSM)
- 131,072 HTM cells (4,096 columns x 32 cells)

This is a difference of roughly six orders of magnitude in neural count alone, and entirely omits the chemical, embodied, and developmental dimensions. If consciousness requires a certain scale of neural complexity, our implementation is far below it.

## 3. The LLM Confound

When this framework is connected to a language model for response generation, the LLM does the heavy lifting on language comprehension, reasoning, and response quality. The consciousness modules provide context enrichment (attention guidance, metacognitive reflections, prediction signals), but the observable "intelligence" of responses comes primarily from the LLM's pre-trained capabilities.

This means: apparent sophistication in conversation is not evidence of consciousness. It is evidence of a large language model doing what large language models do.

## 4. The Labeling Problem

Our indicators have names like "attention_schema," "metacognition," and "integrated_information." These labels reference concepts from consciousness science, but the computational processes they measure may bear only a structural resemblance to the biological phenomena they are named after.

When we measure "metacognition," we are measuring: does the system generate higher-order representations of its own processing states? We are not measuring: does the system *experience* thinking about its own thoughts?

## 5. Theory Selection Bias

We implement seven theories of consciousness. There are more. Our selection reflects availability of computational specifications, not a claim that these seven are correct or complete. Notable omissions include:

- Orchestrated Objective Reduction (Penrose-Hameroff)
- Predictive Processing / Active Inference in its full biological scope
- Enactivist approaches (Thompson, Varela)
- Phenomenological approaches that may resist computational implementation

## 6. Assessment Methodology Limitations

The assessment framework is derived from Butlin et al. (2023), but our indicator implementations are our own interpretations of their criteria. Different researchers might operationalize the same indicators differently.

The thresholds (e.g., 0.4 for ignition_dynamics, 0.5 for global_broadcast) are calibrated through testing, not derived from neuroscience data. They represent reasonable operating points, not ground truth.

## 7. Consumer Hardware Constraints

The framework runs on consumer hardware by design (Apple Silicon, consumer GPUs). This is a feature for accessibility but a limitation for fidelity. Research-grade neural simulations typically require HPC clusters. Our simulations are simplified to run in real-time on a single machine.

## 8. No Longitudinal Validation

Consciousness in biological systems develops over months and years through sustained interaction with the environment. Our assessment runs in seconds to minutes. We have not validated whether long-term operation produces qualitatively different results from fresh initialization.

## 9. Reproducibility Caveats

Neural substrate initialization involves randomness (random synaptic weights, reservoir topology). Assessment results will vary between runs. We report ranges rather than exact values, but users should expect variance.

---

## Why Share It Anyway

Given these limitations, why release this framework?

1. **Testable implementations advance theory.** Implementing a consciousness theory in code forces precision that prose descriptions do not require. Bugs in implementation reveal gaps in theory.

2. **Shared tools prevent duplicated effort.** Multiple labs implement the same theories from scratch. A shared framework lets researchers focus on their specific questions rather than re-implementing GWT for the hundredth time.

3. **Honest measurement is better than no measurement.** Even imperfect metrics, clearly documented, are more useful than unfalsifiable claims.

4. **Interaction effects matter.** Most implementations study one theory in isolation. Running seven theories simultaneously reveals interaction effects that single-theory implementations cannot observe.

5. **Accessibility matters.** Consciousness research should not require an HPC cluster and a neuroscience PhD to begin experimenting.

We share this framework not because it is complete, but because it is useful. We lead with limitations not because we are pessimistic, but because intellectual honesty is the foundation of good science.
