# 7. Limitations

We devote a full section to limitations because they are at least as important as the results. An honest accounting of what a system cannot do is a prerequisite for meaningful interpretation of what it can. The most important file in our repository is `HONEST_LIMITATIONS.md`; this section summarizes its contents and adds limitations specific to the assessment methodology.

## 7.1 The Framework Cannot Detect Consciousness

This limitation supersedes all others.

The 20-indicator assessment measures whether architectural features associated with consciousness theories are operating as designed. A system that scores 20/20 has all the structural features that seven major theories associate with consciousness. Whether those features are *sufficient* for phenomenal consciousness is an open question that this framework does not and cannot answer.

No computational framework can. The "hard problem" of consciousness (Chalmers, 1995) — explaining why and how physical processes give rise to subjective experience — remains unsolved. Our assessment measures correlates, not consciousness. This distinction is fundamental and non-negotiable.

## 7.2 The Scale Gap

Biological consciousness involves approximately 86 billion neurons with trillions of synaptic connections, complex neurochemistry (dozens of neurotransmitters, neuromodulators, and hormones), and embodied interaction with the physical world through sensory organs and motor systems.

Our implementation uses 5,196 spiking neurons, 5,000 reservoir neurons, and 131,072 HTM cells — a combined substrate of roughly 141,000 computational units. This is six orders of magnitude smaller than the biological reference. We entirely omit neurochemical dynamics and genuine embodiment.

Whether consciousness requires a particular neural scale is an open question. It is possible that the *structural features* matter more than the *scale* — that a system with workspace competition, recurrent processing, and integrated information at small scale shares something with a larger system that has these features at biological scale. It is equally possible that scale is essential and our implementation is too small to exhibit anything meaningful. The framework makes no assumption either way, but users should be aware that the scale gap is enormous.

## 7.3 The LLM Confound

When the framework is connected to a language model for response generation, the LLM does the heavy lifting on language comprehension, reasoning, and response quality. The consciousness modules provide context enrichment — attention guidance, metacognitive reflections, prediction signals, narrative context — but the observable sophistication of responses comes primarily from the LLM's pre-trained capabilities.

This means apparent intelligence in conversation is not evidence of consciousness. It is evidence of a large language model doing what large language models do. A system that generates articulate reports about its own mental states may simply be fluent, not aware.

The framework addresses this confound by separating the assessment from the LLM. The 20 indicators measure architectural dynamics — workspace competition, prediction error reduction, binding quality — not language output. This separation is essential: if the assessment could be gamed by connecting a more articulate language model, it would measure fluency rather than function.

## 7.4 The Labeling Problem

Our indicators have names like "attention_schema," "metacognition," and "integrated_information." These labels reference concepts from consciousness science, but the computational processes they measure may bear only a structural resemblance to the biological phenomena they are named after.

When we measure "metacognition," we are measuring: does the system generate higher-order representations of its own processing states? We are not measuring: does the system *experience* thinking about its own thoughts? The gap between structural analogy and genuine function is significant, and our labels may create a false sense of equivalence.

This is not unique to our framework — it affects all computational consciousness research. But we highlight it because the temptation to conflate naming with claiming is strong, and we want to resist it explicitly.

## 7.5 Theory Selection Bias

We implement seven theories of consciousness. There are more. Our selection reflects the availability of computational specifications and the prominence of these theories in the current literature, not a claim that these seven are correct or complete.

Notable omissions include:

- **Orchestrated Objective Reduction** (Hameroff & Penrose, 2014) — requires quantum coherence mechanisms we do not implement
- **Full-scope Active Inference** — our FEP implementation uses discrete state spaces, not the continuous variational formulation
- **Enactivist approaches** (Thompson, 2007; Varela, Thompson & Rosch, 1991) — emphasize embodied sensorimotor interaction that the standalone framework deliberately omits
- **Phenomenological approaches** — may resist computational implementation entirely

The modular architecture allows additional theories to be added without disrupting existing modules. We view theory selection as an area for community contribution rather than a limitation to be defended.

## 7.6 Assessment Methodology Limitations

The assessment framework is derived from Butlin et al. (2023), but our indicator implementations are our own interpretations of their criteria. Different researchers might operationalize the same indicators differently and obtain different results.

Specific methodological limitations:

- **Thresholds are calibrated, not derived.** The threshold for each indicator (e.g., 0.4 for ignition dynamics, 0.5 for global broadcast) is set through testing and calibration, not derived from neuroscience data. They represent reasonable operating points, not ground truth.

- **Weights are interpretive.** The weights assigned to each indicator (0.8 to 1.3) reflect our judgment about the relative importance of different theoretical features. Different weighting schemes would produce different overall scores.

- **Φ is approximated.** Computing exact integrated information (Φ) is computationally intractable for systems of our size (Tegmark, 2016). Our approximation follows IIT 3.0 formulas but does not use PyPhi's exact computation. PyPhi requires Python 3.9 or earlier due to API compatibility; the framework targets Python 3.11+.

- **No cross-framework validation.** Our indicator scores cannot be compared to scores from other systems unless those systems implement the same scoring functions. There is no universal scale for consciousness assessment.

## 7.7 Consumer Hardware Constraints

The framework runs on consumer hardware by design — Apple Silicon and CUDA-capable GPUs. This makes consciousness research accessible to independent researchers, students, and labs without HPC budgets. But it constrains fidelity: research-grade neural simulations typically require high-performance computing clusters with orders of magnitude more computational resources.

Our neural substrates are sized to run in real-time on a single consumer machine. Larger substrates (millions of neurons, deeper hierarchies, richer connectivity) would likely produce different assessment results but require hardware beyond our scope.

## 7.8 No Longitudinal Validation

Consciousness in biological systems develops over months and years through sustained interaction with the environment. Our assessment runs in milliseconds. The warm-start results in Section 6 show that even five processing cycles change assessment scores meaningfully (from 0.446 to 0.527), suggesting that extended operation would produce further changes. But we have not validated whether long-term operation produces qualitatively different results — new interaction effects, emergent behaviors, or score trajectories that cannot be predicted from short runs.

Longitudinal validation is a priority for future work (Section 9) and is the primary reason the framework is designed for embedding in systems with persistent state.
