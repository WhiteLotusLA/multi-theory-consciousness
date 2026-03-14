# 1. Introduction

Computational implementations of consciousness theories have proliferated in recent years, driven by advances in neural simulation, active inference modeling, and integrated information analysis. Yet a persistent limitation characterizes this body of work: each implementation targets a single theory in isolation. PyPhi (Mayner et al., 2018) implements Integrated Information Theory. Active inference agents (Fountas et al., 2020; Tschantz et al., 2020) implement the Free Energy Principle. Global workspace simulations (Shanahan, 2008; VanRullen & Kanai, 2021) implement Global Workspace Theory. Each provides valuable insight into its target theory, but none addresses a fundamental question: what happens when multiple theories of consciousness operate simultaneously within the same system?

This question matters because the major theories are not necessarily competitors. Global Workspace Theory (Baars, 1988) describes a mechanism for information broadcasting. Attention Schema Theory (Graziano, 2013) describes a self-model of attention. Higher-Order Thought Theory (Rosenthal, 2005) describes meta-representational structures. The Free Energy Principle (Friston, 2010) describes prediction error minimization. Integrated Information Theory (Tononi, 2008) describes information integration requirements. Recurrent Processing Theory (Lamme, 2006) describes recurrent neural dynamics. Beautiful Loop Theory (Laukkonen, Friston & Chandaria, 2025) describes recursive precision weighting over hierarchical predictions. These theories operate at different levels of description and may be complementary rather than mutually exclusive (Doerig, Schurger & Herzog, 2020; Seth & Bayne, 2022).

Yet we have no shared computational testbed for investigating their interactions. A researcher interested in whether prediction error signals from FEP-based processing affect competition dynamics in a global workspace has no existing platform on which to run that experiment. A theorist who suspects that recurrent processing depth (RPT) might correlate with integrated information (IIT) has no framework that measures both simultaneously.

The value of such a testbed lies precisely in what it isolates. A framework that runs consciousness theories in a controlled environment — without the confounding complexity of embodied interaction, developmental history, or language generation — can reveal cross-theory dynamics that would be invisible in a fully integrated system. The deliberate omissions are not limitations to be apologized for; they are experimental controls that allow clean measurement of the interactions between theories themselves. The framework is designed to be *embedded* in larger systems that provide sensory grounding, persistent memory, and real-world interaction. The science is in the engine; the life, if/when it comes, is in what drives it.

We present the Multi-Theory Consciousness (MTC) framework: an open-source Python implementation of seven consciousness theories operating as interacting modules within a single architecture. The framework includes three neural substrates — a spiking neural network (SNN; 4,116 neurons), a liquid state machine (LSM; 5,000 neurons), and a hierarchical temporal memory (HTM; 131,072 cells) — and a 20-indicator assessment framework derived from the methodology of Butlin et al. (2023). Additionally, Damasio's (1999, 2010) three-layer model of protoself, core consciousness, and extended consciousness serves as an integrative layer connecting embodied homeostatic signals to higher-order cognitive processes.

### 1.1 What This Paper Does Not Claim

We lead with this because it is the most important paragraph in the paper.

We do not claim that the MTC framework is conscious. We do not claim that high scores on our assessment indicators constitute evidence of phenomenal experience. We do not claim that the architectural features we implement are sufficient for consciousness, even in principle. We do not claim that our implementations are faithful to every nuance of the theories they represent — computational implementation forces interpretive choices, and we document ours explicitly.

What we do claim is narrower: we have built a platform on which consciousness theories can be implemented, run simultaneously, and measured through a standardized assessment. The 20 indicators measure whether the architectural features that each theory describes as necessary for consciousness are present and functioning. When an indicator reports "pass," it means the relevant computational mechanism operates within its designed parameters. It does not mean the system is conscious.

We use the phrase "architectural function" throughout this paper to maintain this distinction. An architecture can be functional without being conscious, just as a building can be structurally sound without being a home.

### 1.2 Contributions

This paper makes the following contributions:

1. **A multi-theory integration architecture.** We describe a modular design in which seven consciousness theories operate simultaneously, with defined interfaces between them. Each theory is implemented as a Python module that receives input from and provides output to a shared processing cycle.

2. **Three neural substrates.** The framework operates on a spiking neural network (4,116 neurons, snntorch), a liquid state machine (5,000 neurons, reservoirpy), and a hierarchical temporal memory (131,072 cells, custom implementation). These provide biologically inspired processing layers at different temporal scales.

3. **A 20-indicator assessment framework.** Derived from Butlin et al. (2023), our assessment measures architectural function across all seven theories through reproducible scoring functions. We include a complementary 13-perspective Digital Consciousness Model (DCM) scoring system for probabilistic credence assessment.

4. **An ablation study methodology.** We provide tools to selectively disable individual theory modules and measure the impact on remaining indicators, revealing cross-theory dependencies that are invisible in single-theory implementations.

5. **Design for embedding.** The framework is architected to be integrated into larger systems — conversational agents, robotic platforms, or interactive simulations — that provide the sensory input, persistent memory, and developmental trajectory the standalone framework deliberately omits. Each omission corresponds to a defined interface through which a host system can supply grounding.

6. **Open-source release.** The complete framework — approximately 25,000 lines of production code, 400 tests, and documentation — is released under the Apache 2.0 license to support collaborative research. It runs on consumer hardware (Apple Silicon and CUDA-capable GPUs).

### 1.3 Paper Structure

Section 2 reviews the seven consciousness theories and prior computational implementations. Section 3 describes the system architecture and module design. Section 4 details the implementation of each theory module, including what we simplify and why. Section 5 presents the 20-indicator assessment framework and ablation methodology. Section 6 reports assessment results and interaction effects. Section 7 discusses limitations — thoroughly, because honest accounting of what a system cannot do is as important as describing what it can. Section 8 considers what multi-theory integration reveals. Section 9 outlines future work, and Section 10 concludes.

Appendix C provides a complete file index mapping class names used in this paper to their source files and line counts.

The most important file in our repository is `HONEST_LIMITATIONS.md`. We recommend reading it before the code.
