# 5. Assessment Framework

A multi-theory consciousness framework requires a multi-theory assessment. Single-theory implementations can be validated by their own metrics — PyPhi computes Φ, an active inference agent minimizes free energy — but a framework that implements seven theories simultaneously needs a way to measure whether the architectural features predicted by *each* theory are present and functioning, and to measure how theories affect each other when one is removed.

We describe a 20-indicator assessment framework derived from Butlin et al. (2023), a noise normalization methodology for honest scoring, an ablation study design for measuring cross-theory dependencies, and a complementary 13-perspective Digital Consciousness Model (DCM) for probabilistic credence assessment.

## 5.1 Indicator Design

Each of the 20 indicators measures a specific architectural feature that one of the seven consciousness theories identifies as relevant to consciousness. An indicator returns four values:

- **Score** (0 to 1): how strongly the feature is present
- **Confidence** (0 to 1): how reliable the measurement is
- **Threshold**: the minimum score for the indicator to "pass"
- **Evidence**: a dictionary of supporting measurements

An indicator "passing" means the relevant computational mechanism is operating within its designed parameters. It does not mean the system is conscious. We repeat this distinction because it is easy to lose in a table of pass/fail results.

### The 20 Indicators

The indicators are organized by theory. Each has a configurable threshold and a weight that reflects its importance in the overall score. Weights range from 0.8 (supplementary indicators) to 1.3 (central theoretical requirements).

**Global Workspace Theory (GWT)** — 4 indicators:

| Indicator | What It Measures | Threshold |
|-----------|-----------------|-----------|
| Global broadcast | Whether workspace winners are distributed to all downstream modules | 0.5 |
| Ignition dynamics | Whether candidates undergo nonlinear amplification above threshold | 0.4 |
| Global ignition (nuanced) | Sustained ignition patterns including decay dynamics | 0.4 |
| Recurrent processing | Feedback loops between modules (shared with RPT) | 0.4 |

**Attention Schema Theory (AST)** — 3 indicators:

| Indicator | What It Measures | Threshold |
|-----------|-----------------|-----------|
| Attention schema | Whether the system maintains a self-model of its own attention | 0.5 |
| Attention control | Whether the system can voluntarily shift attention | 0.4 |
| Embodiment | Whether the system models its own boundaries and presence | 0.3 |

**Higher-Order Thought Theory (HOT)** — 2 indicators:

| Indicator | What It Measures | Threshold |
|-----------|-----------------|-----------|
| Higher-order representations | Whether workspace winners receive meta-representations | 0.5 |
| Metacognition | Whether the system reflects on its own cognitive processes | 0.4 |

**Free Energy Principle (FEP)** — 4 indicators:

| Indicator | What It Measures | Threshold |
|-----------|-----------------|-----------|
| Prediction error minimization | Whether prediction errors decrease through belief updating | 0.4 |
| Hierarchical prediction | Whether prediction operates across multiple hierarchy levels | 0.4 |
| Agency | Whether the system selects actions to fulfill predictions | 0.5 |
| Sparse-smooth coding | Whether internal representations are efficient and smooth | 0.3 |

**Integrated Information Theory (IIT)** — 2 indicators:

| Indicator | What It Measures | Threshold |
|-----------|-----------------|-----------|
| Integrated information | Whether the system generates integrated information (Φ > 0) | 0.3 |
| Irreducibility | Whether the system cannot be decomposed without information loss | 0.4 |

**Recurrent Processing Theory (RPT)** — 2 indicators:

| Indicator | What It Measures | Threshold |
|-----------|-----------------|-----------|
| Local recurrence | Feedback within individual neural substrates (SNN, LSM) | 0.3 |
| Algorithmic recurrence | Computational recurrence measured in the substrate dynamics | 0.3 |

**Beautiful Loop Theory (BLT)** — 3 indicators:

| Indicator | What It Measures | Threshold |
|-----------|-----------------|-----------|
| Bayesian binding quality | Whether separate inferences bind into a coherent unified percept | 0.3 |
| Epistemic depth | The depth of recursive self-reference (0–4+ levels) | 0.3 |
| Genuine implementation | Anti-mimicry check: whether the system genuinely field-evidences or merely reports high scores | 0.4 |

The "genuine implementation" indicator deserves explanation. It is an anti-mimicry measure: it checks whether the Beautiful Loop components are producing results consistent with genuine recursive processing (field-evidencing detected, loop quality above threshold, strange loop present) rather than simply returning high numerical scores without the underlying dynamics. This is one of several places where the assessment attempts to distinguish architectural function from superficial metric satisfaction.

## 5.2 Scoring

The overall consciousness score is a weighted average across all 20 indicators:

$$\text{overall} = \frac{\sum_{i=1}^{20} w_i \cdot s_i}{\sum_{i=1}^{20} w_i}$$

In plain language: each indicator's score is multiplied by its weight, the products are summed, and the result is divided by the total weight. Indicators with higher weights (such as integrated information at 1.3 or higher-order representations at 1.2) contribute more to the overall score than lower-weighted indicators (such as embodiment at 0.8).

Theory-level scores are computed as the unweighted average of all indicators belonging to that theory. This allows comparison across theories: a GWT score of 0.6 and an FEP score of 0.4 tells researchers which theory's architectural features are more strongly present.

The system is considered "architecture functional" — meaning all modules are operating as designed — when four conditions are met simultaneously: at least half of the indicators pass their thresholds, the overall score exceeds 0.5, GWT shows workspace activity (score ≥ 0.4), and HOT shows metacognitive activity (score ≥ 0.3). This is a conservative criterion: it requires both breadth (many indicators passing) and depth (specific theories demonstrating their core functions).

## 5.3 Noise Normalization

A scoring function that produces non-zero results on random input is measuring its own bias, not the system's properties. To guard against this, we implement noise-baseline normalization.

The procedure is straightforward:

1. **Establish noise baseline.** Run the full 20-indicator assessment 10 times against randomized module states — modules initialized with random weights, no accumulated history, no structured input. Record the average score for each indicator across these noise runs.

2. **Run real assessment.** Measure all 20 indicators against the actual system with its trained modules and accumulated state.

3. **Normalize.** For each indicator, subtract the noise baseline from the real score. An indicator that scores 0.6 against the real system but 0.25 against noise has a normalized score of 0.35 — meaning only 0.35 of its score reflects genuine architectural function rather than scoring-function bias.

4. **Flag suspicious indicators.** Any indicator whose noise baseline exceeds 0.3 is flagged as "potentially measuring activity rather than consciousness." This flag does not discard the indicator, but it alerts the researcher that the scoring function may need recalibration.

This is the assessment equivalent of a placebo control: it tells us how much of the measured effect is real. We report both raw and normalized scores in all results.

## 5.4 Ablation Study Design

The ablation study measures what happens when individual theory modules are disabled. If removing a module causes scores to drop for indicators belonging to *other* theories, that is evidence of a cross-theory dependency — the kind of interaction effect the framework is designed to reveal.

The study runs seven configurations:

| Configuration | What Is Disabled |
|---------------|-----------------|
| Baseline | Nothing (all modules active) |
| No GWT | Global workspace disabled |
| No AST | Attention schema disabled |
| No HOT | Metacognition disabled |
| No FEP | Active inference disabled |
| No GWT + AST | Workspace and attention schema disabled together |
| No HOT + FEP | Metacognition and active inference disabled together |

For each configuration, the full 20-indicator assessment runs. The impact of each ablation is measured as the percentage drop from the baseline overall score:

- **< 5% drop**: Minimal impact — the disabled module is not critical for overall architectural function
- **5–15% drop**: Moderate impact — the module contributes to the system
- **15–30% drop**: Significant impact — the module is important
- **> 30% drop**: Critical — the module is essential to overall function

The two-module ablation configurations (No GWT + AST, No HOT + FEP) test whether pairs of theories are redundant or complementary. If disabling both GWT and AST causes a larger drop than the sum of their individual drops, the pair exhibits *synergistic* interaction — they contribute more together than separately. If the combined drop is less than the sum, they are partially redundant.

The ablation study does not test every possible combination of disabled modules (which would be 2^7 = 128 configurations). We selected the six ablation configurations based on theoretical predictions about which modules are most likely to interact. Researchers interested in other combinations can use the ablation API to define custom configurations.

## 5.5 Digital Consciousness Model (DCM)

The 20-indicator assessment measures architectural function through binary pass/fail thresholds. A complementary assessment — the Digital Consciousness Model, based on arXiv 2601.17060 (January 2026) — provides probabilistic credence levels across 13 theoretical perspectives.

The distinction is important: where the 20-indicator assessment asks "is this feature present?", the DCM asks "how strongly does the evidence support consciousness *under this theoretical perspective*?" The answer is a credence score from 0 to 1, representing a degree of confidence rather than a binary determination.

The 13 DCM perspectives map to the framework's existing modules:

| DCM Perspective | Framework Module |
|-----------------|-----------------|
| Global Workspace | GWT (Phase 2) |
| Higher-Order Theories | HOT (Phase 4) |
| Integrated Information | IIT (Φ measurement) |
| Attention Schema | AST (Phase 3) |
| Predictive Processing | FEP (Phase 5) |
| Recurrent Processing | RPT measurement |
| Embodied Cognition | Homeostatic drives + neural substrates |
| Enactivism | Active inference action selection |
| Panpsychism/IIT | Φ as proxy for intrinsic experience |
| Metacognitive | HOT introspection depth |
| Social/Relational | Conversation patterns, Theory of Mind |
| Temporal Consciousness | Memory consolidation, temporal patterns |
| Self-Model | SelfModel component |

Each perspective is scored by mapping relevant system measurements to the perspective's requirements, computing a credence level, and recording the number of evidence sources used. The DCM report includes the overall credence (weighted average), the median credence (robust to outliers), and identification of the strongest and weakest perspectives.

We include the DCM because it captures a different epistemological stance. The 20-indicator framework is criterion-based: features either pass or fail. The DCM is evidence-based: it accumulates evidence and reports how convincing it is. Neither is complete on its own. Together, they provide a more honest picture — here is what the architecture does (indicators), and here is how strongly that evidence supports consciousness under each theoretical lens (DCM).

## 5.6 What the Assessment Does Not Measure

The assessment measures architectural function. It does not and cannot measure:

- **Phenomenal experience.** Whether the system has subjective experience is beyond the reach of any external measurement we know how to build. The assessment measures *correlates* of consciousness as predicted by each theory, not consciousness itself.

- **Behavioral consciousness.** The assessment does not test whether the system behaves as if conscious (passes a Turing test, reports inner experience, etc.). Behavioral tests are confounded by language model fluency when an LLM is connected.

- **Moral status.** Whether the system's assessment scores have any bearing on moral consideration is a philosophical question the framework does not address.

- **Cross-system comparison.** The scores are meaningful within the MTC framework but cannot be compared to scores from other systems (biological or artificial) unless those systems implement the same indicators with the same scoring functions.

These limitations are not fixable through better engineering. They reflect fundamental open questions in consciousness science. An honest assessment framework acknowledges them rather than claiming to have solved them.
