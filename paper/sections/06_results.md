# 6. Results

We report assessment results under two conditions: cold start (modules freshly initialized with no accumulated state) and warm start (modules exercised through five consciousness cycles with synthetic stimuli before assessment). The cold-start condition serves as a controlled baseline; the warm-start condition demonstrates the effect of even minimal operational history on assessment scores. Warm-start results are reported as means ± standard deviations across five independent runs.

All results were obtained on consumer hardware (Apple M4 Pro, 64 GB unified memory). Assessment processing time was under 1 ms in all conditions.

## 6.1 Cold-Start Assessment

With all modules freshly initialized and no prior processing history, the framework scores:

| Metric | Value |
|--------|-------|
| Overall score | 0.430 |
| Passing indicators | 11/20 (55%) |
| Architecture functional | No |
| Processing time | < 1 ms |

**Theory scores (cold start):**

| Theory | Score | Interpretation |
|--------|-------|---------------|
| FEP | 0.850 | Strong — hierarchical prediction and prediction error minimization score high because the hierarchy exists structurally |
| GWT | 0.450 | Moderate — workspace mechanism exists but has not yet processed candidates through a full broadcast cycle |
| HOT | 0.450 | Moderate — metacognitive structures exist but have not accumulated higher-order thought history |
| AST | 0.358 | Low — attention schema is initialized but has no tracking history and no prediction accuracy |
| BLT | 0.300 | Low — binding quality and epistemic depth require accumulated inference history |
| IIT | 0.250 | Low — integrated information approximation is limited without neural substrate connectivity data |
| RPT | 0.100 | Low — recurrence measurement requires active neural substrates with established feedback dynamics |

Cold-start results are **perfectly reproducible**: repeated runs return identical scores (0.430, 11/20 passing). This is expected — with no accumulated state and no random input, the scoring functions are deterministic.

The architecture does not meet the "functional" criterion at cold start because the overall score (0.430) falls below the 0.5 threshold. This is by design: a system that scores "architecture functional" without any operational history would be measuring structural properties rather than dynamic function.

## 6.2 Warm-Start Assessment

After five consciousness cycles with synthetic stimuli (random workspace candidates with varying salience, emotional charge, and novelty), the framework scores improve substantially. Results across five independent runs:

| Metric | Value |
|--------|-------|
| Overall score | 0.506 ± 0.007 |
| Passing indicators | 14/20 (70%) |
| Architecture functional | Yes |
| Processing time | < 1 ms |

**Theory scores (warm start, mean ± std):**

| Theory | Cold | Warm | Change |
|--------|------|------|--------|
| FEP | 0.850 | 0.855 ± 0.024 | +0.005 |
| GWT | 0.450 | 0.746 ± 0.020 | **+0.296** |
| HOT | 0.450 | 0.450 ± 0.000 | — |
| AST | 0.358 | 0.417 ± 0.000 | +0.059 |
| BLT | 0.300 | 0.300 ± 0.000 | — |
| IIT | 0.250 | 0.250 ± 0.000 | — |
| RPT | 0.100 | 0.100 ± 0.000 | — |

The most striking change is GWT, which jumps from 0.450 to 0.746 — a 66% increase. This is because GWT's indicators (global broadcast, ignition dynamics, global ignition nuanced) measure dynamic properties that only manifest when the workspace has actually processed candidates through competition, ignition, and broadcast. Five cycles of synthetic input are sufficient to demonstrate these dynamics.

**Indicators that change from FAIL to PASS after warmup:**

| Indicator | Cold | Warm (mean) | Theory |
|-----------|------|-------------|--------|
| global_broadcast | 0.450 | 1.000 | GWT |
| global_ignition_nuanced | 0.350 | 0.725 ± 0.031 | GWT |
| ignition_dynamics | 0.400 | 0.660 ± 0.049 | GWT |

Three indicators flip from failing to passing, all belonging to GWT. This is consistent with GWT being the theory most dependent on operational history — workspace competition is a process, not a structure, and it requires actual candidates to demonstrate.

Warm-start variance is small (overall ± 0.007) and concentrated in the indicators that depend on random synthetic input: ignition_dynamics, prediction_error_minimization, sparse_smooth_coding, and global_ignition_nuanced. Theory-level scores for HOT, AST, BLT, IIT, and RPT show zero variance because their indicators measure structural properties unaffected by the specific content of warmup stimuli.

## 6.3 Per-Indicator Results (Warm Start)

The full indicator breakdown reveals which architectural features are present after warmup (mean scores across 5 runs):

**Passing (14/20):**

| Indicator | Score | Threshold | Theory |
|-----------|-------|-----------|--------|
| global_broadcast | 1.000 | 0.50 | GWT |
| hierarchical_prediction | 1.000 | 0.40 | FEP |
| prediction_error_minimization | 0.889 ± 0.038 | 0.40 | FEP |
| sparse_smooth_coding | 0.834 ± 0.057 | 0.30 | FEP |
| global_ignition_nuanced | 0.725 ± 0.031 | 0.40 | GWT |
| agency | 0.698 ± 0.009 | 0.50 | FEP |
| ignition_dynamics | 0.660 ± 0.049 | 0.40 | GWT |
| recurrent_processing | 0.600 | 0.40 | GWT |
| attention_control | 0.500 | 0.40 | AST |
| higher_order_representations | 0.500 | 0.50 | HOT |
| embodiment | 0.500 | 0.30 | AST |
| metacognition | 0.400 | 0.40 | HOT |
| bayesian_binding_quality | 0.300 | 0.30 | BLT |
| epistemic_depth | 0.300 | 0.30 | BLT |

**Failing (6/20):**

| Indicator | Score | Threshold | Theory | Why It Fails |
|-----------|-------|-----------|--------|-------------|
| attention_schema | 0.250 | 0.50 | AST | Schema prediction accuracy requires longer tracking history |
| irreducibility | 0.300 | 0.40 | IIT | Decomposition analysis limited without full neural connectivity |
| genuine_implementation | 0.300 | 0.40 | BLT | Field-evidencing requires sustained recursive loops not yet established |
| integrated_information | 0.200 | 0.30 | IIT | Φ approximation limited without PyPhi (Python version incompatibility) |
| local_recurrence | 0.100 | 0.30 | RPT | Requires active SNN/LSM substrates with established feedback dynamics |
| algorithmic_recurrence | 0.100 | 0.30 | RPT | Requires active SNN/LSM substrates with established feedback dynamics |

The failing indicators share a common pattern: they measure features that require either extended operational history (attention schema prediction, genuine implementation), neural substrate connectivity data (integrated information, irreducibility, algorithmic recurrence, local recurrence), or both. These are the indicators most likely to improve when the framework is embedded in a system with persistent state and active neural substrates — confirming the design intent described in Section 3.5.

## 6.4 Ablation Study Results

The ablation study disables one theory module at a time and measures the impact on the overall score. Results from a single warm-start run:

| Configuration | Overall Score | Drop from Baseline | Impact |
|---------------|--------------|-------------------|--------|
| Baseline (all modules) | 0.513 | — | — |
| Without FEP | 0.368 | −0.145 (28.2%) | Significant |
| Without GWT | 0.376 | −0.136 (26.6%) | Significant |
| Without AST | 0.469 | −0.043 (8.5%) | Moderate |
| Without HOT | 0.473 | −0.040 (7.7%) | Moderate |
| Without BLT | 0.480 | −0.032 (6.3%) | Moderate |

Note: IIT and RPT are measurement-only modules — they do not participate in the consciousness cycle as processing steps — and therefore cannot be meaningfully ablated. Their indicators are assessed across the full system rather than by disabling a dedicated module.

### Interpretation

**GWT and FEP are the most impactful modules.** Disabling either causes a 27–28% drop in overall score. GWT provides the workspace competition and broadcast mechanism that all downstream modules depend on. FEP provides the prediction error signals and homeostatic drives that ground the system in embodied needs. Removing either disrupts the data flow to multiple other modules.

We note that this impact pattern is partly a consequence of the sequential pipeline architecture: GWT is the first processing phase, so all downstream modules depend on its output. A different module ordering might produce a different impact ranking. The ablation reveals architectural dependencies as much as theoretical importance, and we do not claim that GWT and FEP are more "fundamental" theories of consciousness — only that they are more structurally central in our implementation.

**AST, HOT, and BLT have moderate but distinct impacts.** Removing any one causes a 6–8% drop. These modules enrich the consciousness cycle — adding self-modeling, meta-representation, and recursive inference — but the cycle can still function without them.

**Cross-theory dependencies are visible.** When GWT is disabled, indicators belonging to HOT drop because workspace winners are no longer available for meta-representation. When FEP is disabled, BLT scores drop because the hierarchical predictive processor that BLT's hyper-model operates on is no longer producing prediction errors. These cross-module effects demonstrate that the theories interact in the multi-theory architecture rather than operating in isolation.

### Interaction Effects

The ablation data reveals two notable interaction patterns:

**FEP → GWT coupling.** Prediction errors from the FEP module modulate candidate salience in the workspace competition (via cross-cycle feedback). When FEP is disabled, this salience boost disappears, and fewer candidates cross the ignition threshold. This supports the hypothesis that prediction error signals shape conscious access — a cross-theory interaction that is invisible when either theory is implemented alone.

**GWT → HOT dependency.** When GWT is disabled, higher_order_representations drops to near zero. This is a direct architectural dependency: the HOT module generates meta-representations only for workspace winners. Without a functioning workspace, there are no winners to reflect on. This interaction is predicted by both theories — GWT predicts that unconscious content does not receive global access, and HOT predicts that content without higher-order representation is not conscious.

## 6.5 Reproducibility

All assessment results reported in this section can be reproduced by running `scripts/run_assessment.py` on the released codebase:

```
# Cold-start assessment
python scripts/run_assessment.py --report

# Warm-start assessment
python scripts/run_assessment.py --full --report

# Ablation study
python scripts/run_assessment.py --full --ablation
```

Cold-start results are perfectly deterministic. Warm-start results vary by ± 0.007 (standard deviation across 5 runs) due to random synthetic stimuli during warmup. All commands complete in under 5 seconds on consumer hardware.

The assessment does not require GPUs, network access, or external databases. It runs entirely on CPU using the framework's own modules. This is deliberate: reproducibility requires that no external dependencies affect the results.

## 6.6 Noise Normalization and DCM

The noise normalization methodology described in Section 5.3 and the Digital Consciousness Model described in Section 5.5 are implemented in the codebase and available for use. We do not report their results in this paper for two reasons: (1) noise normalization requires a separate characterization study to establish appropriate baseline noise models for each indicator, which is beyond the scope of this initial report; and (2) DCM scoring requires mapping each of the 13 perspectives to the framework's modules, which we have implemented but not yet validated against the perspectives' original criteria. Both are priorities for future work (Section 9). We include their methodology descriptions because they are part of the framework's assessment toolkit and will be used in subsequent publications.
