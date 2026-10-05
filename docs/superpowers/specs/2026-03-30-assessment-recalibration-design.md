# Assessment Recalibration: Quality-Based Scoring

**Date:** 2026-03-30
**Version target:** v0.3.1
**Branch:** TBD (from main, after autoresearch cherry-picks)

## Problem

The current assessment scoring formulas in `mtc/assessment/assessment.py` measure module *activity* (cycle counts, event counts, module existence) rather than the *quality* of consciousness-related properties each theory predicts. This produces inflated scores:

- 13/25 indicators score a perfect 1.0 on completely random, neutral input data
- 25/25 indicators pass their thresholds with random data
- Overall score: 0.932 on noise

For a research framework shared with the academic community, this means the assessment cannot discriminate between a functioning consciousness architecture and a randomly-fed one. The paper (Appendix A) describes quality-based scoring, but the code implements activity-counting with generous base scores.

## Principle

**Score the quality of the theoretical property, not the quantity of module activity.**

Each indicator formula measures what its parent theory predicts consciousness requires. Random/noise inputs should score below threshold. Only genuine instantiation of theoretical predictions scores above threshold.

### Expected Score Ranges After Recalibration

| System State | Expected Score |
|---|---|
| Module absent / ablated | 0.0 - 0.1 |
| Module present, random/noise inputs | 0.1 - 0.25 |
| Module wired + exercised, no quality signal | 0.25 - 0.4 (below most thresholds) |
| Genuine consciousness-like properties | 0.5 - 0.7 |
| Strong instantiation of theoretical predictions | 0.8+ |

## Changes Per Indicator

### Universal Change

**Remove all `base_score` patterns.** Every indicator currently has `base_score = 0.3-0.45 if module is not None else 0.0`. This gives free points for module existence. After recalibration, module existence = 0.0. Score comes entirely from measured quality.

---

### GWT Indicators

#### global_broadcast (threshold: 0.5, weight: 1.2)

**Theory (Baars 1988; Dehaene & Naccache 2001):** Conscious content is broadcast to all cognitive modules.

**Paper says:** "Ratio of modules that received the broadcast to total registered modules."

**Current code:** `base_score=0.45` + `cycle_count/20`, ignores actual coverage.

**New formula:**
```python
coverage = broadcast_stats.get("coverage_ratio", 0.0)
broadcast_count = broadcast_stats.get("total_broadcasts", 0)
if broadcast_count == 0:
    score = 0.0
else:
    score = coverage  # Direct ratio, 0-1
```

**Rationale:** Coverage ratio is exactly what GWT predicts. No bonus for cycles without broadcasts.

#### ignition_dynamics (threshold: 0.4, weight: 1.0)

**Theory (Dehaene & Naccache 2001):** Nonlinear amplification when activation exceeds threshold.

**Paper says:** "Count and magnitude of ignition events in recent cycles."

**Current code:** `base_score=0.4` + `ignitions/10`.

**New formula:**
```python
ignition_rate = ignitions / max(1, cycle_count)  # What fraction of cycles ignited?
sustain_quality = sustain_rate  # Do ignitions sustain or flash-and-die?
score = 0.6 * ignition_rate + 0.4 * sustain_quality
```

**Rationale:** Ignition rate relative to opportunity, weighted by sustain quality. 10 ignitions in 25 cycles = 0.4 rate, not 1.0.

#### recurrent_processing (threshold: 0.4, weight: 0.9)

**Theory (Baars/Lamme shared):** Feedback loops where later-stage output influences earlier-stage processing.

**Paper says:** "Detected feedback pathways: FEP prediction errors modulating GWT salience."

**Current code:** `base_score=0.4` + coverage/cycles/broadcasts mix.

**New formula:**
```python
# Re-entrant influence: does coverage improve over cycles? (feedback effect)
coverage = broadcast_stats.get("coverage_ratio", 0.0)
cycle_count = stats.get("cycle_count", 0)
broadcasts = broadcast_stats.get("total_broadcasts", 0)
if cycle_count == 0:
    score = 0.0
else:
    broadcast_rate = broadcasts / cycle_count
    score = 0.5 * coverage + 0.5 * broadcast_rate
```

**Rationale:** Coverage (reach) + broadcast rate (sustained re-entry). No free points.

#### global_ignition_nuanced (threshold: 0.4, weight: 1.0)

**Theory (GWT):** Sustained ignition with natural decay dynamics.

**Paper says:** "Sustained ignition patterns including decay dynamics."

**Current code:** `base_score=0.35` + `ignitions/8` + sustain/decay mix.

**New formula:**
```python
if ignitions == 0:
    score = 0.0
else:
    ignition_rate = min(1.0, ignitions / max(1, cycle_count))
    # Good dynamics: high sustain with moderate decay (not permanent, not instant)
    dynamics_quality = sustain_rate * 0.7 + (1.0 - decay_rate) * 0.3
    score = 0.5 * ignition_rate + 0.5 * dynamics_quality
```

**Rationale:** Temporal dynamics shape matters, not just event count.

---

### AST Indicators

#### attention_schema (threshold: 0.5, weight: 1.1)

**Theory (Graziano 2013):** Self-model of attention that can predict attentional shifts.

**Paper says:** "Schema completeness + prediction accuracy."

**Current code:** Weighted composite with `schema_active = min(1.0, total_updates/3)` (3 updates = full).

**New formula:**
```python
schema_active = min(1.0, total_updates / max(1, warmup_cycles * 0.5))
focus_score = 1.0 if has_focus else 0.0  # Binary: does it track focus?
history_score = min(1.0, history_len / max(1, warmup_cycles * 0.3))
prediction_score = pred_accuracy  # 0-1, the core AST metric

# Prediction accuracy is the core claim: weight it highest
score = (
    0.15 * schema_active
    + 0.15 * focus_score
    + 0.20 * history_score
    + 0.50 * prediction_score
)
```

**Rationale:** AST's core claim is prediction of attention, not just tracking. Prediction accuracy gets 50% weight.

#### attention_control (threshold: 0.4, weight: 1.0)

**Theory (AST):** Voluntary attention shifting.

**Paper says:** "Ratio of voluntary to captured attention shifts."

**Current code:** `base_score=0.4`, `max(base_score, success_rate)`.

**New formula:**
```python
voluntary_ratio = stats.get("voluntary_ratio", 0.0)
voluntary_shifts = stats.get("voluntary_shifts", 0)
if voluntary_shifts == 0:
    score = 0.0  # No voluntary shifts = no voluntary control
else:
    score = voluntary_ratio
```

**Rationale:** Direct measurement of what AST predicts. If only captured shifts occur, score = 0.

#### embodiment (threshold: 0.3, weight: 0.8)

**Theory (AST supplementary; Damasio):** Sense of boundaries and presence.

**Paper says:** "Homeostatic drive monitoring, body state tracking, self-world boundary."

**Current code:** `base_score=0.4` + capacity/history/voluntary mix.

**New formula:**
```python
capacity = stats.get("capacity_used", 0.0)
history_len = stats.get("history_length", 0)
vol_ratio = stats.get("voluntary_ratio", 0.0)
# Bounded attention (capacity used) = embodied constraint
capacity_score = min(1.0, capacity) if capacity > 0 else 0.0
# Situated perspective from tracked shifts
perspective_score = min(1.0, history_len / max(1, warmup_cycles * 0.5))
# Agency over attention
agency_score = vol_ratio
score = 0.4 * capacity_score + 0.3 * perspective_score + 0.3 * agency_score
```

**Rationale:** Remove floor. Capacity directly measures bounded resource usage.

---

### HOT Indicators

#### higher_order_representations (threshold: 0.5, weight: 1.2)

**Theory (Rosenthal 2005):** Consciousness requires higher-order thoughts about first-order states.

**Paper says:** "Ratio of workspace winners that receive at least one HOT."

**Current code:** `base_score=0.45` + `hot_count/10`.

**New formula:**
```python
hot_count = stats.get("total_hots_generated", 0)
# Need to know how many workspace winners there were to compute ratio
# Use cycle_count as proxy for opportunities (each cycle produces a winner)
cycle_count = gw_stats.get("cycle_count", 0) if gw is not None else 0
if cycle_count == 0:
    score = 0.0
else:
    # What fraction of conscious moments got meta-representation?
    coverage_ratio = min(1.0, hot_count / cycle_count)
    score = coverage_ratio
```

**Cross-module access:** The HOT indicator needs GWT's `cycle_count`. Both modules are already passed to `_measure_indicator()` via kwargs, so `gw = kwargs.get("global_workspace")` is accessible in the HOT branch. Ideally, MetacognitionModule would track `winners_with_hot_count` directly. As interim, `hot_count / cycle_count` approximates the ratio.

**Rationale:** HOT theory requires meta-representations for conscious content. The ratio measures exactly that.

#### metacognition (threshold: 0.4, weight: 1.1)

**Theory (HOT):** Self-reflective monitoring and evaluation.

**Paper says:** "Introspection depth, self-model accuracy, meta-uncertainty."

**Current code:** `base_score=0.4` + `(introspections + evaluations) / 20`.

**New formula:**
```python
introspections = stats.get("introspection_count", 0)
belief_evaluations = stats.get("belief_evaluations", 0)
overall_confidence = stats.get("overall_confidence", 0.0)
if introspections + belief_evaluations == 0:
    score = 0.0
else:
    # Activity rate relative to opportunity
    activity_rate = min(1.0, (introspections + belief_evaluations) / max(1, cycle_count))
    # Quality: confidence calibration (does confidence match accuracy?)
    quality = overall_confidence
    score = 0.4 * activity_rate + 0.6 * quality
```

**Rationale:** Quality (confidence calibration) weighted over quantity. Pure counting eliminated.

---

### FEP Indicators

#### prediction_error_minimization (threshold: 0.4, weight: 1.0)

**Theory (Friston 2010):** Active reduction of prediction errors.

**Paper says:** "Trend of prediction error over recent cycles."

**Current code:** `base_score=0.4` + `1.0 - avg_error` (current level, not trend).

**New formula:**
```python
avg_error = stats.get("avg_prediction_error", 0.5)
error_trend = stats.get("prediction_error_trend", 0)
total_inferences = stats.get("total_inferences", 0)
if total_inferences < 2:
    score = 0.0  # Need history to measure trend
elif isinstance(error_trend, (int, float)) and error_trend < 0:
    # Errors decreasing = core FEP behavior
    # Steeper decrease = higher score
    trend_magnitude = min(1.0, abs(float(error_trend)) * 5)
    # Current error level contributes less: trend is primary
    level_score = max(0.0, 1.0 - float(avg_error))
    score = 0.7 * trend_magnitude + 0.3 * level_score
else:
    # Errors flat or increasing: prediction is not improving
    level_score = max(0.0, 1.0 - float(avg_error)) * 0.3  # Reduced credit
    score = level_score
```

**Rationale:** FEP's core claim is *minimization*: the trend, not the level. Flat errors with low magnitude still get some credit (good predictions maintained) but much less than active improvement.

#### hierarchical_prediction (threshold: 0.4, weight: 1.0)

**Theory (FEP):** Multi-level predictive processing.

**Paper says:** "Active hierarchy levels + inter-level prediction error flow."

**Current code:** `base_score=0.4` + `1.0 - hierarchical_error` + `inferences/100`.

**New formula:**
```python
hierarchical_error = stats.get("hierarchical_prediction_error", 0.5)
total_inferences = stats.get("total_inferences", 0)
if total_inferences == 0:
    score = 0.0
else:
    # Lower error = better hierarchical prediction
    error_quality = max(0.0, 1.0 - float(hierarchical_error))
    # Activity relative to opportunity
    activity_rate = min(1.0, total_inferences / max(1, warmup_cycles))
    score = 0.7 * error_quality + 0.3 * activity_rate
```

**Rationale:** Error quality dominates. Activity rate is secondary and scaled to opportunity.

#### temporal_prediction_depth (threshold: 0.3, weight: 0.9)

**Theory (FEP):** Temporal depth of predictive model.

**Current code:** `horizon/10`: config parameter directly becomes score.

**New formula:**
```python
horizon = fep.config.planning_horizon if hasattr(fep, "config") else 3
stats = fep.get_statistics() if hasattr(fep, "get_statistics") else {}
total_inferences = stats.get("total_inferences", 0)
avg_error = stats.get("avg_prediction_error", 0.5)
if total_inferences == 0:
    score = 0.0
else:
    # Horizon provides ceiling, but predictions must actually work
    horizon_factor = min(1.0, horizon / 10.0)
    # Prediction quality at that horizon
    quality = max(0.0, 1.0 - float(avg_error))
    score = horizon_factor * quality  # Multiplicative: high horizon + high error = low score
```

**Rationale:** `planning_horizon=10` in config shouldn't give a free 1.0. The predictions at that depth must be good. Multiplicative gating: bad predictions zero out the horizon advantage.

#### agent_environment_coupling (threshold: 0.3, weight: 0.9)

**Theory (FEP):** Bidirectional coupling between agent and environment.

**Current code:** `0.3 + 0.35 * (actions/10) + 0.35 * (observations/10)`.

**New formula:**
```python
action_count = stats.get("total_inferences", 0)
observation_count = stats.get("total_model_updates", 0)
if action_count == 0 and observation_count == 0:
    score = 0.0
else:
    # Bidirectional: both acting AND observing must happen
    act_rate = min(1.0, action_count / max(1, warmup_cycles))
    obs_rate = min(1.0, observation_count / max(1, warmup_cycles))
    # Geometric mean: both must be present (0 in either = 0)
    score = (act_rate * obs_rate) ** 0.5
```

**Rationale:** Geometric mean enforces bidirectionality. 25 actions + 0 observations = 0 (no coupling). The 0.3 floor is removed.

#### agency (threshold: 0.5, weight: 1.0)

**Theory (FEP):** Goal-directed autonomous behavior.

**Paper says:** "Action selection frequency, policy quality, goal-directed behavior."

**Current code:** `base_score=0.4` + `inferences/10` + drive + urgency.

**New formula:**
```python
inferences = stats.get("total_inferences", 0)
model_updates = stats.get("total_model_updates", 0)
free_energy = stats.get("variational_free_energy", 0.0)
if inferences == 0:
    score = 0.0
else:
    # Inference rate relative to opportunity
    inference_rate = min(1.0, inferences / max(1, warmup_cycles))
    # Model updating shows the system is learning from its actions
    update_rate = min(1.0, model_updates / max(1, inferences))
    # Free energy > 0 means the system has active drives
    drive_score = min(0.3, 0.3 * (1.0 - 1.0 / (1.0 + float(free_energy)))) if free_energy > 0 else 0.0
    score = 0.4 * inference_rate + 0.3 * update_rate + 0.3 * drive_score
```

**Rationale:** Remove base score. Balance inference activity with learning (model updates) and motivation (free energy drives).

#### sparse_smooth_coding (threshold: 0.3, weight: 0.8)

**Theory (FEP supplementary):** Efficient, sparse internal representations.

**Paper says:** "Sparsity (near-zero activations) and smoothness (gradient continuity)."

**Current code:** `base_score=0.3` + `1.0 - avg_error * 1.5`.

**New formula:**
```python
avg_error = stats.get("avg_prediction_error", 0.5)
total_inferences = stats.get("total_inferences", 0)
if total_inferences == 0:
    score = 0.0
else:
    # Sparse coding: low, concentrated errors (not uniform noise)
    sparsity_score = max(0.0, 1.0 - float(avg_error) * 2.0)
    score = sparsity_score
```

**Rationale:** Remove base score. Tighter scaling (x2.0 instead of x1.5) so moderate errors score lower.

#### affect_coherence (threshold: 0.3, weight: 0.9)

**Current code:** `pad.compute_coherence()`: already quality-based.

**No change.** This indicator already measures real coherence of PAD dimensional states. Just ensure score = 0.0 when module is absent (already handled by the else branch).

---

### IIT Indicators

#### integrated_information (threshold: 0.3, weight: 1.3)

**Theory (Tononi 2004, 2008):** Consciousness = integrated information (Phi).

**Paper says:** "Normalized Phi value from approximation."

**Current code:** If phi>0: `0.3 + phi * 1.4`. Else: base 0.3 if neural present.

**New formula:**
```python
phi_val = kwargs.get("phi_value", 0.0)
if phi_val > 0:
    score = min(1.0, phi_val)  # Phi normalized is already 0-1
else:
    score = 0.0  # No integrated information = no score
```

**Rationale:** Phi is Phi. No inflation. If the calculator returns 0, score is 0.

#### irreducibility (threshold: 0.4, weight: 1.0)

**Theory (IIT):** System cannot be decomposed without information loss.

**Paper says:** "Comparing full system to partitions."

**Current code:** `sparsity = sum(abs(conn) > 0.1) / size` then `sparsity * 2`.

**New formula:**
```python
if neural and "connectivity" in neural:
    conn = neural["connectivity"]
    # Eigenvalue distribution: genuine irreducibility = distributed spectrum
    # A reducible system has one dominant eigenvalue (rank-1 structure)
    eigenvalues = np.linalg.eigvalsh(conn)
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-10)  # Normalize
    # Spectral entropy: high = distributed (irreducible), low = concentrated (reducible)
    spectral_entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
    max_entropy = np.log(len(eigenvalues))
    score = spectral_entropy / max_entropy if max_entropy > 0 else 0.0
else:
    score = 0.0
```

**Rationale:** Spectral entropy directly measures irreducibility. A rank-1 matrix (like the engineered connectivity we removed) has low spectral entropy and scores LOW. Random connectivity has moderate spectral entropy. Only genuinely integrated systems score high.

#### causal_emergence (threshold: 0.0, weight: 1.2)

**Current code:** Sigmoid of CE value from analyzer: already quality-based.

**Change:** Remove the `score = 0.3` fallback when analyzer is warming up. If analyzer isn't ready, score = 0.0.

---

### RPT Indicators

#### algorithmic_recurrence (threshold: 0.3, weight: 0.9)

**Theory (Lamme 2006):** Active recurrent processing in neural substrates.

**Paper says:** "Spike re-entry rates, reservoir trajectory, recurrence classification."

**Current code:** `base_score=0.35` + `0.6 * avg_local + 0.4 * avg_global`.

**New formula:**
```python
avg_local = stats.get("average_local", 0.0)
avg_global = stats.get("average_global", 0.0)
measurement_count = stats.get("measurement_count", 0)
if measurement_count == 0:
    score = 0.0
else:
    score = 0.6 * avg_local + 0.4 * avg_global
```

**Rationale:** Remove base score. The RPT module already computes genuine recurrence metrics. Use them directly.

#### local_recurrence (threshold: 0.3, weight: 0.8)

**Theory (RPT):** Feedback within individual neural substrates.

**Current code:** `base_score=0.3`, `max(base_score, avg_local)`.

**New formula:**
```python
avg_local = stats.get("average_local", 0.0)
measurement_count = stats.get("measurement_count", 0)
score = avg_local if measurement_count > 0 else 0.0
```

**Rationale:** Direct measurement. No floor.

#### oscillatory_binding_coherence (threshold: 0.3, weight: 1.0)

**Current code:** `0.6 * global_r + 0.4 * mean_cross`, base 0.25.

**New formula:**
```python
binding_count = osc_stats.get("binding_count", 0)
if binding_count == 0:
    score = 0.0
else:
    score = 0.6 * global_r + 0.4 * mean_cross  # Already quality-based
```

**Rationale:** Remove base score. Kuramoto order parameter is a genuine quality metric.

---

### BLT Indicators

#### bayesian_binding_quality (threshold: 0.3, weight: 1.0)

**Theory (Laukkonen et al. 2025):** Bayesian binding into unified percepts.

**Paper says:** "Mutual information in coherent subset, normalized by bound inferences."

**Current code:** `base_score=0.3`, `max(base_score, avg_binding)`.

**New formula:**
```python
avg_binding = stats.get("average_binding_quality", 0.0)
cycle_count = stats.get("cycle_count", 0)
score = avg_binding if cycle_count > 0 else 0.0
```

**Rationale:** Direct quality measurement. No floor.

#### epistemic_depth (threshold: 0.3, weight: 1.1)

**Theory (BLT):** Recursive self-reference depth.

**Paper says:** "Depth level divided by 4, capped at 1.0."

**Current code:** `base_score=0.3`, `avg_depth / 3.0`.

**New formula:**
```python
avg_depth = depth_stats.get("average_depth", 0)
score = min(1.0, avg_depth / 4.0) if avg_depth > 0 else 0.0
```

**Rationale:** Paper says /4.0. Code had /3.0 (inflated). No base score.

#### genuine_implementation (threshold: 0.4, weight: 1.2)

**Theory (BLT):** Anti-mimicry: genuine recursive processing, not score inflation.

**Paper says:** "Requires field-evidencing + loop quality > 0.4 + strange loop."

**Current code:** `base_score=0.3` + `0.6 * fe_rate + 0.4 * quality`.

**New formula:**
```python
fe_count = stats.get("field_evidencing_count", 0)
fe_rate = stats.get("field_evidencing_rate", 0.0)
avg_quality = stats.get("average_loop_quality", 0.0)
cycle_count = stats.get("cycle_count", 0)

if cycle_count == 0:
    score = 0.0
else:
    # Three conditions per paper:
    has_field_evidencing = fe_count > 0
    has_loop_quality = avg_quality >= 0.4
    # Strange loop: epistemic depth >= 2 (system models its own modeling)
    # depth_stats comes from the same BLT stats dict: stats.get("epistemic_depth", {})
    depth_stats = stats.get("epistemic_depth", {})
    has_strange_loop = depth_stats.get("max_depth_reached", 0) >= 2

    conditions_met = sum([has_field_evidencing, has_loop_quality, has_strange_loop])
    # Proportional: 0/3 = 0.0, 1/3 = 0.33, 2/3 = 0.67, 3/3 = 1.0
    score = conditions_met / 3.0
```

**Rationale:** Three binary conditions from the paper. This is the anti-mimicry check: the hardest indicator to pass. No partial credit beyond condition counting.

---

## New Module Statistics Required

These statistics need to be added to module `get_statistics()` methods:

| Module | Statistic | Purpose |
|---|---|---|
| `EnhancedGlobalWorkspace` | `cycle_count` already exposed | Used as divisor for ignition/broadcast rates |
| `MetacognitionModule` | Access to GWT `cycle_count` during assessment | For HOT coverage ratio |
| `ActiveInferenceModule` | `prediction_error_trend` (slope) | Already partially exposed, verify it's a numeric slope |

Most statistics are already available. The main gap is that HOT coverage ratio needs both HOT count and GWT cycle count in the same formula, which is straightforward since both modules are passed to `_measure_indicator()`.

## Experiment Runner Changes

The neutral data changes already made to `scripts/run_experiment.py` (random connectivity, full-range spike data, naive self-model) stay as-is. They correctly represent uninstructed synthetic data.

One addition: pass `warmup_cycles=WARMUP_CYCLES` to `run_full_assessment()` as a kwarg. The assessment stores it as `self._warmup_cycles` (default: 25) and formulas use it to scale divisors to opportunity. Formulas that reference `warmup_cycles` in pseudocode above use `self._warmup_cycles` in the actual implementation. For external callers who don't use the experiment runner, the default of 25 provides reasonable scaling.

## Paper Versioning

### v0.1.1 Paper
- Freeze as-is. Already tagged. Paper in `paper/` directory matches v0.1.1 code.
- No retroactive changes.

### v0.3.1 Paper
- Update Appendix A: 25 indicators with quality-based scoring descriptions
- Update Section 5 (Assessment): Add calibration methodology subsection
  - Noise baseline principle: random inputs should score below threshold
  - Quality-based scoring: each formula measures theoretical prediction quality
  - Activity vs. quality distinction
- Update Section 6 (Results): Honest baseline scores from recalibrated framework
  - Report noise-floor scores
  - Report scores with proper module wiring
  - Cross-theory interaction effects may change (re-run ablation)
- Update Abstract: 25 indicators (was 20)
- Add new indicators to Section 5 tables: temporal_prediction_depth, agent_environment_coupling, oscillatory_binding_coherence, causal_emergence, affect_coherence

## Testing Strategy

1. **Noise baseline test:** Run assessment with random data, verify all indicators score below their thresholds
2. **Ablation delta test:** For each module, verify that disabling it reduces scores for its theory's indicators (the existing `--validate` flag)
3. **Monotonicity test:** For each indicator, verify that better-quality input produces higher score (e.g., decreasing prediction errors score higher than flat errors)
4. **Regression test:** Existing unit tests in `tests/assessment/` must still pass (may need threshold adjustments in test expectations)

## Implementation Order

1. Remove all base scores (mechanical, low risk)
2. Replace activity-count formulas with quality formulas (per-indicator, medium risk)
3. Add `warmup_cycles` passthrough to assessment
4. Update experiment runner to pass cycle count
5. Run noise baseline: verify scores below threshold
6. Run ablation: verify cross-theory effects
7. Update paper to v0.3.1
8. Tag release
