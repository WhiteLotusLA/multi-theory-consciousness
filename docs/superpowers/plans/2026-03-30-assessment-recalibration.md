# Assessment Recalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace activity-counting assessment formulas with quality-based scoring that measures theoretical predictions, so random inputs score below threshold and only genuine consciousness-like properties score above threshold.

**Architecture:** All changes are in `mtc/assessment/assessment.py` (scoring formulas) and `scripts/run_experiment.py` (warmup_cycles passthrough). Tests updated to match new expectations. No module code changes: modules already produce the quality statistics we need.

**Tech Stack:** Python 3.14, numpy, pytest, asyncio

**Spec:** `docs/superpowers/specs/2026-03-30-assessment-recalibration-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `mtc/assessment/assessment.py` | Modify (lines 640-1380) | All 25 indicator scoring formulas |
| `scripts/run_experiment.py` | Modify (line 451) | Pass warmup_cycles to assessment |
| `tests/assessment/test_recalibration.py` | Create | Noise baseline + quality signal tests |
| `tests/consciousness/test_assessment.py` | Modify | Adjust score expectations |
| `tests/consciousness/test_assessment_upgrade.py` | Modify | Adjust score/threshold expectations |
| `tests/assessment/test_new_indicators.py` | Modify | Adjust score expectations |

---

### Task 1: Add warmup_cycles parameter to ConsciousnessAssessment

**Files:**
- Modify: `mtc/assessment/assessment.py:421-440` (constructor)
- Modify: `mtc/assessment/assessment.py:630-650` (run_full_assessment signature)
- Modify: `scripts/run_experiment.py:451` (pass warmup_cycles)

- [ ] **Step 1: Add warmup_cycles to run_full_assessment**

In `mtc/assessment/assessment.py`, find the `run_full_assessment` method signature (around line 630) and add `warmup_cycles` parameter:

```python
    async def run_full_assessment(
        self,
        global_workspace=None,
        attention_schema=None,
        metacognition=None,
        active_inference=None,
        neural_states=None,
        conversation_data=None,
        ablation_config=None,
        beautiful_loop=None,
        rpt_measurement=None,
        pad_affect_model=None,
        causal_emergence_analyzer=None,
        warmup_cycles: int = 25,
    ) -> "ConsciousnessReport":
```

Store it for use in formulas by adding this line at the start of the method body (after `start_time`):

```python
        self._warmup_cycles = warmup_cycles
```

Also pass `warmup_cycles=warmup_cycles` to the `_measure_indicator` call by adding it to the kwargs dict. Find the line that calls `await self._measure_indicator(` and add `warmup_cycles=warmup_cycles` to its kwargs:

```python
                score, confidence, evidence = await self._measure_indicator(
                    name,
                    config,
                    global_workspace=global_workspace,
                    attention_schema=attention_schema,
                    metacognition=metacognition,
                    active_inference=active_inference,
                    neural_states=neural_states,
                    conversation_data=conversation_data,
                    ablation_config=ablation_config,
                    beautiful_loop=beautiful_loop,
                    rpt_measurement=rpt_measurement,
                    pad_affect_model=pad_affect_model,
                    causal_emergence_analyzer=causal_emergence_analyzer,
                    phi_value=phi_value,
                    warmup_cycles=warmup_cycles,
                )
```

- [ ] **Step 2: Update experiment runner to pass warmup_cycles**

In `scripts/run_experiment.py`, find the line (around 451):

```python
    return await assessment.run_full_assessment(**modules)
```

Replace with:

```python
    return await assessment.run_full_assessment(**modules, warmup_cycles=WARMUP_CYCLES)
```

Do the same in `run_ablation` (around line 519):

```python
        ablated_report = await assessment.run_full_assessment(**modules, warmup_cycles=WARMUP_CYCLES)
```

- [ ] **Step 3: Run existing tests to verify no breakage**

Run: `python -m pytest tests/consciousness/test_assessment.py tests/consciousness/test_assessment_upgrade.py tests/assessment/test_new_indicators.py -v --timeout=60 2>&1 | tail -30`

Expected: All existing tests still pass (warmup_cycles has a default of 25).

- [ ] **Step 4: Commit**

```bash
git add mtc/assessment/assessment.py scripts/run_experiment.py
git commit -m "feat: add warmup_cycles parameter to assessment for quality-based scaling"
```

---

### Task 2: Recalibrate GWT indicators (4 indicators)

**Files:**
- Modify: `mtc/assessment/assessment.py:782-855` (GWT indicator block)
- Modify: `mtc/assessment/assessment.py:1352-1377` (global_ignition_nuanced block)

- [ ] **Step 1: Rewrite global_broadcast formula**

In `mtc/assessment/assessment.py`, find the `global_broadcast` block (starts around line 785). Replace the entire block from `if name == "global_broadcast":` through the `evidence = {` dict ending with:

```python
                if name == "global_broadcast":
                    stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                    broadcast_stats = stats.get("broadcast", {})
                    broadcast_count = broadcast_stats.get("total_broadcasts", 0)
                    coverage = broadcast_stats.get("coverage_ratio", 0.0)
                    if broadcast_count == 0:
                        score = 0.0
                    else:
                        score = min(1.0, coverage)
                    confidence = 0.7 if broadcast_count > 10 else 0.4
                    evidence = {
                        "broadcasts": broadcast_count,
                        "coverage": coverage,
                        "module_active": True,
                    }
```

- [ ] **Step 2: Rewrite ignition_dynamics formula**

Find the `ignition_dynamics` block. Replace entirely:

```python
                elif name == "ignition_dynamics":
                    stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                    ignition_stats = stats.get("ignition", {})
                    ignitions = ignition_stats.get("total_ignitions", 0)
                    sustain_rate = ignition_stats.get("sustain_rate", 0.0)
                    cycle_count = stats.get("cycle_count", 0)
                    if ignitions == 0:
                        score = 0.0
                    else:
                        ignition_rate = min(1.0, ignitions / max(1, cycle_count))
                        sustain_quality = sustain_rate
                        score = 0.6 * ignition_rate + 0.4 * sustain_quality
                    confidence = 0.7 if ignitions > 5 else 0.4
                    evidence = {
                        "ignitions": ignitions,
                        "ignition_rate": ignitions / max(1, cycle_count) if cycle_count > 0 else 0.0,
                        "sustain_rate": sustain_rate,
                        "module_active": True,
                    }
```

- [ ] **Step 3: Rewrite recurrent_processing formula**

Find the `recurrent_processing` block. Replace entirely:

```python
                elif name == "recurrent_processing":
                    stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                    broadcast_stats = stats.get("broadcast", {})
                    coverage = broadcast_stats.get("coverage_ratio", 0.0)
                    cycle_count = stats.get("cycle_count", 0)
                    broadcasts = broadcast_stats.get("total_broadcasts", 0)
                    if cycle_count == 0:
                        score = 0.0
                    else:
                        broadcast_rate = min(1.0, broadcasts / cycle_count)
                        score = 0.5 * coverage + 0.5 * broadcast_rate
                    confidence = 0.5 + 0.2 * min(1.0, cycle_count / 10)
                    evidence = {
                        "coverage": coverage,
                        "cycles": cycle_count,
                        "broadcasts": broadcasts,
                        "module_active": True,
                    }
```

- [ ] **Step 4: Rewrite global_ignition_nuanced formula**

Find the `global_ignition_nuanced` block (around line 1352). Replace entirely:

```python
        elif name == "global_ignition_nuanced":
            gw = kwargs.get("global_workspace")
            if gw is not None:
                stats = gw.get_statistics() if hasattr(gw, "get_statistics") else {}
                ignition_stats = stats.get("ignition", {})
                ignitions = ignition_stats.get("total_ignitions", 0)
                sustain_rate = ignition_stats.get("sustain_rate", 0.0)
                decay_rate = ignition_stats.get("decay_rate", 0.0)
                cycle_count = stats.get("cycle_count", 0)
                if ignitions == 0:
                    score = 0.0
                else:
                    ignition_rate = min(1.0, ignitions / max(1, cycle_count))
                    dynamics_quality = sustain_rate * 0.7 + (1.0 - decay_rate) * 0.3
                    score = 0.5 * ignition_rate + 0.5 * dynamics_quality
                confidence = 0.6 if ignitions > 3 else 0.4
                evidence = {
                    "ignitions": ignitions,
                    "sustain_rate": sustain_rate,
                    "decay_rate": decay_rate,
                    "module_active": True,
                }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True}
```

- [ ] **Step 5: Run experiment to see GWT score change**

Run: `python scripts/run_experiment.py 2>&1 | grep -E "gwt_score|overall_score|passing_count"`

Expected: GWT score should drop from 1.0. Exact value TBD but should be below 1.0.

- [ ] **Step 6: Commit**

```bash
git add mtc/assessment/assessment.py
git commit -m "recalibrate: GWT indicators, quality-based scoring, no base scores"
```

---

### Task 3: Recalibrate AST indicators (3 indicators)

**Files:**
- Modify: `mtc/assessment/assessment.py:857-957` (AST indicator block)

- [ ] **Step 1: Rewrite attention_schema formula**

Find the `attention_schema` block (around line 861). Replace the scoring logic inside the try block:

```python
                if name == "attention_schema":
                    try:
                        stats = (
                            ast.get_statistics()
                            if hasattr(ast, "get_statistics")
                            else {}
                        )
                        schema = getattr(ast, "schema", None)
                        warmup_cycles = kwargs.get("warmup_cycles", 25)

                        has_focus = (
                            schema is not None
                            and getattr(schema, "current_focus", None) is not None
                        )
                        total_updates = stats.get("total_updates", 0)
                        history_len = stats.get("history_length", 0)
                        pred_accuracy = stats.get("prediction_accuracy", 0.0)

                        schema_active = min(1.0, total_updates / max(1, warmup_cycles * 0.5))
                        focus_score = 1.0 if has_focus else 0.0
                        history_score = min(1.0, history_len / max(1, warmup_cycles * 0.3))
                        prediction_score = pred_accuracy

                        score = (
                            0.15 * schema_active
                            + 0.15 * focus_score
                            + 0.20 * history_score
                            + 0.50 * prediction_score
                        )
                        confidence = 0.5 + 0.3 * min(1.0, total_updates / 5)
                        evidence = {
                            "has_focus": has_focus,
                            "total_updates": total_updates,
                            "history_length": history_len,
                            "prediction_accuracy": round(pred_accuracy, 3),
                            "module_active": True,
                        }
                    except Exception as e:
                        score = 0.0
                        confidence = 0.3
                        evidence = {"module_present": True, "access_error": str(e)}
```

- [ ] **Step 2: Rewrite attention_control formula**

```python
                elif name == "attention_control":
                    stats = (
                        ast.get_statistics() if hasattr(ast, "get_statistics") else {}
                    )
                    voluntary_shifts = stats.get("voluntary_shifts", 0)
                    voluntary_ratio = stats.get("voluntary_ratio", 0.0)
                    if voluntary_shifts == 0:
                        score = 0.0
                    else:
                        score = min(1.0, voluntary_ratio)
                    confidence = 0.6 if voluntary_shifts > 3 else 0.3
                    evidence = {
                        "voluntary_shifts": voluntary_shifts,
                        "voluntary_ratio": voluntary_ratio,
                        "module_active": True,
                    }
```

- [ ] **Step 3: Rewrite embodiment formula**

```python
                elif name == "embodiment":
                    stats = ast.get_statistics() if hasattr(ast, "get_statistics") else {}
                    warmup_cycles = kwargs.get("warmup_cycles", 25)
                    capacity = stats.get("capacity_used", 0.0)
                    history_len = stats.get("history_length", 0)
                    vol_ratio = stats.get("voluntary_ratio", 0.0)
                    capacity_score = min(1.0, capacity) if capacity > 0 else 0.0
                    perspective_score = min(1.0, history_len / max(1, warmup_cycles * 0.5))
                    agency_score = vol_ratio
                    score = 0.4 * capacity_score + 0.3 * perspective_score + 0.3 * agency_score
                    confidence = 0.5 + 0.2 * min(1.0, history_len / 5)
                    evidence = {
                        "capacity_used": capacity,
                        "history_length": history_len,
                        "voluntary_ratio": vol_ratio,
                        "module_active": True,
                    }
```

- [ ] **Step 4: Commit**

```bash
git add mtc/assessment/assessment.py
git commit -m "recalibrate: AST indicators, prediction accuracy weighted, no base scores"
```

---

### Task 4: Recalibrate HOT indicators (2 indicators)

**Files:**
- Modify: `mtc/assessment/assessment.py:959-1003` (HOT indicator block)

- [ ] **Step 1: Rewrite higher_order_representations formula**

```python
                if name == "higher_order_representations":
                    stats = (
                        meta.get_statistics() if hasattr(meta, "get_statistics") else {}
                    )
                    hot_count = stats.get("total_hots_generated", 0)
                    # Use GWT cycle_count as proxy for conscious moment count
                    gw = kwargs.get("global_workspace")
                    gw_stats = gw.get_statistics() if gw is not None and hasattr(gw, "get_statistics") else {}
                    cycle_count = gw_stats.get("cycle_count", 0)
                    if cycle_count == 0 or hot_count == 0:
                        score = 0.0
                    else:
                        coverage_ratio = min(1.0, hot_count / cycle_count)
                        score = coverage_ratio
                    confidence = 0.7 if hot_count > 5 else 0.4
                    evidence = {
                        "hots_generated": hot_count,
                        "cycle_count": cycle_count,
                        "coverage_ratio": hot_count / max(1, cycle_count),
                        "module_active": True,
                    }
```

- [ ] **Step 2: Rewrite metacognition formula**

```python
                elif name == "metacognition":
                    stats = (
                        meta.get_statistics() if hasattr(meta, "get_statistics") else {}
                    )
                    introspections = stats.get("introspection_count", 0)
                    belief_evaluations = stats.get("belief_evaluations", 0)
                    overall_confidence = stats.get("overall_confidence", 0.0)
                    gw = kwargs.get("global_workspace")
                    gw_stats = gw.get_statistics() if gw is not None and hasattr(gw, "get_statistics") else {}
                    cycle_count = gw_stats.get("cycle_count", 0)
                    if introspections + belief_evaluations == 0:
                        score = 0.0
                    else:
                        activity_rate = min(1.0, (introspections + belief_evaluations) / max(1, cycle_count))
                        quality = overall_confidence
                        score = 0.4 * activity_rate + 0.6 * quality
                    confidence = 0.6 if introspections > 2 else 0.3
                    evidence = {
                        "introspections": introspections,
                        "evaluations": belief_evaluations,
                        "overall_confidence": overall_confidence,
                        "module_active": True,
                    }
```

- [ ] **Step 3: Commit**

```bash
git add mtc/assessment/assessment.py
git commit -m "recalibrate: HOT indicators, coverage ratio, confidence quality weighting"
```

---

### Task 5: Recalibrate FEP indicators (7 indicators)

**Files:**
- Modify: `mtc/assessment/assessment.py:1005-1143` (FEP indicator block)
- Modify: `mtc/assessment/assessment.py:1328-1350` (sparse_smooth_coding block)

- [ ] **Step 1: Rewrite prediction_error_minimization formula**

```python
                if name == "prediction_error_minimization":
                    stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                    avg_error = stats.get("avg_prediction_error", 0.5)
                    error_trend = stats.get("prediction_error_trend", 0)
                    total_inferences = stats.get("total_inferences", 0)
                    if isinstance(avg_error, str):
                        avg_error = 0.5
                    if total_inferences < 2:
                        score = 0.0
                    elif isinstance(error_trend, (int, float)) and error_trend < 0:
                        trend_magnitude = min(1.0, abs(float(error_trend)) * 5)
                        level_score = max(0.0, 1.0 - float(avg_error))
                        score = 0.7 * trend_magnitude + 0.3 * level_score
                    else:
                        level_score = max(0.0, 1.0 - float(avg_error)) * 0.3
                        score = level_score
                    confidence = 0.6 if total_inferences > 5 else 0.3
                    evidence = {
                        "avg_error": avg_error,
                        "trend": error_trend,
                        "total_inferences": total_inferences,
                        "module_active": True,
                    }
```

- [ ] **Step 2: Rewrite hierarchical_prediction formula**

```python
                elif name == "hierarchical_prediction":
                    stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                    hierarchical_error = stats.get("hierarchical_prediction_error", 0.5)
                    total_inferences = stats.get("total_inferences", 0)
                    warmup_cycles = kwargs.get("warmup_cycles", 25)
                    if total_inferences == 0:
                        score = 0.0
                    else:
                        if isinstance(hierarchical_error, (int, float)):
                            error_quality = max(0.0, 1.0 - float(hierarchical_error))
                        else:
                            error_quality = 0.0
                        activity_rate = min(1.0, total_inferences / max(1, warmup_cycles))
                        score = 0.7 * error_quality + 0.3 * activity_rate
                    confidence = 0.6 if total_inferences > 5 else 0.3
                    evidence = {
                        "hierarchical_error": hierarchical_error,
                        "inferences": total_inferences,
                        "module_active": True,
                    }
```

- [ ] **Step 3: Rewrite temporal_prediction_depth formula**

```python
                elif name == "temporal_prediction_depth":
                    fep = kwargs.get("active_inference")
                    if fep is not None:
                        horizon = fep.config.planning_horizon if hasattr(fep, "config") else 3
                        stats = fep.get_statistics() if hasattr(fep, "get_statistics") else {}
                        total_inferences = stats.get("total_inferences", 0)
                        avg_error = stats.get("avg_prediction_error", 0.5)
                        if isinstance(avg_error, str):
                            avg_error = 0.5
                        if total_inferences == 0:
                            score = 0.0
                        else:
                            horizon_factor = min(1.0, horizon / 10.0)
                            quality = max(0.0, 1.0 - float(avg_error))
                            score = horizon_factor * quality
                        confidence = 0.5 + 0.2 * min(1.0, total_inferences / 10)
                        evidence = {"planning_horizon": horizon, "total_inferences": total_inferences, "avg_error": avg_error}
                    else:
                        score, confidence = 0.0, 0.3
                        evidence = {"module_absent": True}
```

- [ ] **Step 4: Rewrite agent_environment_coupling formula**

```python
                elif name == "agent_environment_coupling":
                    fep = kwargs.get("active_inference")
                    if fep is not None:
                        stats = fep.get_statistics() if hasattr(fep, "get_statistics") else {}
                        warmup_cycles = kwargs.get("warmup_cycles", 25)
                        action_count = stats.get("total_inferences", 0)
                        observation_count = stats.get("total_model_updates", 0)
                        if action_count == 0 and observation_count == 0:
                            score = 0.0
                        else:
                            act_rate = min(1.0, action_count / max(1, warmup_cycles))
                            obs_rate = min(1.0, observation_count / max(1, warmup_cycles))
                            score = (act_rate * obs_rate) ** 0.5
                        confidence = 0.5 + 0.3 * min(1.0, (action_count + observation_count) / 20)
                        evidence = {"actions": action_count, "observations": observation_count}
                    else:
                        score, confidence = 0.0, 0.3
                        evidence = {"module_absent": True}
```

- [ ] **Step 5: Rewrite agency formula**

```python
                elif name == "agency":
                    stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                    warmup_cycles = kwargs.get("warmup_cycles", 25)
                    inferences = stats.get("total_inferences", 0)
                    model_updates = stats.get("total_model_updates", 0)
                    free_energy = stats.get("variational_free_energy", 0.0)
                    if inferences == 0:
                        score = 0.0
                    else:
                        inference_rate = min(1.0, inferences / max(1, warmup_cycles))
                        update_rate = min(1.0, model_updates / max(1, inferences))
                        drive_score = (
                            min(0.3, 0.3 * (1.0 - 1.0 / (1.0 + float(free_energy))))
                            if isinstance(free_energy, (int, float)) and free_energy > 0
                            else 0.0
                        )
                        score = min(1.0, 0.4 * inference_rate + 0.3 * update_rate + 0.3 * drive_score)
                    confidence = 0.6 if inferences > 5 else 0.3
                    evidence = {
                        "inferences": inferences,
                        "model_updates": model_updates,
                        "free_energy": free_energy,
                        "module_active": True,
                    }
```

- [ ] **Step 6: Rewrite affect_coherence, no change needed, just verify**

The `affect_coherence` block already uses `pad.compute_coherence()` which is quality-based. Verify it has no base score. The existing code should be:

```python
                elif name == "affect_coherence":
                    pad = kwargs.get("pad_affect_model")
                    if pad is not None:
                        score = pad.compute_coherence()
                        ...
```

No changes needed here.

- [ ] **Step 7: Rewrite sparse_smooth_coding formula**

Find the `sparse_smooth_coding` block (around line 1328):

```python
        if name == "sparse_smooth_coding":
            ai = kwargs.get("active_inference")
            if ai is not None:
                stats = ai.get_statistics() if hasattr(ai, "get_statistics") else {}
                avg_error = stats.get("avg_prediction_error", 0.5)
                total_inferences = stats.get("total_inferences", 0)
                if isinstance(avg_error, str):
                    avg_error = 0.5
                if total_inferences == 0:
                    score = 0.0
                else:
                    sparsity_score = max(0.0, 1.0 - float(avg_error) * 2.0)
                    score = sparsity_score
                confidence = 0.5
                evidence = {
                    "avg_error": avg_error,
                    "sparsity_proxy": score,
                    "module_active": True,
                }
            else:
                score = 0.1
                confidence = 0.3
                evidence = {"ablated": True}
```

- [ ] **Step 8: Commit**

```bash
git add mtc/assessment/assessment.py
git commit -m "recalibrate: FEP indicators, trend-based, multiplicative gating, geometric mean coupling"
```

---

### Task 6: Recalibrate IIT indicators (3 indicators)

**Files:**
- Modify: `mtc/assessment/assessment.py:1145-1200` (IIT indicator block)

- [ ] **Step 1: Rewrite integrated_information formula**

```python
            if name == "integrated_information":
                phi_val = kwargs.get("phi_value", 0.0)
                if phi_val > 0:
                    score = min(1.0, phi_val)
                    confidence = 0.7
                    evidence = {
                        "phi_normalized": phi_val,
                        "source": "pre_computed_phi",
                    }
                else:
                    score = 0.0
                    confidence = 0.4
                    evidence = {"phi_zero": True, "neural_present": bool(neural)}
```

- [ ] **Step 2: Rewrite irreducibility formula with spectral entropy**

```python
            elif name == "irreducibility":
                if neural and "connectivity" in neural:
                    conn = np.array(neural["connectivity"])
                    try:
                        eigenvalues = np.linalg.eigvalsh(conn)
                        eigenvalues = np.abs(eigenvalues)
                        total = eigenvalues.sum()
                        if total > 0:
                            eigenvalues = eigenvalues / total
                            spectral_entropy = -np.sum(
                                eigenvalues * np.log(eigenvalues + 1e-10)
                            )
                            max_entropy = np.log(len(eigenvalues))
                            score = spectral_entropy / max_entropy if max_entropy > 0 else 0.0
                        else:
                            score = 0.0
                        evidence = {
                            "spectral_entropy": float(spectral_entropy) if total > 0 else 0.0,
                            "max_entropy": float(max_entropy) if total > 0 else 0.0,
                            "n_eigenvalues": len(eigenvalues),
                        }
                    except Exception as e:
                        score = 0.0
                        evidence = {"computation_error": str(e)}
                else:
                    score = 0.0
                    evidence = {"connectivity_missing": True}
                confidence = 0.5
```

- [ ] **Step 3: Fix causal_emergence warming-up fallback**

Find the `causal_emergence` block. Change the warming-up fallback from `score = 0.3` to `score = 0.0`:

```python
            elif name == "causal_emergence":
                ce_analyzer = kwargs.get("causal_emergence_analyzer")
                if ce_analyzer is not None:
                    ce_val = ce_analyzer.get_latest_ce()
                    if ce_val is not None:
                        score = 1.0 / (1.0 + np.exp(-ce_val * 2.0))
                        confidence = 0.7
                        evidence = {
                            "causal_emergence": ce_val,
                            "macro_beats_micro": ce_val > 0,
                            "module_active": True,
                        }
                    else:
                        score = 0.0
                        confidence = 0.2
                        evidence = {"warming_up": True}
                else:
                    score = 0.0
                    confidence = 0.3
                    evidence = {"ablated": True, "reason": "analyzer not provided"}
```

- [ ] **Step 4: Commit**

```bash
git add mtc/assessment/assessment.py
git commit -m "recalibrate: IIT indicators, direct phi, spectral entropy for irreducibility"
```

---

### Task 7: Recalibrate RPT indicators (3 indicators)

**Files:**
- Modify: `mtc/assessment/assessment.py:1202-1262` (RPT indicator block)

- [ ] **Step 1: Rewrite algorithmic_recurrence formula**

```python
                if name == "algorithmic_recurrence":
                    stats = (
                        rpt.get_statistics() if hasattr(rpt, "get_statistics") else {}
                    )
                    avg_local = stats.get("average_local", 0.0)
                    avg_global = stats.get("average_global", 0.0)
                    measurement_count = stats.get("measurement_count", 0)
                    if measurement_count == 0:
                        score = 0.0
                    else:
                        score = 0.6 * avg_local + 0.4 * avg_global
                    confidence = 0.5 + 0.3 * min(1.0, measurement_count / 10)
                    evidence = {
                        "avg_local": avg_local,
                        "avg_global": avg_global,
                        "measurements": measurement_count,
                        "module_active": True,
                    }
```

- [ ] **Step 2: Rewrite local_recurrence formula**

```python
                elif name == "local_recurrence":
                    stats = rpt.get_statistics() if hasattr(rpt, "get_statistics") else {}
                    avg_local = stats.get("average_local", 0.0)
                    measurement_count = stats.get("measurement_count", 0)
                    score = avg_local if measurement_count > 0 else 0.0
                    confidence = 0.5 + 0.2 * min(1.0, measurement_count / 10)
                    evidence = {
                        "avg_local_recurrence": avg_local,
                        "measurements": measurement_count,
                        "module_active": True,
                    }
```

- [ ] **Step 3: Rewrite oscillatory_binding_coherence formula**

```python
                elif name == "oscillatory_binding_coherence":
                    osc_stats = rpt.get_oscillatory_binding_stats() if hasattr(
                        rpt, "get_oscillatory_binding_stats"
                    ) else {}
                    global_r = osc_stats.get("global_order_parameter", 0.0)
                    cross_vals = osc_stats.get("cross_coherence", {})
                    binding_count = osc_stats.get("binding_count", 0)
                    if cross_vals:
                        mean_cross = max(0.0, float(np.mean(list(cross_vals.values()))))
                    else:
                        mean_cross = 0.0
                    if binding_count == 0:
                        score = 0.0
                    else:
                        score = 0.6 * global_r + 0.4 * mean_cross
                    confidence = 0.5 + 0.3 * min(1.0, binding_count / 10)
                    evidence = {
                        "global_order_parameter": global_r,
                        "cross_coherence": cross_vals,
                        "mean_cross_coherence": mean_cross,
                        "binding_count": binding_count,
                        "module_active": bool(osc_stats),
                    }
```

- [ ] **Step 4: Commit**

```bash
git add mtc/assessment/assessment.py
git commit -m "recalibrate: RPT indicators, direct recurrence metrics, no base scores"
```

---

### Task 8: Recalibrate BLT indicators (3 indicators)

**Files:**
- Modify: `mtc/assessment/assessment.py:1264-1325` (BLT indicator block)

- [ ] **Step 1: Rewrite bayesian_binding_quality formula**

```python
                if name == "bayesian_binding_quality":
                    avg_binding = stats.get("average_binding_quality", 0.0)
                    cycle_count = stats.get("cycle_count", 0)
                    score = avg_binding if cycle_count > 0 else 0.0
                    confidence = 0.5 + 0.3 * min(1.0, cycle_count / 10)
                    evidence = {
                        "avg_binding": avg_binding,
                        "cycles": cycle_count,
                        "module_active": True,
                    }
```

- [ ] **Step 2: Rewrite epistemic_depth formula**

```python
                elif name == "epistemic_depth":
                    depth_stats = stats.get("epistemic_depth", {})
                    avg_depth = depth_stats.get("average_depth", 0)
                    max_depth = depth_stats.get("max_depth_reached", 0)
                    score = min(1.0, avg_depth / 4.0) if avg_depth > 0 else 0.0
                    confidence = 0.6 if max_depth >= 2 else 0.3
                    evidence = {
                        "avg_depth": avg_depth,
                        "max_depth": max_depth,
                        "module_active": True,
                    }
```

- [ ] **Step 3: Rewrite genuine_implementation formula**

```python
                elif name == "genuine_implementation":
                    avg_quality = stats.get("average_loop_quality", 0.0)
                    fe_count = stats.get("field_evidencing_count", 0)
                    cycle_count = stats.get("cycle_count", 0)
                    depth_stats = stats.get("epistemic_depth", {})

                    if cycle_count == 0:
                        score = 0.0
                    else:
                        has_field_evidencing = fe_count > 0
                        has_loop_quality = avg_quality >= 0.4
                        has_strange_loop = depth_stats.get("max_depth_reached", 0) >= 2
                        conditions_met = sum([has_field_evidencing, has_loop_quality, has_strange_loop])
                        score = conditions_met / 3.0
                    confidence = 0.5 + 0.3 * min(1.0, cycle_count / 20)
                    evidence = {
                        "avg_loop_quality": avg_quality,
                        "field_evidencing_count": fe_count,
                        "has_field_evidencing": fe_count > 0,
                        "has_loop_quality": avg_quality >= 0.4,
                        "has_strange_loop": depth_stats.get("max_depth_reached", 0) >= 2,
                        "conditions_met": sum([fe_count > 0, avg_quality >= 0.4, depth_stats.get("max_depth_reached", 0) >= 2]),
                        "module_active": True,
                    }
```

- [ ] **Step 4: Commit**

```bash
git add mtc/assessment/assessment.py
git commit -m "recalibrate: BLT indicators, direct quality, conditions-based genuine_implementation"
```

---

### Task 9: Run full noise baseline and verify scores below threshold

**Files:**
- No file changes: verification only

- [ ] **Step 1: Run experiment and show all indicator scores**

Run:
```bash
python -c "
import asyncio, random, numpy as np
random.seed(42); np.random.seed(42)
from scripts.run_experiment import build_modules, warmup, WARMUP_CYCLES
from mtc.core.param_overrides import load_overrides
from mtc.assessment.assessment import ConsciousnessAssessment

async def main():
    overrides = load_overrides('experiments/params.toml', 'experiments/param_schema.toml')
    modules = build_modules(overrides)
    await warmup(modules, WARMUP_CYCLES)
    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(**modules, warmup_cycles=WARMUP_CYCLES)
    failing = []
    print(f\"{'Indicator':<40} {'Score':>6} {'Thresh':>6} {'Pass':>5}\")
    print('-' * 60)
    for name, r in sorted(report.indicator_results.items(), key=lambda x: x[1].score, reverse=True):
        status = 'YES' if r.passes_threshold else 'NO'
        print(f'{name:<40} {r.score:>6.3f} {r.threshold:>6.2f} {status:>5}')
        if r.passes_threshold:
            failing.append(name)
    print(f'\nOverall: {report.overall_score:.3f}  Passing: {len(failing)}/25')
asyncio.run(main())
" 2>&1 | grep -v "pyphi\|PyPhi\|PLOS\|Mayner\|WELCOME\|https\|suppress\|set\|export\|forum\|Documentation\|RuntimeWarning\|VisibleDeprecation\|numpy\|return _methods\|ret = ret"
```

Expected: Most indicators should score below their thresholds with neutral random data. Some may still pass if the modules genuinely produce quality output (e.g., FEP prediction errors may genuinely decrease). The key validation is that no indicator scores 1.0 on random data.

- [ ] **Step 2: Document results**

Record the actual scores. If any indicator is still scoring above threshold on random data, investigate whether that's genuine module quality or a formula that still needs adjustment.

- [ ] **Step 3: Run ablation check on key modules**

Run: `python scripts/run_experiment.py --validate gwt,ast,hot,fep,blt,rpt 2>&1 | tail -20`

Expected: Each module's ablation should show a measurable score drop. If a module's ablation causes no drop, that module's indicators aren't measuring anything real.

---

### Task 10: Update existing tests to match new score expectations

**Files:**
- Modify: `tests/consciousness/test_assessment.py`
- Modify: `tests/consciousness/test_assessment_upgrade.py`
- Modify: `tests/assessment/test_new_indicators.py`

- [ ] **Step 1: Run all existing assessment tests to see what fails**

Run: `python -m pytest tests/consciousness/test_assessment.py tests/consciousness/test_assessment_upgrade.py tests/assessment/test_new_indicators.py tests/assessment/test_causal_emergence.py -v --timeout=60 2>&1 | tail -40`

Record which tests fail and what the assertion errors are.

- [ ] **Step 2: Fix test_assessment.py assertions**

The main changes:
- Scores are still 0.0-1.0 (range checks unchanged)
- `score > 0.0` assertions may fail for indicators that now score 0.0 when modules don't have activity. Replace `> 0.0` with `>= 0.0` where the test doesn't run warmup.
- `module_active` evidence fields are still present.

- [ ] **Step 3: Fix test_assessment_upgrade.py assertions**

Key changes:
- `score ≤ 0.15` for ablated indicators: ablated indicators now score 0.0-0.1, so this assertion should still pass. Verify.
- `architecture_functional` thresholds: with lower scores, architecture may not be functional in tests that don't run proper warmup. Update test expectations.
- `BLT score > 0.0` and `RPT score > 0.0`: these may now be 0.0 if the test doesn't exercise modules. Update to `>= 0.0` or add proper warmup.
- `genuine_implementation` comparison test: scoring is now conditions-based (0/3, 1/3, 2/3, 3/3). High quality (0.7 quality + field-evidencing) should still score higher than low quality (0.3 quality + no field-evidencing).

- [ ] **Step 4: Fix test_new_indicators.py assertions**

Key changes:
- `score >= 0.0` assertions should still pass.
- Threshold value assertions (`threshold == 0.3`) are unchanged: we didn't change thresholds.
- ConTraSt weight assertions (`GWT=1.0`, `HOT=0.5`) are unchanged.

- [ ] **Step 5: Run all assessment tests and confirm green**

Run: `python -m pytest tests/consciousness/test_assessment.py tests/consciousness/test_assessment_upgrade.py tests/assessment/test_new_indicators.py tests/assessment/test_causal_emergence.py -v --timeout=60 2>&1 | tail -40`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: update assessment test expectations for quality-based scoring"
```

---

### Task 11: Create noise baseline regression test

**Files:**
- Create: `tests/assessment/test_recalibration.py`

- [ ] **Step 1: Write noise baseline test**

Create `tests/assessment/test_recalibration.py`:

```python
"""
Regression tests for assessment recalibration.

Verifies that random/neutral inputs score below threshold for all indicators,
preventing score inflation from returning.
"""

import asyncio
import random

import numpy as np
import pytest

from mtc.assessment.assessment import ConsciousnessAssessment
from scripts.run_experiment import build_modules, warmup, WARMUP_CYCLES


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _seed():
    random.seed(42)
    np.random.seed(42)


@pytest.mark.asyncio
async def test_noise_baseline_below_thresholds():
    """No indicator should pass its threshold on neutral random data.

    This is the core calibration invariant: if the experiment runner uses
    neutral synthetic data (no engineered connectivity, full-range spike
    data, naive self-model), scores should reflect the absence of genuine
    consciousness-like properties.

    Some indicators MAY pass if the modules genuinely produce quality
    output during warmup (e.g., FEP errors may decrease). If an indicator
    consistently passes on noise, its formula needs investigation.
    """
    _seed()
    overrides = {}  # No parameter overrides: pure defaults
    modules = build_modules(overrides)
    await warmup(modules, WARMUP_CYCLES)

    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        **modules, warmup_cycles=WARMUP_CYCLES
    )

    passing_on_noise = []
    for name, result in report.indicator_results.items():
        if result.passes_threshold:
            passing_on_noise.append(
                f"{name}: score={result.score:.3f} threshold={result.threshold:.2f}"
            )

    # Allow up to 5 indicators to pass on noise: modules may genuinely
    # produce some quality signal during warmup. But 13/25 (pre-calibration)
    # is a failure. If more than 5 pass, the calibration has regressed.
    max_allowed = 5
    assert len(passing_on_noise) <= max_allowed, (
        f"{len(passing_on_noise)} indicators pass on noise (max allowed: {max_allowed}):\n"
        + "\n".join(passing_on_noise)
    )


@pytest.mark.asyncio
async def test_no_perfect_scores_on_noise():
    """No indicator should score 1.0 on neutral random data."""
    _seed()
    overrides = {}
    modules = build_modules(overrides)
    await warmup(modules, WARMUP_CYCLES)

    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        **modules, warmup_cycles=WARMUP_CYCLES
    )

    perfect_scores = [
        name for name, r in report.indicator_results.items()
        if r.score >= 0.999
    ]
    assert len(perfect_scores) == 0, (
        f"Indicators scoring 1.0 on noise: {perfect_scores}"
    )


@pytest.mark.asyncio
async def test_ablated_modules_score_near_zero():
    """Ablated (None) modules should produce scores near zero."""
    _seed()
    assessment = ConsciousnessAssessment()
    # Run with NO modules at all
    report = await assessment.run_full_assessment(warmup_cycles=WARMUP_CYCLES)

    for name, result in report.indicator_results.items():
        assert result.score <= 0.15, (
            f"{name} scores {result.score:.3f} with all modules ablated (expected ≤ 0.15)"
        )


@pytest.mark.asyncio
async def test_overall_score_below_half_on_noise():
    """Overall score should be below 0.5 on neutral random data."""
    _seed()
    overrides = {}
    modules = build_modules(overrides)
    await warmup(modules, WARMUP_CYCLES)

    assessment = ConsciousnessAssessment()
    report = await assessment.run_full_assessment(
        **modules, warmup_cycles=WARMUP_CYCLES
    )

    assert report.overall_score < 0.5, (
        f"Overall score {report.overall_score:.3f} on noise (expected < 0.5)"
    )
```

- [ ] **Step 2: Run the new test**

Run: `python -m pytest tests/assessment/test_recalibration.py -v --timeout=120 2>&1 | tail -20`

Expected: All 4 tests pass. If `test_noise_baseline_below_thresholds` fails with more than 5 indicators passing, investigate which formulas still need tightening.

- [ ] **Step 3: Commit**

```bash
git add tests/assessment/test_recalibration.py
git commit -m "test: add noise baseline regression tests for calibration invariant"
```

---

### Task 12: Final verification and summary

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=120 2>&1 | tail -40`

Expected: All tests pass.

- [ ] **Step 2: Run experiment and print final scores**

Run: `python scripts/run_experiment.py 2>&1 | tail -15`

Record the final scores for the session log.

- [ ] **Step 3: Run ablation validation**

Run: `python scripts/run_experiment.py --validate gwt,ast,hot,fep,blt,rpt 2>&1 | tail -25`

Record ablation results.

- [ ] **Step 4: Commit any remaining changes**

```bash
git add -A
git commit -m "recalibrate: quality-based assessment scoring, complete"
```
