# Appendix D: Complete Assessment Output Example

The following is the complete output of a single assessment run after 5 warmup cycles with synthetic stimuli. This output can be reproduced by running:

```
python scripts/run_assessment.py --full --report
```

## Session Metadata

```
Session:              8b88ce1e
Timestamp:            2026-03-14T14:22:16.335983
Overall Score:        0.518
Architecture Func.:   True
Confidence:           0.475
Passing Indicators:   15/20
Processing Time:      0.29 ms
```

## Theory Scores

| Theory | Score |
|--------|-------|
| FEP | 0.864 |
| GWT | 0.677 |
| HOT | 0.450 |
| AST | 0.417 |
| BLT | 0.300 |
| IIT | 0.250 |
| RPT | 0.100 |

## Per-Indicator Results

| Status | Indicator | Score | Threshold | Confidence | Time (ms) |
|--------|-----------|-------|-----------|------------|-----------|
| PASS | global_broadcast | 1.000 | 0.50 | 0.500 | 0.06 |
| PASS | hierarchical_prediction | 1.000 | 0.40 | 0.400 | 0.01 |
| PASS | prediction_error_minimization | 0.907 | 0.40 | 0.600 | 0.02 |
| PASS | sparse_smooth_coding | 0.860 | 0.30 | 0.500 | 0.01 |
| PASS | agency | 0.690 | 0.50 | 0.400 | 0.01 |
| PASS | global_ignition_nuanced | 0.688 | 0.40 | 0.400 | 0.03 |
| PASS | ignition_dynamics | 0.600 | 0.40 | 0.500 | 0.04 |
| PASS | recurrent_processing | 0.600 | 0.40 | 0.600 | 0.00 |
| PASS | attention_control | 0.500 | 0.40 | 0.400 | 0.00 |
| PASS | higher_order_representations | 0.500 | 0.50 | 0.500 | 0.00 |
| PASS | local_recurrence | 0.500 | 0.30 | 0.500 | 0.00 |
| PASS | embodiment | 0.500 | 0.30 | 0.500 | 0.00 |
| PASS | metacognition | 0.400 | 0.40 | 0.400 | 0.00 |
| PASS | bayesian_binding_quality | 0.300 | 0.30 | 0.500 | 0.01 |
| PASS | epistemic_depth | 0.300 | 0.30 | 0.400 | 0.00 |
| FAIL | irreducibility | 0.300 | 0.40 | 0.500 | 0.00 |
| FAIL | genuine_implementation | 0.300 | 0.40 | 0.500 | 0.00 |
| FAIL | attention_schema | 0.250 | 0.50 | 0.500 | 0.01 |
| FAIL | integrated_information | 0.200 | 0.30 | 0.600 | 0.00 |
| FAIL | algorithmic_recurrence | 0.100 | 0.30 | 0.300 | 0.00 |

## Interpretation Notes

- **Total processing time under 1 ms.** All 20 indicators are computed in a single pass. Assessment does not bottleneck the consciousness cycle.
- **FEP and GWT score highest** because their architectural features (hierarchical prediction, workspace competition, broadcast) are structural — they function from the moment the modules are initialized.
- **IIT and RPT score lowest** because they require neural substrate connectivity data and active recurrent dynamics that are not fully present after only 5 warmup cycles.
- **Three BLT indicators cluster at 0.300.** Bayesian binding and epistemic depth pass at their thresholds; genuine implementation fails because field-evidencing requires sustained recursive loops not yet established.
- **Architecture functional = True** because all four conditions are met: 15/20 pass (≥ 10), overall 0.518 (≥ 0.5), GWT 0.677 (≥ 0.4), HOT 0.450 (≥ 0.3).

## Reproducing This Output

```bash
# Clone repository
git clone https://github.com/WhiteLotusLA/multi-theory-consciousness.git
cd multi-theory-consciousness

# Install dependencies
pip install -e .

# Run assessment
python scripts/run_assessment.py --full --report
```

Results will vary slightly from the above due to random synthetic stimuli during warmup. Cold-start results (without `--full`) are perfectly deterministic.
