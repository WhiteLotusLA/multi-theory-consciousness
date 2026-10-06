# Mapping to the Butlin et al. indicator properties

The assessment's indicators follow the indicator-property method of Butlin
et al. (2023), updated in 2025:

> Butlin, P., Long, R., Elmoznino, E., et al. (2023). "Consciousness in
> Artificial Intelligence: Insights from the Science of Consciousness."
> arXiv:2308.08708.
>
> Butlin, P., Long, R., et al. (2025). "Consciousness in Artificial
> Intelligence: A Taxonomy and Implications." *Trends in Cognitive Sciences*.

Every row below is read from `ConsciousnessAssessment().indicator_configs`
and `negative_indicator_configs` in `mtc/assessment/assessment.py`, and a test
(`tests/docs/test_butlin_doc.py`) compares this table with the code. The
paper's id column holds the `butlin_id` the code records; `extension` marks an
indicator the paper does not list, which this framework adds and labels as its
own. The theory column uses the code's `ConsciousnessTheory` names; FEP stands
for the paper's predictive processing (PP) group, and IIT and BLT are outside
the paper's indicator list.

The "needs" column names what the indicator reads before it reports
anything other than `insufficient_data` (the module must also be supplied, or
the indicator is `absent`). [FIRST_RUN.md](FIRST_RUN.md) gives the warmup
thresholds with their source constants.

## Positive indicators (30)

| Indicator | Theory | Paper id | Polarity | What MTC computes | Needs |
|---|---|---|---|---|---|
| `global_broadcast` | GWT | GWT-3 | positive | Broadcast coverage of the workspace across registered modules | workspace cycles and a broadcast coverage ratio |
| `ignition_dynamics` | GWT | GWT-2 | positive | Non-linear amplification when the workspace threshold is crossed | workspace cycles and the ignition sustain rate |
| `global_ignition_nuanced` | GWT | GWT-3 | positive | Ignition with its sustain rate (an extension of GWT-3) | workspace cycles and the ignition sustain rate |
| `sequential_module_querying` | GWT | GWT-4 | positive | Successive competitions over many candidates from the bottleneck's statistics | workspace cycles and bottleneck statistics |
| `flexible_responsiveness` | GWT | AE-1 | positive | Diversity of winning sources when conditions change | workspace cycles and bottleneck statistics |
| `attention_schema` | AST | AST-1 | positive | Accuracy of the attention schema's predictions of its own focus | schema updates and scored attention predictions |
| `attention_control` | AST | AST-1 | positive | Voluntary ratio of typed attention shifts (a sub-aspect of AST-1) | schema updates and typed shifts |
| `higher_order_representations` | HOT | HOT-1 | positive | Higher-order thoughts generated about workspace content | workspace cycles and higher-order thoughts |
| `metacognition` | HOT | HOT-2 | positive | Metacognitive monitoring with a measured overall confidence | higher-order thoughts and a confidence reading |
| `sparse_smooth_coding` | HOT | HOT-4 | positive | Sparse and smooth predictive coding from inference statistics | active inference (the `fep` extra) with prediction errors |
| `online_reportable_learning` | HOT | extension | positive | Learning during operation that metacognition can report | metacognition or active inference activity |
| `prediction_error_minimization` | FEP | PP-1 | positive | Least-squares trend of the last prediction errors plus the error level | active inference with 10 prediction errors |
| `hierarchical_prediction` | FEP | PP-1 | positive | Hierarchical prediction error (a deeper aspect of PP-1) | active inference with a hierarchical error |
| `temporal_prediction_depth` | FEP | extension | positive | Planning horizon and belief and free-energy history depth (extends PP-1) | active inference with prediction errors |
| `agent_environment_coupling` | FEP | extension | positive | Bidirectional coupling of action and observation (bridges PP-1 and AE-2) | active inference with inferences |
| `affect_coherence` | FEP | extension | positive | Mean lag-1 autocorrelation of the PAD dimensions over 20 states | a PAD affect model with 10 states |
| `complex_agency` | FEP | AE-1 | positive | Navigation of competing policies (AE-1 expanded) | active inference with policies |
| `integrated_information` | IIT | extension | positive | The PhiTracker's module-graph integration proxy, else the PhiCalculator over neural states | a warm PhiTracker (128 transitions; exact Phi needs the `iit` extra) or neural states |
| `irreducibility` | IIT | extension | positive | Algebraic connectivity of the weighted connectivity (weighted Fiedler) | neural states with a connectivity matrix |
| `causal_emergence` | IIT | extension | positive | Macro against micro effective information (Hoel) | a causal emergence analyzer and a warm PhiTracker |
| `recurrent_processing` | RPT | RPT-1 | positive | Broadcast coverage blended with RPT global recurrence when measured | workspace cycles |
| `algorithmic_recurrence` | RPT | RPT-1 | positive | Algorithmic recurrence in the substrates (an implementation detail of RPT-1) | RPT measurements |
| `local_recurrence` | RPT | RPT-2 | positive | Local recurrent amplification before global access | RPT measurements |
| `organized_percepts` | RPT | RPT-2 | positive | Organized, integrated perceptual representations | workspace bottleneck statistics or RPT measurements |
| `oscillatory_binding_coherence` | RPT | extension | positive | Kuramoto phase synchronization across substrates | RPT measurements and oscillatory binding statistics |
| `agency` | AE | AE-1 | positive | Inference activity, model updates and free energy, with no presence floor | active inference with inferences |
| `embodiment` | AE | AE-2 | positive | Output-input contingencies from the attention schema's perspective statistics | schema updates |
| `bayesian_binding_quality` | BLT | extension | positive | Binding quality of inferences into a unified percept | a beautiful loop that has run a conscious moment |
| `epistemic_depth` | BLT | extension | positive | Recursive self-reference depth (keyword-circular; see the limitations) | a beautiful loop that has run a conscious moment |
| `genuine_implementation` | BLT | extension | positive | Anti-mimicry check (addresses the paper's gaming problem) | a beautiful loop that has run a conscious moment |

## Negative indicators (3)

Each negative indicator above its threshold (0.3 for all three) subtracts its
score times its weight times 0.05 from the overall score. None carries a paper
id in the code. A module the caller did not supply keeps the v0.4.0 rule level
(`absent`, which still counts against the score); a supplied module with no
data yet is `insufficient_data` and applies no penalty.

| Indicator | Theory | Paper id | Polarity | What MTC computes | Needs |
|---|---|---|---|---|---|
| `purely_reactive` | GWT | extension | negative | 0.0 when the workspace has cycled and ignited, 0.5 when it cycled without ignition | workspace cycles |
| `fixed_repertoire` | AE | extension | negative | 0.0 when active inference has inferred and updated its model, 0.4 otherwise | active inference with inferences |
| `no_metacognitive_access` | HOT | extension | negative | 0.0 when higher-order thoughts were generated, 0.3 otherwise | metacognition activity |

## Framework extensions

These indicators carry no paper id and are this framework's additions, each
labelled `extension` above: `online_reportable_learning`,
`temporal_prediction_depth`, `agent_environment_coupling`, `affect_coherence`,
`integrated_information`, `irreducibility`, `causal_emergence`,
`oscillatory_binding_coherence`, `bayesian_binding_quality`,
`epistemic_depth` and `genuine_implementation`, plus the three negative
indicators. IIT and the beautiful loop are outside the paper's indicator list
entirely; the others extend or bridge a listed property.

The specificity and sensitivity values each indicator carries (served by
`GET /api/meta`) are expert estimates informed by the 2025 paper's discussion,
not measurements.
