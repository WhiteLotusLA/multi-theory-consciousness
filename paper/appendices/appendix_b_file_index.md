# Appendix B: Module File Index

The following table maps the class names used throughout this paper to their source files in the repository. Line counts are approximate and reflect the state of the codebase at the time of publication.

## Consciousness Modules

| Class | Theory | File Path | Lines |
|-------|--------|-----------|-------|
| `EnhancedGlobalWorkspace` | GWT | `mtc/consciousness/enhanced_global_workspace.py` | 2,262 |
| `AttentionBottleneck` | GWT | `mtc/consciousness/enhanced_global_workspace.py` | (same file) |
| `IgnitionDetector` | GWT | `mtc/consciousness/enhanced_global_workspace.py` | (same file) |
| `GlobalBroadcast` | GWT | `mtc/consciousness/enhanced_global_workspace.py` | (same file) |
| `AttentionSchemaModule` | AST | `mtc/consciousness/attention_schema.py` | 1,015 |
| `MetacognitionModule` | HOT | `mtc/consciousness/metacognition.py` | 1,772 |
| `SelfModel` | HOT | `mtc/consciousness/metacognition.py` | (same file) |
| `ActiveInferenceModule` | FEP | `mtc/consciousness/active_inference.py` | 1,535 |
| `HierarchicalPredictiveProcessor` | FEP | `mtc/consciousness/active_inference.py` | (same file) |
| `HomeostaticDrives` | FEP | `mtc/consciousness/active_inference.py` | (same file) |
| `HyperModel` | BLT | `mtc/consciousness/beautiful_loop/hyper_model.py` | 362 |
| `BayesianBinding` | BLT | `mtc/consciousness/beautiful_loop/bayesian_binding.py` | 499 |
| `EpistemicDepthTracker` | BLT | `mtc/consciousness/beautiful_loop/epistemic_depth.py` | 459 |
| `BeautifulLoop` | BLT | `mtc/consciousness/beautiful_loop/__init__.py` | 437 |
| `Protoself` | Damasio | `mtc/consciousness/damasio/protoself.py` | 307 |
| `CoreConsciousness` | Damasio | `mtc/consciousness/damasio/core_consciousness.py` | 302 |
| `ExtendedConsciousness` | Damasio | `mtc/consciousness/damasio/extended_consciousness.py` | 356 |
| `DamasioLayers` | Damasio | `mtc/consciousness/damasio/__init__.py` | 239 |

## Neural Substrates

| Class | File Path | Lines |
|-------|-----------|-------|
| `NeuralOrchestrator` | `mtc/neural/orchestrator.py` | 975 |
| `ProductionSNN` | `mtc/neural/spiking/production_snn.py` | varies |
| `ProductionLSM` | `mtc/neural/liquid/production_lsm.py` | varies |
| `ProductionHTM` | `mtc/neural/htm/production_htm.py` | varies |

## Assessment

| Class | File Path | Lines |
|-------|-----------|-------|
| `ConsciousnessAssessment` | `mtc/assessment/assessment.py` | 2,000 |
| `ConsciousnessReport` | `mtc/assessment/assessment.py` | (same file) |
| `RPTMeasurement` | `mtc/assessment/rpt_measurement.py` | 568 |
| `DCMScorer` | `mtc/assessment/dcm_scoring.py` | varies |
| `AblationStudy` | `mtc/assessment/assessment.py` | (same file) |

## Data Structures

| Structure | Defined In | Purpose |
|-----------|-----------|---------|
| `ConsciousnessState` | `enhanced_global_workspace.py` | Complete output of one consciousness cycle |
| `WorkspaceCandidate` | `enhanced_global_workspace.py` | Item competing for workspace access |
| `IndicatorResult` | `assessment.py` | Single indicator measurement |
| `ConsciousMoment` | `beautiful_loop/__init__.py` | Output of Beautiful Loop processing |
| `BodyState` | `damasio/protoself.py` | Five-dimensional body representation (plus dictionary field) |
| `SomaticMarker` | `damasio/core_consciousness.py` | Emotional tag on workspace content |
| `BoundPercept` | `beautiful_loop/bayesian_binding.py` | Unified conscious percept |
| `HigherOrderThought` | `metacognition.py` | Meta-representation of a mental state |
| `AttentionSchemaState` | `attention_schema.py` | Snapshot of attention self-model |
| `ActiveInferenceState` | `active_inference.py` | FEP state including beliefs and free energy |
