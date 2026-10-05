"""The integration module must import on a base install."""

import importlib


def test_consciousness_integration_imports():
    module = importlib.import_module("mtc.consciousness.consciousness_integration")
    assert hasattr(module, "ConsciousnessIntegration")
    assert hasattr(module, "ConsciousNeuralOrchestrator")


def test_integration_tolerates_missing_mongo_schemas():
    module = importlib.import_module("mtc.consciousness.consciousness_integration")
    # NeuralDataManager is optional until the Mongo layer lands; the name must
    # exist (possibly as None) so type hints and constructors keep working.
    assert hasattr(module, "NeuralDataManager")


def test_integration_binds_the_real_orchestrator_and_legacy_metrics():
    module = importlib.import_module("mtc.consciousness.consciousness_integration")
    orchestrator_mod = importlib.import_module("mtc.neural.orchestrator")
    metrics_mod = importlib.import_module("mtc.consciousness.consciousness_metrics")
    assert module.NeuralOrchestrator is orchestrator_mod.NeuralOrchestrator
    assert issubclass(
        module.ConsciousNeuralOrchestrator, orchestrator_mod.NeuralOrchestrator
    )
    assert module.ConsciousnessMetrics is metrics_mod.ConsciousnessMetrics
    assert len(metrics_mod.ConsciousnessMetrics().indicators) == 14
