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
