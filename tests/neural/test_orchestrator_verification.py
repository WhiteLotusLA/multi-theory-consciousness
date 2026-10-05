"""The orchestrator's verification reads only what the latest experience
produced: with no substrate output every test is insufficient_data (with or
without conversation history), a block is real only when its substrate
produced output, and the call never raises."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import mtc.neural.orchestrator as orch_mod
from mtc.consciousness.verification_framework import (
    ConsciousnessVerifier,
    VerificationSuiteResult,
)
from mtc.neural.orchestrator import NeuralOrchestrator


def _run(coro):
    return asyncio.run(coro)


def test_verification_component_available():
    assert orch_mod.OPTIONAL_COMPONENTS["verification_framework"] == (
        "mtc.consciousness.verification_framework",
        None,
    )
    orchestrator = NeuralOrchestrator(enable_mongodb=False)
    assert orchestrator.available_components()["verification_framework"] is True
    assert isinstance(orchestrator.consciousness_verifier, ConsciousnessVerifier)


def test_verify_with_absent_substrates_is_insufficient():
    orchestrator = NeuralOrchestrator(enable_mongodb=False)
    result = _run(orchestrator.verify_consciousness())
    assert isinstance(result, VerificationSuiteResult)
    assert result.overall_score is None
    assert result.status == "insufficient_data"
    assert result.measured == []
    assert all(r.insufficient_data for r in result.results.values())

    # After an experience with no substrate installed, nothing is real either.
    orchestrator.process_experience("conversation", {"joy": 0.6}, context="hello")
    result = _run(orchestrator.verify_consciousness())
    assert result.overall_score is None
    assert all(r.insufficient_data for r in result.results.values())


def test_verify_with_history_and_absent_substrates_is_insufficient():
    # Conversation turns alone are not evidence: a reply that repeats the
    # user's words must not produce a coherence score without a substrate.
    orchestrator = NeuralOrchestrator(enable_mongodb=False)
    history = [
        {"role": "user", "content": "tell me about cats"},
        {"role": "assistant", "content": "cats are about cats"},
    ]
    for pad_state in (None, {"pleasure": 0.2, "arousal": 0.45}):
        result = _run(
            orchestrator.verify_consciousness(
                conversation_history=history, pad_state=pad_state
            )
        )
        assert isinstance(result, VerificationSuiteResult)
        assert result.overall_score is None
        assert result.status == "insufficient_data"
        assert result.measured == []
        assert all(r.insufficient_data for r in result.results.values())
        assert result.results["coherence"].components == {}

    orchestrator.process_experience("conversation", {"joy": 0.6}, context="hello")
    result = _run(orchestrator.verify_consciousness(conversation_history=history))
    assert result.overall_score is None
    assert all(r.insufficient_data for r in result.results.values())


def test_verify_marks_only_produced_outputs_real():
    orchestrator = NeuralOrchestrator(enable_mongodb=False)
    subconscious = SimpleNamespace(
        emotional_coherence=0.7,
        intuitive_valence=0.2,
        intuitive_arousal=-0.1,
        state_trajectory=np.linspace(0.0, 1.0, 8),
    )
    orchestrator.state.current_experience = SimpleNamespace(
        snn_result={"metrics": {"synchrony": 0.9}}, subconscious_state=subconscious
    )
    # The SNN produced output this experience; the LSM did not (no stashed
    # trajectory), so its coherence on the stale state is not read.
    orchestrator.latest_substrate_states = {
        "snn_state": {"metrics": {"synchrony": 0.9}}
    }
    orchestrator.latest_neural_signals = {"snn": np.array([0.1, 0.4, 0.2])}

    result = _run(orchestrator.verify_consciousness(conversation_history=None))
    coherence = result.results["coherence"]
    assert coherence.components == {"snn": pytest.approx(0.9)}
    assert result.measured == ["coherence"]

    orchestrator.latest_neural_signals["lsm"] = np.linspace(0.0, 1.0, 8)
    result = _run(
        orchestrator.verify_consciousness(
            conversation_history=[
                {"role": "user", "content": "one two"},
                {"role": "assistant", "content": "one two"},
            ],
            pad_state={"pleasure": 0.2, "arousal": 0.45},
        )
    )
    coherence = result.results["coherence"]
    assert set(coherence.components) == {"snn", "lsm", "emotional", "contextual"}
    assert "emotional_authenticity" in result.measured


def test_verify_never_raises(monkeypatch):
    orchestrator = NeuralOrchestrator(enable_mongodb=False)

    def boom(*args, **kwargs):
        raise RuntimeError("suite exploded")

    monkeypatch.setattr(
        orchestrator.consciousness_verifier, "run_verification_suite", boom
    )
    assert _run(orchestrator.verify_consciousness()) is None

    orchestrator.latest_substrate_states = {"snn_state": "not a dict"}
    orchestrator.latest_neural_signals = {"snn": "not an array"}
    orchestrator.state.current_experience = object()
    monkeypatch.undo()
    result = _run(orchestrator.verify_consciousness())
    assert result is not None and result.overall_score is None

    orchestrator.consciousness_verifier = None
    assert _run(orchestrator.verify_consciousness()) is None
