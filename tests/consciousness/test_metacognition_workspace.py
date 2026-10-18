"""Metacognition as the workspace drives it.

A default EnhancedGlobalWorkspace runs a few cycles on synthetic candidates
and the tests read the metacognition module and its self-model afterwards.
The workspace module is imported read-only; nothing here changes it.
"""

import pytest

from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    TemporalWorkspaceConfig,
    WorkspaceCandidateSource,
)


def _workspace() -> EnhancedGlobalWorkspace:
    """A small default workspace whose winners are not held back by the
    refractory period, so every cycle admits its candidates."""
    return EnhancedGlobalWorkspace(
        capacity=5,
        ignition_threshold=0.3,
        temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0),
    )


async def _run_cycle(workspace: EnhancedGlobalWorkspace, index: int):
    candidate = await workspace.submit_candidate(
        content=f"synthetic thought {index}",
        content_type="thought",
        summary=f"Synthetic thought number {index}",
        source=WorkspaceCandidateSource.CTM,
        activation_level=0.9,
        emotional_salience=0.6,
    )
    return await workspace.process_consciousness_cycle(candidates=[candidate])


@pytest.mark.asyncio
async def test_two_cycles_keep_the_self_model_predictive_fields_numeric():
    workspace = _workspace()
    sm = workspace.metacognition.self_model

    await _run_cycle(workspace, 0)
    assert isinstance(sm.prediction_accuracy, float)
    assert isinstance(sm.model_confidence, float)

    await _run_cycle(workspace, 1)
    assert isinstance(sm.prediction_accuracy, float)
    assert isinstance(sm.model_confidence, float)
    assert sm.update_count == 2


def _buffered_hot_mean(workspace: EnhancedGlobalWorkspace) -> float:
    confidences = [h.confidence for h in workspace.metacognition.higher_order_thoughts]
    return sum(confidences) / len(confidences)


@pytest.mark.asyncio
async def test_self_model_confidence_follows_the_measured_hot_mean():
    workspace = _workspace()
    assert workspace.metacognition.get_statistics()["overall_confidence"] is None

    state = await _run_cycle(workspace, 0)
    assert state.higher_order_thoughts
    expected = _buffered_hot_mean(workspace)
    assert workspace.metacognition.get_statistics()[
        "overall_confidence"
    ] == pytest.approx(expected)
    assert workspace.metacognition.self_model.confidence == pytest.approx(expected)
    assert expected != pytest.approx(0.5)

    await _run_cycle(workspace, 1)
    assert workspace.metacognition.self_model.confidence == pytest.approx(
        _buffered_hot_mean(workspace)
    )


@pytest.mark.asyncio
async def test_v04_constant_source_keeps_the_self_model_confidence_at_half():
    workspace = EnhancedGlobalWorkspace(
        capacity=5,
        ignition_threshold=0.3,
        temporal_config=TemporalWorkspaceConfig(refractory_period_ms=0.0),
        metacognition_kwargs={"overall_confidence_source": "v04_constant"},
    )
    state = await _run_cycle(workspace, 0)
    assert state.higher_order_thoughts
    assert workspace.metacognition.get_statistics()["overall_confidence"] == 0.5
    assert workspace.metacognition.get_statistics()["doubt_level"] == 0.0
    assert workspace.metacognition.self_model.confidence == 0.5
