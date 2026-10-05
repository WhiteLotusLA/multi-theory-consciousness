"""
Loop closure in the enhanced global workspace.

Workspace winners feed the active inference precision bias, the discrete
agent observes the content-derived vector, the self-model predicts each
cycle and is scored against itself afterwards, the protoself receives real
body metrics, and the observation encoder follows the world model.
"""

import numpy as np
import pytest

import mtc.consciousness.enhanced_global_workspace as egw
from mtc.consciousness.active_inference import ActiveInferenceConfig
from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    TemporalWorkspaceConfig,
    WorkspaceCandidate,
    WorkspaceCandidateSource,
)

FAST_FEP = dict(num_hidden_states=4, num_actions=2, planning_horizon=2)


def _workspace(fep=None, **kwargs) -> EnhancedGlobalWorkspace:
    kwargs.setdefault(
        "active_inference_config", ActiveInferenceConfig(**{**FAST_FEP, **(fep or {})})
    )
    kwargs.setdefault(
        "temporal_config", TemporalWorkspaceConfig(refractory_period_ms=0.0)
    )
    return EnhancedGlobalWorkspace(**kwargs)


def _candidate(summary: str, content_type: str = "emotion", activation: float = 0.9):
    return WorkspaceCandidate(
        content=summary,
        content_type=content_type,
        summary=summary,
        source=WorkspaceCandidateSource.SENSORY,
        source_module="synthetic",
        activation_level=activation,
        emotional_salience=0.5,
    )


# ============================================================================
# Workspace to FEP precision feedback
# ============================================================================


@pytest.mark.asyncio
async def test_winners_raise_mapped_channel_precision():
    workspace = _workspace()
    module = workspace.active_inference
    channel = module.channel_for_content_type("emotion")
    assert channel is not None

    state = await workspace.process_consciousness_cycle([_candidate("a warm feeling")])
    assert len(state.workspace_contents) == 1

    bias = module.workspace_precision_bias
    expected = 0.9 * module.WORKSPACE_BIAS_GAIN
    assert bias[channel] == pytest.approx(expected)
    others = np.delete(bias, channel)
    assert np.all(others == 0.0)


@pytest.mark.asyncio
async def test_bias_decays_without_winners():
    workspace = _workspace()
    module = workspace.active_inference
    channel = module.channel_for_content_type("emotion")
    await workspace.process_consciousness_cycle([_candidate("a warm feeling")])
    before = module.workspace_precision_bias[channel]
    assert before > 0.0

    await workspace.process_consciousness_cycle([])
    after = module.workspace_precision_bias[channel]
    assert after == pytest.approx(before * module.WORKSPACE_BIAS_DECAY)


@pytest.mark.asyncio
async def test_precision_feedback_switch_off_leaves_bias_untouched():
    workspace = _workspace(precision_feedback_enabled=False)
    await workspace.process_consciousness_cycle([_candidate("a warm feeling")])
    assert np.all(workspace.active_inference.workspace_precision_bias == 0.0)


# ============================================================================
# Observation reconnect
# ============================================================================


async def _capture_observations(workspace, monkeypatch, candidates):
    captured = {}
    encode = workspace._workspace_to_observation
    infer = workspace.active_inference.infer_and_act
    discrete = workspace.active_inference.observation_for_discrete_inference
    calls = {"discrete": 0}

    async def spy_encode(*args, **kwargs):
        captured["raw"] = await encode(*args, **kwargs)
        return captured["raw"]

    async def spy_infer(observation, attention_precision=None):
        captured["observed"] = np.array(observation, copy=True)
        captured["precision"] = attention_precision
        return await infer(observation, attention_precision=attention_precision)

    def spy_discrete(vector):
        calls["discrete"] += 1
        return discrete(vector)

    monkeypatch.setattr(workspace, "_workspace_to_observation", spy_encode)
    monkeypatch.setattr(workspace.active_inference, "infer_and_act", spy_infer)
    monkeypatch.setattr(
        workspace.active_inference, "observation_for_discrete_inference", spy_discrete
    )
    await workspace.process_consciousness_cycle(candidates)
    return captured, calls


@pytest.mark.asyncio
async def test_discrete_agent_observes_content_vector(monkeypatch):
    workspace = _workspace()
    captured, calls = await _capture_observations(
        workspace, monkeypatch, [_candidate("a warm feeling")]
    )
    assert calls["discrete"] == 1
    np.testing.assert_allclose(captured["observed"], captured["raw"])


@pytest.mark.asyncio
async def test_legacy_route_observes_hierarchy_top_level(monkeypatch):
    workspace = _workspace(fep={"hierarchy_observation": "hpp_top_level"})
    captured, calls = await _capture_observations(
        workspace, monkeypatch, [_candidate("a warm feeling")]
    )
    assert calls["discrete"] == 0
    assert len(captured["observed"]) == workspace.active_inference.num_observations


# ============================================================================
# Encoder selection
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fep, channels",
    [
        ({}, 5),
        (
            {
                "world_model": "conversational_16x10x8",
                "num_hidden_states": None,
                "num_actions": None,
                "planning_horizon": 1,
            },
            10,
        ),
    ],
)
async def test_observation_length_matches_world_model(monkeypatch, fep, channels):
    workspace = _workspace(fep=fep)
    assert workspace.active_inference.num_observations == channels
    captured, _ = await _capture_observations(
        workspace,
        monkeypatch,
        [_candidate("a warm feeling"), _candidate("a passing thought", "thought")],
    )
    raw = captured["raw"]
    assert len(raw) == channels
    assert raw.sum() == pytest.approx(1.0)
    assert np.all(raw >= 0.0)
    if captured["precision"] is not None:
        assert len(captured["precision"]) == channels


@pytest.mark.asyncio
async def test_conversational_encoder_reads_tone_and_intent():
    workspace = _workspace(
        fep={
            "world_model": "conversational_16x10x8",
            "num_hidden_states": None,
            "num_actions": None,
            "planning_horizon": 1,
        }
    )
    index = workspace.active_inference.observation_index
    winners = [
        egw.WorkspaceContent(candidate=_candidate("x", "emotion"), salience=0.6),
        egw.WorkspaceContent(candidate=_candidate("y", "emotion"), salience=0.6),
    ]
    warm = await workspace._workspace_to_observation(
        winners, winners, 0.5, {"joy": 0.8}
    )
    tense = await workspace._workspace_to_observation(
        winners, winners, 0.5, {"fear": 0.8}
    )
    assert warm[index["warm_tone"]] > tense[index["warm_tone"]]
    assert tense[index["tense_tone"]] > warm[index["tense_tone"]]
    assert warm[index["emote_intent"]] > warm[index["inform_intent"]]


# ============================================================================
# HOT strange loop
# ============================================================================


@pytest.mark.asyncio
async def test_self_prediction_scored_after_self_model_update(monkeypatch):
    workspace = _workspace()
    meta = workspace.metacognition
    order = []
    predict = meta.predict_self
    update = meta.update_self_model
    score = meta.score_prediction_from_state
    predictions = []

    def spy_predict(context):
        order.append("predict")
        prediction = predict(context)
        predictions.append(prediction)
        return prediction

    def spy_update(**kwargs):
        order.append("update")
        return update(**kwargs)

    def spy_score(prediction_id):
        order.append(("score", prediction_id))
        return score(prediction_id)

    monkeypatch.setattr(meta, "predict_self", spy_predict)
    monkeypatch.setattr(meta, "update_self_model", spy_update)
    monkeypatch.setattr(meta, "score_prediction_from_state", spy_score)

    await workspace.process_consciousness_cycle([_candidate("a warm feeling")])
    assert order == ["predict", "update", ("score", predictions[0].id)]


@pytest.mark.asyncio
async def test_prediction_accuracy_history_fills():
    workspace = _workspace()
    sm = workspace.metacognition.self_model
    # A self-model that commits to checkable claims about the next cycle
    sm.curiosity_level = 0.9
    sm.social_need = 0.2
    for i in range(6):
        await workspace.process_consciousness_cycle([_candidate(f"feeling {i}")])
    scored = [p for p in sm.predictions if p.accurate is not None]
    assert len(scored) >= 3
    assert len(sm.prediction_accuracy_history) > 0


@pytest.mark.asyncio
async def test_self_prediction_switch_off(monkeypatch):
    workspace = _workspace(self_prediction_enabled=False)
    calls = []
    monkeypatch.setattr(
        workspace.metacognition, "predict_self", lambda context: calls.append(context)
    )
    await workspace.process_consciousness_cycle([_candidate("a warm feeling")])
    assert calls == []
    assert workspace.metacognition.self_model.predictions == []


# ============================================================================
# Body grounding
# ============================================================================


@pytest.mark.asyncio
async def test_body_metrics_reach_the_protoself(monkeypatch):
    latencies = []

    def fake_metrics(cycle_latency_ms=None):
        latencies.append(cycle_latency_ms)
        metrics = {"memory_usage_pct": 0.4}
        if cycle_latency_ms is not None:
            metrics["response_latency_ms"] = cycle_latency_ms
        return metrics

    monkeypatch.setattr(egw, "gather_system_body_metrics", fake_metrics)
    workspace = _workspace()
    received = []
    process = workspace.damasio.process

    async def spy_process(*args, **kwargs):
        received.append(kwargs.get("system_metrics"))
        return await process(*args, **kwargs)

    monkeypatch.setattr(workspace.damasio, "process", spy_process)

    await workspace.process_consciousness_cycle([_candidate("first")])
    await workspace.process_consciousness_cycle([_candidate("second")])

    # The first cycle has no previous duration; the second sees the first
    assert latencies[0] is None
    assert latencies[1] is not None and latencies[1] > 0.0
    assert received[0] == {"memory_usage_pct": 0.4}
    assert received[1]["response_latency_ms"] == latencies[1]


@pytest.mark.asyncio
async def test_body_metrics_switch_off(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("body metrics must not be sampled")

    monkeypatch.setattr(egw, "gather_system_body_metrics", fail)
    workspace = _workspace(body_metrics_enabled=False)
    received = []
    process = workspace.damasio.process

    async def spy_process(*args, **kwargs):
        received.append(kwargs.get("system_metrics"))
        return await process(*args, **kwargs)

    monkeypatch.setattr(workspace.damasio, "process", spy_process)
    await workspace.process_consciousness_cycle([_candidate("first")])
    assert received == [None]


# ============================================================================
# Construction switches for the workspace-owned modules
# ============================================================================


def test_default_construction_matches_previous_geometry():
    workspace = EnhancedGlobalWorkspace()
    config = workspace.active_inference.config
    assert (
        config.num_hidden_states,
        config.num_observations,
        config.num_actions,
        config.planning_horizon,
    ) == (8, 5, 4, 5)
    assert config.enable_homeostasis is True
    assert workspace.metacognition.hot_threshold == 0.3
    assert workspace.beautiful_loop.field_evidencing_threshold == 0.4


def test_module_kwargs_reach_the_workspace_modules():
    workspace = EnhancedGlobalWorkspace(
        active_inference_config=EnhancedGlobalWorkspace.default_active_inference_config(
            meta_accuracy_method="median"
        ),
        beautiful_loop_kwargs={"loop_quality_formula": "v04"},
        damasio_kwargs={"circadian": "v04_sawtooth"},
        metacognition_kwargs={"max_hot_buffer": 7},
    )
    assert workspace.active_inference.config.meta_accuracy_method == "median"
    assert workspace.active_inference.config.num_hidden_states == 8
    assert workspace.beautiful_loop.loop_quality_formula == "v04"
    assert workspace.beautiful_loop.field_evidencing_threshold == 0.4
    assert workspace.damasio.protoself.circadian == "v04_sawtooth"
    assert workspace.metacognition.max_hots == 7
    assert workspace.metacognition.hot_threshold == 0.3


def test_default_config_helper_keeps_conversational_dimensions():
    config = EnhancedGlobalWorkspace.default_active_inference_config(
        world_model="conversational_16x10x8"
    )
    assert config.num_observations == 10
