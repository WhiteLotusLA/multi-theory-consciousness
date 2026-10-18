"""The contract keys the legacy GWT and IIT indicators read are built from real
sources only: substrate signals, attention-schema counters, published
retrievals, real components and binding clusters mapped onto components."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from mtc.consciousness.consciousness_integration import ConsciousnessIntegration
from mtc.consciousness.consciousness_metrics import (
    GlobalWorkspaceIntegration,
    IntegratedInformationPhi,
    MissingSignal,
)

# Oscillator indices by binding group: snn 0-7, lsm 8-15, htm 16-23, cross 24-29.


def _integration(orchestrator) -> ConsciousnessIntegration:
    return ConsciousnessIntegration(
        neural_orchestrator=orchestrator, enable_persistence=False
    )


def _workspace(latencies=(), on_state=True):
    counters = SimpleNamespace(
        voluntary_shifts=3, captured_shifts=2, uncertain_shifts=1
    )
    schema = SimpleNamespace(schema=counters) if on_state else counters
    log = [SimpleNamespace(latency_ms=value) for value in latencies]
    return SimpleNamespace(
        attention_schema=schema, broadcaster=SimpleNamespace(broadcast_log=log)
    )


def _orchestrator(**overrides):
    values = {
        "latest_neural_signals": {"snn": np.array([0.1, 0.4, 0.2, 0.7])},
        "enhanced_workspace": _workspace(latencies=(10.0, 30.0)),
        "_last_retrieved_memories": ["first memory", "second memory"],
        "_last_binding_result": SimpleNamespace(
            clusters=[[0, 9], [17, 20], [2, 18, 25]]
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _blocks(snn=True, lsm=True, htm=True, ctm=False):
    return {
        "snn": {"is_real": snn},
        "lsm": {"is_real": lsm},
        "htm": {"is_real": htm},
        "thoughts": {"is_real": ctm},
    }


def test_contract_keys_built_from_real_sources():
    keys = _integration(_orchestrator())._build_indicator_contract_keys(_blocks())

    assert keys["neural_activity"]["activation_pattern"] == pytest.approx(
        [0.1, 0.4, 0.2, 0.7]
    )
    assert keys["attention_focus"]["focus_changes"] == 6
    assert keys["memory_access"]["retrieved_memories"] == [
        "first memory",
        "second memory",
    ]
    assert keys["system_state"]["active_components"] == ["snn", "lsm", "htm"]
    assert "partition_independence" not in keys["system_state"]
    assert keys["causal_connections"]["active_links"] == [
        ("htm", "snn"),
        ("lsm", "snn"),
    ]
    assert keys["broadcast_latency_ms"] == pytest.approx(20.0)

    # The counters may also sit on the module itself.
    module_layout = _orchestrator(enhanced_workspace=_workspace(on_state=False))
    keys = _integration(module_layout)._build_indicator_contract_keys(_blocks())
    assert keys["attention_focus"]["focus_changes"] == 6
    # No broadcast has run, so no latency is reported.
    assert "broadcast_latency_ms" not in keys


def test_contract_keys_absent_sources_stay_missing():
    keys = _integration(SimpleNamespace())._build_indicator_contract_keys(
        _blocks(False, False, False)
    )
    # No SNN signal: the activation pattern is missing, not an empty vector.
    assert keys["neural_activity"]["activation_pattern"] is None
    assert keys["attention_focus"]["focus_changes"] is None
    assert keys["memory_access"]["retrieved_memories"] is None
    assert keys["system_state"]["active_components"] == []
    assert keys["causal_connections"]["active_links"] == []
    assert "broadcast_latency_ms" not in keys


def test_token_attention_is_absent_and_named():
    integration = _integration(_orchestrator())
    keys = integration._build_indicator_contract_keys(_blocks())
    # No component produces token attention, so the key is left out.
    assert "token_attention" not in keys.get("llm_processing", {})

    with pytest.raises(MissingSignal) as caught:
        asyncio.run(GlobalWorkspaceIntegration().measure(keys))
    assert "llm_processing.token_attention" in str(caught.value)
    assert "neural_activity.activation_pattern" not in str(caught.value)

    # Without an SNN signal the activation pattern is named as well.
    keys = _integration(
        _orchestrator(latest_neural_signals=None)
    )._build_indicator_contract_keys(_blocks())
    with pytest.raises(MissingSignal) as caught:
        asyncio.run(GlobalWorkspaceIntegration().measure(keys))
    assert "neural_activity.activation_pattern" in str(caught.value)


def test_ctm_component_read_from_thoughts_block():
    integration = _integration(_orchestrator())
    keys = integration._build_indicator_contract_keys(_blocks(ctm=True))
    assert keys["system_state"]["active_components"] == ["snn", "lsm", "htm", "ctm"]

    # A "ctm" block is not where the machine reports; only thoughts counts.
    blocks = _blocks(ctm=False)
    blocks["ctm"] = {"is_real": True}
    keys = integration._build_indicator_contract_keys(blocks)
    assert "ctm" not in keys["system_state"]["active_components"]


def test_active_links_are_component_pairs():
    binding = SimpleNamespace(
        clusters=[
            [0, 1, 2],  # snn only: no link
            [3, 10],  # snn and lsm, but lsm is not real this turn
            [4, 19, 26],  # snn, htm and a cross-modal oscillator
            [5, 20],  # snn and htm again: counted once
            [27, 28],  # cross-modal only
        ]
    )
    integration = _integration(_orchestrator(_last_binding_result=binding))
    keys = integration._build_indicator_contract_keys(_blocks(lsm=False))

    links = keys["causal_connections"]["active_links"]
    assert links == [("htm", "snn")]
    components = set(keys["system_state"]["active_components"])
    for a, b in links:
        assert a != b and a in components and b in components

    keys["system_state"]["partition_independence"] = 1.0
    score = asyncio.run(IntegratedInformationPhi().measure(keys))
    # One linked pair of the one possible pair between snn and htm.
    assert score == pytest.approx(1.0 * 0.8)
