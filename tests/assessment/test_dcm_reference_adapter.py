"""The reference DCM evidence adapter over MTC's own module set."""

import json
from pathlib import Path

import pytest

from mtc.assessment.dcm_benchmark import DCMEvidenceAdapter
from mtc.assessment.dcm_reference_adapter import (
    IDENTITY_INDICATORS,
    KNOWN_ZEROS,
    ReferenceDCMAdapter,
    build_default_adapter,
)

ROOT = Path(__file__).resolve().parents[2]

# Known zeros the bundled 106-indicator spec subset omits; the full
# 206-indicator scheme carries them.
FULL_SPEC_ONLY = {
    "Predator Evasion",
    "Body Maintenance",
    "Dedicated Speech Systems",
    "Dedicated Action Systems",
    "Dedicated Sensory Systems",
    "Social Contagion",
    "Navigation Performance",
    "Single Moving Perspective",
}


def _bundled_names():
    data = json.loads((ROOT / "data" / "dcm" / "bundled_spec.json").read_text())
    raw = data["raw"] if isinstance(data, dict) and "raw" in data else data
    names = set()

    def walk(evidencers):
        for ev in evidencers:
            children = ev.get("evidencers", [])
            if children:
                walk(children)
            else:
                names.add(ev["name"])

    for stance in raw:
        walk(stance.get("evidencers", []))
    return names


class _Workspace:
    def get_statistics(self):
        return {
            "cycle_count": 10,
            "broadcast": {"total_broadcasts": 6, "coverage_ratio": 0.6},
            "ignition": {"total_ignitions": 3, "sustain_rate": 0.5},
            "bottleneck": {"total_competitions": 10, "avg_candidates": 5.0},
            "self_model": {"self_calibration": 0.7, "predictions_made": 4},
        }


class _Attention:
    def get_statistics(self):
        return {
            "total_updates": 8,
            "prediction_accuracy": 0.55,
            "voluntary_ratio": 0.4,
            "voluntary_shifts": 3,
        }


class _Meta:
    def get_statistics(self):
        return {
            "total_hots_generated": 6,
            "introspection_count": 4,
            "first_order_buffer_size": 6,
            "doubt_level": 0.2,
        }


class _Drive:
    def __init__(self, current, optimal, urgency):
        self.current_level = current
        self.optimal_level = optimal
        self._urgency = urgency

    def get_deviation(self):
        return abs(self.current_level - self.optimal_level)

    def get_urgency(self):
        return self._urgency


class _Inference:
    def __init__(self):
        self.config = type("C", (), {"planning_horizon": 10})()
        self.homeostatic_drives = type(
            "H",
            (),
            {
                "drives": {
                    "curiosity": _Drive(0.4, 0.7, 0.3),
                    "safety": _Drive(0.9, 0.9, 0.0),
                    "coherence": _Drive(0.5, 0.8, 0.25),
                }
            },
        )()

    def get_statistics(self):
        return {
            "total_inferences": 12,
            "total_model_updates": 6,
            "avg_prediction_error": 0.3,
        }


class _RPT:
    def __init__(self, binding=True, floor=None):
        self._binding = binding
        self._floor = floor

    def get_statistics(self):
        return {"average_local": 0.4, "average_global": 0.3, "measurement_count": 5}

    def get_oscillatory_binding_stats(self):
        if not self._binding:
            return {}
        stats = {
            "global_order_parameter": 0.6,
            "cross_coherence": {"snn_lsm": 0.2, "snn_htm": 0.4},
            "binding_count": 7,
        }
        if self._floor is not None:
            stats["zero_stimulus_order_parameter"] = self._floor
        return stats


def _modules(binding=True, floor=None):
    return {
        "global_workspace": _Workspace(),
        "attention_schema": _Attention(),
        "metacognition": _Meta(),
        "active_inference": _Inference(),
        "rpt_measurement": _RPT(binding, floor),
    }


BINDING_NAMES = ("Feature Binding", "Holistic Dependency", "Wiring Convergence")
HORIZON_NAMES = (
    "Coherent Goal-directed Behavior",
    "Anticipation Gradient",
    "Simulations of Future",
    "Plans for Future Needs",
)


def test_known_zeros_are_in_the_bundled_spec():
    names = _bundled_names()
    assert len(KNOWN_ZEROS) == 29
    missing = set(KNOWN_ZEROS) - names
    assert missing == FULL_SPEC_ONLY
    adapter = ReferenceDCMAdapter({})
    assert adapter.get_known_zeros() == set(KNOWN_ZEROS)
    override = ReferenceDCMAdapter({}, known_zeros={"Carbon-Based"})
    assert override.get_known_zeros() == {"Carbon-Based"}


def test_produced_names_subset_of_spec():
    names = _bundled_names()
    adapter = ReferenceDCMAdapter(
        _modules(),
        substrate_sizes={"layers": 6, "nodes": 5000, "connections": 200000},
        identity_hook=lambda: {name: 0.5 for name in IDENTITY_INDICATORS},
    )
    evidence = adapter.collect_module_evidence()
    assert set(evidence) <= names
    assert set(evidence) <= ReferenceDCMAdapter.MODULE_INDICATOR_NAMES
    assert ReferenceDCMAdapter.MODULE_INDICATOR_NAMES <= names
    assert adapter.produced_indicator_names == frozenset(evidence)
    assert isinstance(adapter, DCMEvidenceAdapter)


def test_no_evidence_without_modules():
    adapter = ReferenceDCMAdapter({})
    assert adapter.collect_module_evidence() == {}
    assert adapter.get_conversation_samples() == []


def test_evidence_from_module_statistics():
    evidence = ReferenceDCMAdapter(_modules()).collect_module_evidence()
    assert evidence["Threshold Activation"] == 1.0
    assert evidence["Information Transfer"] == pytest.approx(0.6)
    assert evidence["Attentional Schema"] == pytest.approx(0.55)
    assert evidence["Metacognition"] == pytest.approx(0.4)
    assert evidence["Self-Representation"] == pytest.approx(0.7)
    assert evidence["Compressibility"] == pytest.approx(0.7)
    # The planning horizon is configuration, not a measurement: the four
    # names it used to fill (with a horizon of 10 here) are not reported.
    for name in HORIZON_NAMES:
        assert name not in evidence, name
        assert name not in ReferenceDCMAdapter.MODULE_INDICATOR_NAMES, name
    assert evidence["Adaptive Learning"] == pytest.approx(0.6)
    assert evidence["Curiosity"] == pytest.approx(0.3)
    assert evidence["Maintains Homeostasis"] == pytest.approx(
        1.0 - (0.3 + 0.0 + 0.3) / 3
    )
    assert evidence["Conflicting Subparts"] == pytest.approx(2 / 3)
    assert evidence["Function Re-application"] == pytest.approx(0.4)
    for value in evidence.values():
        assert 0.0 <= value <= 1.0
    # No constant stands in for continuous information or structure
    assert "Continuous Information" not in evidence
    assert "Number of Nodes" not in evidence
    assert "Functional Specialization" not in evidence


class _FullCoverageWorkspace:
    def __init__(self, coverage, broadcasts=6):
        self.coverage = coverage
        self.broadcasts = broadcasts

    def get_statistics(self):
        return {
            "cycle_count": 10,
            "broadcast": {
                "total_broadcasts": self.broadcasts,
                "coverage_ratio": self.coverage,
            },
            "ignition": {"total_ignitions": 3, "sustain_rate": 0.5},
        }


def test_bottleneck_omitted_when_undetermined():
    full = ReferenceDCMAdapter(
        {"global_workspace": _FullCoverageWorkspace(1.0)}
    ).collect_module_evidence()
    assert full["Information Transfer"] == pytest.approx(1.0)
    assert "Informational Bottleneck" not in full
    partial = ReferenceDCMAdapter(
        {"global_workspace": _FullCoverageWorkspace(0.4)}
    ).collect_module_evidence()
    assert partial["Informational Bottleneck"] == 1.0
    # No coverage at all, no broadcast, or no coverage reported: undetermined.
    zero = ReferenceDCMAdapter(
        {"global_workspace": _FullCoverageWorkspace(0.0)}
    ).collect_module_evidence()
    assert "Informational Bottleneck" not in zero
    silent = ReferenceDCMAdapter(
        {"global_workspace": _FullCoverageWorkspace(0.4, broadcasts=0)}
    ).collect_module_evidence()
    assert "Informational Bottleneck" not in silent
    unreported = ReferenceDCMAdapter(
        {"global_workspace": _FullCoverageWorkspace(None)}
    ).collect_module_evidence()
    assert "Informational Bottleneck" not in unreported
    assert "Information Transfer" not in unreported


@pytest.mark.asyncio
async def test_adapter_reads_only_producer_keys():
    from tests.assessment.test_stats_contract import (
        build_and_run_modules,
        navigate,
        recorded_modules,
    )
    from mtc.assessment.assessment import STATS_METHODS

    modules = await build_and_run_modules()
    log = set()
    adapter = ReferenceDCMAdapter(recorded_modules(modules, log))
    evidence = adapter.collect_module_evidence()
    assert evidence
    assert log
    for contract_key, path in sorted(log):
        module_key, method = STATS_METHODS[contract_key]
        output = getattr(modules[module_key], method)()
        try:
            navigate(output, path)
        except KeyError:
            pytest.fail(f"adapter read {contract_key}: {path}, which is not emitted")


def test_binding_reads_cross_coherence():
    evidence = ReferenceDCMAdapter(_modules(binding=True)).collect_module_evidence()
    assert evidence["Feature Binding"] == pytest.approx(0.6)
    assert evidence["Wiring Convergence"] == pytest.approx((0.2 + 0.4) / 2)
    assert evidence["Holistic Dependency"] == pytest.approx(0.6 * 0.6)

    fallback = ReferenceDCMAdapter(_modules(binding=False)).collect_module_evidence()
    # Without binding data nothing stands in for the binding indicators
    assert "Feature Binding" not in fallback
    assert "Holistic Dependency" not in fallback
    assert "Wiring Convergence" not in fallback
    assert fallback["Information Transfer"] == pytest.approx(0.6)

    # Binding data without a workspace coverage: holistic dependency, which
    # needs both, is omitted; the binding-only indicators stay.
    no_workspace = ReferenceDCMAdapter(
        {"rpt_measurement": _RPT(binding=True)}
    ).collect_module_evidence()
    assert no_workspace["Feature Binding"] == pytest.approx(0.6)
    assert no_workspace["Wiring Convergence"] == pytest.approx(0.3)
    assert "Holistic Dependency" not in no_workspace


def test_saturated_binding_is_omitted():
    from mtc.assessment.dcm_reference_adapter import BINDING_FLOOR_MIN_HEADROOM

    assert BINDING_FLOOR_MIN_HEADROOM == 0.1
    # A floor of 0.95 leaves 0.05 headroom: the layer synchronizes without
    # input, so its order parameter is no binding evidence.
    saturated = ReferenceDCMAdapter(_modules(floor=0.95)).collect_module_evidence()
    for name in BINDING_NAMES:
        assert name not in saturated, name
    assert saturated["Information Transfer"] == pytest.approx(0.6)
    # Enough headroom, or a floor the producer did not report: today's rule.
    for floor in (0.85, None, float("nan"), "0.95"):
        evidence = ReferenceDCMAdapter(_modules(floor=floor)).collect_module_evidence()
        assert evidence["Feature Binding"] == pytest.approx(0.6), floor
        assert evidence["Holistic Dependency"] == pytest.approx(0.36), floor
        assert evidence["Wiring Convergence"] == pytest.approx(0.3), floor


def test_binding_floor_at_the_headroom_keeps_the_indicators():
    # A floor of exactly 0.9 leaves a headroom of 0.1, which is not less than
    # BINDING_FLOOR_MIN_HEADROOM (1.0 - 0.9 is 0.0999... in floating point).
    evidence = ReferenceDCMAdapter(_modules(floor=0.9)).collect_module_evidence()
    assert evidence["Feature Binding"] == pytest.approx(0.6)
    assert evidence["Holistic Dependency"] == pytest.approx(0.36)
    assert evidence["Wiring Convergence"] == pytest.approx(0.3)
    above = ReferenceDCMAdapter(_modules(floor=0.9000001)).collect_module_evidence()
    for name in BINDING_NAMES:
        assert name not in above, name


def test_structural_evidence_only_from_supplied_counts():
    import math

    without = ReferenceDCMAdapter(_modules()).collect_module_evidence()
    assert not {"Number of Layers", "Number of Nodes", "Number of Connections"} & set(
        without
    )
    adapter = ReferenceDCMAdapter(
        {},
        substrate_sizes={"layers": 6, "nodes": 5000},
        structural_references={"Number of Layers": 10.0, "Number of Nodes": 1e6},
    )
    evidence = adapter.collect_module_evidence()
    assert evidence["Number of Layers"] == pytest.approx(math.log10(7) / math.log10(11))
    assert evidence["Number of Nodes"] == pytest.approx(
        math.log10(5001) / math.log10(1e6 + 1)
    )
    assert "Number of Connections" not in evidence


def test_description_from_inventory_has_no_identity_text(monkeypatch):
    from mtc.assessment import dcm_reference_adapter
    from mtc.core.config import Settings

    monkeypatch.setattr(
        dcm_reference_adapter,
        "get_settings",
        lambda: Settings(SYSTEM_NAME="TestSystem"),
    )
    hook_values = {"Stable Personality": 0.9}
    adapter = ReferenceDCMAdapter(
        _modules(),
        substrate_sizes={"snn": 5000},
        identity_hook=lambda: hook_values,
        conversation_provider=lambda n: ["a private line"],
    )
    description = adapter.get_system_description()
    assert description.startswith("TestSystem")
    assert "30 positive and 3 negative indicators" in description
    assert "Global Workspace Theory" in description
    assert "snn: 5000" in description
    assert "private line" not in description
    assert "Stable Personality" not in description
    assert description.isascii()


def test_conversation_samples_pass_privacy_filter():
    samples = ["keep me", "drop me", "redact SECRET"]

    def privacy_filter(text):
        if text.startswith("drop"):
            return None
        return text.replace("SECRET", "[redacted]")

    adapter = ReferenceDCMAdapter(
        {},
        conversation_provider=lambda n: samples[:n],
        privacy_filter=privacy_filter,
    )
    assert adapter.get_conversation_samples(3) == ["keep me", "redact [redacted]"]
    assert ReferenceDCMAdapter({}).get_conversation_samples(5) == []


def test_identity_indicators_only_via_hook():
    without = ReferenceDCMAdapter(_modules()).collect_module_evidence()
    assert not set(IDENTITY_INDICATORS) & set(without)
    hooked = ReferenceDCMAdapter(
        _modules(),
        identity_hook=lambda: {
            "Stable Personality": 0.8,
            "Self-Recognition": 1.4,
            "Curiosity": 0.99,  # not an identity indicator: ignored
            "Self-Representations": float("nan"),
        },
    ).collect_module_evidence()
    assert hooked["Stable Personality"] == pytest.approx(0.8)
    assert hooked["Self-Recognition"] == 1.0
    assert "Self-Representations" not in hooked
    assert hooked["Curiosity"] == without["Curiosity"]


def test_build_default_adapter_without_state_dir():
    pytest.importorskip("pymdp")
    adapter = build_default_adapter()
    assert isinstance(adapter, ReferenceDCMAdapter)
    assert "global_workspace" in adapter.modules
    # Fresh modules: nothing has run, so no module evidence yet
    assert adapter.collect_module_evidence() == {}


def test_real_binding_floor_omits_the_binding_indicators():
    """With the real binding layer, whose static coupling synchronizes the
    oscillators without input, the measured zero-stimulus floor leaves no
    headroom and the binding indicators are omitted, through the RPT
    statistics and through the binding module alone."""
    from mtc.assessment.dcm_reference_adapter import BINDING_FLOOR_MIN_HEADROOM
    from mtc.assessment.rpt_measurement import RPTMeasurement
    from mtc.neural.oscillatory_binding import OSCILLATOR_NAMES, OscillatoryBinding

    binding = OscillatoryBinding()
    rpt = RPTMeasurement()
    rpt.attach_oscillatory_binding(binding)
    for value in (0.2, 0.5, 0.8):
        binding.bind({name: value for name in OSCILLATOR_NAMES})
    stats = rpt.get_oscillatory_binding_stats()
    floor = stats["zero_stimulus_order_parameter"]
    assert 1.0 - floor < BINDING_FLOOR_MIN_HEADROOM
    for modules in (
        {"rpt_measurement": rpt, "global_workspace": _Workspace()},
        {"oscillatory_binding": binding, "global_workspace": _Workspace()},
    ):
        evidence = ReferenceDCMAdapter(modules).collect_module_evidence()
        for name in BINDING_NAMES:
            assert name not in evidence, (sorted(modules), name)
        assert evidence["Information Transfer"] == pytest.approx(0.6)
