"""The assessment reads only statistics its producers actually emit."""

import random

import numpy as np
import pytest

from mtc.assessment import assessment as assessment_module
from mtc.assessment.assessment import (
    INSUFFICIENT_DATA,
    MEASURED,
    STATS_CONTRACT,
    STATS_METHODS,
    ConsciousnessAssessment,
)

# ---------------------------------------------------------------------------
# Recording wrappers: every key read from a statistics dict is logged as
# (contract key, dotted path), including reads of keys that are missing.
# ---------------------------------------------------------------------------


class RecordingDict(dict):
    def __init__(self, data, contract_key, prefix, log):
        super().__init__(data)
        self._contract_key = contract_key
        self._prefix = prefix
        self._log = log

    def _record(self, key):
        path = f"{self._prefix}{key}"
        self._log.add((self._contract_key, path))
        return path

    def _wrap(self, key, value):
        path = self._record(key)
        if isinstance(value, dict) and not isinstance(value, RecordingDict):
            return RecordingDict(value, self._contract_key, path + ".", self._log)
        return value

    def __getitem__(self, key):
        return self._wrap(key, super().__getitem__(key))

    def get(self, key, default=None):
        if key in self:
            return self._wrap(key, super().__getitem__(key))
        self._record(key)
        return default


class Recorder:
    """Delegates to a module; its statistics methods return RecordingDicts."""

    def __init__(self, module, methods, log):
        self._module = module
        self._methods = methods  # method name -> contract key
        self._log = log

    def __getattr__(self, name):
        attr = getattr(self._module, name)
        if name in self._methods:
            contract_key = self._methods[name]

            def wrapped(*args, **kwargs):
                return RecordingDict(attr(*args, **kwargs), contract_key, "", self._log)

            return wrapped
        return attr


def recorded_modules(modules, log):
    """Wrap every module keyword that has a statistics contract."""
    methods_by_module = {}
    for contract_key, (module_key, method) in STATS_METHODS.items():
        methods_by_module.setdefault(module_key, {})[method] = contract_key
    wrapped = dict(modules)
    for module_key, methods in methods_by_module.items():
        if modules.get(module_key) is not None:
            wrapped[module_key] = Recorder(modules[module_key], methods, log)
    return wrapped


def navigate(data, path):
    value = data
    for part in path.split("."):
        value = value[part]
    return value


def producer_output(modules, contract_key):
    module_key, method = STATS_METHODS[contract_key]
    return getattr(modules[module_key], method)()


def assert_reads_match_producers(log, modules):
    assert log, "nothing was read"
    for contract_key, path in sorted(log):
        declared = STATS_CONTRACT[contract_key]
        assert path in declared, f"{contract_key}: {path} read but not declared"
        output = producer_output(modules, contract_key)
        try:
            value = navigate(output, path)
        except KeyError:
            pytest.fail(f"{contract_key}: {path} read but not emitted")
        assert isinstance(
            value, declared[path]
        ), f"{contract_key}: {path} is {type(value).__name__}"


# ---------------------------------------------------------------------------
# Real modules driven through a few synthetic cycles
# ---------------------------------------------------------------------------


class _SelfModel:
    def __init__(self):
        self.update_count = 0
        self.prediction_accuracy = 0.5
        self.model_confidence = 0.5
        self.self_calibration_score = 0.5
        self.prediction_accuracy_history = []

    def step(self):
        self.update_count += 1
        self.prediction_accuracy = float(
            np.clip(self.prediction_accuracy + np.random.uniform(-0.08, 0.08), 0.1, 0.9)
        )
        self.prediction_accuracy_history.append(self.prediction_accuracy)


class _MetaState:
    def __init__(self):
        self.total_predictions = 0
        self.predicted_accuracy = 0.5
        self.prediction_accuracy = 0.5
        self.model_confidence = 0.5

    def step(self, self_model):
        self.total_predictions += 1
        self.prediction_accuracy = self_model.prediction_accuracy
        self.predicted_accuracy = self_model.prediction_accuracy + 0.05


async def build_and_run_modules(cycles=12):
    """Real modules after ``cycles`` synthetic cycles (needs pymdp)."""
    pytest.importorskip("pymdp")
    from mtc.assessment.rpt_measurement import RPTMeasurement
    from mtc.consciousness.active_inference import ActiveInferenceModule
    from mtc.consciousness.attention_schema import (
        AttentionSchemaModule,
        AttentionShiftType,
        AttentionTarget,
    )
    from mtc.consciousness.beautiful_loop import BeautifulLoop
    from mtc.consciousness.enhanced_global_workspace import (
        EnhancedGlobalWorkspace,
        WorkspaceCandidate,
        WorkspaceCandidateSource,
    )
    from mtc.consciousness.metacognition import (
        FirstOrderStateType,
        MetacognitionModule,
        MetaType,
    )
    from mtc.consciousness.pad_affect import PADAffectModel, PADSources
    from mtc.neural.oscillatory_binding import OscillatoryBinding

    random.seed(3)
    np.random.seed(3)
    workspace = EnhancedGlobalWorkspace(body_metrics_enabled=False)
    attention = AttentionSchemaModule()
    metacognition = MetacognitionModule()
    inference = ActiveInferenceModule()
    loop = BeautifulLoop()
    rpt = RPTMeasurement()
    binding = OscillatoryBinding()
    rpt.attach_oscillatory_binding(binding)
    pad = PADAffectModel()
    self_model, meta_state = _SelfModel(), _MetaState()
    sources = [
        WorkspaceCandidateSource.SENSORY,
        WorkspaceCandidateSource.MEMORY,
        WorkspaceCandidateSource.CONCEPT,
    ]
    shifts = [AttentionShiftType.VOLUNTARY, AttentionShiftType.CAPTURED]

    for i in range(cycles):
        candidates = [
            WorkspaceCandidate(
                content=f"item_{i}_{j}",
                content_type="perception",
                summary=f"item {i} {j}",
                source=sources[(i + j) % len(sources)],
                activation_level=float(np.random.uniform(0.3, 0.9)),
                emotional_salience=float(np.random.uniform(0.0, 0.6)),
                novelty_score=float(np.random.uniform(0.1, 0.8)),
            )
            for j in range(5)
        ]
        state = await workspace.process_consciousness_cycle(candidates)
        hot = None
        if state.primary_content:
            attention.schema.current_focus = AttentionTarget(
                content=state.primary_content.candidate.content,
                content_type="perception",
                summary=state.primary_content.candidate.summary,
                source_module="global_workspace",
                attention_strength=state.primary_content.salience,
                salience=state.primary_content.salience,
                shift_type=shifts[i % 2],
            )
            first_order = metacognition.register_first_order_state(
                content=state.primary_content.candidate.content,
                content_summary=state.primary_content.candidate.summary,
                state_type=FirstOrderStateType.PERCEPTION,
                source_module="global_workspace",
            )
            hot = await metacognition.generate_hot(first_order, MetaType.AWARENESS)
        result = await inference.infer_and_act(np.random.dirichlet(np.ones(5)))
        self_model.step()
        meta_state.step(self_model)
        await loop.process_conscious_moment(
            prediction_errors=[float(result.prediction_error), 0.2, 0.3],
            workspace_winners=state.workspace_contents or [],
            inference_result=result,
            self_model=self_model,
            meta_state=meta_state,
            higher_order_thoughts=[hot] if hot is not None else None,
            attention_schema_state=attention.schema,
            context=f"cycle_{i}",
        )
        stats = workspace.get_statistics()
        rpt.measure_full(
            snn_state={"total_spikes": 150, "num_neurons": 100},
            lsm_state={"spectral_radius": 0.9, "active_neuron_ratio": 0.4},
            workspace_state={
                "broadcast": stats.get("broadcast", {}),
                "cycle_count": stats.get("cycle_count", i + 1),
                "ignition": stats.get("ignition", {}),
            },
        )
        binding.bind(
            {
                "attention_focus": float(np.random.random()),
                "valence": float(np.random.uniform(-1, 1)),
                "novelty": float(np.random.random()),
            }
        )
        pad.compute(PADSources(somatic_valence=float(np.sin(i / 3.0))))

    return {
        "global_workspace": workspace,
        "attention_schema": attention,
        "metacognition": metacognition,
        "active_inference": inference,
        "beautiful_loop": loop,
        "rpt_measurement": rpt,
        "pad_affect_model": pad,
    }


# Statistics a producer reports as None until it has measured them.
OPTIONAL_STATISTICS = {
    ("global_workspace", "broadcast.coverage_ratio"),
    ("global_workspace", "ignition.sustain_rate"),
    ("attention_schema", "prediction_accuracy"),
    ("attention_schema", "voluntary_ratio"),
    ("metacognition", "overall_confidence"),
    ("beautiful_loop", "average_binding_quality"),
    ("rpt_measurement", "average_local"),
    ("rpt_measurement.oscillatory_binding", "zero_stimulus_order_parameter"),
}

# Statistics a producer emits only once it has the data; the formulas read
# them only when present, so the noise stubs need not emit them.
PRESENT_WHEN_MEASURED = {
    ("rpt_measurement.oscillatory_binding", "zero_stimulus_order_parameter"),
}


@pytest.mark.asyncio
async def test_stats_contract_matches_real_producers():
    modules = await build_and_run_modules()
    log = set()
    report = await ConsciousnessAssessment().run_full_assessment(
        **recorded_modules(modules, log), normalize=False
    )
    assert_reads_match_producers(log, modules)
    read_keys = {key for key, _path in log}
    assert read_keys == set(STATS_CONTRACT)
    # The synthetic cycles leave the core indicators measured.
    for name in ("sequential_module_querying", "metacognition"):
        assert report.indicator_results[name].status == MEASURED, name
    # A default workspace registers no broadcast receiver, so its coverage
    # is unmeasured and the broadcast indicator says so.
    broadcast = report.indicator_results["global_broadcast"]
    assert broadcast.status == INSUFFICIENT_DATA
    assert broadcast.evidence["missing"] == "global_workspace.broadcast.coverage_ratio"


def test_stats_contract_accepts_none_only_for_optional_statistics():
    for contract_key, declared in STATS_CONTRACT.items():
        for path, expected_type in declared.items():
            accepts_none = isinstance(None, expected_type)
            optional = (contract_key, path) in OPTIONAL_STATISTICS
            assert accepts_none is optional, f"{contract_key}: {path}"
            if optional:
                assert isinstance(0.5, expected_type), f"{contract_key}: {path}"
    assert PRESENT_WHEN_MEASURED <= OPTIONAL_STATISTICS


def test_noise_stubs_emit_the_producer_schema():
    stubs = {
        "global_workspace": assessment_module._NoiseGlobalWorkspace(),
        "attention_schema": assessment_module._NoiseAttentionSchema(),
        "metacognition": assessment_module._NoiseMetacognition(),
        "active_inference": assessment_module._NoiseActiveInference(),
        "beautiful_loop": assessment_module._NoiseBeautifulLoop(),
        "rpt_measurement": assessment_module._NoiseRPTMeasurement(),
    }
    for contract_key, declared in STATS_CONTRACT.items():
        output = producer_output(stubs, contract_key)
        for path, expected_type in declared.items():
            if (contract_key, path) in PRESENT_WHEN_MEASURED:
                # The stub keeps the pre-measurement branch: no such key.
                with pytest.raises(KeyError):
                    navigate(output, path)
                continue
            value = navigate(output, path)
            assert isinstance(value, expected_type), f"{contract_key}: {path}"
            # Every stub statistic is a number: None would drop the indicator
            # out of the noise baseline.
            assert value is not None, f"{contract_key}: {path}"
    errors = [r.prediction_error for r in stubs["active_inference"].inference_history]
    assert len(errors) >= 10 and all(isinstance(e, float) for e in errors)
    pad = assessment_module._NoisePADAffectModel()
    from mtc.consciousness.pad_affect import PADState

    assert isinstance(pad.get_latest(), PADState)


def test_v04_noise_baseline_draws_v040_stub_inputs():
    # Under the v04 formulas the noise pass runs on the v0.4.0 stubs:
    # counters drawn from zero, a numeric trend, the v0.4.0 key set.
    random.seed(7)
    lows = {"cycle_count": [], "total_inferences": [], "loop_cycles": []}
    for _ in range(200):
        stubs = assessment_module._noise_modules(legacy_v04=True)
        gw_stats = stubs["global_workspace"].get_statistics()
        ai_stats = stubs["active_inference"].get_statistics()
        loop_stats = stubs["beautiful_loop"].get_loop_statistics()
        assert "bottleneck" not in gw_stats
        assert isinstance(ai_stats["prediction_error_trend"], float)
        assert "peak_depth" not in loop_stats["epistemic_depth"]
        assert not hasattr(stubs["rpt_measurement"], "get_oscillatory_binding_stats")
        assert stubs["pad_affect_model"].get_latest() is None
        lows["cycle_count"].append(gw_stats["cycle_count"])
        lows["total_inferences"].append(ai_stats["total_inferences"])
        lows["loop_cycles"].append(loop_stats["cycle_count"])
    assert all(min(values) == 0 for values in lows.values())
    # The v05 formulas keep the producer-schema stubs
    stubs = assessment_module._noise_modules(legacy_v04=False)
    assert isinstance(
        stubs["global_workspace"], assessment_module._NoiseGlobalWorkspace
    )


class _ErrorRecord:
    def __init__(self, error):
        self.prediction_error = error


class _TrendInference:
    def __init__(self, errors, avg_error=0.3):
        self.inference_history = [_ErrorRecord(e) for e in errors]
        self._avg = avg_error

    def get_statistics(self):
        return {
            "total_inferences": len(self.inference_history),
            "total_model_updates": 3,
            "avg_prediction_error": self._avg,
            "prediction_error_trend": "decreasing",
            "hierarchical_prediction_error": 0.2,
            "variational_free_energy": 0.5,
        }


@pytest.mark.asyncio
async def test_pem_trend_reads_numeric_slope():
    errors = list(np.linspace(0.6, 0.2, 20))
    report = await ConsciousnessAssessment().run_full_assessment(
        active_inference=_TrendInference(errors), normalize=False
    )
    result = report.indicator_results["prediction_error_minimization"]
    assert result.status == MEASURED
    slope = np.polyfit(np.arange(20), errors, 1)[0]
    relative = slope * 20 / np.mean(errors)
    magnitude = min(1.0, max(0.0, -relative))
    assert result.evidence["trend_magnitude"] == pytest.approx(magnitude)
    assert magnitude > 0.5
    assert result.score == pytest.approx(0.7 * magnitude + 0.3 * 0.7)

    rising = await ConsciousnessAssessment().run_full_assessment(
        active_inference=_TrendInference(list(np.linspace(0.2, 0.6, 20))),
        normalize=False,
    )
    rising_result = rising.indicator_results["prediction_error_minimization"]
    assert rising_result.evidence["trend_magnitude"] == 0.0
    assert rising_result.score == pytest.approx(0.3 * 0.7)

    short = await ConsciousnessAssessment().run_full_assessment(
        active_inference=_TrendInference(errors[:9]), normalize=False
    )
    assert (
        short.indicator_results["prediction_error_minimization"].status
        == INSUFFICIENT_DATA
    )


class _DepthLoop:
    def __init__(self, peak):
        self.peak = peak

    def get_loop_statistics(self):
        return {
            "cycle_count": 10,
            "average_loop_quality": 0.5,
            "average_binding_quality": 0.4,
            "field_evidencing_count": 1,
            "epistemic_depth": {"average_depth": 2.0, "peak_depth": self.peak},
        }


@pytest.mark.asyncio
async def test_genuine_implementation_reads_peak_depth():
    deep = await ConsciousnessAssessment().run_full_assessment(
        beautiful_loop=_DepthLoop(peak=3), normalize=False
    )
    genuine = deep.indicator_results["genuine_implementation"]
    # The third condition is the depth tracker reaching its meta-prediction
    # level (peak_depth >= 2), read from module state; it is not the
    # tracker's strange-loop verdict, which reaches this indicator only
    # through field_evidencing_count, so the evidence key says so.
    assert genuine.evidence["has_meta_prediction_depth"] is True
    assert "has_strange_loop" not in genuine.evidence
    assert genuine.score == pytest.approx(1.0)
    assert deep.indicator_results["epistemic_depth"].confidence == 0.6

    shallow = await ConsciousnessAssessment().run_full_assessment(
        beautiful_loop=_DepthLoop(peak=1), normalize=False
    )
    shallow_genuine = shallow.indicator_results["genuine_implementation"]
    assert shallow_genuine.evidence["has_meta_prediction_depth"] is False
    assert shallow_genuine.score == pytest.approx(2.0 / 3.0)
    assert shallow.indicator_results["epistemic_depth"].confidence == 0.3


class _BottleneckWorkspace:
    def get_statistics(self):
        return {
            "cycle_count": 10,
            "broadcast": {"total_broadcasts": 4, "coverage_ratio": 0.5},
            "ignition": {"total_ignitions": 0},
            "bottleneck": {
                "total_competitions": 10,
                "avg_candidates": 6.0,
                "source_distribution": {"sensory": 4, "memory": 2},
            },
        }


@pytest.mark.asyncio
async def test_sequential_querying_and_percepts_read_bottleneck():
    log = set()
    modules = {"global_workspace": _BottleneckWorkspace()}
    report = await ConsciousnessAssessment().run_full_assessment(
        **recorded_modules(modules, log), normalize=False
    )
    sequential = report.indicator_results["sequential_module_querying"]
    assert sequential.score == pytest.approx(0.5 * 1.0 + 0.5 * 0.6)
    percepts = report.indicator_results["organized_percepts"]
    assert percepts.score == pytest.approx(0.5 * min(1.0, 4 / 5))
    paths = {path for _key, path in log}
    assert "bottleneck.total_competitions" in paths
    assert not any(path.startswith("competition") for path in paths)


class _HotMeta:
    def get_statistics(self):
        return {
            "total_hots_generated": 5,
            "introspection_count": 1,
            "first_order_buffer_size": 5,
            "overall_confidence": 0.6,
        }


@pytest.mark.asyncio
async def test_no_metacognitive_access_reads_total_hots_generated():
    report = await ConsciousnessAssessment().run_full_assessment(
        metacognition=_HotMeta(), normalize=False
    )
    neg = report.negative_indicators["no_metacognitive_access"]
    assert neg["score"] == 0.0 and neg["status"] == MEASURED
    legacy = await ConsciousnessAssessment(
        indicator_formulas="v04"
    ).run_full_assessment(metacognition=_HotMeta(), normalize=False)
    # v0.4.0 read a never-emitted key and saw no HOTs
    assert legacy.negative_indicators["no_metacognitive_access"]["score"] == 0.3


@pytest.mark.asyncio
async def test_v04_formulas_keep_v04_reads():
    from tests.assessment.test_missing_signals import (
        V04_FIXTURE_EXPECTED,
        fixture_modules,
    )

    report = await ConsciousnessAssessment(
        indicator_formulas="v04"
    ).run_full_assessment(**fixture_modules(), normalize=False)
    # The indicators whose v0.4.0 bodies read dead keys keep their numbers
    for name in (
        "sequential_module_querying",
        "organized_percepts",
        "flexible_responsiveness",
        "epistemic_depth",
        "genuine_implementation",
        "metacognition",
        "prediction_error_minimization",
        "global_ignition_nuanced",
        "complex_agency",
        "online_reportable_learning",
        "temporal_prediction_depth",
        "agency",
        "embodiment",
    ):
        expected_score, expected_confidence, _passes = V04_FIXTURE_EXPECTED[name]
        result = report.indicator_results[name]
        assert result.score == expected_score, name
        assert result.confidence == expected_confidence, name
