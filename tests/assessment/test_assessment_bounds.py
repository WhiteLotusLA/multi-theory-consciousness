"""Report hygiene: clamped scores, raw scores, negative indicators, studies."""

from pathlib import Path

import pytest

from mtc.assessment import assessment as assessment_module
from mtc.assessment.assessment import (
    AblationStudy,
    ConsciousnessAssessment,
    LongitudinalStudy,
)

ROOT = Path(__file__).resolve().parents[2]


class _OverflowRPT:
    """Statistics that push the recurrence formulas above 1."""

    def get_statistics(self):
        return {"average_local": 3.0, "average_global": 2.5, "measurement_count": 7}

    def get_oscillatory_binding_stats(self):
        return {
            "global_order_parameter": 4.0,
            "cross_coherence": {"a": 2.0},
            "binding_count": 9,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("formulas", ["v05", "v04"])
async def test_scores_clamped_to_unit_interval(formulas):
    report = await ConsciousnessAssessment(
        indicator_formulas=formulas
    ).run_full_assessment(rpt_measurement=_OverflowRPT(), normalize=False)
    for name in (
        "algorithmic_recurrence",
        "local_recurrence",
        "oscillatory_binding_coherence",
    ):
        assert report.indicator_results[name].score == 1.0, name
    for result in report.indicator_results.values():
        assert 0.0 <= result.score <= 1.0
    assert 0.0 <= report.overall_score <= 1.0


@pytest.mark.asyncio
async def test_normalized_assessment_raw_scores_are_raw(monkeypatch):
    calls = []
    original = ConsciousnessAssessment.run_full_assessment

    async def spy(self, *args, **kwargs):
        calls.append(kwargs.get("normalize", True))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ConsciousnessAssessment, "run_full_assessment", spy)
    assessment = ConsciousnessAssessment()
    result = await assessment.run_normalized_assessment(noise_iterations=2)
    assert calls == [False]
    for name, raw in result.raw_scores.items():
        assert raw == result.raw_report.indicator_results[name].score
        assert (
            "noise_baseline" not in result.raw_report.indicator_results[name].evidence
        )


class _ActiveWorkspace:
    def get_statistics(self):
        return {
            "cycle_count": 5,
            "broadcast": {"total_broadcasts": 3, "coverage_ratio": 0.5},
            "ignition": {"total_ignitions": 0},
            "bottleneck": {"total_competitions": 5, "avg_candidates": 3.0},
        }


@pytest.mark.asyncio
async def test_negative_indicators_on_report():
    report = await ConsciousnessAssessment().run_full_assessment(
        global_workspace=_ActiveWorkspace(), normalize=False
    )
    negative = report.negative_indicators
    assert set(negative) == {
        "purely_reactive",
        "fixed_repertoire",
        "no_metacognitive_access",
    }
    # Cycles but no ignition: the purely-reactive rule level 0.5, penalized
    assert negative["purely_reactive"]["score"] == 0.5
    assert negative["purely_reactive"]["penalty"] == pytest.approx(0.5 * 0.8 * 0.05)
    assert negative["fixed_repertoire"]["status"] == "absent"
    exported = report.to_dict()
    assert exported["negative_indicators"] == negative
    assert exported["status"] == report.status
    assert exported["coverage"] == report.coverage
    assert exported["insufficient_indicators"] == report.insufficient_indicators
    assert exported["indicator_results"]["global_broadcast"]["status"] == "measured"


_EXPECTED_KEYS = {
    "global_workspace",
    "attention_schema",
    "metacognition",
    "active_inference",
    "neural_states",
    "beautiful_loop",
    "rpt_measurement",
    "pad_affect_model",
    "causal_emergence_analyzer",
    "phi_tracker",
}


def _system_state():
    return {key: object() for key in _EXPECTED_KEYS}


@pytest.mark.asyncio
async def test_studies_pass_full_module_dict(monkeypatch, tmp_path):
    seen = []
    real_run = ConsciousnessAssessment.run_full_assessment

    async def capture(self, **kwargs):
        seen.append(kwargs)
        return await real_run(self, normalize=False)

    monkeypatch.setattr(ConsciousnessAssessment, "run_full_assessment", capture)

    study = LongitudinalStudy(output_dir=str(tmp_path / "longitudinal"))
    study.start_time = assessment_module.datetime.now()
    state = _system_state()
    measurement = await study._take_measurement(state)
    assert _EXPECTED_KEYS <= set(seen[-1])
    for key in _EXPECTED_KEYS:
        assert seen[-1][key] is state[key]
    # The captured call measured nothing, so the study records no score
    assert measurement.consciousness_score is None

    async def get_state():
        return _system_state()

    ablation = AblationStudy(output_dir=str(tmp_path / "ablation"))
    results = await ablation.run_full_ablation(get_state)
    assert all(_EXPECTED_KEYS <= set(kwargs) for kwargs in seen[1:])
    assert len(results["configurations"]) == 7


@pytest.mark.asyncio
async def test_studies_record_missing_overall(monkeypatch, tmp_path):
    original = ConsciousnessAssessment._overall_score
    monkeypatch.setattr(ConsciousnessAssessment, "_overall_score", lambda self, r: None)
    study = LongitudinalStudy(output_dir=str(tmp_path / "longitudinal"))
    study.start_time = assessment_module.datetime.now()
    for _ in range(3):
        study.measurements.append(await study._take_measurement({}))
    assert all(m.consciousness_score is None for m in study.measurements)
    results = study._analyze_results()
    assert results.consciousness_mean is None
    assert results.emergence_detected is False

    async def get_state():
        return {}

    ablation = AblationStudy(output_dir=str(tmp_path / "ablation"))
    outcome = await ablation.run_full_ablation(get_state)
    assert all(i["score_drop"] is None for i in outcome["impacts"])
    monkeypatch.setattr(ConsciousnessAssessment, "_overall_score", original)


def _project_settings(tmp_path, monkeypatch):
    from mtc.core.config import Settings

    project = tmp_path / "project"
    settings = Settings(_env_file=None, PROJECT_ROOT=str(project))
    monkeypatch.setattr(assessment_module, "get_settings", lambda: settings)
    working = tmp_path / "cwd"
    working.mkdir()
    monkeypatch.chdir(working)
    return project, working


def test_study_construction_creates_no_directory(tmp_path, monkeypatch):
    project, working = _project_settings(tmp_path, monkeypatch)
    LongitudinalStudy()
    AblationStudy()
    assert not project.exists()
    assert list(working.iterdir()) == []
    # An explicit directory is not created at construction either
    LongitudinalStudy(output_dir=str(tmp_path / "explicit"))
    assert not (tmp_path / "explicit").exists()


@pytest.mark.asyncio
async def test_study_default_dirs_follow_settings(tmp_path, monkeypatch):
    project, working = _project_settings(tmp_path, monkeypatch)
    study = LongitudinalStudy()
    assert study.output_dir == project / "results" / "longitudinal_studies"
    study.start_time = assessment_module.datetime.now()
    await study._save_intermediate_results()
    written = list((project / "results" / "longitudinal_studies").iterdir())
    assert [p.name for p in written] == [f"study_{study.study_id}_intermediate.json"]
    assert list(working.iterdir()) == []

    ablation = AblationStudy()
    assert ablation.output_dir == project / "results" / "ablation_studies"
    ablation._save_results({"configurations": [], "impacts": []})
    files = list((project / "results" / "ablation_studies").iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("ablation_")
    assert "phase" not in files[0].name
    assert "phase" not in str(ablation.output_dir)

    # An explicit directory is used as given
    explicit = AblationStudy(output_dir=str(tmp_path / "explicit"))
    assert explicit.output_dir == tmp_path / "explicit"
    explicit._save_results({"configurations": [], "impacts": []})
    assert len(list((tmp_path / "explicit").iterdir())) == 1


def test_header_and_counts():
    source = (ROOT / "mtc" / "assessment" / "assessment.py").read_text()
    assert "30 positive and 3 negative" in source
    assert "20-indicator" not in source and "20 indicators" not in source
    assert chr(0xA7) not in source  # section signs became "section 3"
    assessment = ConsciousnessAssessment()
    assert len(assessment.indicator_configs) == 30
    assert len(assessment.negative_indicator_configs) == 3
    assert assessment.theory_empirical_weights["GWT"] == 1.0
    for cfg in assessment.indicator_configs.values():
        assert cfg["description"].isascii()
