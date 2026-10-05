"""Tests for the DCM LLM-based indicator evaluator."""

import json
import pytest
import tempfile
from datetime import datetime, timedelta


class TestDCMEvaluator:
    """Tests for LLM-based indicator evaluation."""

    def test_build_batch_prompt(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp())
        prompt = evaluator._build_batch_prompt(
            indicators=["Curiosity", "Play", "Metacognition"],
            system_description="A consciousness research platform with 7 theories.",
            conversation_samples=["Hello there", "Hi! I'm curious about stars."],
        )
        assert "Curiosity" in prompt
        assert "Play" in prompt
        assert "Metacognition" in prompt
        assert "consciousness research platform" in prompt
        assert "curious about stars" in prompt

    def test_parse_evaluation_response(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp())
        # Simulated LLM response
        response = json.dumps(
            {
                "Curiosity": {"score": 0.8, "reasoning": "Shows exploratory behavior"},
                "Play": {"score": 0.3, "reasoning": "Limited playful behavior"},
                "Metacognition": {"score": 0.9, "reasoning": "Strong self-reflection"},
            }
        )
        scores = evaluator._parse_response(
            response, ["Curiosity", "Play", "Metacognition"]
        )
        assert scores["Curiosity"] == pytest.approx(0.8)
        assert scores["Play"] == pytest.approx(0.3)
        assert scores["Metacognition"] == pytest.approx(0.9)

    def test_parse_response_handles_malformed(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp())
        # Malformed response: nothing scored, no stand-in value
        scores = evaluator._parse_response("not valid json", ["Curiosity", "Play"])
        assert scores == {}

    def test_parse_response_leaves_out_missing_and_unparseable_entries(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp())
        response = json.dumps(
            {
                "Curiosity": {"score": 0.8, "reasoning": "x"},
                "Play": {"reasoning": "no score given"},
                "Metacognition": "high",
                "Agency": {"score": None},
            }
        )
        scores = evaluator._parse_response(
            response, ["Curiosity", "Play", "Metacognition", "Agency", "Memory"]
        )
        assert scores == {"Curiosity": pytest.approx(0.8)}

    def test_failed_batch_is_left_out_and_not_cached(self, monkeypatch):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp(), batch_size=1)
        calls = []

        def flaky_local(prompt):
            calls.append(prompt)
            if "- Play" in prompt:
                raise ConnectionError("server down")
            return json.dumps({"Curiosity": {"score": 0.7, "reasoning": "x"}})

        monkeypatch.setattr(evaluator, "_call_local", flaky_local)
        first = evaluator.evaluate_indicators(
            ["Curiosity", "Play"], None, "desc", ["s1"]
        )
        assert first == {"Curiosity": pytest.approx(0.7)}
        assert list(evaluator.cache_dir.glob("eval_*.json")) == []
        # Nothing cached, so the next call evaluates again
        evaluator.evaluate_indicators(["Curiosity", "Play"], None, "desc", ["s1"])
        assert len(calls) == 4

    def test_evaluator_has_no_default_score(self):
        from mtc.assessment import dcm_evaluator

        assert not hasattr(dcm_evaluator, "DEFAULT_SCORE")

    def test_cache_ttl(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        cache_dir = tempfile.mkdtemp()
        evaluator = DCMEvaluator(cache_dir=cache_dir)
        key = evaluator._cache_key("desc", ["sample"], ["Curiosity"])

        # Store a cached result
        evaluator._save_cache({"Curiosity": 0.7}, datetime.now(), key)

        # Should be fresh
        cached = evaluator._load_cache(["Curiosity"], key)
        assert cached is not None
        assert cached["Curiosity"] == pytest.approx(0.7)

    def test_cache_expired(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        cache_dir = tempfile.mkdtemp()
        evaluator = DCMEvaluator(cache_dir=cache_dir, cache_ttl_hours=24)
        key = evaluator._cache_key("desc", ["sample"], ["Curiosity"])

        # Store an old cached result
        old_time = datetime.now() - timedelta(hours=25)
        evaluator._save_cache({"Curiosity": 0.7}, old_time, key)

        # Should be expired
        cached = evaluator._load_cache(["Curiosity"], key)
        assert cached is None

    def test_cache_keyed_by_description_samples_and_indicators(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp())
        base = evaluator._cache_key("desc", ["a", "b"], ["Play", "Curiosity"])
        # Indicator order does not matter; every input does.
        assert base == evaluator._cache_key("desc", ["a", "b"], ["Curiosity", "Play"])
        assert base != evaluator._cache_key("other", ["a", "b"], ["Play", "Curiosity"])
        assert base != evaluator._cache_key("desc", ["a"], ["Play", "Curiosity"])
        assert base != evaluator._cache_key("desc", ["a", "b"], ["Play"])

        evaluator._save_cache({"Play": 0.4, "Curiosity": 0.6}, datetime.now(), base)
        assert evaluator._load_cache(["Play", "Curiosity"], base) is not None
        other = evaluator._cache_key("other", ["a", "b"], ["Play", "Curiosity"])
        assert evaluator._load_cache(["Play", "Curiosity"], other) is None

    def test_evaluate_reuses_cache_only_for_the_same_inputs(self, monkeypatch):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp())
        calls = []

        def fake_local(prompt):
            calls.append(prompt)
            return json.dumps({"Play": {"score": 0.6, "reasoning": "x"}})

        monkeypatch.setattr(evaluator, "_call_local", fake_local)
        first = evaluator.evaluate_indicators(["Play"], None, "desc", ["s1"])
        again = evaluator.evaluate_indicators(["Play"], None, "desc", ["s1"])
        evaluator.evaluate_indicators(["Play"], None, "changed", ["s1"])
        assert first == again == {"Play": pytest.approx(0.6)}
        assert len(calls) == 2

    def test_claude_mode_requires_configured_model(self, monkeypatch):
        from mtc.assessment import dcm_evaluator
        from mtc.core.config import Settings

        monkeypatch.setattr(
            dcm_evaluator,
            "get_settings",
            lambda: Settings(ANTHROPIC_MODEL="", ANTHROPIC_API_KEY="test-key"),
        )
        evaluator = dcm_evaluator.DCMEvaluator(cache_dir=tempfile.mkdtemp())
        with pytest.raises(ValueError, match="ANTHROPIC_MODEL"):
            evaluator.evaluate_indicators(
                ["Play"], None, "desc", ["s1"], mode="claude", force=True
            )

    def test_no_model_literal_in_evaluator(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "mtc"
            / "assessment"
            / "dcm_evaluator.py"
        ).read_text()
        assert "claude-" not in source

    def test_batching(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp(), batch_size=3)
        indicators = [f"Ind_{i}" for i in range(10)]
        batches = evaluator._make_batches(indicators)
        assert len(batches) == 4  # 3+3+3+1
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1
