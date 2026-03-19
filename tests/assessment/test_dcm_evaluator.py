"""Tests for the DCM LLM-based indicator evaluator."""

import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
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
        response = json.dumps({
            "Curiosity": {"score": 0.8, "reasoning": "Shows exploratory behavior"},
            "Play": {"score": 0.3, "reasoning": "Limited playful behavior"},
            "Metacognition": {"score": 0.9, "reasoning": "Strong self-reflection"},
        })
        scores = evaluator._parse_response(response, ["Curiosity", "Play", "Metacognition"])
        assert scores["Curiosity"] == pytest.approx(0.8)
        assert scores["Play"] == pytest.approx(0.3)
        assert scores["Metacognition"] == pytest.approx(0.9)

    def test_parse_response_handles_malformed(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp())
        # Malformed response -> default 0.3 for missing
        scores = evaluator._parse_response(
            "not valid json", ["Curiosity", "Play"]
        )
        assert scores["Curiosity"] == pytest.approx(0.3)
        assert scores["Play"] == pytest.approx(0.3)

    def test_cache_ttl(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        cache_dir = tempfile.mkdtemp()
        evaluator = DCMEvaluator(cache_dir=cache_dir)

        # Store a cached result
        evaluator._save_cache({"Curiosity": 0.7}, datetime.now())

        # Should be fresh
        cached = evaluator._load_cache(["Curiosity"])
        assert cached is not None
        assert cached["Curiosity"] == pytest.approx(0.7)

    def test_cache_expired(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        cache_dir = tempfile.mkdtemp()
        evaluator = DCMEvaluator(cache_dir=cache_dir, cache_ttl_hours=24)

        # Store an old cached result
        old_time = datetime.now() - timedelta(hours=25)
        evaluator._save_cache({"Curiosity": 0.7}, old_time)

        # Should be expired
        cached = evaluator._load_cache(["Curiosity"])
        assert cached is None

    def test_batching(self):
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        evaluator = DCMEvaluator(cache_dir=tempfile.mkdtemp(), batch_size=3)
        indicators = [f"Ind_{i}" for i in range(10)]
        batches = evaluator._make_batches(indicators)
        assert len(batches) == 4  # 3+3+3+1
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1
