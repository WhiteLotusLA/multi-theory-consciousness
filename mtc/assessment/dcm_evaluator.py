"""
DCM LLM-based Indicator Evaluator
==================================

Evaluates DCM indicators that can't be measured directly from modules.
Uses any OpenAI-compatible API (local LLM) or Anthropic API (Claude).

Framework-agnostic: no host-system-specific imports.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Default conservative score for indicators that can't be evaluated
DEFAULT_SCORE = 0.3

EVALUATION_SYSTEM_PROMPT = """You are an expert consciousness researcher evaluating whether an AI system exhibits specific consciousness-related indicators from the Digital Consciousness Model (DCM).

For each indicator, assess whether the system satisfies it based on:
1. The system's architecture description
2. Recent conversation samples showing the system's behavior
3. The indicator's definition

Rate each indicator from 0.0 (definitely absent) to 1.0 (strongly present).
Be calibrated: 0.5 means genuinely uncertain, not a default.

Respond with ONLY a JSON object mapping indicator names to objects with "score" (float) and "reasoning" (string, 1 sentence).

Example:
{
  "Curiosity": {"score": 0.7, "reasoning": "System asks unprompted questions about novel topics."},
  "Play": {"score": 0.2, "reasoning": "No evidence of playful or exploratory behavior without prompting."}
}"""


class DCMEvaluator:
    """LLM-based evaluator for DCM indicators."""

    def __init__(
        self,
        cache_dir: str = "data/dcm",
        batch_size: int = 10,
        cache_ttl_hours: int = 24,
    ):
        self.cache_dir = Path(cache_dir) / "eval_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

    def evaluate_indicators(
        self,
        indicator_names: List[str],
        spec: Any,
        system_description: str,
        conversation_samples: List[str],
        mode: str = "local",
        force: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate a list of indicators via LLM.

        Args:
            indicator_names: Indicators to evaluate
            spec: ParsedSpec (for indicator descriptions, if available)
            system_description: Architecture summary
            conversation_samples: Recent conversations
            mode: "local" (Qwen) or "claude" (Anthropic API)
            force: Bypass cache

        Returns:
            Dict mapping indicator name -> score [0, 1]
        """
        # Check cache first (unless forced)
        if not force:
            cached = self._load_cache(indicator_names)
            if cached is not None:
                logger.info(
                    f"DCM evaluator: loaded {len(cached)} cached scores"
                )
                return cached

        # Batch and evaluate
        all_scores: Dict[str, float] = {}
        batches = self._make_batches(indicator_names)

        for i, batch in enumerate(batches):
            logger.info(
                f"DCM evaluator: batch {i+1}/{len(batches)} "
                f"({len(batch)} indicators, mode={mode})"
            )
            prompt = self._build_batch_prompt(
                batch, system_description, conversation_samples
            )

            try:
                if mode == "claude":
                    response = self._call_claude(prompt)
                else:
                    response = self._call_local(prompt)

                scores = self._parse_response(response, batch)
                all_scores.update(scores)
            except Exception as e:
                logger.warning(f"DCM evaluator batch {i+1} failed: {e}")
                # Default scores for failed batch
                for name in batch:
                    all_scores[name] = DEFAULT_SCORE

        # Cache results
        self._save_cache(all_scores, datetime.now())

        return all_scores

    def _build_batch_prompt(
        self,
        indicators: List[str],
        system_description: str,
        conversation_samples: List[str],
    ) -> str:
        """Build evaluation prompt for a batch of indicators."""
        conv_text = "\n".join(
            f"- {msg[:200]}" for msg in conversation_samples[:20]
        )

        indicator_list = "\n".join(f"- {name}" for name in indicators)

        return (
            f"## System Under Evaluation\n\n{system_description}\n\n"
            f"## Recent Conversation Samples\n\n{conv_text}\n\n"
            f"## Indicators to Evaluate\n\n{indicator_list}\n\n"
            f"Evaluate each indicator and respond with JSON only."
        )

    def _parse_response(
        self, response: str, indicator_names: List[str]
    ) -> Dict[str, float]:
        """Parse LLM response into indicator scores."""
        scores: Dict[str, float] = {}

        try:
            # Try to extract JSON from response
            text = response.strip()
            # Handle markdown code blocks
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    text = text[start:end]

            data = json.loads(text)

            for name in indicator_names:
                if name in data:
                    entry = data[name]
                    if isinstance(entry, dict):
                        scores[name] = float(entry.get("score", DEFAULT_SCORE))
                    elif isinstance(entry, (int, float)):
                        scores[name] = float(entry)
                    else:
                        scores[name] = DEFAULT_SCORE
                else:
                    scores[name] = DEFAULT_SCORE

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse evaluator response: {e}")
            for name in indicator_names:
                scores[name] = DEFAULT_SCORE

        # Clamp all scores to [0, 1]
        return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

    def _call_local(self, prompt: str) -> str:
        """Call local LLM (OpenAI-compatible API)."""
        import requests

        host = os.environ.get("LLAMA_CPP_HOST", os.environ.get("MLX_LLM_HOST", "localhost"))
        port = os.environ.get("LLAMA_CPP_PORT", os.environ.get("MLX_LLM_PORT", "8080"))
        url = f"http://{host}:{port}/v1/chat/completions"

        resp = requests.post(
            url,
            json={
                "model": "default",
                "messages": [
                    {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_claude(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Required for mode='claude'."
            )

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=EVALUATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _make_batches(self, items: List[str]) -> List[List[str]]:
        """Split items into batches."""
        return [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

    def _save_cache(
        self, scores: Dict[str, float], timestamp: datetime
    ) -> None:
        """Save evaluation scores to cache."""
        cache_file = self.cache_dir / "latest_eval.json"
        data = {
            "timestamp": timestamp.isoformat(),
            "scores": scores,
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_cache(
        self, indicator_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Load cached scores if fresh enough and covers all indicators."""
        cache_file = self.cache_dir / "latest_eval.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - cached_time > self.cache_ttl:
                return None  # Expired

            scores = data["scores"]
            # Check if cache covers all requested indicators
            if all(name in scores for name in indicator_names):
                return {name: scores[name] for name in indicator_names}

            return None  # Incomplete coverage
        except Exception:
            return None
