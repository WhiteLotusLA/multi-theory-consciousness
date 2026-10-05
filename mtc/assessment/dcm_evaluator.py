"""
DCM LLM-based Indicator Evaluator
==================================

Evaluates DCM indicators that can't be measured directly from modules.
Uses a local chat-completions server (``mode="local"``, the Settings LLM host
and port) or the Anthropic API (``mode="claude"``, model id from
``Settings.ANTHROPIC_MODEL``).

Data leaving the machine: ``mode="claude"`` sends the system description and
the conversation samples the adapter supplies to an external API. The local
mode keeps them on the configured host.

Framework-agnostic: no host-system-specific imports.
"""

import hashlib
import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from mtc.core.config import get_settings

logger = logging.getLogger(__name__)

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
            mode: "local" (chat-completions server) or "claude" (Anthropic
                API; sends the description and samples off the machine)
            force: Bypass cache

        Returns:
            Dict mapping indicator name -> score [0, 1]. An indicator the
            model did not score (failed batch, missing or unparseable entry)
            is left out rather than given a stand-in value; the benchmark
            then treats it as unmeasured. Only a complete evaluation is
            cached.

        Raises:
            ValueError: ``mode="claude"`` with ``Settings.ANTHROPIC_MODEL``
                empty.
        """
        if mode == "claude":
            self._claude_model()  # configuration errors surface here, not per batch

        cache_key = self._cache_key(
            system_description, conversation_samples, indicator_names
        )

        # Check cache first (unless forced)
        if not force:
            cached = self._load_cache(indicator_names, cache_key)
            if cached is not None:
                logger.info(f"DCM evaluator: loaded {len(cached)} cached scores")
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
                # The batch's indicators stay unscored; no stand-in value.
                logger.warning(f"DCM evaluator batch {i+1} failed: {e}")

        missing = [name for name in indicator_names if name not in all_scores]
        if missing:
            logger.warning(
                f"DCM evaluator: {len(missing)} indicators unscored, "
                "result not cached"
            )
        else:
            self._save_cache(all_scores, datetime.now(), cache_key)

        return all_scores

    def _build_batch_prompt(
        self,
        indicators: List[str],
        system_description: str,
        conversation_samples: List[str],
    ) -> str:
        """Build evaluation prompt for a batch of indicators."""
        conv_text = "\n".join(f"- {msg[:200]}" for msg in conversation_samples[:20])

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
        """Parse LLM response into indicator scores.

        Entries that are missing, non-numeric or non-finite are left out.
        """
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

            if not isinstance(data, dict):
                raise ValueError("evaluator response is not a JSON object")

            for name in indicator_names:
                entry = data.get(name)
                if isinstance(entry, dict):
                    entry = entry.get("score")
                if isinstance(entry, bool) or not isinstance(entry, (int, float, str)):
                    continue
                try:
                    value = float(entry)
                except ValueError:
                    continue
                if math.isfinite(value):
                    scores[name] = value

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse evaluator response: {e}")
            return {}

        # Clamp all scores to [0, 1]
        return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

    def _call_local(self, prompt: str) -> str:
        """Call local LLM (OpenAI-compatible API)."""
        import requests

        settings = get_settings()
        host = settings.LLM_HOST
        port = str(settings.LLM_PORT)
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

    @staticmethod
    def _claude_model() -> str:
        """The configured Anthropic model id; raises when it is empty."""
        model = get_settings().ANTHROPIC_MODEL
        if not model:
            raise ValueError(
                "Settings.ANTHROPIC_MODEL (env MTC_ANTHROPIC_MODEL) is empty; "
                "set it to a model id to use mode='claude'"
            )
        return model

    def _call_claude(self, prompt: str) -> str:
        """Call the Anthropic API (sends the prompt off the machine)."""
        import anthropic

        model = self._claude_model()
        api_key = get_settings().ANTHROPIC_API_KEY or None
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Required for mode='claude'.")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
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

    @staticmethod
    def _cache_key(
        system_description: str,
        conversation_samples: List[str],
        indicator_names: List[str],
    ) -> str:
        """sha256 over the description, the samples and the sorted indicators.

        A cached evaluation is reused only for the same evidence and the same
        indicator set.
        """
        payload = json.dumps(
            {
                "system_description": system_description,
                "conversation_samples": list(conversation_samples),
                "indicators": sorted(indicator_names),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_file(self, cache_key: str) -> Path:
        return self.cache_dir / f"eval_{cache_key[:32]}.json"

    def _save_cache(
        self, scores: Dict[str, float], timestamp: datetime, cache_key: str
    ) -> None:
        """Save evaluation scores to the cache file for ``cache_key``."""
        cache_file = self._cache_file(cache_key)
        data = {
            "timestamp": timestamp.isoformat(),
            "cache_key": cache_key,
            "scores": scores,
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_cache(
        self, indicator_names: List[str], cache_key: str
    ) -> Optional[Dict[str, float]]:
        """Load cached scores for ``cache_key`` if fresh and complete."""
        cache_file = self._cache_file(cache_key)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)
            if data.get("cache_key") != cache_key:
                return None

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
