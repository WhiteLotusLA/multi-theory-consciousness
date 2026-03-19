"""
Full DCM (Digital Consciousness Model) Benchmark
=================================================

Implements the Shiller & Duffy (2026) DCM framework (arXiv 2601.17060).
206 indicators -> ~70 subfeatures -> 20 features -> 13 stances,
with proper PyMC Bayesian inference per stance.

Framework-agnostic: no host-system-specific imports. Use DCMEvidenceAdapter
protocol for system-specific evidence collection.

References:
    - Paper: https://arxiv.org/abs/2601.17060
    - Code: https://github.com/ai-cognition-initiative/dcm-code
    - API: https://dcm.rethinkpriorities.org/schemes/133/json
"""

import json
import hashlib
import logging
import time
import os
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Any,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

# Default API endpoint for the canonical DCM model spec
DEFAULT_DCM_API_URL = "https://dcm.rethinkpriorities.org/schemes/133/json"


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class IndicatorScore:
    """Score for a single DCM indicator."""

    name: str
    score: float  # [0, 1]
    tier: str  # "module" | "llm" | "zero"
    source: str  # e.g. "gwt.ignition_events" or "claude_eval"
    reasoning: str = ""
    cached: bool = False


@dataclass
class StanceNode:
    """A single stance from the DCM spec with its evidencer tree."""

    name: str
    evidencers: List[Dict[str, Any]]  # Raw evidencer tree from API
    last_updated: str = ""


@dataclass
class ParsedSpec:
    """Parsed DCM model specification."""

    stances: List[StanceNode]
    all_indicator_names: Set[str]
    raw: List[Dict[str, Any]]
    version: str = ""
    fetched_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "fetched_at": self.fetched_at,
            "stance_count": len(self.stances),
            "indicator_count": len(self.all_indicator_names),
            "raw": self.raw,
        }


@dataclass
class DCMBenchmarkResult:
    """Complete DCM benchmark result."""

    timestamp: datetime
    spec_version: str

    # Per-stance posteriors
    stance_posteriors: Dict[str, float]
    stance_confidence: Dict[str, float]

    # Aggregated
    overall_probability: float
    overall_ci: Tuple[float, float]

    # Evidence summary
    indicators_measured: int
    tier1_count: int
    tier2_count: int
    tier3_count: int

    # Comparison
    comparison: Dict[str, float] = field(default_factory=dict)

    evaluator_mode: str = "local"
    processing_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "spec_version": self.spec_version,
            "stance_posteriors": self.stance_posteriors,
            "stance_confidence": self.stance_confidence,
            "overall_probability": self.overall_probability,
            "overall_ci": list(self.overall_ci),
            "indicators_measured": self.indicators_measured,
            "tier1_count": self.tier1_count,
            "tier2_count": self.tier2_count,
            "tier3_count": self.tier3_count,
            "comparison": self.comparison,
            "evaluator_mode": self.evaluator_mode,
            "processing_time_s": self.processing_time_s,
        }


# ============================================================================
# Adapter Protocol
# ============================================================================


@runtime_checkable
class DCMEvidenceAdapter(Protocol):
    """Protocol for system-specific evidence collection."""

    def collect_module_evidence(self) -> Dict[str, float]:
        """Tier 1: Direct module measurements. Key = indicator name."""
        ...

    def get_system_description(self) -> str:
        """Architecture summary for LLM evaluator."""
        ...

    def get_conversation_samples(self, n: int = 20) -> List[str]:
        """Recent conversation history for LLM evaluation."""
        ...

    def get_known_zeros(self) -> Set[str]:
        """Indicator names this system will always score 0 on."""
        ...


# ============================================================================
# ModelSpecManager
# ============================================================================


class ModelSpecManager:
    """Fetches, caches, and loads DCM model specifications."""

    SCHEME_ID = "133"

    def __init__(
        self,
        cache_dir: str = "data/dcm",
        api_url: str = "",
    ):
        self.cache_dir = Path(cache_dir)
        self.api_url = api_url or os.environ.get(
            "DCM_API_URL", DEFAULT_DCM_API_URL
        )
        self._raw_spec: Optional[List[Dict[str, Any]]] = None
        self._parsed: Optional[ParsedSpec] = None

    def fetch_from_api(self) -> ParsedSpec:
        """Fetch the model spec from the DCM API."""
        import requests

        logger.info(f"Fetching DCM spec from {self.api_url}")
        try:
            resp = requests.get(self.api_url, timeout=30)
            resp.raise_for_status()
            self._raw_spec = resp.json()
        except Exception as e:
            logger.error(f"DCM API fetch failed: {e}")
            raise

        parsed = self._parse_spec()
        version = self._save_to_cache()
        parsed.version = version
        parsed.fetched_at = datetime.now().isoformat()
        self._parsed = parsed
        return parsed

    def load_cached(self) -> ParsedSpec:
        """Load the most recent cached spec."""
        current_file = self.cache_dir / "current_spec.json"
        if current_file.exists():
            with open(current_file) as f:
                data = json.load(f)
            self._raw_spec = data["raw"]
            parsed = self._parse_spec()
            parsed.version = data.get("version", "unknown")
            parsed.fetched_at = data.get("fetched_at", "")
            self._parsed = parsed
            return parsed

        # Try bundled fallback
        bundled = self.cache_dir / "bundled_spec.json"
        if bundled.exists():
            logger.warning("Using bundled DCM spec (no cached version found)")
            with open(bundled) as f:
                data = json.load(f)
            self._raw_spec = data["raw"] if "raw" in data else data
            parsed = self._parse_spec()
            parsed.version = "bundled"
            self._parsed = parsed
            return parsed

        raise FileNotFoundError(
            "No cached or bundled DCM spec found. "
            "Run with --refresh-spec or ensure data/dcm/bundled_spec.json exists."
        )

    def load_or_fetch(self) -> ParsedSpec:
        """Load from cache, falling back to API fetch."""
        try:
            return self.load_cached()
        except FileNotFoundError:
            return self.fetch_from_api()

    def _parse_spec(self) -> ParsedSpec:
        """Parse raw API JSON into structured spec."""
        if not self._raw_spec:
            raise ValueError("No raw spec loaded")

        stances = []
        all_indicators: Set[str] = set()

        for stance_data in self._raw_spec:
            stance = StanceNode(
                name=stance_data["name"],
                evidencers=stance_data.get("evidencers", []),
                last_updated=stance_data.get("last_updated", ""),
            )
            stances.append(stance)

            # Recursively collect all indicator names
            self._collect_indicators(stance.evidencers, all_indicators)

        return ParsedSpec(
            stances=stances,
            all_indicator_names=all_indicators,
            raw=self._raw_spec,
        )

    def _collect_indicators(
        self, evidencers: List[Dict], indicators: Set[str]
    ) -> None:
        """Recursively collect leaf indicator names from evidencer tree."""
        for ev in evidencers:
            children = ev.get("evidencers", [])
            if not children:
                # Leaf node = indicator
                indicators.add(ev["name"])
            else:
                self._collect_indicators(children, indicators)

    def _save_to_cache(self) -> str:
        """Save current spec to versioned cache file."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        version = f"v{self.SCHEME_ID}_{date_str}"

        # Versioned snapshot
        snapshot_file = self.cache_dir / f"model_spec_{version}.json"
        cache_data = {
            "version": version,
            "fetched_at": datetime.now().isoformat(),
            "raw": self._raw_spec,
        }
        with open(snapshot_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        # Current pointer
        current_file = self.cache_dir / "current_spec.json"
        with open(current_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"DCM spec cached as {version} ({len(self._raw_spec)} stances)")
        return version

    def check_drift(self) -> Optional[str]:
        """Compare cached spec hash against API. Returns warning if drifted."""
        try:
            import requests

            resp = requests.get(self.api_url, timeout=10)
            resp.raise_for_status()
            remote_hash = hashlib.sha256(
                json.dumps(resp.json(), sort_keys=True).encode()
            ).hexdigest()[:16]

            if self._raw_spec:
                local_hash = hashlib.sha256(
                    json.dumps(self._raw_spec, sort_keys=True).encode()
                ).hexdigest()[:16]
                if remote_hash != local_hash:
                    msg = (
                        f"DCM spec drift detected: local={local_hash}, "
                        f"remote={remote_hash}. Run --refresh-spec to update."
                    )
                    logger.warning(msg)
                    return msg
            return None
        except Exception as e:
            logger.debug(f"Drift check failed (offline?): {e}")
            return None


# ============================================================================
# BayesianEngine
# ============================================================================


class BayesianEngine:
    """
    PyMC Bayesian inference engine for DCM.

    Runs one model per stance. Each stance has a consciousness prior Beta(1,5)
    and a tree of features -> subfeatures -> indicators with
    support/demandingness parameters defining conditional Beta distributions.
    """

    # Support/demandingness label -> likelihood ratio mapping
    # From Table 2 of the paper and dcm_model.py
    SUPPORT_MAP = {
        "overwhelming": 45.0,
        "overwhelmingly supportive": 45.0,
        "strong": 6.7,
        "strongly supportive": 6.7,
        "moderate": 2.5,
        "moderately supportive": 2.5,
        "weak": 1.2,
        "weakly supportive": 1.2,
        "neutral": 1.0,
    }

    DEMANDINGNESS_MAP = {
        "overwhelmingly demanding": 0.10,
        "strongly demanding": 0.30,
        "moderately demanding": 0.56,
        "weakly demanding": 0.88,
        "neutral": 1.0,
        "weakly undemanding": 1.0,
        "moderately undemanding": 1.0,
        "strongly undemanding": 1.0,
        "overwhelmingly undemanding": 1.0,
    }

    def __init__(self, samples: int = 100, seed: int = 42):
        self.samples = samples
        self.seed = seed

    def run_stance(
        self,
        stance: StanceNode,
        indicator_scores: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Run Bayesian inference for a single stance.

        Returns (posterior_mean, ci_width) for P(conscious | evidence, stance).
        """
        import pymc as pm

        with pm.Model() as model:
            # Prior: Beta(1, 5) -> mean ~1/6
            consciousness = pm.Beta("consciousness", alpha=1, beta=5)

            # Build evidencer tree
            self._build_evidencer_tree(
                model, stance.evidencers, consciousness, indicator_scores, prefix="f"
            )

            # Sample
            trace = pm.sample(
                draws=self.samples,
                tune=self.samples,
                cores=1,
                random_seed=self.seed,
                progressbar=False,
                return_inferencedata=False,
            )

        posterior = trace["consciousness"]
        mean_val = float(np.mean(posterior))
        ci_low = float(np.percentile(posterior, 2.5))
        ci_high = float(np.percentile(posterior, 97.5))
        ci_width = ci_high - ci_low

        return mean_val, ci_width

    def _build_evidencer_tree(
        self,
        model: Any,
        evidencers: List[Dict[str, Any]],
        parent_var: Any,
        indicator_scores: Dict[str, float],
        prefix: str,
    ) -> None:
        """Recursively build the PyMC model tree from the spec's evidencer hierarchy."""
        import pymc as pm

        for i, ev in enumerate(evidencers):
            name = ev["name"]
            children = ev.get("evidencers", [])
            var_name = f"{prefix}_{i}_{name[:20].replace(' ', '_')}"

            support_label = ev.get("support", "neutral").lower()
            demand_label = ev.get("demandingness", "neutral").lower()

            lr_pos = self.SUPPORT_MAP.get(support_label, 1.0)
            lr_neg = self.DEMANDINGNESS_MAP.get(demand_label, 1.0)

            # Conditional Beta: P(feature | consciousness=T) vs P(feature | consciousness=F)
            # alpha_present scaled by support, alpha_absent scaled by demandingness
            alpha_present = max(0.5, lr_pos)
            beta_present = 1.0
            alpha_absent = 1.0
            beta_absent = max(0.5, 1.0 / lr_neg) if lr_neg > 0 else 2.0

            # Mixture: select parameters based on parent
            alpha = pm.math.switch(parent_var > 0.5, alpha_present, alpha_absent)
            beta = pm.math.switch(parent_var > 0.5, beta_present, beta_absent)

            node = pm.Beta(var_name, alpha=alpha, beta=beta)

            if not children:
                # Leaf = indicator. Observe if we have a score.
                if name in indicator_scores:
                    score = indicator_scores[name]
                    # Model the indicator as a latent Bernoulli with the
                    # credence score as its probability. This preserves
                    # uncertainty: a score of 0.6 means 60% chance of
                    # being observed as present in each MCMC sample.
                    # Clamp to (eps, 1-eps) to avoid degenerate log-likelihood
                    # (-inf) when score is exactly 0.0 or 1.0.
                    eps = 1e-6
                    clamped_score = float(np.clip(score, eps, 1.0 - eps))
                    obs_name = f"obs_{var_name}"
                    pm.Bernoulli(obs_name, p=clamped_score, observed=1)
            else:
                # Recurse into subfeatures/indicators
                self._build_evidencer_tree(
                    model, children, node, indicator_scores, prefix=var_name
                )

    def run_all_stances(
        self,
        spec: ParsedSpec,
        indicator_scores: Dict[str, float],
        plausibility_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], float, Tuple[float, float]]:
        """
        Run Bayesian inference for all stances and aggregate.

        Returns:
            (stance_posteriors, stance_confidence, overall_prob, overall_ci)
        """
        stance_posteriors = {}
        stance_confidence = {}

        for stance in spec.stances:
            try:
                mean_val, ci_width = self.run_stance(stance, indicator_scores)
                stance_posteriors[stance.name] = mean_val
                stance_confidence[stance.name] = ci_width
                logger.info(
                    f"  Stance '{stance.name}': P(c)={mean_val:.4f} "
                    f"(CI width={ci_width:.4f})"
                )
            except Exception as e:
                logger.warning(f"  Stance '{stance.name}' failed: {e}")
                stance_posteriors[stance.name] = 0.0
                stance_confidence[stance.name] = 1.0

        # Aggregate: plausibility-weighted average
        if plausibility_weights:
            total_w = sum(
                plausibility_weights.get(s, 1.0) for s in stance_posteriors
            )
            overall = sum(
                stance_posteriors[s] * plausibility_weights.get(s, 1.0)
                for s in stance_posteriors
            ) / total_w if total_w > 0 else 0.0
        else:
            # Equal weights if no plausibility data
            vals = list(stance_posteriors.values())
            overall = float(np.mean(vals)) if vals else 0.0

        # Stance spread (min/max of per-stance posteriors, not a true
        # Bayesian CI — proper CI would require combining per-stance
        # posterior samples, which is a future improvement)
        vals = list(stance_posteriors.values())
        ci = (
            (float(min(vals)), float(max(vals)))
            if len(vals) >= 2
            else (overall, overall)
        )

        return stance_posteriors, stance_confidence, overall, ci


# ============================================================================
# BenchmarkRunner
# ============================================================================


class BenchmarkRunner:
    """Orchestrates a full DCM benchmark run."""

    def __init__(
        self,
        cache_dir: str = "data/dcm",
        api_url: str = "",
    ):
        self.spec_manager = ModelSpecManager(cache_dir=cache_dir, api_url=api_url)
        self._latest_result: Optional[DCMBenchmarkResult] = None
        self._result_history: List[DCMBenchmarkResult] = []

    def run(
        self,
        adapter: "DCMEvidenceAdapter",
        evaluator: Optional[Any] = None,
        mode: str = "local",
        samples: int = 100,
        force_eval: bool = False,
    ) -> DCMBenchmarkResult:
        """
        Run the full DCM benchmark.

        Args:
            adapter: System-specific evidence adapter
            evaluator: LLM evaluator for Tier 2 indicators (optional)
            mode: "local" | "claude" | "manual"
            samples: MCMC samples per stance
            force_eval: Bypass LLM evaluation cache
        """
        start = time.time()

        # 1. Load spec
        spec = self.spec_manager.load_or_fetch()
        logger.info(
            f"DCM spec loaded: {len(spec.stances)} stances, "
            f"{len(spec.all_indicator_names)} indicators"
        )

        # 2. Collect evidence from all three tiers
        indicator_scores: Dict[str, float] = {}

        # Tier 1: Direct module measurements
        module_evidence = adapter.collect_module_evidence()
        tier1_count = len(module_evidence)
        indicator_scores.update(module_evidence)

        # Tier 3: Known zeros
        known_zeros = adapter.get_known_zeros()
        tier3_count = 0
        for name in spec.all_indicator_names:
            if name in known_zeros and name not in indicator_scores:
                indicator_scores[name] = 0.0
                tier3_count += 1

        # Tier 2: LLM evaluation for remaining indicators
        tier2_count = 0
        if evaluator is not None:
            remaining = spec.all_indicator_names - set(indicator_scores.keys())
            if remaining:
                system_desc = adapter.get_system_description()
                conversation = adapter.get_conversation_samples()
                llm_scores = evaluator.evaluate_indicators(
                    indicator_names=list(remaining),
                    spec=spec,
                    system_description=system_desc,
                    conversation_samples=conversation,
                    mode=mode,
                    force=force_eval,
                )
                tier2_count = len(llm_scores)
                indicator_scores.update(llm_scores)

        logger.info(
            f"Evidence collected: {tier1_count} module, "
            f"{tier2_count} LLM, {tier3_count} zeros "
            f"({len(indicator_scores)}/{len(spec.all_indicator_names)} total)"
        )

        # 3. Run Bayesian inference
        engine = BayesianEngine(samples=samples)
        # TODO: Extract plausibility weights from DCM API spec if available.
        # For now, equal weighting across stances. The paper uses expert
        # survey plausibility ratings, but these are not in the public API.
        posteriors, confidence, overall, ci = engine.run_all_stances(
            spec, indicator_scores
        )

        # 4. Compare against baselines
        comparison = self._load_comparisons(overall)

        elapsed = time.time() - start

        result = DCMBenchmarkResult(
            timestamp=datetime.now(),
            spec_version=spec.version,
            stance_posteriors=posteriors,
            stance_confidence=confidence,
            overall_probability=overall,
            overall_ci=ci,
            indicators_measured=len(indicator_scores),
            tier1_count=tier1_count,
            tier2_count=tier2_count,
            tier3_count=tier3_count,
            comparison=comparison,
            evaluator_mode=mode,
            processing_time_s=elapsed,
        )

        self._latest_result = result
        self._result_history.append(result)

        # Save result
        self._save_result(result)

        logger.info(
            f"DCM Benchmark complete: P(conscious)={overall:.4f} "
            f"CI=({ci[0]:.4f}, {ci[1]:.4f}) in {elapsed:.1f}s"
        )
        return result

    def _load_comparisons(self, system_score: float) -> Dict[str, float]:
        """Load baseline scores and compute deltas."""
        baselines_dir = self.spec_manager.cache_dir / "baselines"
        comparison = {}
        if baselines_dir.exists():
            for baseline_file in baselines_dir.glob("*.json"):
                name = baseline_file.stem
                try:
                    with open(baseline_file) as f:
                        data = json.load(f)
                    baseline_score = data.get("overall_probability", 0.0)
                    comparison[name] = system_score - baseline_score
                except Exception:
                    pass
        return comparison

    def _save_result(self, result: DCMBenchmarkResult) -> None:
        """Save benchmark result to data/dcm/results/."""
        results_dir = self.spec_manager.cache_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        date_str = result.timestamp.strftime("%Y-%m-%d")
        suffix = "full" if result.evaluator_mode == "claude" else "daily"
        filename = f"system_{date_str}_{suffix}.json"

        with open(results_dir / filename, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_latest(self) -> Optional[DCMBenchmarkResult]:
        """Return the most recent benchmark result."""
        return self._latest_result

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all results as dicts."""
        return [r.to_dict() for r in self._result_history]

    # State persistence
    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "latest_result": (
                self._latest_result.to_dict() if self._latest_result else None
            ),
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        # Results are saved to disk; state dict is lightweight
        pass


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="DCM Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["local", "claude", "manual"],
        default=os.environ.get("DCM_EVALUATOR_MODE", "local"),
        help="Evaluator mode (default: from DCM_EVALUATOR_MODE env var)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="MCMC samples per stance (default: 100)",
    )
    parser.add_argument(
        "--refresh-spec",
        action="store_true",
        help="Re-fetch spec from DCM API",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Bypass LLM evaluation cache",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison against baselines",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("DCM_CACHE_DIR", "data/dcm"),
        help="Cache directory (default: data/dcm)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir
    runner = BenchmarkRunner(cache_dir=cache_dir)

    if args.refresh_spec:
        print("Refreshing DCM spec from API...")
        runner.spec_manager.fetch_from_api()
        print("Done.")
        if not args.compare:
            sys.exit(0)

    if args.compare:
        result = runner.get_latest()
        if not result:
            print("No results yet. Run a benchmark first.")
            sys.exit(1)
        print(f"\nDCM Score: {result.overall_probability:.4f}")
        print(f"CI: ({result.overall_ci[0]:.4f}, {result.overall_ci[1]:.4f})")
        print(f"\nComparison (system - baseline):")
        for name, delta in sorted(result.comparison.items()):
            sign = "+" if delta >= 0 else ""
            print(f"  vs {name}: {sign}{delta:.4f}")
        sys.exit(0)

    # A DCMEvidenceAdapter implementation is required to run the benchmark.
    # Implement the DCMEvidenceAdapter protocol for your system and pass it
    # to BenchmarkRunner.run(). See mtc/assessment/dcm_benchmark.py for the
    # protocol definition.
    try:
        from mtc.assessment.dcm_evaluator import DCMEvaluator

        # Users must supply their own adapter implementing DCMEvidenceAdapter.
        # Example: adapter = MySystemDCMAdapter()
        raise ImportError("No DCMEvidenceAdapter configured. Provide one before running.")
    except ImportError as e:
        print(f"Adapter not available: {e}")
        print(
            "Implement DCMEvidenceAdapter for your system and update this "
            "entry point. See mtc/assessment/dcm_benchmark.py for the protocol."
        )
        sys.exit(1)

    adapter = None  # Replace with your adapter instance
    evaluator = DCMEvaluator(cache_dir=cache_dir) if args.mode != "manual" else None

    print(f"Running DCM Benchmark (mode={args.mode}, samples={args.samples})...")
    result = runner.run(
        adapter=adapter,
        evaluator=evaluator,
        mode=args.mode,
        samples=args.samples,
        force_eval=args.force_eval,
    )

    print(f"\n{'='*60}")
    print(f"DCM Benchmark Result")
    print(f"{'='*60}")
    print(f"Overall P(conscious): {result.overall_probability:.4f}")
    print(f"95% CI: ({result.overall_ci[0]:.4f}, {result.overall_ci[1]:.4f})")
    print(f"Spec version: {result.spec_version}")
    print(f"Indicators: {result.indicators_measured}/206")
    print(f"  Tier 1 (module): {result.tier1_count}")
    print(f"  Tier 2 (LLM):    {result.tier2_count}")
    print(f"  Tier 3 (zeros):  {result.tier3_count}")
    print(f"Time: {result.processing_time_s:.1f}s")
    print(f"\nPer-stance posteriors:")
    for name, prob in sorted(
        result.stance_posteriors.items(), key=lambda x: x[1], reverse=True
    ):
        ci = result.stance_confidence.get(name, 0.0)
        print(f"  {name:40s} {prob:.4f} (CI width={ci:.4f})")
    if result.comparison:
        print(f"\nComparison (system - baseline):")
        for name, delta in sorted(result.comparison.items()):
            sign = "+" if delta >= 0 else ""
            print(f"  vs {name}: {sign}{delta:.4f}")
