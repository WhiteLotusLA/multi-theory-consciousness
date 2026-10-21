"""
DCM (Digital Consciousness Model) Benchmark
===========================================

Follows the Shiller & Duffy (2026) DCM scheme (arXiv 2601.17060): 206
indicators -> ~70 subfeatures -> 20 features -> 13 stances.

By default a run collects and reports the indicator evidence of the three
tiers (module measurements, the evaluator, known zeros) and computes no
posterior: ``overall_probability`` is None and ``posterior_status`` is
``not_computed``. ``run(compute_posterior=True)`` (the command line flag
``--experimental-posterior``) also runs the PyMC engine per stance; its
number does not respond to the evidence (see ``BayesianEngine``) and is
labelled ``experimental``. The engine needs pymc (the research extra);
without it the run raises ``DCMEngineUnavailable``. A stance that fails is
logged, counted under ``dcm_stance``, named in ``failed_stances`` and left
out of the aggregate (re-raised when ``Settings.DEBUG`` is true); with no
successful stance the overall and its interval are None and nothing is
saved.

Framework-agnostic: no host-system-specific imports. Use DCMEvidenceAdapter
protocol for system-specific evidence collection; the reference adapter for
MTC's own module set lives in ``mtc.assessment.dcm_reference_adapter``.

File locations: without an explicit ``cache_dir`` the cache is
``DCM_CACHE_DIR`` resolved under ``PROJECT_ROOT`` at call time (fetched specs
and benchmark results are written there, the directory created on the first
write), and the shipped ``data/dcm/bundled_spec.json`` and
``data/dcm/baselines/`` are read through ``mtc.core.resources.resolve_shipped``
(the project copy, else the packaged copy). An explicit ``cache_dir`` holds
everything, as before: the spec, the bundled fallback, baselines and results.

Baselines: ``data/dcm/baselines/*.json`` hold published comparison values.
A baseline marked ``"approximate": true`` (all four shipped ones: they are
unverified) is skipped by ``_load_comparisons`` unless asked, so no delta
against an unverified value is printed.

Command line:
    python -m mtc.assessment.dcm_benchmark [--adapter package.module:callable]
        [--state-dir PATH] [--mode local|claude|manual] [--compare]
        [--experimental-posterior] ...

References:
    - Paper: https://arxiv.org/abs/2601.17060
    - Code: https://github.com/ai-cognition-initiative/dcm-code
    - API: the scheme endpoint in ``Settings.DCM_API_URL``
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from mtc.core.config import get_settings
from mtc.core.paths import resolve_path
from mtc.core.resources import resolve_shipped

logger = logging.getLogger(__name__)

# Adapter factory the CLI loads when --adapter is not given
DEFAULT_ADAPTER = "mtc.assessment.dcm_reference_adapter:build_default_adapter"

# posterior_status values of a result
POSTERIOR_NOT_COMPUTED = "not_computed"
POSTERIOR_EXPERIMENTAL = "experimental"
POSTERIOR_STATUSES = (POSTERIOR_NOT_COMPUTED, POSTERIOR_EXPERIMENTAL)

# Exit code of the command line for a missing extra (a configuration error,
# the code the other entry points return for it)
EXIT_USAGE = 2

# Failure counter step of a stance whose inference raised
STANCE_FAILURE_STEP = "dcm_stance"

ENGINE_UNAVAILABLE_MESSAGE = (
    "the DCM posterior needs pymc: "
    "pip install 'multi-theory-consciousness[research]'"
)
LIMITATIONS_POINTER = "see the DCM benchmark entry of docs/HONEST_LIMITATIONS.md"
EXPERIMENTAL_NOTE = (
    "This number does not respond to the indicator evidence: the engine "
    "returns the prior mean (about 0.167) for any system."
)


class DCMEngineUnavailable(ImportError):
    """The PyMC engine cannot run because pymc is not installed."""


def pymc_available() -> bool:
    """True when pymc can be imported (it is never imported here)."""
    try:
        return importlib.util.find_spec("pymc") is not None
    except (ImportError, ValueError):
        return False


def _finite_real(value: Any) -> Optional[float]:
    """A finite real number as float, else None (bools are not numbers)."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _format_probability(value: Optional[float]) -> str:
    return "not computed" if value is None else f"{value:.4f}"


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
    """A DCM benchmark result.

    ``overall_probability`` and ``overall_ci`` are None when no posterior
    was computed (``posterior_status`` ``not_computed``, the default run) or
    when every stance failed. ``failed_stances`` maps a failed stance to its
    exception type and ``failures`` counts failures by step.
    """

    timestamp: datetime
    spec_version: str

    # Per-stance posteriors (successful stances only)
    stance_posteriors: Dict[str, float]
    stance_confidence: Dict[str, float]

    # Aggregated
    overall_probability: Optional[float]
    overall_ci: Optional[Tuple[float, float]]

    # Evidence summary
    indicators_measured: int
    tier1_count: int
    tier2_count: int
    tier3_count: int

    # Comparison
    comparison: Dict[str, float] = field(default_factory=dict)

    evaluator_mode: str = "local"
    processing_time_s: float = 0.0
    posterior_status: Optional[str] = None
    failed_stances: Dict[str, str] = field(default_factory=dict)
    failures: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A result built without a status reads as experimental only when it
        # holds a probability.
        if self.posterior_status is None:
            self.posterior_status = (
                POSTERIOR_EXPERIMENTAL
                if self.overall_probability is not None
                else POSTERIOR_NOT_COMPUTED
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "spec_version": self.spec_version,
            "stance_posteriors": self.stance_posteriors,
            "stance_confidence": self.stance_confidence,
            "overall_probability": self.overall_probability,
            "overall_ci": (
                list(self.overall_ci) if self.overall_ci is not None else None
            ),
            "posterior_status": self.posterior_status,
            "failed_stances": dict(self.failed_stances),
            "failures": dict(self.failures),
            "indicators_measured": self.indicators_measured,
            "tier1_count": self.tier1_count,
            "tier2_count": self.tier2_count,
            "tier3_count": self.tier3_count,
            "comparison": self.comparison,
            "evaluator_mode": self.evaluator_mode,
            "processing_time_s": self.processing_time_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DCMBenchmarkResult":
        """Rebuild a result from ``to_dict()`` output.

        ``overall_probability`` must be present; null means not computed. A
        record written before ``posterior_status`` existed is experimental
        when it holds a probability.
        """
        raw_overall = data["overall_probability"]
        overall = None if raw_overall is None else float(raw_overall)
        raw_ci = data.get("overall_ci")
        ci = (
            (float(raw_ci[0]), float(raw_ci[1]))
            if raw_ci is not None and overall is not None
            else None
        )
        status = data.get("posterior_status")
        if status is None:
            status = (
                POSTERIOR_EXPERIMENTAL
                if overall is not None
                else POSTERIOR_NOT_COMPUTED
            )
        if status not in POSTERIOR_STATUSES:
            raise ValueError(f"unknown posterior_status {status!r}")
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            spec_version=str(data.get("spec_version", "")),
            stance_posteriors={
                str(k): float(v) for k, v in data.get("stance_posteriors", {}).items()
            },
            stance_confidence={
                str(k): float(v) for k, v in data.get("stance_confidence", {}).items()
            },
            overall_probability=overall,
            overall_ci=ci,
            indicators_measured=int(data.get("indicators_measured", 0)),
            tier1_count=int(data.get("tier1_count", 0)),
            tier2_count=int(data.get("tier2_count", 0)),
            tier3_count=int(data.get("tier3_count", 0)),
            comparison={
                str(k): float(v) for k, v in data.get("comparison", {}).items()
            },
            evaluator_mode=str(data.get("evaluator_mode", "local")),
            processing_time_s=float(data.get("processing_time_s", 0.0)),
            posterior_status=status,
            failed_stances={
                str(k): str(v) for k, v in (data.get("failed_stances") or {}).items()
            },
            failures={str(k): int(v) for k, v in (data.get("failures") or {}).items()},
        )


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
        cache_dir: Optional[str] = None,
        api_url: str = "",
    ):
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.api_url = api_url or get_settings().DCM_API_URL
        self._raw_spec: Optional[List[Dict[str, Any]]] = None
        self._parsed: Optional[ParsedSpec] = None

    @property
    def cache_dir(self) -> Path:
        """The explicit cache directory, else ``DCM_CACHE_DIR`` resolved now."""
        if self._cache_dir is not None:
            return self._cache_dir
        cfg = get_settings()
        return resolve_path(cfg.DCM_CACHE_DIR, settings=cfg)

    def shipped_path(self, name: str) -> Optional[Path]:
        """A shipped DCM file or directory (``bundled_spec.json``, ``baselines``).

        An explicit cache directory is the only place looked at; the default
        cache reads the shipped copy through ``resolve_shipped``.
        """
        local = self.cache_dir / name
        if local.exists():
            return local
        if self._cache_dir is not None:
            return None
        try:
            path, _ = resolve_shipped(f"data/dcm/{name}")
        except FileNotFoundError:
            return None
        return path

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
        bundled = self.shipped_path("bundled_spec.json")
        if bundled is not None:
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

    def _collect_indicators(self, evidencers: List[Dict], indicators: Set[str]) -> None:
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
    PyMC inference engine for DCM (experimental).

    Runs one model per stance. Each stance has a consciousness prior Beta(1,5)
    and a tree of features -> subfeatures -> indicators with
    support/demandingness parameters defining conditional Beta distributions.

    The number it produces does not respond to the evidence, for three
    structural reasons:

    1. Each indicator score enters as an observed Bernoulli whose probability
       is the score itself (a float), so it adds a constant term to the log
       probability that no model variable depends on.
    2. The scheme's sub-feature and indicator nodes carry no support or
       demandingness labels; under the label mapping a neutral node is
       Beta(1, 1) under both branches of its parent's switch, so the leaves
       are independent of their parents.
    3. The consciousness variable is continuous and thresholded at 0.5 in the
       switches, so no posterior mean can exceed E[c | c > 0.5] (about 0.58
       under the Beta(1, 5) prior).

    The posterior mean therefore equals the prior mean (about 0.167) for any
    evidence and is not comparable with published values.

    ``run_all_stances`` returns ``(posteriors, confidence, overall, ci)`` over
    the stances that succeeded; a failed stance is named in
    ``failed_stances`` (stance -> exception type) and counted in
    ``failures["dcm_stance"]``.
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
        self.failed_stances: Dict[str, str] = {}
        self.failures: Dict[str, int] = {}

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
            inference_data = pm.sample(
                draws=self.samples,
                tune=self.samples,
                cores=1,
                random_seed=self.seed,
                progressbar=False,
            )

        # The sampler returns an InferenceData whose posterior holds one row of
        # draws per chain; the statistics read every chain's draws as one array.
        posterior = np.asarray(inference_data.posterior["consciousness"]).ravel()
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
        Run inference for all stances and aggregate the successful ones.

        Raises ``DCMEngineUnavailable`` before the loop when pymc is not
        installed. A stance that raises is logged at WARNING with its name
        and the exception type, named in ``self.failed_stances``, counted in
        ``self.failures["dcm_stance"]`` and left out of the posteriors and
        the aggregate; it is re-raised when ``Settings.DEBUG`` is true.

        Returns:
            (stance_posteriors, stance_confidence, overall_prob, overall_ci);
            the overall and the interval are None when no stance succeeded.
        """
        if not pymc_available():
            raise DCMEngineUnavailable(ENGINE_UNAVAILABLE_MESSAGE)
        self.failed_stances = {}
        self.failures = {}
        stance_posteriors: Dict[str, float] = {}
        stance_confidence: Dict[str, float] = {}

        for stance in spec.stances:
            try:
                mean_val, ci_width = self.run_stance(stance, indicator_scores)
            except Exception as exc:
                logger.warning(
                    " Stance '%s' failed: %s", stance.name, type(exc).__name__
                )
                self.failed_stances[stance.name] = type(exc).__name__
                self.failures[STANCE_FAILURE_STEP] = (
                    self.failures.get(STANCE_FAILURE_STEP, 0) + 1
                )
                if bool(get_settings().DEBUG):
                    raise
                continue
            stance_posteriors[stance.name] = float(mean_val)
            stance_confidence[stance.name] = float(ci_width)
            logger.info(
                f" Stance '{stance.name}': P(c)={mean_val:.4f} "
                f"(CI width={ci_width:.4f})"
            )

        if not stance_posteriors:
            return stance_posteriors, stance_confidence, None, None

        # Aggregate: plausibility-weighted average
        overall: Optional[float]
        if plausibility_weights:
            total_w = sum(plausibility_weights.get(s, 1.0) for s in stance_posteriors)
            overall = (
                sum(
                    stance_posteriors[s] * plausibility_weights.get(s, 1.0)
                    for s in stance_posteriors
                )
                / total_w
                if total_w > 0
                else None
            )
        else:
            # Equal weights if no plausibility data
            overall = float(np.mean(list(stance_posteriors.values())))
        if overall is None:
            return stance_posteriors, stance_confidence, None, None

        # Stance spread (min/max of per-stance posteriors, not a true
        # Bayesian CI: proper CI would require combining per-stance
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
        cache_dir: Optional[str] = None,
        api_url: str = "",
        history_size: int = 50,
    ):
        if int(history_size) < 1:
            raise ValueError(f"history_size must be at least 1, got {history_size}")
        self.spec_manager = ModelSpecManager(cache_dir=cache_dir, api_url=api_url)
        self.history_size = int(history_size)
        self._latest_result: Optional[DCMBenchmarkResult] = None
        self._result_history: List[DCMBenchmarkResult] = []

    def run(
        self,
        adapter: "DCMEvidenceAdapter",
        evaluator: Optional[Any] = None,
        mode: str = "local",
        samples: int = 100,
        force_eval: bool = False,
        compute_posterior: bool = False,
        engine: Optional[Any] = None,
    ) -> DCMBenchmarkResult:
        """
        Run the DCM benchmark.

        Args:
            adapter: System-specific evidence adapter
            evaluator: LLM evaluator for Tier 2 indicators (optional)
            mode: "local" | "claude" | "manual"
            samples: MCMC samples per stance
            force_eval: Bypass LLM evaluation cache
            compute_posterior: False (the default) reports the collected
                evidence only, with ``posterior_status`` ``not_computed``;
                True also runs the engine and labels the result
                ``experimental`` (its number does not respond to the
                evidence; see ``BayesianEngine``).
            engine: The engine to run when ``compute_posterior`` is true
                (anything with ``run_all_stances``); a ``BayesianEngine``
                with ``samples`` when None.

        Raises:
            DCMEngineUnavailable: ``compute_posterior`` is true and pymc is
                not installed; nothing is saved.

        A result is saved under ``<cache_dir>/results/`` and becomes the
        latest one unless a posterior was requested and no stance
        succeeded.
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

        posteriors: Dict[str, float] = {}
        confidence: Dict[str, float] = {}
        overall: Optional[float] = None
        ci: Optional[Tuple[float, float]] = None
        failed_stances: Dict[str, str] = {}
        failures: Dict[str, int] = {}
        comparison: Dict[str, float] = {}
        status = POSTERIOR_NOT_COMPUTED

        if compute_posterior:
            status = POSTERIOR_EXPERIMENTAL
            # 3. Run the engine (experimental; see BayesianEngine).
            if engine is None:
                engine = BayesianEngine(samples=samples)
            # Equal weighting across stances. The DCM paper (Shiller and Duffy
            # 2026) weights stances by expert-survey plausibility ratings, but
            # those ratings are not published in the public API or the
            # supplementary data. If they are released, pass them through
            # run_all_stances(plausibility_weights=...).
            posteriors, confidence, overall, ci = engine.run_all_stances(
                spec, indicator_scores
            )
            failed_stances = dict(getattr(engine, "failed_stances", None) or {})
            failures = dict(getattr(engine, "failures", None) or {})
            # 4. Compare against the baselines that are not approximate
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
            posterior_status=status,
            failed_stances=failed_stances,
            failures=failures,
        )

        if compute_posterior and overall is None:
            logger.warning(
                "DCM benchmark: every stance failed (%s); nothing saved",
                ", ".join(sorted(failed_stances)) or "no stance in the spec",
            )
            return result

        self._latest_result = result
        self._result_history.append(result)
        if len(self._result_history) > self.history_size:
            self._result_history = self._result_history[-self.history_size :]

        # Save result
        self._save_result(result)

        if overall is None:
            logger.info(
                "DCM Benchmark complete: evidence only (%d indicators), "
                "posterior not computed, in %.1fs",
                len(indicator_scores),
                elapsed,
            )
        else:
            logger.info(
                f"DCM Benchmark complete: experimental P(conscious)={overall:.4f} "
                f"CI=({ci[0]:.4f}, {ci[1]:.4f}) in {elapsed:.1f}s"
            )
        return result

    def _load_comparisons(
        self, system_score: Optional[float], include_approximate: bool = False
    ) -> Dict[str, float]:
        """Deltas (system - baseline) against the baseline files.

        Empty when the system score is None. A file that cannot be read or
        parsed is skipped with a WARNING naming the file and the exception
        type; a baseline without a finite real ``overall_probability`` is
        skipped; a baseline marked ``"approximate": true`` is skipped unless
        ``include_approximate`` is true.
        """
        comparison: Dict[str, float] = {}
        if system_score is None:
            return comparison
        baselines_dir = self.spec_manager.shipped_path("baselines")
        if baselines_dir is None or not baselines_dir.is_dir():
            return comparison
        for baseline_file in sorted(baselines_dir.glob("*.json")):
            name = baseline_file.stem
            try:
                with open(baseline_file) as f:
                    data = json.load(f)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Skipping DCM baseline %s: %s",
                    baseline_file.name,
                    type(exc).__name__,
                )
                continue
            if not isinstance(data, dict):
                continue
            if data.get("approximate") is True and not include_approximate:
                continue
            baseline_score = _finite_real(data.get("overall_probability"))
            if baseline_score is None:
                continue
            comparison[name] = system_score - baseline_score
        return comparison

    def _save_result(self, result: DCMBenchmarkResult) -> None:
        """Save the benchmark result under ``<cache_dir>/results/``.

        An evidence-only result (``posterior_status`` ``not_computed``) is
        saved under its own ``_evidence`` name, so a default run never
        overwrites a same-day result that holds an experimental posterior.
        """
        results_dir = self.spec_manager.cache_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        date_str = result.timestamp.strftime("%Y-%m-%d")
        suffix = "full" if result.evaluator_mode == "claude" else "daily"
        if result.posterior_status == POSTERIOR_NOT_COMPUTED:
            suffix += "_evidence"
        filename = f"system_{date_str}_{suffix}.json"

        with open(results_dir / filename, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_latest(self) -> Optional[DCMBenchmarkResult]:
        """The most recent result, reloaded from disk when none is in memory."""
        if self._latest_result is None:
            self._latest_result = self._load_newest_saved_result()
        return self._latest_result

    def _load_newest_saved_result(self) -> Optional[DCMBenchmarkResult]:
        """The newest readable JSON result under ``<cache_dir>/results/``."""
        results_dir = self.spec_manager.cache_dir / "results"
        if not results_dir.is_dir():
            return None
        newest: Optional[DCMBenchmarkResult] = None
        for path in results_dir.glob("*.json"):
            try:
                with open(path) as f:
                    candidate = DCMBenchmarkResult.from_dict(json.load(f))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping unreadable DCM result %s: %s", path, exc)
                continue
            if newest is None or candidate.timestamp > newest.timestamp:
                newest = candidate
        return newest

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the in-memory results (at most ``history_size``) as dicts."""
        return [r.to_dict() for r in self._result_history]

    # State persistence
    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "latest_result": (
                self._latest_result.to_dict() if self._latest_result else None
            ),
        }

    def from_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the latest result from the state, else from disk.

        Results are saved to disk as JSON, so the state dict stays light:
        when it holds no result and nothing is in memory, the newest saved
        result is reloaded.
        """
        latest = (state or {}).get("latest_result")
        if isinstance(latest, dict):
            try:
                self._latest_result = DCMBenchmarkResult.from_dict(latest)
                return
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Ignoring malformed DCM latest_result: %s", exc)
        if self._latest_result is None:
            self._latest_result = self._load_newest_saved_result()


# ============================================================================
# CLI Entry Point
# ============================================================================


def load_adapter_factory(spec: str) -> Callable[..., Any]:
    """Resolve ``package.module:callable`` to the adapter factory it names."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(
            f"adapter must look like package.module:callable, got {spec!r}"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise ValueError(f"{spec!r} does not name a callable")
    return factory


def _print_comparison(runner: "BenchmarkRunner", result: DCMBenchmarkResult) -> None:
    """The newest saved result, labelled, against the baselines that are not
    approximate (recomputed from the baseline files now)."""
    if result.overall_probability is None:
        print(f"\nPosterior: not computed ({LIMITATIONS_POINTER})")
        print(f"Indicators: {result.indicators_measured}")
        return
    print(f"\nExperimental P(conscious): {result.overall_probability:.4f}")
    if result.overall_ci is not None:
        print(
            f"Stance spread: ({result.overall_ci[0]:.4f}, "
            f"{result.overall_ci[1]:.4f})"
        )
    print(EXPERIMENTAL_NOTE)
    comparison = runner._load_comparisons(result.overall_probability)
    if comparison:
        print("\nComparison with verified baselines (system - baseline):")
        for name, delta in sorted(comparison.items()):
            sign = "+" if delta >= 0 else ""
            print(f" vs {name}: {sign}{delta:.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    """Run the DCM benchmark from the command line; returns the exit code."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="DCM Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["local", "claude", "manual"],
        default=settings.DCM_EVALUATOR_MODE,
        help="Evaluator mode (default: from MTC_DCM_EVALUATOR_MODE); claude "
        "sends the system description and conversation samples to an "
        "external API",
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
        help="Show the newest saved result against the baselines",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: MTC_DCM_CACHE_DIR under the project root, "
        "with the shipped spec and baselines read from data/dcm or the package)",
    )
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="Adapter factory as package.module:callable "
        f"(default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Persisted module state passed to the adapter factory",
    )
    parser.add_argument(
        "--experimental-posterior",
        action="store_true",
        help="Also run the PyMC engine (needs the research extra); its number "
        "does not respond to the evidence and is labelled experimental",
    )
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir
    runner = BenchmarkRunner(cache_dir=cache_dir)

    if args.refresh_spec:
        print("Refreshing DCM spec from API...")
        runner.spec_manager.fetch_from_api()
        print("Done.")
        if not args.compare:
            return 0

    if args.compare:
        result = runner.get_latest()
        if not result:
            print("No results yet. Run a benchmark first.")
            return 1
        _print_comparison(runner, result)
        return 0

    try:
        factory = load_adapter_factory(args.adapter)
        adapter = factory(state_dir=args.state_dir)
    except (ImportError, ValueError, TypeError) as exc:
        print(f"Adapter not available ({args.adapter}): {exc}", file=sys.stderr)
        return 1

    from mtc.assessment.dcm_evaluator import DCMEvaluator

    evaluator = DCMEvaluator(cache_dir=cache_dir) if args.mode != "manual" else None

    print(f"Running DCM Benchmark (mode={args.mode}, samples={args.samples})...")
    try:
        result = runner.run(
            adapter=adapter,
            evaluator=evaluator,
            mode=args.mode,
            samples=args.samples,
            force_eval=args.force_eval,
            compute_posterior=args.experimental_posterior,
        )
    except DCMEngineUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    rule = "=" * 60
    print(f"\n{rule}")
    print("DCM Benchmark Result")
    print(rule)
    print(f"Spec version: {result.spec_version}")
    print(f"Indicators: {result.indicators_measured}")
    print(f" Tier 1 (module): {result.tier1_count}")
    print(f" Tier 2 (LLM):    {result.tier2_count}")
    print(f" Tier 3 (zeros):  {result.tier3_count}")
    print(f"Time: {result.processing_time_s:.1f}s")
    if not args.experimental_posterior:
        print(f"\nPosterior: not computed ({LIMITATIONS_POINTER})")
        return 0
    if result.overall_probability is None:
        print(
            "Posterior: not computed; every stance failed: "
            + (", ".join(sorted(result.failed_stances)) or "no stance in the spec"),
            file=sys.stderr,
        )
        return 1
    print(f"\nExperimental P(conscious): {result.overall_probability:.4f}")
    print(f"Stance spread: ({result.overall_ci[0]:.4f}, {result.overall_ci[1]:.4f})")
    print(EXPERIMENTAL_NOTE)
    print(f"({LIMITATIONS_POINTER})")
    print("\nPer-stance posteriors (experimental):")
    for name, prob in sorted(
        result.stance_posteriors.items(), key=lambda x: x[1], reverse=True
    ):
        ci = result.stance_confidence.get(name)
        width = "n/a" if ci is None else f"{ci:.4f}"
        print(f" {name:40s} {prob:.4f} (CI width={width})")
    for name, error in sorted(result.failed_stances.items()):
        print(f" {name:40s} failed ({error})")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
