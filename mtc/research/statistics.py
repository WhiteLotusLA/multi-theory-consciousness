"""
Statistics helper
=================

Small, dependency-light statistics for the research tooling: an effect size,
a reproducibility index, two-sample and one-sample tests, a linear trend,
ablation effects and a plain JSON export. numpy and ``scipy.stats`` only.

Warning: this module is named ``statistics``, like the standard library
module. If ``mtc/research`` is the first entry on ``sys.path`` (for example
when a file in this directory is run as a script), ``import statistics``
resolves here and breaks the standard library. Never run these files as
scripts; import them as ``mtc.research.statistics``.

Honest measurement: every function that cannot compute a quantity returns None
(or a result whose ``status`` is ``insufficient_data`` with a reason) instead
of a number: fewer than two observations, zero variance, a zero mean for a
ratio. Results are descriptive; thresholds are module constants:

- ``ALPHA`` (0.05): a p-value below it is flagged ``significant``;
- ``MIN_EFFECT_SIZE`` (0.5): an absolute Cohen's d at or above it is flagged
  ``meaningful_effect``;
- ``MIN_REPRODUCIBILITY`` (0.85): the reproducibility index at or above it
  counts as reproducible in ``is_reproducible``.

Config: none. Persisted state: none (``export_results`` writes a file only
when called).
"""

import json
import logging
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

ALPHA = 0.05
MIN_EFFECT_SIZE = 0.5
MIN_REPRODUCIBILITY = 0.85

MEASURED = "measured"
INSUFFICIENT_DATA = "insufficient_data"

_EXPORT_NOTE = (
    "Descriptive statistics of the recorded samples. A statistical difference "
    "or trend says nothing by itself about experience."
)


@dataclass
class ComparisonResult:
    """A comparison of one sample with another sample or with a constant."""

    method: str
    status: str
    n_a: int
    n_b: Optional[int] = None
    mean_a: Optional[float] = None
    mean_b: Optional[float] = None
    difference: Optional[float] = None
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    significant: Optional[bool] = None
    meaningful_effect: Optional[bool] = None
    reason: Optional[str] = None


@dataclass
class TrendResult:
    """Least-squares line through a series."""

    slope: float
    intercept: float
    r_squared: float
    p_value: Optional[float]
    n: int


def _clean(values: Optional[Sequence[Any]]) -> np.ndarray:
    """Finite floats only; None and non-finite entries are dropped."""
    if values is None:
        return np.asarray([], dtype=float)
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=float)


def cohen_d(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Cohen's d (a minus b) with the pooled standard deviation.

    None with fewer than two observations on either side or zero pooled
    variance.
    """
    x, y = _clean(a), _clean(b)
    if x.size < 2 or y.size < 2:
        return None
    pooled = ((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1)) / (
        x.size + y.size - 2
    )
    if pooled <= 0.0 or not math.isfinite(pooled):
        return None
    return float((x.mean() - y.mean()) / math.sqrt(pooled))


def reproducibility(values: Sequence[float]) -> Optional[float]:
    """One minus the coefficient of variation, clipped to [0, 1].

    None with fewer than two observations or a zero mean.
    """
    x = _clean(values)
    if x.size < 2:
        return None
    mean = float(x.mean())
    if abs(mean) < 1e-12:
        return None
    cv = float(x.std(ddof=1)) / abs(mean)
    return float(min(1.0, max(0.0, 1.0 - cv)))


def is_reproducible(values: Sequence[float]) -> Optional[bool]:
    """Whether the reproducibility index reaches ``MIN_REPRODUCIBILITY``."""
    index = reproducibility(values)
    return None if index is None else index >= MIN_REPRODUCIBILITY


def _flags(p_value: Optional[float], effect: Optional[float]) -> Dict[str, Any]:
    return {
        "significant": None if p_value is None else bool(p_value < ALPHA),
        "meaningful_effect": (
            None if effect is None else bool(abs(effect) >= MIN_EFFECT_SIZE)
        ),
    }


def _insufficient(method: str, n_a: int, n_b: Optional[int], reason: str):
    return ComparisonResult(
        method=method,
        status=INSUFFICIENT_DATA,
        n_a=n_a,
        n_b=n_b,
        reason=reason,
    )


def compare_samples(
    a: Sequence[float],
    b: Sequence[float],
    method: str = "t_test",
    n_permutations: int = 10000,
    seed: Optional[int] = None,
) -> ComparisonResult:
    """Two-sample comparison of ``a`` against ``b`` (difference is a minus b).

    ``method`` is ``t_test`` (Welch's unequal-variance test) or
    ``permutation`` (two-sided difference of means; ``seed`` makes it
    reproducible).
    """
    if method not in ("t_test", "permutation"):
        raise ValueError("method must be 't_test' or 'permutation'")
    x, y = _clean(a), _clean(b)
    if x.size < 2 or y.size < 2:
        return _insufficient(method, x.size, y.size, "fewer than two observations")
    diff = float(x.mean() - y.mean())
    effect = cohen_d(x, y)
    if method == "t_test":
        if x.var(ddof=1) == 0.0 and y.var(ddof=1) == 0.0:
            return _insufficient(method, x.size, y.size, "zero variance")
        res = scipy_stats.ttest_ind(x, y, equal_var=False)
        statistic, p_value = float(res.statistic), float(res.pvalue)
    else:
        rng = np.random.default_rng(seed)
        pooled = np.concatenate([x, y])
        count = 0
        for _ in range(int(n_permutations)):
            rng.shuffle(pooled)
            stat = pooled[: x.size].mean() - pooled[x.size :].mean()
            if abs(stat) >= abs(diff) - 1e-12:
                count += 1
        statistic = diff
        p_value = (count + 1) / (int(n_permutations) + 1)
    if not (math.isfinite(statistic) and math.isfinite(p_value)):
        return _insufficient(method, x.size, y.size, "statistic not finite")
    return ComparisonResult(
        method=method,
        status=MEASURED,
        n_a=int(x.size),
        n_b=int(y.size),
        mean_a=float(x.mean()),
        mean_b=float(y.mean()),
        difference=diff,
        statistic=statistic,
        p_value=p_value,
        effect_size=effect,
        **_flags(p_value, effect),
    )


def one_sample_test(values: Sequence[float], chance: float = 0.5) -> ComparisonResult:
    """One-sample t-test of the mean against ``chance`` with a 95% interval.

    The effect size is Cohen's d against chance: (mean minus chance) over the
    sample standard deviation.
    """
    x = _clean(values)
    method = "one_sample_t_test"
    if x.size < 2:
        return _insufficient(method, int(x.size), None, "fewer than two observations")
    sd = float(x.std(ddof=1))
    mean = float(x.mean())
    if sd == 0.0:
        return _insufficient(method, int(x.size), None, "zero variance")
    res = scipy_stats.ttest_1samp(x, chance)
    sem = sd / math.sqrt(x.size)
    half = float(scipy_stats.t.ppf(0.975, x.size - 1)) * sem
    effect = (mean - chance) / sd
    p_value = float(res.pvalue)
    return ComparisonResult(
        method=method,
        status=MEASURED,
        n_a=int(x.size),
        n_b=None,
        mean_a=mean,
        mean_b=float(chance),
        difference=mean - float(chance),
        statistic=float(res.statistic),
        p_value=p_value,
        effect_size=float(effect),
        ci_low=mean - half,
        ci_high=mean + half,
        **_flags(p_value, effect),
    )


def trend(
    values: Sequence[float], times: Optional[Sequence[float]] = None
) -> Optional[TrendResult]:
    """Least-squares slope of ``values`` against ``times`` (default 0..n-1).

    None with fewer than three points or a constant time axis.
    """
    y_raw = list(values)
    t_raw = list(range(len(y_raw))) if times is None else list(times)
    if len(t_raw) != len(y_raw):
        raise ValueError("values and times must have the same length")
    pairs = []
    for t, v in zip(t_raw, y_raw):
        try:
            tf, vf = float(t), float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(tf) and math.isfinite(vf):
            pairs.append((tf, vf))
    if len(pairs) < 3:
        return None
    t_arr = np.asarray([p[0] for p in pairs])
    y_arr = np.asarray([p[1] for p in pairs])
    if float(t_arr.max() - t_arr.min()) == 0.0:
        return None
    fit = scipy_stats.linregress(t_arr, y_arr)
    p_value = float(fit.pvalue) if math.isfinite(fit.pvalue) else None
    return TrendResult(
        slope=float(fit.slope),
        intercept=float(fit.intercept),
        r_squared=float(fit.rvalue**2),
        p_value=p_value,
        n=len(pairs),
    )


def ablation_effects(runs: Sequence[Dict[str, Any]]) -> Dict[str, ComparisonResult]:
    """Compare each ablated configuration with the baseline across runs.

    ``runs`` are repeated outputs of ``AblationStudy.run_full_ablation``: each
    holds ``configurations``, a list of ``{"name", "overall_score", ...}``
    whose first entry (or the one named ``Baseline``) is the baseline. Runs
    whose overall score is None for a configuration are left out of that
    configuration's sample. The comparison is baseline minus configuration, so
    a positive difference means the ablation lowered the score.
    """
    baseline: List[float] = []
    by_name: Dict[str, List[float]] = {}
    order: List[str] = []
    for run in runs:
        configs = run.get("configurations") or []
        if not configs:
            continue
        base = next((c for c in configs if c.get("name") == "Baseline"), configs[0])
        score = base.get("overall_score")
        if score is not None:
            baseline.append(float(score))
        for config in configs:
            name = config.get("name")
            if config is base or name is None:
                continue
            if name not in by_name:
                by_name[name] = []
                order.append(name)
            if config.get("overall_score") is not None:
                by_name[name].append(float(config["overall_score"]))
    return {name: compare_samples(baseline, by_name[name]) for name in order}


def _plain(value: Any) -> Any:
    """Convert dataclasses and numpy values to plain JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def export_results(results: Dict[str, Any], path: Path) -> Path:
    """Write ``results`` as plain JSON (non-finite numbers become null)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": _EXPORT_NOTE,
        "thresholds": {
            "alpha": ALPHA,
            "min_effect_size": MIN_EFFECT_SIZE,
            "min_reproducibility": MIN_REPRODUCIBILITY,
        },
        "results": _plain(results),
    }
    target.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return target
