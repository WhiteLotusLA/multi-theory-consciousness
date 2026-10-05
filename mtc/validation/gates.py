"""
Acceptance Gates
================

The acceptance-gate harness: a signal that runs in shadow (logged by
``mtc.validation.shadow_log``, never narrated) must pass these gates on its
shadow data before anyone proposes narrating it. Each gate encodes a generic
lesson about inner-state signals that look alive but are not:

- ``movement``: the signal moves. A constant value, or one value stamped on
  most samples (a template-dominated signal), fails.
- ``content_coupling``: the signal moves with the independent feature it
  claims to track: Spearman correlation, with significance from a seeded
  permutation null (no distributional assumption).
- ``range_honesty``: the signal stays inside its claimed range and visits
  every state it claims to distinguish. A channel that never goes negative
  cannot claim negative states.
- ``null_beating``: a signal-level metric beats a matched random baseline,
  so a statistic that random inputs also produce (phase-locking that is not
  binding, for example) does not count. Callers own the null generation.

Every gate fails closed: too few samples is "not provable yet", never a pass,
and a non-finite input, a constant series that makes a correlation undefined
or an empty null distribution is a failure rather than a comparison.
Regions are half-open ``[low, high)``, except that a region ending at the
claimed maximum includes it.

``run_signal_gates(spec, records)`` runs the suite a spec describes over
shadow records. Spec shape (JSON or YAML, read by ``load_spec``)::

    signal_id: <the shadow signal id>
    dimensions: [<value keys>]
    gates:
      movement: {min_std, max_modal_fraction, min_distinct, min_samples}
      range_honesty: {claimed_min, claimed_max, required_regions:
                      [[low, high, label], ...], min_hits_per_region,
                      min_samples}
      content_coupling: {feature_key (a context key) or feature_values,
                         feature_name, min_abs_corr, max_p, seed,
                         n_permutations, min_samples}
      null_beating: {real_metric, null_metrics, metric_name, percentile}

Each per-dimension gate may also name ``dimensions`` to run on a subset; the
null-beating gate runs once. A record's value that is None is a missing
measurement and is skipped. An unknown gate name raises ValueError.

``write_artifacts`` writes ``<signal_id>-gates-<date>.json`` and ``.md`` with
every verdict and the numbers behind it. The records the gates read can hold
conversation-derived context and are private; the artifacts hold only
statistics. PyYAML is imported only when a YAML spec is loaded.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

DEFAULT_MIN_SAMPLES = 50
DEFAULT_MIN_PAIRS = 30
PER_DIMENSION_GATES = ("movement", "range_honesty", "content_coupling")
GATE_NAMES = PER_DIMENSION_GATES + ("null_beating",)


@dataclass
class GateResult:
    """One gate's verdict with the numbers behind it."""

    gate: str
    signal_id: str
    dimension: Optional[str]
    passed: bool
    metrics: Dict[str, Any]
    reason: str


@dataclass
class GateReport:
    """Every gate result for one signal; it passes only when every gate ran
    and passed."""

    signal_id: str
    generated_at: str
    n_samples: int
    window_start: Optional[str]
    window_end: Optional[str]
    results: List[GateResult] = field(default_factory=list)

    @property
    def overall_pass(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)


def _array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _non_finite(arr: np.ndarray) -> int:
    return int(np.sum(~np.isfinite(arr)))


# ------------------------------------------------------------------ gates


def movement_gate(
    values: Sequence[float],
    *,
    signal_id: str,
    dimension: Optional[str],
    min_std: float,
    max_modal_fraction: float,
    min_distinct: int,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> GateResult:
    """The signal moves: not constant, not template-dominated."""
    arr = _array(values)
    n = int(arr.size)
    base: Dict[str, Any] = {"n": n}
    if n < min_samples:
        return GateResult(
            "movement",
            signal_id,
            dimension,
            False,
            base,
            f"insufficient samples: n={n} < {min_samples}",
        )
    bad = _non_finite(arr)
    if bad:
        return GateResult(
            "movement",
            signal_id,
            dimension,
            False,
            {**base, "non_finite": bad},
            f"{bad} non-finite values; failing closed",
        )
    std = float(np.std(arr))
    # Rounded so float jitter cannot hide a stamped value.
    _, counts = np.unique(np.round(arr, 6), return_counts=True)
    modal_fraction = float(counts.max() / n)
    distinct = int(counts.size)
    metrics = {
        **base,
        "std": std,
        "modal_fraction": modal_fraction,
        "distinct_values": distinct,
    }
    if std <= min_std:
        reason, passed = f"effectively constant: std={std:.6g} <= {min_std}", False
    elif modal_fraction > max_modal_fraction:
        reason, passed = (
            f"template-dominated: modal fraction {modal_fraction:.2f} > "
            f"{max_modal_fraction} (one value stamped on most samples)",
            False,
        )
    elif distinct < min_distinct:
        reason, passed = f"too few distinct values: {distinct} < {min_distinct}", False
    else:
        reason, passed = (
            f"moves: std={std:.4g}, modal fraction {modal_fraction:.2f}, "
            f"{distinct} distinct values over {n} samples",
            True,
        )
    return GateResult("movement", signal_id, dimension, passed, metrics, reason)


def content_coupling_gate(
    signal_values: Sequence[float],
    feature_values: Sequence[float],
    *,
    signal_id: str,
    dimension: Optional[str],
    feature_name: str,
    min_abs_corr: float,
    max_p: float,
    seed: int,
    n_permutations: int = 1000,
    min_samples: int = DEFAULT_MIN_PAIRS,
) -> GateResult:
    """The signal correlates with the feature it claims to track; the p value
    comes from a seeded permutation null."""
    sig = _array(signal_values)
    feat = _array(feature_values)
    if sig.size != feat.size:
        raise ValueError(f"signal/feature length mismatch: {sig.size} != {feat.size}")
    n = int(sig.size)
    base: Dict[str, Any] = {"n": n, "feature": feature_name}
    if n < min_samples:
        return GateResult(
            "content_coupling",
            signal_id,
            dimension,
            False,
            base,
            f"insufficient pairs: n={n} < {min_samples}",
        )
    bad = _non_finite(sig) + _non_finite(feat)
    if bad:
        return GateResult(
            "content_coupling",
            signal_id,
            dimension,
            False,
            {**base, "correlation": None, "non_finite": bad},
            f"{bad} non-finite values; failing closed",
        )
    constant = bool(np.ptp(sig) == 0 or np.ptp(feat) == 0)
    rho = float("nan") if constant else float(stats.spearmanr(sig, feat).statistic)
    # A constant input makes rho NaN, and NaN compares False with everything:
    # without this guard an untestable coupling would collect no null hits
    # and pass. Untestable claims fail.
    if not math.isfinite(rho):
        return GateResult(
            "content_coupling",
            signal_id,
            dimension,
            False,
            {**base, "correlation": None},
            f"coupling to {feature_name} undefined: the signal or the feature is "
            "constant; failing closed",
        )
    rng = np.random.default_rng(seed)
    shuffled = feat.copy()
    hits = 0
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        null_rho = stats.spearmanr(sig, shuffled).statistic
        # A non-finite null counts against the signal (the conservative side).
        if not np.isfinite(null_rho) or abs(null_rho) >= abs(rho):
            hits += 1
    p_perm = (hits + 1) / (n_permutations + 1)
    metrics = {
        **base,
        "correlation": rho,
        "p_permutation": float(p_perm),
        "n_permutations": n_permutations,
    }
    if abs(rho) < min_abs_corr:
        reason, passed = (
            f"uncoupled from {feature_name}: |rho|={abs(rho):.3f} < {min_abs_corr}",
            False,
        )
    elif p_perm > max_p:
        reason, passed = (
            f"coupling to {feature_name} not significant: p={p_perm:.4f} > {max_p}",
            False,
        )
    else:
        reason, passed = (
            f"couples to {feature_name}: rho={rho:.3f}, p={p_perm:.4f} ({n} pairs)",
            True,
        )
    return GateResult("content_coupling", signal_id, dimension, passed, metrics, reason)


def range_honesty_gate(
    values: Sequence[float],
    *,
    signal_id: str,
    dimension: Optional[str],
    claimed_min: float,
    claimed_max: float,
    required_regions: Sequence[Tuple[float, float, str]],
    min_hits_per_region: int,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> GateResult:
    """The signal stays inside its claimed range and visits every claimed
    state."""
    arr = _array(values)
    n = int(arr.size)
    base: Dict[str, Any] = {"n": n}
    if n < min_samples:
        return GateResult(
            "range_honesty",
            signal_id,
            dimension,
            False,
            base,
            f"insufficient samples: n={n} < {min_samples}",
        )
    bad = _non_finite(arr)
    if bad:
        return GateResult(
            "range_honesty",
            signal_id,
            dimension,
            False,
            {**base, "non_finite": bad},
            f"{bad} non-finite values; failing closed",
        )
    outside = int(np.sum((arr < claimed_min) | (arr > claimed_max)))
    hits: Dict[str, int] = {}
    for low, high, label in required_regions:
        upper = arr <= high if high >= claimed_max else arr < high
        hits[str(label)] = int(np.sum((arr >= low) & upper))
    metrics = {
        **base,
        "observed_min": float(arr.min()),
        "observed_max": float(arr.max()),
        "out_of_claimed_range": outside,
        "region_hits": hits,
    }
    if outside:
        return GateResult(
            "range_honesty",
            signal_id,
            dimension,
            False,
            metrics,
            f"{outside} samples outside the claimed range "
            f"[{claimed_min}, {claimed_max}]; the claim itself is wrong",
        )
    unreached = [label for label, count in hits.items() if count < min_hits_per_region]
    if unreached:
        return GateResult(
            "range_honesty",
            signal_id,
            dimension,
            False,
            metrics,
            "claimed states never genuinely reached: "
            + ", ".join(
                f"{label} ({hits[label]} hits < {min_hits_per_region})"
                for label in unreached
            ),
        )
    return GateResult(
        "range_honesty",
        signal_id,
        dimension,
        True,
        metrics,
        f"visits all claimed states within [{claimed_min}, {claimed_max}]: "
        + ", ".join(f"{k}={v}" for k, v in hits.items()),
    )


def null_beating_gate(
    *,
    real_metric: float,
    null_metrics: Sequence[float],
    signal_id: str,
    dimension: Optional[str],
    metric_name: str,
    percentile: float = 95.0,
) -> GateResult:
    """The signal's metric beats the given percentile of a matched null."""
    nulls = _array(null_metrics)
    real_ok = isinstance(real_metric, (int, float)) and math.isfinite(real_metric)
    if not real_ok or nulls.size == 0 or _non_finite(nulls):
        return GateResult(
            "null_beating",
            signal_id,
            dimension,
            False,
            {"metric": metric_name, "n_nulls": int(nulls.size)},
            f"{metric_name}: non-finite metric or empty or non-finite null "
            "distribution; comparison undefined, failing closed",
        )
    threshold = float(np.percentile(nulls, percentile))
    key = f"null_p{percentile:.0f}"
    metrics = {
        "metric": metric_name,
        "real": float(real_metric),
        key: threshold,
        "null_mean": float(nulls.mean()),
        "null_std": float(nulls.std()),
        "n_nulls": int(nulls.size),
    }
    if real_metric <= threshold:
        return GateResult(
            "null_beating",
            signal_id,
            dimension,
            False,
            metrics,
            f"{metric_name}={real_metric:.4g} does not beat the null "
            f"(p{percentile:.0f}={threshold:.4g}); indistinguishable from noise",
        )
    return GateResult(
        "null_beating",
        signal_id,
        dimension,
        True,
        metrics,
        f"{metric_name}={real_metric:.4g} beats the null "
        f"p{percentile:.0f}={threshold:.4g}",
    )


# ------------------------------------------------------ shadow log + runner


def load_shadow_log(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a shadow JSONL log, one record per non-blank line.

    A line that is not a JSON object raises ValueError naming its number.
    """
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except ValueError as exc:
                raise ValueError(f"{path}: line {number} is not JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {number} is not a JSON object")
            records.append(record)
    return records


def load_spec(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a gate spec from JSON (``.json``) or YAML (any other suffix)."""
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() == ".json":
        spec = json.loads(text)
    else:
        import yaml  # PyYAML, imported only for a YAML spec

        spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise ValueError(f"{target}: a gate spec must be a mapping")
    return spec


def _value(record: Dict[str, Any], dimension: str) -> Optional[float]:
    values = record.get("values") or {}
    raw = values.get(dimension) if isinstance(values, dict) else None
    return None if raw is None else float(raw)


def run_signal_gates(
    spec: Dict[str, Any],
    records: List[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> GateReport:
    """Run the gate suite ``spec`` describes over shadow ``records``."""
    try:
        signal_id = str(spec["signal_id"])
        dimensions = list(spec["dimensions"])
        gates = dict(spec["gates"] or {})
    except (KeyError, TypeError) as exc:
        raise ValueError("a gate spec needs signal_id, dimensions and gates") from exc
    unknown = sorted(set(gates) - set(GATE_NAMES))
    if unknown:
        raise ValueError(f"unknown gates {unknown}; known: {GATE_NAMES}")

    moment = now or datetime.now(timezone.utc)
    stamps = [str(r["ts"]) for r in records if r.get("ts")]
    report = GateReport(
        signal_id=signal_id,
        generated_at=moment.isoformat(),
        n_samples=len(records),
        window_start=min(stamps) if stamps else None,
        window_end=max(stamps) if stamps else None,
    )

    def config_for(name: str, dimension: str) -> Optional[Dict[str, Any]]:
        if name not in gates:
            return None
        cfg = dict(gates[name] or {})
        only = cfg.pop("dimensions", None)
        if only is not None and dimension not in only:
            return None
        return cfg

    for dimension in dimensions:
        values = [v for v in (_value(r, dimension) for r in records) if v is not None]
        cfg = config_for("movement", dimension)
        if cfg is not None:
            report.results.append(
                movement_gate(values, signal_id=signal_id, dimension=dimension, **cfg)
            )
        cfg = config_for("range_honesty", dimension)
        if cfg is not None:
            cfg["required_regions"] = [tuple(r) for r in cfg["required_regions"]]
            report.results.append(
                range_honesty_gate(
                    values, signal_id=signal_id, dimension=dimension, **cfg
                )
            )
        cfg = config_for("content_coupling", dimension)
        if cfg is not None:
            feature_key = cfg.pop("feature_key", None)
            if feature_key is not None:
                pairs = []
                for record in records:
                    value = _value(record, dimension)
                    context = record.get("context") or {}
                    feature = context.get(feature_key)
                    if value is not None and feature is not None:
                        pairs.append((value, float(feature)))
                signal_values = [p[0] for p in pairs]
                feature_values = [p[1] for p in pairs]
            else:
                signal_values = values
                feature_values = list(cfg.pop("feature_values"))
            report.results.append(
                content_coupling_gate(
                    signal_values,
                    feature_values,
                    signal_id=signal_id,
                    dimension=dimension,
                    **cfg,
                )
            )
    if "null_beating" in gates:
        cfg = dict(gates["null_beating"] or {})
        cfg.pop("dimensions", None)
        report.results.append(
            null_beating_gate(signal_id=signal_id, dimension=None, **cfg)
        )
    return report


def _payload(report: GateReport) -> Dict[str, Any]:
    return {
        "signal_id": report.signal_id,
        "generated_at": report.generated_at,
        "n_samples": report.n_samples,
        "window_start": report.window_start,
        "window_end": report.window_end,
        "overall_pass": report.overall_pass,
        "results": [
            {
                "gate": r.gate,
                "dimension": r.dimension,
                "passed": r.passed,
                "metrics": r.metrics,
                "reason": r.reason,
            }
            for r in report.results
        ],
    }


def render_markdown(report: GateReport) -> str:
    """The plain-language artifact: verdicts first, then the numbers."""
    verdict = "PASS" if report.overall_pass else "FAIL"
    meaning = (
        "every gate passed on the shadow data; narrating this signal may now be "
        "proposed, which still needs naming and consent."
        if report.overall_pass
        else "at least one gate failed, so this signal stays silent. A failed "
        "gate is a finding: silence is the honest state until the signal passes."
    )
    lines = [
        f"# Acceptance gates: `{report.signal_id}`",
        "",
        f"Generated {report.generated_at}. {report.n_samples} shadow samples, "
        f"window {report.window_start} to {report.window_end}.",
        "",
        f"**Overall: {verdict}**: {meaning}",
        "",
        "| Gate | Dimension | Verdict | Why |",
        "|---|---|---|---|",
    ]
    for r in report.results:
        lines.append(
            f"| {r.gate} | {r.dimension or '-'} | "
            f"{'PASS' if r.passed else 'FAIL'} | {r.reason.replace('|', '/')} |"
        )
    lines += ["", "## The numbers behind the verdicts", ""]
    for r in report.results:
        label = f"{r.gate} ({r.dimension})" if r.dimension else r.gate
        numbers = ", ".join(f"{k}={v}" for k, v in r.metrics.items())
        lines.append(f"- **{label}:** {numbers}")
    lines.append("")
    return "\n".join(lines)


def write_artifacts(report: GateReport, directory: Union[str, Path]) -> List[Path]:
    """Write ``<signal_id>-gates-<date>.json`` and ``.md``; return both paths."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    base = f"{report.signal_id}-gates-{report.generated_at[:10]}"
    json_path = out / f"{base}.json"
    md_path = out / f"{base}.md"
    json_path.write_text(
        json.dumps(_payload(report), indent=2, default=str), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return [json_path, md_path]
