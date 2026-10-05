#!/usr/bin/env python3
"""Run the acceptance gates for one shadow signal.

Usage:
  python scripts/validation/run_gates.py --spec config/validation/<signal>.yaml
      [--shadow-dir DIR] [--log FILE] [--out DIR]

The spec (YAML or JSON) names the signal; its shadow log is read from
<shadow-dir>/<signal_id>.jsonl (default directory: SHADOW_LOG_DIR) unless --log
names the file. The JSON and markdown artifacts go to --out (default
<RESULTS_DIR>/validation).

Exit codes: 0 every gate passed (narration becomes proposable; naming and
consent still lie ahead), 1 at least one gate failed (the signal stays
silent), 2 the spec or the shadow log could not be read. A missing shadow log
is reported as missing evidence and never counts as a pass.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtc.core.config import get_settings  # noqa: E402
from mtc.core.paths import results_dir, shadow_log_dir  # noqa: E402
from mtc.validation.gates import (  # noqa: E402
    load_shadow_log,
    load_spec,
    run_signal_gates,
    write_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the acceptance gates for one shadow signal."
    )
    parser.add_argument("--spec", required=True, help="gate spec (YAML or JSON)")
    parser.add_argument(
        "--shadow-dir", help="shadow log directory (default: SHADOW_LOG_DIR)"
    )
    parser.add_argument("--log", help="shadow log file (overrides --shadow-dir)")
    parser.add_argument(
        "--out", help="artifact directory (default: <RESULTS_DIR>/validation)"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    try:
        spec = load_spec(args.spec)
        signal_id = str(spec["signal_id"])
    except Exception as exc:
        print(f"OPERATIONAL ERROR: spec {args.spec} could not be read: {exc}")
        return 2

    if args.log:
        log_path = Path(args.log)
    else:
        directory = (
            Path(args.shadow_dir) if args.shadow_dir else shadow_log_dir(settings)
        )
        log_path = directory / f"{signal_id}.jsonl"
    if not log_path.is_file():
        print(
            f"MISSING EVIDENCE: shadow log {log_path} does not exist; "
            f"{signal_id} has no samples and its gates are not passed"
        )
        return 2
    try:
        records = load_shadow_log(log_path)
        report = run_signal_gates(spec, records)
    except Exception as exc:
        print(f"OPERATIONAL ERROR: {exc}")
        return 2

    out_dir = Path(args.out) if args.out else results_dir(settings) / "validation"
    json_path, md_path = write_artifacts(report, out_dir)
    print(f"== Acceptance gates: {report.signal_id}")
    print(
        f"   samples: {report.n_samples}  "
        f"window: {report.window_start} to {report.window_end}"
    )
    for result in report.results:
        verdict = "PASS" if result.passed else "FAIL"
        dimension = f" [{result.dimension}]" if result.dimension else ""
        print(f"   {result.gate}{dimension}: {verdict}: {result.reason}")
    print(f"   overall: {'PASS' if report.overall_pass else 'FAIL'}")
    print(f"   artifacts: {json_path}  {md_path}")
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
