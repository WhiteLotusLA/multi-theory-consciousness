#!/usr/bin/env python3
"""
Run Consciousness Assessment - CLI
====================================

Command-line tool to run the full multi-theory consciousness
assessment framework.

Modes:
    --quick     Minimal assessment with freshly initialized modules (default)
    --full      Run several consciousness cycles first to warm up modules
    --report    Print a detailed Markdown-formatted report
    --ablation  Run ablation study (disable one component at a time)
    --json      Write results to a JSON file under ASSESSMENT_OUTPUT_DIR
    --json-path Write the JSON report to this path instead (implies --json)
    --seed      Seed random, numpy and torch before any module is built (42)
    --state-dir Restore persisted module state from this directory, then assess

Usage:
    python scripts/run_assessment.py
    python scripts/run_assessment.py --full --report
    python scripts/run_assessment.py --ablation
    python scripts/run_assessment.py --full --json
    python scripts/run_assessment.py --preset legacy_v04 --full  # v0.4.0 behaviour
    python scripts/run_assessment.py --params my.toml            # parameter overrides

State directory (--state-dir): the modules are built, registered with the
ConsciousnessStateManager under six names, restored with restore_all() and then
assessed, using the workspace's own attention schema, metacognition, active
inference and beautiful loop as the AST, HOT, FEP and BLT inputs. Later phases
persist under the same six names:

    gwt                    the EnhancedGlobalWorkspace (and its owned modules)
    rpt                    RPTMeasurement
    phi                    PhiTracker
    causal_emergence       CausalEmergenceAnalyzer
    pad_affect             PADAffectModel
    oscillatory_binding    OscillatoryBinding

Without --state-dir, RPT, PAD, the causal emergence analyzer and neural states
are not built and report absent; scripts/run_experiment.py is the synthetic
testbed and --state-dir is the path for real runs.

Overrides are optional here: with neither --preset nor --params the modules
run on their defaults. --preset NAME loads experiments/presets/NAME.toml on
top of the defaults, and --params PATH loads on top of the preset.

This script is a shim: the runner lives in mtc.cli.assess (``mtc assess``),
and the names a caller reads from this module come from there.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtc.cli import assess as _assess  # noqa: E402
from mtc.cli.assess import main  # noqa: E402,F401


def __getattr__(name):
    """Expose the runner's names (helpers and constants) through this module."""
    return getattr(_assess, name)


if __name__ == "__main__":
    raise SystemExit(main())
