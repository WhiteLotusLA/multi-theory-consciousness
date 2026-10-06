#!/usr/bin/env python3
"""Check every enabled store against the schema registry (mtc db check).

Usage: python scripts/db/verify.py [--only STORE...] [--json] [--strict] [--env-file PATH]
A thin wrapper: the work is done by mtc.db.verify, so an installed package can run
the same step as mtc db check from any directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mtc.db.verify import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
