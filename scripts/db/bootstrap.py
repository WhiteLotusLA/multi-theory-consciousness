#!/usr/bin/env python3
"""Initialize every enabled store, record the install date and seed (mtc db init).

Usage: python scripts/db/bootstrap.py [--only STORE...] [--no-seed] [--env-file PATH] [--json]
A thin wrapper: the work is done by mtc.db.bootstrap, so an installed package can run
the same step as mtc db init from any directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mtc.db.bootstrap import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
