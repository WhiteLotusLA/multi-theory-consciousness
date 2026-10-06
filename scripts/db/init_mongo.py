#!/usr/bin/env python3
"""Create the MongoDB indexes from the schema registry (mtc db init --only mongodb).

Usage: python scripts/db/init_mongo.py [--no-seed] [--env-file PATH] [--json]
A thin wrapper: the work is done by mtc.db.bootstrap, so an installed package can run
the same step as mtc db init --only mongodb from any directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mtc.db.bootstrap import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["--only", "mongodb", *sys.argv[1:]]))
