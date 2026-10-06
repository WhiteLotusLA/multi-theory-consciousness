#!/usr/bin/env python3
"""Apply the Neo4j constraints and seed the graph (mtc db init --only neo4j).

Usage: python scripts/db/init_neo4j.py [--no-seed] [--env-file PATH] [--json]
A thin wrapper: the work is done by mtc.db.bootstrap, so an installed package can run
the same step as mtc db init --only neo4j from any directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mtc.db.bootstrap import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["--only", "neo4j", *sys.argv[1:]]))
