#!/usr/bin/env python3
"""Backfill Qdrant points for PostgreSQL episodic rows that have none.

Usage: python scripts/db/reembed_postgres_to_qdrant.py [--dry-run] [--all] [--batch-size N] [--limit N] [--json]
A thin wrapper: the work is done by mtc.db.reembed, so an installed package can run
the same step as python3 -m mtc.db.reembed from any directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mtc.db.reembed import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
