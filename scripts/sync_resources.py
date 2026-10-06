#!/usr/bin/env python3
"""Copy the shipped runtime defaults into mtc/resources/ as package data.

Usage: python scripts/sync_resources.py [--check]
--check writes nothing and exits 1 when a packaged copy is missing, differs
from its source or is stale. The manifest is
mtc.core.resources.RESOURCE_MANIFEST; edit the sources, never the copies.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtc.core.resources import sync_resources  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    changed = sync_resources(ROOT, check=args.check)
    for relative in changed:
        if args.check:
            print(f"DRIFT: mtc/resources/{relative}")
        else:
            print(f"wrote mtc/resources/{relative}")
    if args.check:
        return 1 if changed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
