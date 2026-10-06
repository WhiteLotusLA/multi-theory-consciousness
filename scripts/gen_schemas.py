#!/usr/bin/env python3
"""Regenerate the JSON schemas under docs/schemas from the onboarding models.

Usage: python scripts/gen_schemas.py [--check] [--dir DIR]

Writes docs/schemas/persona.schema.json (the persona file) and
docs/schemas/onboarding-answers.schema.json (the setup answers) from
mtc.onboarding.schema.export_json_schemas(), as JSON with sorted keys, a
two-space indent and a trailing newline. The files are generated and never
edited by hand.

--check writes nothing and exits 1 when a committed file is missing or its
parsed JSON differs from the models. A pydantic upgrade can change how a
schema is rendered; when the check fails after one, regenerate the files
with this script and commit them.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtc.onboarding.schema import export_json_schemas  # noqa: E402

SCHEMA_DIR = ROOT / "docs" / "schemas"


def render(document: Dict) -> str:
    """The committed text of one schema."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def drift(directory: Path) -> List[str]:
    """File names that are missing or whose parsed JSON differs."""
    changed = []
    for name, document in export_json_schemas().items():
        path = directory / name
        try:
            committed = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            changed.append(name)
            continue
        if committed != json.loads(render(document)):
            changed.append(name)
    return sorted(changed)


def write(directory: Path) -> List[str]:
    """Write every schema; return the file names written."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, document in export_json_schemas().items():
        (directory / name).write_text(render(document), encoding="utf-8")
        written.append(name)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dir", type=Path, default=SCHEMA_DIR)
    args = parser.parse_args(argv)
    directory = args.dir
    if args.check:
        changed = drift(directory)
        for name in changed:
            print(f"DRIFT: {name}")
        if changed:
            print(
                "The committed JSON schemas differ from the onboarding models. "
                "Regenerate them with: python scripts/gen_schemas.py "
                "(a pydantic upgrade can change the rendering; regenerate "
                "after one).",
                file=sys.stderr,
            )
            return 1
        return 0
    for name in write(directory):
        print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
