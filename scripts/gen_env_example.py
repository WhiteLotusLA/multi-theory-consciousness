#!/usr/bin/env python3
"""Regenerate .env.example and docs/CONFIGURATION.md from Settings.

Usage: python scripts/gen_env_example.py [--check]
--check exits 1 when the committed files differ from the rendered output.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtc.core.config import Settings  # noqa: E402
from mtc.core.env_example import (  # noqa: E402
    render_configuration_md,
    render_env_example,
)

TARGETS = {
    ROOT / ".env.example": render_env_example,
    ROOT / "docs" / "CONFIGURATION.md": render_configuration_md,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    drift = 0
    for path, render in TARGETS.items():
        rendered = render(Settings)
        if args.check:
            if not path.exists() or path.read_text() != rendered:
                print(f"DRIFT: {path.relative_to(ROOT)}")
                drift += 1
        else:
            path.write_text(rendered)
            print(f"wrote {path.relative_to(ROOT)}")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
