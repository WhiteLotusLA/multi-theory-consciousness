"""docs/BUTLIN_2025_MAPPING.md has one row per indicator with the code's theory,
paper id and polarity, and no row for an indicator the code does not define."""

import re
from pathlib import Path

from mtc.assessment.assessment import ConsciousnessAssessment

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "BUTLIN_2025_MAPPING.md"
EXTENSION = "extension"


def _rows(text: str):
    rows = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        match = re.fullmatch(r"`([a-z_]+)`", cells[0])
        if match:
            assert match.group(1) not in rows, f"two rows for {match.group(1)}"
            rows[match.group(1)] = cells[1:4]
    return rows


def test_every_indicator_and_id_documented():
    assessment = ConsciousnessAssessment()
    expected = {}
    for configs, polarity in (
        (assessment.indicator_configs, "positive"),
        (assessment.negative_indicator_configs, "negative"),
    ):
        for name, config in configs.items():
            expected[name] = [
                config["theory"].name,
                config.get("butlin_id") or EXTENSION,
                config.get("polarity") or polarity,
            ]
    rows = _rows(DOC.read_text(encoding="utf-8"))
    assert set(rows) == set(expected), (
        f"missing: {sorted(set(expected) - set(rows))}; "
        f"unknown: {sorted(set(rows) - set(expected))}"
    )
    for name, cells in expected.items():
        assert rows[name] == cells, name
