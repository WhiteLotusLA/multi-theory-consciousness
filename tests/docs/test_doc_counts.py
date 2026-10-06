"""The theory and indicator counts in the documents equal the code's.

The documents are read from the repository and every expected number comes
from ``ConsciousnessAssessment`` and ``ConsciousnessTheory`` at test time, so
a document that keeps an old count, or code that adds an indicator without
the documents following, fails here.
"""

import re
from pathlib import Path

import pytest

from mtc.assessment.assessment import ConsciousnessAssessment, ConsciousnessTheory

ROOT = Path(__file__).resolve().parents[2]
SCANNED = (
    "README.md",
    "docs/GETTING_STARTED.md",
    "docs/ARCHITECTURE.md",
    "docs/CONSCIOUSNESS_THEORIES.md",
    "docs/HONEST_LIMITATIONS.md",
    "docs/BUTLIN_2025_MAPPING.md",
)
STALE = (
    re.compile(r"\b20 indicators\b", re.I),
    re.compile(r"\b25 indicators\b", re.I),
    re.compile(r"\b7 theories\b", re.I),
    re.compile(r"\bseven theories\b", re.I),
    re.compile(r"dcm_scoring"),
    re.compile(r"mtc-assess"),
    re.compile(r"C\+\+[^\n]{0,80}\bHTM\b|\bHTM\b[^\n]{0,80}C\+\+"),
)
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
THEORY_IDS = "|".join(t.name for t in ConsciousnessTheory)


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _number(token: str) -> int:
    token = token.lower()
    return NUMBER_WORDS[token] if token in NUMBER_WORDS else int(token)


@pytest.fixture(scope="module")
def assessment() -> ConsciousnessAssessment:
    return ConsciousnessAssessment()


@pytest.mark.parametrize("relative", SCANNED)
def test_no_stale_counts(relative):
    text = _text(relative)
    hits = [p.pattern for p in STALE if p.search(text)]
    assert hits == [], f"{relative} carries stale wording: {hits}"


def test_indicator_counts_match_code(assessment):
    positive = len(assessment.indicator_configs)
    negative = len(assessment.negative_indicator_configs)
    theories = len(list(ConsciousnessTheory))
    by_theory = {}
    for config in assessment.indicator_configs.values():
        name = config["theory"].name
        by_theory[name] = by_theory.get(name, 0) + 1

    positive_re = re.compile(r"\b(\d+) positive indicators\b", re.I)
    negative_re = re.compile(r"\b(\d+|three) negative (?:indicators|checks)\b", re.I)
    plus_re = re.compile(r"\b(\d+) indicators? \(?plus (\d+) negative", re.I)
    theory_re = re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten) theories\b", re.I
    )
    mentions = {"positive": 0, "negative": 0, "theories": 0}
    for relative in SCANNED:
        text = _text(relative)
        for match in positive_re.finditer(text):
            mentions["positive"] += 1
            assert int(match.group(1)) == positive, f"{relative}: {match.group(0)}"
        for match in negative_re.finditer(text):
            mentions["negative"] += 1
            assert _number(match.group(1)) == negative, f"{relative}: {match.group(0)}"
        for match in plus_re.finditer(text):
            assert int(match.group(1)) == positive, f"{relative}: {match.group(0)}"
            assert int(match.group(2)) == negative, f"{relative}: {match.group(0)}"
        for match in theory_re.finditer(text):
            mentions["theories"] += 1
            assert _number(match.group(1)) == theories, f"{relative}: {match.group(0)}"
    # Every kind of count is stated somewhere, so the check above is not vacuous.
    assert all(count > 0 for count in mentions.values()), mentions

    # The per-theory counts in the theory overview and the architecture table.
    overview = _text("docs/CONSCIOUSNESS_THEORIES.md")
    stated = dict(
        (name, int(count))
        for name, count in re.findall(
            rf"\b({THEORY_IDS})\s+\((\d+)(?:\s+indicators)?\)", overview
        )
    )
    assert stated == by_theory

    architecture = _text("docs/ARCHITECTURE.md")
    rows = dict(
        (name, int(count))
        for name, count in re.findall(
            rf"^\| ({THEORY_IDS}) \|[^\n]*\| (\d+) \|$", architecture, re.M
        )
    )
    assert rows == by_theory

    readme = _text("README.md")
    rows = dict(
        (name, int(count))
        for name, count in re.findall(
            rf"^\| [^|\n]*\(({THEORY_IDS})\) \|[^\n]*\| (\d+) \|$", readme, re.M
        )
    )
    assert rows == by_theory
