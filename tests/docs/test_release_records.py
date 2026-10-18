"""The release records say what was decided, and retired files stay gone.

- docs/MIGRATION_FROM_V04.md records results, not open questions: it holds
  neither the words "decision pending" nor "a report, not a result".
- experiments/program.md, the research-loop notes the release removed, does
  not come back.
- The four headline numbers in the migration record's opening and in the
  headline paragraph of the changelog's 0.5.0 section, each with its passing
  count and, where the fixture records them, its status and coverage, and the
  seeded --full result on the first-run page, equal the committed fixtures
  they come from.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "docs" / "MIGRATION_FROM_V04.md"


def test_migration_record_holds_no_pending_decision():
    text = " ".join(MIGRATION.read_text(encoding="utf-8").split()).lower()
    for phrase in ("decision pending", "a report, not a result"):
        assert phrase not in text, f"MIGRATION_FROM_V04.md still says {phrase!r}"


def test_experiments_program_file_does_not_exist():
    assert not (ROOT / "experiments" / "program.md").exists()


FIXTURES = ROOT / "tests" / "fixtures"


def _printed(name: str, fixture: str) -> str:
    """The printed value of one metric line of an experiment-runner fixture."""
    text = (FIXTURES / fixture).read_text(encoding="utf-8")
    match = re.search(rf"^{name}:\s+(\S+)$", text, re.M)
    assert match, f"{fixture} has no {name} line"
    return match.group(1)


def _report(fixture: str) -> dict:
    return json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))


def _headline_phrases() -> dict:
    """The four headline numbers as the documents print them, read from the
    committed fixtures: the experiment runner under the legacy preset (the
    v0.4.0 run) and under the defaults, and the seeded assessment under
    both. Each run is one phrase: its overall, its passing count out of the
    indicator total and, where the fixture records them, its status and its
    coverage, in that order and joined, so that no fact can be met by another
    run's phrase or by a sentence elsewhere."""
    legacy_run = "baseline_v040_experiment.txt"
    default_run = "baseline_v050_experiment.txt"
    legacy = _report("baseline_legacy_assessment.json")
    default = _report("baseline_v050_assessment.json")

    def experiment(fixture: str) -> str:
        phrase = (
            f"{_printed('overall_score', fixture)}, "
            f"{_printed('passing_count', fixture)} of "
            f"{_printed('total_indicators', fixture)} passing"
        )
        text = (FIXTURES / fixture).read_text(encoding="utf-8")
        if re.search(r"^assessment_status:", text, re.M):
            phrase += (
                f", status {_printed('assessment_status', fixture)}"
                f", coverage {_printed('coverage', fixture)}"
            )
        return phrase

    def assessment(report: dict) -> str:
        return (
            f"{report['overall_score']:.6f}, "
            f"{report['passing_count']} of {report['total_indicators']} passing, "
            f"status {report['status']}, coverage {report['coverage']:.3f}"
        )

    return {
        "legacy experiment": experiment(legacy_run),
        "default experiment": experiment(default_run),
        "legacy assessment": assessment(legacy),
        "default assessment": assessment(default),
    }


def _stated(phrase: str, text: str) -> bool:
    """The phrase stands in the text as a whole: no digit or decimal point
    runs into its first number, and no digit runs on from its last."""
    pattern = rf"(?<![\d.]){re.escape(phrase)}(?!\d)"
    return re.search(pattern, text) is not None


def _migration_opening() -> str:
    text = MIGRATION.read_text(encoding="utf-8")
    return text.split("## Why the numbers changed", 1)[0]


def _changelog_headline() -> str:
    """The headline paragraph of the changelog's 0.5.0 section."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = text.split("## [0.5.0]", 1)[1].split("\n### ", 1)[0]
    paragraphs = [
        p for p in section.split("\n\n") if p.startswith("The four headline numbers")
    ]
    assert len(paragraphs) == 1, "the 0.5.0 section has no single headline paragraph"
    return paragraphs[0]


@pytest.mark.parametrize(
    "where, read",
    [
        ("MIGRATION_FROM_V04.md", _migration_opening),
        ("CHANGELOG.md", _changelog_headline),
    ],
)
def test_headline_numbers_equal_the_committed_fixtures(where, read):
    text = " ".join(read().split())
    missing = [
        (run, phrase)
        for run, phrase in _headline_phrases().items()
        if not _stated(phrase, text)
    ]
    assert missing == [], f"{where} lacks {missing}"


def test_migration_opening_announces_the_erratum_note_without_a_path():
    """No erratum file exists in the tree, so the opening announces the note
    and links nothing: the paragraph that announces it holds no link, no
    slash and no file name."""
    paragraphs = [
        " ".join(p.split())
        for p in _migration_opening().split("\n\n")
        if "erratum" in p
    ]
    assert len(paragraphs) == 1, "the opening does not announce the erratum note once"
    paragraph = paragraphs[0]
    assert re.search(
        r"The maintainer will publish an erratum note for the paper's v0\.4\.0 "
        r"numbers; until then",
        paragraph,
    ), paragraph
    assert "](" not in paragraph and "/" not in paragraph, paragraph
    file_name = re.search(
        r"[\w-]+\.(?:md|pdf|txt|tex|html?|rst|docx?)\b", paragraph, re.I
    )
    assert file_name is None, file_name.group(0)


def test_first_run_full_mode_paragraph_equals_the_fixture():
    """The first-run page states the seeded --full result as the committed
    fixture records it, and names every indicator that has no data."""
    report = _report("baseline_v050_assessment.json")
    text = (ROOT / "docs" / "FIRST_RUN.md").read_text(encoding="utf-8")
    match = re.search(r"^`mtc assess --full`.*?(?=\n\n)", text, re.M | re.S)
    assert match, "FIRST_RUN.md has no --full paragraph"
    paragraph = " ".join(match.group(0).split())
    statuses = [r["status"] for r in report["indicator_results"].values()]
    expected = [
        f"overall {report['overall_score']:.6f}",
        f"{report['passing_count']} of {report['total_indicators']} passing",
        f"status `{report['status']}`",
        f"coverage {report['coverage']:.3f}",
        f"{statuses.count('measured')} indicators measured",
        f"{statuses.count('insufficient_data')} `insufficient_data`",
        f"{statuses.count('absent')} `absent`",
    ]
    expected += [f"`{name}`" for name in report["insufficient_indicators"]]
    missing = [phrase for phrase in expected if phrase not in paragraph]
    assert missing == [], f"the --full paragraph lacks {missing}"
