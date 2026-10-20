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
- The changelog's interface lists name the release's breaking changes to
  the brain protocol, the DCM result, the neural orchestrator's state, the
  dataclass fields that became Optional and the legacy framework, and its
  new-field list names the fields the release added to them; every named
  change is read back from the source, so the record and the code agree.
"""

import ast
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


def test_migration_opening_announces_the_paper_update_without_a_path():
    """The paper draft was never submitted (no arXiv endorsement), so it gets
    no erratum: the opening announces that the maintainer will update the
    draft with a v0.5.0 results section, states that the draft reports twenty
    indicators (the v0.1.1 count, so its numbers are not v0.4.0's) and links
    nothing: the paragraph holds no link, no slash and no file name."""
    joined = (" ".join(p.split()) for p in _migration_opening().split("\n\n"))
    paragraphs = [p for p in joined if "paper draft" in p]
    assert len(paragraphs) == 1, "the opening does not announce the paper update once"
    paragraph = paragraphs[0]
    assert re.search(
        r"The maintainer will update the paper draft, which reports twenty "
        r"indicators \(the v0\.1\.1 count\) and has not been submitted, with a "
        r"v0\.5\.0 results section; until then this record is the reference for "
        r"the difference between v0\.4\.0 and v0\.5\.0\.",
        paragraph,
    ), paragraph
    assert "erratum" not in paragraph and "v0.4.0 numbers" not in paragraph, paragraph
    assert "](" not in paragraph and "/" not in paragraph, paragraph
    file_name = re.search(
        r"[\w-]+\.(?:md|pdf|txt|tex|html?|rst|docx?)\b", paragraph, re.I
    )
    assert file_name is None, file_name.group(0)


def test_version_history_states_the_paper_draft_was_not_submitted():
    """The version history names the paper as a draft prepared for arXiv in
    March 2026 and not submitted (no endorsement), referencing v0.1.1; it
    never calls it a submission."""
    text = (ROOT / "VERSION_HISTORY.md").read_text(encoding="utf-8")
    assert (
        "The MTC paper draft (prepared for arXiv in March 2026 and not submitted) "
        "references **v0.1.1**." in text
    ), "the version history does not state the paper draft's status"
    assert (
        "arXiv submission" not in text
    ), "the version history calls the draft a submission"


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


def _changelog_list(title: str) -> str:
    """One interface list of the changelog's 0.5.0 section, from its heading
    to the next heading."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = text.split("## [0.5.0]", 1)[1].split("\n## [", 1)[0]
    parts = section.split(f"\n### {title}\n", 1)
    assert len(parts) == 2, f"the 0.5.0 section has no list {title!r}"
    return parts[1].split("\n### ", 1)[0]


def _bullet_naming(text: str, name: str) -> str:
    """The one bullet of a list that names ``name`` in backticks."""
    bullets = [
        " ".join(b.split()) for b in re.split(r"\n(?=- )", text) if f"`{name}" in b
    ]
    assert len(bullets) == 1, f"{len(bullets)} bullets name `{name}`"
    return bullets[0]


def _class_node(relpath: str, cls: str) -> ast.ClassDef:
    tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    assert len(nodes) == 1, f"{relpath} defines {cls} {len(nodes)} times"
    return nodes[0]


def _field_annotations(relpath: str, cls: str) -> dict:
    """The annotated attributes of a class as their source text."""
    return {
        n.target.id: ast.unparse(n.annotation)
        for n in _class_node(relpath, cls).body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }


def _parameters(relpath: str, cls: str, method: str) -> list:
    methods = [
        n
        for n in _class_node(relpath, cls).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method
    ]
    assert len(methods) == 1, f"{cls} defines {method} {len(methods)} times"
    args = methods[0].args
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]


# Dataclass fields that became Optional in the release, as (file, class,
# fields): the changelog names the first field qualified by its class and the
# others bare in the same bullet, and the source annotation admits None.
OPTIONAL_FIELDS = [
    (
        "mtc/assessment/dcm_benchmark.py",
        "DCMBenchmarkResult",
        ("overall_probability", "overall_ci"),
    ),
    (
        "mtc/consciousness/beautiful_loop/__init__.py",
        "ConsciousMoment",
        ("binding_quality",),
    ),
    (
        "mtc/consciousness/beautiful_loop/bayesian_binding.py",
        "BoundPercept",
        ("binding_quality", "coherence"),
    ),
    (
        "mtc/consciousness/beautiful_loop/bayesian_binding.py",
        "BindingPair",
        ("mutual_information", "coherence_score", "combined_uncertainty_reduction"),
    ),
    (
        "mtc/consciousness/damasio/protoself.py",
        "ProtoSelfState",
        ("body_delta", "stability", "primordial_feelings"),
    ),
    (
        "mtc/consciousness/damasio/core_consciousness.py",
        "CoreExperience",
        ("feeling_of_knowing",),
    ),
    (
        "mtc/consciousness/damasio/extended_consciousness.py",
        "AutobiographicSelf",
        ("identity_continuity", "narrative_coherence", "temporal_horizon"),
    ),
    (
        "mtc/consciousness/damasio/extended_consciousness.py",
        "ExtendedState",
        ("episodic_resonance",),
    ),
    (
        "mtc/consciousness/beautiful_loop/epistemic_depth.py",
        "SelfReferenceMetrics",
        ("self_model_coherence", "meta_prediction_error", "calibration_quality"),
    ),
    (
        "mtc/assessment/rpt_measurement.py",
        "RecurrenceMetrics",
        (
            "local_recurrence_score",
            "snn_feedback_strength",
            "lsm_reservoir_recurrence",
            "local_spike_reentry",
        ),
    ),
    (
        "mtc/consciousness/rsi/canon_schema.py",
        "ResponseSample",
        ("behavioral_marker_hits", "should_not_contain_hits"),
    ),
]

# Fields the release added, as (file, class, field): the new-field list names
# each one qualified by its class.
NEW_FIELDS = [
    (
        "mtc/neural/oscillatory_binding.py",
        "BindingResult",
        "zero_stimulus_order_parameter",
    ),
    (
        "mtc/consciousness/enhanced_global_workspace.py",
        "IgnitionEvent",
        "sustain_checked",
    ),
    ("mtc/safety/safety_monitor.py", "SafetyAssessment", "alignment_evidence"),
    (
        "mtc/consciousness/enhanced_global_workspace.py",
        "WorkspaceCandidate",
        "submitted_at_s",
    ),
]

# Groups of fields the release added to one class, as (file, class, fields):
# the new-field list names the first qualified by its class and the others
# bare or qualified in the same bullet.
NEW_FIELD_GROUPS = [
    ("mtc/api/schemas.py", "ErrorData", ("blocked", "safety_gate", "safety_decision")),
]

# Methods and properties the release added, as (file, class, member): the
# new-field list names each one qualified by its class.
NEW_MEMBERS = [
    (
        "mtc/consciousness/beautiful_loop/hyper_model.py",
        "HyperModel",
        "has_observed_update",
    ),
    (
        "mtc/consciousness/enhanced_global_workspace.py",
        "AttentionBottleneck",
        "set_simulated_time",
    ),
    (
        "mtc/consciousness/enhanced_global_workspace.py",
        "AttentionBottleneck",
        "simulated_time_s",
    ),
    (
        "mtc/consciousness/enhanced_global_workspace.py",
        "EnhancedGlobalWorkspace",
        "recency_clock",
    ),
]

# Classes whose constructor gained the keyword ``settings`` (read for DEBUG, or
# for the state directory and the project root).
SETTINGS_KEYWORD = [
    ("mtc/consciousness/attention_schema.py", "AttentionSchemaModule"),
    ("mtc/consciousness/active_inference.py", "ActiveInferenceModule"),
    ("mtc/consciousness/autonomous_life.py", "AutonomousLife"),
    ("mtc/consciousness/metacognition.py", "MetacognitionModule"),
    ("mtc/consciousness/state_persistence.py", "ConsciousnessStateManager"),
]

NEURAL_STATE_REMOVED = ("snn_active", "lsm_active", "htm_active")


def test_changelog_breaking_list_names_the_phase9_interface_changes():
    lists = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    lists = lists.split("### Breaking for code built on mtc", 1)[1]
    lists = lists.split("## [0.4.0]", 1)[0]
    for name in (
        "last_turn_at",
        "overall_probability",
        "snn_active",
        "ConsciousMoment.binding_quality",
        "ProtoSelfState",
    ):
        assert name in lists, f"the changelog's interface lists lack {name!r}"

    breaking = _changelog_list("Breaking for code built on mtc")

    # The pipeline passes the keyword on every turn, so a brain written
    # against the earlier protocol fails on its first turn.
    assert "last_turn_at" in _parameters(
        "mtc/brain/protocol.py", "ConversationalBrain", "think_with_context"
    )
    bullet = _bullet_naming(breaking, "ConversationalBrain.think_with_context")
    assert "`last_turn_at" in bullet and "TypeError" in bullet, bullet

    for relpath, cls, fields in OPTIONAL_FIELDS:
        annotations = _field_annotations(relpath, cls)
        bullet = _bullet_naming(breaking, f"{cls}.{fields[0]}")
        for field in fields:
            assert "Optional[" in annotations[field], (cls, field)
            assert (
                f"`{field}`" in bullet or f"`{cls}.{field}`" in bullet
            ), f"the bullet that names {cls}.{fields[0]} lacks {field}: {bullet}"

    state = _field_annotations("mtc/neural/orchestrator.py", "ConsciousnessState")
    bullet = _bullet_naming(breaking, "snn_active")
    assert "mtc.neural.orchestrator" in bullet, bullet
    for field in NEURAL_STATE_REMOVED:
        assert field not in state, field
        assert f"`{field}`" in bullet, (field, bullet)

    framework = ast.parse(
        (ROOT / "mtc/assessment/framework.py").read_text(encoding="utf-8")
    )
    top_level = {
        n.name
        for n in framework.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "test_consciousness_measurement" not in top_level
    _bullet_naming(breaking, "mtc.assessment.framework.test_consciousness_measurement")


def test_changelog_new_field_list_names_the_phase9_additions():
    new = _changelog_list("New fields and public names")

    params = _parameters(
        "mtc/assessment/framework.py",
        "ConsciousnessMeasurementFramework",
        "measure_consciousness",
    )
    assert params[1:4] == ["snn_states", "lsm_states", "htm_states"], params
    bullet = _bullet_naming(
        new, "ConsciousnessMeasurementFramework.measure_consciousness"
    )
    assert "htm_states=None" in bullet, bullet

    for relpath, cls, field in NEW_FIELDS:
        assert field in _field_annotations(relpath, cls), (cls, field)
        _bullet_naming(new, f"{cls}.{field}")

    for relpath, cls, fields in NEW_FIELD_GROUPS:
        annotations = _field_annotations(relpath, cls)
        bullet = _bullet_naming(new, f"{cls}.{fields[0]}")
        for field in fields:
            assert field in annotations, (cls, field)
            assert (
                f"`{field}`" in bullet or f"`{cls}.{field}`" in bullet
            ), f"the bullet that names {cls}.{fields[0]} lacks {field}: {bullet}"

    for relpath, cls, member in NEW_MEMBERS:
        members = {
            n.name
            for n in _class_node(relpath, cls).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert member in members, (cls, member)
        _bullet_naming(new, f"{cls}.{member}")

    bullet = _bullet_naming(new, "settings=")
    for relpath, cls in SETTINGS_KEYWORD:
        assert "settings" in _parameters(relpath, cls, "__init__"), cls
        assert f"`{cls}`" in bullet, (cls, bullet)

    statistics = (ROOT / "mtc/consciousness/metacognition.py").read_text(
        encoding="utf-8"
    )
    assert '"failures": dict(self.failures)' in statistics
    bullet = _bullet_naming(new, "sustain_checked_count")
    assert "`failures` (metacognition)" in bullet, bullet

    lsm = (ROOT / "mtc/neural/liquid/production_lsm.py").read_text(encoding="utf-8")
    assert "self.active_readout = READOUT_RLS" in lsm
    _bullet_naming(new, "ProductionLSM.active_readout")


def _workspace_parameters() -> list:
    return _parameters(
        "mtc/consciousness/enhanced_global_workspace.py",
        "EnhancedGlobalWorkspace",
        "__init__",
    )


def test_changelog_names_this_releases_late_interface_changes():
    """The breaking list names the positional shift of the workspace's
    constructor, the renamed genuine_implementation evidence key and the
    display path of the persistence status; the changed-defaults list names
    the recency clock's default. Each is read back from the source."""
    breaking = _changelog_list("Breaking for code built on mtc")

    params = _workspace_parameters()
    index = params.index("recency_clock")
    assert params[index + 1] == "body_metrics_enabled", params
    bullet = _bullet_naming(breaking, "EnhancedGlobalWorkspace(recency_clock=")
    assert "`body_metrics_enabled`" in bullet and "ValueError" in bullet, bullet

    assessment = (ROOT / "mtc/assessment/assessment.py").read_text(encoding="utf-8")
    assert '"has_meta_prediction_depth": has_meta_prediction_depth' in assessment
    assert '"has_strange_loop": depth_stats.get("max_depth_reached", 0)' in assessment
    bullet = _bullet_naming(breaking, "has_meta_prediction_depth")
    assert "`has_strange_loop`" in bullet and "`genuine_implementation`" in bullet
    assert "v04" in bullet, bullet

    persistence = (ROOT / "mtc/consciousness/state_persistence.py").read_text(
        encoding="utf-8"
    )
    assert re.search(r'"state_dir": display_path\(\s*self\.state_dir\b', persistence)
    bullet = _bullet_naming(breaking, "ConsciousnessStateManager.get_status")
    assert "`state_dir`" in bullet and "never" in bullet, bullet

    tree = ast.parse(
        (ROOT / "mtc/consciousness/enhanced_global_workspace.py").read_text(
            encoding="utf-8"
        )
    )
    constants = {
        n.targets[0].id: n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Constant)
    }
    changed = _changelog_list("Changed defaults")
    bullet = _bullet_naming(changed, "recency_clock")
    assert f"`{constants['RECENCY_CLOCK_SIMULATED']}`" in bullet, bullet
    assert f"`{constants['RECENCY_CLOCK_V04_WALL']}`" in bullet, bullet


def test_migration_record_states_the_v04_stand_ins_and_the_renamed_key():
    """The record says how the v04 formula bodies read a statistic their
    producers now leave None (the v0.4.0 constant where there was one, and 0.0
    for the broadcast coverage and the sustain rate, where v0.4.0 read 1.0): in
    the widened-reach bullet, in the one-switch row and in the Switches-table
    row of assessment.indicator_formulas. It names the renamed evidence key
    has_meta_prediction_depth in the blt.strange_loop_test row and in the
    seeded assessment's genuine_implementation row, and keeps has_strange_loop
    only where it records the rename."""
    text = MIGRATION.read_text(encoding="utf-8")
    assert text.count("where v0.4.0 read 1.0") >= 2, text.count("where v0.4.0 read 1.0")
    lines = text.splitlines()
    rows = [
        line for line in lines if line.startswith("| `assessment.indicator_formulas` |")
    ]
    assert len(rows) == 1, rows
    assert "stand-in" in rows[0] and "where v0.4.0 read 1.0" in rows[0], rows[0]
    assert "as the v0.4.0 constants, so they count" not in rows[0], rows[0]
    strange = [line for line in lines if line.startswith("| `blt.strange_loop_test` |")]
    assert len(strange) == 1 and "has_meta_prediction_depth" in strange[0], strange
    seeded = text.split("\n### The seeded assessment, every indicator that moved\n", 1)[
        1
    ].split("\n### ", 1)[0]
    genuine = [
        line
        for line in seeded.splitlines()
        if line.startswith("| genuine_implementation |")
    ]
    assert len(genuine) == 1 and "has_meta_prediction_depth" in genuine[0], genuine
    for paragraph in text.split("\n\n"):
        if "has_strange_loop" in paragraph:
            assert "has_meta_prediction_depth" in paragraph, paragraph[:200]


COUNT_WORDS = {
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
}


def _v050_switch_rows() -> list:
    """The switches of the migration record's table of the rows added by the
    0.5.0 fixes."""
    text = MIGRATION.read_text(encoding="utf-8")
    section = text.split("\n### Phase 9 rows\n", 1)[1].split("\n### ", 1)[0]
    return re.findall(r"^\| `([a-z_]+\.[a-z_0-9]+)` \|", section, re.M)


def test_changelog_counts_the_new_schema_rows():
    """Every place in the 0.5.0 section that counts the new schema rows of
    the 0.5.0 fixes gives the number of rows the migration record's table lists,
    and every one of them is a row of the parameter schema."""
    rows = _v050_switch_rows()
    # The recency clock is the release gate's last row (the bonus read the
    # wall clock until then); the table and the counts include it.
    assert "gwt.recency_clock" in rows, rows
    schema = (ROOT / "experiments" / "param_schema.toml").read_text(encoding="utf-8")
    missing = [row for row in rows if f"\n[{row}]\n" not in schema]
    assert rows and missing == [], missing
    section = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    section = " ".join(section.split("## [0.5.0]", 1)[1].split("\n## [", 1)[0].split())
    counted = re.findall(
        r"\b(" + "|".join(COUNT_WORDS) + r") (?:new |parameter )?schema rows\b",
        section,
    )
    assert len(counted) == 3, counted
    assert {COUNT_WORDS[word] for word in counted} == {len(rows)}, (counted, rows)
