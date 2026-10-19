"""The runtime, brain, thought, testing and audit pages follow the code.

Each expected value is read from the code at test time: the absent reasons
and the safety gate states of the turn record, the post-turn bound, the
brain's health re-check interval, the seed packs' slot words and refused
affect slots, and the three wall-clock evidence keys of the baseline
comparators. The privacy audit page is held to the boundary rule's wording.
Bold markup and line breaks are ignored.
"""

import ast
import re
from pathlib import Path

import pytest

from mtc.brain.brain import HEALTH_RECHECK_S
from mtc.consciousness.continuous.seeds import AFFECT_SLOTS, SLOT_KEYS
from mtc.pipeline.context_types import ABSENT_REASONS, SAFETY_GATE_STATES
from tests.baseline_compare import WALL_CLOCK_EVIDENCE_KEYS

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _text(name: str) -> str:
    raw = (DOCS / name).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", raw.replace("**", "")).strip()


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def test_runtime_doc_names_every_absent_reason():
    text = _text("RUNTIME_PIPELINE.md")
    section = text.split("## Absent signals", 1)[1].split(" ## ", 1)[0]
    missing = [reason for reason in ABSENT_REASONS if f"`{reason}`" not in section]
    assert missing == [], f"RUNTIME_PIPELINE.md does not name {missing}"


def test_runtime_doc_names_every_safety_gate_state():
    text = _text("RUNTIME_PIPELINE.md")
    step = text.split("4. Safety gate.", 1)[1].split("5. Store the message.", 1)[0]
    missing = [state for state in SAFETY_GATE_STATES if f"`{state}`" not in step]
    assert missing == [], f"the safety gate step does not name {missing}"
    assert "`SAFETY_GATE_FAIL_CLOSED`" in step
    assert "`safety_unavailable`" in step


def test_runtime_doc_states_the_post_turn_bound():
    pytest.importorskip("pymdp")
    from mtc.pipeline.conversation_pipeline import PipelineConfig

    seconds = _number(PipelineConfig().post_turn_timeout_s)
    text = _text("RUNTIME_PIPELINE.md")
    assert f"`PipelineConfig.post_turn_timeout_s` ({seconds} seconds)" in text
    assert "`on_generated`" in text


def test_llm_setup_states_the_health_recheck_interval():
    text = _text("LLM_SETUP.md")
    assert (
        f"at most once every {_number(HEALTH_RECHECK_S)} seconds "
        "(`HEALTH_RECHECK_S`" in text
    )


def test_continuous_thought_doc_lists_the_slot_words_and_refuses_the_affect_slots():
    text = _text("CONTINUOUS_THOUGHT.md")
    keys = text.split("the slot word lists", 1)[1].split("; and `description`", 1)[0]
    listed = set(re.findall(r"`([a-z_]+)`", keys))
    assert listed == set(SLOT_KEYS)
    for slot in AFFECT_SLOTS:
        assert f"`{{{slot}}}`" in text, slot
    assert "refused in every template family" in text


def test_testing_doc_names_the_three_wall_clock_keys():
    text = _text("TESTING.md")
    for indicator, key in WALL_CLOCK_EVIDENCE_KEYS:
        assert f"`{indicator}`" in text, indicator
        assert f"`{key}`" in text, key
    assert "`WALL_CLOCK_EVIDENCE_KEYS`" in text
    assert "no score, status, confidence, count or printed line depends on them" in (
        text
    )


def test_privacy_audit_doc_states_the_boundary_rule():
    text = _text("PRIVACY_AUDIT.md")
    assert "match whole words" not in text
    for phrase in (
        "where it starts after anything that is not a letter",
        "at a camelCase boundary: a lower-case letter, then the token with its "
        "first letter capitalised and not followed by another lower-case letter",
        "Inside a longer word after a letter it is not a hit",
        "The folded view is not the full Unicode confusables set.",
        "only literal-built expressions are assembled",
    ):
        assert phrase in text, phrase


def test_privacy_audit_doc_says_which_patch_hits_carry_a_commit():
    """In patch mode only a Python literal-built hit is assembled per commit
    and names it; a decoded-view or folded-view hit counts added lines and
    carries no commit."""
    text = _text("PRIVACY_AUDIT.md")
    assert "an assembled hit carries the file's line and" not in text
    assert "the hidden forms are assembled in each commit's own version" not in text
    for phrase in (
        "a hit in the decoded or folded view of a line counts added lines within "
        "that file and carries no commit",
        "Only literal-built expressions in a Python file are assembled in each "
        "commit's own version of the file on the lines that commit added",
        "a Python literal-built hit carries the file's line in that commit's "
        "version and `@<12-hex hash>`",
    ):
        assert phrase in text, phrase


def _salience_bases() -> set:
    """Every salience basis value the thought machine's sources write."""
    from mtc.consciousness.continuous.enhancer import LLM_SALIENCE_BASIS

    package = ROOT / "mtc" / "consciousness" / "continuous"
    generator = (package / "generator.py").read_text(encoding="utf-8")
    machine = (package / "machine.py").read_text(encoding="utf-8")
    found = set(re.findall(r'\bbasis = (?:\w+, )?"(\w+)"', generator))
    found |= set(re.findall(r'"salience_basis": "(\w+)"', machine))
    found.add(LLM_SALIENCE_BASIS)
    return found


def test_continuous_thought_doc_lists_every_salience_basis():
    bases = _salience_bases()
    assert {"affect", "memory_importance", "strategy_default"} <= bases
    assert "injected_default" in bases
    text = _text("CONTINUOUS_THOUGHT.md")
    missing = sorted(b for b in bases if f"`{b}`" not in text)
    assert missing == [], missing


def _workspace_constants() -> dict:
    """The module-level string constants of the workspace module, read from
    its source so that no workspace is built or imported."""
    path = ROOT / "mtc" / "consciousness" / "enhanced_global_workspace.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def test_testing_doc_states_the_recency_clock():
    """The one clock read the cooldown pin left on the default path, the
    recency bonus, is named as the cause of the recorded 1.2e-9 move and as
    running on the simulated clock under the defaults, with the switch value
    that restores the wall-clock read taken from the workspace module."""
    constants = _workspace_constants()
    simulated = constants["RECENCY_CLOCK_SIMULATED"]
    wall = constants["RECENCY_CLOCK_V04_WALL"]
    text = _text("TESTING.md")
    assert (
        "The pin removes the gate, not every clock read: one evidence value of "
        "the default seeded assessment" not in text
    )
    for phrase in (
        "the attention bottleneck's recency bonus",
        "moved `genuine_implementation.evidence.avg_loop_quality` by 1.2e-9 when "
        "every clock ran 25 times faster",
        f"(`gwt.recency_clock`, default `{simulated}`; `{wall}` restores the "
        "v0.4.0 wall-clock age)",
        "under the defaults the seeded assessment reads no wall clock into a "
        "score or an evidence value",
        "the legacy seeded assessment run at clock speeds 1, 1 and 25 differs in "
        "exactly those three readouts",
        "the 1e-9 bound was not widened for it",
    ):
        assert phrase in text, phrase


def test_llm_setup_states_that_a_raising_recheck_is_unhealthy():
    text = _text("LLM_SETUP.md")
    assert (
        "A re-check whose health probe raises any error counts as unhealthy, so "
        "the call still raises `BrainUnavailableError` and the turn answers 503 "
        "(`brain_unavailable`)." in text
    )
