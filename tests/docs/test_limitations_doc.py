"""docs/HONEST_LIMITATIONS.md states every limitation the release records.

Each limitation sentence is pinned by key phrases in the section it belongs
to, so a sentence that is dropped, reworded away or moved to another section
fails its own case. The three sentences that replaced older bullets also pin
that the older wording is gone, and three sentences that belong beside an
existing bullet are checked inside that bullet. Bold markup and line breaks
are ignored; every other character of a phrase must match.

The exact sentences for other documents (the night prune in the first-run
guide and the operations guide, the idle sampler note in the first-run
guide) are pinned here as well, and so is the absence of the two reserve
sentences that would apply only had the maintainer ruled the other way.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
LIMITATIONS = DOCS / "HONEST_LIMITATIONS.md"


def _normalized(text: str) -> str:
    """The text with bold markup removed and every whitespace run as one space."""
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def _section(number: int) -> str:
    """The normalized body of the numbered section ``## <number>. ...``."""
    text = LIMITATIONS.read_text(encoding="utf-8")
    match = re.search(rf"^## {number}\. [^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"HONEST_LIMITATIONS.md has no section {number}"
    return _normalized(match.group(1))


def _bullet(section: int, lead: str) -> str:
    """The normalized bullet of a section whose text contains ``lead``."""
    text = LIMITATIONS.read_text(encoding="utf-8")
    match = re.search(rf"^## {section}\. [^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"HONEST_LIMITATIONS.md has no section {section}"
    bullets = re.split(r"\n(?=- )", match.group(1))
    found = [_normalized(b) for b in bullets if lead in _normalized(b)]
    assert len(found) == 1, f"section {section}: {len(found)} bullets hold {lead!r}"
    return found[0]


# (sentence id, section, key phrases). Every phrase must appear in that
# section; the ids follow the release plan's list of limitation sentences.
SENTENCES = (
    (
        "L1",
        14,
        (
            "Oscillatory binding does not discriminate inputs.",
            "the stimulus projection keeps only whether a metric is nonzero, "
            "not its size",
            "scores 0.0 as floor-saturated",
        ),
    ),
    (
        "L2",
        14,
        (
            "The DCM benchmark does not compute a usable posterior.",
            "equals the prior mean (about 0.167) for any evidence",
            "reports the collected indicator evidence only unless "
            "`--experimental-posterior` is given",
        ),
    ),
    (
        "L3",
        6,
        (
            "indicators score 0.0 as noise-saturated",
            "cannot register there until the runner is fed structured content",
        ),
    ),
    (
        "L4",
        14,
        (
            "The temporal binding experiment is not part of the default battery.",
            "the expected answer cannot be perceived by the system under test",
        ),
    ),
    (
        "L5",
        6,
        (
            "The three negative indicators are two-level rules, not graded "
            "measurements",
            "Their penalty can lower the overall score by at most about 0.1.",
        ),
    ),
    (
        "L6",
        14,
        (
            "The production HTM and the compact HTM both ship",
            "no accelerated or native variant ships",
        ),
    ),
    (
        "L7",
        11,
        (
            "gives active inference an observation vector with no mass",
            "counted in zero_mass_observations",
        ),
    ),
    (
        "L8",
        11,
        (
            "needs a token-attention vector that no component produces",
            "is left out of the legacy aggregate",
        ),
    ),
    (
        "L9",
        11,
        (
            "computes an emotional regulation value that nothing applies",
            "the value is reported under skipped",
        ),
    ),
    (
        "L10",
        12,
        (
            "With no brain the memory consolidation pipeline writes no gist.",
            "mechanical digests of its own recent thoughts",
        ),
    ),
    (
        "L11",
        11,
        (
            "scores its focus prediction only in the cycle after a hyperfocus",
            "until a hyperfocus cycle has been scored",
        ),
    ),
    (
        "L12",
        11,
        (
            "A voluntary attention request does not enter the workspace competition",
            "ast.voluntary_boost has no effect",
        ),
    ),
    (
        "L13",
        6,
        (
            "global_broadcast and recurrent_processing report insufficient_data "
            "unless a caller registers broadcast receivers",
        ),
    ),
    (
        "L14",
        6,
        ("The ignition detector has no sustain measurement",),
    ),
    (
        "L15",
        11,
        (
            "is not a time-series integration measure",
            "the pipeline's momentary measure is therefore None on every turn",
        ),
    ),
    (
        "L16",
        13,
        (
            "an input missing from pad_sources_measured contributes nothing",
            "The affect label reads Neutral inside a pleasure band of 0.1 "
            "around zero.",
        ),
    ),
    (
        "L17",
        11,
        (
            "two of them read other layers",
            "separates those signals, not the attention schema or the "
            "higher-order thought module",
        ),
    ),
    (
        "L18",
        14,
        (
            "a result is labelled exact for the algorithm and not for the data",
            "observes at most 24 of 32",
        ),
    ),
    (
        "L19",
        11,
        (
            "every candidate receives a neutral relevance of 0.5",
            "The goal relevance a producer sets on a candidate is read by the "
            "gate only.",
        ),
    ),
    (
        "L20",
        13,
        (
            "The NARRATE_ keys govern the pipeline",
            "a module constructed directly keeps its v0.4.0 report strings",
        ),
    ),
    (
        "L21",
        13,
        (
            "The proxy has a sampling floor",
            "a positive value soon after warm-up is not evidence of integration",
        ),
    ),
    (
        "L22",
        11,
        ("the sustain timer is keyed by an identifier that is new every cycle",),
    ),
    (
        "L23",
        12,
        (
            "During a Redis outage the working-memory wrapper answers reads "
            "from what was written during the outage only.",
            "(`served_from_fallback`)",
        ),
    ),
    (
        "L24",
        12,
        (
            "The safety gate is a keyword heuristic",
            "suspends every untrusted conversation until a trusted speaker "
            "resumes them with `mtc safety resume`",
        ),
    ),
    (
        "L25",
        13,
        (
            "the HTTP routes do not validate the Host header",
            "keep authentication on whenever a browser is used on that machine",
        ),
    ),
    (
        "L26",
        11,
        (
            "Bayesian binding quality is a correlation proxy for mutual information",
            "candidates without their own vector are not bound",
        ),
    ),
    (
        "L27",
        11,
        ("it is a proxy for recursive self-reference, not an observation of it",),
    ),
    (
        "L28",
        11,
        ("it does not measure whether the goal was achieved",),
    ),
    (
        "L29",
        11,
        (
            "describes the system's affect around that label, not anything the "
            "speaker did",
        ),
    ),
    (
        "L30",
        14,
        (
            "The Goal Drift Index is a heuristic",
            "as fully drifted as soon as it appears",
        ),
    ),
    (
        "L31",
        11,
        (
            "A thought composed from a seed template can win the workspace",
            "its record and its broadcast frame carry the form template",
        ),
    ),
    (
        "L32",
        12,
        (
            "A bare install cannot run the assessment, the server, the examples "
            "or the demo",
            "each of them stops with one line that names the extra",
        ),
    ),
    (
        "refractory-clock",
        13,
        (
            "A workspace built directly keeps the simulated per-cycle clock for "
            "its refractory period",
            "sets `gwt.refractory_clock = elapsed`",
        ),
    ),
)


@pytest.mark.parametrize(
    "section, phrases", [s[1:] for s in SENTENCES], ids=[s[0] for s in SENTENCES]
)
def test_limitation_sentence_is_stated_in_its_section(section, phrases):
    body = _section(section)
    missing = [phrase for phrase in phrases if phrase not in body]
    assert missing == [], f"section {section} lacks {missing}"


def test_replaced_bullets_keep_none_of_their_old_wording():
    """The gist, PAD and HTM bullets were replaced, not extended."""
    text = _normalized(LIMITATIONS.read_text(encoding="utf-8"))
    for old in (
        "With no brain the memory layer stores and retrieves text and writes "
        "no gist.",
        "PAD inputs absent in a cycle use declared neutral defaults",
        "an input missing from that list contributed its default, not a reading",
        "Only `ProductionHTM` ships.",
    ):
        assert old not in text, old


def test_limitations_name_the_thought_machine_digests_beside_the_gist_sentence():
    bullet = _bullet(12, "Memory gists and knowledge synthesis exist only when")
    assert "The continuous thought machine separately writes mechanical" in bullet
    assert "into the same gist store" in bullet
    assert "surface into the prompt like any gist" in bullet


def test_limitations_state_where_pad_provenance_is_kept():
    bullet = _bullet(13, "PAD is computed from the inputs measured in the cycle")
    assert (
        "keep the coordinate without the list of inputs it was computed from" in bullet
    )
    assert "only the turn's provenance snapshot carries `pad_sources_measured`" in (
        bullet
    )
    assert "the two SNN inputs are never among them" in bullet


def test_appended_sentences_sit_in_the_bullet_they_qualify():
    phi = _bullet(14, "Module-graph Phi is an integration proxy, never narrated.")
    assert "The matrix is empirical" in phi
    proxy = _bullet(13, "integration proxy over five module signals")
    assert "The proxy has a sampling floor" in proxy
    narration = _bullet(13, "Narration is off by default")
    assert "The NARRATE_ keys govern the pipeline" in narration
    safety = _bullet(12, "Safety scoring, the trust compass's simple mode")
    assert "The safety gate is a keyword heuristic" in safety


def test_reserve_sentences_are_absent():
    """The maintainer kept the CI job and the fail-closed gate, so neither
    reserve sentence applies."""
    text = _normalized(LIMITATIONS.read_text(encoding="utf-8"))
    assert "The safety gate fails open" not in text
    assert "CI runs neither" not in text


NIGHT_PRUNE = (
    "With the sleep scheduler on, each night sleep deletes memories the pruner "
    "scores as weak; set SLEEP_PRUNE_DRY_RUN=true to log what would be removed "
    "without deleting it."
)
IDLE_SAMPLES = (
    "idle samples repeat the last state, so they add transitions that carry no "
    "module dynamics and pull the proxy toward its floor"
)


@pytest.mark.parametrize("document", ["FIRST_RUN.md", "OPERATIONS.md"])
def test_night_prune_deletion_is_stated(document):
    text = _normalized((DOCS / document).read_text(encoding="utf-8"))
    assert NIGHT_PRUNE in text


def test_first_run_states_what_idle_samples_do_to_the_proxy():
    text = _normalized((DOCS / "FIRST_RUN.md").read_text(encoding="utf-8"))
    assert IDLE_SAMPLES in text


def test_module_graph_bullet_qualifies_the_separation_by_cycle():
    """On a cycle where the protoself and core-layer signals are unmeasured
    the AST and HOT nodes read the attention focus and the thought count, so
    the separation claim is qualified the way the tracker's docstring is."""
    bullet = _bullet(11, "two of them read other layers")
    assert (
        "separates those signals, not the attention schema or the higher-order "
        "thought module, once they are measured, and the attention focus or the "
        "thought count on a cycle where they are not" in bullet
    )


def test_operations_states_the_pass_level_delete_confirmation():
    text = _normalized((DOCS / "OPERATIONS.md").read_text(encoding="utf-8"))
    assert (
        "an unconfirmed delete keeps the memory's vector point and graph node"
        not in (text)
    )
    assert (
        "a pass whose delete the store confirmed for no row keeps the vector "
        "points and graph nodes of all its rows; once at least one row is "
        "confirmed, the index entries of every row of the pass are deleted" in text
    )
