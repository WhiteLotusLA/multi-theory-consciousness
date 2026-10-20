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
# section; the ids are stable labels, one per limitation sentence.
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
            "In the experiment runner five indicators score 0.0 as "
            "noise-saturated (higher_order_representations, "
            "hierarchical_prediction, temporal_prediction_depth, "
            "agent_environment_coupling and genuine_implementation): the "
            "runner's synthetic warmup fills them on any seed, so they cannot "
            "register there until the runner is fed structured content.",
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
        "L33",
        11,
        (
            "carries a wall-clock recency term (0.2 minus 0.01 per second of the "
            "state's age, floored at zero)",
            "constant to within 1e-5",
        ),
    ),
    (
        "L34",
        11,
        (
            "adds 0.15 times the self-model's confidence",
            "stays at its declared prior of 0.5 until a higher-order thought has "
            "been buffered",
        ),
    ),
    (
        "L35",
        12,
        (
            "The privacy audit's assembled-literal pass evaluates only the literal "
            "forms it lists",
            "HTML character reference in a markdown or HTML file is not seen",
        ),
    ),
    (
        "L36",
        12,
        (
            "In the trust compass's deep mode the brain reads the whole message",
            "only the keyword path leaves already scored phrases out",
        ),
    ),
    (
        "L37",
        12,
        (
            "there is no cooldown, so a hostile untrusted speaker produces one "
            "alert per message",
        ),
    ),
    (
        "L38",
        12,
        (
            "The resume audit row names the escalation role",
            "not the trusted speaker who presented the request",
        ),
    ),
    (
        "L39",
        12,
        (
            "still contacts the model hub at its first load unless "
            "`HF_HUB_OFFLINE=1` is set",
            "the demo injects the hash embedder and loads none",
        ),
    ),
    (
        "L40",
        13,
        (
            "A claimed name written with combining accents (the decomposed form) "
            "loses them in the reduction",
        ),
    ),
    (
        "L41",
        13,
        (
            "the reply is returned only after the turn's post-turn work has "
            "finished",
            "no heartbeat is sent meanwhile",
        ),
    ),
    (
        "L42",
        13,
        (
            "A degraded brain is re-checked only when a call arrives",
            "`/health` keeps reading degraded until the next turn or background "
            "brain call",
        ),
    ),
    (
        "L43",
        13,
        (
            "A turn the safety gate refuses still records a wellbeing contact and "
            "the privacy layer's data-read event",
            "refers to the conversation rows and the memory stores",
        ),
    ),
    (
        "L44",
        14,
        (
            "is reported raw and is never noise-normalised unless the caller "
            "supplies baselines made by the same producer",
        ),
    ),
    (
        "L45",
        14,
        (
            "`has_meta_prediction_depth` (`has_strange_loop` under the v04 "
            "formulas, where it reads a key no producer reports) reports that "
            "the epistemic depth reached 2",
            "not the depth tracker's strange-loop verdict",
        ),
    ),
    (
        "L46",
        14,
        (
            "In the legacy 14-indicator panel the text metrics read missing text "
            "as 0.0",
            "the array metrics are None without their substrates",
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
    (
        "recency-clock",
        13,
        (
            "A workspace built directly ages its candidates' recency bonus on its "
            "simulated clock",
            "`gwt.recency_clock = v04_wall` restores the v0.4.0 wall-clock read",
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


def test_binding_bullet_states_that_vectors_of_different_lengths_are_never_compared():
    bullet = _bullet(11, "Bayesian binding quality is a correlation proxy")
    for phrase in (
        "Two vectors of different lengths are never compared",
        "a three-entry content vector and a longer embedding bind only with "
        "vectors of their own length",
        "a percept with no comparable pair has no measured binding quality",
        "`blt.binding_evidence = v04_salience_posterior` truncates both vectors "
        "to their common prefix, as v0.4.0 did",
    ):
        assert phrase in bullet, phrase


def test_strange_loop_bullet_states_how_loose_its_threshold_is():
    """The agreement threshold is the tracker's default, and the bullet says
    what it admits when the history minimums are first met."""
    import inspect

    from mtc.consciousness.beautiful_loop.epistemic_depth import (
        EpistemicDepthTracker,
    )

    default = (
        inspect.signature(EpistemicDepthTracker)
        .parameters["strange_loop_threshold"]
        .default
    )
    bullet = _bullet(11, "The strange-loop condition is agreement")
    for phrase in (
        f"Its agreement threshold of {default} on two quantities in [0, 1] is "
        "loose by design",
        "moves a fifth of the way toward each scored batch of self-predictions",
        "five meta predictions, three scored batches",
        "it lies between 0.256 and 0.744 and any ledger accuracy between 0.244 "
        "and 0.756 agrees with it whatever it is",
        "the hyper-model having compared a precision prediction with the update "
        "that followed",
    ):
        assert phrase in bullet, phrase


def test_seeding_bullet_states_that_the_defaults_read_no_wall_clock():
    """The recency bonus runs on the simulated clock by default, so the
    seeding bullet no longer records a default evidence value that follows
    the clock."""
    bullet = _bullet(9, "`mtc assess` seeds random, numpy and torch")
    assert "moves 1.2e-9 at a clock 25 times" not in bullet
    for phrase in (
        "Under the defaults the seeded assessment reads no wall clock into a "
        "score or an evidence value",
        "the candidate recency bonus runs on the workspace's simulated clock "
        "(`gwt.recency_clock = simulated`)",
    ):
        assert phrase in bullet, phrase


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


NOISE_SATURATED = re.compile(
    r"In the experiment runner (\w+) indicators score 0\.0 as noise-saturated "
    r"\(([^)]*)\)"
)


def _listed_names(listing: str) -> list:
    """The names of a prose list such as ``a, b and c``."""
    return [n.strip() for n in re.split(r",\s*|\s+and\s+", listing) if n.strip()]


def _migration_noise_table() -> list:
    """The indicators the migration record's noise-saturation table flags as
    noise-saturated (the floor-saturated row excluded)."""
    text = (DOCS / "MIGRATION_FROM_V04.md").read_text(encoding="utf-8")
    match = re.search(
        r"^### Noise saturation under the defaults[^\n]*\n(.*?)(?=^#{2,3} )",
        text,
        re.M | re.S,
    )
    assert match, "MIGRATION_FROM_V04.md has no noise-saturation section"
    names = []
    for line in match.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or not re.fullmatch(r"[a-z_]+", cells[0]):
            continue
        names.append(cells[0])
    return names


def test_noise_saturated_list_matches_its_count_and_the_migration_record():
    from mtc.assessment.assessment import ConsciousnessAssessment

    match = NOISE_SATURATED.search(_section(6))
    assert match, "section 6 does not list the noise-saturated indicators"
    count_word, listing = match.groups()
    names = _listed_names(listing)
    words = ("one", "two", "three", "four", "five", "six", "seven", "eight")
    assert count_word in words, count_word
    assert words.index(count_word) + 1 == len(names), (count_word, names)
    known = set(ConsciousnessAssessment().indicator_configs)
    assert [n for n in names if n not in known] == []
    assert sorted(names) == sorted(_migration_noise_table())
