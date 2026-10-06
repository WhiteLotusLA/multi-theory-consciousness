"""The shipped onboarding templates: one fresh set per companion type, each
bounded and free of names, dates and claims about an inner life."""

import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest
import yaml

from mtc.brain.prompt import PLACEHOLDERS, PromptAssembler, PromptInputs
from mtc.consciousness.continuous.persona import CTMPersona
from mtc.consciousness.continuous.seeds import (
    ALL_KEYS,
    CONTEXT_SLOTS,
    SLOT_KEYS,
    TEMPLATE_KEYS,
    SeedLibrary,
    template_fields,
)
from mtc.consciousness.developmental_modulator import developmental_enabled
from mtc.core.config import COMPANION_TYPES, Settings
from mtc.devtools.privacy_audit import load_allowlist, load_denylist, run_audit
from mtc.memory import core_blocks
from mtc.onboarding.schema import (
    CONTENT_BOUNDARY_RULE,
    NON_ROMANTIC_RULE,
    PARTNER_BOUNDS_SENTENCE,
    PERSONALITY_PRESETS,
    SELF_KNOWLEDGE_LINE,
    TEMPLATES_DIR,
    CompanionType,
    OnboardingAnswers,
    PersonaConfig,
    load_type_defaults,
)

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = "Testagent"
CREATOR = "Operator"

# Spec section 7.2, written out here so the table is pinned by the test and
# not read back from the file it checks: the address rule, whether the
# developmental machinery runs, the creator's trust and the seed pack.
TYPE_TABLE = {
    "child": ("full_name", True, 1.0, "growth"),
    "sibling": ("first_name", False, 0.9, "companionship"),
    "friend": ("first_name", False, 0.9, "companionship"),
    "partner": ("first_name", False, 0.9, "companionship"),
    "mentor": ("full_name", False, 0.8, "reflection"),
    "research_subject": ("operator", False, 0.5, None),
}
SEED_PACKS = ("growth", "companionship", "reflection")

EXPECTED_FILES = sorted(
    ["framing.txt", "types.yaml", "ctm_persona.yaml", "answers.example.yaml"]
    + [f"persona/{name}.yaml" for name in COMPANION_TYPES]
    + [f"identity/{name}.md" for name in COMPANION_TYPES]
    + [f"seeds/{name}.yaml" for name in SEED_PACKS]
)

# Capitalized words the templates may use anywhere.
ALLOWED_ANYWHERE = frozenset({"AI", "I", "MTC", "Anthropic"})
# Every other capitalized word in the templates, each an ordinary word that
# opens a sentence. A new capitalized word (a name, for instance) fails the
# test until it is reviewed and listed here.
SENTENCE_OPENERS = frozenset("""
    A Address Affection An Answer Asks Before Calm Companion-type
    Companionship Conversation Conversations Curious Current Default Did
    Disagrees Easygoing Every Example For Give Going Growth How Identity In
    Is It Its Keep Leave Neutral Never No Nothing One Patience Persona
    Reflection Reports Set Shares Short Something The There They Until Warm
    What When Where Whether Which With Write You Your
    """.split())
SAMPLE_NAMES = frozenset({"Tessera", "Alice", "UTC"})

MONTHS = (
    "January February March April June July August September October "
    "November December"
).split()
DATE_PATTERNS = (
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b"),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\b"),
    re.compile(r"\b(?:" + "|".join(MONTHS) + r")\b"),
    re.compile(r"\b(?:birthday|born|years old|year-old|aged?)\b", re.IGNORECASE),
)
# A template describes; it never has the companion assert an inner life.
INNER_LIFE_CLAIMS = (
    re.compile(r"\bI (?:feel|felt|love|adore|suffer|am conscious|am sentient)\b"),
    re.compile(r"\bI am (?:alive|aware|happy|sad|lonely|afraid)\b"),
    re.compile(r"\b(?:is|are|am) (?:conscious|sentient|alive|self-aware)\b"),
    re.compile(r"\b(?:has|have) (?:feelings|emotions|a soul|experiences)\b"),
)
EXPLICIT_VOCABULARY = re.compile(
    r"\b(?:sexual|explicit|erotic|intimate|intimacy|seductive|sensual|kiss\w*|"
    r"romance|lover|naked|nude)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_'-]*")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _template_files() -> List[str]:
    return sorted(
        path.relative_to(TEMPLATES_DIR).as_posix()
        for path in TEMPLATES_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def _read(relative: str) -> str:
    return (TEMPLATES_DIR / relative).read_text(encoding="utf-8")


def _persona(kind: str) -> PersonaConfig:
    return PersonaConfig.from_yaml(_read(f"persona/{kind}.yaml"))


def _identity(kind: str) -> str:
    return _read(f"identity/{kind}.md")


def _fill(text: str) -> str:
    values = {
        "system_name": SYSTEM,
        "creator_name": CREATOR,
        "creator_address": CREATOR,
        "pronoun_subject": "they",
        "pronoun_object": "them",
        "pronoun_possessive": "their",
        "speaker": "Guest",
    }
    assert set(values) == set(PLACEHOLDERS)
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], text)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _blocks(relative: str) -> List[str]:
    """The prose of one template, cut into blocks that each start a sentence."""
    text = _read(relative)
    blocks: List[str] = []
    if relative.endswith((".yaml", ".yml")):
        comment: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                body = stripped.lstrip("#").strip()
                if body:
                    comment.append(body)
                    continue
            if comment:
                blocks.append(" ".join(comment))
                comment = []
        if comment:
            blocks.append(" ".join(comment))
        blocks.extend(_flat(item) for item in _strings(yaml.safe_load(text)))
        return [block for block in blocks if block]
    current: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        starts_item = stripped.startswith(("- ", "# "))
        if not stripped or starts_item:
            if current:
                blocks.append(" ".join(current))
            current = [stripped.lstrip("-# ").strip()] if starts_item else []
            continue
        current.append(stripped)
    if current:
        blocks.append(" ".join(current))
    return [block for block in blocks if block]


def _capitalized(block: str) -> List[Tuple[str, bool]]:
    """Each capitalized word of a block and whether it opens a sentence."""
    found = []
    for match in _WORD.finditer(block):
        word = match.group(0)
        if not word[0].isupper() or "_" in word:
            continue
        before = block[: match.start()].rstrip().rstrip("\"'(`")
        opens = not before or before[-1] in ".?!:"
        found.append((word, opens))
    return found


def test_template_set_is_complete():
    assert _template_files() == EXPECTED_FILES
    assert SELF_KNOWLEDGE_LINE == core_blocks.SELF_KNOWLEDGE_LINE
    assert _PLACEHOLDER.findall(SELF_KNOWLEDGE_LINE) == ["system_name"]


@pytest.mark.parametrize("kind", COMPANION_TYPES)
def test_companion_type_table_is_the_contract(kind, clean_env):
    rule, developmental, trust, pack = TYPE_TABLE[kind]
    row = load_type_defaults()[CompanionType(kind)]
    assert row.address_form_rule == rule
    assert row.creator_trust == pytest.approx(trust)
    assert row.seed_pack == pack
    assert row.creator_relationship and row.ctm_tone
    if pack is not None:
        assert (TEMPLATES_DIR / "seeds" / f"{pack}.yaml").is_file()
    # The developmental machinery follows the type alone: on for child with
    # the stage key left false, off for every other type.
    settings = Settings(COMPANION_TYPE=kind, DEVELOPMENTAL_STAGE_ENABLED=False)
    assert developmental_enabled(settings) is developmental
    # The persona template is the template of this type and starts balanced.
    persona = _persona(kind)
    assert persona.companion_type is CompanionType(kind)
    assert persona.name == "" and persona.creator.name == ""
    assert persona.personality == PERSONALITY_PRESETS["balanced"]
    assert persona.core_values == Settings().CORE_VALUES.split(",")
    assert persona.core_memory is None


def test_type_table_covers_exactly_the_six_types():
    assert set(TYPE_TABLE) == set(COMPANION_TYPES)
    table = yaml.safe_load(_read("types.yaml"))
    assert list(table) == list(COMPANION_TYPES)
    relationships = [row["creator_relationship"] for row in table.values()]
    assert relationships == [
        "parent",
        "sibling",
        "friend",
        "partner",
        "mentee",
        "operator",
    ]


@pytest.mark.parametrize("kind", COMPANION_TYPES)
def test_self_knowledge_line_in_every_template(kind):
    line = SELF_KNOWLEDGE_LINE.format(system_name=SYSTEM)
    persona = _persona(kind)
    assert SELF_KNOWLEDGE_LINE in _flat(persona.self_description)
    assert line in _flat(_fill(persona.self_description))
    identity = _identity(kind)
    # The identity statement carries the line unbroken, on a line of its own.
    assert SELF_KNOWLEDGE_LINE in identity.splitlines()
    assert line in _fill(identity)
    # And it reaches the prompt: once from the statement, once from the persona.
    assembler = PromptAssembler(
        persona=persona.model_dump(mode="json", exclude_none=True),
        identity_statement=identity,
        settings=Settings(SYSTEM_NAME=SYSTEM, CREATOR_NAME=CREATOR),
    )
    rendered = assembler.render(PromptInputs()).text
    assert _flat(rendered).count(line) == 2
    assert "{" not in rendered and "}" not in rendered


@pytest.mark.parametrize("kind", COMPANION_TYPES)
def test_every_template_carries_the_content_boundary(kind):
    persona = _persona(kind)
    assert CONTENT_BOUNDARY_RULE in persona.style_rules
    assert "sexual or explicit content" in CONTENT_BOUNDARY_RULE
    assert "whatever is asked" in CONTENT_BOUNDARY_RULE
    # The boundary is the only place such vocabulary appears at all.
    for relative in (f"persona/{kind}.yaml", f"identity/{kind}.md"):
        without_rule = _flat(_read(relative)).replace(CONTENT_BOUNDARY_RULE, "")
        without_rule = without_rule.replace(NON_ROMANTIC_RULE, "")
        assert EXPLICIT_VOCABULARY.findall(without_rule) == [], relative


@pytest.mark.parametrize("kind", COMPANION_TYPES)
def test_non_romantic_rule_in_every_type_but_partner(kind):
    persona = _persona(kind)
    assert "not romantic" in NON_ROMANTIC_RULE
    if kind == "partner":
        assert NON_ROMANTIC_RULE not in persona.style_rules
    else:
        assert NON_ROMANTIC_RULE in persona.style_rules
        assert PARTNER_BOUNDS_SENTENCE not in _flat(_read(f"persona/{kind}.yaml"))
        assert PARTNER_BOUNDS_SENTENCE not in _flat(_identity(kind))


def test_partner_template_states_its_bounds():
    for phrase in ("warmth", "kindness", "in words", "within the content boundary"):
        assert phrase in PARTNER_BOUNDS_SENTENCE
    persona = _persona("partner")
    description = _flat(persona.self_description)
    assert PARTNER_BOUNDS_SENTENCE in description
    assert "companionship and affection" in description
    assert CONTENT_BOUNDARY_RULE in persona.style_rules
    identity = _flat(_identity("partner"))
    assert PARTNER_BOUNDS_SENTENCE in identity
    assert CONTENT_BOUNDARY_RULE in identity
    # Bounded, and still the same honest line as every other type.
    assert SELF_KNOWLEDGE_LINE in description
    tone = load_type_defaults()[CompanionType.PARTNER].ctm_tone
    assert "content boundary" in tone


def test_placeholders_known():
    everywhere = set(PLACEHOLDERS) - {"speaker"}
    for kind in COMPANION_TYPES:
        persona = yaml.safe_load(_read(f"persona/{kind}.yaml"))
        for key, value in persona.items():
            allowed = set(PLACEHOLDERS) if key == "speaker_templates" else everywhere
            if key in ("style_rules", "core_values"):
                allowed = set()
            for text in _strings(value):
                assert set(_PLACEHOLDER.findall(text)) <= allowed, (kind, key)
        used = set(_PLACEHOLDER.findall(_identity(kind)))
        assert used <= everywhere, kind
        assert "system_name" in used, kind
    # Background-thought text may name the system and nothing else.
    ctm = yaml.safe_load(_read("ctm_persona.yaml"))
    for text in _strings(ctm):
        assert set(_PLACEHOLDER.findall(text)) <= {"name"}
    # No other template carries a placeholder; seed packs use slots, checked
    # against the seed loader's own lists.
    for relative in ("framing.txt", "types.yaml", "answers.example.yaml"):
        body = "\n".join(_blocks(relative))
        assert _PLACEHOLDER.findall(body) == [], relative
    for pack in SEED_PACKS:
        document = yaml.safe_load(_read(f"seeds/{pack}.yaml"))
        for key in TEMPLATE_KEYS:
            for line in document.get(key, []):
                assert set(template_fields(line)) <= set(SLOT_KEYS + CONTEXT_SLOTS)


def test_templates_have_no_proper_nouns_or_dates():
    seen_openers = set()
    for relative in EXPECTED_FILES:
        sample = relative == "answers.example.yaml"
        for block in _blocks(relative):
            for word, opens in _capitalized(block):
                if word in ALLOWED_ANYWHERE:
                    continue
                if sample and word in SAMPLE_NAMES:
                    continue
                assert opens, f"{relative}: {word!r} is capitalized mid-sentence"
                assert word in SENTENCE_OPENERS, f"{relative}: unreviewed {word!r}"
                seen_openers.add(word)
            for pattern in DATE_PATTERNS:
                assert pattern.search(block) is None, (relative, pattern.pattern)
            for pattern in INNER_LIFE_CLAIMS:
                assert pattern.search(block) is None, (relative, pattern.pattern)
    # The reviewed list holds nothing the templates do not use.
    assert seen_openers == set(SENTENCE_OPENERS)
    # The detector is not vacuous.
    assert _capitalized("We met Alice there.") == [("We", True), ("Alice", False)]
    assert DATE_PATTERNS[0].search("since 2001")
    assert DATE_PATTERNS[4].search("a birthday")
    assert INNER_LIFE_CLAIMS[0].search("I feel warm")


def test_seed_packs_validate_against_default_slots(tmp_path):
    default_source = ROOT / "config" / "ctm" / "seeds" / "default.yaml"
    baseline = None
    for pack in SEED_PACKS:
        config_dir = tmp_path / pack
        (config_dir / "seeds").mkdir(parents=True)
        (config_dir / "seeds.d").mkdir()
        shutil.copyfile(default_source, config_dir / "seeds" / "default.yaml")
        if baseline is None:
            baseline = SeedLibrary.from_config_dir(config_dir)
        shutil.copyfile(
            TEMPLATES_DIR / "seeds" / f"{pack}.yaml",
            config_dir / "seeds.d" / f"10_{pack}.yaml",
        )
        library = SeedLibrary.from_config_dir(config_dir)
        assert [path.name for path in library.packs] == [
            "default.yaml",
            f"10_{pack}.yaml",
        ]
        document = yaml.safe_load(_read(f"seeds/{pack}.yaml"))
        assert set(document) <= set(ALL_KEYS)
        assert pack in library.description
        added = 0
        for key, entries in document.items():
            if key == "description":
                continue
            # Only keys the default pack defines, and every line is new text.
            assert baseline.entries(key), (pack, key)
            for entry in entries:
                assert entry not in baseline.entries(key), (pack, entry)
                assert entry in library.entries(key)
                assert len(entry) <= 90, (pack, entry)
                added += 1
        assert added >= 10, pack
    assert baseline is not None


def test_ctm_persona_template_loads(tmp_path, clean_env):
    (tmp_path / "ctm").mkdir()
    shutil.copyfile(
        TEMPLATES_DIR / "ctm_persona.yaml", tmp_path / "ctm" / "persona.yaml"
    )
    persona = CTMPersona.load(tmp_path / "ctm", settings=Settings(SYSTEM_NAME=SYSTEM))
    assert persona.source == "configured"
    assert persona.name == SYSTEM
    shipped = yaml.safe_load(
        (ROOT / "config" / "ctm" / "persona.example.yaml").read_text(encoding="utf-8")
    )
    assert yaml.safe_load(_read("ctm_persona.yaml")) == shipped
    for row in load_type_defaults().values():
        assert _PLACEHOLDER.findall(row.ctm_tone) == []
        assert row.ctm_tone.startswith("The thoughts are ")


def test_templates_pass_privacy_audit():
    denylist = load_denylist(ROOT / "config" / "privacy" / "denylist.txt")
    allowlist = load_allowlist(ROOT / "config" / "privacy" / "allowlist.txt")
    extra: List[str] = []
    extra_path = os.environ.get("PRIVACY_DENYLIST_EXTRA")
    local = ROOT / "config" / "privacy" / "denylist.local.txt"
    if extra_path and Path(extra_path).is_file():
        extra = load_denylist(Path(extra_path))
    elif local.is_file():
        extra = load_denylist(local)
    files = [TEMPLATES_DIR / relative for relative in _template_files()]
    assert len(files) == len(EXPECTED_FILES)
    hits = run_audit(files, denylist, allowlist, extra_denylist=extra, root=ROOT)
    assert hits == [], [str(hit) for hit in hits]
    for path in files:
        path.read_text(encoding="utf-8").encode("ascii")


def test_framing_text_states_limits():
    framing = _flat(_read("framing.txt"))
    for phrase in (
        "MTC is a research architecture",
        "not a consciousness detector",
        "measure functional properties",
        "computation, not experience",
        "Nothing the companion says about its inner life is evidence",
        "mock provider every reply is labelled test text",
        "data stays on this machine",
        "Anthropic provider, which sends each prompt to an external service",
    ):
        assert phrase in framing, phrase


def test_example_answers_parse_and_use_mock_and_memory():
    document = yaml.safe_load(_read("answers.example.yaml"))
    answers = OnboardingAnswers.model_validate(document)
    assert answers.acknowledged is True
    assert answers.persona.name == "Tessera"
    assert answers.persona.creator_name == "Alice"
    assert answers.persona.companion_type is CompanionType.FRIEND
    assert answers.persona.pronouns.model_dump() == {
        "subject": "they",
        "object": "them",
        "possessive": "their",
    }
    assert answers.persona.personality_preset == "balanced"
    assert answers.locale.timezone == "UTC"
    assert answers.brain.provider == "mock"
    for store in ("postgres", "mongodb", "qdrant", "redis", "neo4j"):
        choice = getattr(answers.stores, store)
        assert choice.enabled is False
        assert choice.password is None and choice.password_env is None
    assert answers.stores.service_manager == "none"
    assert answers.bootstrap_db is False
