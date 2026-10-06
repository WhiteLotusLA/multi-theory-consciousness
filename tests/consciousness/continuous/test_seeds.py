"""Seed packs: loading, merging, neutrality and honest errors."""

import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mtc.consciousness.continuous.seeds import (
    ALL_KEYS,
    CONTEXT_SLOTS,
    LIST_KEYS,
    SLOT_KEYS,
    TEMPLATE_KEYS,
    SeedError,
    SeedLibrary,
    template_fields,
)
from mtc.core.config import Settings
from mtc.devtools.privacy_audit import load_allowlist, load_denylist, run_audit

ROOT = Path(__file__).resolve().parents[3]
SHIPPED = ROOT / "config" / "ctm"
DEFAULT_PACK = SHIPPED / "seeds" / "default.yaml"

# Capitalized words allowed anywhere in a line: the first-person pronoun.
CAPITAL_ALLOWLIST = {"I"}


def _write(tmp_path, name, text, folder="seeds.d"):
    path = tmp_path / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _base(tmp_path):
    """A config dir holding a copy of the default pack."""
    _write(tmp_path, "default.yaml", DEFAULT_PACK.read_text("utf-8"), folder="seeds")
    return tmp_path


def _settings(tmp_path) -> Settings:
    return Settings(PROJECT_ROOT=str(tmp_path))


def test_default_pack_loads_and_validates(tmp_path):
    lib = SeedLibrary.from_config_dir(SHIPPED, settings=_settings(tmp_path))
    assert lib.source == "file" and lib.packs == [DEFAULT_PACK]
    assert set(lib.keys()) == set(LIST_KEYS)
    for key in LIST_KEYS:
        assert len(lib.entries(key)) >= 6, key
    assert len(lib.entries("concept_pairs")) >= 10
    for key in SLOT_KEYS:
        assert key in lib.keys()
    # The four dream families and the plain query terms are present.
    for key in (
        "dream_consolidation_templates",
        "dream_creative_templates",
        "dream_emotion_templates",
        "dream_spontaneous_templates",
        "creative_concepts",
        "memory_query_terms",
    ):
        assert lib.entries(key)
    # Every shipped template fills with the full context available.
    rng = random.Random(1)
    context = {name: f"<{name}>" for name in CONTEXT_SLOTS}
    for key in TEMPLATE_KEYS:
        for template in lib.entries(key):
            text = lib.fill(template, rng, **context)
            assert "{" not in text and text


def test_unknown_slot_is_error(tmp_path):
    base = _base(tmp_path)
    path = _write(base, "bad.yaml", 'questions:\n  - "What about {nonesuch} today?"\n')
    with pytest.raises(SeedError) as err:
        SeedLibrary.from_config_dir(base, settings=_settings(tmp_path))
    message = str(err.value)
    assert str(path) in message and "questions" in message and "nonesuch" in message
    # A bad default pack is rejected the same way.
    broken = tmp_path / "two"
    _write(broken, "default.yaml", 'memory_templates:\n  - "{what}"\n', folder="seeds")
    with pytest.raises(SeedError, match="what"):
        SeedLibrary.from_config_dir(broken, settings=_settings(tmp_path))


def test_invalid_packs_raise_naming_the_file(tmp_path):
    for text in (
        "bogus_key: [a]\n",
        "questions: not a list\n",
        "questions: [3]\n",
        "concept_pairs: [[one]]\n",
        "reflection: ['a {brace}']\n",
        "- a list\n",
        "questions: [unclosed\n",
        "description: [1]\n",
        'questions:\n  - "{memory:>10}"\n',
        'questions:\n  - "{}"\n',
    ):
        folder = tmp_path / str(abs(hash(text)))
        path = _write(folder, "bad.yaml", text)
        with pytest.raises(SeedError) as err:
            SeedLibrary.from_config_dir(folder, settings=_settings(tmp_path))
        assert str(path) in str(err.value), text


def test_user_pack_extends_and_overrides(tmp_path):
    base = _base(tmp_path)
    default = SeedLibrary.from_config_dir(base, settings=_settings(tmp_path))
    _write(
        base,
        "10-first.yaml",
        "description: first\n"
        'questions:\n  - "What is the {descriptor} part really for?"\n'
        "concept_pairs:\n  - [sea, shore]\n"
        "feeling:\n  - brisk\n",
    )
    _write(
        base,
        "20-second.yaml",
        "description: second\n"
        "questions:\n"
        '  - "What is the {descriptor} part really for?"\n'
        '  - "A new open question?"\n',
    )
    lib = SeedLibrary.from_config_dir(base, settings=_settings(tmp_path))
    assert lib.description == "second"  # scalars override, in name order
    questions = lib.entries("questions")
    assert questions[: len(default.entries("questions"))] == default.entries(
        "questions"
    )
    # no duplicate
    assert questions.count("What is the {descriptor} part really for?") == 1
    assert questions[-1] == "A new open question?"
    assert ["sea", "shore"] in lib.entries("concept_pairs")
    assert lib.entries("feeling")[-1] == "brisk"
    assert [p.name for p in lib.packs] == [
        "default.yaml",
        "10-first.yaml",
        "20-second.yaml",
    ]
    # The user pack changed nothing in the shipped file's library object.
    assert "brisk" not in default.entries("feeling")


def test_default_pack_is_neutral():
    import yaml

    data = yaml.safe_load(DEFAULT_PACK.read_text("utf-8"))
    assert set(data) <= set(ALL_KEYS)
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            for item in value:
                lines.extend(item if isinstance(item, list) else [item])
        else:
            lines.append(value)
    assert len(lines) > 100
    sentence_start = re.compile(r"(^|[.?!]\s+)([A-Z][A-Za-z']*)")
    for line in lines:
        # Strip slot names, which are lowercase by construction.
        text = re.sub(r"\{[a-z_]+\}", "slot", line)
        allowed_positions = {m.start(2) for m in sentence_start.finditer(text)}
        for match in re.finditer(r"\b[A-Z][A-Za-z']*", text):
            word = match.group(0)
            ok = match.start() in allowed_positions or word in CAPITAL_ALLOWLIST
            assert ok, f"capitalized word {word!r} in {line!r}"
        assert all(ord(ch) < 128 for ch in line), line
    hits = run_audit(
        [DEFAULT_PACK],
        load_denylist(ROOT / "config" / "privacy" / "denylist.txt"),
        load_allowlist(ROOT / "config" / "privacy" / "allowlist.txt"),
        root=ROOT,
    )
    assert hits == []


def test_choose_is_seeded(tmp_path):
    lib = SeedLibrary.from_config_dir(SHIPPED, settings=_settings(tmp_path))

    def draw(seed):
        rng = random.Random(seed)
        return [
            (
                lib.choose("questions", rng),
                lib.choose("concept_pairs", rng),
                lib.fill(lib.choose("spontaneous_lines", rng), rng),
            )
            for _ in range(20)
        ]

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)
    pair = lib.choose("concept_pairs", random.Random(0))
    assert isinstance(pair, tuple) and len(pair) == 2


def test_fill_names_a_missing_context_slot_and_invents_nothing(tmp_path):
    lib = SeedLibrary.from_config_dir(SHIPPED, settings=_settings(tmp_path))
    rng = random.Random(0)
    with pytest.raises(SeedError, match="memory"):
        lib.fill("Thinking back to {memory}.", rng)
    assert lib.fill("thinking back to {memory}.", rng, memory="a walk") == (
        "Thinking back to a walk."
    )
    assert lib.fill("A {descriptor} moment.", rng).startswith("A ")
    with pytest.raises(SeedError):
        lib.fill("{other_slot}", rng)


def test_choose_only_templates_that_can_be_filled(tmp_path):
    lib = SeedLibrary.from_config_dir(SHIPPED, settings=_settings(tmp_path))
    rng = random.Random(3)
    for _ in range(40):
        template = lib.choose("dream_consolidation_templates", rng, available=[])
        assert "{memory}" not in template  # never offered when it cannot be filled
        template = lib.choose("memory_templates", rng, available=["memory"])
        assert "{memory}" in template
    with pytest.raises(SeedError, match="memory_templates"):
        lib.choose("memory_templates", rng, available=[])
    # every shipped emotion template names the measured emotion
    with pytest.raises(SeedError, match="emotion_templates"):
        lib.choose("emotion_templates", rng, available=[])
    with pytest.raises(SeedError):
        lib.choose("not_a_key", rng)


def test_topic_is_not_a_slot_no_strategy_supplies_it(tmp_path):
    assert "topic" not in CONTEXT_SLOTS
    assert "{topic}" not in DEFAULT_PACK.read_text("utf-8")
    base = _base(tmp_path)
    path = _write(base, "topic.yaml", 'questions:\n  - "What about {topic}?"\n')
    with pytest.raises(SeedError) as err:
        SeedLibrary.from_config_dir(base, settings=_settings(tmp_path))
    message = str(err.value)
    assert str(path) in message and "questions" in message and "topic" in message


def test_template_fields():
    assert template_fields("a {first} and {second} and {first}") == ["first", "second"]
    assert template_fields("no slots, {{escaped}}") == []
    with pytest.raises(SeedError):
        template_fields("{broken")


def test_missing_default_pack_leaves_an_empty_library(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        lib = SeedLibrary.from_config_dir(tmp_path, settings=_settings(tmp_path))
    assert lib.source == "missing" and lib.keys() == []
    with pytest.raises(SeedError):
        lib.choose("questions", random.Random(0))
    assert any("default seed pack" in r.getMessage() for r in caplog.records)


def test_config_dir_comes_from_settings_at_call_time(tmp_path):
    first = _base(tmp_path / "a")
    _write(first, "x.yaml", 'description: "from a"\n')
    cfg = Settings(PROJECT_ROOT=str(tmp_path), CTM_CONFIG_DIR="a")
    assert SeedLibrary.from_config_dir(settings=cfg).description == "from a"
    cfg.CTM_CONFIG_DIR = "elsewhere"
    assert SeedLibrary.from_config_dir(settings=cfg).source == "missing"


def test_yaml_imported_lazily():
    code = (
        "import sys\n"
        "import mtc.consciousness.continuous.seeds\n"
        "assert 'yaml' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stderr
