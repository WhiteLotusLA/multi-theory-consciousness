"""Prompt assembler: section order, stable prefix, persona resolution, speaker blocks."""

import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mtc.brain import prompt as prompt_module
from mtc.brain.prompt import (
    RESERVED_SECTIONS,
    SECTION_ORDER,
    STABLE_PREFIX_SECTIONS,
    AssembledPrompt,
    PromptAssembler,
    PromptCaps,
    PromptInputs,
    SpeakerView,
)
from mtc.core import persona as persona_module
from mtc.core.config import Settings

T0 = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)

EXPECTED_ORDER = (
    "persona",
    "core_memory",
    "speaker",
    "style",
    "thinking",
    "capabilities",
    "ctm_thoughts",
    "thought_gists",
    "neural_state",
    "attention_guidance",
    "metacognitive_insights",
    "consciousness_narrative",
    "active_inference",
    "global_workspace",
    "memories",
    "concepts",
    "coherence",
    "summary",
    "directives",
    "self_awareness",
    "clock",
    "window_notes",
)

CAPABILITY_WORDS = (
    "brows",
    "file",
    "code",
    "tool",
    "voice",
    "messaging",
    "camera",
    "image",
)


@pytest.fixture(autouse=True)
def _fresh_warning_memory(monkeypatch):
    monkeypatch.setattr(persona_module, "_WARNED_MISSING", set())


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        PROJECT_ROOT=str(tmp_path),
        SYSTEM_NAME="Testagent",
        CREATOR_NAME="Operator",
        AGENT_SPACE_DIR=str(tmp_path / "agent_space"),
        IDENTITY_STATEMENT_PATH=str(tmp_path / "agent_space" / "absent.md"),
    )
    values.update(overrides)
    return Settings(**values)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def _assembler(tmp_path: Path, **kwargs) -> PromptAssembler:
    settings = kwargs.pop("settings", None) or _settings(tmp_path)
    kwargs.setdefault("identity_statement", "")
    return PromptAssembler(settings=settings, **kwargs)


def _known(name: str = "Alice") -> SpeakerView:
    return SpeakerView(name=name, role="friend", authenticated=True, label="verified")


def _section_names(rendered: AssembledPrompt):
    return [name for name, _chars in rendered.sections]


def test_section_order_matches_spec():
    assert SECTION_ORDER == EXPECTED_ORDER
    assert STABLE_PREFIX_SECTIONS == EXPECTED_ORDER[:6]
    assert "perception" in RESERVED_SECTIONS
    assert "perception" not in SECTION_ORDER
    assert len(set(SECTION_ORDER)) == len(SECTION_ORDER)


def test_stable_prefix_identical_across_dynamic_changes(tmp_path):
    assembler = _assembler(tmp_path)
    speaker = _known()
    core = {"identity": "Testagent is an AI research system."}
    first = assembler.render(
        PromptInputs(
            speaker=speaker,
            core_memory=core,
            ctm_thoughts=[{"content": "a quiet thought", "timestamp": T0}],
            memories=[{"content": "the first memory", "timestamp": T0}],
            now=T0,
        )
    )
    second = assembler.render(
        PromptInputs(
            speaker=speaker,
            core_memory=core,
            ctm_thoughts=[{"content": "another thought entirely", "timestamp": T0}],
            memories=[{"content": "a later memory", "timestamp": T0}],
            concepts=["tides"],
            now=T0 + timedelta(hours=5),
        )
    )
    assert first.stable_prefix_chars == second.stable_prefix_chars > 0
    prefix = first.text[: first.stable_prefix_chars]
    assert prefix == second.text[: second.stable_prefix_chars]
    assert first.text != second.text
    prefix_names = [n for n in _section_names(first) if n in STABLE_PREFIX_SECTIONS]
    assert prefix_names == ["persona", "core_memory", "speaker", "style"]
    # The suffix follows SECTION_ORDER.
    names = _section_names(second)
    assert names == sorted(names, key=SECTION_ORDER.index)
    assert sum(chars for _n, chars in first.sections) <= len(first.text)


def test_capability_sections_absent_unless_registered(tmp_path):
    assembler = _assembler(tmp_path)
    rendered = assembler.render(PromptInputs(speaker=_known(), now=T0))
    lowered = rendered.text.lower()
    for word in CAPABILITY_WORDS:
        assert word not in lowered, word
    assert "capabilities" not in _section_names(rendered)
    assembler.register_capability("echo", "You can echo a line back when asked.")
    with_cap = assembler.render(PromptInputs(speaker=_known(), now=T0))
    assert "capabilities" in _section_names(with_cap)
    assert "echo a line back" in with_cap.text
    # The capability joins the stable prefix.
    assert with_cap.text.index("echo a line back") < with_cap.stable_prefix_chars


def test_perception_section_never_registered(tmp_path):
    assembler = _assembler(tmp_path)
    with pytest.raises(ValueError, match="vision"):
        assembler.register_capability("perception", "You can see images.")
    with pytest.raises(ValueError, match="memories"):
        assembler.register_capability("memories", "a section name")
    with pytest.raises(ValueError):
        assembler.register_capability("", "empty name")
    rendered = assembler.render(PromptInputs(now=T0))
    assert "perception" not in _section_names(rendered)
    assert "image" not in rendered.text.lower()


def test_persona_resolution_order(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    # Configured persona wins.
    _write(
        tmp_path / "config" / "persona.yaml",
        """
        name: Configured Unit
        style_rules:
          - Answer directly.
        """,
    )
    configured = PromptAssembler(settings=settings, identity_statement="")
    assert configured.persona_source == "configured"
    assert "Configured Unit" in configured.render(PromptInputs(now=T0)).text
    # Without it, the shipped example applies.
    (tmp_path / "config" / "persona.yaml").unlink()
    example = PromptAssembler(settings=settings, identity_statement="")
    assert example.persona_source == "example"
    assert example.persona["speaker_templates"]["unverified"]
    # Without the example, the built-in neutral text applies.
    monkeypatch.setattr(
        prompt_module, "PERSONA_EXAMPLE_PATH", tmp_path / "missing.example.yaml"
    )
    builtin = PromptAssembler(settings=settings, identity_statement="")
    assert builtin.persona_source == "builtin"
    text = builtin.render(PromptInputs(speaker=_known(), now=T0)).text
    assert "Testagent" in text
    assert "AI" in text
    # An explicit mapping is configured by definition.
    explicit = PromptAssembler(
        persona={"name": "Explicit"}, settings=settings, identity_statement=""
    )
    assert explicit.persona_source == "configured"


def test_persona_placeholders_rendered(tmp_path):
    persona = {
        "name": "",
        "pronouns": {"subject": "she", "object": "her", "possessive": "her"},
        "creator": {"name": "", "address_form": "Boss"},
        "self_description": "{system_name} is an AI system; {pronoun_subject} says so.",
        "communication_style": "Plain and {pronoun_possessive} own.",
        "style_rules": ["Answer directly."],
        "speaker_templates": {
            "creator": "Talking with {creator_name}, addressed as {creator_address}.",
            "known": "Talking with {speaker}, verified.",
            "unverified": "Talking with someone called {speaker}, unverified.",
        },
    }
    assembler = _assembler(tmp_path, persona=persona)
    creator = SpeakerView(
        name="Operator", role="creator", authenticated=True, label="creator"
    )
    text = assembler.render(PromptInputs(speaker=creator, now=T0)).text
    assert "Testagent is an AI system; she says so." in text
    assert "Plain and her own." in text
    assert "Talking with Operator, addressed as Boss." in text
    assert "{" not in text and "}" not in text
    bad = dict(persona, self_description="Hello {nickname}.")
    with pytest.raises(ValueError, match="nickname"):
        _assembler(tmp_path, persona=bad).render(PromptInputs(now=T0))


def test_identity_statement_rendered_when_present(tmp_path):
    statement = tmp_path / "agent_space" / "identity_statement.md"
    _write(
        statement,
        """
        # Identity statement

        You are {system_name}; {creator_name} set you up for research.
        """,
    )
    settings = _settings(tmp_path, IDENTITY_STATEMENT_PATH=str(statement))
    assembler = PromptAssembler.from_settings(settings)
    text = assembler.render(PromptInputs(now=T0)).text
    assert "You are Testagent; Operator set you up for research." in text
    assert (
        text.index("Identity statement")
        < assembler.render(PromptInputs(now=T0)).stable_prefix_chars
    )
    without = PromptAssembler(settings=settings, identity_statement="")
    assert "set you up for research" not in without.render(PromptInputs(now=T0)).text


def test_speaker_block_for_unverified_name(tmp_path):
    settings = _settings(tmp_path)
    assembler = PromptAssembler(settings=settings, identity_statement="")
    assert assembler.persona_source == "example"
    impostor = SpeakerView(
        name=settings.CREATOR_NAME, role=None, authenticated=False, label="unverified"
    )
    rendered = assembler.render(PromptInputs(speaker=impostor, now=T0))
    assert "speaker" in _section_names(rendered)
    assert "not verified" in rendered.text
    assert "set this system up" not in rendered.text
    assert "Operator" in rendered.text  # the given name is shown, as a claim


def test_creator_template_only_for_authenticated_creator(tmp_path):
    assembler = _assembler(tmp_path)
    creator = SpeakerView(
        name="Operator", role="creator", authenticated=True, label="creator"
    )
    text = assembler.render(PromptInputs(speaker=creator, now=T0)).text
    assert "set this system up" in text
    assert "not verified" not in text
    known = SpeakerView(name="Alice", role="friend", authenticated=True, label="known")
    text = assembler.render(PromptInputs(speaker=known, now=T0)).text
    assert "verified speaker you know" in text
    assert "set this system up" not in text
    claimed = SpeakerView(
        name="Operator", role="creator", authenticated=False, label="unverified"
    )
    text = assembler.render(PromptInputs(speaker=claimed, now=T0)).text
    assert "not verified" in text
    assert "set this system up" not in text
    none = assembler.render(PromptInputs(now=T0))
    assert "speaker" not in _section_names(none)


def test_disabled_sections_ablate(tmp_path):
    assembler = _assembler(tmp_path, disabled_sections=("memories", "clock"))
    rendered = assembler.render(
        PromptInputs(
            memories=[{"content": "a memory that must not render", "timestamp": T0}],
            concepts=["tides"],
            now=T0,
        )
    )
    names = _section_names(rendered)
    assert "memories" not in names and "clock" not in names
    assert "concepts" in names
    assert "must not render" not in rendered.text
    assert "Current time" not in rendered.text
    with pytest.raises(ValueError, match="telepathy"):
        _assembler(tmp_path, disabled_sections=("telepathy",))


def test_caps_from_settings(tmp_path):
    settings = _settings(
        tmp_path,
        PROMPT_MAX_THOUGHTS=2,
        PROMPT_MAX_GISTS=1,
        PROMPT_MAX_MEMORIES=3,
        PROMPT_MAX_MEMORY_CHARS=20,
        PROMPT_MAX_CONCEPTS=1,
        PROMPT_DISABLED_SECTIONS="coherence, window_notes",
    )
    caps = PromptCaps.from_settings(settings)
    assert caps == PromptCaps(
        thoughts=2, gists=1, memories=3, memory_chars=20, concepts=1
    )
    assembler = PromptAssembler.from_settings(settings)
    assert assembler.caps == caps
    assert assembler.disabled_sections == ("coherence", "window_notes")
    rendered = assembler.render(
        PromptInputs(
            ctm_thoughts=[
                {"content": f"thought {i}", "timestamp": T0} for i in range(5)
            ],
            memories=[
                {"content": "x" * 80, "timestamp": T0},
                {"content": "y" * 10, "timestamp": T0},
                {"content": "z" * 10, "timestamp": T0},
                {"content": "w" * 10, "timestamp": T0},
            ],
            concepts=["one", "two"],
            coherence="must not render",
            now=T0,
        )
    )
    assert "thought 0" in rendered.text and "thought 1" in rendered.text
    assert "thought 2" not in rendered.text
    assert "x" * 20 in rendered.text and "x" * 21 not in rendered.text
    assert "w" * 10 not in rendered.text
    assert "one" in rendered.text and "two" not in rendered.text.split("concepts")[-1]
    assert "must not render" not in rendered.text


def test_thinking_instructions_only_for_thinking_families(tmp_path):
    thinking = _assembler(tmp_path, family="qwen", thinking_mode=True)
    names = _section_names(thinking.render(PromptInputs(now=T0)))
    assert "thinking" in names
    generic = _assembler(tmp_path, family="generic", thinking_mode=True)
    assert "thinking" not in _section_names(generic.render(PromptInputs(now=T0)))
    off = _assembler(tmp_path, family="qwen", thinking_mode=False)
    assert "thinking" not in _section_names(off.render(PromptInputs(now=T0)))
