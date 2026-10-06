"""The onboarding schema: persona, speakers and the wizard's answers."""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mtc.core.config import COMPANION_TYPES, Settings
from mtc.core.persona import PERSONA_KEYS, load_persona, unknown_persona_keys
from mtc.onboarding import schema
from mtc.onboarding.schema import (
    PERSONALITY_PRESETS,
    CompanionType,
    CreatorConfig,
    OnboardingAnswers,
    PersonaAnswers,
    PersonaConfig,
    PersonalityConfig,
    Pronouns,
    SpeakerConfig,
    SpeakersFile,
    SpeakerTemplates,
    StoreChoice,
    StoresAnswers,
    default_address_form,
    export_json_schemas,
    load_type_defaults,
)
from mtc.pipeline.speaker_trust import SpeakerTrust

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _persona(**overrides) -> PersonaConfig:
    values = dict(
        name="Tessera",
        pronouns=Pronouns(),
        companion_type=CompanionType.FRIEND,
        creator=CreatorConfig(name="Alice", address_form="Alice"),
        self_description="{system_name} is an AI system: plain, and honest.",
        communication_style="Warm, plain and unhurried.",
        core_values=["curiosity", "honesty"],
        personality=PERSONALITY_PRESETS["balanced"],
        style_rules=["No flattery.", "When something is not remembered, say so."],
        speaker_templates=SpeakerTemplates(
            creator="You are talking with {creator_name}.",
            known="You are talking with {speaker}.",
            unverified="Someone gives the name {speaker}.",
        ),
    )
    values.update(overrides)
    return PersonaConfig(**values)


def _answers(**overrides) -> dict:
    data = {
        "acknowledged": True,
        "persona": {
            "name": "Tessera",
            "pronouns": {"subject": "they", "object": "them", "possessive": "their"},
            "companion_type": "friend",
            "creator_name": "Alice",
        },
        "locale": {},
        "stores": {},
        "brain": {},
        "api": {},
    }
    data.update(overrides)
    return data


def test_companion_types_equal_config_constant():
    assert tuple(member.value for member in CompanionType) == COMPANION_TYPES
    assert set(load_type_defaults()) == set(CompanionType)


def test_persona_fields_equal_persona_keys():
    assert set(PersonaConfig.model_fields) == set(PERSONA_KEYS)
    assert len(PersonaConfig.model_fields) == len(PERSONA_KEYS)
    nested = {
        "creator": CreatorConfig,
        "personality": PersonalityConfig,
        "core_memory": schema.CoreMemoryOverrides,
        "speaker_templates": SpeakerTemplates,
    }
    for key, sub_keys in PERSONA_KEYS.items():
        if sub_keys is None:
            assert key not in nested, key
            continue
        assert tuple(nested[key].model_fields) == tuple(sub_keys), key
    assert tuple(Pronouns.model_fields) == ("subject", "object", "possessive")
    assert PersonaConfig.model_fields["core_memory"].is_required() is False


def test_persona_round_trip_yaml():
    persona = _persona(
        core_memory=schema.CoreMemoryOverrides(identity="{system_name} is here.")
    )
    text = persona.to_yaml()
    assert PersonaConfig.from_yaml(text) == persona
    loaded = yaml.safe_load(text)
    assert set(loaded) == set(PERSONA_KEYS)
    assert list(loaded) == list(PersonaConfig.model_fields)
    assert loaded["companion_type"] == "friend"
    assert loaded["core_memory"] == {"identity": "{system_name} is here."}
    # A persona without overrides writes no core_memory key at all.
    assert "core_memory" not in yaml.safe_load(_persona().to_yaml())
    with pytest.raises(ValueError):
        PersonaConfig.from_yaml("- not\n- a mapping\n")
    with pytest.raises(ValidationError):
        PersonaConfig.from_yaml(text + "favourite_colour: blue\n")
    # The shipped example is a valid persona document.
    example = (ROOT / "config" / "persona.example.yaml").read_text(encoding="utf-8")
    parsed = PersonaConfig.from_yaml(example)
    assert parsed.name == "" and parsed.creator.name == ""
    assert parsed.companion_type is CompanionType.RESEARCH_SUBJECT


def test_written_persona_loads_through_load_persona_without_unknown_keys(clean_env):
    target = clean_env / "config" / "persona.yaml"
    target.parent.mkdir(parents=True)
    persona = _persona()
    target.write_text(persona.to_yaml(), encoding="utf-8")
    loaded = load_persona(settings=Settings())
    assert loaded is not None
    assert unknown_persona_keys(loaded) == []
    assert loaded["name"] == "Tessera"
    assert loaded["creator"] == {"name": "Alice", "address_form": "Alice"}
    assert loaded["pronouns"] == {
        "subject": "they",
        "object": "them",
        "possessive": "their",
    }
    assert loaded["personality"] == PERSONALITY_PRESETS["balanced"].model_dump()
    assert loaded["style_rules"] == persona.style_rules


@pytest.mark.parametrize(
    "bad",
    [
        "Tes,sera",
        "Tes:sera",
        "Tes\nsera",
        "Tes$sera",
        'Tes"sera',
        "Tes'sera",
        "Tes\\sera",
        "Tes`sera",
        "Tes{sera}",
        "Tes\tsera",
        "",
        "   ",
        "x" * 201,
    ],
)
def test_name_validation_rejects_separators_and_unsafe_chars(bad):
    with pytest.raises(ValidationError):
        PersonaAnswers(
            name=bad, pronouns=Pronouns(), companion_type="friend", creator_name="Alice"
        )
    with pytest.raises(ValidationError):
        PersonaAnswers(
            name="Tessera",
            pronouns=Pronouns(),
            companion_type="friend",
            creator_name=bad,
        )
    with pytest.raises(ValidationError):
        SpeakerConfig(name=bad, role="guest", trust=0.5)
    if bad.strip():
        with pytest.raises(ValidationError):
            PersonaAnswers(
                name="Tessera",
                pronouns=Pronouns(),
                companion_type="friend",
                creator_name="Alice",
                address_form=bad,
            )


def test_names_accept_ordinary_text_up_to_the_limit():
    longest = "x" * 200
    answers = PersonaAnswers(
        name=f"  {longest} ",
        pronouns=Pronouns(),
        companion_type="mentor",
        creator_name="Alice Example-Name",
        address_form="Ms Example",
    )
    assert answers.name == longest
    assert answers.creator_name == "Alice Example-Name"
    for word in ("the y", "they!", "", "x" * 21):
        with pytest.raises(ValidationError):
            Pronouns(subject=word)


def test_speaker_round_trip_and_trust_bounds(clean_env):
    for trust in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValidationError):
            SpeakerConfig(name="Alice", role="creator", trust=trust)
    with pytest.raises(ValidationError):
        SpeakerConfig(name="Alice", role="admin", trust=0.5)
    with pytest.raises(ValidationError):
        SpeakerConfig(name="Alice", role="guest", trust=0.5, token_env="NOT A NAME")
    with pytest.raises(ValidationError):
        SpeakersFile(
            speakers=[
                SpeakerConfig(name="Alice", role="creator", trust=1.0),
                SpeakerConfig(name="Alice", role="guest", trust=0.2),
            ]
        )
    document = SpeakersFile(
        speakers=[
            SpeakerConfig(name="Alice", role="creator", trust=0.9),
            SpeakerConfig(
                name="Guest", role="guest", trust=0.25, token_env="GUEST_TOKEN"
            ),
        ]
    )
    text = document.to_yaml()
    assert SpeakersFile.from_yaml(text) == document
    assert "token_env" not in yaml.safe_load(text)["speakers"][0]
    target = clean_env / "config" / "speakers.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(text, encoding="utf-8")
    status = SpeakerTrust.from_settings(Settings()).status()
    assert status["entries"] == 2
    assert status["by_role"] == {"creator": 1, "companion": 0, "guest": 1}
    assert status["with_token_env"] == 1


def test_answers_require_acknowledgement():
    assert OnboardingAnswers.model_validate(_answers()).acknowledged is True
    for value in (False, None):
        with pytest.raises(ValidationError) as caught:
            OnboardingAnswers.model_validate(_answers(acknowledged=value))
        assert "acknowledged" in str(caught.value)
    missing = _answers()
    del missing["acknowledged"]
    with pytest.raises(ValidationError):
        OnboardingAnswers.model_validate(missing)
    with pytest.raises(ValidationError):
        OnboardingAnswers.model_validate(_answers(schema_version=2))
    with pytest.raises(ValidationError):
        OnboardingAnswers.model_validate(_answers(surprise=True))


def test_db_prefix_pattern():
    assert schema.DB_PREFIX_PATTERN == r"^[a-z][a-z0-9_]{0,40}$"
    for good in ("mtc", "a", "mtc_test_0a1b2c3d", "x" * 41):
        assert StoresAnswers(db_prefix=good).db_prefix == good
    for bad in ("", "Mtc", "1mtc", "mtc-prod", "mtc prod", "_mtc", "x" * 42, "m$"):
        with pytest.raises(ValidationError):
            StoresAnswers(db_prefix=bad)


def test_answers_secrets_are_write_only_in_schema():
    document = export_json_schemas()["onboarding-answers.schema.json"]
    definitions = document["$defs"]

    def secret_shape(model: str, field: str) -> dict:
        options = definitions[model]["properties"][field]["anyOf"]
        return next(option for option in options if option.get("type") == "string")

    for model, field in (
        ("StoreChoice", "password"),
        ("BrainAnswers", "anthropic_api_key"),
    ):
        shape = secret_shape(model, field)
        assert shape["format"] == "password", (model, field)
        assert shape["writeOnly"] is True, (model, field)
    # A secret never renders: not in a repr, not in a dump.
    choice = StoreChoice(enabled=True, password="correct-horse")
    assert "correct-horse" not in repr(choice)
    assert "correct-horse" not in choice.model_dump_json()
    for unsafe in (
        "has space",
        "dollar$",
        'quote"',
        "back\\slash",
        "tick`",
        "a\nb",
        "",
    ):
        with pytest.raises(ValidationError) as caught:
            StoreChoice(enabled=True, password=unsafe)
        if unsafe.strip():
            assert unsafe not in str(caught.value)
    with pytest.raises(ValidationError):
        StoreChoice(enabled=True, password="one", password_env="TWO")


def test_personality_presets_in_range():
    assert list(PERSONALITY_PRESETS) == [
        "balanced",
        "curious",
        "calm",
        "playful",
        "analytical",
    ]
    expected = {
        "balanced": (0.6, 0.6, 0.5, 0.6, 0.4),
        "curious": (0.85, 0.55, 0.55, 0.6, 0.35),
        "calm": (0.55, 0.7, 0.4, 0.7, 0.2),
        "playful": (0.75, 0.45, 0.75, 0.65, 0.35),
        "analytical": (0.7, 0.8, 0.35, 0.5, 0.3),
    }
    for name, preset in PERSONALITY_PRESETS.items():
        values = tuple(preset.model_dump().values())
        assert values == expected[name], name
        assert all(0.0 <= value <= 1.0 for value in values)
    for bad in (-0.01, 1.01, float("inf"), True):
        with pytest.raises(ValidationError):
            PersonalityConfig(
                openness=bad,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
            )
    with pytest.raises(ValidationError):
        PersonaAnswers(
            name="Tessera",
            pronouns=Pronouns(),
            companion_type="friend",
            creator_name="Alice",
            personality_preset="grumpy",
        )
    with pytest.raises(ValidationError):
        PersonaAnswers(
            name="Tessera",
            pronouns=Pronouns(),
            companion_type="friend",
            creator_name="Alice",
            personality_preset=None,
        )


@pytest.mark.parametrize(
    "companion_type, expected",
    [
        ("child", "Alice Example"),
        ("sibling", "Alice"),
        ("friend", "Alice"),
        ("partner", "Alice"),
        ("mentor", "Alice Example"),
        ("research_subject", "operator"),
    ],
)
def test_default_address_form_per_type(companion_type, expected):
    assert default_address_form(companion_type, "Alice Example") == expected
    assert default_address_form(CompanionType(companion_type), " Alice Example ") == (
        expected
    )


def test_wake_and_sleep_hours_differ():
    locale = schema.LocaleAnswers()
    assert (locale.wake_hour, locale.sleep_hour) == (6, 22)
    with pytest.raises(ValidationError) as caught:
        schema.LocaleAnswers(wake_hour=7, sleep_hour=7)
    assert "differ" in str(caught.value)
    for bad in (-1, 24):
        with pytest.raises(ValidationError):
            schema.LocaleAnswers(wake_hour=bad)
        with pytest.raises(ValidationError):
            schema.LocaleAnswers(sleep_hour=bad)


def test_timezone_validated_with_zoneinfo():
    for zone in ("UTC", "Europe/Berlin", "Asia/Tokyo"):
        assert schema.LocaleAnswers(timezone=zone).timezone == zone
    for bad in ("Mars/Olympus", "", "Europe/", "../etc/passwd", "UTC+25"):
        with pytest.raises(ValidationError):
            schema.LocaleAnswers(timezone=bad)


def test_brain_and_api_answers_bounds():
    brain = schema.BrainAnswers()
    assert (brain.provider, brain.template_family) == ("mock", "generic")
    with pytest.raises(ValidationError):
        schema.BrainAnswers(provider="remote")
    with pytest.raises(ValidationError) as caught:
        schema.BrainAnswers(provider="anthropic")
    assert "anthropic_model" in str(caught.value)
    with pytest.raises(ValidationError):
        schema.BrainAnswers(port=0)
    with pytest.raises(ValidationError):
        schema.BrainAnswers(host="bad host")
    with pytest.raises(ValidationError):
        schema.BrainAnswers(base_url="not a url")
    with pytest.raises(ValidationError):
        schema.ApiAnswers(port=70000)
    assert schema.ApiAnswers(port=8765).port == 8765
    assert StoresAnswers().service_manager == "none"
    with pytest.raises(ValidationError):
        StoresAnswers(service_manager="launchd")
    stores = StoresAnswers()
    for store in schema.STORE_NAMES:
        assert getattr(stores, store).enabled is False
