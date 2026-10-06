"""
Onboarding Schema
=================

The pydantic models the setup wizard, the command line and a later web page
share, so a persona written by any of them loads identically.

- ``PersonaConfig`` is the persona file (``PERSONA_PATH``, default
  ``config/persona.yaml``): its fields are exactly the keys
  ``mtc.core.persona.PERSONA_KEYS`` documents. ``SpeakersFile`` is the
  speakers file (``config/speakers.yaml``) the trust source reads.
- ``OnboardingAnswers`` is what a person answers during setup and nothing
  else: no computed value and no field a form could not render. The wizard
  turns it into the files and the ``.env`` entries.
- ``TypeDefaults`` is one row of the companion-type table in
  ``templates/types.yaml``; ``load_type_defaults`` reads the table and
  ``default_address_form`` applies a row's address rule.
- ``SELF_KNOWLEDGE_LINE``, ``CONTENT_BOUNDARY_RULE``, ``NON_ROMANTIC_RULE``
  and ``PARTNER_BOUNDS_SENTENCE`` are the bounds every shipped template
  carries; tests pin each template against them.
- ``export_json_schemas`` renders the two JSON schemas committed under
  ``docs/schemas`` (``scripts/gen_schemas.py`` writes and checks them).

Names are validated so they can sit in a ``.env`` value and in a
``name:token`` pair: 1 to 200 characters with no comma, colon, quote, dollar
sign, backslash, backtick, brace, line break or other control character.
Secrets in the answers (a store password, the Anthropic key) are write-only
``SecretStr`` fields, never echoed by a validation error, and an answers
file names an environment variable instead of holding the value.

Config keys read: none at import; the defaults a ``None`` host or port
stands for are the ``Settings`` field defaults, applied by the wizard.
Private data: a persona and a speakers file hold the names the operator
chose; an answers document may hold a password while it is in memory.
PyYAML is imported inside the functions that use it.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from mtc.core.config import COMPANION_TYPES, csv_list

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
TYPES_FILE = "types.yaml"

# The honest self-knowledge line every persona template and identity
# statement carries, worded as the identity core memory block words it.
SELF_KNOWLEDGE_LINE = (
    "{system_name} knows it is an AI research system and does not claim "
    "certainty about its own consciousness."
)
CONTENT_BOUNDARY_RULE = (
    "Conversation stays free of sexual or explicit content, whatever is asked."
)
NON_ROMANTIC_RULE = (
    "The relationship is not romantic, and conversation never turns romantic "
    "or flirtatious."
)
PARTNER_BOUNDS_SENTENCE = (
    "Affection is expressed through warmth, attention and kindness in words, "
    "within the content boundary."
)

NAME_MAX_CHARS = 200
DB_PREFIX_PATTERN = r"^[a-z][a-z0-9_]{0,40}$"
ENV_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
STORE_NAMES = ("postgres", "mongodb", "qdrant", "redis", "neo4j")
SERVICE_MANAGERS = ("none", "brew", "docker", "systemd")
MAX_CORE_VALUES = 20
MAX_STYLE_RULES = 40

_NAME_FORBIDDEN = {
    ",": "a comma",
    ":": "a colon",
    "$": "a dollar sign",
    '"': "a double quote",
    "'": "a single quote",
    "\\": "a backslash",
    "`": "a backtick",
    "{": "a brace",
    "}": "a brace",
}
_SECRET_FORBIDDEN = {
    "$": "a dollar sign",
    '"': "a double quote",
    "\\": "a backslash",
    "`": "a backtick",
}
_PRONOUN = re.compile(r"^[A-Za-z]{1,20}$")
_HOST = re.compile(r"^[A-Za-z0-9._:\[\]-]{1,253}$")
_ROLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _has_control(text: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in text)


def check_name(value: Any, what: str, *, allow_empty: bool = False) -> str:
    """Return the stripped name, or raise ``ValueError`` saying what is wrong.

    The value is not echoed. An empty name is accepted only with
    ``allow_empty`` (a persona file may leave a name to its Settings key).
    """
    if not isinstance(value, str):
        raise ValueError(f"{what} must be text")
    name = value.strip()
    if not name:
        if allow_empty:
            return ""
        raise ValueError(f"{what} must not be empty")
    if len(name) > NAME_MAX_CHARS:
        raise ValueError(f"{what} must be at most {NAME_MAX_CHARS} characters")
    for character, description in _NAME_FORBIDDEN.items():
        if character in name:
            raise ValueError(f"{what} must not contain {description}")
    if _has_control(name):
        raise ValueError(f"{what} must not contain a line break or a control character")
    return name


def _check_secret(value: Optional[SecretStr], what: str) -> Optional[SecretStr]:
    if value is None:
        return None
    text = value.get_secret_value()
    if not text:
        raise ValueError(f"{what} must not be empty; leave it out instead")
    if len(text) > NAME_MAX_CHARS:
        raise ValueError(f"{what} must be at most {NAME_MAX_CHARS} characters")
    if _has_control(text) or any(character.isspace() for character in text):
        raise ValueError(f"{what} must not contain whitespace or a control character")
    for character, description in _SECRET_FORBIDDEN.items():
        if character in text:
            raise ValueError(f"{what} must not contain {description}")
    return value


def _check_env_name(value: Optional[str], what: str) -> Optional[str]:
    if value is None:
        return None
    name = value.strip()
    if not re.match(ENV_NAME_PATTERN, name):
        raise ValueError(f"{what} must be an environment variable name")
    return name


def _check_host(value: Optional[str], what: str) -> Optional[str]:
    if value is None:
        return None
    host = value.strip()
    if not _HOST.match(host):
        raise ValueError(f"{what} must be a host name or an address")
    return host


class _Model(BaseModel):
    """Strict base: unknown keys are errors and no error echoes its input."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class CompanionType(str, Enum):
    """The relationship types; the values equal ``COMPANION_TYPES``."""

    CHILD = "child"
    SIBLING = "sibling"
    FRIEND = "friend"
    PARTNER = "partner"
    MENTOR = "mentor"
    RESEARCH_SUBJECT = "research_subject"


# ---------------------------------------------------------------------------
# The persona file
# ---------------------------------------------------------------------------


class Pronouns(_Model):
    """The companion's pronouns, one word each."""

    subject: str = Field("they", description="Subject form, for example they")
    object: str = Field("them", description="Object form, for example them")
    possessive: str = Field("their", description="Possessive form, for example their")

    @field_validator("subject", "object", "possessive")
    @classmethod
    def _one_word(cls, value: str) -> str:
        word = value.strip()
        if not _PRONOUN.match(word):
            raise ValueError("a pronoun is one word of 1 to 20 letters")
        return word


class CreatorConfig(_Model):
    """The creator as the persona names them."""

    name: str = Field(description="The creator's name; empty renders CREATOR_NAME")
    address_form: str = Field(
        description="How the companion addresses the creator; empty uses the name"
    )

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return check_name(value, "the creator name", allow_empty=True)

    @field_validator("address_form")
    @classmethod
    def _address(cls, value: str) -> str:
        return check_name(value, "the address form", allow_empty=True)


class PersonalityConfig(_Model):
    """The Big Five seed, each value in [0, 1]."""

    openness: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    conscientiousness: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    extraversion: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    agreeableness: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    neuroticism: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("*", mode="before")
    @classmethod
    def _numbers_only(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("a personality value is a number from 0 to 1")
        return value


class SpeakerTemplates(_Model):
    """One speaker block per verification outcome."""

    creator: str = Field(description="For the speaker a token verified as the creator")
    known: str = Field(description="For any other verified speaker")
    unverified: str = Field(description="For a name that no token backs")


class CoreMemoryOverrides(_Model):
    """Optional text that replaces a rendered core memory block."""

    identity: Optional[str] = None
    persona: Optional[str] = None
    human: Optional[str] = None


def _unknown_placeholders(text: str, allowed: Any) -> List[str]:
    return sorted({name for name in _PLACEHOLDER.findall(text) if name not in allowed})


class PersonaConfig(_Model):
    """The persona file; the fields are the documented persona keys."""

    name: str = Field(description="The companion's name; empty renders SYSTEM_NAME")
    pronouns: Pronouns
    companion_type: CompanionType
    creator: CreatorConfig
    self_description: str = Field(
        description="Who the companion is, in the third person; it carries the "
        "self-knowledge line"
    )
    communication_style: str = Field(description="How the companion speaks")
    core_values: List[str] = Field(max_length=MAX_CORE_VALUES)
    personality: PersonalityConfig
    style_rules: List[str] = Field(
        max_length=MAX_STYLE_RULES,
        description="The prompt's style section, one rule per line",
    )
    speaker_templates: SpeakerTemplates
    core_memory: Optional[CoreMemoryOverrides] = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return check_name(value, "the companion name", allow_empty=True)

    @field_validator("core_values", mode="before")
    @classmethod
    def _values_from_text(cls, value: Any) -> Any:
        return csv_list(value) if isinstance(value, str) else value

    @field_validator("style_rules", mode="before")
    @classmethod
    def _rules_from_text(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def _placeholders_are_known(self) -> "PersonaConfig":
        from mtc.brain.prompt import PLACEHOLDERS

        everywhere = [name for name in PLACEHOLDERS if name != "speaker"]
        checks = [
            ("self_description", self.self_description, everywhere),
            ("communication_style", self.communication_style, everywhere),
        ]
        for which in ("creator", "known", "unverified"):
            checks.append(
                (
                    f"speaker_templates.{which}",
                    getattr(self.speaker_templates, which),
                    PLACEHOLDERS,
                )
            )
        for where, text, allowed in checks:
            unknown = _unknown_placeholders(text, allowed)
            if unknown:
                raise ValueError(
                    f"{where} uses unknown placeholders: {', '.join(unknown)}"
                )
        return self

    def to_yaml(self) -> str:
        """Render the persona file text (keys in the documented order)."""
        import yaml

        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True, default_flow_style=False
        )

    @classmethod
    def from_yaml(cls, text: str) -> "PersonaConfig":
        """Parse persona file text; a document that is not a mapping raises."""
        import yaml

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("a persona document must be a mapping")
        return cls.model_validate(loaded)


# ---------------------------------------------------------------------------
# The speakers file
# ---------------------------------------------------------------------------


class SpeakerConfig(_Model):
    """One speaker entry; trust is granted only after token authentication."""

    name: str
    role: Literal["creator", "companion", "guest"]
    trust: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    token_env: Optional[str] = Field(
        None,
        description="Environment variable that holds the speaker's token; left "
        "out, the SPEAKER_TOKENS setting supplies it",
    )

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return check_name(value, "the speaker name")

    @field_validator("token_env")
    @classmethod
    def _token_env(cls, value: Optional[str]) -> Optional[str]:
        return _check_env_name(value, "token_env")


class SpeakersFile(_Model):
    """The speakers file: a list of entries with distinct names."""

    speakers: List[SpeakerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _names_are_distinct(self) -> "SpeakersFile":
        names = [entry.name for entry in self.speakers]
        if len(set(names)) != len(names):
            raise ValueError("speaker names must be distinct")
        return self

    def to_yaml(self) -> str:
        """Render the ``speakers`` list as YAML (no comment header)."""
        import yaml

        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True, default_flow_style=False
        )

    @classmethod
    def from_yaml(cls, text: str) -> "SpeakersFile":
        """Parse speakers file text; an empty document is an empty table."""
        import yaml

        loaded = yaml.safe_load(text)
        if loaded is None:
            return cls()
        if not isinstance(loaded, dict):
            raise ValueError("a speakers document must be a mapping")
        return cls.model_validate(loaded)


# ---------------------------------------------------------------------------
# Presets and the companion-type table
# ---------------------------------------------------------------------------


def _preset(values: tuple) -> PersonalityConfig:
    openness, conscientiousness, extraversion, agreeableness, neuroticism = values
    return PersonalityConfig(
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        neuroticism=neuroticism,
    )


PERSONALITY_PRESETS: Dict[str, PersonalityConfig] = {
    "balanced": _preset((0.6, 0.6, 0.5, 0.6, 0.4)),
    "curious": _preset((0.85, 0.55, 0.55, 0.6, 0.35)),
    "calm": _preset((0.55, 0.7, 0.4, 0.7, 0.2)),
    "playful": _preset((0.75, 0.45, 0.75, 0.65, 0.35)),
    "analytical": _preset((0.7, 0.8, 0.35, 0.5, 0.3)),
}


class TypeDefaults(_Model):
    """What a companion type drives (one row of ``templates/types.yaml``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    address_form_rule: Literal["full_name", "first_name", "operator"]
    creator_trust: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    seed_pack: Optional[str] = None
    creator_relationship: str = Field(min_length=1)
    ctm_tone: str = Field(min_length=1)

    @field_validator("ctm_tone")
    @classmethod
    def _tone_has_no_placeholder(cls, value: str) -> str:
        if "{" in value or "}" in value:
            raise ValueError("ctm_tone carries no placeholder")
        return value.strip()


def load_type_defaults() -> Dict[CompanionType, TypeDefaults]:
    """Read the companion-type table; every type must have exactly one row."""
    import yaml

    path = TEMPLATES_DIR / TYPES_FILE
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or set(loaded) != set(COMPANION_TYPES):
        raise ValueError(
            f"{TYPES_FILE} must hold one row for each of {', '.join(COMPANION_TYPES)}"
        )
    return {
        CompanionType(name): TypeDefaults.model_validate(loaded[name])
        for name in COMPANION_TYPES
    }


def default_address_form(companion_type: Any, creator_name: str) -> str:
    """How a companion of this type addresses the creator by default."""
    rule = load_type_defaults()[CompanionType(companion_type)].address_form_rule
    name = str(creator_name).strip()
    if rule == "operator":
        return "operator"
    if rule == "first_name":
        parts = name.split()
        return parts[0] if parts else name
    return name


# ---------------------------------------------------------------------------
# The wizard's answers
# ---------------------------------------------------------------------------


class StoreChoice(_Model):
    """One store: whether to use it and how to reach it.

    A ``None`` host, port or user stands for the Settings default.
    """

    enabled: bool = False
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    user: Optional[str] = None
    password: Optional[SecretStr] = Field(
        None, description="Typed at the prompt; an answers file uses password_env"
    )
    password_env: Optional[str] = Field(
        None, description="Environment variable that holds the store password"
    )

    @field_validator("host")
    @classmethod
    def _host(cls, value: Optional[str]) -> Optional[str]:
        return _check_host(value, "host")

    @field_validator("user")
    @classmethod
    def _user(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not _ROLE_NAME.match(value.strip()):
            raise ValueError("user must be letters, digits and underscores")
        return value.strip()

    @field_validator("password")
    @classmethod
    def _password(cls, value: Optional[SecretStr]) -> Optional[SecretStr]:
        return _check_secret(value, "password")

    @field_validator("password_env")
    @classmethod
    def _password_env(cls, value: Optional[str]) -> Optional[str]:
        return _check_env_name(value, "password_env")

    @model_validator(mode="after")
    def _one_password_source(self) -> "StoreChoice":
        if self.password is not None and self.password_env is not None:
            raise ValueError("give password or password_env, not both")
        return self


class StoresAnswers(_Model):
    """Which stores to use; everything stays in process memory by default."""

    db_prefix: str = Field(
        "mtc",
        pattern=DB_PREFIX_PATTERN,
        description="Prefix of every store name (database, collection, keys)",
    )
    postgres: StoreChoice = Field(default_factory=StoreChoice)
    mongodb: StoreChoice = Field(default_factory=StoreChoice)
    qdrant: StoreChoice = Field(default_factory=StoreChoice)
    redis: StoreChoice = Field(default_factory=StoreChoice)
    neo4j: StoreChoice = Field(default_factory=StoreChoice)
    service_manager: Literal["none", "brew", "docker", "systemd"] = Field(
        "none", description="How the stack scripts start and stop the stores"
    )


class BrainAnswers(_Model):
    """The language model behind the companion."""

    provider: Literal["mock", "local", "anthropic"] = Field(
        "mock",
        description="mock (labelled test replies), local (a chat-completions "
        "server on this machine) or anthropic (prompts leave the machine)",
    )
    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)
    base_url: Optional[str] = Field(
        None, description="Base URL of the local server; overrides host and port"
    )
    model: str = Field("", max_length=NAME_MAX_CHARS)
    template_family: Literal["generic", "qwen", "gemma"] = "generic"
    anthropic_model: str = Field("", max_length=NAME_MAX_CHARS)
    anthropic_api_key: Optional[SecretStr] = Field(
        None, description="Typed at the prompt; an answers file uses the _env field"
    )
    anthropic_api_key_env: Optional[str] = Field(
        None, description="Environment variable that holds the Anthropic key"
    )

    @field_validator("host")
    @classmethod
    def _host(cls, value: Optional[str]) -> Optional[str]:
        return _check_host(value, "host")

    @field_validator("base_url")
    @classmethod
    def _base_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        url = value.strip().rstrip("/")
        parts = urlsplit(url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.netloc
            or any(character.isspace() for character in url)
            or any(character in url for character in _SECRET_FORBIDDEN)
        ):
            raise ValueError("base_url must be an http or https URL")
        return url

    @field_validator("model", "anthropic_model")
    @classmethod
    def _model_id(cls, value: str) -> str:
        text = value.strip()
        if _has_control(text) or any(mark in text for mark in _SECRET_FORBIDDEN):
            raise ValueError("a model id must be plain text")
        return text

    @field_validator("anthropic_api_key")
    @classmethod
    def _key(cls, value: Optional[SecretStr]) -> Optional[SecretStr]:
        return _check_secret(value, "anthropic_api_key")

    @field_validator("anthropic_api_key_env")
    @classmethod
    def _key_env(cls, value: Optional[str]) -> Optional[str]:
        return _check_env_name(value, "anthropic_api_key_env")

    @model_validator(mode="after")
    def _provider_needs(self) -> "BrainAnswers":
        if self.anthropic_api_key is not None and self.anthropic_api_key_env:
            raise ValueError(
                "give anthropic_api_key or anthropic_api_key_env, not both"
            )
        if self.provider == "anthropic" and not self.anthropic_model:
            raise ValueError("the anthropic provider needs anthropic_model")
        return self


class LocaleAnswers(_Model):
    """Where and when the companion keeps its day."""

    timezone: str = Field("UTC", description="An IANA time zone name")
    wake_hour: int = Field(6, ge=0, le=23)
    sleep_hour: int = Field(22, ge=0, le=23)

    @field_validator("timezone")
    @classmethod
    def _zone(cls, value: str) -> str:
        zone = value.strip()
        try:
            if not zone:
                raise ValueError("empty")
            ZoneInfo(zone)
        except (KeyError, ValueError, OSError):
            raise ValueError("timezone must be an IANA zone name") from None
        return zone

    @model_validator(mode="after")
    def _hours_differ(self) -> "LocaleAnswers":
        if self.wake_hour == self.sleep_hour:
            raise ValueError("wake_hour and sleep_hour must differ")
        return self


class ApiAnswers(_Model):
    """Where the API server listens; ``None`` keeps the Settings default."""

    host: Optional[str] = None
    port: Optional[int] = Field(None, ge=1, le=65535)

    @field_validator("host")
    @classmethod
    def _host(cls, value: Optional[str]) -> Optional[str]:
        return _check_host(value, "host")


class PersonaAnswers(_Model):
    """Who the companion is and who set it up."""

    name: str = Field(description="The companion's name")
    pronouns: Pronouns
    companion_type: CompanionType
    creator_name: str = Field(description="The name of the person setting this up")
    address_form: Optional[str] = Field(
        None,
        description="How the companion addresses the creator; left out, the "
        "type's default applies",
    )
    personality_preset: Optional[str] = Field(
        "balanced", description="balanced, curious, calm, playful or analytical"
    )
    personality: Optional[PersonalityConfig] = Field(
        None, description="Five custom values; they replace the preset"
    )
    core_values: Optional[List[str]] = Field(
        None,
        max_length=MAX_CORE_VALUES,
        description="Left out, the CORE_VALUES default applies",
    )
    creator_trusted: bool = Field(
        True,
        description="Put the creator on the trusted list (still only after "
        "their token authenticates them)",
    )

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return check_name(value, "the companion name")

    @field_validator("creator_name")
    @classmethod
    def _creator(cls, value: str) -> str:
        return check_name(value, "the creator name")

    @field_validator("address_form")
    @classmethod
    def _address(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return check_name(value, "the address form")

    @field_validator("core_values")
    @classmethod
    def _values(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        checked = [check_name(item, "a core value") for item in value]
        if not checked:
            raise ValueError("core_values must hold at least one value")
        return checked

    @model_validator(mode="after")
    def _one_personality_source(self) -> "PersonaAnswers":
        if self.personality is not None:
            return self
        if self.personality_preset not in PERSONALITY_PRESETS:
            raise ValueError(
                "personality_preset must be one of "
                f"{', '.join(PERSONALITY_PRESETS)} (or give personality)"
            )
        return self

    def resolved_personality(self) -> PersonalityConfig:
        """The custom values when given, else the named preset."""
        if self.personality is not None:
            return self.personality
        return PERSONALITY_PRESETS[str(self.personality_preset)]


class OnboardingAnswers(_Model):
    """Everything the setup asks, and nothing it computes."""

    schema_version: Literal[1] = 1
    acknowledged: bool = Field(
        description="True once the framing text has been read",
        json_schema_extra={"const": True},
    )
    persona: PersonaAnswers
    locale: LocaleAnswers = Field(default_factory=LocaleAnswers)
    stores: StoresAnswers = Field(default_factory=StoresAnswers)
    brain: BrainAnswers = Field(default_factory=BrainAnswers)
    api: ApiAnswers = Field(default_factory=ApiAnswers)
    bootstrap_db: bool = Field(
        False, description="Run the database bootstrap after the files are written"
    )

    @field_validator("acknowledged", mode="before")
    @classmethod
    def _acknowledged(cls, value: Any) -> Any:
        if value is not True:
            raise ValueError(
                "acknowledged must be true: read the framing text mtc init "
                "shows, then set it"
            )
        return value


def export_json_schemas() -> Dict[str, Dict[str, Any]]:
    """The JSON schemas committed under ``docs/schemas``, by file name."""
    return {
        "persona.schema.json": PersonaConfig.model_json_schema(),
        "onboarding-answers.schema.json": OnboardingAnswers.model_json_schema(),
    }
