"""
Setup Wizard
============

``mtc init``: eleven steps that turn a person's answers into a configured
project. Steps 1 to 8 and the bootstrap question collect an
``OnboardingAnswers`` document (interactively through a ``Prompter``, or from
an answers file); step 9 writes the files; step 10 runs the database
bootstrap when asked; step 11 prints what to do next.

What the write step produces, all inside the project directory:

- the persona file (``PERSONA_PATH``, default ``config/persona.yaml``) from
  the companion type's template, with the name, pronouns, creator,
  personality and core values filled in;
- the speakers file (the shipped ``speakers.example.yaml`` comment header and
  one creator entry with the type's trust; no token and no ``token_env``);
- the identity statement (``IDENTITY_STATEMENT_PATH``) from the type's
  template, with its placeholders left for the prompt assembler;
- ``<CTM_CONFIG_DIR>/persona.yaml`` with the type's tone and, for a type
  that has one, the seed pack as ``<CTM_CONFIG_DIR>/seeds.d/10_<pack>.yaml``;
- ``.env`` entries: ``SYSTEM_NAME``, ``CREATOR_NAME``,
  ``CREATOR_RELATIONSHIP``, ``COMPANION_TYPE``, ``CORE_VALUES``,
  ``TIMEZONE``, ``WAKE_HOUR``, ``SLEEP_HOUR``, ``DB_PREFIX``, the five store
  enable keys and, for an enabled store, its host, port, derived names and
  credentials, ``CTM_STORE``, ``SERVICE_MANAGER``, ``LLM_PROVIDER`` with the
  provider's keys, ``API_HOST``, ``API_PORT``, ``API_AUTH_ENABLED``,
  ``API_KEY`` and the two trust entries from
  ``mtc.pipeline.speaker_trust.env_entries``. ``SYSTEM_BIRTH_DATE`` is written
  only by the bootstrap.

Secrets (the API key, the creator's speaker token, store passwords, an
Anthropic key) are generated with ``secrets`` or typed at a hidden prompt,
written only to ``.env`` (mode 0600, existing values kept) and never printed,
logged or placed in any other file. Every file written here is private (it
holds the names the operator chose) and gitignored. An existing file is
never replaced without ``force``.

Probes use the standard library only and never block setup. Importing this
module loads no store client, no web framework and no HTTP client.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from pydantic import SecretStr, ValidationError

from mtc.core.checks import exit_code_for, render_text
from mtc.core.config import Settings, csv_list
from mtc.core.envfile import read_env, render_value, upsert_env
from mtc.core.paths import resolve_path
from mtc.core.resources import read_shipped_text
from mtc.onboarding.schema import (
    DB_PREFIX_PATTERN,
    PERSONALITY_PRESETS,
    SERVICE_MANAGERS,
    STORE_NAMES,
    TEMPLATES_DIR,
    ApiAnswers,
    BrainAnswers,
    CreatorConfig,
    LocaleAnswers,
    OnboardingAnswers,
    PersonaAnswers,
    PersonaConfig,
    PersonalityConfig,
    Pronouns,
    SpeakerConfig,
    SpeakersFile,
    StoreChoice,
    StoresAnswers,
    TypeDefaults,
    check_name,
    default_address_form,
    load_type_defaults,
)
from mtc.pipeline import speaker_trust

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

ENV_FILE_NAME = ".env"
PRIVATE_FILE_MODE = 0o600
PRIVATE_DIR_MODE = 0o700
SPEAKERS_EXAMPLE = "config/speakers.example.yaml"
SEED_PACK_PREFIX = "10_"
SEED_PACK_DIR = "seeds.d"
ENV_HEADER = "Written by mtc init"

STEP_TITLES = (
    "Honest framing",
    "Name and pronouns",
    "Companion type",
    "Creator",
    "Personality and core values",
    "Timezone and hours",
    "Stores",
    "Brain",
    "Write",
    "Bootstrap",
    "Next steps",
)

PRONOUN_SETS: Dict[str, Tuple[str, str, str]] = {
    "they": ("they", "them", "their"),
    "she": ("she", "her", "her"),
    "he": ("he", "him", "his"),
    "it": ("it", "it", "its"),
}
TYPE_SUMMARIES: Dict[str, str] = {
    "child": "grows and learns from you; the developmental stages are on",
    "sibling": "a peer with banter that stays kind",
    "friend": "a peer with warmth and views of its own",
    "partner": "companionship and affection in words, inside the content boundary",
    "mentor": "a patient guide who asks before telling",
    "research_subject": "the engine alone, with a neutral description and no role",
}
STORE_LABELS: Dict[str, str] = {
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "qdrant": "Qdrant",
    "redis": "Redis",
    "neo4j": "Neo4j",
}
STORE_PURPOSE: Dict[str, str] = {
    "mongodb": "recommended: thoughts, turns and telemetry",
    "qdrant": "recommended: semantic recall",
    "redis": "optional: working memory across restarts",
    "neo4j": "optional: the knowledge graph",
}
# Compose service names, by store.
COMPOSE_SERVICES: Dict[str, str] = {
    "postgres": "postgres",
    "mongodb": "mongo",
    "qdrant": "qdrant",
    "redis": "redis",
    "neo4j": "neo4j",
}
_ENABLE_FIELDS: Dict[str, str] = {
    "postgres": "POSTGRES_ENABLED",
    "mongodb": "MONGODB_ENABLED",
    "qdrant": "QDRANT_ENABLED",
    "redis": "REDIS_ENABLED",
    "neo4j": "NEO4J_ENABLED",
}
_HOST_PORT_FIELDS: Dict[str, Tuple[str, str]] = {
    "postgres": ("POSTGRES_HOST", "POSTGRES_PORT"),
    "mongodb": ("MONGODB_HOST", "MONGODB_PORT"),
    "qdrant": ("QDRANT_HOST", "QDRANT_PORT"),
    "redis": ("REDIS_HOST", "REDIS_PORT"),
}
_PASSWORD_FIELDS: Dict[str, str] = {
    "postgres": "POSTGRES_PASSWORD",
    "mongodb": "MONGODB_PASSWORD",
    "redis": "REDIS_PASSWORD",
    "neo4j": "NEO4J_PASSWORD",
}
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ---------------------------------------------------------------------------
# Small helpers over Settings
# ---------------------------------------------------------------------------


def _default(field_name: str) -> Any:
    """The Settings field default (the one place defaults come from)."""
    return Settings.model_fields[field_name].default


def env_key(field_name: str) -> str:
    """The environment variable that carries a Settings field."""
    info = Settings.model_fields[field_name]
    alias = info.validation_alias
    choices = getattr(alias, "choices", None)
    if choices:
        return str(choices[0])
    if isinstance(alias, str):
        return alias
    return f"{Settings.model_config.get('env_prefix', '')}{field_name}"


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _neo4j_default_endpoint() -> Tuple[str, int]:
    parts = urlsplit(str(_default("NEO4J_URI")))
    return str(parts.hostname), int(parts.port or 0)


def store_endpoint(store: str, choice: StoreChoice) -> Tuple[str, int]:
    """The host and port a store choice means, defaults from Settings."""
    if store == "neo4j":
        default_host, default_port = _neo4j_default_endpoint()
    else:
        host_field, port_field = _HOST_PORT_FIELDS[store]
        default_host, default_port = str(_default(host_field)), int(
            _default(port_field)
        )
    return choice.host or default_host, int(choice.port or default_port)


def _settings_for(project_dir: Path, existing_env: Mapping[str, str]) -> Settings:
    """Settings as the project will load them: its ``.env`` values over the
    defaults, rooted at the project directory."""
    prefix = str(Settings.model_config.get("env_prefix", ""))
    values: Dict[str, Any] = {}
    for key, value in existing_env.items():
        name = key[len(prefix) :] if key.startswith(prefix) else ""
        if name in Settings.model_fields and name != "PROJECT_ROOT":
            values[name] = value
    values["PROJECT_ROOT"] = str(project_dir)
    return Settings(_env_file=None, **values)


def _derived_names(db_prefix: str) -> Settings:
    """Settings whose store names derive from ``db_prefix`` alone."""
    return Settings(
        _env_file=None,
        DB_PREFIX=db_prefix,
        POSTGRES_DB="",
        POSTGRES_USER="",
        MONGODB_DB="",
        QDRANT_COLLECTION="",
        REDIS_KEY_PREFIX="",
    )


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

Validator = Callable[[str], str]


class Prompter(Protocol):
    """How the wizard talks to a person."""

    def say(self, text: str) -> None: ...

    def ask(
        self,
        question: str,
        default: Optional[str] = None,
        validate: Optional[Validator] = None,
    ) -> str: ...

    def ask_choice(
        self, question: str, choices: Sequence[str], default: str
    ) -> str: ...

    def ask_bool(self, question: str, default: bool) -> bool: ...

    def ask_secret(self, question: str) -> str: ...


class InputEnded(RuntimeError):
    """The input ran out before the questions did."""


class NotAcknowledged(RuntimeError):
    """The framing text was not acknowledged; nothing is written."""


def _parse_bool(text: str, default: bool) -> bool:
    answer = text.strip().lower()
    if not answer:
        return default
    if answer in ("y", "yes", "true", "1"):
        return True
    if answer in ("n", "no", "false", "0"):
        return False
    raise ValueError("answer yes or no")


def _parse_choice(text: str, choices: Sequence[str], default: str) -> str:
    answer = text.strip()
    if not answer:
        return default
    if answer.isdigit() and 1 <= int(answer) <= len(choices):
        return choices[int(answer) - 1]
    lowered = {choice.lower(): choice for choice in choices}
    if answer.lower() in lowered:
        return lowered[answer.lower()]
    raise ValueError(f"choose one of: {', '.join(choices)}")


class _LinePrompter:
    """Shared prompt logic over ``_read`` (a line) and ``_read_secret``."""

    def say(self, text: str) -> None:
        raise NotImplementedError

    def _read(self, prompt: str) -> str:
        raise NotImplementedError

    def _read_secret(self, prompt: str) -> str:
        raise NotImplementedError

    def ask(
        self,
        question: str,
        default: Optional[str] = None,
        validate: Optional[Validator] = None,
    ) -> str:
        suffix = f" [{default}]" if default not in (None, "") else ""
        while True:
            answer = self._read(f"{question}{suffix}: ").strip()
            if not answer and default is not None:
                answer = default
            try:
                return validate(answer) if validate is not None else answer
            except ValueError as exc:
                self.say(f"  {exc}")

    def ask_choice(self, question: str, choices: Sequence[str], default: str) -> str:
        listing = ", ".join(f"{i}) {choice}" for i, choice in enumerate(choices, 1))
        while True:
            answer = self._read(f"{question} ({listing}) [{default}]: ")
            try:
                return _parse_choice(answer, choices, default)
            except ValueError as exc:
                self.say(f"  {exc}")

    def ask_bool(self, question: str, default: bool) -> bool:
        hint = "Y/n" if default else "y/N"
        while True:
            answer = self._read(f"{question} [{hint}]: ")
            try:
                return _parse_bool(answer, default)
            except ValueError as exc:
                self.say(f"  {exc}")

    def ask_secret(self, question: str) -> str:
        return self._read_secret(f"{question}: ")


class ConsolePrompter(_LinePrompter):
    """Prompts on the terminal; secrets are read without echo."""

    def say(self, text: str) -> None:
        print(text)

    def _read(self, prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:
            raise InputEnded("the input ended before setup finished") from None

    def _read_secret(self, prompt: str) -> str:
        import getpass

        try:
            return getpass.getpass(prompt)
        except EOFError:
            raise InputEnded("the input ended before setup finished") from None


class ScriptedPrompter(_LinePrompter):
    """Answers from a list, for tests and scripted runs.

    ``said`` holds every line shown, ``questions`` every question asked and
    ``secret_questions`` the ones asked through the hidden path.
    """

    def __init__(self, answers: Sequence[str]) -> None:
        self._answers = list(answers)
        self.said: List[str] = []
        self.questions: List[str] = []
        self.secret_questions: List[str] = []

    @property
    def remaining(self) -> int:
        return len(self._answers)

    def say(self, text: str) -> None:
        self.said.append(text)

    def _next(self) -> str:
        if not self._answers:
            raise InputEnded("the scripted answers ran out")
        return self._answers.pop(0)

    def _read(self, prompt: str) -> str:
        self.questions.append(prompt)
        return self._next()

    def _read_secret(self, prompt: str) -> str:
        self.secret_questions.append(prompt)
        return self._next()


# ---------------------------------------------------------------------------
# Probes (standard library only; a probe never blocks setup)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """A probe outcome in plain language; ``status`` uses the check words."""

    ok: bool
    status: str
    message: str


def _brain_base_url(brain: BrainAnswers) -> str:
    if brain.base_url:
        return brain.base_url
    host = brain.host or str(_default("LLM_HOST"))
    port = int(brain.port or _default("LLM_PORT"))
    return urlunsplit(("http", f"{host}:{port}", "", "", ""))


def probe_brain(brain: BrainAnswers, *, timeout_s: float = 5.0) -> ProbeResult:
    """Ask the local model server's health paths whether it is up."""
    if brain.provider == "mock":
        return ProbeResult(
            True,
            "skipped",
            "The mock provider needs no server. Its replies are labelled test text.",
        )
    if brain.provider == "anthropic":
        return ProbeResult(
            False,
            "insufficient_data",
            "Not probed: checking the key would be a paid request. It is "
            "checked when the first message is sent.",
        )
    import socket
    import urllib.error
    import urllib.request

    base = _brain_base_url(brain)
    where = urlsplit(base).netloc
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    refused = ProbeResult(
        False,
        "unreachable",
        f"Nothing is listening at {where}. Start the model server, then run "
        "mtc doctor. Setup continues.",
    )
    slow = ProbeResult(
        False,
        "unreachable",
        f"The server at {where} did not answer within {timeout_s:g} seconds. A "
        "large model can take minutes to load; run mtc doctor once it is up.",
    )
    last: Optional[ProbeResult] = None
    for path in csv_list(str(_default("LLM_HEALTH_PATHS"))):
        url = f"{base}/{path.lstrip('/')}"
        try:
            with opener.open(url, timeout=timeout_s) as response:
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            last = ProbeResult(
                False,
                "failed",
                f"The server at {where} answered with HTTP {exc.code}. It is "
                "running but not ready, or it is another kind of server.",
            )
            continue
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                return slow
            if isinstance(reason, ConnectionRefusedError):
                return refused
            return ProbeResult(
                False,
                "unreachable",
                f"The server at {where} could not be reached "
                f"({type(reason).__name__}). Setup continues.",
            )
        except (TimeoutError, socket.timeout):
            return slow
        except OSError as exc:
            return ProbeResult(
                False,
                "unreachable",
                f"The server at {where} could not be reached "
                f"({type(exc).__name__}). Setup continues.",
            )
        if status == 200:
            return ProbeResult(True, "ok", f"The model server at {where} answered.")
        last = ProbeResult(
            False, "failed", f"The server at {where} answered with HTTP {status}."
        )
    if last is not None:
        return last
    return ProbeResult(False, "insufficient_data", "No health path is configured.")


def probe_store(
    choice: StoreChoice, store: str, *, timeout_s: float = 2.0
) -> ProbeResult:
    """Check that something accepts connections where the store should be."""
    if store not in STORE_NAMES:
        raise ValueError(f"unknown store {store!r}")
    label = STORE_LABELS[store]
    if not choice.enabled:
        return ProbeResult(True, "skipped", f"{label} is not enabled.")
    import socket

    host, port = store_endpoint(store, choice)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except (TimeoutError, socket.timeout):
        return ProbeResult(
            False,
            "unreachable",
            f"{label} at {host} port {port} did not answer within "
            f"{timeout_s:g} seconds.",
        )
    except OSError:
        return ProbeResult(
            False,
            "unreachable",
            f"Nothing is listening for {label} at {host} port {port}. Start it "
            "before mtc db init.",
        )
    return ProbeResult(
        True, "ok", f"{label} accepts connections at {host} port {port}."
    )


# ---------------------------------------------------------------------------
# Collecting answers
# ---------------------------------------------------------------------------


def detect_timezone(
    environ: Optional[Mapping[str, str]] = None,
    localtime: Optional[Path] = None,
) -> Optional[str]:
    """The system's IANA zone from ``TZ`` or the zone link, else None."""
    env = os.environ if environ is None else environ
    candidates = [str(env.get("TZ", "")).strip().lstrip(":")]
    link = Path("/etc/localtime") if localtime is None else Path(localtime)
    try:
        target = os.path.realpath(link)
    except OSError:
        target = ""
    marker = "zoneinfo/"
    if marker in target:
        candidates.append(target.split(marker, 1)[1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            ZoneInfo(candidate)
        except (KeyError, ValueError, OSError):
            continue
        return candidate
    return None


def _step(prompter: Prompter, number: int) -> None:
    prompter.say("")
    prompter.say(f"Step {number} of {len(STEP_TITLES)}: {STEP_TITLES[number - 1]}")


def framing_text() -> str:
    """The honest framing shown before anything is asked."""
    return (TEMPLATES_DIR / "framing.txt").read_text(encoding="utf-8")


def example_answers_text() -> str:
    """The synthetic sample answers file."""
    return (TEMPLATES_DIR / "answers.example.yaml").read_text(encoding="utf-8")


def _name_validator(what: str) -> Validator:
    return lambda text: check_name(text, what)


def _hour(text: str) -> str:
    if not text.strip().isdigit() or not 0 <= int(text) <= 23:
        raise ValueError("an hour is a whole number from 0 to 23")
    return str(int(text))


def _port(text: str) -> str:
    if not text.strip().isdigit() or not 1 <= int(text) <= 65535:
        raise ValueError("a port is a whole number from 1 to 65535")
    return str(int(text))


def _unit_value(text: str) -> str:
    try:
        value = float(text)
    except ValueError:
        raise ValueError("give a number from 0 to 1") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError("give a number from 0 to 1")
    return str(value)


def _zone(text: str) -> str:
    try:
        return LocaleAnswers(timezone=text).timezone
    except ValidationError:
        raise ValueError(
            "that is not an IANA zone name (for example UTC or Europe/Berlin)"
        ) from None


def _db_prefix(text: str) -> str:
    if not re.match(DB_PREFIX_PATTERN, text):
        raise ValueError(
            "a prefix starts with a lowercase letter and holds lowercase "
            "letters, digits and underscores"
        )
    return text


def _secret_or_none(text: str) -> Optional[SecretStr]:
    return SecretStr(text) if text else None


def _ask_password(prompter: Prompter, question: str, what: str) -> Optional[SecretStr]:
    """A typed secret, re-asked until it can sit in ``.env``; Enter is None."""
    while True:
        typed = prompter.ask_secret(question)
        if not typed:
            return None
        try:
            return StoreChoice(enabled=True, password=typed).password
        except ValidationError as exc:
            for line in _error_lines(exc):
                reason = line.split(": ", 1)[-1]
                prompter.say(f"  The {what} cannot be used: {reason}")


def _collect_persona(prompter: Prompter) -> PersonaAnswers:
    _step(prompter, 2)
    name = prompter.ask(
        "What is the companion called", validate=_name_validator("the name")
    )
    choice = prompter.ask_choice(
        "Pronouns", list(PRONOUN_SETS) + ["custom"], default="they"
    )
    if choice == "custom":

        def word(text: str) -> str:
            try:
                return Pronouns(subject=text).subject
            except ValidationError:
                raise ValueError("a pronoun is one word of letters") from None

        pronouns = Pronouns(
            subject=prompter.ask("Subject form (as in they)", validate=word),
            object=prompter.ask("Object form (as in them)", validate=word),
            possessive=prompter.ask("Possessive form (as in their)", validate=word),
        )
    else:
        subject, object_form, possessive = PRONOUN_SETS[choice]
        pronouns = Pronouns(subject=subject, object=object_form, possessive=possessive)

    _step(prompter, 3)
    for kind, summary in TYPE_SUMMARIES.items():
        prompter.say(f"  {kind}: {summary}")
    kind = prompter.ask_choice(
        "Companion type", list(TYPE_SUMMARIES), default="research_subject"
    )

    _step(prompter, 4)
    creator = prompter.ask("Your name", validate=_name_validator("the creator name"))
    suggested = default_address_form(kind, creator)
    if kind == "child":
        question = f"How does {name} address you (a parental word, or your name)"
    else:
        question = f"How does {name} address you"
    address = prompter.ask(
        question, default=suggested, validate=_name_validator("the address form")
    )

    _step(prompter, 5)
    preset = prompter.ask_choice(
        "Personality", list(PERSONALITY_PRESETS) + ["custom"], default="balanced"
    )
    personality: Optional[PersonalityConfig] = None
    if preset == "custom":
        prompter.say("  Five values, each from 0 to 1.")
        personality = PersonalityConfig(
            **{
                trait: float(prompter.ask(trait.capitalize(), validate=_unit_value))
                for trait in PersonalityConfig.model_fields
            }
        )
    default_values = str(_default("CORE_VALUES"))

    def values(text: str) -> str:
        items = csv_list(text)
        if not items:
            raise ValueError("give at least one value")
        return ",".join(check_name(item, "a core value") for item in items)

    chosen = prompter.ask(
        "Core values, comma-separated", default=default_values, validate=values
    )
    return PersonaAnswers(
        name=name,
        pronouns=pronouns,
        companion_type=kind,
        creator_name=creator,
        address_form=None if address == suggested else address,
        personality_preset=None if personality is not None else preset,
        personality=personality,
        core_values=None if chosen == default_values else csv_list(chosen),
    )


def _collect_locale(prompter: Prompter) -> LocaleAnswers:
    _step(prompter, 6)
    detected = detect_timezone()
    if detected:
        prompter.say(f"  Detected time zone: {detected}. Press Enter to keep it.")
    zone = prompter.ask(
        "Time zone (an IANA name)",
        default=detected or str(_default("TIMEZONE")),
        validate=_zone,
    )
    wake = prompter.ask(
        "Hour the day starts (0 to 23)",
        default=str(_default("WAKE_HOUR")),
        validate=_hour,
    )

    def sleep_hour(text: str) -> str:
        hour = _hour(text)
        if hour == wake:
            raise ValueError("the night must start at a different hour")
        return hour

    sleep = prompter.ask(
        "Hour the night starts (0 to 23)",
        default=str(_default("SLEEP_HOUR")),
        validate=sleep_hour,
    )
    return LocaleAnswers(timezone=zone, wake_hour=int(wake), sleep_hour=int(sleep))


def _collect_store(
    prompter: Prompter, store: str, manager: str, db_prefix: str, probe: bool
) -> StoreChoice:
    label = STORE_LABELS[store]
    default_host, default_port = store_endpoint(store, StoreChoice())
    host = prompter.ask(
        f"{label} host",
        default=default_host,
        validate=lambda text: str(StoreChoice(host=text).host),
    )
    port = int(prompter.ask(f"{label} port", default=str(default_port), validate=_port))
    user: Optional[str] = None
    password: Optional[SecretStr] = None
    if store == "postgres":
        password = _ask_password(
            prompter,
            f"{label} password (Enter generates one)",
            f"{label} password",
        )
    elif store == "mongodb":
        wants = prompter.ask_bool(
            f"Does {label} use a user and a password", default=False
        )
        if wants:
            user = prompter.ask(
                f"{label} user",
                default=db_prefix,
                validate=lambda text: str(StoreChoice(user=text).user),
            )
            password = _ask_password(
                prompter,
                f"{label} password (Enter generates one)",
                f"{label} password",
            )
    elif store == "redis":
        password = _ask_password(
            prompter, f"{label} password (Enter for none)", f"{label} password"
        )
    elif store == "neo4j":
        password = _ask_password(
            prompter,
            f"{label} password (Enter generates one)",
            f"{label} password",
        )
    choice = StoreChoice(
        enabled=True,
        host=None if host == default_host else host,
        port=None if port == default_port else port,
        user=user,
        password=password,
    )
    if probe:
        prompter.say(f"  {probe_store(choice, store).message}")
    return choice


def _collect_stores(prompter: Prompter, probe: bool) -> StoresAnswers:
    _step(prompter, 7)
    prompter.say(
        "  PostgreSQL is required for anything to persist. The other stores "
        "are optional."
    )
    if not prompter.ask_bool("Use PostgreSQL", default=False):
        prompter.say(
            "  Warning: nothing will persist. Conversations, memories and "
            "thoughts stay in process memory and are gone when the process "
            "stops. Run mtc init --force later to add stores."
        )
        return StoresAnswers()
    enabled = {"postgres": True}
    for store in ("mongodb", "qdrant", "redis", "neo4j"):
        recommended = store in ("mongodb", "qdrant")
        enabled[store] = prompter.ask_bool(
            f"Use {STORE_LABELS[store]} ({STORE_PURPOSE[store]})", default=recommended
        )
    manager = prompter.ask_choice(
        "How are the stores started",
        list(SERVICE_MANAGERS),
        default=str(_default("SERVICE_MANAGER")),
    )
    db_prefix = prompter.ask(
        "Prefix for store names",
        default=str(_default("DB_PREFIX")),
        validate=_db_prefix,
    )
    choices = {
        store: _collect_store(prompter, store, manager, db_prefix, probe)
        for store in STORE_NAMES
        if enabled[store]
    }
    return StoresAnswers(db_prefix=db_prefix, service_manager=manager, **choices)


def _collect_brain(prompter: Prompter, probe: bool) -> Tuple[BrainAnswers, ApiAnswers]:
    _step(prompter, 8)
    prompter.say("  mock: labelled test replies, no model needed")
    prompter.say("  local: a chat-completions server on this machine")
    prompter.say("  anthropic: an external service; prompts leave this machine")
    provider = prompter.ask_choice(
        "Brain provider", ["mock", "local", "anthropic"], default="mock"
    )
    brain = BrainAnswers(provider="mock")
    if provider == "local":
        default_host = str(_default("LLM_HOST"))
        default_port = int(_default("LLM_PORT"))
        host = prompter.ask(
            "Model server host",
            default=default_host,
            validate=lambda text: str(BrainAnswers(host=text).host),
        )
        port = int(
            prompter.ask("Model server port", default=str(default_port), validate=_port)
        )
        model = prompter.ask("Served model id (Enter for the loaded model)", default="")
        family = prompter.ask_choice(
            "Chat template family", ["generic", "qwen", "gemma"], default="generic"
        )
        brain = BrainAnswers(
            provider="local",
            host=None if host == default_host else host,
            port=None if port == default_port else port,
            model=model,
            template_family=family,
        )
        if probe:
            prompter.say(f"  {probe_brain(brain).message}")
    elif provider == "anthropic":

        def model_id(text: str) -> str:
            if not text.strip():
                raise ValueError("the Anthropic provider needs a model id")
            return text.strip()

        model = prompter.ask("Anthropic model id", validate=model_id)
        typed = prompter.ask_secret(
            "Anthropic key (Enter to add it to the environment file later)"
        )
        brain = BrainAnswers(
            provider="anthropic",
            anthropic_model=model,
            anthropic_api_key=_secret_or_none(typed),
        )
        prompter.say(f"  {probe_brain(brain).message}")
    else:
        prompter.say(f"  {probe_brain(brain).message}")
    default_api_port = int(_default("API_PORT"))
    api_port = int(
        prompter.ask(
            "Port for the API server on this machine",
            default=str(default_api_port),
            validate=_port,
        )
    )
    return brain, ApiAnswers(port=None if api_port == default_api_port else api_port)


def collect_interactive(
    prompter: Prompter,
    *,
    probe: bool = True,
    settings: Optional[Settings] = None,
) -> OnboardingAnswers:
    """Ask steps 1 to 8 and the bootstrap question; return the answers.

    ``settings`` is accepted for callers that hold one; every default shown
    is the Settings field default. Raises ``NotAcknowledged`` when the
    framing is not confirmed with the word yes.
    """
    del settings  # defaults come from the Settings model, not an instance
    _step(prompter, 1)
    prompter.say(framing_text())
    typed = prompter.ask("Type yes to confirm that you have read this", default="")
    if typed.strip().lower() != "yes":
        raise NotAcknowledged("the framing was not acknowledged; nothing was written")
    persona = _collect_persona(prompter)
    locale = _collect_locale(prompter)
    stores = _collect_stores(prompter, probe)
    brain, api = _collect_brain(prompter, probe)
    _step(prompter, 10)
    prompter.say(
        "  The bootstrap runs after the files are written. It records the "
        "install date, even with no store, and prepares each enabled store."
    )
    bootstrap = prompter.ask_bool("Run the bootstrap after the write", default=True)
    return OnboardingAnswers(
        acknowledged=True,
        persona=persona,
        locale=locale,
        stores=stores,
        brain=brain,
        api=api,
        bootstrap_db=bootstrap,
    )


# ---------------------------------------------------------------------------
# Answers files
# ---------------------------------------------------------------------------


class AnswersError(ValueError):
    """An answers document that cannot be used; ``errors`` lists every reason."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _error_lines(exc: ValidationError, prefix: str = "") -> List[str]:
    """One line per validation error: where and why, never the input."""
    lines = []
    for error in exc.errors(include_input=False, include_url=False):
        where = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "invalid")).replace("Value error, ", "")
        label = ".".join(part for part in (prefix, where) if part)
        if where == "acknowledged" and "framing" not in message:
            message += " (read the framing text mtc init shows, then set it to true)"
        lines.append(f"{label}: {message}" if label else message)
    return lines


def resolve_env_references(
    answers: OnboardingAnswers, environ: Optional[Mapping[str, str]] = None
) -> OnboardingAnswers:
    """Replace every ``*_env`` reference with the value of its variable.

    Raises ``AnswersError`` naming each variable that is unset or holds a
    value ``.env`` cannot carry; the values are never echoed.
    """
    env = os.environ if environ is None else environ
    errors: List[str] = []
    stores_update: Dict[str, StoreChoice] = {}
    for store in STORE_NAMES:
        choice: StoreChoice = getattr(answers.stores, store)
        if choice.password_env is None:
            continue
        where = f"stores.{store}.password_env"
        value = env.get(choice.password_env, "")
        if not value:
            errors.append(
                f"{where}: the environment variable {choice.password_env} is not set"
            )
            continue
        try:
            stores_update[store] = StoreChoice(
                **{
                    **choice.model_dump(exclude={"password", "password_env"}),
                    "password": value,
                }
            )
        except ValidationError as exc:
            errors.extend(_error_lines(exc, where))
    brain = answers.brain
    if brain.anthropic_api_key_env is not None:
        where = "brain.anthropic_api_key_env"
        value = env.get(brain.anthropic_api_key_env, "")
        if not value:
            errors.append(
                f"{where}: the environment variable "
                f"{brain.anthropic_api_key_env} is not set"
            )
        else:
            try:
                brain = BrainAnswers(
                    **{
                        **brain.model_dump(
                            exclude={"anthropic_api_key", "anthropic_api_key_env"}
                        ),
                        "anthropic_api_key": value,
                    }
                )
            except ValidationError as exc:
                errors.extend(_error_lines(exc, where))
    if errors:
        raise AnswersError(errors)
    stores = answers.stores.model_copy(update=stores_update)
    return answers.model_copy(update={"stores": stores, "brain": brain})


def load_answers(path: Path) -> OnboardingAnswers:
    """Read an answers file (YAML or JSON) and resolve its ``*_env`` references.

    Raises ``AnswersError`` listing every problem, without echoing a value.
    """
    import yaml

    source = Path(path)
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise AnswersError([f"answers file {source.name} not found"]) from None
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AnswersError(
            [f"answers file {source.name} could not be read ({type(exc).__name__})"]
        ) from None
    if not isinstance(loaded, dict):
        raise AnswersError([f"answers file {source.name} must hold a mapping"])
    try:
        answers = OnboardingAnswers.model_validate(loaded)
    except ValidationError as exc:
        raise AnswersError(_error_lines(exc)) from None
    return resolve_env_references(answers)


# ---------------------------------------------------------------------------
# Planning the write
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedFile:
    """One file the wizard writes."""

    path: Path
    content: str
    mode: int


@dataclass
class WritePlan:
    """Everything one run writes.

    ``env_updates`` holds secret values; ``describe`` shows paths and key
    names only. ``env_keep_existing`` names the keys whose value already in
    ``.env`` is kept. ``remove`` lists wizard-written seed packs a re-run
    replaces, and ``private_dirs`` the directories created with mode 0700
    (the agent space).
    """

    files: List[PlannedFile]
    env_updates: Dict[str, str]
    env_keep_existing: Set[str]
    env_path: Path = Path(ENV_FILE_NAME)
    project_dir: Path = Path(".")
    remove: List[Path] = field(default_factory=list)
    private_dirs: List[Path] = field(default_factory=list)

    def _shown(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_dir).as_posix()
        except ValueError:
            return path.name

    def describe(self) -> str:
        """Paths, modes and key names; never a value."""
        lines = ["Files to write:"]
        for planned in self.files:
            lines.append(f"  {self._shown(planned.path)} (mode {planned.mode:04o})")
        if self.remove:
            lines.append("Seed packs from an earlier run to remove:")
            lines.extend(f"  {self._shown(path)}" for path in self.remove)
        lines.append(
            f"Entries to write in {self._shown(self.env_path)} "
            f"(mode {PRIVATE_FILE_MODE:04o}):"
        )
        for key in sorted(self.env_updates):
            kept = " (kept when already set)" if key in self.env_keep_existing else ""
            lines.append(f"  {key}{kept}")
        return "\n".join(lines)


class PersonaExists(RuntimeError):
    """A file the wizard would write already exists and force was not given."""

    def __init__(self, existing: Sequence[str]) -> None:
        self.existing = list(existing)
        super().__init__(
            "these files already exist: "
            + ", ".join(self.existing)
            + "; pass --force to replace them"
        )


def _fill_known(text: str, values: Mapping[str, str]) -> str:
    return _PLACEHOLDER.sub(
        lambda match: values.get(match.group(1), match.group(0)), text
    )


def build_persona(answers: OnboardingAnswers) -> PersonaConfig:
    """The persona for these answers, from the companion type's template.

    The name, pronoun and creator placeholders in the two free-text fields
    are filled here, because the core memory blocks show those fields as
    written; the speaker templates keep theirs for the prompt assembler.
    """
    persona = answers.persona
    kind = persona.companion_type.value
    template = PersonaConfig.from_yaml(
        (TEMPLATES_DIR / "persona" / f"{kind}.yaml").read_text(encoding="utf-8")
    )
    address = persona.address_form or default_address_form(kind, persona.creator_name)
    values = {
        "system_name": persona.name,
        "creator_name": persona.creator_name,
        "creator_address": address,
        "pronoun_subject": persona.pronouns.subject,
        "pronoun_object": persona.pronouns.object,
        "pronoun_possessive": persona.pronouns.possessive,
    }
    core_values = persona.core_values or csv_list(str(_default("CORE_VALUES")))
    data = template.model_dump()
    data.update(
        name=persona.name,
        pronouns=persona.pronouns,
        creator=CreatorConfig(name=persona.creator_name, address_form=address),
        self_description=_fill_known(template.self_description, values).strip(),
        communication_style=_fill_known(template.communication_style, values).strip(),
        core_values=core_values,
        personality=persona.resolved_personality(),
    )
    return PersonaConfig.model_validate(data)


def _comment_header(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.rstrip())
    return "\n".join(lines)


def _speakers_text(creator: str, defaults: TypeDefaults, cfg: Settings) -> str:
    example, _where = read_shipped_text(SPEAKERS_EXAMPLE, settings=cfg)
    document = SpeakersFile(
        speakers=[
            SpeakerConfig(name=creator, role="creator", trust=defaults.creator_trust)
        ]
    )
    header = _comment_header(example)
    note = (
        "#\n# Written by mtc init: one creator entry. Its token is in the "
        "environment\n# file, never here."
    )
    return f"{header}\n{note}\n{document.to_yaml()}"


def _ctm_persona_text(defaults: TypeDefaults) -> str:
    import yaml

    template = (TEMPLATES_DIR / "ctm_persona.yaml").read_text(encoding="utf-8")
    document = yaml.safe_load(template)
    document["system_prompt"] = (
        f"{document['system_prompt'].strip()} {defaults.ctm_tone}"
    )
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=78)
    return f"{_comment_header(template)}\n\n{body}"


def _persona_text(persona: PersonaConfig) -> str:
    header = (
        "# Persona written by mtc init. It holds the names you chose: keep it "
        "private.\n# Edit it by hand, or run mtc init --force to write it again.\n"
    )
    return f"{header}\n{persona.to_yaml()}"


def _store_password(
    store: str, choice: StoreChoice, manager: str
) -> Tuple[Optional[str], bool]:
    """``(value, generated)``: a typed password, a generated one or none.

    PostgreSQL and Neo4j always need one, so one is generated when none is
    typed. MongoDB gets a generated one only together with a user, and Redis
    only a typed one: both run without credentials on loopback, locally and
    in the shipped Compose file, and a password the server does not know
    would lock the client out.
    """
    del manager  # the rule is the same under every service manager
    if choice.password is not None:
        return choice.password.get_secret_value(), False
    generate = store in ("postgres", "neo4j") or (
        store == "mongodb" and choice.user is not None
    )
    return (secrets.token_hex(16), True) if generate else (None, False)


def _store_env(
    answers: OnboardingAnswers, updates: Dict[str, str], keep: Set[str]
) -> None:
    stores = answers.stores
    names = _derived_names(stores.db_prefix)
    updates[env_key("DB_PREFIX")] = stores.db_prefix
    updates[env_key("SERVICE_MANAGER")] = stores.service_manager
    for store in STORE_NAMES:
        choice: StoreChoice = getattr(stores, store)
        updates[env_key(_ENABLE_FIELDS[store])] = _flag(choice.enabled)
        if not choice.enabled:
            continue
        host, port = store_endpoint(store, choice)
        password, generated = _store_password(store, choice, stores.service_manager)
        if store == "neo4j":
            parts = urlsplit(str(_default("NEO4J_URI")))
            updates[env_key("NEO4J_URI")] = urlunsplit(
                (parts.scheme, f"{host}:{port}", parts.path, "", "")
            )
            updates[env_key("NEO4J_USER")] = choice.user or str(_default("NEO4J_USER"))
            updates[env_key("NEO4J_DATABASE")] = str(_default("NEO4J_DATABASE"))
        else:
            host_field, port_field = _HOST_PORT_FIELDS[store]
            updates[env_key(host_field)] = host
            updates[env_key(port_field)] = str(port)
        if store == "postgres":
            updates[env_key("POSTGRES_DB")] = names.POSTGRES_DB
            updates[env_key("POSTGRES_USER")] = choice.user or names.POSTGRES_USER
        elif store == "mongodb":
            updates[env_key("MONGODB_DB")] = names.MONGODB_DB
            if password is not None:
                updates[env_key("MONGODB_USER")] = choice.user or names.POSTGRES_USER
        elif store == "qdrant":
            updates[env_key("QDRANT_COLLECTION")] = names.QDRANT_COLLECTION
        elif store == "redis":
            updates[env_key("REDIS_KEY_PREFIX")] = names.REDIS_KEY_PREFIX
        if password is not None and store in _PASSWORD_FIELDS:
            key = env_key(_PASSWORD_FIELDS[store])
            updates[key] = password
            if generated:
                keep.add(key)
    updates[env_key("CTM_STORE")] = "mongo" if stores.mongodb.enabled else "memory"


def _brain_env(answers: OnboardingAnswers, updates: Dict[str, str]) -> None:
    brain = answers.brain
    updates[env_key("LLM_PROVIDER")] = brain.provider
    if brain.provider == "local":
        updates[env_key("LLM_HOST")] = brain.host or str(_default("LLM_HOST"))
        updates[env_key("LLM_PORT")] = str(brain.port or _default("LLM_PORT"))
        updates[env_key("LLM_BASE_URL")] = brain.base_url or ""
        updates[env_key("LLM_MODEL")] = brain.model
        updates[env_key("LLM_TEMPLATE_FAMILY")] = brain.template_family
    elif brain.provider == "anthropic":
        updates[env_key("ANTHROPIC_MODEL")] = brain.anthropic_model
        if brain.anthropic_api_key is not None:
            updates[env_key("ANTHROPIC_API_KEY")] = (
                brain.anthropic_api_key.get_secret_value()
            )


def plan_writes(
    answers: OnboardingAnswers,
    project_dir: Path,
    *,
    existing_env: Mapping[str, str],
) -> WritePlan:
    """Turn answers into the files and ``.env`` entries of one run.

    ``existing_env`` is the project's parsed ``.env`` (empty for a new
    project): its path keys decide where files go, the creator's token and
    other speakers' entries in it are kept, and a creator it names under
    another name is replaced. Nothing is written here.
    """
    answers = resolve_env_references(answers)
    root = Path(project_dir)
    cfg = _settings_for(root, existing_env)
    persona_answers = answers.persona
    defaults = load_type_defaults()[persona_answers.companion_type]
    persona = build_persona(answers)
    creator = persona_answers.creator_name

    ctm_dir = resolve_path(cfg.CTM_CONFIG_DIR, settings=cfg)
    files = [
        PlannedFile(
            resolve_path(cfg.PERSONA_PATH, settings=cfg),
            _persona_text(persona),
            PRIVATE_FILE_MODE,
        ),
        PlannedFile(
            speaker_trust.speakers_path(cfg),
            _speakers_text(creator, defaults, cfg),
            PRIVATE_FILE_MODE,
        ),
        PlannedFile(
            resolve_path(cfg.IDENTITY_STATEMENT_PATH, settings=cfg),
            (
                TEMPLATES_DIR
                / "identity"
                / f"{persona_answers.companion_type.value}.md"
            ).read_text(encoding="utf-8"),
            PRIVATE_FILE_MODE,
        ),
        PlannedFile(
            ctm_dir / "persona.yaml", _ctm_persona_text(defaults), PRIVATE_FILE_MODE
        ),
    ]
    pack_dir = ctm_dir / SEED_PACK_DIR
    planned_pack: Optional[Path] = None
    if defaults.seed_pack is not None:
        planned_pack = pack_dir / f"{SEED_PACK_PREFIX}{defaults.seed_pack}.yaml"
        files.append(
            PlannedFile(
                planned_pack,
                (TEMPLATES_DIR / "seeds" / f"{defaults.seed_pack}.yaml").read_text(
                    encoding="utf-8"
                ),
                PRIVATE_FILE_MODE,
            )
        )
    stale = []
    for source in sorted((TEMPLATES_DIR / "seeds").glob("*.yaml")):
        candidate = pack_dir / f"{SEED_PACK_PREFIX}{source.name}"
        if candidate != planned_pack and candidate.is_file():
            stale.append(candidate)

    updates: Dict[str, str] = {}
    keep: Set[str] = set()
    updates[env_key("SYSTEM_NAME")] = persona_answers.name
    updates[env_key("CREATOR_NAME")] = creator
    updates[env_key("CREATOR_RELATIONSHIP")] = defaults.creator_relationship
    updates[env_key("COMPANION_TYPE")] = persona_answers.companion_type.value
    updates[env_key("CORE_VALUES")] = ",".join(persona.core_values)
    updates[env_key("TIMEZONE")] = answers.locale.timezone
    updates[env_key("WAKE_HOUR")] = str(answers.locale.wake_hour)
    updates[env_key("SLEEP_HOUR")] = str(answers.locale.sleep_hour)
    _store_env(answers, updates, keep)
    _brain_env(answers, updates)
    updates[env_key("API_HOST")] = answers.api.host or str(_default("API_HOST"))
    updates[env_key("API_PORT")] = str(answers.api.port or _default("API_PORT"))
    updates[env_key("API_AUTH_ENABLED")] = _flag(True)
    api_key = env_key("API_KEY")
    updates[api_key] = secrets.token_hex(32)
    keep.add(api_key)
    previous = str(existing_env.get(env_key("CREATOR_NAME"), "")).strip()
    try:
        trust_entries = speaker_trust.env_entries(
            creator,
            secrets.token_hex(32),
            trusted=persona_answers.creator_trusted,
            existing_env=existing_env,
            previous_name=previous if previous and previous != creator else None,
            keep_existing=True,
        )
    except ValueError:
        # A creator name in the old file that cannot sit in a pair: there is
        # no old pair to replace.
        trust_entries = speaker_trust.env_entries(
            creator,
            secrets.token_hex(32),
            trusted=persona_answers.creator_trusted,
            existing_env=existing_env,
            keep_existing=True,
        )
    updates.update(trust_entries)
    return WritePlan(
        files=files,
        env_updates=updates,
        env_keep_existing=keep,
        env_path=root / ENV_FILE_NAME,
        project_dir=root,
        remove=stale,
        private_dirs=[resolve_path(cfg.AGENT_SPACE_DIR, settings=cfg)],
    )


# ---------------------------------------------------------------------------
# Applying the plan
# ---------------------------------------------------------------------------


def _write_private(path: Path, content: str, mode: int) -> None:
    """Write ``content`` atomically (a temporary file, then a rename) with
    ``mode``."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def apply_plan(plan: WritePlan, *, force: bool) -> List[Path]:
    """Write the plan; return every path written, ``.env`` last.

    Without ``force`` an existing planned file raises ``PersonaExists`` and
    nothing is written. Every ``.env`` value is rendered before the first
    write, so an unsafe value also leaves everything untouched.
    """
    existing = [
        plan._shown(planned.path) for planned in plan.files if planned.path.exists()
    ]
    if existing and not force:
        raise PersonaExists(existing)
    for value in plan.env_updates.values():
        render_value(value)
    for directory in plan.private_dirs:
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, PRIVATE_DIR_MODE)
    written: List[Path] = []
    for planned in plan.files:
        _write_private(planned.path, planned.content, planned.mode)
        written.append(planned.path)
    for stale in plan.remove:
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    plan.env_path.parent.mkdir(parents=True, exist_ok=True)
    keys = upsert_env(
        plan.env_path,
        plan.env_updates,
        keep_existing=plan.env_keep_existing,
        header=ENV_HEADER,
    )
    written.append(plan.env_path)
    logger.info(
        "setup wrote %d file(s) and %d environment entries",
        len(plan.files),
        len(keys),
    )
    return written


# ---------------------------------------------------------------------------
# Next steps and the entry point
# ---------------------------------------------------------------------------


def enabled_stores(answers: OnboardingAnswers) -> List[str]:
    """The enabled stores, in the answers model's order."""
    return [store for store in STORE_NAMES if getattr(answers.stores, store).enabled]


# The scripts and the Makefile ship in a repository checkout (and the sdist),
# not in a wheel; a step that names one says so.
CHECKOUT = "in a repository checkout"


def _mongodb_has_user(choice: StoreChoice) -> bool:
    """True when the wizard writes MONGODB_USER (and so a password)."""
    return choice.enabled and (
        choice.user is not None
        or choice.password is not None
        or choice.password_env is not None
    )


def next_steps(answers: OnboardingAnswers, project_dir: Path) -> str:
    """What to run after setup, for these answers. No secret is shown."""
    del project_dir  # the steps are the same wherever the project lives
    lines = ["Next steps:"]
    provider = answers.brain.provider
    if provider == "local":
        lines.append(
            "  - Start the model server: set LLM_BACKEND and MODEL_PATH in the "
            f"environment file, then run scripts/ops/start_llm.sh ({CHECKOUT}), "
            "or start your own chat-completions server."
        )
    elif provider == "anthropic":
        lines.append(
            "  - The Anthropic provider sends each prompt to an external "
            "service. Its key belongs in the environment file as "
            f"{env_key('ANTHROPIC_API_KEY')}."
        )
    else:
        lines.append(
            "  - The mock provider is active: replies are labelled test text. "
            "Run mtc init --force to choose a model later."
        )
    stores = enabled_stores(answers)
    manager = answers.stores.service_manager
    if stores:
        services = " ".join(COMPOSE_SERVICES[store] for store in stores)
        if manager == "docker":
            lines.append(
                f'  - Start the stores ({CHECKOUT}): make db-up DB_SERVICES="{services}"'
            )
        elif manager == "brew":
            lines.append(
                f"  - Start the stores ({CHECKOUT}): scripts/ops/start_stack.sh "
                "--no-brain --no-api (it uses brew services); otherwise start "
                "each with brew services start."
            )
        elif manager == "systemd":
            lines.append(
                f"  - Start the stores ({CHECKOUT}): scripts/ops/start_stack.sh "
                "--no-brain --no-api (it uses systemctl); otherwise start each "
                "with systemctl start."
            )
        else:
            lines.append(
                "  - Start the stores yourself; scripts/ops/start_stack.sh "
                f"--dry-run ({CHECKOUT}) prints what it checks."
            )
        if "postgres" in stores and manager != "docker":
            lines.append(
                "  - Create the PostgreSQL role and database once: "
                f"scripts/db/init_postgres.sh ({CHECKOUT}); otherwise create, "
                "as the PostgreSQL administrator, a login role named by "
                "MTC_POSTGRES_USER with the MTC_POSTGRES_PASSWORD value and a "
                "database named by MTC_POSTGRES_DB that it owns."
            )
        if _mongodb_has_user(answers.stores.mongodb):
            line = (
                "  - MongoDB with a user: nothing creates the MongoDB user. "
                "Create it once in mongosh, connected as an administrator to "
                "the database MTC_MONGODB_AUTH_SOURCE names (MTC_MONGODB_DB "
                "when it is empty): db.createUser({user: the MTC_MONGODB_USER "
                "value, pwd: passwordPrompt(), roles: [{role: 'readWrite', db: "
                "the MTC_MONGODB_DB value}]}), and paste the MTC_MONGODB_PASSWORD "
                "value from the environment file at the prompt."
            )
            if manager == "docker":
                line += (
                    " The shipped Compose MongoDB service runs without "
                    "authentication: create the user there as well, or leave "
                    "the user empty with mtc init --force."
                )
            lines.append(line)
        if not answers.bootstrap_db:
            lines.append("  - Prepare the stores: mtc db init")
    elif not answers.bootstrap_db:
        lines.append("  - Record the install date: mtc db init (no store is needed).")
    lines.extend(
        [
            "  - Check the setup: mtc doctor",
            "  - Start the API server: mtc serve",
            "  - Talk: mtc chat",
            "  - Measure: mtc assess --quick",
        ]
    )
    return "\n".join(lines)


def _bootstrap(settings: Settings, env_path: Path) -> Tuple[Any, int]:
    """Run the database bootstrap for the freshly written project.

    Returns its result and exit code (a missing store client reads as a
    usage error, as it does for ``mtc db init``).
    """
    import asyncio

    from mtc.db import bootstrap as db_bootstrap

    result = asyncio.run(db_bootstrap.bootstrap(settings, env_path=env_path))
    rule = getattr(db_bootstrap, "command_exit_code", exit_code_for)
    return result, int(rule(list(getattr(result, "results", []))))


def run_init(
    *,
    from_path: Optional[Path],
    yes: bool,
    force: bool,
    bootstrap_db: bool,
    project_dir: Path,
    probe: bool,
    print_plan: bool,
    prompter: Optional[Prompter] = None,
) -> int:
    """Run the wizard; return an exit code (0 done, 2 refused or invalid)."""
    root = Path(project_dir)
    interactive = from_path is None
    voice: Prompter = prompter if prompter is not None else ConsolePrompter()
    if interactive and yes:
        voice.say("--yes needs --from: a run without questions needs an answers file.")
        return EXIT_USAGE
    if interactive and prompter is None and not sys.stdin.isatty():
        voice.say(
            "mtc init needs a terminal for its questions. Without one, pass "
            "--from answers.yaml --yes (mtc init --print-example-answers "
            "prints a sample)."
        )
        return EXIT_USAGE
    try:
        if interactive:
            answers = collect_interactive(voice, probe=probe)
        else:
            answers = load_answers(Path(from_path))
    except AnswersError as exc:
        voice.say("The answers cannot be used:")
        for line in exc.errors:
            voice.say(f"  - {line}")
        return EXIT_USAGE
    except (NotAcknowledged, InputEnded) as exc:
        voice.say(f"Setup stopped: {exc}.")
        return EXIT_USAGE
    if bootstrap_db:
        answers = answers.model_copy(update={"bootstrap_db": True})

    env_path = root / ENV_FILE_NAME
    try:
        plan = plan_writes(answers, root, existing_env=read_env(env_path))
    except AnswersError as exc:
        voice.say("The answers cannot be used:")
        for line in exc.errors:
            voice.say(f"  - {line}")
        return EXIT_USAGE
    except ValidationError as exc:
        voice.say("The existing environment file does not load:")
        for line in _error_lines(exc):
            voice.say(f"  - {line}")
        return EXIT_USAGE

    if interactive:
        _step(voice, 9)
    voice.say(plan.describe())
    if print_plan:
        voice.say("Nothing was written (--print-plan).")
        return EXIT_OK
    if not yes:
        try:
            confirmed = voice.ask_bool("Write these files", default=True)
        except InputEnded as exc:
            voice.say(f"Setup stopped: {exc}.")
            return EXIT_USAGE
        if not confirmed:
            voice.say("Nothing was written.")
            return EXIT_USAGE
    try:
        written = apply_plan(plan, force=force)
    except PersonaExists as exc:
        voice.say(f"Refused: {exc}. Nothing was written.")
        return EXIT_USAGE
    except ValueError as exc:
        voice.say(f"Refused: {exc}. Nothing was written.")
        return EXIT_USAGE
    voice.say(
        f"Wrote {len(written)} files. Secrets are in {ENV_FILE_NAME} only "
        f"(mode {PRIVATE_FILE_MODE:04o}); none was printed."
    )
    if answers.brain.provider == "anthropic" and not (
        answers.brain.anthropic_api_key is not None
        or read_env(env_path).get(env_key("ANTHROPIC_API_KEY"))
    ):
        voice.say(
            f"The Anthropic key is not set yet: add {env_key('ANTHROPIC_API_KEY')} "
            "to the environment file before mtc serve."
        )

    code = EXIT_OK
    if answers.bootstrap_db:
        if interactive:
            _step(voice, 10)
        settings = Settings(_env_file=str(env_path), PROJECT_ROOT=str(root))
        try:
            result, code = _bootstrap(settings, env_path)
        except ImportError as exc:
            voice.say(
                "The files were written, but the bootstrap could not start: a "
                f"module is missing ({exc.name or 'unknown'}). Install the "
                "matching extra, then run mtc db init."
            )
            return EXIT_USAGE
        except Exception as exc:  # noqa: BLE001 - the message may hold a DSN
            logger.error("bootstrap failed: %s", type(exc).__name__)
            voice.say(
                "The files were written, but the bootstrap failed "
                f"({type(exc).__name__}). Fix it, then run mtc db init."
            )
            return EXIT_FAILED
        results = list(getattr(result, "results", []))
        voice.say(render_text(results, title="Bootstrap"))
        birth_date = getattr(result, "birth_date", None)
        if birth_date:
            voice.say(f"Install date: {birth_date}")
        if code != EXIT_OK:
            voice.say(
                "The files were written, but the bootstrap reported a problem. "
                "Fix it, then run mtc db init."
            )
    if interactive:
        _step(voice, 11)
    voice.say(next_steps(answers, root))
    return code
