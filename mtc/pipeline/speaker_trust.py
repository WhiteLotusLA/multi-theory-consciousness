"""
Speaker Trust
=============

The single source of speaker trust for the runtime pipeline. A speaker is
authenticated only when the token they present matches the configured token
for their entry, compared with ``secrets.compare_digest``. A name on its own
grants nothing: an unknown or unauthenticated speaker receives
``DEFAULT_SPEAKER_TRUST``, no activation bonus, no safety bypass and the
default emotion trust, whatever they are called.

A name no token backs is reduced by ``claimed_name`` to one safe line before
it becomes the identity's name: letters, digits, space, period, apostrophe,
hyphen and underscore, at most ``MAX_CLAIMED_NAME_CHARS`` characters; a claim
with nothing usable left is the anonymous default speaker. The entry lookup
and the token comparison use the presented name as given (stripped), and an
authenticated identity keeps its configured entry name.

Single-source rule: this module is the only reader of ``SPEAKERS_PATH``,
``SPEAKER_TOKENS``, ``TRUSTED_SPEAKERS``, ``TRUST_ACTIVATION_BONUS``,
``TRUSTED_EMOTION_TRUST`` and ``DEFAULT_EMOTION_TRUST`` (a tree-wide test pins
it). Every other module derives trust from the ``SpeakerIdentity`` returned by
``SpeakerTrust.verify`` through the helpers here, which refuse a bare string.
``DEFAULT_SPEAKER`` names the anonymous speaker and ``DEFAULT_SPEAKER_TRUST``
is the trust everyone starts from.

Tokens come from the environment variable an entry's ``token_env`` names,
read at call time, else from the entry's ``SPEAKER_TOKENS`` pair; an entry
with neither never authenticates and an empty presented token is never
compared. Speaker names and tokens are private at runtime: ``status()`` reports
counts by role only, nothing here is logged with a name or a token, and no
state is persisted.

Three helpers serve local tools so that no other module names the trust keys:
``speakers_path`` resolves the speakers file, ``token_for`` returns the token
a local client presents for a configured speaker (it authenticates nobody;
the server still compares what is presented), and ``env_entries`` renders the
two environment entries a setup tool writes for the creator.
"""

from __future__ import annotations

import contextvars
import logging
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from mtc.core.config import Settings, csv_list, get_settings
from mtc.core.paths import resolve_path

logger = logging.getLogger(__name__)

ROLES = ("creator", "companion", "guest")

# The role label the safety and privacy layers see for a speaker who is not
# authenticated; it is one of the lowercase labels those layers recognise.
UNAUTHENTICATED_SAFETY_ROLE = "guest"

_SESSION_AUTHENTICATED = "verified"
_SESSION_CLAIMED = "claimed"

_ENTRY_FIELDS = ("name", "role", "trust", "token_env")

# The longest name an unauthenticated speaker keeps, after ``claimed_name``.
MAX_CLAIMED_NAME_CHARS = 64
# Characters a claimed name keeps besides letters and digits.
_CLAIMED_NAME_EXTRA = frozenset(" .'-_")


def claimed_name(raw: Optional[str]) -> str:
    """A claimed speaker name as one safe line.

    Every character that is not a letter or digit (``str.isalnum``) and not
    one of space, period, apostrophe, hyphen and underscore becomes a space;
    runs of whitespace collapse to one space; the result is stripped and cut
    to ``MAX_CLAIMED_NAME_CHARS``. An empty result means nothing usable was
    given.
    """
    text = "".join(
        char if char.isalnum() or char in _CLAIMED_NAME_EXTRA else " "
        for char in str(raw or "")
    )
    return " ".join(text.split())[:MAX_CLAIMED_NAME_CHARS].strip()


@dataclass(frozen=True)
class SpeakerEntry:
    """One configured speaker: a name, a role, the trust granted on
    authentication and the environment variable that holds the token."""

    name: str
    role: str
    trust: float
    token_env: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a speaker entry needs a non-empty name")
        if self.role not in ROLES:
            # The value is not echoed: a misplaced name or token may sit there
            raise ValueError(f"speaker role must be one of {ROLES}")
        try:
            trust = float(self.trust)
        except (TypeError, ValueError) as exc:
            raise ValueError("speaker trust must be a number in [0, 1]") from exc
        if not 0.0 <= trust <= 1.0 or trust != trust:
            raise ValueError(f"speaker trust must be in [0, 1], got {self.trust!r}")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "trust", trust)
        if self.token_env is not None and not str(self.token_env).strip():
            object.__setattr__(self, "token_env", None)


@dataclass(frozen=True)
class SpeakerIdentity:
    """The verified outcome for one speaker on one turn.

    ``role`` and the configured ``trust`` are set only when ``authenticated``;
    ``trusted`` adds the allowlist check; ``is_default`` marks the anonymous
    speaker who gave no name.
    """

    name: str
    role: Optional[str]
    trust: float
    authenticated: bool
    trusted: bool
    is_default: bool

    def view(self) -> Any:
        """The prompt assembler's ``SpeakerView`` for this identity."""
        from mtc.brain.prompt import SpeakerView

        if self.is_default:
            label = "anonymous"
        elif self.authenticated and self.role == "creator":
            label = ""
        elif self.authenticated:
            label = "verified"
        else:
            label = "unverified"
        return SpeakerView(
            name=self.name,
            role=self.role if self.authenticated else None,
            authenticated=bool(self.authenticated),
            label=label,
        )


def _require_identity(identity: Any) -> SpeakerIdentity:
    if not isinstance(identity, SpeakerIdentity):
        raise TypeError(
            "trust helpers take a SpeakerIdentity from SpeakerTrust.verify, "
            f"not {type(identity).__name__}"
        )
    return identity


def _parse_entries(document: Any) -> List[SpeakerEntry]:
    if document is None:
        return []
    if not isinstance(document, dict):
        raise ValueError("the speakers file must be a mapping with a speakers list")
    raw = document.get("speakers")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("speakers must be a list of entries")
    entries: List[SpeakerEntry] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"speakers[{index}] must be a mapping")
        unknown = set(item) - set(_ENTRY_FIELDS)
        if unknown:
            # Counted, not named: a misplaced key may be a name or a token
            raise ValueError(
                f"speakers[{index}] has {len(unknown)} unknown field(s); the "
                f"allowed fields are {', '.join(_ENTRY_FIELDS)}"
            )
        if "role" not in item:
            raise ValueError(f"speakers[{index}] needs a role from {ROLES}")
        if "trust" not in item:
            # No default: a missing trust is a configuration error, never a
            # value the parser invents for an authenticated speaker.
            raise ValueError(f"speakers[{index}] needs a trust from 0.0 to 1.0")
        entries.append(
            SpeakerEntry(
                name=str(item.get("name", "")),
                role=item.get("role"),
                trust=item.get("trust"),
                token_env=item.get("token_env"),
            )
        )
    return entries


class SpeakerTrust:
    """Token-checked speaker trust built from the configured entries.

    Args:
        entries: The speaker table; duplicate names raise ``ValueError``.
        settings: Settings to read; None reads ``get_settings()`` at call time.
    """

    def __init__(
        self, entries: Sequence[SpeakerEntry] = (), settings: Optional[Settings] = None
    ) -> None:
        table: Dict[str, SpeakerEntry] = {}
        for entry in entries:
            if not isinstance(entry, SpeakerEntry):
                raise TypeError("entries must be SpeakerEntry instances")
            if entry.name in table:
                raise ValueError("duplicate speaker entry")
            table[entry.name] = entry
        self._entries = table
        self._settings = settings
        self._current: contextvars.ContextVar[Optional[SpeakerIdentity]] = (
            contextvars.ContextVar("mtc_speaker_identity", default=None)
        )
        self._verifications = 0
        self._authenticated = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "SpeakerTrust":
        """Load the table at ``SPEAKERS_PATH``; a missing file is an empty table.

        PyYAML is imported here, not at module import. A malformed file raises
        ``ValueError`` naming the problem (never a name or a token): a parse
        error names the file and the line only, because the parser's own
        message quotes the offending line, and that message is neither chained
        nor logged.
        """
        cfg = settings if settings is not None else get_settings()
        path = resolve_path(cfg.SPEAKERS_PATH, settings=cfg)
        if not path.is_file():
            logger.info("speaker table not found; every speaker gets the default")
            return cls((), settings=settings)
        import yaml

        problem: Optional[str] = None
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None) or getattr(
                exc, "context_mark", None
            )
            line = getattr(mark, "line", None)
            where = f" near line {line + 1}" if isinstance(line, int) else ""
            problem = f"is not valid YAML{where}"
        except UnicodeDecodeError:
            problem = "is not valid UTF-8"
        if problem is not None:
            message = f"speakers file {path.name} (SPEAKERS_PATH) {problem}"
            logger.error("%s", message)
            raise ValueError(message) from None
        entries = _parse_entries(document)
        logger.info("speaker table loaded with %d entries", len(entries))
        return cls(entries, settings=settings)

    def _cfg(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _configured_token(self, entry: SpeakerEntry) -> Optional[bytes]:
        """The token configured for an entry, read now; None when there is none."""
        if entry.token_env:
            value = os.environ.get(entry.token_env, "")
            return value.encode("utf-8") if value else None
        for pair in csv_list(self._cfg().SPEAKER_TOKENS):
            if ":" not in pair:
                continue
            pair_name, secret_value = pair.split(":", 1)
            if pair_name.strip() != entry.name:
                continue
            return secret_value.encode("utf-8") if secret_value else None
        return None

    def verify(self, name: Optional[str], token: Optional[str]) -> SpeakerIdentity:
        """Resolve a presented name and token into a ``SpeakerIdentity``.

        An empty name is the anonymous default speaker. An unknown name, a
        missing or empty token, an entry without a configured token and a
        mismatch all yield the default trust and no role, under the name
        ``claimed_name`` makes of the presented one (the anonymous default
        when nothing usable is left). The entry lookup and the token
        comparison use the presented name as given, stripped.
        """
        cfg = self._cfg()
        self._verifications += 1
        default_trust = float(cfg.DEFAULT_SPEAKER_TRUST)
        given = (name or "").strip()
        anonymous = SpeakerIdentity(
            name=cfg.DEFAULT_SPEAKER,
            role=None,
            trust=default_trust,
            authenticated=False,
            trusted=False,
            is_default=True,
        )
        if not given:
            return anonymous
        entry = self._entries.get(given)
        authenticated = False
        if entry is not None and isinstance(token, str) and token:
            expected = self._configured_token(entry)
            if expected is not None:
                authenticated = secrets.compare_digest(token.encode("utf-8"), expected)
        if not authenticated or entry is None:
            safe = claimed_name(given)
            if not safe:
                return anonymous
            return SpeakerIdentity(
                name=safe,
                role=None,
                trust=default_trust,
                authenticated=False,
                trusted=False,
                is_default=False,
            )
        self._authenticated += 1
        allowlisted = given in set(csv_list(cfg.TRUSTED_SPEAKERS))
        return SpeakerIdentity(
            name=given,
            role=entry.role,
            trust=float(entry.trust),
            authenticated=True,
            trusted=allowlisted,
            is_default=False,
        )

    def is_trusted(self, name: Optional[str], token: Optional[str]) -> bool:
        """True only for an authenticated speaker on the ``TRUSTED_SPEAKERS`` list."""
        return self.verify(name, token).trusted

    # ------------------------------------------------------------------
    # Derived values (every one takes a SpeakerIdentity)
    # ------------------------------------------------------------------

    def trust_bonus(self, identity: SpeakerIdentity) -> float:
        """Workspace activation bonus: ``TRUST_ACTIVATION_BONUS`` when trusted."""
        identity = _require_identity(identity)
        if not identity.trusted:
            return 0.0
        return float(self._cfg().TRUST_ACTIVATION_BONUS)

    def emotion_trust(self, identity: SpeakerIdentity) -> float:
        """Emotion trust weight for the post-turn record."""
        identity = _require_identity(identity)
        cfg = self._cfg()
        if identity.trusted:
            return float(cfg.TRUSTED_EMOTION_TRUST)
        return float(cfg.DEFAULT_EMOTION_TRUST)

    def label_for(self, identity: SpeakerIdentity) -> str:
        """Bracketed name for history lines; empty for the anonymous default
        speaker and for the authenticated creator."""
        identity = _require_identity(identity)
        if identity.is_default:
            return ""
        if identity.authenticated and identity.role == "creator":
            return ""
        return f"[{identity.name}]"

    def safety_role(self, identity: SpeakerIdentity) -> str:
        """Role label for the safety and privacy layers: the configured role
        once authenticated, ``guest`` otherwise. Every role in ``ROLES`` is a
        label of the privacy layer's ``UserRole``, so an authenticated
        speaker is never recorded as an unrecognized role."""
        identity = _require_identity(identity)
        if identity.authenticated and identity.role:
            return identity.role
        return UNAUTHENTICATED_SAFETY_ROLE

    def session_key(self, identity: SpeakerIdentity) -> str:
        """Key for the speaker's session; an authenticated speaker never shares
        a session with someone who merely claims the name."""
        identity = _require_identity(identity)
        kind = _SESSION_AUTHENTICATED if identity.authenticated else _SESSION_CLAIMED
        if identity.is_default:
            kind = "anonymous"
        return f"{kind}:{identity.name}"

    # ------------------------------------------------------------------
    # Current identity (for name-keyed lookups such as theory of mind)
    # ------------------------------------------------------------------

    @contextmanager
    def use_identity(self, identity: SpeakerIdentity) -> Iterator[SpeakerIdentity]:
        """Make ``identity`` the current one for ``trust_lookup`` inside the block."""
        identity = _require_identity(identity)
        token = self._current.set(identity)
        try:
            yield identity
        finally:
            self._current.reset(token)

    def current_identity(self) -> Optional[SpeakerIdentity]:
        """The identity set by ``use_identity`` in this context, if any."""
        return self._current.get()

    def trust_lookup(self, name: str) -> float:
        """Trust for a name-keyed consumer: the current authenticated identity's
        trust when the name is that identity's, else ``DEFAULT_SPEAKER_TRUST``."""
        current = self._current.get()
        if (
            current is not None
            and current.authenticated
            and isinstance(name, str)
            and name.strip() == current.name
        ):
            return float(current.trust)
        return float(self._cfg().DEFAULT_SPEAKER_TRUST)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Counts only: entries by role, token sources and verifications."""
        by_role = {role: 0 for role in ROLES}
        with_env = 0
        for entry in self._entries.values():
            by_role[entry.role] += 1
            if entry.token_env:
                with_env += 1
        return {
            "entries": len(self._entries),
            "by_role": by_role,
            "with_token_env": with_env,
            "allowlisted": len(csv_list(self._cfg().TRUSTED_SPEAKERS)),
            "verifications": self._verifications,
            "authenticated": self._authenticated,
        }


# ----------------------------------------------------------------------
# Helpers for local tools (the setup wizard and the command-line client)
# ----------------------------------------------------------------------

_PAIR_BREAKERS = (",", ":", "\n", "\r")


def _env_key(field: str) -> str:
    """The environment variable that carries a Settings field."""
    return f"{Settings.model_config.get('env_prefix', '')}{field}"


def _pair_name(value: Any, what: str) -> str:
    """A speaker name that can sit in a ``name:token`` pair, stripped.

    The value is not echoed: a misplaced token may sit there.
    """
    name = value.strip() if isinstance(value, str) else ""
    if not name:
        raise ValueError(f"{what} must be a non-empty name")
    if any(mark in name for mark in _PAIR_BREAKERS):
        raise ValueError(f"{what} must not contain a comma, a colon or a line break")
    return name


def speakers_path(settings: Optional[Settings] = None) -> Path:
    """The speakers file ``SPEAKERS_PATH`` names, resolved; it may not exist."""
    cfg = settings if settings is not None else get_settings()
    return resolve_path(cfg.SPEAKERS_PATH, settings=cfg)


def token_for(name: str, settings: Optional[Settings] = None) -> Optional[str]:
    """The token a local client presents for a configured speaker, or None.

    The entry's ``token_env`` variable when the entry names one (read now;
    None while it is not exported), else the entry's ``SPEAKER_TOKENS`` pair.
    A name without an entry has no token: the trust source would not
    authenticate it either. The value is never logged and authenticates
    nobody here; the server compares what the client presents.
    """
    cfg = settings if settings is not None else get_settings()
    given = name.strip() if isinstance(name, str) else ""
    if not given:
        return None
    trust = SpeakerTrust.from_settings(cfg)
    entry = trust._entries.get(given)
    if entry is None:
        return None
    configured = trust._configured_token(entry)
    return configured.decode("utf-8") if configured else None


def env_entries(
    name: str,
    token: str,
    *,
    trusted: bool,
    existing_tokens: str = "",
    existing_trusted: str = "",
    previous_name: Optional[str] = None,
    existing_env: Optional[Mapping[str, str]] = None,
    keep_existing: bool = False,
) -> Dict[str, str]:
    """The two environment entries a setup tool writes for one speaker.

    Returns the ``SPEAKER_TOKENS`` and ``TRUSTED_SPEAKERS`` values under their
    environment variable names, tokens first. The speaker's ``name:token``
    pair leads; every other speaker's pair and allowlist entry is kept as
    written; a pair or allowlist entry under ``previous_name`` (the speaker
    was renamed) is dropped. With ``trusted`` false the name is left off the
    allowlist. ``existing_env`` is a parsed environment file that supplies
    the two existing values when they are not passed as strings. With
    ``keep_existing`` a token already paired with the same name is kept and
    ``token`` is used only when there is none; a renamed speaker always gets
    ``token``. A name with a comma, a colon or a line break and a token that
    is empty or holds a comma or whitespace raise ``ValueError`` without
    echoing the value. Nothing is logged.
    """
    speaker = _pair_name(name, "the speaker name")
    replaced = {speaker}
    if previous_name is not None and str(previous_name).strip():
        replaced.add(_pair_name(previous_name, "the previous speaker name"))
    given = token if isinstance(token, str) else ""
    if not given or any(mark == "," or mark.isspace() for mark in given):
        raise ValueError(
            "a speaker token must be non-empty with no comma and no whitespace"
        )

    tokens_key = _env_key("SPEAKER_TOKENS")
    trusted_key = _env_key("TRUSTED_SPEAKERS")
    if existing_env is not None:
        existing_tokens = existing_tokens or str(existing_env.get(tokens_key, ""))
        existing_trusted = existing_trusted or str(existing_env.get(trusted_key, ""))

    chosen = given
    others: List[str] = []
    for pair in csv_list(existing_tokens):
        pair_name, mark, pair_token = pair.partition(":")
        if not mark or pair_name.strip() not in replaced:
            others.append(pair)
        elif keep_existing and pair_name.strip() == speaker and pair_token.strip():
            chosen = pair_token.strip()
    allowlist = [item for item in csv_list(existing_trusted) if item not in replaced]
    if trusted:
        allowlist.insert(0, speaker)
    return {
        tokens_key: ",".join([f"{speaker}:{chosen}"] + others),
        trusted_key: ",".join(allowlist),
    }
