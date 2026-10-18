"""
Access Control Enforcer
=======================

Decides who may read which privacy tier (``DataClassification``). A decision
combines the requester's role, the access stage (``access_stage`` in
``mtc.privacy.data_classifier``), the tier and any registered consent.

Roles are labels, never names: ``SUBJECT`` (the system itself, which always
has full access to its own data), ``CREATOR``, ``GUARDIAN``, ``COMPANION`` (an
authenticated companion speaker, the speaker-trust role of that name),
``SYSTEM`` (the framework's own processes), ``RESEARCHER`` and ``GUEST``.
``UserRole.from_string`` accepts exactly those lowercase labels; any other
string, a person's name included, raises ValueError, so no name maps to a
role. Every speaker-trust role (creator, companion, guest) is one of them.

The policy is data (``AccessPolicy``): one row per access stage giving each
role's highest readable tier (None for no access), the stages at which consent
can unlock a higher tier and at which access above ``PUBLIC`` is audited, a
stage-independent row and the emergency rules. ``AccessPolicy.default()``
encodes a guardian-style policy shape, in which a creator sees everything at
first and less as the stages advance while a researcher needs consent; it is
a documented default meant to be replaced by an operator's own policy.

Default matrix (highest readable tier; "none" is no access; stages 3 to 6 are
consent stages; the companion column is explicit and equals the guardian's):

    stage  creator       guardian      companion     system  researcher  guest
    0      core          core          core          core    none        none
    1      relationship  relationship  relationship  core    none        public
    2      relationship  public        public        core    none        public
    3      relationship  public        public        core    public      public
    4      public        public        public        core    public      public
    5      public        public        public        core    public      public
    6      public        public        public        core    public      public

The subject reads core at every stage.

Which row applies:

- The developmental machinery on (``developmental_enabled``: the ``child``
  companion type or ``DEVELOPMENTAL_STAGE_ENABLED``): the row of the current
  stage. An unknown stage (no ``SYSTEM_BIRTH_DATE``) takes the most protective
  row, the per-role minimum over every stage, and consent unlocks nothing.
- The machinery off: the stage-independent row (by default the final stage's
  row, with consent), so role limits apply and the calendar changes nothing.

Consent is registered per role and tier and covers that tier and the ones
below it; it can carry an expiry. Consents live in memory, and in a JSON file
only when ``consent_path`` is given (written on the first change, never
earlier). The file names who granted each consent and is private.

Emergency override: only the roles in ``AccessPolicy.emergency_roles``
(creator and system by default), with a reason of at least ten characters,
and at most once per ten minutes across all roles (``clock`` is injectable).

Config keys read: ``SYSTEM_BIRTH_DATE``, ``DEVELOPMENTAL_STAGE_ENABLED`` and
``COMPANION_TYPE`` (through ``developmental_enabled``), at call time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from mtc.consciousness.developmental_modulator import developmental_enabled
from mtc.core.config import Settings, get_settings
from mtc.core.paths import display_path
from mtc.privacy.data_classifier import DataClassification, StagePolicy, access_stage

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
Row = Mapping["UserRole", Optional[DataClassification]]


class UserRole(Enum):
    """Requester roles; labels only, never names."""

    SUBJECT = "subject"
    CREATOR = "creator"
    GUARDIAN = "guardian"
    COMPANION = "companion"
    SYSTEM = "system"
    RESEARCHER = "researcher"
    GUEST = "guest"

    @classmethod
    def from_string(cls, value: str) -> UserRole:
        """Map an exact lowercase role label to its role.

        Surrounding whitespace is ignored; anything else that is not one of the
        labels (a capitalised label or a person's name included) raises
        ValueError.
        """
        label = str(value).strip()
        for role in cls:
            if role.value == label:
                return role
        raise ValueError(f"unknown role label {label!r}")


class AccessOutcome(Enum):
    """What a decision allows."""

    ALLOW = "allow"
    ALLOW_WITH_AUDIT = "allow_with_audit"
    DENY = "deny"
    EMERGENCY_OVERRIDE = "emergency_override"


@dataclass
class AccessContext:
    """One access request: who, which tier, why."""

    role: UserRole
    user_id: str
    classification: DataClassification
    purpose: str = ""
    emergency: bool = False
    emergency_reason: Optional[str] = None


@dataclass
class AccessDecision:
    """The verdict on one request.

    ``basis`` names the row that decided: ``subject``, ``stage``,
    ``most_protective``, ``stage_independent`` or ``emergency``. ``stage`` is
    the access stage when the stage rows were in force, else None.
    """

    outcome: AccessOutcome
    reason: str
    role: UserRole
    classification: Optional[DataClassification] = None
    stage: Optional[int] = None
    basis: str = ""

    @property
    def allowed(self) -> bool:
        return self.outcome is not AccessOutcome.DENY


_C = DataClassification
_R = UserRole


def _default_rows() -> Tuple[Dict[UserRole, Optional[DataClassification]], ...]:
    def row(creator, guardian, companion, researcher, guest):
        return {
            _R.CREATOR: creator,
            _R.GUARDIAN: guardian,
            _R.COMPANION: companion,
            _R.SYSTEM: _C.CORE,
            _R.RESEARCHER: researcher,
            _R.GUEST: guest,
            _R.SUBJECT: _C.CORE,
        }

    # The companion column is explicit and equals the guardian's: an
    # authenticated companion speaker reads what a guardian reads.
    return (
        row(_C.CORE, _C.CORE, _C.CORE, None, None),
        row(_C.RELATIONSHIP, _C.RELATIONSHIP, _C.RELATIONSHIP, None, _C.PUBLIC),
        row(_C.RELATIONSHIP, _C.PUBLIC, _C.PUBLIC, None, _C.PUBLIC),
        row(_C.RELATIONSHIP, _C.PUBLIC, _C.PUBLIC, _C.PUBLIC, _C.PUBLIC),
        row(_C.PUBLIC, _C.PUBLIC, _C.PUBLIC, _C.PUBLIC, _C.PUBLIC),
        row(_C.PUBLIC, _C.PUBLIC, _C.PUBLIC, _C.PUBLIC, _C.PUBLIC),
        row(_C.PUBLIC, _C.PUBLIC, _C.PUBLIC, _C.PUBLIC, _C.PUBLIC),
    )


def _freeze_row(row: Row) -> Dict[UserRole, Optional[DataClassification]]:
    frozen: Dict[UserRole, Optional[DataClassification]] = {}
    for role in UserRole:
        value = row.get(role)
        frozen[role] = None if value is None else DataClassification.parse(value)
    return frozen


def _rank(value: Optional[DataClassification]) -> int:
    return -1 if value is None else value.rank


@dataclass(frozen=True)
class AccessPolicy:
    """The access matrix and its rules as data (see the module docstring)."""

    rows: Tuple[Row, ...]
    consent_stages: FrozenSet[int]
    stage_independent: Row
    stage_independent_consent: bool = True
    emergency_roles: FrozenSet[UserRole] = field(
        default_factory=lambda: frozenset({UserRole.CREATOR, UserRole.SYSTEM})
    )
    emergency_min_reason: int = 10
    emergency_cooldown_s: float = 600.0

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("an access policy needs at least one stage row")
        object.__setattr__(self, "rows", tuple(_freeze_row(row) for row in self.rows))
        object.__setattr__(
            self, "stage_independent", _freeze_row(self.stage_independent)
        )
        object.__setattr__(
            self, "consent_stages", frozenset(int(s) for s in self.consent_stages)
        )
        object.__setattr__(self, "emergency_roles", frozenset(self.emergency_roles))

    @classmethod
    def default(cls) -> AccessPolicy:
        """The documented default policy (a guardian-style shape)."""
        rows = _default_rows()
        return cls(
            rows=rows,
            consent_stages=frozenset({3, 4, 5, 6}),
            stage_independent=dict(rows[-1]),
        )

    def row_for_stage(self, stage: int) -> Dict[UserRole, Optional[DataClassification]]:
        """The row of ``stage``; a stage past the last row takes the last row."""
        return dict(self.rows[min(max(stage, 0), len(self.rows) - 1)])

    def most_protective_row(self) -> Dict[UserRole, Optional[DataClassification]]:
        """Per role, the lowest tier any stage row grants (None below public)."""
        lowest: Dict[UserRole, Optional[DataClassification]] = {}
        for role in UserRole:
            values = [row.get(role) for row in self.rows]
            lowest[role] = min(values, key=_rank)
        return lowest


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc)


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


class AccessControlEnforcer:
    """Stage-, role- and consent-aware access decisions."""

    def __init__(
        self,
        policy: Optional[AccessPolicy] = None,
        consent_path: Optional[Path] = None,
        settings: Optional[Settings] = None,
        clock: Optional[Clock] = None,
        *,
        stage_policy: Optional[StagePolicy] = None,
    ) -> None:
        self.policy = policy if policy is not None else AccessPolicy.default()
        self._consent_path = Path(consent_path) if consent_path is not None else None
        self._settings = settings
        self._clock = clock or _system_clock
        self._stage_policy = stage_policy or StagePolicy()
        self._last_override: Optional[datetime] = None
        self._consents: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._load_consents()

    # ------------------------------------------------------------ state

    def _active_settings(self) -> Settings:
        return self._settings if self._settings is not None else get_settings()

    def _now(self) -> datetime:
        return _utc(self._clock())

    def now(self) -> datetime:
        """The enforcer's clock, as an aware UTC instant."""
        return self._now()

    @property
    def settings(self) -> Settings:
        """The Settings in force (the passed ones, else ``get_settings()``)."""
        return self._active_settings()

    @property
    def stage_policy(self) -> StagePolicy:
        return self._stage_policy

    @property
    def enabled(self) -> bool:
        """True when the stage rows are in force (read at call time)."""
        return developmental_enabled(self._active_settings())

    def current_stage(self) -> Optional[int]:
        """The access stage now, or None when it cannot be established."""
        return access_stage(
            self._now(), settings=self._active_settings(), policy=self._stage_policy
        )

    def _row(
        self,
    ) -> Tuple[Dict[UserRole, Optional[DataClassification]], str, Optional[int], bool]:
        """The row in force, its basis, the stage and whether consent applies."""
        if not self.enabled:
            return (
                dict(self.policy.stage_independent),
                "stage_independent",
                None,
                self.policy.stage_independent_consent,
            )
        stage = self.current_stage()
        if stage is None:
            return self.policy.most_protective_row(), "most_protective", None, False
        return (
            self.policy.row_for_stage(stage),
            "stage",
            stage,
            stage in self.policy.consent_stages,
        )

    # ------------------------------------------------------------ decisions

    def check_access(self, context: AccessContext) -> AccessDecision:
        """Decide one request (see the module docstring for the order)."""
        role = context.role
        tier = DataClassification.parse(context.classification)
        if role is UserRole.SUBJECT:
            return AccessDecision(
                AccessOutcome.ALLOW,
                "the subject always has access to its own data",
                role,
                tier,
                basis="subject",
            )
        if context.emergency:
            decision = self.emergency_override(role, context.emergency_reason or "")
            decision.classification = tier
            return decision

        row, basis, stage, consent_applies = self._row()
        where = f"stage {stage}" if basis == "stage" else f"the {basis} row"
        highest = row.get(role)
        if highest is not None and tier <= highest:
            audited = consent_applies and tier > DataClassification.PUBLIC
            return AccessDecision(
                AccessOutcome.ALLOW_WITH_AUDIT if audited else AccessOutcome.ALLOW,
                f"role {role.value} may read up to {highest.value} under {where}",
                role,
                tier,
                stage,
                basis,
            )
        if consent_applies:
            consent = self._valid_consent(role, tier)
            if consent is not None:
                return AccessDecision(
                    AccessOutcome.ALLOW_WITH_AUDIT,
                    f"consent for role {role.value} up to "
                    f"{consent['classification']} under {where}",
                    role,
                    tier,
                    stage,
                    basis,
                )
        limit = "no access" if highest is None else f"up to {highest.value}"
        return AccessDecision(
            AccessOutcome.DENY,
            f"role {role.value} has {limit} under {where}; {tier.value} denied",
            role,
            tier,
            stage,
            basis,
        )

    def emergency_override(self, role: UserRole, reason: str) -> AccessDecision:
        """Grant or refuse an emergency override (roles, reason, cooldown)."""
        if role not in self.policy.emergency_roles:
            return AccessDecision(
                AccessOutcome.DENY,
                f"role {role.value} cannot use an emergency override",
                role,
                basis="emergency",
            )
        if len((reason or "").strip()) < self.policy.emergency_min_reason:
            return AccessDecision(
                AccessOutcome.DENY,
                "an emergency override needs a reason of at least "
                f"{self.policy.emergency_min_reason} characters",
                role,
                basis="emergency",
            )
        now = self._now()
        cooldown = timedelta(seconds=self.policy.emergency_cooldown_s)
        if self._last_override is not None and now - self._last_override < cooldown:
            remaining = cooldown - (now - self._last_override)
            return AccessDecision(
                AccessOutcome.DENY,
                f"emergency override cooldown: {remaining.total_seconds():.0f} s left",
                role,
                basis="emergency",
            )
        self._last_override = now
        logger.warning("emergency override granted to role %s", role.value)
        return AccessDecision(
            AccessOutcome.EMERGENCY_OVERRIDE,
            f"emergency override granted to role {role.value}",
            role,
            basis="emergency",
        )

    # ------------------------------------------------------------ consent

    def register_consent(
        self,
        role: UserRole,
        classification: DataClassification,
        granted_by: str,
        *,
        expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Record a consent for ``role`` up to ``classification``; return it.

        A later consent for the same role and tier replaces the earlier one.
        """
        tier = DataClassification.parse(classification)
        record = {
            "role": role.value,
            "classification": tier.value,
            "granted_by": str(granted_by),
            "granted_at": self._now().isoformat(),
            "expires_at": _utc(expires_at).isoformat() if expires_at else None,
        }
        self._consents[(role.value, tier.value)] = record
        self._save_consents()
        logger.info("consent registered for role %s up to %s", role.value, tier.value)
        return dict(record)

    def revoke_consent(
        self, role: UserRole, classification: DataClassification
    ) -> bool:
        """Remove a consent; False when none was registered."""
        key = (role.value, DataClassification.parse(classification).value)
        if self._consents.pop(key, None) is None:
            return False
        self._save_consents()
        return True

    def consents(self) -> List[Dict[str, Any]]:
        """Copies of the registered consents."""
        return [dict(record) for record in self._consents.values()]

    def _valid_consent(
        self, role: UserRole, tier: DataClassification
    ) -> Optional[Dict[str, Any]]:
        now = self._now()
        for record in self._consents.values():
            if record["role"] != role.value:
                continue
            if DataClassification.parse(record["classification"]) < tier:
                continue
            expires = record.get("expires_at")
            if expires and datetime.fromisoformat(expires) <= now:
                continue
            return record
        return None

    def _load_consents(self) -> None:
        if self._consent_path is None or not self._consent_path.exists():
            return
        try:
            records = json.loads(self._consent_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("consent file could not be read: %s", exc)
            return
        for record in records:
            try:
                key = (
                    UserRole.from_string(record["role"]).value,
                    DataClassification.parse(record["classification"]).value,
                )
            except (KeyError, ValueError):
                logger.warning("consent file entry skipped: unknown role or tier")
                continue
            self._consents[key] = dict(record)

    def _save_consents(self) -> None:
        if self._consent_path is None:
            return
        path = self._consent_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(list(self._consents.values()), indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)

    # ------------------------------------------------------------ reports

    def access_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Role to tier to readable under the row in force (consent aside)."""
        row, _basis, _stage, _consent = self._row()
        matrix: Dict[str, Dict[str, bool]] = {}
        for role in UserRole:
            highest = DataClassification.CORE if role is UserRole.SUBJECT else row[role]
            matrix[role.value] = {
                tier.value: highest is not None and tier <= highest
                for tier in DataClassification
            }
        return matrix

    def status(self) -> Dict[str, Any]:
        """The row in force and the consents; ``consent_file`` is shown
        through ``mtc.core.paths.display_path``, never as an absolute path.
        """
        _row, basis, stage, consent_applies = self._row()
        return {
            "enabled": self.enabled,
            "stage": stage,
            "basis": basis,
            "consent_applies": consent_applies,
            "consents": len(self._consents),
            "consent_file": (
                display_path(self._consent_path, self._settings)
                if self._consent_path
                else None
            ),
        }
