"""
Privacy
=======

Privacy tiers and who may read them: the data classifier (whole-word keyword,
tag and content-type rules over four tiers), the access stage, the access
control enforcer (a replaceable policy matrix with consent and emergency
rules), the privacy auditor (bounded log, JSONL trail, audit store, violation
detection and reports) and the facade that combines them.

``mtc.privacy.stores`` holds the MongoDB audit store and is imported by its own
path. Importing this package opens no store client. Audit events hold user ids
and access records and are private data at runtime.
"""

from mtc.privacy.access_control_enforcer import (
    AccessContext,
    AccessControlEnforcer,
    AccessDecision,
    AccessOutcome,
    AccessPolicy,
    UserRole,
)
from mtc.privacy.data_classifier import (
    ClassificationResult,
    DataClassification,
    DataClassifier,
    StagePolicy,
    access_stage,
    match_keywords,
    stage_info,
)
from mtc.privacy.privacy_auditor import (
    EventType,
    PrivacyAuditor,
    PrivacyEvent,
    PrivacyViolation,
    ViolationType,
)
from mtc.privacy.privacy_integration import (
    IntegratedPrivacySystem,
    get_privacy_system,
    reset_privacy_system,
)

__all__ = [
    "AccessContext",
    "AccessControlEnforcer",
    "AccessDecision",
    "AccessOutcome",
    "AccessPolicy",
    "ClassificationResult",
    "DataClassification",
    "DataClassifier",
    "EventType",
    "IntegratedPrivacySystem",
    "PrivacyAuditor",
    "PrivacyEvent",
    "PrivacyViolation",
    "StagePolicy",
    "UserRole",
    "ViolationType",
    "access_stage",
    "get_privacy_system",
    "match_keywords",
    "reset_privacy_system",
    "stage_info",
]
