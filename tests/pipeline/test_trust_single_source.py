"""Tree-wide audit: the speaker trust module is the only reader of the trust
keys and the only source of trust; its helpers refuse bare strings."""

import ast
from pathlib import Path
from typing import List

import pytest

from mtc.pipeline.speaker_trust import SpeakerTrust

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "mtc"

TRUST_KEYS = frozenset(
    {
        "SPEAKERS_PATH",
        "SPEAKER_TOKENS",
        "TRUSTED_SPEAKERS",
        "TRUST_ACTIVATION_BONUS",
        "TRUSTED_EMOTION_TRUST",
        "DEFAULT_EMOTION_TRUST",
    }
)
ALLOWED = {
    PACKAGE / "pipeline" / "speaker_trust.py",
    PACKAGE / "core" / "config.py",
}


def _key_reads(source: str, label: str) -> List[str]:
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in TRUST_KEYS:
            hits.append(f"{label}:{node.lineno}:{node.attr}")
        elif isinstance(node, ast.Constant) and node.value in TRUST_KEYS:
            hits.append(f"{label}:{node.lineno}:{node.value}")
    return hits


def test_trust_keys_read_only_by_speaker_trust():
    hits: List[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or path in ALLOWED:
            continue
        hits.extend(
            _key_reads(
                path.read_text(encoding="utf-8"), path.relative_to(ROOT).as_posix()
            )
        )
    assert not hits, "trust key read outside speaker_trust.py: " + ", ".join(hits)
    own = _key_reads((PACKAGE / "pipeline" / "speaker_trust.py").read_text(), "own")
    assert own, "speaker_trust.py must read the trust keys"


def test_trust_helpers_refuse_bare_strings():
    trust = SpeakerTrust()
    for helper in (
        trust.trust_bonus,
        trust.emotion_trust,
        trust.label_for,
        trust.safety_role,
        trust.session_key,
    ):
        with pytest.raises(TypeError):
            helper("Alice")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        with trust.use_identity("Alice"):  # type: ignore[arg-type]
            pass


def test_audit_detects_a_planted_read():
    assert _key_reads("x = settings.TRUSTED_SPEAKERS", "planted")
    assert _key_reads('x = getattr(s, "SPEAKER_TOKENS")', "planted")
    assert not _key_reads("x = settings.DEFAULT_SPEAKER_TRUST", "planted")
