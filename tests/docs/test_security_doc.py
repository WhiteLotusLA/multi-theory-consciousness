"""docs/SECURITY.md states what the API and the safety stack do today.

Each test pins the sentences of one behaviour: what turning authentication
off exposes, the Host header limit, the websocket Origin rule, the claimed
name rule (with the length read from the trust source), and the safety gate
and emergency suspension. Bold markup and line breaks are ignored.
"""

import re
from pathlib import Path

from mtc.pipeline.speaker_trust import MAX_CLAIMED_NAME_CHARS

ROOT = Path(__file__).resolve().parents[2]
SECURITY = ROOT / "docs" / "SECURITY.md"


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def _section(title: str) -> str:
    """The normalized body of the section headed ``## <title>``."""
    text = SECURITY.read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"SECURITY.md has no section {title!r}"
    return _normalized(match.group(1))


def test_security_doc_states_what_auth_off_exposes():
    body = _section("Loopback by default")
    for phrase in (
        "With API_AUTH_ENABLED=false every process on the machine that can "
        "reach the port can use every route (chat, state, memory search).",
        "Only the spontaneous-thought websocket frames are withheld, because no "
        "connection counts as a keyed client.",
        "A state frame can quote the text of the workspace winner, which may "
        "be a thought or a recalled memory.",
    ):
        assert phrase in body, phrase


def test_security_doc_states_the_host_header_limit():
    body = _section("Loopback by default")
    assert (
        "With API_AUTH_ENABLED=false the HTTP routes do not validate the Host "
        "header, so a web page that rebinds its domain to the loopback address "
        "can call them from a browser on the same machine; keep authentication "
        "on whenever a browser is used on that machine." in body
    )


def test_security_doc_states_the_websocket_origin_rule():
    body = _section("Loopback by default")
    for phrase in (
        "A handshake without an `Origin` header (a client that is not a "
        "browser) is served",
        "the literal `null` included, is closed with 1008",
        "With `API_CORS_ORIGINS` set to `*` the check refuses every browser origin",
        "a handshake without an `Origin` header is still served",
    ):
        assert phrase in body, phrase


def test_security_doc_states_the_claimed_name_rule():
    body = _section("Speakers: a token, never a name")
    assert f"cut to {MAX_CLAIMED_NAME_CHARS} characters" in body
    assert "A claim with nothing usable left is the anonymous default speaker." in (
        body
    )
    assert "quotes it under the unverified template" in body
    assert "with a fixed message that does not echo it" in body


def test_security_doc_says_who_may_resume_a_suspension():
    body = _section("The safety gate and emergency suspension")
    for phrase in (
        "The gate fails closed. With `SAFETY_GATE_FAIL_CLOSED` true (the default)",
        "ends with the error code `safety_unavailable` (HTTP 503) and nothing "
        "of it is stored",
        "With it false (the default) the message is denied",
        "suspends every untrusted conversation",
        "Only a trusted speaker may lift a suspension",
        "`POST /api/safety/resume`",
        "`mtc safety resume`",
    ):
        assert phrase in body, phrase
