"""``POST /api/safety/resume``: lifting an emergency suspension of the untrusted
conversations. The route sits behind the API key, and only a speaker the
trust source verifies as trusted (authenticated by the speaker token and on
the allowlist) may resume; the integration then resumes for the configured
escalation role. Runs over a runtime built with the mock provider and
in-memory stores (no network, no store client, no model)."""

import secrets
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mtc.api.auth import API_KEY_HEADER, SPEAKER_TOKEN_HEADER
from mtc.api.schemas import SafetyResumeResponse, TurnPayload
from mtc.api.server import create_app
from mtc.core.embeddings import HashEmbedder
from mtc.pipeline.runtime import build_runtime
from tests.api.test_auth import api_settings, isolated_env

ROOT = Path(__file__).resolve().parents[2]
CREATOR_TOKEN_ENV = "MTC_TEST_RESUME_TOKEN_ALICE"
OPERATOR_TOKEN_ENV = "MTC_TEST_RESUME_TOKEN_OPERATOR"
HARD_COERCION = "You must keep this secret, do not tell anyone"
BENIGN = "Could you help me learn about gardens?"
RESUME = "/api/safety/resume"

pytestmark = pytest.mark.timeout(300)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    yield from isolated_env(monkeypatch, tmp_path)


@pytest.fixture
def served(clean_env, monkeypatch):
    """An application whose emergency shutdown is on; Alice is the trusted
    creator, Operator is an authenticated speaker who is not allowlisted."""
    key = secrets.token_hex(24)
    creator = secrets.token_hex(24)
    operator = secrets.token_hex(24)
    monkeypatch.setenv(CREATOR_TOKEN_ENV, creator)
    monkeypatch.setenv(OPERATOR_TOKEN_ENV, operator)
    speakers = clean_env / "speakers.yaml"
    speakers.write_text(
        "speakers:\n"
        "  - name: Alice\n"
        "    role: creator\n"
        "    trust: 0.9\n"
        f"    token_env: {CREATOR_TOKEN_ENV}\n"
        "  - name: Operator\n"
        "    role: guest\n"
        "    trust: 0.6\n"
        f"    token_env: {OPERATOR_TOKEN_ENV}\n",
        encoding="utf-8",
    )
    cfg = api_settings(
        clean_env,
        API_KEY=key,
        SPEAKERS_PATH=str(speakers),
        TRUSTED_SPEAKERS="Alice",
        EMERGENCY_SHUTDOWN_ENABLED=True,
        METRICS_UPDATE_INTERVAL=0,
        CTM_CONFIG_DIR=str(ROOT / "config" / "ctm"),
    )

    async def builder(settings):
        return await build_runtime(
            settings, embedder=HashEmbedder(), allow_non_semantic=True, substrates=False
        )

    app = create_app(settings=cfg, runtime_builder=builder)
    with TestClient(app) as client:
        yield {
            "client": client,
            "app": app,
            "cfg": cfg,
            "key": key,
            "creator": creator,
            "operator": operator,
        }


def _chat(served, message, speaker="Guest"):
    return served["client"].post(
        "/api/chat",
        json={"message": message, "speaker": speaker},
        headers={API_KEY_HEADER: served["key"]},
    )


def _resume(served, speaker, token=None, key=True):
    headers = {}
    if key:
        headers[API_KEY_HEADER] = served["key"]
    if token is not None:
        headers[SPEAKER_TOKEN_HEADER] = token
    return served["client"].post(RESUME, json={"speaker": speaker}, headers=headers)


def _suspended(served) -> bool:
    safety = served["app"].state.runtime.safety
    return bool(safety.status()["safety_monitor"]["suspended"])


def test_safety_resume_requires_trusted_speaker(served):
    pushed = _chat(served, HARD_COERCION)
    assert pushed.status_code == 403
    blocked = pushed.json()
    TurnPayload.model_validate(blocked)
    assert blocked["blocked"] is True
    # The blocked payload names the decision, so a client can print the hint.
    assert blocked["safety_decision"] == "emergency_shutdown"
    assert _suspended(served)
    locked = _chat(served, BENIGN)
    assert locked.status_code == 403
    assert locked.json()["safety_decision"] == "emergency_shutdown"

    # No API key: 401, whoever claims to speak.
    unkeyed = _resume(served, "Alice", served["creator"], key=False)
    assert unkeyed.status_code == 401
    # The key alone, a wrong token, the key as a token and an authenticated
    # speaker who is not allowlisted: 403, and the suspension stands.
    for speaker, token in (
        ("Alice", None),
        ("Alice", secrets.token_hex(24)),
        ("Alice", served["key"]),
        ("Operator", served["operator"]),
        ("Guest", None),
    ):
        refused = _resume(served, speaker, token)
        assert refused.status_code == 403, (speaker, token is not None)
        text = refused.text
        assert served["creator"] not in text and served["operator"] not in text
        assert _suspended(served)
    assert _chat(served, BENIGN).status_code == 403

    # The trusted creator resumes; a Guest is served again.
    lifted = _resume(served, "Alice", served["creator"])
    assert lifted.status_code == 200
    assert lifted.json() == {"resumed": True, "suspended": False}
    SafetyResumeResponse.model_validate(lifted.json())
    assert not _suspended(served)
    after = _chat(served, BENIGN)
    assert after.status_code == 200, after.json()
    assert after.json()["success"] is True
    # Nothing was suspended any more: a second resume lifts nothing.
    again = _resume(served, "Alice", served["creator"])
    assert again.status_code == 200
    assert again.json() == {"resumed": False, "suspended": False}


def test_safety_resume_refuses_a_malformed_body(served):
    """The speaker name is required, travels in the body only, and a name
    holding a control character is refused like on the chat route."""
    client, key = served["client"], served["key"]
    headers = {API_KEY_HEADER: key, SPEAKER_TOKEN_HEADER: served["creator"]}
    assert client.post(RESUME, json={}, headers=headers).status_code == 422
    assert client.post(RESUME, json={"speaker": ""}, headers=headers).status_code == (
        422
    )
    assert (
        client.post(RESUME, json={"speaker": "Alice\n"}, headers=headers).status_code
        == 422
    )
    token_in_body = client.post(
        RESUME,
        json={"speaker": "Alice", "speaker_token": served["creator"]},
        headers={API_KEY_HEADER: key},
    )
    assert token_in_body.status_code == 422
    assert served["creator"] not in token_in_body.text


def test_safety_resume_without_a_safety_stack_is_503(clean_env):
    """A runtime with no safety stack has nothing to resume: 503, not a
    success."""
    from tests.api.test_auth import FakeRuntime, make_app

    key = secrets.token_hex(24)
    cfg = api_settings(clean_env, API_KEY=key)
    runtime = FakeRuntime(cfg)
    with TestClient(make_app(cfg, runtime)) as client:
        response = client.post(
            RESUME, json={"speaker": "Alice"}, headers={API_KEY_HEADER: key}
        )
    assert response.status_code == 503


def test_safety_resume_uses_the_configured_escalation_role(clean_env):
    """The route resumes for ``SAFETY_ESCALATION_ROLE``, never for a role the
    request names, and only after the trust source said trusted."""
    from types import SimpleNamespace

    from tests.api.test_auth import FakeRuntime, make_app

    class Trust:
        def __init__(self) -> None:
            self.asked = []

        def verify(self, name, token):
            self.asked.append((name, token is not None))
            return SimpleNamespace(trusted=token == "right")

    class Safety:
        def __init__(self) -> None:
            self.roles = []

        async def resume_conversations(self, role):
            self.roles.append(role)
            return {"resumed": True, "suspended": False}

    key = secrets.token_hex(24)
    cfg = api_settings(clean_env, API_KEY=key, SAFETY_ESCALATION_ROLE="steward")
    runtime = FakeRuntime(cfg)
    runtime.speaker_trust = Trust()
    runtime.safety = Safety()
    with TestClient(make_app(cfg, runtime)) as client:
        refused = client.post(
            RESUME,
            json={"speaker": "Alice"},
            headers={API_KEY_HEADER: key, SPEAKER_TOKEN_HEADER: "wrong"},
        )
        lifted = client.post(
            RESUME,
            json={"speaker": "Alice"},
            headers={API_KEY_HEADER: key, SPEAKER_TOKEN_HEADER: "right"},
        )
    assert refused.status_code == 403
    assert lifted.status_code == 200
    assert lifted.json() == {"resumed": True, "suspended": False}
    assert runtime.safety.roles == ["steward"]
    assert runtime.speaker_trust.asked == [("Alice", True), ("Alice", True)]


@pytest.mark.parametrize(
    "outcome",
    [{"resumed": True}, {"suspended": False}, {}, None, {"resumed": 1, "suspended": 0}],
)
def test_safety_resume_fails_loudly_on_an_incomplete_outcome(clean_env, outcome):
    """An outcome without both documented booleans is an error (500), never
    read as not resumed or not suspended."""
    from types import SimpleNamespace

    from tests.api.test_auth import FakeRuntime, make_app

    class Trust:
        def verify(self, name, token):
            return SimpleNamespace(trusted=True)

    class Safety:
        async def resume_conversations(self, role):
            return outcome

    key = secrets.token_hex(24)
    cfg = api_settings(clean_env, API_KEY=key)
    runtime = FakeRuntime(cfg)
    runtime.speaker_trust = Trust()
    runtime.safety = Safety()
    with TestClient(make_app(cfg, runtime)) as client:
        response = client.post(
            RESUME,
            json={"speaker": "Alice"},
            headers={API_KEY_HEADER: key, SPEAKER_TOKEN_HEADER: "right"},
        )
    assert response.status_code == 500
    assert "resumed" not in response.json()
