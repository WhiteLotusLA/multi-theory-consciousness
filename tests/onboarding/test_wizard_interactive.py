"""The wizard's questions, driven by scripted answers."""

import http.server
import socket
import threading
import time
from contextlib import contextmanager

import pytest

from mtc.core.config import Settings
from mtc.core.envfile import read_env
from mtc.onboarding import wizard
from mtc.onboarding.schema import (
    BrainAnswers,
    OnboardingAnswers,
    PersonalityConfig,
    StoreChoice,
)
from mtc.onboarding.wizard import (
    STEP_TITLES,
    ScriptedPrompter,
    collect_interactive,
    detect_timezone,
    probe_brain,
    probe_store,
    run_init,
)

from .test_wizard_noninteractive import ENV_NAME, make_project

LOOPBACK = "127.0.0.1"

# Every answer of a session that takes the defaults: a friend called Tessera,
# no store, the mock provider. Comments name the step each answer belongs to.
DEFAULT_SESSION = (
    "yes",  # 1 framing
    "Tessera",  # 2 name
    "",  # 2 pronouns: they
    "friend",  # 3 type
    "Alice Example",  # 4 creator
    "",  # 4 address form: the type's default
    "",  # 5 personality: balanced
    "",  # 5 core values: the default
    "",  # 6 time zone: the detected one
    "",  # 6 wake hour
    "",  # 6 sleep hour
    "",  # 7 PostgreSQL: no
    "",  # 8 provider: mock
    "18765",  # 8 API port
    "n",  # 10 bootstrap
)


@pytest.fixture
def project(monkeypatch, tmp_path):
    return make_project(monkeypatch, tmp_path)


@contextmanager
def fake_server(behaviour: str):
    """A stdlib server on an ephemeral loopback port: ``ok`` answers 200,
    ``error`` answers 500 and ``hang`` accepts and never answers."""
    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - the handler API's name
            if behaviour == "hang":
                release.wait(10)
                return
            status = 200 if behaviour == "ok" else 500
            body = b"{}"
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = http.server.ThreadingHTTPServer((LOOPBACK, 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1]


def step_lines(said) -> list:
    return [line for line in said if line.startswith("Step ")]


def test_scripted_session_covers_eleven_steps(project):
    prompter = ScriptedPrompter(DEFAULT_SESSION)
    answers = collect_interactive(prompter, probe=False)
    assert prompter.remaining == 0
    expected = OnboardingAnswers.model_validate(
        {
            "acknowledged": True,
            "persona": {
                "name": "Tessera",
                "pronouns": {
                    "subject": "they",
                    "object": "them",
                    "possessive": "their",
                },
                "companion_type": "friend",
                "creator_name": "Alice Example",
                "personality_preset": "balanced",
            },
            "locale": {"timezone": "UTC", "wake_hour": 6, "sleep_hour": 22},
            "brain": {"provider": "mock"},
            "api": {"port": 18765},
            "bootstrap_db": False,
        }
    )
    assert answers == expected
    # The framing is shown before the first question and has to be confirmed.
    assert "not a consciousness detector" in "\n".join(prompter.said)
    assert prompter.questions[0].startswith("Type yes")
    assert "Nothing will persist" not in prompter.questions
    assert any("nothing will persist" in line for line in prompter.said)

    # The whole run: eleven step headers, in order, and a written project.
    voice = ScriptedPrompter(DEFAULT_SESSION + ("y",))
    code = run_init(
        from_path=None,
        yes=False,
        force=False,
        bootstrap_db=False,
        project_dir=project,
        probe=False,
        print_plan=False,
        prompter=voice,
    )
    assert code == 0 and voice.remaining == 0
    headers = step_lines(voice.said)
    assert headers == [
        f"Step {number} of 11: {STEP_TITLES[number - 1]}"
        for number in (1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11)
    ]
    assert len(STEP_TITLES) == 11
    env = read_env(project / ENV_NAME)
    assert env["MTC_SYSTEM_NAME"] == "Tessera"
    assert env["MTC_CREATOR_NAME"] == "Alice Example"
    # The friend default addresses the creator by first name.
    persona_text = (project / "config" / "persona.yaml").read_text()
    assert "address_form: Alice\n" in persona_text


def test_framing_must_be_acknowledged_with_yes(project):
    for typed in ("", "y", "no", "sure"):
        voice = ScriptedPrompter([typed])
        code = run_init(
            from_path=None,
            yes=False,
            force=False,
            bootstrap_db=False,
            project_dir=project,
            probe=False,
            print_plan=False,
            prompter=voice,
        )
        assert code == 2
        assert list(project.iterdir()) == []
        assert "not acknowledged" in "\n".join(voice.said)
    # Input that ends early stops the run without writing.
    voice = ScriptedPrompter(DEFAULT_SESSION[:5])
    code = run_init(
        from_path=None,
        yes=False,
        force=False,
        bootstrap_db=False,
        project_dir=project,
        probe=False,
        print_plan=False,
        prompter=voice,
    )
    assert code == 2 and list(project.iterdir()) == []


def test_timezone_detection_is_confirmable(project, monkeypatch, tmp_path):
    assert detect_timezone({"TZ": "Europe/Berlin"}, tmp_path / "absent") == (
        "Europe/Berlin"
    )
    assert detect_timezone({"TZ": "Not/AZone"}, tmp_path / "absent") is None
    # The system zone link: its target names the zone.
    zone_file = tmp_path / "share" / "zoneinfo" / "Asia" / "Tokyo"
    zone_file.parent.mkdir(parents=True)
    zone_file.write_bytes(b"")
    link = tmp_path / "localtime"
    link.symlink_to(zone_file)
    assert detect_timezone({}, link) == "Asia/Tokyo"
    assert detect_timezone({}, tmp_path / "absent") is None

    # Enter keeps the detected zone ...
    monkeypatch.setenv("TZ", "Europe/Berlin")
    session = list(DEFAULT_SESSION)
    prompter = ScriptedPrompter(session)
    kept = collect_interactive(prompter, probe=False)
    assert kept.locale.timezone == "Europe/Berlin"
    assert any("Detected time zone: Europe/Berlin" in line for line in prompter.said)
    # ... a typed zone replaces it, and a wrong one is asked again.
    session[8:9] = ["Mars/Olympus", "Asia/Tokyo"]
    session[10:12] = ["7", "7", "23"]
    prompter = ScriptedPrompter(session)
    typed = collect_interactive(prompter, probe=False)
    assert typed.locale.timezone == "Asia/Tokyo"
    assert (typed.locale.wake_hour, typed.locale.sleep_hour) == (7, 23)
    notes = "\n".join(prompter.said)
    assert "not an IANA zone name" in notes
    assert "different hour" in notes
    assert prompter.remaining == 0


def test_brain_probe_plain_language_results(project):
    with fake_server("ok") as port:
        result = probe_brain(BrainAnswers(provider="local", host=LOOPBACK, port=port))
        assert (result.ok, result.status) == (True, "ok")
        assert "answered" in result.message
        by_url = probe_brain(
            BrainAnswers(provider="local", base_url=f"http://{LOOPBACK}:{port}")
        )
        assert by_url.ok is True

    closed = unused_port()
    refused = probe_brain(BrainAnswers(provider="local", host=LOOPBACK, port=closed))
    assert (refused.ok, refused.status) == (False, "unreachable")
    assert "Nothing is listening" in refused.message
    assert "Setup continues" in refused.message

    with fake_server("hang") as port:
        started = time.monotonic()
        slow = probe_brain(
            BrainAnswers(provider="local", host=LOOPBACK, port=port), timeout_s=0.3
        )
        assert time.monotonic() - started < 5
        assert (slow.ok, slow.status) == (False, "unreachable")
        assert "did not answer within 0.3 seconds" in slow.message

    with fake_server("error") as port:
        broken = probe_brain(BrainAnswers(provider="local", host=LOOPBACK, port=port))
        assert (broken.ok, broken.status) == (False, "failed")
        assert "HTTP 500" in broken.message

    # Providers with nothing to probe say so; neither reads as measured.
    mock = probe_brain(BrainAnswers(provider="mock"))
    assert mock.status == "skipped" and "test text" in mock.message
    paid = probe_brain(
        BrainAnswers(provider="anthropic", anthropic_model="example-model")
    )
    assert (paid.ok, paid.status) == (False, "insufficient_data")
    for outcome in (refused, slow, broken, mock, paid):
        outcome.message.encode("ascii")
        assert "Traceback" not in outcome.message

    # The store probe speaks the same way, on a listening and a closed port.
    with fake_server("ok") as port:
        up = probe_store(StoreChoice(enabled=True, host=LOOPBACK, port=port), "redis")
        assert (up.ok, up.status) == (True, "ok")
    down = probe_store(StoreChoice(enabled=True, host=LOOPBACK, port=closed), "neo4j")
    assert (down.ok, down.status) == (False, "unreachable")
    assert "Neo4j" in down.message
    off = probe_store(StoreChoice(), "postgres")
    assert off.status == "skipped"
    with pytest.raises(ValueError):
        probe_store(StoreChoice(enabled=True), "sqlite")


def test_interactive_probe_reports_and_never_blocks(project):
    closed = unused_port()
    session = list(DEFAULT_SESSION)
    # Step 8: a local server where nothing listens.
    session[12:13] = ["local", LOOPBACK, str(closed), "example-model", "qwen"]
    prompter = ScriptedPrompter(session)
    answers = collect_interactive(prompter, probe=True)
    assert prompter.remaining == 0
    assert answers.brain.provider == "local"
    assert (answers.brain.host, answers.brain.port) == (LOOPBACK, closed)
    assert answers.brain.model == "example-model"
    assert answers.brain.template_family == "qwen"
    assert any("Nothing is listening" in line for line in prompter.said)


def test_secret_prompts_use_the_secret_path(project):
    typed_password = "typed-store-password-1"
    typed_key = "typed-anthropic-key-1"
    session = list(DEFAULT_SESSION)
    # Step 7: PostgreSQL with a typed password, Redis with none, by hand.
    session[11:12] = [
        "y",  # PostgreSQL
        "n",  # MongoDB
        "n",  # Qdrant
        "y",  # Redis
        "n",  # Neo4j
        "brew",  # service manager
        "acme",  # prefix
        "",  # PostgreSQL host
        "",  # PostgreSQL port
        "bad password",  # refused: whitespace
        typed_password,
        "",  # Redis host
        "",  # Redis port
        "",  # Redis password: none
    ]
    # Step 8: Anthropic with a typed key.
    provider_at = session.index("18765") - 1
    session[provider_at : provider_at + 1] = ["anthropic", "example-model", typed_key]
    prompter = ScriptedPrompter(session)
    answers = collect_interactive(prompter, probe=False)
    assert prompter.remaining == 0

    assert len(prompter.secret_questions) == 4
    assert all("password" in q or "key" in q for q in prompter.secret_questions)
    visible = "\n".join(prompter.questions + prompter.said)
    assert (typed_password in visible) is False
    assert (typed_key in visible) is False
    assert "bad password" not in visible
    assert "cannot be used" in visible

    postgres = answers.stores.postgres
    assert postgres.enabled is True
    assert (postgres.password.get_secret_value() == typed_password) is True
    assert answers.stores.redis.enabled is True
    assert answers.stores.redis.password is None
    assert answers.stores.service_manager == "brew"
    assert answers.stores.db_prefix == "acme"
    assert (answers.brain.anthropic_api_key.get_secret_value() == typed_key) is True
    assert (typed_password in repr(answers)) is False
    assert (typed_key in answers.model_dump_json()) is False

    # Written, the typed values land in the environment file and nowhere else.
    plan = wizard.plan_writes(answers, project, existing_env={})
    assert (plan.env_updates["MTC_POSTGRES_PASSWORD"] == typed_password) is True
    assert "MTC_POSTGRES_PASSWORD" not in plan.env_keep_existing
    assert "MTC_REDIS_PASSWORD" not in plan.env_updates
    assert (plan.env_updates["ANTHROPIC_API_KEY"] == typed_key) is True
    described = plan.describe()
    assert (typed_password in described) is False and (typed_key in described) is False
    for planned in plan.files:
        assert (typed_password in planned.content) is False
        assert (typed_key in planned.content) is False


def test_custom_personality_values_validated(project):
    session = list(DEFAULT_SESSION)
    session[6:7] = [
        "custom",
        "1.5",  # refused: above 1
        "abc",  # refused: not a number
        "0.9",  # openness
        "0.1",  # conscientiousness
        "-0.2",  # refused: below 0
        "0.5",  # extraversion
        "1",  # agreeableness
        "0",  # neuroticism
    ]
    session[7 + 8] = "candor, patience"
    prompter = ScriptedPrompter(session)
    answers = collect_interactive(prompter, probe=False)
    assert prompter.remaining == 0
    assert answers.persona.personality == PersonalityConfig(
        openness=0.9,
        conscientiousness=0.1,
        extraversion=0.5,
        agreeableness=1.0,
        neuroticism=0.0,
    )
    assert answers.persona.personality_preset is None
    assert answers.persona.resolved_personality().openness == pytest.approx(0.9)
    assert answers.persona.core_values == ["candor", "patience"]
    notes = [line for line in prompter.said if "number from 0 to 1" in line]
    assert len(notes) == 3

    # A custom pronoun set and a typed address form are kept as typed.
    session = list(DEFAULT_SESSION)
    session[2:3] = ["custom", "xe", "xem", "xyr"]
    session[5 + 3] = "Coach"
    prompter = ScriptedPrompter(session)
    answers = collect_interactive(prompter, probe=False)
    assert answers.persona.pronouns.model_dump() == {
        "subject": "xe",
        "object": "xem",
        "possessive": "xyr",
    }
    assert answers.persona.address_form == "Coach"
    # The core values question offers the Settings default.
    default_values = Settings.model_fields["CORE_VALUES"].default
    asked = [q for q in prompter.questions if q.startswith("Core values")]
    assert len(asked) == 1 and f"[{default_values}]" in asked[0]
    assert answers.persona.core_values is None
