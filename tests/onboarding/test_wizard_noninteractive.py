"""The wizard without questions: an answers file in, a configured project out.

The helpers here (``make_project``, ``write_answers`` and ``run``) are shared
by the sibling wizard test modules.
"""

import copy
import os
import re
import socket
import stat
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
import yaml

from mtc.core import config as config_module
from mtc.core.envfile import read_env
from mtc.core.persona import load_persona
from mtc.onboarding import wizard
from mtc.onboarding.schema import OnboardingAnswers
from mtc.onboarding.wizard import ScriptedPrompter, run_init

ROOT = Path(__file__).resolve().parents[2]
HEX_RUN = re.compile(r"[0-9a-f]{32,}")

WRITTEN_FILES = (
    "config/persona.yaml",
    "config/speakers.yaml",
    "agent_space/identity_statement.md",
    "config/ctm/persona.yaml",
    "config/ctm/seeds.d/10_companionship.yaml",
)
ENV_NAME = wizard.ENV_FILE_NAME


def make_project(monkeypatch, tmp_path) -> Path:
    """An empty project directory with no MTC settings in the environment."""
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "_settings", None)
    target = tmp_path / "proj"
    target.mkdir()
    return target


@pytest.fixture
def project(monkeypatch, tmp_path):
    return make_project(monkeypatch, tmp_path)


def base_answers() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "acknowledged": True,
        "persona": {
            "name": "Tessera",
            "pronouns": {"subject": "they", "object": "them", "possessive": "their"},
            "companion_type": "friend",
            "creator_name": "Alice",
        },
        "locale": {"timezone": "UTC", "wake_hour": 6, "sleep_hour": 22},
        "stores": {},
        "brain": {"provider": "mock"},
        "api": {"port": 18765},
        "bootstrap_db": False,
    }


def merged(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged(result[key], value)
        else:
            result[key] = value
    return result


def write_answers(directory: Path, **overrides: Any) -> Path:
    path = directory / "answers.yaml"
    path.write_text(
        yaml.safe_dump(merged(base_answers(), overrides), sort_keys=False),
        encoding="utf-8",
    )
    return path


def run(
    project_dir: Path,
    answers_path: Optional[Path],
    *,
    yes: bool = True,
    force: bool = False,
    bootstrap_db: bool = False,
    probe: bool = False,
    print_plan: bool = False,
    script: Tuple[str, ...] = (),
) -> Tuple[int, ScriptedPrompter]:
    voice = ScriptedPrompter(list(script))
    code = run_init(
        from_path=answers_path,
        yes=yes,
        force=force,
        bootstrap_db=bootstrap_db,
        project_dir=project_dir,
        probe=probe,
        print_plan=print_plan,
        prompter=voice,
    )
    return code, voice


def snapshot(project_dir: Path) -> Dict[str, bytes]:
    return {
        path.relative_to(project_dir).as_posix(): path.read_bytes()
        for path in sorted(project_dir.rglob("*"))
        if path.is_file()
    }


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_from_answers_writes_expected_files(project, tmp_path):
    code, voice = run(project, write_answers(tmp_path))
    assert code == 0
    written = snapshot(project)
    assert sorted(written) == sorted(WRITTEN_FILES + (ENV_NAME,))
    for relative in WRITTEN_FILES + (ENV_NAME,):
        assert mode_of(project / relative) == 0o600, relative
    assert mode_of(project / "agent_space") == 0o700

    persona = load_persona(
        settings=config_module.Settings(_env_file=None, PROJECT_ROOT=str(project))
    )
    assert persona["name"] == "Tessera"
    assert persona["companion_type"] == "friend"
    assert persona["creator"] == {"name": "Alice", "address_form": "Alice"}
    assert "Tessera knows it is an AI research system" in persona["self_description"]
    assert "{" not in persona["self_description"]
    assert "{" not in persona["communication_style"]
    assert "{speaker}" in persona["speaker_templates"]["known"]

    env = read_env(project / ENV_NAME)
    assert env["MTC_SYSTEM_NAME"] == "Tessera"
    assert env["MTC_LLM_PROVIDER"] == "mock"
    assert env["MTC_API_PORT"] == "18765"
    # The install date belongs to the bootstrap, never to the wizard.
    assert "MTC_SYSTEM_BIRTH_DATE" not in env
    text = "\n".join(voice.said)
    assert "Next steps:" in text
    assert "mtc doctor" in text and "mtc db init" in text
    assert voice.questions == [] and voice.secret_questions == []


def test_refuses_overwrite_without_force(project, tmp_path):
    answers = write_answers(tmp_path)
    assert run(project, answers)[0] == 0
    before = snapshot(project)
    changed = write_answers(
        tmp_path, persona={"name": "Testagent", "companion_type": "mentor"}
    )
    code, voice = run(project, changed)
    assert code == 2
    assert snapshot(project) == before
    text = "\n".join(voice.said)
    assert "--force" in text
    assert "config/persona.yaml" in text
    assert "Nothing was written" in text
    # One existing file is enough to refuse, whichever it is.
    for relative in WRITTEN_FILES:
        (project / relative).unlink()
    (project / "config" / "speakers.yaml").write_text("speakers: []\n")
    code, _voice = run(project, changed)
    assert code == 2
    assert (project / "config" / "speakers.yaml").read_text() == "speakers: []\n"
    assert not (project / "config" / "persona.yaml").exists()


def test_force_rewrites_and_keeps_existing_secrets(project, tmp_path):
    first = write_answers(
        tmp_path,
        stores={"postgres": {"enabled": True}, "neo4j": {"enabled": True}},
    )
    assert run(project, first)[0] == 0
    env_before = read_env(project / ENV_NAME)
    persona_before = (project / "config" / "persona.yaml").read_text()

    second = write_answers(
        tmp_path,
        persona={"name": "Testagent", "companion_type": "mentor"},
        stores={"postgres": {"enabled": True}, "neo4j": {"enabled": True}},
    )
    code, _voice = run(project, second, force=True)
    assert code == 0
    env_after = read_env(project / ENV_NAME)
    assert (project / "config" / "persona.yaml").read_text() != persona_before
    assert env_after["MTC_SYSTEM_NAME"] == "Testagent"
    assert env_after["MTC_COMPANION_TYPE"] == "mentor"
    for key in (
        "MTC_API_KEY",
        "MTC_POSTGRES_PASSWORD",
        "MTC_NEO4J_PASSWORD",
        "MTC_SPEAKER_TOKENS",
    ):
        assert env_before[key], key
        assert (env_after[key] == env_before[key]) is True, key

    # A new creator is a new person: the old pair goes and a new token comes.
    third = write_answers(tmp_path, persona={"creator_name": "Operator"})
    assert run(project, third, force=True)[0] == 0
    env_renamed = read_env(project / ENV_NAME)
    assert env_renamed["MTC_CREATOR_NAME"] == "Operator"
    old_token = env_before["MTC_SPEAKER_TOKENS"].split(":", 1)[1]
    assert env_renamed["MTC_SPEAKER_TOKENS"].startswith("Operator:")
    assert (old_token in env_renamed["MTC_SPEAKER_TOKENS"]) is False
    assert env_renamed["MTC_TRUSTED_SPEAKERS"] == "Operator"
    assert (env_renamed["MTC_API_KEY"] == env_before["MTC_API_KEY"]) is True


def test_missing_acknowledgement_exits_2(project, tmp_path):
    for value in (False, None):
        code, voice = run(project, write_answers(tmp_path, acknowledged=value))
        assert code == 2
        assert snapshot(project) == {}
        text = "\n".join(voice.said)
        assert "acknowledged" in text
        assert "framing" in text
    document = base_answers()
    del document["acknowledged"]
    path = tmp_path / "answers.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    code, voice = run(project, path)
    assert code == 2 and snapshot(project) == {}
    assert "acknowledged" in "\n".join(voice.said)


def test_invalid_answers_list_every_error(project, tmp_path, monkeypatch):
    monkeypatch.delenv("WIZARD_TEST_UNSET_PASSWORD", raising=False)
    bad = write_answers(
        tmp_path,
        persona={"name": "Tes,sera", "companion_type": "pet"},
        locale={"timezone": "Mars/Olympus"},
        stores={"db_prefix": "Bad-Prefix"},
        brain={"provider": "anthropic"},
    )
    code, voice = run(project, bad)
    assert code == 2
    assert snapshot(project) == {}
    text = "\n".join(voice.said)
    for where in (
        "persona.name",
        "persona.companion_type",
        "locale.timezone",
        "stores.db_prefix",
        "brain",
    ):
        assert where in text, where
    # The offending values are not echoed back.
    assert "Tes,sera" not in text and "Mars/Olympus" not in text

    unset = write_answers(
        tmp_path,
        stores={
            "postgres": {
                "enabled": True,
                "password_env": "WIZARD_TEST_UNSET_PASSWORD",
            }
        },
    )
    code, voice = run(project, unset)
    assert code == 2 and snapshot(project) == {}
    assert "WIZARD_TEST_UNSET_PASSWORD is not set" in "\n".join(voice.said)

    code, voice = run(project, tmp_path / "missing.yaml")
    assert code == 2 and "not found" in "\n".join(voice.said)
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just text\n", encoding="utf-8")
    assert run(project, scalar)[0] == 2


def test_yes_without_answers_exits_2(project):
    code, voice = run(project, None, yes=True)
    assert code == 2
    assert snapshot(project) == {}
    assert "--from" in "\n".join(voice.said)
    assert voice.questions == []


def test_from_without_yes_asks_before_writing(project, tmp_path):
    answers = write_answers(tmp_path)
    code, voice = run(project, answers, yes=False, script=("n",))
    assert code == 2 and snapshot(project) == {}
    assert len(voice.questions) == 1
    code, voice = run(project, answers, yes=False, script=("y",))
    assert code == 0 and (project / "config" / "persona.yaml").is_file()
    # No answer available at all: nothing is written.
    code, _voice = run(tmp_path / "other", answers, yes=False, script=())
    assert code == 2 and not (tmp_path / "other").exists()


def test_print_plan_writes_nothing_and_shows_no_secret(project, tmp_path):
    answers = write_answers(
        tmp_path,
        stores={
            "service_manager": "docker",
            "postgres": {"enabled": True},
            "mongodb": {"enabled": True},
            "redis": {"enabled": True},
            "neo4j": {"enabled": True},
        },
    )
    code, voice = run(project, answers, print_plan=True)
    assert code == 0
    assert snapshot(project) == {}
    text = "\n".join(voice.said)
    for expected in (
        "config/persona.yaml (mode 0600)",
        "config/speakers.yaml",
        "agent_space/identity_statement.md",
        "MTC_API_KEY (kept when already set)",
        "MTC_POSTGRES_PASSWORD (kept when already set)",
        "MTC_NEO4J_PASSWORD (kept when already set)",
        "MTC_SPEAKER_TOKENS",
        "Nothing was written",
    ):
        assert expected in text, expected
    assert HEX_RUN.search(text) is None
    assert "=" not in text
    assert "Next steps" not in text


def test_no_probe_makes_no_network_call(project, monkeypatch):
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("the wizard touched the network with probing off")

    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    script = (
        "yes",
        "Tessera",
        "",
        "friend",
        "Alice",
        "",
        "",
        "",
        "",
        "",
        "",
        # stores: PostgreSQL and Qdrant, started by hand
        "y",
        "n",
        "y",
        "n",
        "n",
        "none",
        "",
        "",
        "",
        "",
        "",
        "",
        # brain: a local server on its default endpoint, then the API port
        "local",
        "",
        "",
        "",
        "",
        "18765",
        "n",
        "y",
    )
    code, voice = run(project, None, yes=False, probe=False, script=script)
    assert code == 0, voice.said
    assert voice.remaining == 0
    assert attempts == []
    env = read_env(project / ENV_NAME)
    assert env["MTC_LLM_PROVIDER"] == "local"
    assert env["MTC_POSTGRES_ENABLED"] == "true"
    assert env["MTC_QDRANT_ENABLED"] == "true"

    # With probing on the same session does reach for the probes, so the
    # guard above is not vacuous.
    calls = []
    monkeypatch.setattr(
        wizard,
        "probe_store",
        lambda choice, store, **kwargs: calls.append(store)
        or wizard.ProbeResult(False, "unreachable", f"{store} probed"),
    )
    monkeypatch.setattr(
        wizard,
        "probe_brain",
        lambda brain, **kwargs: calls.append(brain.provider)
        or wizard.ProbeResult(False, "unreachable", "brain probed"),
    )
    other = project.parent / "probed"
    code, voice = run(other, None, yes=False, probe=True, script=script)
    assert code == 0
    assert calls == ["postgres", "qdrant", "local"]
    assert attempts == []
    assert "brain probed" in "\n".join(voice.said)


def test_print_example_answers_is_valid(project, tmp_path):
    text = wizard.example_answers_text()
    document = yaml.safe_load(text)
    assert OnboardingAnswers.model_validate(document).persona.name == "Tessera"
    path = tmp_path / "example.yaml"
    path.write_text(text, encoding="utf-8")
    assert wizard.load_answers(path).brain.provider == "mock"
    code, _voice = run(project, path)
    assert code == 0
    env = read_env(project / ENV_NAME)
    assert env["MTC_SYSTEM_NAME"] == "Tessera"
    assert env["MTC_CREATOR_NAME"] == "Alice"
    for store in ("POSTGRES", "MONGODB", "QDRANT", "REDIS", "NEO4J"):
        assert env[f"MTC_{store}_ENABLED"] == "false"
    assert env["MTC_CTM_STORE"] == "memory"


def _answers(**overrides: Any) -> OnboardingAnswers:
    return OnboardingAnswers.model_validate(merged(base_answers(), overrides))


@pytest.mark.parametrize("manager", ["none", "brew", "docker", "systemd"])
def test_next_steps_say_the_mongodb_user_needs_creating(manager, tmp_path):
    """A MongoDB user gets a password in the environment file, and no script
    or Compose service creates that user: the steps say how, without the
    password, and say that the shipped Compose service runs without
    authentication."""
    with_user = wizard.next_steps(
        _answers(
            stores={
                "mongodb": {"enabled": True, "user": "acme_app"},
                "service_manager": manager,
            }
        ),
        tmp_path,
    )
    assert "nothing creates the MongoDB user" in with_user
    assert "db.createUser" in with_user
    assert ("runs without authentication" in with_user) is (manager == "docker")
    without_user = wizard.next_steps(
        _answers(stores={"mongodb": {"enabled": True}, "service_manager": manager}),
        tmp_path,
    )
    assert "MongoDB user" not in without_user


@pytest.mark.parametrize("manager", ["none", "brew", "docker", "systemd"])
def test_next_steps_mark_checkout_only_commands(manager, tmp_path):
    """scripts/ and the Makefile ship only in a repository checkout (and the
    sdist), not in a wheel: every step that names one says so."""
    text = wizard.next_steps(
        _answers(
            brain={"provider": "local"},
            stores={
                "postgres": {"enabled": True},
                "redis": {"enabled": True},
                "service_manager": manager,
            },
        ),
        tmp_path,
    )
    lines = [line for line in text.splitlines() if line.startswith("  - ")]
    checkout = [line for line in lines if "scripts/" in line or "make " in line]
    assert checkout, text
    for line in checkout:
        assert "in a repository checkout" in line, line
