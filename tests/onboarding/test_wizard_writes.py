"""What the wizard writes: where each value lands and where no secret does."""

import logging
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from mtc.consciousness.continuous.persona import CTMPersona
from mtc.consciousness.continuous.seeds import SeedLibrary
from mtc.core.checks import CheckResult, CheckStatus
from mtc.core.config import COMPANION_TYPES, Settings
from mtc.core.envfile import read_env, render_value
from mtc.core.logger import reset_logging
from mtc.memory.core_blocks import render_core_blocks
from mtc.onboarding import wizard
from mtc.onboarding.schema import (
    SELF_KNOWLEDGE_LINE,
    TEMPLATES_DIR,
    OnboardingAnswers,
    load_type_defaults,
)
from mtc.onboarding.wizard import plan_writes, run_init
from mtc.pipeline.speaker_trust import SpeakerTrust, token_for

from .test_wizard_noninteractive import (
    ENV_NAME,
    ROOT,
    base_answers,
    make_project,
    merged,
    mode_of,
    run,
    snapshot,
    write_answers,
)

ALL_STORES = {
    "db_prefix": "acme",
    "service_manager": "docker",
    "postgres": {"enabled": True},
    "mongodb": {"enabled": True, "user": "acme_app"},
    "qdrant": {"enabled": True},
    "redis": {"enabled": True},
    "neo4j": {"enabled": True},
}
SECRET_KEYS = (
    "MTC_API_KEY",
    "MTC_POSTGRES_PASSWORD",
    "MTC_MONGODB_PASSWORD",
    "MTC_NEO4J_PASSWORD",
)


@pytest.fixture
def project(monkeypatch, tmp_path):
    return make_project(monkeypatch, tmp_path)


def _secrets_of(project_dir: Path) -> list:
    env = read_env(project_dir / ENV_NAME)
    values = [env[key] for key in SECRET_KEYS if env.get(key)]
    for pair in env["MTC_SPEAKER_TOKENS"].split(","):
        values.append(pair.split(":", 1)[1])
    assert all(len(value) >= 32 for value in values)
    return values


def _settings(project_dir: Path) -> Settings:
    return Settings(
        _env_file=str(project_dir / ENV_NAME), PROJECT_ROOT=str(project_dir)
    )


def test_secrets_only_in_env_file_mode_0600(project, tmp_path):
    code, voice = run(project, write_answers(tmp_path, stores=ALL_STORES))
    assert code == 0
    assert mode_of(project / ENV_NAME) == 0o600
    values = _secrets_of(project)
    # The API key, the creator's token and three generated store passwords
    # (Redis runs without one unless a password is typed).
    assert len(values) == 5
    assert len(set(values)) == 5
    others = {
        name: data.decode("utf-8")
        for name, data in snapshot(project).items()
        if name != ENV_NAME
    }
    assert len(others) == 5
    shown = "\n".join(voice.said)
    for value in values:
        for name, text in others.items():
            assert (value in text) is False, name
        assert (value in shown) is False
        assert (value in repr(voice.questions)) is False
    # The speakers file names the creator and carries no token field at all.
    speakers = yaml.safe_load(others["config/speakers.yaml"])
    assert speakers == {
        "speakers": [{"name": "Alice", "role": "creator", "trust": 0.9}]
    }
    # No temporary file is left behind next to anything written.
    leftovers = [path.name for path in project.rglob("*.tmp")]
    assert leftovers == []


def test_env_keys_written(project, tmp_path):
    answers = write_answers(
        tmp_path,
        persona={"core_values": ["candor", "patience"]},
        locale={"timezone": "Europe/Berlin", "wake_hour": 7, "sleep_hour": 23},
        stores=merged(
            ALL_STORES,
            {
                "postgres": {"enabled": True, "host": "db.internal", "port": 15432},
                "neo4j": {"enabled": True, "port": 17687},
            },
        ),
        brain={
            "provider": "local",
            "port": 18080,
            "model": "example-model",
            "template_family": "qwen",
        },
    )
    assert run(project, answers)[0] == 0
    env = read_env(project / ENV_NAME)
    fields = Settings.model_fields
    expected = {
        # identity
        "MTC_SYSTEM_NAME": "Tessera",
        "MTC_CREATOR_NAME": "Alice",
        "MTC_CREATOR_RELATIONSHIP": "friend",
        "MTC_COMPANION_TYPE": "friend",
        "MTC_CORE_VALUES": "candor,patience",
        # locale
        "MTC_TIMEZONE": "Europe/Berlin",
        "MTC_WAKE_HOUR": "7",
        "MTC_SLEEP_HOUR": "23",
        # stores, with the names the prefix derives written out
        "MTC_DB_PREFIX": "acme",
        "MTC_SERVICE_MANAGER": "docker",
        "MTC_POSTGRES_ENABLED": "true",
        "MTC_POSTGRES_HOST": "db.internal",
        "MTC_POSTGRES_PORT": "15432",
        "MTC_POSTGRES_DB": "acme_consciousness",
        "MTC_POSTGRES_USER": "acme",
        "MTC_MONGODB_ENABLED": "true",
        "MTC_MONGODB_HOST": fields["MONGODB_HOST"].default,
        "MTC_MONGODB_PORT": str(fields["MONGODB_PORT"].default),
        "MTC_MONGODB_DB": "acme_neural",
        "MTC_MONGODB_USER": "acme_app",
        "MTC_QDRANT_ENABLED": "true",
        "MTC_QDRANT_HOST": fields["QDRANT_HOST"].default,
        "MTC_QDRANT_PORT": str(fields["QDRANT_PORT"].default),
        "MTC_QDRANT_COLLECTION": "acme_memories",
        "MTC_REDIS_ENABLED": "true",
        "MTC_REDIS_HOST": fields["REDIS_HOST"].default,
        "MTC_REDIS_PORT": str(fields["REDIS_PORT"].default),
        "MTC_REDIS_KEY_PREFIX": "acme:",
        "MTC_NEO4J_ENABLED": "true",
        "MTC_NEO4J_URI": fields["NEO4J_URI"].default.replace("7687", "17687"),
        "MTC_NEO4J_USER": fields["NEO4J_USER"].default,
        "MTC_NEO4J_DATABASE": fields["NEO4J_DATABASE"].default,
        "MTC_CTM_STORE": "mongo",
        # brain
        "MTC_LLM_PROVIDER": "local",
        "MTC_LLM_HOST": fields["LLM_HOST"].default,
        "MTC_LLM_PORT": "18080",
        "MTC_LLM_BASE_URL": "",
        "MTC_LLM_MODEL": "example-model",
        "MTC_LLM_TEMPLATE_FAMILY": "qwen",
        # API, authenticated
        "MTC_API_HOST": fields["API_HOST"].default,
        "MTC_API_PORT": "18765",
        "MTC_API_AUTH_ENABLED": "true",
        # trust, through the trust module's helper
        "MTC_TRUSTED_SPEAKERS": "Alice",
    }
    for key, value in expected.items():
        assert env.get(key) == value, key
    secret_keys = set(SECRET_KEYS) | {"MTC_SPEAKER_TOKENS"}
    assert set(env) == set(expected) | secret_keys
    for key in secret_keys:
        assert bool(env[key]) is True, key
    assert env["MTC_SPEAKER_TOKENS"].startswith("Alice:")
    # Nothing is invented for a server that runs without credentials.
    assert "MTC_REDIS_PASSWORD" not in env
    assert "MTC_MONGODB_AUTH_SOURCE" not in env
    # The file loads as Settings and says the same thing.
    cfg = _settings(project)
    assert cfg.POSTGRES_DB == "acme_consciousness"
    assert cfg.QDRANT_COLLECTION == "acme_memories"
    assert cfg.API_AUTH_ENABLED is True and bool(cfg.API_KEY) is True
    assert cfg.LLM_BASE_URL.endswith(":18080")

    # The Anthropic keys appear only for that provider.
    assert "ANTHROPIC_API_KEY" not in env and "MTC_ANTHROPIC_MODEL" not in env


def test_anthropic_key_from_environment_reference(project, tmp_path, monkeypatch):
    monkeypatch.setenv("WIZARD_TEST_ANTHROPIC", "sample-key-value-000")
    answers = write_answers(
        tmp_path,
        brain={
            "provider": "anthropic",
            "anthropic_model": "example-model",
            "anthropic_api_key_env": "WIZARD_TEST_ANTHROPIC",
        },
    )
    code, voice = run(project, answers)
    assert code == 0
    env = read_env(project / ENV_NAME)
    assert env["MTC_LLM_PROVIDER"] == "anthropic"
    assert env["MTC_ANTHROPIC_MODEL"] == "example-model"
    assert (env["ANTHROPIC_API_KEY"] == "sample-key-value-000") is True
    assert "MTC_LLM_HOST" not in env
    assert ("sample-key-value-000" in "\n".join(voice.said)) is False
    assert "external service" in "\n".join(voice.said)
    for name, data in snapshot(project).items():
        if name != ENV_NAME:
            assert (b"sample-key-value-000" in data) is False, name


def test_speakers_file_loads_in_speaker_trust(project, tmp_path):
    assert run(project, write_answers(tmp_path))[0] == 0
    cfg = _settings(project)
    trust = SpeakerTrust.from_settings(cfg)
    assert trust.status()["by_role"] == {"creator": 1, "companion": 0, "guest": 0}
    assert trust.status()["with_token_env"] == 0

    token = token_for("Alice", cfg)
    assert isinstance(token, str) and len(token) == 64
    verified = trust.verify("Alice", token)
    assert verified.authenticated is True
    assert verified.trusted is True
    assert verified.role == "creator"
    assert verified.trust == pytest.approx(0.9)

    # The name alone, a wrong token and another name all get the default.
    for name, presented in (("Alice", None), ("Alice", "0" * 64), ("Guest", token)):
        identity = trust.verify(name, presented)
        assert identity.authenticated is False
        assert identity.trusted is False
        assert identity.role is None
        assert identity.trust == pytest.approx(cfg.DEFAULT_SPEAKER_TRUST)

    # Declining the trusted list keeps the token and drops the allowlist entry.
    untrusted = write_answers(tmp_path, persona={"creator_trusted": False})
    assert run(project, untrusted, force=True)[0] == 0
    cfg = _settings(project)
    again = SpeakerTrust.from_settings(cfg).verify("Alice", token_for("Alice", cfg))
    assert again.authenticated is True and again.trusted is False
    assert (token_for("Alice", cfg) == token) is True


@pytest.mark.parametrize("kind", ["child", "partner", "research_subject"])
def test_identity_statement_from_type_template(project, tmp_path, kind):
    answers = write_answers(tmp_path, persona={"companion_type": kind})
    assert run(project, answers)[0] == 0
    written = (project / "agent_space" / "identity_statement.md").read_text()
    template = (TEMPLATES_DIR / "identity" / f"{kind}.md").read_text()
    assert written == template
    # Placeholders stay for the prompt assembler; no name is baked in.
    assert SELF_KNOWLEDGE_LINE in written.splitlines()
    assert "{system_name}" in written and "Tessera" not in written
    env = read_env(project / ENV_NAME)
    row = load_type_defaults()[kind]
    assert env["MTC_CREATOR_RELATIONSHIP"] == row.creator_relationship
    speakers = yaml.safe_load((project / "config" / "speakers.yaml").read_text())
    assert speakers["speakers"][0]["trust"] == pytest.approx(row.creator_trust)


def test_ctm_persona_and_seed_pack_selected(project, tmp_path):
    seeds_dir = project / "config" / "ctm" / "seeds.d"
    default_pack = project / "config" / "ctm" / "seeds" / "default.yaml"
    default_pack.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "config" / "ctm" / "seeds" / "default.yaml", default_pack)

    assert (
        run(project, write_answers(tmp_path, persona={"companion_type": "mentor"}))[0]
        == 0
    )
    assert sorted(path.name for path in seeds_dir.iterdir()) == ["10_reflection.yaml"]
    cfg = _settings(project)
    persona = CTMPersona.load(settings=cfg)
    assert persona.source == "configured"
    assert persona.name == "Tessera"
    tone = load_type_defaults()["mentor"].ctm_tone
    assert persona.system_prompt.endswith(tone)
    assert "{name}" in persona.system_prompt
    library = SeedLibrary.from_config_dir(settings=cfg)
    assert [path.name for path in library.packs] == [
        "default.yaml",
        "10_reflection.yaml",
    ]
    assert "What is the clearest way to explain this?" in library.entries("questions")

    # Operator packs sit beside the wizard's and are never touched.
    (seeds_dir / "20_mine.yaml").write_text('questions:\n  - "Why?"\n')
    (seeds_dir / "10_custom.yaml").write_text('questions:\n  - "How?"\n')
    child = write_answers(tmp_path, persona={"companion_type": "child"})
    assert run(project, child, force=True)[0] == 0
    assert sorted(path.name for path in seeds_dir.iterdir()) == [
        "10_custom.yaml",
        "10_growth.yaml",
        "20_mine.yaml",
    ]
    engine = write_answers(tmp_path, persona={"companion_type": "research_subject"})
    assert run(project, engine, force=True)[0] == 0
    assert sorted(path.name for path in seeds_dir.iterdir()) == [
        "10_custom.yaml",
        "20_mine.yaml",
    ]
    assert CTMPersona.load(settings=_settings(project)).system_prompt.endswith(
        load_type_defaults()["research_subject"].ctm_tone
    )


def test_every_written_path_is_gitignored(project, tmp_path):
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout: the ignore rules cannot be evaluated")
    answers = OnboardingAnswers.model_validate(
        merged(base_answers(), {"persona": {"companion_type": "child"}})
    )
    plan = plan_writes(answers, project, existing_env={})
    relatives = [planned.path.relative_to(project).as_posix() for planned in plan.files]
    relatives.append(plan.env_path.relative_to(project).as_posix())
    assert len(relatives) == 6
    for relative in relatives:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", relative],
            cwd=ROOT,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, f"{relative} is not gitignored"
    # The check is real: a tracked source path is not ignored.
    control = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "mtc/onboarding/wizard.py"],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )
    assert control.returncode == 1


def test_identity_written_to_persona_and_env_consistently(project, tmp_path):
    answers = write_answers(
        tmp_path,
        persona={
            "name": "Testagent",
            "creator_name": "Alice Example",
            "companion_type": "sibling",
            "core_values": ["candor", "play"],
            "pronouns": {"subject": "she", "object": "her", "possessive": "her"},
            "personality_preset": "playful",
        },
    )
    assert run(project, answers)[0] == 0
    env = read_env(project / ENV_NAME)
    persona = yaml.safe_load((project / "config" / "persona.yaml").read_text())
    assert persona["name"] == env["MTC_SYSTEM_NAME"] == "Testagent"
    assert persona["creator"]["name"] == env["MTC_CREATOR_NAME"] == "Alice Example"
    assert persona["companion_type"] == env["MTC_COMPANION_TYPE"] == "sibling"
    assert ",".join(persona["core_values"]) == env["MTC_CORE_VALUES"] == "candor,play"
    # The sibling default: the first name.
    assert persona["creator"]["address_form"] == "Alice"
    assert persona["pronouns"]["possessive"] == "her"
    assert persona["personality"]["extraversion"] == pytest.approx(0.75)
    assert "Testagent calls Alice" in " ".join(persona["self_description"].split())
    speakers = yaml.safe_load((project / "config" / "speakers.yaml").read_text())
    assert speakers["speakers"][0]["name"] == "Alice Example"
    assert env["MTC_SPEAKER_TOKENS"].startswith("Alice Example:")
    # A name with a space is still read back whole.
    assert _settings(project).CREATOR_NAME == "Alice Example"


def test_ctm_store_mongo_only_with_mongodb(project, tmp_path):
    cases = (
        ({}, "memory"),
        ({"postgres": {"enabled": True}}, "memory"),
        ({"postgres": {"enabled": True}, "mongodb": {"enabled": True}}, "mongo"),
    )
    for index, (stores, expected) in enumerate(cases):
        target = project / f"case{index}"
        assert run(target, write_answers(tmp_path, stores=stores))[0] == 0
        env = read_env(target / ENV_NAME)
        assert env["MTC_CTM_STORE"] == expected
        assert env["MTC_MONGODB_ENABLED"] == (
            "true" if expected == "mongo" else "false"
        )
        # A local MongoDB runs without credentials: none is invented.
        assert "MTC_MONGODB_PASSWORD" not in env
        assert "MTC_REDIS_PASSWORD" not in env
        assert _settings(target).CTM_STORE == expected


def _fake_bootstrap(monkeypatch, status=CheckStatus.OK):
    calls = []

    async def bootstrap(settings=None, *, env_path=None, **kwargs):
        calls.append((settings, env_path, kwargs))
        return types.SimpleNamespace(
            results=[CheckResult("birth_date", status, "recorded")],
            birth_date="2000-01-01",
            birth_date_written=True,
            seed=None,
        )

    module = types.ModuleType("mtc.db.bootstrap")
    module.bootstrap = bootstrap
    import mtc.db

    monkeypatch.setitem(sys.modules, "mtc.db.bootstrap", module)
    monkeypatch.setattr(mtc.db, "bootstrap", module, raising=False)
    return calls


def test_bootstrap_flag_calls_bootstrap_with_fresh_settings(
    project, tmp_path, monkeypatch
):
    calls = _fake_bootstrap(monkeypatch)
    # Without the flag and with bootstrap_db false, nothing is bootstrapped.
    assert run(project / "plain", write_answers(tmp_path))[0] == 0
    assert calls == []

    code, voice = run(project, write_answers(tmp_path), bootstrap_db=True)
    assert code == 0
    assert len(calls) == 1
    settings, env_path, extra = calls[0]
    assert extra == {}
    assert env_path == project / ENV_NAME
    assert isinstance(settings, Settings)
    assert settings.PROJECT_ROOT == str(project)
    # Fresh settings: they carry what this run just wrote.
    assert settings.SYSTEM_NAME == "Tessera"
    assert settings.API_PORT == 18765
    assert bool(settings.API_KEY) is True
    assert settings.SYSTEM_BIRTH_DATE == ""
    text = "\n".join(voice.said)
    assert "Bootstrap" in text and "Install date: 2000-01-01" in text
    # Bootstrapped already, so the next steps do not ask for it again.
    assert "mtc db init" not in text

    # bootstrap_db in the answers does the same as the flag.
    from_answers = project.parent / "answers_flag"
    assert run(from_answers, write_answers(tmp_path, bootstrap_db=True))[0] == 0
    assert len(calls) == 2 and calls[1][1] == from_answers / ENV_NAME

    # A bootstrap problem is reported with its exit code after the write.
    calls_failed = _fake_bootstrap(monkeypatch, CheckStatus.UNREACHABLE)
    failed_dir = project.parent / "unreachable"
    code, voice = run(failed_dir, write_answers(tmp_path), bootstrap_db=True)
    assert code == 3 and len(calls_failed) == 1
    assert (failed_dir / "config" / "persona.yaml").is_file()
    assert "bootstrap reported a problem" in "\n".join(voice.said)


def test_no_secret_in_stdout_or_logs(project, tmp_path, capsys, monkeypatch):
    _fake_bootstrap(monkeypatch)
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.DEBUG)
    root_logger = logging.getLogger()
    mtc_logger = logging.getLogger("mtc")
    levels = (root_logger.level, mtc_logger.level)
    root_logger.addHandler(handler)
    mtc_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)
    mtc_logger.setLevel(logging.DEBUG)
    try:
        answers = write_answers(tmp_path, stores=ALL_STORES)
        # The console path: everything the wizard says goes to stdout.
        code = run_init(
            from_path=answers,
            yes=True,
            force=False,
            bootstrap_db=True,
            project_dir=project,
            probe=False,
            print_plan=False,
            prompter=None,
        )
        assert code == 0
        again = run_init(
            from_path=answers,
            yes=True,
            force=True,
            bootstrap_db=False,
            project_dir=project,
            probe=False,
            print_plan=True,
            prompter=None,
        )
        assert again == 0
    finally:
        root_logger.removeHandler(handler)
        mtc_logger.removeHandler(handler)
        root_logger.setLevel(levels[0])
        mtc_logger.setLevel(levels[1])
        reset_logging()
    captured = capsys.readouterr()
    output = captured.out + captured.err
    logged = "\n".join(records)
    assert "Wrote 6 files" in output
    assert any("setup wrote" in line for line in records)
    values = _secrets_of(project)
    assert len(values) == 5
    for value in values:
        assert (value in output) is False
        assert (value in logged) is False
    assert re.search(r"[0-9a-f]{32,}", output) is None
    assert re.search(r"[0-9a-f]{32,}", logged) is None


def test_generated_passwords_are_safe_values(project, tmp_path):
    assert run(project, write_answers(tmp_path, stores=ALL_STORES))[0] == 0
    env = read_env(project / ENV_NAME)
    for key in SECRET_KEYS[1:]:
        value = env[key]
        assert re.fullmatch(r"[0-9a-f]{32}", value) is not None, key
        assert (render_value(value) == value) is True, key
    assert re.fullmatch(r"[0-9a-f]{64}", env["MTC_API_KEY"]) is not None
    token = env["MTC_SPEAKER_TOKENS"].split(":", 1)[1]
    assert re.fullmatch(r"[0-9a-f]{64}", token) is not None
    # Two runs never generate the same values.
    other = project.parent / "second"
    assert run(other, write_answers(tmp_path, stores=ALL_STORES))[0] == 0
    env_other = read_env(other / ENV_NAME)
    for key in SECRET_KEYS + ("MTC_SPEAKER_TOKENS",):
        assert (env[key] == env_other[key]) is False, key

    # A typed password with a character the file cannot carry is refused
    # before anything is written, and it is not echoed.
    answers = OnboardingAnswers.model_validate(
        merged(base_answers(), {"stores": {"postgres": {"enabled": True}}})
    )
    plan = plan_writes(answers, project.parent / "third", existing_env={})
    plan.env_updates["MTC_POSTGRES_PASSWORD"] = "bad$value"
    with pytest.raises(ValueError) as caught:
        wizard.apply_plan(plan, force=False)
    assert "bad$value" not in str(caught.value)
    assert not (project.parent / "third").exists()


@pytest.mark.parametrize("kind", COMPANION_TYPES)
def test_wizard_persona_renders_clean_core_blocks(project, tmp_path, kind):
    answers = write_answers(tmp_path, persona={"companion_type": kind})
    assert run(project, answers)[0] == 0
    persona = yaml.safe_load((project / "config" / "persona.yaml").read_text())
    blocks = render_core_blocks(persona, settings=_settings(project))
    line = SELF_KNOWLEDGE_LINE.format(system_name=persona["name"])
    assert blocks["identity"].count(line) == 1
    assert "Pronouns: they/them/their." in blocks["identity"]
    for name, text in blocks.items():
        assert "{" not in text and "}" not in text, name
        assert ".." not in text, (name, text)
        assert text.isascii(), name
