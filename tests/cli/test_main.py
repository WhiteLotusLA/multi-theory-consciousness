"""The ``mtc`` parser and how each subcommand reaches its module."""

import argparse
import os
import socket
import sys
import types
from pathlib import Path

import pytest
import yaml

import mtc
from mtc.cli import main as cli
from mtc.core import config as config_module
from mtc.core.envfile import upsert_env
from mtc.onboarding import wizard
from mtc.onboarding.schema import OnboardingAnswers

LOOPBACK = "127.0.0.1"
PLANTED = "Plantedname"

# Every subcommand with exactly the flags the CLI documents.
DOCUMENTED = {
    "init": {
        "--from",
        "--yes",
        "--force",
        "--bootstrap-db",
        "--dir",
        "--no-probe",
        "--print-plan",
        "--print-example-answers",
    },
    "doctor": {"--json", "--strict", "--fetch-models", "--no-network"},
    "db": set(),
    "assess": set(),
    "serve": set(),
    "chat": {"--message", "--speaker", "--anonymous", "--show-thinking"},
    "ctm": set(),
    "privacy-audit": {"--paths", "--staged", "--patch-from", "--runtime"},
}
DOCUMENTED_NESTED = {
    ("db", "init"): {"--only", "--no-seed", "--json"},
    ("db", "check"): {"--only", "--json", "--strict"},
    ("ctm", "start"): {"--state"},
    ("ctm", "stop"): set(),
    ("ctm", "status"): set(),
}


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MTC_DEVICE", "cpu")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "_settings", None)
    yield tmp_path
    config_module._settings = None


def _subparsers(parser: argparse.ArgumentParser) -> dict:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _flags(parser: argparse.ArgumentParser) -> set:
    found = set()
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith("--") and option != "--help":
                found.add(option)
    return found


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1]


def _fake_module(monkeypatch, package, name: str, **attributes):
    module = types.ModuleType(f"{package.__name__}.{name}")
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(package, name, module, raising=False)
    return module


def test_parser_has_exactly_the_documented_subcommands():
    parser = cli.build_parser()
    commands = _subparsers(parser)
    assert set(commands) == set(DOCUMENTED) == set(cli.SUBCOMMANDS)
    assert set(cli.HANDLERS) == set(cli.SUBCOMMANDS)
    assert _flags(parser) == {"--version", "--project-dir", "--verbose"}
    for name, flags in DOCUMENTED.items():
        assert _flags(commands[name]) == flags, name
    nested = {
        (name, action): _flags(sub)
        for name in ("db", "ctm")
        for action, sub in _subparsers(commands[name]).items()
    }
    assert nested == DOCUMENTED_NESTED
    assert (cli.EXIT_OK, cli.EXIT_FAILED, cli.EXIT_USAGE) == (0, 1, 2)
    assert (cli.EXIT_UNREACHABLE, cli.EXIT_INTERRUPTED) == (3, 130)


def test_version_flag(clean, capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"mtc {mtc.__version__}"
    # The package entry point is the same function.
    from mtc.cli import main as exported

    assert exported.main is cli.main


def test_project_dir_changes_root_before_settings(clean, monkeypatch, tmp_path):
    project = tmp_path / "elsewhere"
    project.mkdir()
    upsert_env(project / wizard.ENV_FILE_NAME, {"MTC_SYSTEM_NAME": "Testagent"})
    seen = {}

    def handler(args, extras):
        settings = config_module.get_settings()
        seen["cwd"] = Path.cwd()
        seen["root"] = settings.PROJECT_ROOT
        seen["name"] = settings.SYSTEM_NAME
        return 0

    monkeypatch.setitem(cli.HANDLERS, "serve", handler)
    # Settings cached before the switch must not survive it.
    assert config_module.get_settings().SYSTEM_NAME != "Testagent"
    assert cli.main(["--project-dir", str(project), "serve"]) == 0
    assert seen["cwd"].resolve() == project.resolve()
    assert Path(seen["root"]).resolve() == project.resolve()
    assert seen["name"] == "Testagent"


def test_usage_errors_exit_2(clean, capsys):
    cases = (
        [],
        ["nonsense"],
        ["doctor", "--no-such-flag"],
        ["db"],
        ["db", "migrate"],
        ["ctm"],
        ["chat", "--anonymous", "--speaker", "Alice", "--message", "hi"],
        ["--project-dir", str(clean / "missing"), "doctor"],
        ["init", "--yes"],
        ["serve", "extra"],
    )
    for arguments in cases:
        assert cli.main(list(arguments)) == 2, arguments
        # Every refusal says why (the wizard speaks on stdout, the rest on
        # stderr).
        said = capsys.readouterr()
        assert (said.out + said.err).strip() != "", arguments
    assert cli.main(["--help"]) == 0
    assert "privacy-audit" in capsys.readouterr().out


def test_keyboard_interrupt_exits_130(clean, monkeypatch, capsys):
    def interrupted(args, extras):
        raise KeyboardInterrupt

    monkeypatch.setitem(cli.HANDLERS, "doctor", interrupted)
    assert cli.main(["doctor"]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_init_dispatches_to_wizard(clean, monkeypatch, capsys):
    calls = []

    def run_init(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(wizard, "run_init", run_init)
    assert cli.main(["init"]) == 0
    assert calls[-1] == {
        "from_path": None,
        "yes": False,
        "force": False,
        "bootstrap_db": False,
        "project_dir": Path.cwd(),
        "probe": True,
        "print_plan": False,
    }
    target = clean / "target"
    assert (
        cli.main(
            [
                "init",
                "--from",
                "answers.yaml",
                "--yes",
                "--force",
                "--bootstrap-db",
                "--dir",
                str(target),
                "--no-probe",
                "--print-plan",
            ]
        )
        == 0
    )
    assert calls[-1] == {
        "from_path": Path("answers.yaml"),
        "yes": True,
        "force": True,
        "bootstrap_db": True,
        "project_dir": target,
        "probe": False,
        "print_plan": True,
    }
    # The wizard's exit code is the command's.
    monkeypatch.setattr(wizard, "run_init", lambda **kwargs: 2)
    assert cli.main(["init"]) == 2

    # The sample answers are printed as they ship and never start the wizard.
    capsys.readouterr()
    calls.clear()
    assert cli.main(["init", "--print-example-answers"]) == 0
    printed = capsys.readouterr().out
    assert printed == wizard.example_answers_text()
    assert OnboardingAnswers.model_validate(yaml.safe_load(printed)).acknowledged
    assert calls == []


def test_db_init_and_check_dispatch_with_only(clean, monkeypatch):
    import mtc.db

    seen = []
    _fake_module(
        monkeypatch,
        mtc.db,
        "bootstrap",
        main=lambda argv=None: seen.append(("init", list(argv))) or 0,
    )
    _fake_module(
        monkeypatch,
        mtc.db,
        "verify",
        main=lambda argv=None: seen.append(("check", list(argv))) or 3,
    )
    assert cli.main(["db", "init"]) == 0
    assert cli.main(["db", "init", "--only", "postgres", "qdrant", "--no-seed"]) == 0
    assert cli.main(["db", "init", "--json"]) == 0
    assert cli.main(["db", "check", "--only", "redis", "--json", "--strict"]) == 3
    assert cli.main(["db", "check"]) == 3
    assert seen == [
        ("init", []),
        ("init", ["--only", "postgres", "qdrant", "--no-seed"]),
        ("init", ["--json"]),
        ("check", ["--only", "redis", "--json", "--strict"]),
        ("check", []),
    ]
    # Flags of one action are not accepted by the other.
    assert cli.main(["db", "check", "--no-seed"]) == 2
    assert cli.main(["db", "init", "--strict"]) == 2


def test_assess_dispatches_to_runner(clean, monkeypatch):
    import mtc.cli as package

    seen = []
    _fake_module(
        monkeypatch,
        package,
        "assess",
        main=lambda argv=None: seen.append(list(argv)) or 0,
    )
    assert cli.main(["assess", "--quick", "--json", "--json-path", "out/x.json"]) == 0
    assert cli.main(["assess"]) == 0
    assert cli.main(["assess", "--help"]) == 0
    assert seen == [
        ["--quick", "--json", "--json-path", "out/x.json"],
        [],
        ["--help"],
    ]

    # The runner's own exit code and its usage exit pass through.
    def failing(argv=None):
        raise SystemExit(2)

    package.assess.main = failing
    assert cli.main(["assess", "--no-such-flag"]) == 2
    package.assess.main = lambda argv=None: 1
    assert cli.main(["assess", "--full"]) == 1


def test_serve_calls_api_main(clean, monkeypatch, capsys):
    import mtc.api as package

    calls = []
    outcome = {"code": 0}
    _fake_module(
        monkeypatch,
        package,
        "server",
        main=lambda argv=None: calls.append(list(argv)) or outcome["code"],
    )
    monkeypatch.setenv("MTC_API_PORT", str(_unused_port()))
    assert cli.main(["serve"]) == 0
    assert calls == [[]]
    # A bind the server refuses (no authentication off loopback) is usage.
    outcome["code"] = 2
    assert cli.main(["serve"]) == 2
    assert len(calls) == 2

    # An address already in use is "unreachable", and the server is not started.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind((LOOPBACK, 0))
        busy.listen(1)
        monkeypatch.setenv("MTC_API_PORT", str(busy.getsockname()[1]))
        config_module._settings = None
        assert cli.main(["serve"]) == 3
        assert len(calls) == 2
        assert "already in use" in capsys.readouterr().err

    # The server library exits instead of raising when it loses the port.
    def exits(argv=None):
        raise SystemExit(1)

    monkeypatch.setenv("MTC_API_PORT", str(_unused_port()))
    config_module._settings = None
    package.server.main = exits
    assert cli.main(["serve"]) == 1


def test_serve_missing_api_extra_exits_2(clean, monkeypatch, capsys):
    import mtc.api as package

    monkeypatch.setenv("MTC_API_PORT", str(_unused_port()))
    monkeypatch.delattr(package, "server", raising=False)
    monkeypatch.setitem(sys.modules, "mtc.api.server", None)
    assert cli.main(["serve"]) == 2
    message = capsys.readouterr().err
    assert "api extra" in message
    assert "multi-theory-consciousness[api]" in message


def _checkout(root: Path) -> Path:
    lists = root / "config" / "privacy"
    lists.mkdir(parents=True)
    (lists / "denylist.txt").write_text(f"# planted\n{PLANTED}\n", encoding="utf-8")
    (lists / "allowlist.txt").write_text(
        "# glob kind regex\nconfig/privacy/denylist.txt * .*\n", encoding="utf-8"
    )
    return root


def test_privacy_audit_uses_repository_lists(clean, capsys):
    root = _checkout(clean / "repo")
    (root / "notes.md").write_text("nothing here\n", encoding="utf-8")
    (root / "leak.md").write_text(f"written for {PLANTED}\n", encoding="utf-8")
    base = ["--project-dir", str(root), "privacy-audit"]

    assert cli.main(base + ["--paths", "notes.md", "config/privacy/denylist.txt"]) == 0
    assert "privacy audit: clean" in capsys.readouterr().out
    assert cli.main(base + ["--paths", "leak.md"]) == 1
    captured = capsys.readouterr()
    assert f"leak.md:1:token:{PLANTED}" in captured.out
    assert "1 hit(s)" in captured.err
    assert cli.main(base + ["--paths", "absent.md"]) == 2

    # Runtime directories anywhere on disk, with the same lists.
    logs = clean / "outside" / "logs"
    logs.mkdir(parents=True)
    (logs / "mtc.log").write_text("INFO ready\n", encoding="utf-8")
    (logs / "events.jsonl").write_text('{"event": "ok"}\n', encoding="utf-8")
    assert cli.main(base + ["--runtime", str(logs)]) == 0
    assert "runtime mode scanned 2 file(s)" in capsys.readouterr().out
    (logs / "events.jsonl").write_text(f'{{"who": "{PLANTED}"}}\n', encoding="utf-8")
    assert cli.main(base + ["--runtime", str(logs)]) == 1
    assert f"logs/events.jsonl:1:token:{PLANTED}" in capsys.readouterr().out
    assert cli.main(base + ["--runtime", str(clean / "no-such-dir")]) == 2
    # The private extra list is honoured, and a dangling reference refuses.
    extra = clean / "extra.txt"
    extra.write_text("Plantedpet\n", encoding="utf-8")
    (root / "pet.md").write_text("the dog Plantedpet\n", encoding="utf-8")
    assert cli.main(base + ["--paths", "pet.md"]) == 0
    os.environ["PRIVACY_DENYLIST_EXTRA"] = str(extra)
    try:
        assert cli.main(base + ["--paths", "pet.md"]) == 1
        assert "private#1" in capsys.readouterr().out
        os.environ["PRIVACY_DENYLIST_EXTRA"] = str(clean / "gone.txt")
        assert cli.main(base + ["--paths", "pet.md"]) == 2
    finally:
        del os.environ["PRIVACY_DENYLIST_EXTRA"]
    # Modes that need git plumbing need the repository's own script.
    assert cli.main(base + ["--staged"]) == 2
    assert "repository checkout" in capsys.readouterr().err
    assert cli.main(base + ["--staged", "--paths", "notes.md"]) == 2


def test_privacy_audit_without_lists_exits_2(clean, capsys):
    empty = clean / "not-a-checkout"
    empty.mkdir()
    for mode in (["--paths", "x.md"], ["--runtime", str(empty)], []):
        assert cli.main(["--project-dir", str(empty), "privacy-audit"] + mode) == 2
        message = capsys.readouterr().err
        assert "config/privacy/denylist.txt" in message
    # With only the denylist present, the missing allowlist is the one named.
    lists = empty / "config" / "privacy"
    lists.mkdir(parents=True)
    (lists / "denylist.txt").write_text("x\n", encoding="utf-8")
    assert cli.main(["--project-dir", str(empty), "privacy-audit", "--paths", "x"]) == 2
    assert "config/privacy/allowlist.txt" in capsys.readouterr().err


def test_privacy_audit_runtime_reports_skipped_files(clean, capsys):
    root = _checkout(clean / "repo")
    logs = clean / "outside" / "logs"
    logs.mkdir(parents=True)
    (logs / "mtc.log").write_text("INFO ready\n", encoding="utf-8")
    (logs / "weights.npz").write_bytes(b"\x00\x01")
    base = ["--project-dir", str(root), "privacy-audit"]
    assert cli.main(base + ["--runtime", str(logs)]) == 0
    out = capsys.readouterr().out
    assert "runtime mode scanned 1 file(s)" in out
    assert "skipped 1 file(s) not read as text (.npz 1)" in out
