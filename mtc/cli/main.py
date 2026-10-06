"""
The ``mtc`` Command
===================

One entry point for setup, checks and day-to-day use::

    mtc [--version] [--project-dir PATH] [--verbose] COMMAND ...

    init           the setup wizard (mtc.onboarding.wizard)
    doctor         check the install (mtc.cli.doctor)
    db init|check  prepare or verify the enabled stores (mtc.db)
    assess         run the consciousness assessment (mtc.cli.assess)
    serve          start the API server (mtc.api.server)
    chat           talk through a running server
    ctm            start, stop or inspect continuous thought on a server
    privacy-audit  audit a checkout or runtime directories for private data

Exit codes, shared by every subcommand: 0 success; 1 a required check or the
command's own work failed; 2 a usage or configuration error (a missing extra
and a refused overwrite included); 3 a required service is unreachable; 130
interrupted.

``--project-dir`` changes the working directory before any Settings load, so
``PROJECT_ROOT`` and the environment file follow it. Every handler imports
its heavy modules when it runs: importing this module loads no store client,
no web framework and no HTTP client.

Config keys read here: ``CREATOR_NAME`` (the default speaker of ``mtc chat``;
trust still needs the speaker's token, which comes from
``mtc.pipeline.speaker_trust.token_for``), ``API_HOST`` and ``API_PORT`` (the
port check of ``mtc serve``) and ``PROJECT_ROOT`` (the privacy lists). No
secret is printed: the chat client sends the API key and the speaker token
in headers and shows neither.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNREACHABLE = 3
EXIT_INTERRUPTED = 130

PROG = "mtc"
SUBCOMMANDS = (
    "init",
    "doctor",
    "db",
    "assess",
    "serve",
    "chat",
    "ctm",
    "privacy-audit",
)
CTM_STATES = ("awake", "drowsy", "dreaming", "deep_sleep")
PRIVACY_DIR = Path("config") / "privacy"
PRIVACY_SCRIPT = Path("scripts") / "privacy_audit.py"
PRIVACY_EXTRA_VARIABLE = "PRIVACY_DENYLIST_EXTRA"
QUIT_WORDS = ("/quit", "/exit")

# What a chat turn's HTTP status means, in plain words.
CHAT_STATUS_MESSAGES: Dict[int, str] = {
    401: "The server refused the API key. Check API_KEY in the environment file.",
    403: "The safety gate blocked this message. Nothing was stored.",
    422: "The server rejected the request (an empty or oversized message).",
    502: "The model gave no answer. Nothing was stored.",
    503: "The brain is unavailable: the model server is down or still loading.",
    504: "The model took too long and the turn was cancelled. Nothing was stored.",
}


class UsageError(Exception):
    """A configuration or usage problem the command reports and exits 2 on."""


def _say(text: str) -> None:
    print(text)


def _complain(text: str) -> None:
    print(f"{PROG}: {text}", file=sys.stderr)


def _version() -> str:
    from mtc import __version__

    return str(__version__)


def _load_settings() -> Any:
    """The process Settings, or ``UsageError`` naming the fields that fail."""
    from mtc.core.config import get_settings

    try:
        return get_settings()
    except Exception as exc:  # noqa: BLE001 - named by field, never by value
        names: List[str] = []
        errors = getattr(exc, "errors", None)
        if callable(errors):
            try:
                names = sorted({str(e["loc"][0]) for e in errors() if e.get("loc")})
            except Exception:  # noqa: BLE001
                names = []
        detail = f" (fields: {', '.join(names)})" if names else ""
        raise UsageError(f"the settings do not load{detail}") from None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The ``mtc`` argument parser with every subcommand."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Set up, check and use a Multi-Theory Consciousness install.",
        epilog="Exit codes: 0 ok, 1 failed, 2 usage or configuration, "
        "3 a required service is unreachable, 130 interrupted.",
    )
    parser.add_argument("--version", action="store_true", help="print the version")
    parser.add_argument(
        "--project-dir",
        metavar="PATH",
        help="run as if started in PATH (settings and files follow it)",
    )
    parser.add_argument("--verbose", action="store_true", help="log to stderr")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    init = commands.add_parser("init", help="set up a project (the wizard)")
    init.add_argument(
        "--from", dest="from_path", metavar="PATH", help="read answers from a file"
    )
    init.add_argument("--yes", action="store_true", help="write without asking")
    init.add_argument(
        "--force", action="store_true", help="replace files an earlier run wrote"
    )
    init.add_argument(
        "--bootstrap-db",
        action="store_true",
        help="record the install date and prepare the enabled stores",
    )
    init.add_argument("--dir", metavar="PATH", help="write into this directory")
    init.add_argument(
        "--no-probe", action="store_true", help="never contact a server or a store"
    )
    init.add_argument(
        "--print-plan", action="store_true", help="show what would be written"
    )
    init.add_argument(
        "--print-example-answers",
        action="store_true",
        help="print a sample answers file",
    )

    doctor = commands.add_parser("doctor", help="check the install")
    doctor.add_argument("--json", action="store_true", help="print JSON")
    doctor.add_argument(
        "--strict",
        action="store_true",
        help="count warnings and insufficient_data as failures",
    )
    doctor.add_argument(
        "--fetch-models", action="store_true", help="download the embedding model"
    )
    doctor.add_argument(
        "--no-network", action="store_true", help="probe no store and no server"
    )

    db = commands.add_parser("db", help="prepare or verify the enabled stores")
    db_commands = db.add_subparsers(dest="db_command", metavar="ACTION")
    db_init = db_commands.add_parser("init", help="idempotent bootstrap")
    db_init.add_argument("--only", nargs="+", metavar="STORE")
    db_init.add_argument("--no-seed", action="store_true", help="skip the seeding")
    db_init.add_argument("--json", action="store_true", help="print JSON")
    db_check = db_commands.add_parser("check", help="verify against the registry")
    db_check.add_argument("--only", nargs="+", metavar="STORE")
    db_check.add_argument("--json", action="store_true", help="print JSON")
    db_check.add_argument(
        "--strict",
        action="store_true",
        help="count warnings and insufficient_data as failures",
    )

    commands.add_parser(
        "assess",
        help="run the assessment (its own flags follow; try mtc assess --help)",
        add_help=False,
    )
    commands.add_parser("serve", help="start the API server")

    chat = commands.add_parser("chat", help="talk through a running server")
    chat.add_argument("--message", metavar="TEXT", help="send one message and exit")
    chat.add_argument(
        "--speaker", metavar="NAME", help="speak as NAME (default: the creator)"
    )
    chat.add_argument(
        "--anonymous", action="store_true", help="send no name and no token"
    )
    chat.add_argument(
        "--show-thinking", action="store_true", help="also print the reasoning"
    )

    ctm = commands.add_parser("ctm", help="continuous thought on a running server")
    ctm_commands = ctm.add_subparsers(dest="ctm_command", metavar="ACTION")
    ctm_start = ctm_commands.add_parser("start", help="start the thought stream")
    ctm_start.add_argument("--state", choices=CTM_STATES, metavar="STATE")
    ctm_commands.add_parser("stop", help="stop the thought stream")
    ctm_commands.add_parser("status", help="show the thought stream's status")

    audit = commands.add_parser(
        "privacy-audit", help="audit a checkout or runtime directories"
    )
    mode = audit.add_mutually_exclusive_group()
    mode.add_argument("--paths", nargs="+", metavar="PATH")
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--patch-from", metavar="REF")
    mode.add_argument("--runtime", nargs="+", metavar="DIR")
    return parser


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace, extras: List[str]) -> int:
    from mtc.onboarding import wizard

    if args.print_example_answers:
        sys.stdout.write(wizard.example_answers_text())
        return EXIT_OK
    project_dir = Path(args.dir).expanduser() if args.dir else Path.cwd()
    return wizard.run_init(
        from_path=Path(args.from_path).expanduser() if args.from_path else None,
        yes=bool(args.yes),
        force=bool(args.force),
        bootstrap_db=bool(args.bootstrap_db),
        project_dir=project_dir,
        probe=not args.no_probe,
        print_plan=bool(args.print_plan),
    )


def _cmd_doctor(args: argparse.Namespace, extras: List[str]) -> int:
    from mtc.cli import doctor
    from mtc.core.checks import exit_code_for, render_text, to_json

    results = doctor.run_doctor(
        network=not args.no_network, fetch_models=bool(args.fetch_models)
    )
    code = exit_code_for(results, strict=bool(args.strict))
    if args.json:
        payload = to_json(results, strict=bool(args.strict))
        payload["exit_code"] = code
        payload["version"] = _version()
        _say(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _say(render_text(results, title=f"mtc doctor ({_version()})"))
    return code


def _cmd_db(args: argparse.Namespace, extras: List[str]) -> int:
    if args.db_command not in ("init", "check"):
        raise UsageError("mtc db needs an action: init or check")
    forwarded: List[str] = []
    if args.only:
        forwarded += ["--only", *args.only]
    if args.json:
        forwarded.append("--json")
    if args.db_command == "init":
        from mtc.db import bootstrap

        if args.no_seed:
            forwarded.append("--no-seed")
        return int(bootstrap.main(forwarded))
    from mtc.db import verify

    if args.strict:
        forwarded.append("--strict")
    return int(verify.main(forwarded))


def _cmd_assess(args: argparse.Namespace, extras: List[str]) -> int:
    from mtc.cli import assess

    return int(assess.main(list(extras)) or 0)


def _port_in_use(host: str, port: int) -> bool:
    """True when binding the server's address fails because it is taken."""
    import errno
    import socket

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    family, kind, proto, _name, address = infos[0]
    probe = socket.socket(family, kind, proto)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(address)
    except OSError as exc:
        return exc.errno == errno.EADDRINUSE
    finally:
        probe.close()
    return False


def _cmd_serve(args: argparse.Namespace, extras: List[str]) -> int:
    import errno

    cfg = _load_settings()
    try:
        # Imported when the command runs: this handler is the one caller of
        # the API package from outside it, and the tree-wide audit allows
        # exactly this import, inside this function.
        import mtc.api.server as server
    except ImportError as exc:
        _complain(
            "mtc serve needs the api extra (pip install "
            f"'multi-theory-consciousness[api]'); missing module: {exc.name}"
        )
        return EXIT_USAGE
    host, port = str(cfg.API_HOST), int(cfg.API_PORT)
    if _port_in_use(host, port):
        _complain(
            f"port {port} on {host} is already in use: another server is "
            "running there, or API_PORT needs another value"
        )
        return EXIT_UNREACHABLE
    try:
        return int(server.main([]))
    except ImportError as exc:
        _complain(f"mtc serve needs the api extra: {exc}")
        return EXIT_USAGE
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            _complain(f"port {port} on {host} is already in use")
            return EXIT_UNREACHABLE
        raise
    except SystemExit as exc:
        # The server library exits on a failed bind instead of raising.
        if _port_in_use(host, port):
            _complain(f"port {port} on {host} is already in use")
            return EXIT_UNREACHABLE
        return EXIT_FAILED if exc.code else EXIT_OK


def _chat_identity(args: argparse.Namespace, cfg: Any) -> tuple:
    """The speaker name and token a chat turn presents, or ``(None, None)``."""
    if args.anonymous:
        if args.speaker:
            raise UsageError("--anonymous and --speaker cannot be combined")
        return None, None
    from mtc.pipeline import speaker_trust

    speaker = args.speaker or cfg.CREATOR_NAME
    try:
        token = speaker_trust.token_for(speaker, cfg)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    return speaker, token


def _chat_turn(client: Any, message: str, speaker: Any, token: Any, show: bool) -> int:
    from mtc.cli.http import ApiUnreachable

    try:
        status, payload = client.chat(message, speaker, token)
    except ApiUnreachable:
        _complain(
            f"no API server answered at {client.where}. Start it with mtc serve, "
            "then try again."
        )
        return EXIT_UNREACHABLE
    if status == 200 and payload.get("success", True):
        if show and payload.get("thinking"):
            _say(f"[thinking] {payload['thinking']}")
        _say(str(payload.get("response") or ""))
        return EXIT_OK
    message_text = CHAT_STATUS_MESSAGES.get(status)
    if status == 503 and "authentication" in str(payload.get("detail", "")):
        message_text = (
            "The server has authentication on but no API key configured. Set "
            "API_KEY (mtc init generates one) and restart it."
        )
    if message_text is None:
        message_text = f"The turn failed (HTTP {status})."
    _complain(message_text)
    return EXIT_FAILED


def _cmd_chat(args: argparse.Namespace, extras: List[str]) -> int:
    from mtc.cli.http import ApiClient

    cfg = _load_settings()
    speaker, token = _chat_identity(args, cfg)
    client = ApiClient.from_settings(cfg)
    if speaker is not None and token is None:
        _complain(
            "no token is configured for this speaker, so the server will treat "
            "the name as unverified"
        )
    if args.message is not None:
        if not args.message.strip():
            raise UsageError("--message needs text")
        return _chat_turn(client, args.message, speaker, token, args.show_thinking)
    _say("Type a message and press Enter. /quit or end of input leaves.")
    code = EXIT_OK
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        text = line.strip()
        if not text:
            continue
        if text.lower() in QUIT_WORDS:
            break
        code = _chat_turn(client, text, speaker, token, args.show_thinking)
        if code == EXIT_UNREACHABLE:
            return code
    return EXIT_OK if code != EXIT_UNREACHABLE else code


def _cmd_ctm(args: argparse.Namespace, extras: List[str]) -> int:
    from mtc.cli.http import ApiClient, ApiUnreachable

    if args.ctm_command not in ("start", "stop", "status"):
        raise UsageError("mtc ctm needs an action: start, stop or status")
    client = ApiClient.from_settings(_load_settings())
    try:
        if args.ctm_command == "start":
            status, payload = client.ctm_start(args.state)
        elif args.ctm_command == "stop":
            status, payload = client.ctm_stop()
        else:
            status, payload = client.ctm_status()
    except ApiUnreachable:
        _complain(f"no API server answered at {client.where}. Start it with mtc serve.")
        return EXIT_UNREACHABLE
    if status == 401:
        _complain(CHAT_STATUS_MESSAGES[401])
        return EXIT_FAILED
    if status != 200:
        detail = str(payload.get("detail") or f"HTTP {status}")
        _complain(f"continuous thought: {detail}")
        return EXIT_FAILED
    inner = payload.get("status") if isinstance(payload.get("status"), dict) else {}
    if not payload.get("available", False):
        _say("continuous thought: not built on this server (CTM_ENABLED is false)")
        return EXIT_FAILED if args.ctm_command != "status" else EXIT_OK
    _say(f"running: {'yes' if payload.get('running') else 'no'}")
    _say(f"state: {inner.get('consciousness_state', 'unknown')}")
    frequency = inner.get("current_frequency")
    _say(f"frequency: {frequency if frequency is not None else 'unknown'}")
    _say(f"quiet reason: {payload.get('quiet_reason') or 'none'}")
    return EXIT_OK


def _privacy_lists(root: Path) -> tuple:
    from mtc.devtools import privacy_audit as audit

    lists = root / PRIVACY_DIR
    denylist_path = lists / "denylist.txt"
    allowlist_path = lists / "allowlist.txt"
    for path in (denylist_path, allowlist_path):
        if not path.is_file():
            raise UsageError(
                f"{path.relative_to(root).as_posix()} not found under the project "
                "directory: mtc privacy-audit runs in a repository checkout "
                "(pass --project-dir)"
            )
    extra: List[str] = []
    named = os.environ.get(PRIVACY_EXTRA_VARIABLE)
    local = lists / "denylist.local.txt"
    if named:
        if not Path(named).is_file():
            raise UsageError(
                f"{PRIVACY_EXTRA_VARIABLE} names a file that does not exist; "
                "refusing to audit without the private list"
            )
        extra = audit.load_denylist(Path(named))
    elif local.is_file():
        extra = audit.load_denylist(local)
    return (
        audit,
        audit.load_denylist(denylist_path),
        audit.load_allowlist(allowlist_path),
        extra,
    )


def _cmd_privacy_audit(args: argparse.Namespace, extras: List[str]) -> int:
    from mtc.core.paths import project_root

    root = project_root(_load_settings())
    audit, denylist, allowlist, extra = _privacy_lists(root)
    if args.runtime:
        directories = [Path(item).expanduser().resolve() for item in args.runtime]
        missing = [path.name for path in directories if not path.is_dir()]
        if missing:
            raise UsageError(f"not a directory: {', '.join(missing)}")
        hits = audit.run_runtime_audit(
            directories, denylist, allowlist, extra_denylist=extra
        )
        _say(audit.runtime_summary(directories))
    elif args.paths:
        files = [(root / item).resolve() for item in args.paths]
        missing = [
            str(item) for item, path in zip(args.paths, files) if not path.exists()
        ]
        if missing:
            raise UsageError(f"no such path: {', '.join(missing)}")
        try:
            hits = audit.run_audit(
                files, denylist, allowlist, extra_denylist=extra, root=root
            )
        except ValueError:
            raise UsageError(
                "--paths takes paths inside the project directory"
            ) from None
    else:
        # The tracked tree, the index and a commit range need git plumbing,
        # which lives in the repository's own script.
        import subprocess

        script = root / PRIVACY_SCRIPT
        if not script.is_file():
            raise UsageError(
                f"{PRIVACY_SCRIPT.as_posix()} not found under the project "
                "directory: this mode needs a repository checkout"
            )
        command = [sys.executable, str(script)]
        if args.staged:
            command.append("--staged")
        elif args.patch_from:
            command += ["--patch-from", args.patch_from]
        return int(subprocess.run(command, cwd=root, check=False).returncode)
    for hit in hits:
        _say(str(hit))
    if hits:
        _complain(f"privacy audit: {len(hits)} hit(s)")
        return EXIT_FAILED
    _say("privacy audit: clean")
    return EXIT_OK


HANDLERS: Dict[str, Callable[[argparse.Namespace, List[str]], int]] = {
    "init": _cmd_init,
    "doctor": _cmd_doctor,
    "db": _cmd_db,
    "assess": _cmd_assess,
    "serve": _cmd_serve,
    "chat": _cmd_chat,
    "ctm": _cmd_ctm,
    "privacy-audit": _cmd_privacy_audit,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _enter_project(path: str) -> None:
    """Make ``path`` the working directory and forget cached Settings."""
    target = Path(path).expanduser()
    if not target.is_dir():
        raise UsageError(f"--project-dir: {target.name} is not a directory")
    os.chdir(target)
    from mtc.core import config

    config._settings = None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run ``mtc``; return the exit code (it never raises ``SystemExit``)."""
    parser = build_parser()
    try:
        args, extras = parser.parse_known_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s"
        )
    try:
        if args.version:
            _say(f"{PROG} {_version()}")
            return EXIT_OK
        if args.command is None:
            parser.print_usage(sys.stderr)
            _complain(f"a command is needed: {', '.join(SUBCOMMANDS)}")
            return EXIT_USAGE
        if extras and args.command != "assess":
            parser.print_usage(sys.stderr)
            _complain(f"unrecognized arguments: {' '.join(extras)}")
            return EXIT_USAGE
        if args.project_dir:
            _enter_project(args.project_dir)
        return int(HANDLERS[args.command](args, list(extras)))
    except UsageError as exc:
        _complain(str(exc))
        return EXIT_USAGE
    except KeyboardInterrupt:
        _complain("interrupted")
        return EXIT_INTERRUPTED
    except SystemExit as exc:
        # A delegated argument parser exits on bad usage or --help.
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
