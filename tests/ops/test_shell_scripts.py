"""Rules every operations shell script follows, checked on the source."""

import re
import shutil
import stat
import subprocess

import pytest

from mtc.devtools.privacy_audit import PATTERNS

from .conftest import OPS_DIR, SHELL_SCRIPTS

FORCED_KILL = (
    re.compile(r"\bkill\s+(-s\s+)?-?\s*9\b"),
    re.compile(r"\bkill\s+-s\s*(SIG)?KILL\b", re.IGNORECASE),
    re.compile(r"\bp?kill(all)?\s+[^\n]*-(SIG)?KILL\b", re.IGNORECASE),
    re.compile(r"\bpkill\s+[^\n]*-9\b"),
    re.compile(r"\bkillall\s+[^\n]*-9\b"),
    re.compile(r"SIGKILL"),
    re.compile(r"signal\.SIGKILL|\.kill\(\)"),
)
# Home directories and external volumes: the privacy audit's own pattern,
# plus the shell spellings of a home directory and the usual mount points.
HOME_OR_VOLUME = (
    PATTERNS["homepath"],
    re.compile(r"(^|[\s\"'=(])~/"),
    re.compile(r"\$\{?HOME\b"),
    re.compile(r"/mnt/"),
    re.compile(r"/media/"),
)
NON_LOOPBACK_IPV4 = re.compile(
    r"(?<![\d.])(?!127\.)(?!0\.0\.0\.0)(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"
)
# Defaults that belong to Settings and must never be restated in a script.
DEFAULT_PORTS = re.compile(
    r"(?<![\w.])(5432|27017|6379|6333|6334|7687|7474|8200|8080)\b"
)
BASH4_ONLY = (
    (re.compile(r"\bdeclare\s+-[a-zA-Z]*A"), "an associative array"),
    (re.compile(r"\blocal\s+-[a-zA-Z]*A"), "an associative array"),
    (re.compile(r"\b(mapfile|readarray)\b"), "mapfile or readarray"),
    (
        re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?(,,|\^\^|,|\^)[^}]*\}"),
        "case modification",
    ),
    (
        re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*@[A-Za-z]\}"),
        "a parameter transformation",
    ),
    (re.compile(r"&>>"), "the &>> redirection"),
    (re.compile(r"\|&"), "the |& pipe"),
    (re.compile(r"\bcoproc\b"), "a coprocess"),
    (re.compile(r";;&|;&"), "a case fall-through terminator"),
    (re.compile(r"\[\[\s+-v\s"), "the -v test"),
    (re.compile(r"%\([^)]*\)T"), "the printf time format"),
    (re.compile(r"\bwait\s+-n\b"), "wait -n"),
    (re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\[-\d+\]\}"), "a negative array index"),
)


def _source(name: str) -> str:
    return (OPS_DIR / name).read_text(encoding="utf-8")


def _code_lines(name: str):
    """Lines of a script outside its comments (usage text included)."""
    for number, line in enumerate(_source(name).splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        yield number, line


def test_the_expected_scripts_exist():
    present = sorted(path.name for path in OPS_DIR.glob("*.sh"))
    assert present == sorted(SHELL_SCRIPTS)
    assert (OPS_DIR / "backup_watchdog.py").is_file()


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_bash_syntax_all_scripts(name):
    result = subprocess.run(
        ["/bin/bash", "-n", str(OPS_DIR / name)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_shellcheck_clean():
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck is not installed; bash -n still checks the syntax")
    result = subprocess.run(
        [shellcheck, "--shell=bash", *(str(OPS_DIR / name) for name in SHELL_SCRIPTS)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_strict_mode_header(name):
    lines = _source(name).splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    code = [line for line in lines[1:] if line.strip() and not line.startswith("#")]
    # Strict mode is the first thing the script does.
    assert code[0] == "set -euo pipefail"
    mode = stat.S_IMODE((OPS_DIR / name).stat().st_mode)
    assert mode & 0o111, f"{name} is not executable"
    # Every script says what it is for before any code.
    assert lines[1].startswith("# ")


def test_no_kill_9_in_ops_scripts():
    scanned = 0
    for path in sorted(OPS_DIR.iterdir()):
        if path.suffix not in (".sh", ".py"):
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8")
        for pattern in FORCED_KILL:
            match = pattern.search(text)
            assert match is None, f"{path.name}: forced kill ({match.group(0)!r})"
    assert scanned >= len(SHELL_SCRIPTS) + 1
    # The stop script does signal, with the terminate signal only.
    stop = _source("stop_stack.sh")
    assert "kill -TERM" in stop
    signals = set(re.findall(r"\bkill\s+-([A-Z0-9]+)", stop))
    # Signal 0 only asks whether the process exists; it delivers nothing.
    assert signals == {"0", "TERM"}
    # The detector is not vacuous.
    for sample in ("kill -9 12", "kill -s KILL 12", "pkill -9 name", "kill -KILL 1"):
        assert any(pattern.search(sample) for pattern in FORCED_KILL), sample


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_no_home_or_volume_paths(name):
    text = _source(name)
    for pattern in HOME_OR_VOLUME:
        match = pattern.search(text)
        assert match is None, f"{name}: {match.group(0)!r}"
    # Loopback addresses only.
    assert NON_LOOPBACK_IPV4.findall(text) == []
    assert "localhost" not in text


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_settings_come_from_env_export(name):
    text = _source(name)
    assert "python3 -m mtc.core.env_export" in text
    assert "virtual environment" in text
    for number, line in _code_lines(name):
        match = DEFAULT_PORTS.search(line)
        assert match is None, f"{name}:{number} restates a default ({match.group(0)})"
    # Settings are read through the helper, never from the environment file.
    assert not re.search(r"(^|\s)(source|\.)\s+\S*\.env\b", text, re.MULTILINE)
    assert "grep" not in text or not re.search(r"grep[^\n]*\.env\b", text)


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_every_script_has_help_and_dry_run(name):
    helped = subprocess.run(
        ["/bin/bash", str(OPS_DIR / name), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin"},
    )
    # --help needs no settings, no virtual environment and no tool.
    assert helped.returncode == 0, helped.stderr
    assert helped.stdout.startswith("Usage: scripts/ops/" + name)
    assert "--dry-run" in helped.stdout
    assert "Exit codes:" in helped.stdout
    unknown = subprocess.run(
        ["/bin/bash", str(OPS_DIR / name), "--no-such-option"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert unknown.returncode == 2
    assert "unknown option" in unknown.stderr
    assert re.search(r"--dry-run\)", _source(name)), "the option must be parsed"


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_no_bash4_only_constructs(name):
    for number, line in _code_lines(name):
        for pattern, what in BASH4_ONLY:
            assert pattern.search(line) is None, f"{name}:{number} uses {what}"


def test_bash4_detector_is_not_vacuous():
    samples = (
        "declare -A table",
        "mapfile -t rows < file",
        'echo "${name,,}"',
        'echo "${name^^}"',
        "command &>> log",
        "a |& b",
        "[[ -v name ]]",
    )
    for sample in samples:
        assert any(pattern.search(sample) for pattern, _ in BASH4_ONLY), sample
    # Plain expansions the scripts do use are not flagged.
    for fine in ('"${path#./}"', '"${uri%%:*}"', '"${2:-}"', '"${list[@]}"'):
        assert not any(pattern.search(fine) for pattern, _ in BASH4_ONLY), fine


def test_no_unguarded_recursive_removal():
    """A recursive removal is always preceded by a containment assertion."""
    for name in SHELL_SCRIPTS:
        lines = _source(name).splitlines()
        for index, line in enumerate(lines):
            if line.lstrip().startswith("#"):
                continue
            if not re.search(
                r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\b|\brm\s+-[a-zA-Z]*f[a-zA-Z]*r", line
            ):
                continue
            window = "\n".join(lines[max(0, index - 12) : index])
            assert "assert_inside" in window, (
                f"{name}:{index + 1} removes recursively without a containment "
                "assertion just before it"
            )
