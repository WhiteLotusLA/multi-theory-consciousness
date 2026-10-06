"""The privacy audit fails closed on planted material and stays quiet on clean trees."""

from pathlib import Path

import pytest

from mtc.devtools.privacy_audit import (
    AllowRule,
    load_allowlist,
    load_denylist,
    run_audit,
    run_runtime_audit,
    runtime_files,
)

# Synthetic tokens: no real name appears in this file, whole or in pieces.
PLANTED = "Plantedname"
PET = "Plantedpet"
EM_DASH = "\u2014"
EN_DASH = "\u2013"


def write(tmp: Path, rel: str, text: str, encoding: str = "utf-8") -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding=encoding)
    return p


def test_denylist_token_is_reported(tmp_path):
    f = write(tmp_path, "mtc/x.py", f"# built for {PLANTED}\n")
    hits = run_audit([f], denylist=[PLANTED], allowlist=[])
    assert [(h.kind, h.line) for h in hits] == [("token", 1)]


def test_token_match_is_case_insensitive_and_word_bounded(tmp_path):
    f = write(tmp_path, "a.md", f"{PLANTED.upper()} here\nxs{PLANTED}y not a word\n")
    hits = run_audit([f], denylist=[PLANTED], allowlist=[])
    assert [h.line for h in hits] == [1]


def test_private_extra_list_is_applied(tmp_path):
    f = write(tmp_path, "a.md", f"the dog {PET}\n")
    assert run_audit([f], denylist=[], allowlist=[]) == []
    hits = run_audit([f], denylist=[], allowlist=[], extra_denylist=[PET])
    assert hits and hits[0].kind == "token"


def test_patterns_are_reported(tmp_path):
    f = write(
        tmp_path,
        "notes.md",
        "mail someone@example.com\n"
        "call 555-123-4567\n"
        "at 34.0937, -118.3295\n"
        "path /Users/somebody/x\n"
        "host 10.0.0.12\n"
        "key sk-abcdefghijklmnopqrstuvwxyz0123\n"
        "anthropic sk-" + "ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz012345\n"
        f"a dash {EM_DASH} here and {EN_DASH} there\n",
    )
    kinds = sorted({h.kind for h in run_audit([f], denylist=[], allowlist=[])})
    assert kinds == ["coords", "dash", "email", "homepath", "ipv4", "phone", "secret"]


def test_loopback_and_zero_addresses_are_not_hits(tmp_path):
    f = write(tmp_path, "a.md", "127.0.0.1 and 0.0.0.0 and 1.2.3\n")
    hits = run_audit([f], denylist=[], allowlist=[])
    assert [h for h in hits if h.kind == "ipv4"] == []


def test_emoji_only_reported_in_python_files(tmp_path):
    py = write(tmp_path, "m.py", "x = 1  # \U0001f600\n")
    md = write(tmp_path, "m.md", "\U0001f600\n")
    assert [h.kind for h in run_audit([py], denylist=[], allowlist=[])] == ["emoji"]
    assert run_audit([md], denylist=[], allowlist=[]) == []


def test_artifact_paths_are_reported(tmp_path):
    f = write(tmp_path, "pkg/__pycache__/m.cpython-311.pyc", "")
    hits = run_audit([f], denylist=[], allowlist=[])
    assert hits and hits[0].kind == "artifact"


def test_audit_skips_binary_files(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(bytes(range(256)) * 4)
    assert run_audit([p], denylist=[PLANTED], allowlist=[]) == []


def test_anthropic_format_key_is_reported(tmp_path):
    key = "sk-" + "ant-api03-" + "AbCdEfGhIjKlMnOpQrSt-UvWxYz_0123456789"
    f = write(tmp_path, "settings.md", f"key {key}\n")
    hits = [h for h in run_audit([f], denylist=[], allowlist=[]) if h.kind == "secret"]
    assert [h.match for h in hits] == [key]


def test_non_utf8_text_file_is_flagged_and_still_scanned(tmp_path):
    p = tmp_path / "latin1.py"
    p.write_bytes(f"# caf\xe9 for {PLANTED}\n".encode("latin-1"))
    kinds = sorted(h.kind for h in run_audit([p], denylist=[PLANTED], allowlist=[]))
    assert kinds == ["encoding", "nonascii", "token"]


def test_utf16_text_file_is_flagged_and_still_scanned(tmp_path):
    p = write(tmp_path, "utf16.md", f"note for {PLANTED}\n", encoding="utf-16")
    kinds = sorted(h.kind for h in run_audit([p], denylist=[PLANTED], allowlist=[]))
    assert kinds == ["encoding", "token"]


def test_nul_byte_does_not_hide_a_text_file(tmp_path):
    p = tmp_path / "nul.py"
    p.write_bytes(b"x = 1\x00\n# " + PLANTED.encode("ascii") + b"\n")
    kinds = sorted(h.kind for h in run_audit([p], denylist=[PLANTED], allowlist=[]))
    assert kinds == ["encoding", "token"]


def test_allowlist_glob_matches_nested_paths(tmp_path):
    f = write(tmp_path, "paper/sections/intro.md", f"{PLANTED} appears in the paper\n")
    rule = AllowRule(glob="paper/**", kind="token", pattern=PLANTED)
    assert run_audit([f], denylist=[PLANTED], allowlist=[rule], root=tmp_path) == []
    assert run_audit([f], denylist=[PLANTED], allowlist=[], root=tmp_path)


def test_list_loaders(tmp_path):
    d = write(tmp_path, "deny.txt", f"# comment\n{PLANTED}\n\n/Users/\n")
    a = write(
        tmp_path, "allow.txt", "# glob kind regex\npyproject.toml token Maintainer\n"
    )
    assert load_denylist(d) == [PLANTED, "/Users/"]
    rules = load_allowlist(a)
    assert rules == [
        AllowRule(glob="pyproject.toml", kind="token", pattern="Maintainer")
    ]


def test_directory_paths_are_expanded(tmp_path):
    write(tmp_path, "docs/a/b.md", f"{PLANTED} deep inside\n")
    write(tmp_path, "docs/clean.md", "nothing here\n")
    hits = run_audit(
        [tmp_path / "docs"], denylist=[PLANTED], allowlist=[], root=tmp_path
    )
    assert [h.path for h in hits] == ["docs/a/b.md"]


def test_suffixed_token_forms_are_hits(tmp_path):
    f = write(
        tmp_path, "a.md", f"{PLANTED}s here\n{PLANTED}2 here\n{PLANTED}Bot here\n"
    )
    hits = run_audit([f], denylist=[PLANTED], allowlist=[])
    assert len(hits) == 3


def test_assembled_literal_concatenation_is_reported(tmp_path):
    f = write(tmp_path, "m.py", 'x = "Plan" + "tedname"\n')
    hits = run_audit([f], denylist=[PLANTED], allowlist=[])
    assert [(h.kind, h.line, h.match) for h in hits] == [
        ("assembled-token", 1, PLANTED)
    ]


def test_assembled_join_and_chr_are_reported(tmp_path):
    f = write(
        tmp_path,
        "m.py",
        'a = "".join(["Plan", "ted", "name"])\n'
        'b = chr(80) + "lantedname"\n'
        'c = "-".join(("x", "y"))\n',
    )
    hits = run_audit([f], denylist=[PLANTED], allowlist=[])
    assert [(h.kind, h.line) for h in hits] == [
        ("assembled-token", 1),
        ("assembled-token", 2),
    ]


def test_assembled_check_ignores_variables_and_non_python(tmp_path):
    py = write(tmp_path, "m.py", 'p = "Plan"\nx = p + "tedname"\n')
    md = write(tmp_path, "m.md", '"Plan" + "tedname"\n')
    assert run_audit([py], denylist=[PLANTED], allowlist=[]) == []
    assert run_audit([md], denylist=[PLANTED], allowlist=[]) == []


def test_assembled_check_tolerates_unparseable_python(tmp_path):
    f = write(tmp_path, "broken.py", 'def (:\n    "Plan" + "tedname"\n')
    assert run_audit([f], denylist=[PLANTED], allowlist=[]) == []


def test_whole_literal_is_not_double_reported(tmp_path):
    f = write(tmp_path, "m.py", 'x = "Plantedname" + "!"\n')
    hits = run_audit([f], denylist=[PLANTED], allowlist=[])
    assert [h.kind for h in hits] == ["token"]


def test_non_ascii_in_python_is_reported_by_code_point(tmp_path):
    f = write(tmp_path, "m.py", 'x = "caf\u00e9"\n')
    hits = run_audit([f], denylist=[], allowlist=[])
    assert [(h.kind, h.line, h.match) for h in hits] == [("nonascii", 1, "U+00E9")]


def test_homoglyph_in_identifier_is_reported(tmp_path):
    # Cyrillic small a inside an otherwise Latin name
    f = write(tmp_path, "m.py", "n\u0430me = 1\n")
    hits = run_audit([f], denylist=[], allowlist=[])
    assert [(h.kind, h.match) for h in hits] == [("nonascii", "U+0430")]


def test_zero_width_character_is_reported(tmp_path):
    f = write(tmp_path, "m.py", 'x = "a\u200bb"\n')
    hits = run_audit([f], denylist=[], allowlist=[])
    assert [(h.kind, h.match) for h in hits] == [("nonascii", "U+200B")]


def test_non_ascii_outside_python_is_not_a_hit(tmp_path):
    f = write(tmp_path, "m.md", "caf\u00e9 and n\u0430me\n")
    assert run_audit([f], denylist=[], allowlist=[]) == []


def test_dash_and_emoji_keep_their_own_kinds_in_python(tmp_path):
    f = write(tmp_path, "m.py", f"a = 1  # {EM_DASH}\nb = 2  # \U0001f600\n")
    kinds = [(h.kind, h.line) for h in run_audit([f], denylist=[], allowlist=[])]
    assert kinds == [("dash", 1), ("emoji", 2)]


def _runtime_tree(tmp_path: Path) -> Path:
    """A log directory outside any repository, with clean and planted files."""
    logs = tmp_path / "elsewhere" / "logs"
    write(logs, "mtc.log", "INFO runtime ready\n")
    write(logs, "specialty/safety.jsonl", '{"event": "ok"}\n')
    return logs


def test_runtime_mode_scans_directories_outside_the_repo(tmp_path):
    logs = _runtime_tree(tmp_path)
    results = tmp_path / "other" / "results"
    write(results, "run/report.json", '{"overall": 0.5}\n')
    assert run_runtime_audit([logs, results], [PLANTED], []) == []
    assert sorted(p.name for p in runtime_files([logs, results])) == [
        "mtc.log",
        "report.json",
        "safety.jsonl",
    ]
    # A planted token in each directory is found, in a log, a rotated log and
    # a JSON result alike.
    write(logs, "mtc.log", f"INFO talking with {PLANTED}\n")
    write(logs, "mtc.log.1", f"INFO earlier, {PLANTED}\n")
    write(results, "run/report.json", f'{{"speaker": "{PLANTED}"}}\n')
    hits = run_runtime_audit([logs, results], [PLANTED], [])
    assert sorted((h.path, h.kind) for h in hits) == [
        ("logs/mtc.log", "token"),
        ("logs/mtc.log.1", "token"),
        ("results/run/report.json", "token"),
    ]
    # The private extra list applies, reported by position and never by value.
    write(logs, "mtc.log", f"INFO the dog {PET}\n")
    private = run_runtime_audit([logs], [], [], extra_denylist=[PET])
    assert [(h.kind, h.match) for h in private] == [("token", "private#1")]
    # A directory that cannot be scanned is an error, never a clean result.
    with pytest.raises(NotADirectoryError):
        run_runtime_audit([tmp_path / "missing"], [PLANTED], [])
    with pytest.raises(NotADirectoryError):
        run_runtime_audit([logs / "mtc.log"], [PLANTED], [])


def test_runtime_mode_disables_only_the_artifact_rule(tmp_path):
    logs = _runtime_tree(tmp_path)
    record = logs / "specialty" / "safety.jsonl"
    # The tracked-tree audit flags the file as an artifact and never reads it.
    tracked = run_audit([record], denylist=[PLANTED], allowlist=[], root=tmp_path)
    assert [h.kind for h in tracked] == ["artifact"]
    # Runtime mode scans it and, clean, reports nothing.
    assert run_runtime_audit([logs], [PLANTED], []) == []
    # Planted material in it is still a hit: a token, an address and a dash.
    address = "@".join(["someone", "example.com"])
    record.write_text(
        f'{{"speaker": "{PLANTED}"}}\n'
        f'{{"contact": "{address}"}}\n'
        f'{{"text": "a dash {EM_DASH} here"}}\n',
        encoding="utf-8",
    )
    hits = run_runtime_audit([logs], [PLANTED], [])
    assert [(h.path, h.line, h.kind) for h in hits] == [
        ("logs/specialty/safety.jsonl", 1, "token"),
        ("logs/specialty/safety.jsonl", 2, "email"),
        ("logs/specialty/safety.jsonl", 3, "dash"),
    ]
    assert all(h.kind != "artifact" for h in hits)
    # Every other pattern fires too, and so does the encoding check.
    write(
        logs,
        "mtc.log",
        "call 555-123-4567\n"
        "path /Users/somebody/x\n"
        "host 10.0.0.12\n"
        "key sk-abcdefghijklmnopqrstuvwxyz0123\n",
    )
    (logs / "broken.log").write_bytes(b"caf\xe9 " + PLANTED.encode("ascii") + b"\n")
    kinds = sorted({h.kind for h in run_runtime_audit([logs], [PLANTED], [])})
    assert kinds == [
        "dash",
        "email",
        "encoding",
        "homepath",
        "ipv4",
        "phone",
        "secret",
        "token",
    ]
    # Python in a runtime directory keeps its stricter checks.
    write(logs, "probe.py", 'x = "Planted" + "name"\n')
    assembled = [
        h for h in run_runtime_audit([logs], [PLANTED], []) if h.path == "logs/probe.py"
    ]
    assert [h.kind for h in assembled] == ["assembled-token"]
    # A file kind neither mode reads as text stays unread.
    (logs / "weights.bin").write_bytes(PLANTED.encode("ascii"))
    assert all(
        h.path != "logs/weights.bin" for h in run_runtime_audit([logs], [PLANTED], [])
    )
    assert logs / "weights.bin" not in runtime_files([logs])


def test_runtime_mode_reports_dir_relative_paths(tmp_path):
    logs = _runtime_tree(tmp_path)
    write(logs, "specialty/safety.jsonl", f'{{"speaker": "{PLANTED}"}}\n')
    hits = run_runtime_audit([logs], [PLANTED], [])
    assert [str(h) for h in hits] == [f"logs/specialty/safety.jsonl:1:token:{PLANTED}"]
    # Nothing of the directory's location is printed: only its own name.
    assert str(tmp_path) not in str(hits[0])
    assert "elsewhere" not in hits[0].path
    # Allowlist rules match the reported path.
    covering = AllowRule("logs/specialty/**", "token", PLANTED)
    assert run_runtime_audit([logs], [PLANTED], [covering]) == []
    elsewhere = AllowRule("results/**", "token", PLANTED)
    assert len(run_runtime_audit([logs], [PLANTED], [elsewhere])) == 1


def test_runtime_mode_counts_what_it_skips(tmp_path, capsys):
    """Runtime mode reads only text kinds; every other file is skipped, and
    the count of skipped files is reported next to the scanned count, so a
    directory of binaries never reads as audited."""
    import importlib.util

    from mtc.devtools.privacy_audit import runtime_skipped

    logs = _runtime_tree(tmp_path)
    (logs / "state.npz").write_bytes(PLANTED.encode("ascii"))
    (logs / "cache.pkl").write_bytes(PLANTED.encode("ascii"))
    (logs / "deep").mkdir()
    (logs / "deep" / "store.db").write_bytes(b"\x00\x01")
    skipped = runtime_skipped([logs])
    assert sorted(p.name for p in skipped) == ["cache.pkl", "state.npz", "store.db"]
    assert not set(skipped) & set(runtime_files([logs]))

    script = Path(__file__).resolve().parents[2] / "scripts" / "privacy_audit.py"
    spec = importlib.util.spec_from_file_location("privacy_audit_wrapper", script)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    assert wrapper.main(["--runtime", str(logs)]) == 0
    out = capsys.readouterr().out
    assert "runtime mode scanned 2 file(s)" in out
    assert "skipped 3 file(s) not read as text (.db 1, .npz 1, .pkl 1)" in out
