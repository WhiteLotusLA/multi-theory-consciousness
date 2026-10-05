"""The privacy audit fails closed on planted material and stays quiet on clean trees."""

from pathlib import Path

from mtc.devtools.privacy_audit import (
    AllowRule,
    load_allowlist,
    load_denylist,
    run_audit,
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
    assert kinds == ["encoding", "token"]


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
