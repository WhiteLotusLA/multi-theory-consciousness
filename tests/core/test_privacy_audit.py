"""The privacy audit fails closed on planted material and stays quiet on clean trees."""

import string
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
ALLOWLIST = Path(__file__).resolve().parents[2] / "config" / "privacy" / "allowlist.txt"
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


def test_runtime_mode_allows_adjacent_floats_in_typed_persistence(tmp_path):
    """Typed persistence under STATE_DIR holds the modules' numeric scalars
    and histories as compact JSON, where two adjacent floats are values and
    not a coordinate pair: the tracked allowlist covers that one kind in
    those files and nothing else. (The CI install smoke audits the demo's
    runtime directory; its first run on the pushed v0.5.0 main reported eight
    such pairs.)"""
    rules = load_allowlist(ALLOWLIST)
    runtime = tmp_path / "runtime"
    # The one planted pair (allowlisted for this file, as the self-test's
    # other samples are) stands in a histories file and a scalars file.
    pair = "-0.014460000000000011,-0.035380000000000036"
    write(runtime, "demo/state/gwt_histories.json", f'{{"values":[{pair},0.5]}}\n')
    write(runtime, "demo/state/orchestrator_scalars.json", f'{{"pair":[{pair}]}}\n')
    assert run_runtime_audit([runtime], [PLANTED], rules) == []
    # The allowance is one kind in those files: a name there still counts ...
    write(
        runtime, "demo/state/gwt_histories.json", f'{{"by":"{PLANTED}","v":[{pair}]}}\n'
    )
    hits = run_runtime_audit([runtime], [PLANTED], rules)
    assert [(h.path, h.kind) for h in hits] == [
        ("runtime/demo/state/gwt_histories.json", "token")
    ]
    # ... and the same pair in a log or a result is still a coordinate hit.
    write(runtime, "demo/logs/mtc.log", f"INFO at {pair}\n")
    write(runtime, "demo/results/report.json", f'{{"where":[{pair}]}}\n')
    hits = run_runtime_audit([runtime], [], rules)
    assert sorted((h.path, h.kind) for h in hits) == [
        ("runtime/demo/logs/mtc.log", "coords"),
        ("runtime/demo/results/report.json", "coords"),
    ]


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


# Token boundaries (an alphabetic token after a non-letter or at a camelCase
# boundary is a hit; inside a longer word after a letter it is not).


def test_token_after_underscore_digit_or_camel_boundary_is_a_hit(tmp_path):
    lines = [
        "load_plantedname_state",
        "self._plantedname",
        "2plantedname",
        "myPlantedname",
        "myPlantednameBot",
    ]
    for index, line in enumerate(lines):
        f = write(tmp_path, f"t{index}.md", line + "\n")
        hits = run_audit([f], denylist=[PLANTED], allowlist=[])
        assert [(h.kind, h.match) for h in hits] == [("token", PLANTED)], line
        private = run_audit([f], denylist=[], allowlist=[], extra_denylist=[PLANTED])
        assert [(h.kind, h.match) for h in private] == [("token", "private#1")], line


def test_token_inside_a_longer_word_is_not_a_hit(tmp_path):
    for index, line in enumerate(["xsplantednamey", "XSPLANTEDNAME"]):
        f = write(tmp_path, f"w{index}.md", line + "\n")
        assert run_audit([f], denylist=[PLANTED], allowlist=[]) == [], line


# Literal-only expressions the assembled pass evaluates.


def _assembled(tmp_path, rel, text):
    f = write(tmp_path, rel, text)
    return [(h.kind, h.line, h.match) for h in run_audit([f], [PLANTED], [])]


def test_adjacent_literals_are_assembled(tmp_path):
    text = 'x = "Plan" "tedname"\ny = (\n    "Plan"\n    "tedname"\n)\n'
    assert _assembled(tmp_path, "m.py", text) == [
        ("assembled-token", 1, PLANTED),
        ("assembled-token", 3, PLANTED),
    ]


def test_escapes_inside_a_literal_are_decoded(tmp_path):
    bs = chr(92)
    text = f'x = "{bs}x50lantedname"\ny = "{bs}u0050lantedname"\n'
    assert _assembled(tmp_path, "m.py", text) == [
        ("assembled-token", 1, PLANTED),
        ("assembled-token", 2, PLANTED),
    ]


def test_fstring_constant_pieces_are_assembled(tmp_path):
    text = "x = f\"Plan{'ted'}name\"\ny = f\"{'Plan'}{'tedname'!s}\"\n"
    assert _assembled(tmp_path, "m.py", text) == [
        ("assembled-token", 1, PLANTED),
        ("assembled-token", 2, PLANTED),
    ]


def test_percent_and_format_with_literals_are_assembled(tmp_path):
    text = (
        'a = "Plan%s" % "tedname"\n'
        'b = "Plan{}".format("tedname")\n'
        'c = "%s%s" % ("Plan", "tedname")\n'
        'd = "{0}{1}".format("Plan", "tedname")\n'
    )
    assert _assembled(tmp_path, "m.py", text) == [
        ("assembled-token", 1, PLANTED),
        ("assembled-token", 2, PLANTED),
        ("assembled-token", 3, PLANTED),
        ("assembled-token", 4, PLANTED),
    ]


def test_reversed_literal_slice_is_assembled(tmp_path):
    # The second literal holds the token only inside a longer word, which the
    # line scan does not count; its slice is the token itself.
    text = 'x = "emandetnalP"[::-1]\ny = "xxPlantednamexx"[2:-2]\n'
    assert _assembled(tmp_path, "m.py", text) == [
        ("assembled-token", 1, PLANTED),
        ("assembled-token", 2, PLANTED),
    ]


def test_bytes_chain_is_assembled(tmp_path):
    text = 'x = b"Plan" + b"tedname"\n'
    assert _assembled(tmp_path, "m.py", text) == [("assembled-token", 1, PLANTED)]


def test_json_unicode_escape_is_decoded(tmp_path):
    bs = chr(92)
    text = '{"name": "' + bs + 'u0050lantedname"}\n'
    assert _assembled(tmp_path, "a.json", text) == [("assembled-token", 1, PLANTED)]


def test_multiline_plain_literal_is_reported_once(tmp_path):
    text = 'x = """first line\nsecond Plantedname line\n"""\n'
    assert _assembled(tmp_path, "m.py", text) == [("token", 2, PLANTED)]


def test_assembled_check_still_ignores_names_in_expressions(tmp_path):
    text = (
        'p = "Plan"\n'
        'a = f"{p}tedname"\n'
        'b = "%stedname" % p\n'
        'c = "{}tedname".format(p)\n'
        "d = p[::-1]\n"
    )
    assert _assembled(tmp_path, "m.py", text) == []


# Hidden forms, unknown suffixes, archives, every match on a line, kinds.


def test_zero_width_inside_a_token_is_a_hit_in_markdown(tmp_path):
    text = "Plan" + chr(0x200B) + "tedname here\n"
    assert _assembled(tmp_path, "a.md", text) == [("assembled-token", 1, PLANTED)]


def test_homoglyph_token_is_a_hit_in_yaml(tmp_path):
    # Cyrillic small a in place of the Latin a.
    text = "name: Pl" + chr(0x0430) + "ntedname\n"
    assert _assembled(tmp_path, "a.yaml", text) == [("assembled-token", 1, PLANTED)]
    # Greek capital rho in place of the Latin P.
    greek = chr(0x03A1) + "lantedname\n"
    assert _assembled(tmp_path, "b.yml", greek) == [("assembled-token", 1, PLANTED)]


def test_unknown_text_suffix_is_scanned(tmp_path):
    for rel in ("a.ipynb", "a.pyi", "a.rst", "a.svg", "a.xml"):
        assert _assembled(tmp_path, rel, f"see {PLANTED}\n") == [
            ("token", 1, PLANTED)
        ], rel


def _tar_gz(path: Path, members: dict) -> Path:
    import io
    import tarfile

    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_archive_members_are_scanned(tmp_path):
    archive = _tar_gz(
        tmp_path / "x.tar.gz",
        {
            "inner.txt": f"by {PLANTED}\n".encode("ascii"),
            "clean.txt": b"nothing here\n",
            "._inner.txt": b"\x00\x05\x16\x07metadata",
        },
    )
    hits = run_audit([archive], denylist=[PLANTED], allowlist=[], root=tmp_path)
    assert sorted((h.path, h.kind) for h in hits) == [
        ("x.tar.gz!._inner.txt", "artifact"),
        ("x.tar.gz!inner.txt", "token"),
    ]


def test_zip_and_nested_archive_members_are_scanned(tmp_path):
    import io
    import zipfile

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as nested:
        nested.writestr("deep.md", f"{PLANTED}\n")
    outer = tmp_path / "bundle.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("notes.md", f"{PLANTED}\n")
        archive.writestr("nested.zip", inner.getvalue())
    hits = run_audit([outer], denylist=[PLANTED], allowlist=[], root=tmp_path)
    assert sorted((h.path, h.kind) for h in hits) == [
        ("bundle.zip!nested.zip!deep.md", "token"),
        ("bundle.zip!notes.md", "token"),
    ]


def test_allow_rule_can_name_an_archive_member(tmp_path):
    archive = _tar_gz(
        tmp_path / "x.tar.gz", {"main.tex": f"{PLANTED}\n".encode("ascii")}
    )
    rule = AllowRule("x.tar.gz!main.tex", "token", f"^{PLANTED}$")
    assert run_audit([archive], [PLANTED], [rule], root=tmp_path) == []
    other = AllowRule("x.tar.gz!other.tex", "token", f"^{PLANTED}$")
    assert len(run_audit([archive], [PLANTED], [other], root=tmp_path)) == 1


def test_second_match_on_a_line_is_not_hidden_by_an_allowed_first(tmp_path):
    first = "/Users/somebody"
    second = first + "2"
    f = write(tmp_path, "a.md", f"from {first}/a to {second}/b\n")
    rule = AllowRule("a.md", "homepath", "^/Users/somebody$")
    hits = run_audit([f], denylist=[], allowlist=[rule], root=tmp_path)
    assert [(h.kind, h.match) for h in hits] == [("homepath", second)]


def test_allowlist_accepts_the_nonascii_and_assembled_kinds(tmp_path):
    a = write(
        tmp_path,
        "allow.txt",
        "m.py nonascii ^U\\+0430$\nm.md assembled-token ^Plantedname$\n",
    )
    assert load_allowlist(a) == [
        AllowRule(glob="m.py", kind="nonascii", pattern="^U\\+0430$"),
        AllowRule(glob="m.md", kind="assembled-token", pattern="^Plantedname$"),
    ]


def _wrapper():
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "privacy_audit.py"
    spec = importlib.util.spec_from_file_location("privacy_audit_tree", script)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    return wrapper


def test_tree_mode_counts_skipped_binaries(tmp_path, monkeypatch, capsys):
    wrapper = _wrapper()
    write(tmp_path, "docs/a.md", "nothing here\n")
    (tmp_path / "docs" / "blob.bin").write_bytes(bytes(range(256)) * 4)
    deny = write(tmp_path, "deny.txt", f"{PLANTED}\n")
    allow = write(tmp_path, "allow.txt", "# none\n")
    monkeypatch.delenv("PRIVACY_DENYLIST_EXTRA", raising=False)
    monkeypatch.setattr(wrapper, "ROOT", tmp_path)
    monkeypatch.setattr(wrapper, "DENYLIST", deny)
    monkeypatch.setattr(wrapper, "ALLOWLIST", allow)
    monkeypatch.setattr(wrapper, "LOCAL_EXTRA", tmp_path / "absent.txt")
    assert wrapper.main(["--paths", "docs"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "skipped 1 file(s) not read as text (.bin 1)" in out[-2]
    assert out[-1] == "privacy audit: clean"


# Fail-closed corners of the assembled pass.


def test_split_literal_with_a_lookalike_or_zero_width_escape_is_assembled(tmp_path):
    # The source is ASCII (the escapes are written as text), so no nonascii hit
    # can stand in for the assembled one.
    bs = chr(92)
    text = (
        f'x = "Pl" + "{bs}u0430ntedname"\n'
        f'y = "Plan" + "{bs}u200btedname"\n'
        f'z = "{bs}u03a1lan" + "tedname"\n'
    )
    assert _assembled(tmp_path, "m.py", text) == [
        ("assembled-token", 1, PLANTED),
        ("assembled-token", 2, PLANTED),
        ("assembled-token", 3, PLANTED),
    ]


def test_case_insensitive_match_beyond_ascii_is_not_prefiltered_away(tmp_path):
    # A dotless small i matches a Latin i case-insensitively; it neither
    # lower-cases nor decomposes to one, so a lower-case substring test alone
    # would drop the line before the matcher sees it.
    token = "Pinwheelix"
    dotless = chr(0x0131)
    disguised = "P" + dotless + "nwheelix"
    md = write(tmp_path, "a.md", f"by {disguised}\n")
    assert [(h.kind, h.match) for h in run_audit([md], [token], [])] == [
        ("token", token)
    ]
    py = write(tmp_path, "m.py", f'x = "P" + "{dotless}nwheelix"\n')
    kinds = sorted(h.kind for h in run_audit([py], [token], []))
    assert kinds == ["assembled-token", "nonascii"]


def test_oversize_format_fails_closed(tmp_path):
    text = (
        'a = "%s%s%100000s" % ("Plan", "tedname", "")\n'
        'b = "{}{}{:>{}}".format("Plan", "tedname", "", 200000)\n'
        "c = f\"{'Plan'}{'tedname'}{'':>100000}\"\n"
        'd = "%s%s%*s" % ("Plan", "tedname", 200000, "")\n'
    )
    hits = _assembled(tmp_path, "m.py", text)
    assert [(kind, line) for kind, line, _match in hits] == [
        ("encoding", 1),
        ("encoding", 2),
        ("encoding", 3),
        ("encoding", 4),
    ]
    assert all("format" in match for _kind, _line, match in hits)


def test_dynamic_format_width_is_refused_before_evaluation(tmp_path):
    import tracemalloc

    text = 'b = "{}{}{:>{}}".format("Plan", "tedname", "", 20000000)\n'
    f = write(tmp_path, "m.py", text)
    tracemalloc.start()
    try:
        hits = run_audit([f], [PLANTED], [])
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert [h.kind for h in hits] == ["encoding"]
    assert peak < 5 * 1024 * 1024, peak


def test_string_width_operand_fails_closed_before_allocation(tmp_path):
    import tracemalloc

    text = (
        'a = "{}{}{:>{}}".format("Plan", "tedname", "", "999999999999999999999")\n'
        'b = "{}{}{:>{w}}".format("Plan", "tedname", "", w="999999999999999999999")\n'
        'c = "{}{}{:>{}}".format("Plan", "tedname", "", "20000000")\n'
        'd = "%s%s%*s" % ("Plan", "tedname", "200000", "")\n'
        'e = "{}{}{:>{}}".format("Plan", "tedname", "", "abc")\n'
    )
    f = write(tmp_path, "m.py", text)
    tracemalloc.start()
    try:
        hits = run_audit([f], [PLANTED], [])
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert [(h.kind, h.line) for h in hits] == [
        ("encoding", 1),
        ("encoding", 2),
        ("encoding", 3),
        ("encoding", 4),
        ("encoding", 5),
    ]
    assert all("format" in h.match for h in hits)
    assert peak < 5 * 1024 * 1024, peak


def test_digits_and_stars_outside_a_conversion_specifier_are_not_refused(tmp_path):
    harmless = (
        'a = "{} on 20261017".format("run")\n'
        'b = "%d of 100000 rows" % 5\n'
        'c = "listening on port %d (default 18080)" % 9\n'
        'd = "** %d **" % 20000\n'
        'e = "{{{}}}".format(12345)\n'
        'f = "%*d" % (5, 20000)\n'
        'g = "{:{}}".format(20000, "d")\n'
    )
    assert _assembled(tmp_path, "m.py", harmless) == []
    hidden = (
        'a = "{}{} on 20261017".format("Plan", "tedname")\n'
        'b = "%s%s of 100000 rows" % ("Plan", "tedname")\n'
        'c = "** %s%s %d **" % ("Plan", "tedname", 20000)\n'
        'd = "{{{}{}{}}}".format("Plan", "tedname", 12345)\n'
        'e = "%*d%s%s" % (5, 20000, "Plan", "tedname")\n'
        'f = "{:{}}{}{}".format(20000, "d", "Plan", "tedname")\n'
    )
    hits = _assembled(tmp_path, "n.py", hidden)
    assert hits == [("assembled-token", n, PLANTED) for n in range(1, 7)]


def test_utf8_bytes_chain_with_a_zero_width_escape_is_assembled(tmp_path):
    bs = chr(92)
    text = f'z = b"Pl" + b"{bs}xe2{bs}x80{bs}x8bantedname"\n'
    assert _assembled(tmp_path, "m.py", text) == [("assembled-token", 1, PLANTED)]


def test_bytes_chain_with_a_latin1_lookalike_is_assembled(tmp_path):
    # The joined bytes are valid UTF-8 (an i with acute, which folds to i), but
    # their latin-1 reading holds an A with tilde and a soft hyphen, which fold
    # to A and to nothing: the token is visible in the latin-1 reading only.
    bs = chr(92)
    text = (
        f'x = b"Pl{bs}xc3" + b"{bs}xadntedname"\n'
        f'y = b"".join([b"Pl", b"{bs}xc3{bs}xad", b"ntedname"])\n'
    )
    assert _assembled(tmp_path, "m.py", text) == [
        ("assembled-token", 1, PLANTED),
        ("assembled-token", 2, PLANTED),
    ]


def _pre_314_vformat(
    self, format_string, args, kwargs, used_args, recursion_depth, auto_arg_index=0
):
    """string.Formatter._vformat as the interpreters before 3.14 run it.

    Only an empty field name is auto-numbered there, so an auto-numbered field
    with an index ('{[0]}') is looked up under the empty key and raises
    KeyError, although str.format accepts it.
    """
    if recursion_depth < 0:
        raise ValueError("Max string recursion exceeded")
    result = []
    for literal_text, field_name, format_spec, conversion in self.parse(format_string):
        if literal_text:
            result.append(literal_text)
        if field_name is None:
            continue
        if field_name == "":
            if auto_arg_index is False:
                raise ValueError("cannot switch to automatic field numbering")
            field_name = str(auto_arg_index)
            auto_arg_index += 1
        elif field_name.isdigit():
            if auto_arg_index:
                raise ValueError("cannot switch to manual field specification")
            auto_arg_index = False
        obj, arg_used = self.get_field(field_name, args, kwargs)
        used_args.add(arg_used)
        obj = self.convert_field(obj, conversion)
        format_spec, auto_arg_index = self._vformat(
            format_spec,
            args,
            kwargs,
            used_args,
            recursion_depth - 1,
            auto_arg_index=auto_arg_index,
        )
        result.append(self.format_field(obj, format_spec))
    return "".join(result), auto_arg_index


@pytest.fixture(params=["native", "pre-3.14"])
def formatter_numbering(request, monkeypatch):
    """The interpreter's own string.Formatter, then the pre-3.14 numbering."""
    if request.param == "pre-3.14":
        monkeypatch.setattr(string.Formatter, "_vformat", _pre_314_vformat)
    return request.param


def test_auto_numbered_indexed_fields_are_assembled_on_every_interpreter(
    tmp_path, formatter_numbering
):
    hidden = (
        'a = "{[0]}{[0]}{[0]}{[0]}{}".format("P", "l", "a", "n", "tedname")\n'
        'b = "{}{[1]}{}".format("Pl", "xa", "ntedname")\n'
        'c = "{[0]:>{}}{[0]}{[0]}{[0]}{}".format("P", 1, "l", "a", "n", "tedname")\n'
    )
    assert _assembled(tmp_path, "m.py", hidden) == [
        ("assembled-token", 1, PLANTED),
        ("assembled-token", 2, PLANTED),
        ("assembled-token", 3, PLANTED),
    ]
    # Working code with an auto-numbered indexed field and a nested width is
    # neither refused nor reported.
    assert _assembled(tmp_path, "n.py", 'x = "{[0]:>{}}".format("ab", 5)\n') == []
    # str.format refuses to mix automatic and manual numbering, so such an
    # expression assembles to nothing.
    mixed = 'y = "{}{1}{}".format("Plan", "tedname", "")\n'
    assert _assembled(tmp_path, "o.py", mixed) == []
