"""Patch mode of the privacy audit on a temporary git repository: split
literals are assembled in each commit's post-image, on the lines that commit
added, and a diff header the audit cannot parse fails closed."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "privacy_audit.py"

# A synthetic token: no real name appears in this file, whole or in pieces.
PLANTED = "Plantedname"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not installed"
)


def _wrapper():
    spec = importlib.util.spec_from_file_location("privacy_audit_patch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """An empty repository with a local identity and no user or system config."""
    empty = tmp_path / "gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Testagent")
    _git(path, "config", "user.email", "testagent.local")
    _git(path, "config", "commit.gpgsign", "false")
    return path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _commit(repo: Path, files: dict, message: str) -> str:
    for rel, text in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        _git(repo, "add", "--", rel)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _hits(repo: Path, ref: str):
    wrapper = _wrapper()
    return wrapper.patch_hits(ref, root=repo, denylist=[PLANTED], allowlist=[])


FUNCTION = "def f():\n    a = 1\n    return a\n"
WITH_SPLIT = 'def f():\n    a = 1\n    x = "Plan" + "tedname"\n    return a\n'


def test_split_literal_added_to_an_existing_file_is_reported(repo):
    first = _commit(repo, {"m.py": FUNCTION}, "add a function")
    second = _commit(repo, {"m.py": WITH_SPLIT}, "add a line inside it")
    _commit(repo, {"m.py": FUNCTION}, "remove the line")
    hits = _hits(repo, first)
    assert [(h.path, h.line, h.kind, h.match) for h in hits] == [
        ("m.py", 3, "assembled-token", PLANTED)
    ]
    assert hits[0].commit == second[:12]
    assert str(hits[0]).endswith(f" @{second[:12]}")


def test_file_touched_by_two_commits_is_still_assembled(repo):
    base = _commit(repo, {"README.md": "start\n"}, "start")
    whole = 'x = "Plan" + "tedname"\n\n\ndef g():\n    return 1\n'
    added = _commit(repo, {"n.py": whole}, "add a file with a split literal")
    edited = whole.replace("return 1", "return 2")
    _commit(repo, {"n.py": edited}, "edit another line of it")
    hits = _hits(repo, base)
    assert [(h.path, h.line, h.kind, h.match, h.commit) for h in hits] == [
        ("n.py", 1, "assembled-token", PLANTED, added[:12])
    ]


def test_quoted_header_fails_closed(repo):
    base = _commit(repo, {"README.md": "start\n"}, "start")
    _commit(repo, {'we"ird.md': "plain text\n"}, "a path git quotes")
    hits = _hits(repo, base)
    assert [h.kind for h in hits] == ["encoding"]
    assert "we" in hits[0].path


def test_patch_mode_is_clean_on_a_clean_range(repo):
    base = _commit(repo, {"README.md": "start\n"}, "start")
    _commit(repo, {"m.py": FUNCTION}, "a clean function")
    _commit(repo, {"m.py": FUNCTION + "\n\nb = 'two'\n"}, "a clean edit")
    assert _hits(repo, base) == []


def test_added_text_lines_are_still_scanned(repo):
    base = _commit(repo, {"README.md": "start\n"}, "start")
    _commit(repo, {"notes.md": f"line one\nwritten by {PLANTED}\n"}, "notes")
    hits = _hits(repo, base)
    assert [(h.path, h.line, h.kind, h.match) for h in hits] == [
        ("notes.md", 2, "token", PLANTED)
    ]


def test_git_helpers_take_a_root(repo):
    wrapper = _wrapper()
    _commit(repo, {"a.md": "x\n"}, "one")
    assert wrapper.git_paths("ls-files", "-z", root=repo) == ["a.md"]


SPLIT = 'x = "Plan" + "tedname"\n'


def test_python_file_with_a_space_in_its_path_is_assembled(repo):
    # git ends a +++ header whose path holds a space with a TAB.
    base = _commit(repo, {"README.md": "start\n"}, "start")
    one = _commit(repo, {"my file.py": SPLIT}, "a space in the name")
    two = _commit(repo, {"a dir/m.py": SPLIT}, "a space in the directory")
    hits = _hits(repo, base)
    assert sorted((h.path, h.line, h.kind, h.match, h.commit) for h in hits) == [
        ("a dir/m.py", 1, "assembled-token", PLANTED, two[:12]),
        ("my file.py", 1, "assembled-token", PLANTED, one[:12]),
    ]


def test_text_file_with_a_space_in_its_path_keeps_its_name(repo):
    base = _commit(repo, {"README.md": "start\n"}, "start")
    _commit(repo, {"my notes.md": f"by {PLANTED}\n"}, "a space in the name")
    hits = _hits(repo, base)
    assert [(h.path, h.line, h.kind, h.match) for h in hits] == [
        ("my notes.md", 1, "token", PLANTED)
    ]


def test_post_image_of_a_missing_path_with_a_space_is_skipped(repo):
    wrapper = _wrapper()
    head = _commit(repo, {"m.py": SPLIT}, "one file")
    images = wrapper.post_images([(head, "my file.py"), (head, "m.py")], root=repo)
    assert images == {(head, "m.py"): SPLIT.encode("utf-8")}


def test_lines_a_merge_resolution_adds_are_audited(repo):
    base = _commit(repo, {"m.py": "a = 1\n", "notes.md": "one\n"}, "start")
    _git(repo, "checkout", "-q", "-b", "side")
    _commit(repo, {"m.py": "a = 2\n", "notes.md": "two\n"}, "side change")
    _git(repo, "checkout", "-q", "-")
    _commit(repo, {"m.py": "a = 3\n", "notes.md": "three\n"}, "main change")
    subprocess.run(["git", "merge", "-q", "side"], cwd=repo, capture_output=True)
    (repo / "m.py").write_text("a = 4\n" + SPLIT, encoding="utf-8")
    (repo / "notes.md").write_text(f"four by {PLANTED}\n", encoding="utf-8")
    _git(repo, "add", "m.py", "notes.md")
    _git(repo, "commit", "-q", "--no-edit")
    merge = _git(repo, "rev-parse", "HEAD")
    hits = _hits(repo, base)
    assert sorted((h.path, h.kind, h.match, h.commit) for h in hits) == [
        ("m.py", "assembled-token", PLANTED, merge[:12]),
        ("notes.md", "token", PLANTED, None),
    ]
