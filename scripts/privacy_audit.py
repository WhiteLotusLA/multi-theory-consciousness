#!/usr/bin/env python3
"""Run the privacy audit over tracked files, staged files, or a commit range.

Usage:
  python scripts/privacy_audit.py                 # all tracked files
  python scripts/privacy_audit.py --staged        # staged content (the index)
  python scripts/privacy_audit.py --patch-from origin/main   # lines added since ref
  python scripts/privacy_audit.py --paths a.py b.md
  python scripts/privacy_audit.py --runtime logs results   # runtime directories

Exit code 1 on any hit. The private extra denylist is read from
$PRIVACY_DENYLIST_EXTRA or config/privacy/denylist.local.txt when present.

Tree mode (no option, or --paths) prints how many files it scanned and how
many it skipped as neither text nor an archive, by suffix, before the verdict.

--staged audits the blobs in the index, not the working tree, so a file that
was fixed on disk but not re-staged still blocks the commit. --patch-from
audits every line added by the commits in REF..HEAD, file by file, so path
allowlist rules apply; token and pattern hits then count added lines within
that file. Split literals in Python files are assembled in each commit's own
version of the file, on the lines that commit added: such a hit carries the
file's line number and the commit (``@<short hash>``). A diff header the audit
cannot parse (a path git quotes) is an encoding hit, never silence. A merge
commit counts the lines its resolution adds (its diff against the automatic
re-merge of its parents), so an edit made while merging is audited too.

--runtime audits directories that need not be inside the repository (a log or
a results directory after a run). Only the artifact rule is off there: .jsonl
and .log files are scanned as text instead of being flagged, and every token,
pattern and encoding check still applies. Paths print as
<directory name>/<relative path>, a path that is not a directory is an
error, never a clean result, and the files it does not read as text are
counted as skipped next to the scanned count.
"""

import argparse
import dataclasses
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mtc" / "devtools" / "privacy_audit.py"

# Short commit hashes in patch-mode hits.
SHORT_HASH = 12


def _load_audit_module():
    """Load the audit module by file path.

    Importing it through the ``mtc`` package would run ``mtc/__init__.py``,
    which pulls in the numerical stack. The gate must run in a bare Python
    (the pre-commit hook, the CI audit job) with only the standard library.
    """
    spec = importlib.util.spec_from_file_location("mtc_privacy_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_audit = _load_audit_module()
Hit = _audit.Hit
load_allowlist = _audit.load_allowlist
load_denylist = _audit.load_denylist
run_audit = _audit.run_audit
run_runtime_audit = _audit.run_runtime_audit
runtime_files = _audit.runtime_files
runtime_summary = _audit.runtime_summary

DENYLIST = ROOT / "config" / "privacy" / "denylist.txt"
ALLOWLIST = ROOT / "config" / "privacy" / "allowlist.txt"
LOCAL_EXTRA = ROOT / "config" / "privacy" / "denylist.local.txt"

_COMMIT_LINE = re.compile(r"^commit:([0-9a-f]{40,64})$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def git_bytes(
    *args: str, root: Optional[Path] = None, stdin: Optional[bytes] = None
) -> bytes:
    """Run a git command in ``root`` (default: this repository)."""
    return subprocess.run(
        ["git", *args],
        cwd=root if root is not None else ROOT,
        check=True,
        capture_output=True,
        input=stdin,
    ).stdout


def git_paths(*args: str, root: Optional[Path] = None) -> list:
    """Run a git command with -z output and return the NUL-separated paths."""
    out = git_bytes(*args, root=root).decode("utf-8", errors="surrogateescape")
    return [p for p in out.split("\0") if p]


def extra_denylist() -> list:
    env_path = os.environ.get("PRIVACY_DENYLIST_EXTRA")
    if env_path:
        candidate = Path(env_path)
        if not candidate.is_file():
            raise SystemExit(
                f"privacy audit: PRIVACY_DENYLIST_EXTRA points at {env_path}, "
                "which does not exist; refusing to run without the private list."
            )
        return load_denylist(candidate)
    if LOCAL_EXTRA.is_file():
        return load_denylist(LOCAL_EXTRA)
    return []


def materialize(tmp: Path, rel: str, data: bytes) -> Path:
    target = tmp / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def staged_files(tmp: Path, root: Optional[Path] = None) -> list:
    paths = git_paths(
        "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z", root=root
    )
    return [
        materialize(tmp, rel, git_bytes("show", f":{rel}", root=root)) for rel in paths
    ]


def _patch_text(ref: str, root: Optional[Path], *extra: str) -> str:
    return git_bytes(
        "-c",
        "core.quotePath=false",
        "log",
        "-p",
        "--no-color",
        "--no-ext-diff",
        "--diff-merges=remerge",
        *extra,
        f"{ref}..HEAD",
        root=root,
    ).decode("utf-8", errors="surrogateescape")


def _header_target(line: str) -> Tuple[Optional[str], bool]:
    """``(path, parsed)`` for a ``+++ `` header line.

    ``b/<path>`` gives the path and the null device gives none; any other
    form (a path git quotes) is not parsed. git ends the header of a path
    that holds a space with one TAB, which is not part of the path (a name
    that itself holds a TAB is quoted, so it is never parsed here).
    """
    target = line[4:]
    if target.endswith("\t"):
        target = target[:-1]
    if target.startswith("b/"):
        return target[2:], True
    if target == "/dev/null":
        return None, True
    return None, False


def added_lines(ref: str, root: Optional[Path] = None) -> Tuple[Dict[str, list], list]:
    """Map each path touched in ref..HEAD to the lines the commits added to it.

    Also returns one ``encoding`` hit per diff whose ``+++`` header names
    neither ``b/<path>`` nor the null device: its lines are never dropped
    silently.
    """
    added: Dict[str, list] = {
        rel: []
        for rel in git_paths(
            "log",
            "--format=",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            f"{ref}..HEAD",
            root=root,
        )
    }
    problems = []
    current = None
    in_header = False
    for line in _patch_text(ref, root, "--format=").splitlines():
        if line.startswith("diff --git "):
            current, in_header = None, True
        elif in_header and line.startswith("+++ "):
            current, parsed = _header_target(line)
            if not parsed:
                problems.append(Hit(line[4:], 0, "encoding", "unparsed diff header"))
        elif line.startswith("@@"):
            in_header = False
        elif not in_header and current is not None and line.startswith("+"):
            added.setdefault(current, []).append(line[1:])
    return added, problems


def added_lines_by_file(ref: str, root: Optional[Path] = None) -> dict:
    """Map each path touched in ref..HEAD to the lines the commits added to it."""
    return added_lines(ref, root)[0]


def added_line_numbers(
    ref: str, root: Optional[Path] = None
) -> Dict[Tuple[str, str], Set[int]]:
    """``(commit, path) -> new-side line numbers`` each commit added to a .py file."""
    numbers: Dict[Tuple[str, str], Set[int]] = {}
    commit = None
    current = None
    in_header = False
    for line in _patch_text(ref, root, "--format=commit:%H", "-U0").splitlines():
        marker = _COMMIT_LINE.match(line)
        if marker:
            commit, current, in_header = marker.group(1), None, False
        elif line.startswith("diff --git "):
            current, in_header = None, True
        elif in_header and line.startswith("+++ "):
            current, _parsed = _header_target(line)
        elif line.startswith("@@"):
            in_header = False
            hunk = _HUNK.match(line)
            if hunk is None or commit is None or current is None:
                continue
            if not current.endswith(".py"):
                continue
            start = int(hunk.group(1))
            count = int(hunk.group(2)) if hunk.group(2) is not None else 1
            if count:
                numbers.setdefault((commit, current), set()).update(
                    range(start, start + count)
                )
    return numbers


def post_images(
    keys: Sequence[Tuple[str, str]], root: Optional[Path] = None
) -> Dict[Tuple[str, str], bytes]:
    """Each ``(commit, path)``'s file content, read through one cat-file batch."""
    usable = [key for key in keys if "\n" not in key[1]]
    if not usable:
        return {}
    request = "".join(f"{commit}:{path}\n" for commit, path in usable)
    out = git_bytes(
        "cat-file",
        "--batch",
        root=root,
        stdin=request.encode("utf-8", errors="surrogateescape"),
    )
    images: Dict[Tuple[str, str], bytes] = {}
    position = 0
    for key in usable:
        end = out.index(b"\n", position)
        line = out[position:end]
        position = end + 1
        # A found object is "<oid> <type> <size>"; anything else names the
        # request (which may hold spaces) and ends in "missing" or similar.
        header = line.split(b" ")
        if len(header) != 3 or not header[2].isdigit():
            continue
        if not all(c in b"0123456789abcdef" for c in header[0]):
            continue
        size = int(header[2])
        images[key] = out[position : position + size]
        position += size + 1
    return images


def assembled_hits_in_range(ref: str, root: Optional[Path], tokens) -> List[object]:
    """Assembled-literal hits in the Python files the commits of ref..HEAD
    touched, counted only on the lines each commit added, each carrying the
    commit's short hash and the line number in that commit's version."""
    numbers = added_line_numbers(ref, root)
    images = post_images(list(numbers), root)
    hits = []
    for (commit, path), lines in numbers.items():
        data = images.get((commit, path))
        if data is None:
            continue
        text, _problem = _audit._decode(data)
        for hit in _audit.assembled_hits(path, text, tokens, only_lines=lines):
            hits.append(dataclasses.replace(hit, commit=commit[:SHORT_HASH]))
    return hits


def patch_hits(
    ref: str,
    root: Optional[Path] = None,
    denylist: Sequence[str] = (),
    allowlist: Sequence[object] = (),
    extra: Optional[Sequence[str]] = None,
) -> list:
    """Every hit of patch mode for ref..HEAD in the repository at ``root``."""
    root = root if root is not None else ROOT
    added, problems = added_lines(ref, root)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        files = [
            materialize(
                tmp, rel, "\n".join(lines).encode("utf-8", errors="surrogateescape")
            )
            for rel, lines in added.items()
        ]
        hits = run_audit(
            files,
            denylist,
            allowlist,
            extra_denylist=extra,
            root=tmp,
            assemble_python=False,
        )
    tokens = _audit.compile_tokens(denylist, extra)
    later = problems + assembled_hits_in_range(ref, root, tokens)
    return hits + [h for h in later if not any(r.covers(h.path, h) for r in allowlist)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--patch-from", metavar="REF")
    mode.add_argument("--paths", nargs="+")
    mode.add_argument("--runtime", nargs="+", metavar="DIR")
    args = parser.parse_args(argv)

    denylist = load_denylist(DENYLIST)
    allowlist = load_allowlist(ALLOWLIST)
    extra = extra_denylist()

    def audit(files, root, stats=None):
        return run_audit(
            files, denylist, allowlist, extra_denylist=extra, root=root, stats=stats
        )

    if args.runtime:
        directories = [Path(item).expanduser().resolve() for item in args.runtime]
        missing = [str(item) for item in directories if not item.is_dir()]
        if missing:
            raise SystemExit(f"privacy audit: not a directory: {', '.join(missing)}")
        hits = run_runtime_audit(directories, denylist, allowlist, extra_denylist=extra)
        print(runtime_summary(directories))
    elif args.patch_from:
        hits = patch_hits(args.patch_from, ROOT, denylist, allowlist, extra)
    elif args.staged:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            hits = audit(staged_files(tmp), tmp)
    else:
        if args.paths:
            files = [(ROOT / p).resolve() for p in args.paths]
            missing = [str(f) for f in files if not f.exists()]
            if missing:
                raise SystemExit(f"privacy audit: no such path: {', '.join(missing)}")
        else:
            files = [ROOT / p for p in git_paths("ls-files", "-z")]
            files = [f for f in files if f.exists()]
        stats = _audit.AuditStats()
        hits = audit(files, ROOT, stats)
        print(_audit.tree_summary(stats))

    for hit in hits:
        print(hit)
    if hits:
        print(
            f"\nprivacy audit: {len(hits)} hit(s). "
            "Fix them or add a justified allowlist rule.",
            file=sys.stderr,
        )
        return 1
    print("privacy audit: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
