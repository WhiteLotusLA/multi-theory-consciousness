#!/usr/bin/env python3
"""Run the privacy audit over tracked files, staged files, or a commit range.

Usage:
  python scripts/privacy_audit.py                 # all tracked files
  python scripts/privacy_audit.py --staged        # staged content (the index)
  python scripts/privacy_audit.py --patch-from origin/main   # lines added since ref
  python scripts/privacy_audit.py --paths a.py b.md

Exit code 1 on any hit. The private extra denylist is read from
$PRIVACY_DENYLIST_EXTRA or config/privacy/denylist.local.txt when present.

--staged audits the blobs in the index, not the working tree, so a file that
was fixed on disk but not re-staged still blocks the commit. --patch-from
audits every line added by the commits in REF..HEAD, file by file, so path
allowlist rules apply; line numbers then count added lines within that file.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mtc" / "devtools" / "privacy_audit.py"


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
load_allowlist = _audit.load_allowlist
load_denylist = _audit.load_denylist
run_audit = _audit.run_audit

DENYLIST = ROOT / "config" / "privacy" / "denylist.txt"
ALLOWLIST = ROOT / "config" / "privacy" / "allowlist.txt"
LOCAL_EXTRA = ROOT / "config" / "privacy" / "denylist.local.txt"


def git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True
    ).stdout


def git_paths(*args: str) -> list:
    """Run a git command with -z output and return the NUL-separated paths."""
    out = git_bytes(*args).decode("utf-8", errors="surrogateescape")
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


def staged_files(tmp: Path) -> list:
    paths = git_paths("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [materialize(tmp, rel, git_bytes("show", f":{rel}")) for rel in paths]


def added_lines_by_file(ref: str) -> dict:
    """Map each path touched in ref..HEAD to the lines the commits added to it."""
    added = {
        rel: []
        for rel in git_paths(
            "log",
            "--format=",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            f"{ref}..HEAD",
        )
    }
    patch = git_bytes(
        "-c",
        "core.quotePath=false",
        "log",
        "-p",
        "--no-color",
        "--no-ext-diff",
        "--format=",
        f"{ref}..HEAD",
    ).decode("utf-8", errors="surrogateescape")
    current = None
    in_header = False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            current, in_header = None, True
        elif in_header and line.startswith("+++ "):
            target = line[4:]
            current = target[2:] if target.startswith("b/") else None
        elif line.startswith("@@"):
            in_header = False
        elif not in_header and current is not None and line.startswith("+"):
            added.setdefault(current, []).append(line[1:])
    return added


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--patch-from", metavar="REF")
    mode.add_argument("--paths", nargs="+")
    args = parser.parse_args()

    denylist = load_denylist(DENYLIST)
    allowlist = load_allowlist(ALLOWLIST)
    extra = extra_denylist()

    def audit(files, root):
        return run_audit(files, denylist, allowlist, extra_denylist=extra, root=root)

    if args.patch_from:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            files = [
                materialize(
                    tmp,
                    rel,
                    "\n".join(lines).encode("utf-8", errors="surrogateescape"),
                )
                for rel, lines in added_lines_by_file(args.patch_from).items()
            ]
            hits = audit(files, tmp)
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
        hits = audit(files, ROOT)

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
