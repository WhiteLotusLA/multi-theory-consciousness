"""Privacy audit: fail closed on identity residue, personal data and artifacts.

Two denylists feed the token check: a public one committed in
config/privacy/denylist.txt, and a private extra list that never enters the
repository (PRIVACY_DENYLIST_EXTRA or config/privacy/denylist.local.txt).
Pattern checks need no list. Allowlist rules suppress known-legitimate hits.

Hits on private extra tokens report a placeholder (``private#N``, the 1-based
position in the private list) instead of the token, so the audit output never
prints the names the private list exists to guard.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Sequence, Tuple

KINDS = (
    "token",
    "email",
    "phone",
    "coords",
    "homepath",
    "ipv4",
    "secret",
    "dash",
    "emoji",
    "artifact",
    "encoding",
)

PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?!\d)"),
    "coords": re.compile(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}"),
    "homepath": re.compile(
        r"(?<![A-Za-z0-9])"
        r"(/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|~/Library|/Volumes/)"
    ),
    "ipv4": re.compile(
        r"(?<![\d.])(?!127\.)(?!0\.0\.0\.0)(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"
    ),
    "secret": re.compile(
        r"\b(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,})\b"
    ),
    # En dash (U+2013) and em dash (U+2014), written as escapes.
    "dash": re.compile("[\u2013\u2014]"),
}
# Pictographs, symbols and dingbats.
EMOJI = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]")

ARTIFACT_PARTS = {
    "__pycache__",
    "CMakeFiles",
    "build",
    ".egg-info",
    "node_modules",
    ".DS_Store",
}
ARTIFACT_SUFFIXES = (".pyc", ".pyo", ".so", ".dylib", ".dump", ".jsonl", ".egg-info")
ARTIFACT_TOP_DIRS = {"agent_space", "results", "logs"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".cfg",
    ".ini",
    ".sql",
    ".tex",
    ".bib",
    ".csv",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".env",
    ".example",
    ".cypher",
    ".in",
}


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    kind: str
    match: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.kind}:{self.match}"


@dataclass(frozen=True)
class AllowRule:
    glob: str
    kind: str
    pattern: str

    def covers(self, rel_path: str, hit: Hit) -> bool:
        path_ok = fnmatch.fnmatchcase(rel_path, self.glob) or _glob_match_nested(
            rel_path, self.glob
        )
        kind_ok = self.kind in ("*", hit.kind)
        return path_ok and kind_ok and re.search(self.pattern, hit.match) is not None


def _glob_match_nested(rel_path: str, glob: str) -> bool:
    if glob.endswith("/**"):
        base = glob[:-3]
        return rel_path.startswith(base + "/") or rel_path == base
    return False


def load_denylist(path: Path) -> List[str]:
    tokens: List[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            tokens.append(line)
    return tokens


def load_allowlist(path: Path) -> List[AllowRule]:
    """Parse ``glob kind regex`` lines; a malformed line raises (fail closed)."""
    rules: List[AllowRule] = []
    for lineno, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 2)
        if len(fields) != 3:
            raise ValueError(f"{path}:{lineno}: expected 'glob kind regex': {raw!r}")
        glob, kind, pattern = fields
        if kind != "*" and kind not in KINDS:
            raise ValueError(f"{path}:{lineno}: unknown kind {kind!r}")
        re.compile(pattern)
        rules.append(AllowRule(glob=glob, kind=kind, pattern=pattern))
    return rules


def _token_regex(token: str) -> re.Pattern:
    escaped = re.escape(token)
    if re.fullmatch(r"[A-Za-z' ]+", token):
        return re.compile(rf"(?<![A-Za-z0-9_]){escaped}", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def _is_artifact(rel: PurePosixPath) -> bool:
    parts = rel.parts
    if any(p in ARTIFACT_PARTS or p.endswith(".egg-info") for p in parts):
        return True
    if rel.suffix in ARTIFACT_SUFFIXES or rel.name.endswith(".egg-info"):
        return True
    if parts and parts[0] in ARTIFACT_TOP_DIRS:
        return True
    if len(parts) > 1 and parts[0] == "data" and parts[1] != "dcm":
        return True
    return False


def _read_text(path: Path) -> Tuple[str, Optional[str]]:
    """Return ``(text, problem)`` for a file the audit must scan.

    The gate fails closed: a file that cannot be read, is not valid UTF-8, or
    carries a NUL byte is never skipped. It is decoded best effort (UTF-16 and
    UTF-32 by byte-order mark, otherwise UTF-8 with replacement and NULs
    dropped, which also recovers ASCII text stored as UTF-16 without a mark)
    and still scanned, and ``problem`` names the defect so an ``encoding`` hit
    is reported as well.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        return "", f"unreadable ({exc.__class__.__name__})"
    if b"\x00" not in data:
        try:
            return data.decode("utf-8"), None
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace"), "not utf-8"
    for bom, codec in (
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if data.startswith(bom):
            return data.decode(codec, errors="replace"), codec
    text = data.replace(b"\x00", b"").decode("utf-8", errors="replace")
    return text, "nul byte"


def _compile_tokens(
    denylist: Sequence[str], extra_denylist: Optional[Sequence[str]]
) -> List[Tuple[str, re.Pattern]]:
    labelled = [(t, _token_regex(t)) for t in denylist]
    for index, token in enumerate(extra_denylist or [], start=1):
        labelled.append((f"private#{index}", _token_regex(token)))
    return labelled


def _scan_text(
    rel_str: str,
    text: str,
    tokens: Sequence[Tuple[str, re.Pattern]],
    is_python: bool,
) -> List[Hit]:
    patterns = list(PATTERNS.items()) + ([("emoji", EMOJI)] if is_python else [])
    hits: List[Hit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for label, rx in tokens:
            if rx.search(line):
                hits.append(Hit(rel_str, lineno, "token", label))
        for kind, rx in patterns:
            m = rx.search(line)
            if m:
                hits.append(Hit(rel_str, lineno, kind, m.group(0)))
    return hits


def run_audit(
    paths: Iterable[Path],
    denylist: Sequence[str],
    allowlist: Sequence[AllowRule],
    extra_denylist: Optional[Sequence[str]] = None,
    root: Optional[Path] = None,
) -> List[Hit]:
    tokens = _compile_tokens(denylist, extra_denylist)
    hits: List[Hit] = []
    expanded = []
    for candidate in paths:
        candidate = Path(candidate)
        if candidate.is_dir():
            expanded.extend(sorted(f for f in candidate.rglob("*") if f.is_file()))
        else:
            expanded.append(candidate)
    for path in expanded:
        path = Path(path)
        rel = PurePosixPath(
            str(path.relative_to(root) if root else path).replace("\\", "/")
        )
        if _is_artifact(rel):
            hits.append(Hit(str(rel), 0, "artifact", rel.name))
            continue
        # Files with no suffix (Makefile, LICENSE, .gitignore) are scanned;
        # directories and files with an unknown suffix are skipped. Every
        # scanned file is read fail closed (see _read_text).
        suffix = path.suffix.lower()
        if path.is_dir() or (suffix and suffix not in TEXT_SUFFIXES):
            continue
        text, problem = _read_text(path)
        if problem is not None:
            hits.append(Hit(str(rel), 0, "encoding", problem))
        hits.extend(_scan_text(str(rel), text, tokens, suffix == ".py"))
    return [h for h in hits if not any(r.covers(h.path, h) for r in allowlist)]
