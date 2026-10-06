"""Privacy audit: fail closed on identity residue, personal data and artifacts.

Two denylists feed the token check: a public one committed in
config/privacy/denylist.txt, and a private extra list that never enters the
repository (PRIVACY_DENYLIST_EXTRA or config/privacy/denylist.local.txt).
In Python files, expressions built only from literals (``"a" + "b"``,
``"".join([...])``, ``chr(...)``) are assembled and matched too, so a token
split into pieces is still a hit (kind ``assembled-token``).
Python source is ASCII only: a non-ASCII character that is not an emoji or a
dash is reported by code point (kind ``nonascii``), which closes the homoglyph
and encoded-name routes around the token check.
Pattern checks need no list. Allowlist rules suppress known-legitimate hits.

Hits on private extra tokens report a placeholder (``private#N``, the 1-based
position in the private list) instead of the token, so the audit output never
prints the names the private list exists to guard.

``run_runtime_audit`` audits directories outside the repository (a log or a
results directory). It differs from ``run_audit`` in one rule only: the
artifact rule is off, because a runtime directory legitimately holds files
the tracked tree must not (line-delimited JSON, logs), and those files are
scanned as text instead of being flagged. Tokens, patterns, the encoding
check and the allowlist apply unchanged; paths are reported as
``<directory name>/<relative path>``.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
# Python source is ASCII only: any other non-ASCII character (a homoglyph, an
# encoded or accented name, a zero-width character) is reported by code point.
DASHES = "\u2013\u2014"

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
# Runtime mode only: file kinds a log or results directory holds as text. A
# rotated file (``name.log.1``) counts by its inner suffix.
RUNTIME_TEXT_SUFFIXES = (".jsonl", ".log")


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
        if is_python:
            others = sorted(
                {
                    c
                    for c in line
                    if ord(c) > 0x7F and not EMOJI.match(c) and c not in DASHES
                }
            )
            if others:
                codes = " ".join(f"U+{ord(c):04X}" for c in others[:5])
                hits.append(Hit(rel_str, lineno, "nonascii", codes))
    return hits


def _constant_pieces(node: ast.AST) -> Optional[List[str]]:
    """The string pieces of an expression built only from literals, else None.

    Covers str constants, ``chr(<int>)``, ``+`` chains of those, and
    ``"<sep>".join([...])`` over a literal list or tuple. Anything that
    involves a name, a call with arguments or a format is left to the plain
    text scan.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_pieces(node.left)
        right = _constant_pieces(node.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
        func = node.func
        arg = node.args[0]
        if isinstance(func, ast.Name) and func.id == "chr":
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                try:
                    return [chr(arg.value)]
                except (ValueError, OverflowError):
                    return None
            return None
        if isinstance(func, ast.Attribute) and func.attr == "join":
            sep = _constant_pieces(func.value)
            if sep is None or not isinstance(arg, (ast.List, ast.Tuple)):
                return None
            parts: List[str] = []
            for elt in arg.elts:
                piece = _constant_pieces(elt)
                if piece is None:
                    return None
                parts.append("".join(piece))
            return ["".join(sep).join(parts)]
    return None


def _scan_assembled_literals(
    rel_str: str, text: str, tokens: Sequence[Tuple[str, re.Pattern]]
) -> List[Hit]:
    """Denylist tokens that appear only once literal pieces are joined.

    The text scan sees ``"Pla" + "nted"`` as two harmless pieces, but the
    assembled value is what a reader of the repository ends up with, so every
    literal-only expression is assembled and matched as well. Python that does
    not parse is left to the text scan (and to the linters).
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    hits: List[Hit] = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.BinOp, ast.Call)):
            continue
        pieces = _constant_pieces(node)
        if pieces is None:
            continue
        value = "".join(pieces)
        lineno = getattr(node, "lineno", 0)
        for label, rx in tokens:
            if rx.search(value) and (lineno, label) not in seen:
                seen.add((lineno, label))
                hits.append(Hit(rel_str, lineno, "assembled-token", label))
    return sorted(hits, key=lambda h: (h.line, h.match))


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
        hits.extend(_scan_file(path, str(rel), tokens))
    return [h for h in hits if not any(r.covers(h.path, h) for r in allowlist)]


def _scan_file(
    path: Path, rel_str: str, tokens: Sequence[Tuple[str, re.Pattern]]
) -> List[Hit]:
    """Every hit in one text file: the encoding check, tokens and patterns."""
    hits: List[Hit] = []
    suffix = path.suffix.lower()
    text, problem = _read_text(path)
    if problem is not None:
        hits.append(Hit(rel_str, 0, "encoding", problem))
    scanned = _scan_text(rel_str, text, tokens, suffix == ".py")
    if suffix == ".py":
        whole = {(h.line, h.match) for h in scanned if h.kind == "token"}
        scanned.extend(
            h
            for h in _scan_assembled_literals(rel_str, text, tokens)
            if (h.line, h.match) not in whole
        )
    hits.extend(scanned)
    return hits


def _runtime_scannable(path: Path) -> bool:
    """Whether runtime mode reads a file as text.

    The same rule as the tracked tree (no suffix, or a text suffix), plus the
    runtime text kinds, which the tracked tree treats as artifacts.
    """
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if not suffixes or suffixes[-1] in TEXT_SUFFIXES:
        return True
    return any(suffix in RUNTIME_TEXT_SUFFIXES for suffix in suffixes)


def _runtime_entries(dirs: Iterable[Path]) -> List[Tuple[Path, str]]:
    """``(file, reported path)`` for every file runtime mode scans."""
    entries: List[Tuple[Path, str]] = []
    for directory in dirs:
        base = Path(directory)
        if not base.is_dir():
            raise NotADirectoryError(f"not a directory: {base.name}")
        for path in sorted(f for f in base.rglob("*") if f.is_file()):
            if not _runtime_scannable(path):
                continue
            relative = PurePosixPath(base.name) / PurePosixPath(
                path.relative_to(base).as_posix()
            )
            entries.append((path, str(relative)))
    return entries


def runtime_files(dirs: Iterable[Path]) -> List[Path]:
    """The files ``run_runtime_audit`` scans under ``dirs``."""
    return [path for path, _rel in _runtime_entries(dirs)]


def runtime_skipped(dirs: Iterable[Path]) -> List[Path]:
    """The files under ``dirs`` that runtime mode does not read as text.

    They are reported by count, so a directory of binaries never reads as
    audited; a path that is not a directory raises ``NotADirectoryError``.
    """
    skipped: List[Path] = []
    for directory in dirs:
        base = Path(directory)
        if not base.is_dir():
            raise NotADirectoryError(f"not a directory: {base.name}")
        skipped.extend(
            sorted(
                f for f in base.rglob("*") if f.is_file() and not _runtime_scannable(f)
            )
        )
    return skipped


def runtime_summary(dirs: Iterable[Path]) -> str:
    """One line: how many files runtime mode scans and how many it skips."""
    dirs = list(dirs)
    scanned = len(runtime_files(dirs))
    skipped = runtime_skipped(dirs)
    by_suffix: Dict[str, int] = {}
    for path in skipped:
        suffix = path.suffix.lower()
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
    line = (
        f"privacy audit: runtime mode scanned {scanned} file(s); skipped "
        f"{len(skipped)} file(s) not read as text"
    )
    if by_suffix:
        line += " (" + ", ".join(f"{k} {v}" for k, v in sorted(by_suffix.items())) + ")"
    return line


def run_runtime_audit(
    dirs: Iterable[Path],
    denylist: Sequence[str],
    allowlist: Sequence[AllowRule],
    extra_denylist: Optional[Sequence[str]] = None,
) -> List[Hit]:
    """Audit runtime directories (see the module docstring).

    Only the artifact rule is off. A path that is not a directory raises
    ``NotADirectoryError``: a directory that was not scanned is never clean.
    """
    tokens = _compile_tokens(denylist, extra_denylist)
    hits: List[Hit] = []
    for path, rel_str in _runtime_entries(dirs):
        hits.extend(_scan_file(path, rel_str, tokens))
    return [h for h in hits if not any(r.covers(h.path, h) for r in allowlist)]
