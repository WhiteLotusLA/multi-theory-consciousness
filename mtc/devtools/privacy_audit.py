"""Privacy audit: fail closed on identity residue, personal data and artifacts.

Two denylists feed the token check: a public one committed in
config/privacy/denylist.txt, and a private extra list that never enters the
repository (PRIVACY_DENYLIST_EXTRA or config/privacy/denylist.local.txt).
A purely alphabetic token matches, case-insensitively, where it starts after
anything that is not a letter (so after an underscore or a digit), or at a
camelCase boundary (a lower-case letter, then the token with its first letter
capitalised, not followed by another lower-case letter); inside a longer word
after a letter it is not a hit. Any other token matches as a substring.

Hidden forms are matched as kind ``assembled-token``:

- in Python files, expressions built only from literals: every string and
  bytes constant (so adjacent literals and escapes inside one literal),
  ``+`` chains of strings or bytes, ``"<sep>".join([...])``, ``chr(<int>)``,
  f-strings whose formatted values are literals, ``%`` and ``.format`` with
  literal operands, and a slice of a literal with constant bounds. A name
  computed at run time from non-literal data cannot be seen statically.
- in every text file, a decoded view of each line (backslash-x, backslash-u
  and backslash-U escapes resolved) and a folded view (compatibility
  decomposition, combining marks and format characters such as zero-width
  spaces dropped, and a small table of Cyrillic and Greek letters that look
  like Latin ones; not the full Unicode confusables set).

Python source is ASCII only: a non-ASCII character that is not an emoji or a
dash is reported by code point (kind ``nonascii``). Every distinct match of a
pattern on a line is reported, so an allowlisted first match hides nothing.

A file of unknown suffix is scanned as text when its bytes start with a UTF-16
or UTF-32 mark, or hold no NUL and decode as UTF-8; a gzip, tar or zip archive
is opened and each member is scanned as ``<archive>!<member>`` (two levels of
nesting at most), with AppleDouble ``._*`` members reported as artifacts; any
other binary file is skipped and counted (``AuditStats``).
Pattern checks need no list. Allowlist rules suppress known-legitimate hits;
their globs may name an archive member.

Hits on private extra tokens report a placeholder (``private#N``, the 1-based
position in the private list) instead of the token, so the audit output never
prints the names the private list exists to guard.

``run_runtime_audit`` audits directories outside the repository (a log or a
results directory). It differs from ``run_audit`` in one rule only: the
artifact rule is off, because a runtime directory legitimately holds files
the tracked tree must not (line-delimited JSON, logs), and those files are
scanned as text instead of being flagged. Tokens, patterns, the encoding
check and the allowlist apply unchanged; paths are reported as
``<directory name>/<relative path>``. Runtime mode reads files by suffix only:
it neither sniffs unknown suffixes nor opens archives, and counts every file
it does not read.
"""

from __future__ import annotations

import ast
import fnmatch
import gzip
import io
import re
import string
import tarfile
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    "nonascii",
    "assembled-token",
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

# Cyrillic and Greek letters that look like Latin ones, by code point. A small
# table, not the full Unicode confusables set.
LOOKALIKES: Dict[int, str] = {
    # Cyrillic lower case
    0x0430: "a",
    0x0432: "b",
    0x0435: "e",
    0x043A: "k",
    0x043C: "m",
    0x043D: "h",
    0x043E: "o",
    0x0440: "p",
    0x0441: "c",
    0x0442: "t",
    0x0443: "y",
    0x0445: "x",
    0x0455: "s",
    0x0456: "i",
    0x0458: "j",
    0x0501: "d",
    0x04CF: "l",
    # Cyrillic upper case
    0x0410: "A",
    0x0412: "B",
    0x0415: "E",
    0x041A: "K",
    0x041C: "M",
    0x041D: "H",
    0x041E: "O",
    0x0420: "P",
    0x0421: "C",
    0x0422: "T",
    0x0423: "Y",
    0x0425: "X",
    0x0405: "S",
    0x0406: "I",
    0x0408: "J",
    # Greek lower case
    0x03B1: "a",
    0x03B9: "i",
    0x03BA: "k",
    0x03BD: "v",
    0x03BF: "o",
    0x03C1: "p",
    0x03C4: "t",
    0x03C5: "u",
    0x03C7: "x",
    # Greek upper case
    0x0391: "A",
    0x0392: "B",
    0x0395: "E",
    0x0396: "Z",
    0x0397: "H",
    0x0399: "I",
    0x039A: "K",
    0x039C: "M",
    0x039D: "N",
    0x039F: "O",
    0x03A1: "P",
    0x03A4: "T",
    0x03A5: "Y",
    0x03A7: "X",
}
# Backslash-x, backslash-u and backslash-U escapes written into a line.
ESCAPES = re.compile(r"\\(?:x([0-9A-Fa-f]{2})|u([0-9A-Fa-f]{4})|U([0-9A-Fa-f]{8}))")

# Archives: two levels of nesting at most, and bounded member sizes, so a
# crafted archive can neither recurse nor exhaust memory (both fail closed).
MAX_ARCHIVE_DEPTH = 2
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10000
# A literal-only format is not evaluated when a conversion specifier holds a
# width or precision of 10000 or more, written into the template or taken
# from an operand (a %-format star, a nested replacement field, a nested
# f-string field), when such a dynamic width or precision is not a number,
# or when the result would exceed the size bound. Digits, stars and braces
# outside a conversion specifier are plain text. A refused expression is
# reported as an encoding hit (fail closed), never dropped.
_HUGE_NUMBER = 10000
_MAX_ASSEMBLED_CHARS = 100000
# One %-conversion: '%%', or the mapping key, flags, width, precision, length
# modifier and conversion character of one specifier.
_PERCENT_SPEC = re.compile(
    r"%%|%(?P<key>\()?[-+ #0]*(?P<width>\*|\d+)?(?:\.(?P<prec>\*|\d*))?[hlL]?",
    re.DOTALL,
)
_DIGITS = re.compile(r"\d+")
UNEVALUATED_FORMAT = (
    "unevaluated literal format (width or precision too large or not a number)"
)


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    kind: str
    match: str
    # Patch mode: the short hash of the commit an assembled hit was found in.
    commit: Optional[str] = None

    def __str__(self) -> str:
        base = f"{self.path}:{self.line}:{self.kind}:{self.match}"
        return f"{base} @{self.commit}" if self.commit else base


@dataclass
class AuditStats:
    """What a tree audit read: files and members scanned as text, and the
    paths it skipped because they are neither text nor an archive."""

    scanned: int = 0
    skipped: List[str] = field(default_factory=list)


def tree_summary(stats: AuditStats) -> str:
    """One line: how many files the tree audit scanned and skipped."""
    by_suffix: Dict[str, int] = {}
    for rel in stats.skipped:
        suffix = PurePosixPath(rel.rsplit("!", 1)[-1]).suffix.lower()
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
    line = (
        f"privacy audit: scanned {stats.scanned} file(s); skipped "
        f"{len(stats.skipped)} file(s) not read as text"
    )
    if by_suffix:
        line += " (" + ", ".join(f"{k} {v}" for k, v in sorted(by_suffix.items())) + ")"
    return line


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


# A token's lower-case text, which the lower-cased text of any match of an
# ASCII token in ASCII text contains: a cheap test before the regex runs. Text
# that is not ASCII always goes to the regex, because a case-insensitive match
# can pair an ASCII letter with one that lower-cases to something else (the
# dotless i, the long s, the Kelvin sign).
_NEEDLES: Dict[re.Pattern, str] = {}


def _token_regex(token: str) -> re.Pattern:
    compiled = _token_matcher(token)
    if token.isascii() and token:
        _NEEDLES[compiled] = token.lower()
    return compiled


def _may_hold(lowered: str, rx: re.Pattern) -> bool:
    needle = _NEEDLES.get(rx)
    return needle is None or needle in lowered or not lowered.isascii()


def _token_matcher(token: str) -> re.Pattern:
    """The matcher of one denylist token (see the module docstring).

    A purely alphabetic token matches case-insensitively after anything that
    is not a letter, or at a camelCase boundary: a lower-case letter, then the
    token with its first letter in upper case and the rest in any case, not
    followed by a lower-case letter (so ``myToken`` is a hit and ``xsTokeny``,
    a longer word, is not). Suffixed forms after a boundary stay hits.
    """
    escaped = re.escape(token)
    if re.fullmatch(r"[A-Za-z' ]+", token):
        plain = rf"(?<![A-Za-z])(?i:{escaped})"
        first, rest = token[0], token[1:]
        if not first.isalpha():
            return re.compile(plain)
        camel = rf"(?<=[a-z]){re.escape(first.upper())}(?i:{re.escape(rest)})(?![a-z])"
        return re.compile(rf"{plain}|{camel}")
    return re.compile(escaped, re.IGNORECASE)


def compile_tokens(
    denylist: Sequence[str], extra_denylist: Optional[Sequence[str]] = None
) -> List[Tuple[str, re.Pattern]]:
    """``(label, matcher)`` pairs; a private token is labelled ``private#N``."""
    return _compile_tokens(denylist, extra_denylist)


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


def _is_member_artifact(rel: PurePosixPath) -> bool:
    """An archive member that is a build or metadata artifact.

    AppleDouble members (``._name``, or anything under ``__MACOSX``) carry
    file-system metadata such as extended attributes, never content.
    """
    if rel.name.startswith("._") or "__MACOSX" in rel.parts:
        return True
    if any(p in ARTIFACT_PARTS or p.endswith(".egg-info") for p in rel.parts):
        return True
    return rel.suffix in ARTIFACT_SUFFIXES


def _decode(data: bytes) -> Tuple[str, Optional[str]]:
    """Decode bytes the audit must scan; return ``(text, problem)``.

    The gate fails closed: content that is not valid UTF-8 or carries a NUL
    byte is never skipped. It is decoded best effort (UTF-16 and UTF-32 by
    byte-order mark, otherwise UTF-8 with replacement and NULs dropped, which
    also recovers ASCII text stored as UTF-16 without a mark) and still
    scanned, and ``problem`` names the defect so an ``encoding`` hit is
    reported as well.
    """
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


def _read_text(path: Path) -> Tuple[str, Optional[str]]:
    """Return ``(text, problem)`` for a file the audit must scan (see _decode).

    A file that cannot be read yields an empty text and a problem.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        return "", f"unreadable ({exc.__class__.__name__})"
    return _decode(data)


def _sniff(data: bytes) -> str:
    """``gzip``, ``zip``, ``tar``, ``text`` or ``binary`` by content."""
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if len(data) > 262 and data[257:262] == b"ustar":
        return "tar"
    if data.startswith((b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff")):
        return "text"
    if b"\x00" in data:
        return "binary"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "binary"
    return "text"


def _compile_tokens(
    denylist: Sequence[str], extra_denylist: Optional[Sequence[str]]
) -> List[Tuple[str, re.Pattern]]:
    labelled = [(t, _token_regex(t)) for t in denylist]
    for index, token in enumerate(extra_denylist or [], start=1):
        labelled.append((f"private#{index}", _token_regex(token)))
    return labelled


def _decoded_view(line: str) -> str:
    """The line with backslash-x, -u and -U escapes resolved."""

    def resolve(match: re.Match) -> str:
        code = int(next(group for group in match.groups() if group), 16)
        try:
            return chr(code)
        except (ValueError, OverflowError):
            return match.group(0)

    return ESCAPES.sub(resolve, line)


def _folded_view(line: str) -> str:
    """The line decomposed, with marks and format characters dropped and the
    lookalike letters mapped to Latin ones."""
    decomposed = unicodedata.normalize("NFKD", line)
    kept = "".join(c for c in decomposed if unicodedata.category(c) not in ("Mn", "Cf"))
    return kept.translate(LOOKALIKES)


def _line_views(line: str) -> List[str]:
    """The hidden-form views of a line that differ from it (possibly none)."""
    views: List[str] = []
    candidates = [line]
    if "\\" in line:
        decoded = _decoded_view(line)
        if decoded != line:
            views.append(decoded)
            candidates.append(decoded)
    for candidate in candidates:
        if not candidate.isascii():
            folded = _folded_view(candidate)
            if folded != candidate and folded not in views:
                views.append(folded)
    return views


def _token_in(text: str, lowered: str, rx: re.Pattern) -> bool:
    return _may_hold(lowered, rx) and rx.search(text) is not None


def _line_token_hits(
    text: str, tokens: Sequence[Tuple[str, re.Pattern]]
) -> Dict[Tuple[int, str], str]:
    """``(line, label) -> kind`` for every token on a line: ``token`` when the
    line as written holds it, ``assembled-token`` when only a decoded or
    folded view does."""
    found: Dict[Tuple[int, str], str] = {}
    if not tokens:
        return found
    for lineno, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        views = [(view, view.lower()) for view in _line_views(line)]
        for label, rx in tokens:
            if _token_in(line, lowered, rx):
                found[(lineno, label)] = "token"
            elif any(_token_in(view, low, rx) for view, low in views):
                found[(lineno, label)] = "assembled-token"
    return found


def _scan_text(
    rel_str: str,
    text: str,
    tokens: Sequence[Tuple[str, re.Pattern]],
    is_python: bool,
    token_hits: Optional[Dict[Tuple[int, str], str]] = None,
) -> List[Hit]:
    patterns = list(PATTERNS.items()) + ([("emoji", EMOJI)] if is_python else [])
    if token_hits is None:
        token_hits = _line_token_hits(text, tokens)
    hits: List[Hit] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for label, _rx in tokens:
            kind = token_hits.get((lineno, label))
            if kind is not None:
                hits.append(Hit(rel_str, lineno, kind, label))
        for kind, rx in patterns:
            # Every distinct match: an allowlisted first match hides nothing.
            seen: List[str] = []
            for m in rx.finditer(line):
                if m.group(0) not in seen:
                    seen.append(m.group(0))
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


_NOT_LITERAL = object()
# A literal-only format the guard refused to evaluate (see _PERCENT_SPEC).
_REFUSED = object()


def _int_literal(node: ast.AST) -> Any:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        value = node.operand.value
        return -value if isinstance(node.op, ast.USub) else value
    return _NOT_LITERAL


def _add_operands(node: ast.AST) -> List[ast.AST]:
    """The operands of a left-nested ``+`` chain, without recursion."""
    operands: List[ast.AST] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.BinOp) and isinstance(current.op, ast.Add):
            stack.append(current.right)
            stack.append(current.left)
        else:
            operands.append(current)
    return operands


def _formattable(value: Any) -> bool:
    return isinstance(value, (str, bytes, int, float)) and not isinstance(value, bool)


def _literal(node: ast.AST, depth: int = 0) -> Any:
    """The value of an expression built only from literals, else _NOT_LITERAL.

    Evaluated forms: constants, tuples and lists of literals, ``+`` chains of
    strings or of bytes, ``%`` with literal operands, ``chr(<int>)``,
    ``<literal>.join(<literal sequence>)``, ``<literal>.format(<literals>)``,
    a subscript of a literal with constant bounds, and f-strings whose
    formatted values are literals. Any failure, and any name or other call,
    yields _NOT_LITERAL.
    """
    if depth > 50:
        return _NOT_LITERAL
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, bytes, int, float)) and not isinstance(
            node.value, bool
        ):
            return node.value
        return _NOT_LITERAL
    number = _int_literal(node)
    if number is not _NOT_LITERAL:
        return number
    if isinstance(node, (ast.Tuple, ast.List)):
        items = [_literal(elt, depth + 1) for elt in node.elts]
        blocked = _blocked(items)
        if blocked is not None:
            return blocked
        return tuple(items)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        values = [_literal(op, depth + 1) for op in _add_operands(node)]
        blocked = _blocked(values)
        if blocked is not None:
            return blocked
        if all(isinstance(v, str) for v in values):
            return "".join(values)
        if all(isinstance(v, bytes) for v in values):
            return b"".join(values)
        return _NOT_LITERAL
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        left = _literal(node.left, depth + 1)
        right = _literal(node.right, depth + 1)
        blocked = _blocked([left, right])
        if blocked is not None:
            return blocked
        if not isinstance(left, (str, bytes)):
            return _NOT_LITERAL
        return _guarded_percent(left, right)
    if isinstance(node, ast.Call):
        return _literal_call(node, depth)
    if isinstance(node, ast.Subscript):
        return _literal_subscript(node, depth)
    if isinstance(node, ast.JoinedStr):
        return _literal_fstring(node, depth)
    return _NOT_LITERAL


def _blocked(values: Sequence[Any]) -> Any:
    """_NOT_LITERAL or _REFUSED when a piece is one of them, else None."""
    if any(v is _NOT_LITERAL for v in values):
        return _NOT_LITERAL
    if any(v is _REFUSED for v in values):
        return _REFUSED
    return None


def _huge_digits(digits: str) -> bool:
    """True when a run of decimal digits is a number of 10000 or more."""
    return len(digits.lstrip("0")) >= len(str(_HUGE_NUMBER))


def _small_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and abs(value) < _HUGE_NUMBER
    )


def _huge_spec(spec: str) -> bool:
    """True when a resolved format spec holds a width or precision of 10000+.

    Once the fill and alignment are set aside, a spec holds digits only in its
    width and its precision, so any digit run of that size is one of them.
    """
    if spec[1:2] in ("<", ">", "=", "^"):
        rest = spec[2:]
    elif spec[:1] in ("<", ">", "=", "^"):
        rest = spec[1:]
    else:
        rest = spec
    return any(_huge_digits(m.group()) for m in _DIGITS.finditer(rest))


def _evaluated(evaluate, dynamic: bool) -> Any:
    """``evaluate()`` within the size bound.

    A format that raises is not literal, unless a width or precision of it
    comes from an operand: then the operand is what broke it, and the
    expression is refused (reported) rather than dropped.
    """
    try:
        value = evaluate()
    except Exception:
        return _REFUSED if dynamic else _NOT_LITERAL
    if not isinstance(value, (str, bytes)):
        return _NOT_LITERAL
    if len(value) > _MAX_ASSEMBLED_CHARS:
        return _REFUSED
    return value


def _guarded_percent(template: Any, right: Any) -> Any:
    """``template % right`` when it is safe to run, else _REFUSED.

    Each conversion specifier is read in order ('%%' is text): a written
    width or precision of 10000 or more is refused, and so is a star whose
    operand is not an integer below 10000. Only the operand a star consumes
    is checked; digits and stars elsewhere in the template are text.
    """
    text = template.decode("latin-1") if isinstance(template, bytes) else template
    operands = right if isinstance(right, tuple) else (right,)
    dynamic = False
    index = 0
    for match in _PERCENT_SPEC.finditer(text):
        if match.group(0) == "%%":
            continue
        if match.group("key"):
            # A mapping key needs a mapping operand, and no literal is one.
            return _NOT_LITERAL
        for part in (match.group("width"), match.group("prec")):
            if part == "*":
                dynamic = True
                operand = operands[index] if index < len(operands) else None
                index += 1
                if not _small_int(operand):
                    return _REFUSED
            elif part and _huge_digits(part):
                return _REFUSED
        index += 1
    return _evaluated(lambda: template % right, dynamic)


class _FormatRefused(Exception):
    """A replacement field of a literal format was refused."""


class _GuardedFormatter(string.Formatter):
    """``str.format`` that checks every resolved field spec before formatting.

    Nested replacement fields are substituted into the spec first, so a width
    taken from an operand (an integer or a digit string) is checked as if it
    had been written; the running output size is bounded as well.
    """

    def __init__(self) -> None:
        super().__init__()
        self.total = 0

    def format_field(self, value: Any, format_spec: str) -> Any:
        if _huge_spec(format_spec):
            raise _FormatRefused()
        rendered = format(value, format_spec)
        self.total += len(rendered)
        if self.total > _MAX_ASSEMBLED_CHARS:
            raise _FormatRefused()
        return rendered


_FIELD_FIRST = re.compile(r"[^.\[]*")


def _numbered_template(template: str) -> str:
    """``template`` with every automatic field given its explicit index.

    str.format numbers an automatic field whose name starts with an index or
    an attribute (``{[0]}``, ``{.real}``) as it numbers an empty one, and so
    does string.Formatter from Python 3.14 on; earlier versions look such a
    field up under the empty key and raise KeyError. With the indices written
    out (each field before the fields nested in its spec, the order
    str.format counts in), the guarded formatter agrees with str.format on
    every supported interpreter. Mixing automatic and manual numbering raises
    ValueError, as str.format does.
    """
    state = {"next": 0, "manual": False}

    def rebuild(text: str, depth: int) -> str:
        if depth > 3:
            raise ValueError("format spec nested too deeply")
        out: List[str] = []
        for literal, name, spec, conversion in string.Formatter().parse(text):
            out.append(literal.replace("{", "{{").replace("}", "}}"))
            if name is None:
                continue
            first = _FIELD_FIRST.match(name).group()
            if first == "":
                if state["manual"]:
                    raise ValueError("cannot switch to automatic field numbering")
                name = str(state["next"]) + name
                state["next"] += 1
            elif first.isdecimal():
                if state["next"]:
                    raise ValueError("cannot switch to manual field specification")
                state["manual"] = True
            field_text = "{" + name
            if conversion is not None:
                field_text += "!" + conversion
            if spec:
                field_text += ":" + rebuild(spec, depth + 1)
            out.append(field_text + "}")
        return "".join(out)

    return rebuild(template, 0)


def _guarded_str_format(owner: str, args: List[Any], kwargs: Dict[str, Any]) -> Any:
    formatter = _GuardedFormatter()
    try:
        dynamic = any(
            spec is not None and "{" in spec
            for _text, _name, spec, _conv in formatter.parse(owner)
        )
    except ValueError:
        return _NOT_LITERAL
    try:
        value = formatter.vformat(_numbered_template(owner), args, kwargs)
    except _FormatRefused:
        return _REFUSED
    except Exception:
        return _REFUSED if dynamic else _NOT_LITERAL
    return _evaluated(lambda: value, dynamic)


def _guarded_field(value: Any, spec: str, dynamic: bool) -> Any:
    """``format(value, spec)`` for one f-string field, guarded as above."""
    if _huge_spec(spec):
        return _REFUSED
    return _evaluated(lambda: format(value, spec), dynamic)


def _literal_call(node: ast.Call, depth: int) -> Any:
    func = node.func
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        return _NOT_LITERAL
    if any(kw.arg is None for kw in node.keywords):
        return _NOT_LITERAL
    if isinstance(func, ast.Name) and func.id == "chr":
        if len(node.args) != 1 or node.keywords:
            return _NOT_LITERAL
        code = _int_literal(node.args[0])
        if code is _NOT_LITERAL:
            return _NOT_LITERAL
        try:
            return chr(code)
        except (ValueError, OverflowError):
            return _NOT_LITERAL
    if not isinstance(func, ast.Attribute):
        return _NOT_LITERAL
    owner = _literal(func.value, depth + 1)
    if owner is _REFUSED and func.attr in ("join", "format"):
        return _REFUSED
    if func.attr == "join" and len(node.args) == 1 and not node.keywords:
        items = _literal(node.args[0], depth + 1)
        if items is _REFUSED and isinstance(owner, (str, bytes)):
            return _REFUSED
        if not isinstance(owner, (str, bytes)) or not isinstance(items, tuple):
            return _NOT_LITERAL
        if not all(type(item) is type(owner) for item in items):
            return _NOT_LITERAL
        return owner.join(items)
    if func.attr == "format" and isinstance(owner, str):
        args = [_literal(arg, depth + 1) for arg in node.args]
        kwargs = {kw.arg: _literal(kw.value, depth + 1) for kw in node.keywords}
        values = args + list(kwargs.values())
        blocked = _blocked(values)
        if blocked is not None:
            return blocked
        if any(not _formattable(v) for v in values):
            return _NOT_LITERAL
        return _guarded_str_format(owner, args, kwargs)
    return _NOT_LITERAL


def _literal_subscript(node: ast.Subscript, depth: int) -> Any:
    value = _literal(node.value, depth + 1)
    if value is _REFUSED:
        return _REFUSED
    if not isinstance(value, (str, bytes)):
        return _NOT_LITERAL
    index = node.slice
    if isinstance(index, ast.Slice):
        bounds = []
        for part in (index.lower, index.upper, index.step):
            if part is None:
                bounds.append(None)
                continue
            number = _int_literal(part)
            if number is _NOT_LITERAL:
                return _NOT_LITERAL
            bounds.append(number)
        try:
            return value[slice(*bounds)]
        except (ValueError, TypeError):
            return _NOT_LITERAL
    number = _int_literal(index)
    if number is _NOT_LITERAL:
        return _NOT_LITERAL
    try:
        piece = value[number]
    except IndexError:
        return _NOT_LITERAL
    return piece if isinstance(piece, str) else _NOT_LITERAL


def _literal_fstring(node: ast.JoinedStr, depth: int) -> Any:
    parts: List[str] = []
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            parts.append(part.value)
            continue
        if not isinstance(part, ast.FormattedValue):
            return _NOT_LITERAL
        value = _literal(part.value, depth + 1)
        if value is _REFUSED:
            return _REFUSED
        if not _formattable(value):
            return _NOT_LITERAL
        if part.conversion == ord("r"):
            value = repr(value)
        elif part.conversion == ord("a"):
            value = ascii(value)
        elif part.conversion == ord("s"):
            value = str(value)
        spec = ""
        dynamic = False
        if part.format_spec is not None:
            spec = _literal(part.format_spec, depth + 1)
            if spec is _REFUSED:
                return _REFUSED
            if not isinstance(spec, str):
                return _NOT_LITERAL
            dynamic = any(
                isinstance(piece, ast.FormattedValue)
                for piece in getattr(part.format_spec, "values", ())
            )
        rendered = _guarded_field(value, spec, dynamic)
        if rendered is _REFUSED:
            return _REFUSED
        if not isinstance(rendered, str):
            return _NOT_LITERAL
        parts.append(rendered)
    return "".join(parts)


def _as_texts(value: Any) -> List[str]:
    """The readings of an assembled value that are matched against the tokens.

    A string is read as it is. Bytes are read as latin-1 (every byte one
    character) and, when they decode as such, as UTF-8 too, so a lookalike
    or a zero-width character is caught in whichever reading shows it.
    Integers and other non-text values have no reading.
    """
    if isinstance(value, str):
        return [value]
    if not isinstance(value, bytes):
        return []
    readings = [value.decode("latin-1")]
    try:
        utf8 = value.decode("utf-8")
    except UnicodeDecodeError:
        return readings
    if utf8 != readings[0]:
        readings.append(utf8)
    return readings


_ASSEMBLED_NODES = (ast.BinOp, ast.Call, ast.Constant, ast.JoinedStr, ast.Subscript)


def assembled_hits(
    rel_str: str,
    text: str,
    tokens: Sequence[Tuple[str, re.Pattern]],
    only_lines: Optional[Set[int]] = None,
    reported: Optional[Dict[Tuple[int, str], str]] = None,
) -> List[Hit]:
    """Denylist tokens that appear only once literal pieces are evaluated.

    The text scan sees ``"Pla" + "nted"`` as two harmless pieces, but the
    assembled value is what a reader of the repository ends up with, so every
    literal-only expression is evaluated and matched as well. A node is
    reported at its first line, once per token; it is left out when the line
    scan (the line as written, or its decoded or folded view) already reports
    the same token on a line of the node's span, so a plain literal is never
    reported twice. With ``only_lines`` (patch mode) a node counts only when
    its span holds one of those lines, and only those lines suppress it.
    The assembled value is matched as it is and in its decoded and folded
    views, as a line is. A literal-only format the evaluator refuses to run
    (a width or precision of 10000 or more in a conversion specifier, or a
    dynamic one that is not a number) is an ``encoding`` hit at its first line.
    Python that does not parse is left to the text scan (and to the linters).
    ``reported`` is the line scan's result when the caller already has it.
    """
    if not tokens:
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    if reported is None:
        reported = _line_token_hits(text, tokens)
    hits: List[Hit] = []
    seen: Set[Tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ASSEMBLED_NODES):
            continue
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", None) or start
        span = range(start, end + 1)
        if only_lines is not None:
            span = [n for n in span if n in only_lines]
            if not span:
                continue
        evaluated = _literal(node)
        if evaluated is _REFUSED:
            if (start, UNEVALUATED_FORMAT) not in seen:
                seen.add((start, UNEVALUATED_FORMAT))
                hits.append(Hit(rel_str, start, "encoding", UNEVALUATED_FORMAT))
            continue
        forms = [
            (form, form.lower())
            for value in _as_texts(evaluated)
            for form in [value] + _line_views(value)
        ]
        if not forms:
            continue
        for label, rx in tokens:
            if (start, label) in seen:
                continue
            if not any(_token_in(form, low, rx) for form, low in forms):
                continue
            if any((n, label) in reported for n in span):
                continue
            seen.add((start, label))
            hits.append(Hit(rel_str, start, "assembled-token", label))
    return sorted(hits, key=lambda h: (h.line, h.match))


def _scan_assembled_literals(
    rel_str: str,
    text: str,
    tokens: Sequence[Tuple[str, re.Pattern]],
    only_lines: Optional[Set[int]] = None,
) -> List[Hit]:
    return assembled_hits(rel_str, text, tokens, only_lines=only_lines)


def run_audit(
    paths: Iterable[Path],
    denylist: Sequence[str],
    allowlist: Sequence[AllowRule],
    extra_denylist: Optional[Sequence[str]] = None,
    root: Optional[Path] = None,
    *,
    stats: Optional[AuditStats] = None,
    assemble_python: bool = True,
) -> List[Hit]:
    """Audit files and directories of a tree (see the module docstring).

    ``stats`` collects the scanned count and the skipped paths. Patch mode
    passes ``assemble_python=False`` for its added-line fragments, whose
    literals it assembles in each commit's post-image instead.
    """
    tokens = _compile_tokens(denylist, extra_denylist)
    stats = stats if stats is not None else AuditStats()
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
        if path.is_dir():
            continue
        # Files with no suffix (Makefile, LICENSE, .gitignore) or a text
        # suffix are read fail closed (see _decode); any other file is
        # sniffed: text and archives are scanned, the rest is counted.
        suffix = path.suffix.lower()
        if not suffix or suffix in TEXT_SUFFIXES:
            stats.scanned += 1
            hits.extend(_scan_file(path, str(rel), tokens, assemble_python))
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            hits.append(
                Hit(str(rel), 0, "encoding", f"unreadable ({exc.__class__.__name__})")
            )
            continue
        hits.extend(_scan_blob(data, str(rel), tokens, 0, stats, assemble_python))
    return [h for h in hits if not any(r.covers(h.path, h) for r in allowlist)]


def _scan_file(
    path: Path,
    rel_str: str,
    tokens: Sequence[Tuple[str, re.Pattern]],
    assemble_python: bool = True,
) -> List[Hit]:
    """Every hit in one text file: the encoding check, tokens and patterns."""
    text, problem = _read_text(path)
    return _scan_decoded(
        text, problem, rel_str, tokens, path.suffix.lower() == ".py", assemble_python
    )


def _scan_decoded(
    text: str,
    problem: Optional[str],
    rel_str: str,
    tokens: Sequence[Tuple[str, re.Pattern]],
    is_python: bool,
    assemble_python: bool = True,
) -> List[Hit]:
    hits: List[Hit] = []
    if problem is not None:
        hits.append(Hit(rel_str, 0, "encoding", problem))
    token_hits = _line_token_hits(text, tokens)
    hits.extend(_scan_text(rel_str, text, tokens, is_python, token_hits))
    if is_python and assemble_python:
        hits.extend(assembled_hits(rel_str, text, tokens, reported=token_hits))
    return hits


def _scan_blob(
    data: bytes,
    rel_str: str,
    tokens: Sequence[Tuple[str, re.Pattern]],
    depth: int,
    stats: AuditStats,
    assemble_python: bool = True,
) -> List[Hit]:
    """Scan content of an unknown kind: text, an archive, or a skipped binary."""
    kind = _sniff(data)
    if kind == "text":
        stats.scanned += 1
        text, problem = _decode(data)
        return _scan_decoded(text, problem, rel_str, tokens, False, assemble_python)
    if kind in ("gzip", "zip", "tar"):
        if depth >= MAX_ARCHIVE_DEPTH:
            return [Hit(rel_str, 0, "encoding", "archive nested too deep")]
        return _scan_archive(data, kind, rel_str, tokens, depth, stats)
    stats.skipped.append(rel_str)
    return []


def _archive_members(data: bytes, kind: str, rel_str: str):
    """Yield ``(member name, bytes or None, problem or None)`` for an archive."""
    if kind in ("gzip", "tar"):
        try:
            archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
        except (tarfile.TarError, OSError, EOFError, zlib.error):
            archive = None
        if archive is not None:
            with archive:
                for count, member in enumerate(archive):
                    if count >= MAX_ARCHIVE_MEMBERS:
                        yield "", None, "too many archive members"
                        return
                    if not member.isfile():
                        continue
                    if member.size > MAX_MEMBER_BYTES:
                        yield member.name, None, "archive member too large"
                        continue
                    handle = archive.extractfile(member)
                    yield member.name, handle.read() if handle else b"", None
            return
        if kind == "tar":
            yield "", None, "unreadable archive"
            return
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                payload = stream.read(MAX_MEMBER_BYTES + 1)
        except (OSError, EOFError, zlib.error):
            yield "", None, "unreadable archive"
            return
        name = PurePosixPath(rel_str.rsplit("!", 1)[-1]).name
        inner = name[:-3] if name.lower().endswith(".gz") else name
        if len(payload) > MAX_MEMBER_BYTES:
            yield inner, None, "archive member too large"
            return
        yield inner, payload, None
        return
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, EOFError):
        yield "", None, "unreadable archive"
        return
    with archive:
        for count, info in enumerate(archive.infolist()):
            if count >= MAX_ARCHIVE_MEMBERS:
                yield "", None, "too many archive members"
                return
            if info.is_dir():
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                yield info.filename, None, "archive member too large"
                continue
            try:
                yield info.filename, archive.read(info), None
            except (RuntimeError, zipfile.BadZipFile, OSError, zlib.error):
                yield info.filename, None, "unreadable archive member"


def _scan_archive(
    data: bytes,
    kind: str,
    rel_str: str,
    tokens: Sequence[Tuple[str, re.Pattern]],
    depth: int,
    stats: AuditStats,
) -> List[Hit]:
    """Scan every member of an archive under ``<archive>!<member>``."""
    hits: List[Hit] = []
    for name, payload, problem in _archive_members(data, kind, rel_str):
        member_rel = f"{rel_str}!{name}" if name else rel_str
        if problem is not None:
            hits.append(Hit(member_rel, 0, "encoding", problem))
            continue
        pure = PurePosixPath(name)
        if _is_member_artifact(pure):
            hits.append(Hit(member_rel, 0, "artifact", pure.name))
            continue
        suffix = pure.suffix.lower()
        if not suffix or suffix in TEXT_SUFFIXES:
            stats.scanned += 1
            text, decode_problem = _decode(payload)
            hits.extend(
                _scan_decoded(text, decode_problem, member_rel, tokens, suffix == ".py")
            )
            continue
        hits.extend(_scan_blob(payload, member_rel, tokens, depth + 1, stats))
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
