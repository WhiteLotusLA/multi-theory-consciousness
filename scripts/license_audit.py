#!/usr/bin/env python3
"""Audit the licences of the installed dependency closure of chosen extras.

Usage: python scripts/license_audit.py [--extras NAME...] [--table PATH] [--json]

The project (multi-theory-consciousness) must be installed. For each extra
(default: all), the installed requirement closure is walked through
importlib.metadata with environment markers evaluated by packaging. Each
distribution's licence is read from License-Expression, else its licence
classifiers, else the License field, and classified as permissive, weak
copyleft (LGPL, MPL, EPL, CDDL), strong copyleft (GPL, AGPL) or unknown. A
dual licence (an OR expression or several licence classifiers) takes its most
permissive option.

Exit codes:
  0  no finding;
  1  strong copyleft in the closure of any extra except iit; a weak-copyleft
     or unknown-licence distribution without a row in the licence table
     (THIRD_PARTY_LICENSES.md by default); a required distribution that is
     not installed (the audit cannot vouch for it); or an iit run that finds
     PyPhi while the table lacks the GPL note;
  2  the project is not installed, or a usage error.

Reads installed package metadata only; writes nothing; sends nothing.
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "multi-theory-consciousness"
DEFAULT_TABLE = ROOT / "THIRD_PARTY_LICENSES.md"
COPYLEFT_EXEMPT_EXTRA = "iit"
GPL_NOTE = re.compile(r"(?is)pyphi.*GPLv3 or later|GPLv3 or later.*pyphi")

PERMISSIVE = "permissive"
WEAK = "weak copyleft"
STRONG = "strong copyleft"
UNKNOWN = "unknown"
_RANK = {PERMISSIVE: 0, WEAK: 1, STRONG: 2, UNKNOWN: 3}

_WEAK_MARKERS = (
    "LGPL",
    "LESSER GENERAL PUBLIC",
    "LIBRARY GENERAL PUBLIC",
    "MPL",
    "MOZILLA PUBLIC",
    "EPL",
    "ECLIPSE PUBLIC",
    "CDDL",
    "COMMON DEVELOPMENT AND DISTRIBUTION",
)
_STRONG_MARKERS = ("AGPL", "AFFERO", "GPL", "GENERAL PUBLIC LICENSE")
_PERMISSIVE_MARKERS = (
    "MIT",
    "BSD",
    "APACHE",
    "ISC",
    "PSF",
    "PYTHON SOFTWARE FOUNDATION",
    "UNLICENSE",
    "ZLIB",
    "HPND",
    "CC0",
    "PUBLIC DOMAIN",
    "0BSD",
    "ZPL",
    "ZOPE PUBLIC",
    "X11",
    "BOOST",
    "BSL-1.0",
    "MIT-CMU",
)


def normalize(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def classify_text(text: str) -> str:
    """Classify one licence name or expression term."""
    upper = str(text).upper()
    if not upper.strip():
        return UNKNOWN
    if any(marker in upper for marker in _WEAK_MARKERS):
        return WEAK
    if any(marker in upper for marker in _STRONG_MARKERS):
        return STRONG
    if any(marker in upper for marker in _PERMISSIVE_MARKERS):
        return PERMISSIVE
    return UNKNOWN


def _best(classes: Iterable[str]) -> str:
    chosen = [c for c in classes if c != UNKNOWN]
    if not chosen:
        return UNKNOWN
    return min(chosen, key=lambda c: _RANK[c])


def classify(metadata) -> Tuple[str, str]:
    """Return ``(class, licence)`` from a distribution's metadata."""
    expression = (metadata.get("License-Expression") or "").strip()
    if expression:
        options = re.split(r"\s+OR\s+", expression.strip("() "))
        return _best(classify_text(option) for option in options), expression
    classifiers = [
        value.split("::")[-1].strip()
        for value in (metadata.get_all("Classifier") or [])
        if value.startswith("License ::") and value.strip() != "License :: OSI Approved"
    ]
    if classifiers:
        return _best(classify_text(c) for c in classifiers), "; ".join(classifiers)
    text = (metadata.get("License") or "").strip()
    first = text.splitlines()[0][:120] if text else ""
    return classify_text(first), first


@dataclass
class Entry:
    """One distribution of the closure."""

    name: str
    version: str
    licence: str
    category: str
    extras: List[str] = field(default_factory=list)
    installed: bool = True


@dataclass
class Report:
    """The audit result."""

    extras: List[str]
    entries: List[Entry]
    failures: List[str]

    @property
    def exit_code(self) -> int:
        return 1 if self.failures else 0


Lookup = Callable[[str], Optional[object]]


def _default_lookup(name: str):
    import importlib.metadata as metadata

    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def _requirements(dist) -> List[str]:
    return list(getattr(dist, "requires", None) or [])


def _wanted(requirement, extras: Sequence[str]) -> bool:
    if requirement.marker is None:
        return True
    for extra in extras or ("",):
        if requirement.marker.evaluate({"extra": extra}):
            return True
    return False


def _top_level(project, extra: str) -> List[object]:
    """The project's requirements for one extra, following self-references."""
    from packaging.requirements import Requirement

    selected = []
    pending = [extra]
    seen_extras: Set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen_extras:
            continue
        seen_extras.add(current)
        for line in _requirements(project):
            requirement = Requirement(line)
            if requirement.marker is None:
                continue  # core dependencies are audited as "core"
            if not requirement.marker.evaluate({"extra": current}):
                continue
            if normalize(requirement.name) == normalize(PROJECT):
                pending.extend(sorted(requirement.extras))
            else:
                selected.append(requirement)
    return selected


def _core(project) -> List[object]:
    from packaging.requirements import Requirement

    core = []
    for line in _requirements(project):
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            if normalize(requirement.name) != normalize(PROJECT):
                core.append(requirement)
    return core


def project_extras(project) -> List[str]:
    names = [str(e) for e in (project.metadata.get_all("Provides-Extra") or [])]
    return sorted(set(names))


def walk(
    requirements: Sequence[object], label: str, lookup: Lookup, found: Dict[str, Entry]
) -> None:
    """Add the installed closure of ``requirements`` to ``found``."""
    from packaging.requirements import Requirement

    queue = [(r, tuple(sorted(r.extras))) for r in requirements]
    visited: Set[Tuple[str, Tuple[str, ...]]] = set()
    while queue:
        requirement, extras = queue.pop(0)
        key = normalize(requirement.name)
        if (key, extras) in visited:
            continue
        visited.add((key, extras))
        dist = lookup(requirement.name)
        if dist is None:
            entry = found.setdefault(
                key,
                Entry(key, "", "not installed", UNKNOWN, installed=False),
            )
            if label not in entry.extras:
                entry.extras.append(label)
            continue
        category, licence = classify(dist.metadata)
        entry = found.setdefault(key, Entry(key, str(dist.version), licence, category))
        if label not in entry.extras:
            entry.extras.append(label)
        for line in _requirements(dist):
            child = Requirement(line)
            if _wanted(child, extras):
                queue.append((child, tuple(sorted(child.extras))))


def table_names(text: str) -> Set[str]:
    """Normalized package names in the first cell of every table row."""
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cell = line.strip("|").split("|")[0].strip().strip("`*")
        if cell and cell.lower() not in ("package", "distribution"):
            names.add(normalize(cell))
    return names


def audit(extras: Sequence[str], *, lookup: Lookup, table_text: str) -> Report:
    """Walk and classify the closure of ``extras``; collect the findings."""
    project = lookup(PROJECT)
    if project is None:
        raise LookupError(f"{PROJECT} is not installed")
    known = project_extras(project)
    chosen = list(extras) or ["all"]
    unknown = [e for e in chosen if e not in known]
    if unknown:
        raise ValueError(
            f"unknown extra {', '.join(unknown)}; known: {', '.join(known)}"
        )
    found: Dict[str, Entry] = {}
    walk(_core(project), "core", lookup, found)
    for extra in chosen:
        walk(_top_level(project, extra), extra, lookup, found)
    rows = table_names(table_text)
    failures = []
    for entry in sorted(found.values(), key=lambda e: e.name):
        if not entry.installed:
            failures.append(
                f"{entry.name}: not installed (needed by {', '.join(entry.extras)}); "
                "install the extras before auditing"
            )
            continue
        if entry.category == STRONG:
            outside = [e for e in entry.extras if e != COPYLEFT_EXEMPT_EXTRA]
            if outside:
                failures.append(
                    f"{entry.name}: strong copyleft ({entry.licence}) in the "
                    f"closure of {', '.join(outside)}"
                )
        if entry.category in (WEAK, UNKNOWN) and entry.name not in rows:
            failures.append(
                f"{entry.name}: {entry.category} licence ({entry.licence or 'none'}) "
                "has no row in the licence table"
            )
        if entry.name == "pyphi" and not GPL_NOTE.search(table_text):
            failures.append("pyphi: the licence table lacks the PyPhi GPL note")
    return Report(extras=chosen, entries=list(found.values()), failures=failures)


def main(argv: Optional[Sequence[str]] = None, lookup: Optional[Lookup] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extras", nargs="+", default=["all"], metavar="NAME")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        table_text = args.table.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"license_audit: cannot read {args.table}: {exc}", file=sys.stderr)
        return 2
    try:
        report = audit(
            args.extras, lookup=lookup or _default_lookup, table_text=table_text
        )
    except LookupError as exc:
        print(f"license_audit: {exc}; install it first", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"license_audit: {exc}", file=sys.stderr)
        return 2
    if args.json:
        payload = {
            "extras": report.extras,
            "distributions": [
                asdict(e) for e in sorted(report.entries, key=lambda e: e.name)
            ],
            "failures": report.failures,
            "exit_code": report.exit_code,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        counts: Dict[str, int] = {}
        for entry in report.entries:
            counts[entry.category] = counts.get(entry.category, 0) + 1
        summary = ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
        print(f"license audit for extras {', '.join(report.extras)}: {summary}")
        for failure in report.failures:
            print(f"FINDING: {failure}")
        if not report.failures:
            print("no findings")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
