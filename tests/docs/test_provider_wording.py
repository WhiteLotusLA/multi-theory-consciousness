"""The documents describe a local model server by what it serves.

A local model server is "a chat-completions server" in every document of the
tree, the historical plans and specs under docs included: no document
describes a server, an endpoint or an evaluator mode as compatible with a
named vendor's interface. The check reads every Markdown file under docs and
the top-level release documents and refuses any hyphenated "-compatible"
adjective other than backward and forward compatibility.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP_LEVEL = ("README.md", "CHANGELOG.md", "VERSION_HISTORY.md", "CONTRIBUTING.md")
ALLOWED_PREFIXES = {"backward", "backwards", "forward"}
COMPATIBLE = re.compile(r"\b([A-Za-z][\w.]*)-compatible\b", re.I)


def _documents() -> list:
    paths = sorted((ROOT / "docs").rglob("*.md"))
    paths += [ROOT / name for name in TOP_LEVEL if (ROOT / name).exists()]
    return paths


def test_the_documents_are_read():
    names = {path.relative_to(ROOT).as_posix() for path in _documents()}
    assert "docs/LLM_SETUP.md" in names
    assert "docs/plans/2026-03-18-v0.2.0-release.md" in names
    assert "README.md" in names


def test_no_document_describes_a_server_as_vendor_compatible():
    hits = []
    for path in _documents():
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            for match in COMPATIBLE.finditer(line):
                if match.group(1).lower() not in ALLOWED_PREFIXES:
                    hits.append(f"{path.relative_to(ROOT)}:{number}: {match.group(0)}")
    assert (
        hits == []
    ), "describe a local model server as a chat-completions server: " + "; ".join(hits)
