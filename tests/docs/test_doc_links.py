"""Every relative link in the user-facing documents points at a file that exists."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = sorted(
    [ROOT / name for name in ("README.md", "CHANGELOG.md", "VERSION_HISTORY.md")]
    + [ROOT / "CONTRIBUTING.md"]
    + list((ROOT / "docs").glob("*.md"))
)
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
EXTERNAL = ("http://", "https://", "mailto:", "#")


def _relative_links(path: Path):
    text = path.read_text(encoding="utf-8")
    # Code blocks hold examples, not links.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    for target in LINK.findall(text):
        if target.startswith(EXTERNAL):
            continue
        yield target.split("#", 1)[0]


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(path):
    broken = [
        target
        for target in _relative_links(path)
        if target and not (path.parent / target).exists()
    ]
    assert broken == [], f"{path.relative_to(ROOT)}: {broken}"


def test_documents_have_links():
    total = sum(len(list(_relative_links(path))) for path in DOCUMENTS)
    assert total > 0
