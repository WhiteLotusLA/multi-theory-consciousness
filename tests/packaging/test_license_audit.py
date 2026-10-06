"""The licence audit against fake distribution metadata: copyleft classes,
strong copyleft outside iit fails, weak or unknown licences need a table row,
the iit run needs the PyPhi GPL note, and the committed table lists every
direct dependency."""

import importlib.util
import re
import sys
import tomllib
from email.message import Message
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "_license_audit", ROOT / "scripts" / "license_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


class Dist:
    def __init__(self, name, requires=(), version="1.0.0", **fields):
        self.metadata = Message()
        self.metadata["Name"] = name
        for key, value in fields.items():
            header = key.replace("_", "-")
            values = value if isinstance(value, list) else [value]
            for item in values:
                self.metadata[header] = item
        self.requires = list(requires)
        self.version = version


def world(**dists):
    table = {audit.normalize(name): dist for name, dist in dists.items()}

    def lookup(name):
        return table.get(audit.normalize(name))

    return lookup


def project(requires, extras):
    dist = Dist("multi-theory-consciousness", requires=requires)
    for extra in extras:
        dist.metadata["Provides-Extra"] = extra
    return dist


PERMISSIVE = dict(License_Expression="MIT")


@pytest.mark.parametrize(
    "fields, expected",
    [
        (dict(License_Expression="MIT"), "permissive"),
        (dict(License_Expression="Apache-2.0 OR BSD-2-Clause"), "permissive"),
        (dict(License_Expression="MPL-2.0"), "weak copyleft"),
        (dict(License_Expression="LGPL-3.0-or-later"), "weak copyleft"),
        (dict(License_Expression="EPL-2.0"), "weak copyleft"),
        (dict(License_Expression="GPL-3.0-or-later"), "strong copyleft"),
        (dict(License_Expression="AGPL-3.0-only"), "strong copyleft"),
        (dict(License_Expression="GPL-2.0-only OR MIT"), "permissive"),
        (
            dict(
                Classifier=[
                    "License :: OSI Approved :: GNU General Public License v3 or "
                    "later (GPLv3+)"
                ]
            ),
            "strong copyleft",
        ),
        (
            dict(Classifier=["License :: OSI Approved :: Mozilla Public License 2.0"]),
            "weak copyleft",
        ),
        (dict(License="MPL-2.0 AND MIT"), "weak copyleft"),
        (dict(License="CDDL-1.0"), "weak copyleft"),
        (dict(License="Proprietary terms"), "unknown"),
        (dict(), "unknown"),
    ],
)
def test_classifies_strong_and_weak_copyleft(fields, expected):
    assert audit.classify(Dist("x", **fields).metadata)[0] == expected


def test_strong_copyleft_in_all_fails():
    lookup = world(
        **{
            "multi-theory-consciousness": project(
                [
                    "numpy ; extra == 'all'",
                    "gplthing ; extra == 'research'",
                    "multi-theory-consciousness[research] ; extra == 'all'",
                ],
                ["all", "research"],
            ),
            "numpy": Dist("numpy", **PERMISSIVE),
            "gplthing": Dist("gplthing", ["deep"], License_Expression="GPL-3.0-only"),
            "deep": Dist("deep", **PERMISSIVE),
        }
    )
    report = audit.audit(["all"], lookup=lookup, table_text="")
    assert report.exit_code == 1
    assert any(f.startswith("gplthing: strong copyleft") for f in report.failures)
    entry = {e.name: e for e in report.entries}["gplthing"]
    assert entry.extras == ["all"]
    # The transitive dependency of the GPL package is walked too.
    assert "deep" in {e.name for e in report.entries}


def test_unknown_license_needs_table_row():
    lookup = world(
        **{
            "multi-theory-consciousness": project(
                ["mystery ; extra == 'dev'", "mplthing ; extra == 'dev'"], ["dev"]
            ),
            "mystery": Dist("mystery"),
            "mplthing": Dist("mplthing", License_Expression="MPL-2.0"),
        }
    )
    report = audit.audit(["dev"], lookup=lookup, table_text="")
    assert report.exit_code == 1
    assert len(report.failures) == 2
    table = (
        "| Package | Licence |\n|---|---|\n| mystery | Custom |\n| MPLThing | MPL |\n"
    )
    assert audit.audit(["dev"], lookup=lookup, table_text=table).failures == []
    missing = world(
        **{"multi-theory-consciousness": project(["absent ; extra == 'dev'"], ["dev"])}
    )
    report = audit.audit(["dev"], lookup=missing, table_text=table)
    assert report.failures == [
        "absent: not installed (needed by dev); install the extras before auditing"
    ]


def test_iit_run_requires_gpl_note():
    lookup = world(
        **{
            "multi-theory-consciousness": project(
                ["pyphi ; extra == 'iit'"], ["iit", "all"]
            ),
            "pyphi": Dist(
                "pyphi",
                Classifier=[
                    "License :: OSI Approved :: GNU General Public License v3 or "
                    "later (GPLv3+)"
                ],
            ),
        }
    )
    report = audit.audit(["iit"], lookup=lookup, table_text="| pyphi | GPL |\n")
    assert report.failures == ["pyphi: the licence table lacks the PyPhi GPL note"]
    note = "## PyPhi\n\nPyPhi is licensed GPLv3 or later.\n"
    assert audit.audit(["iit"], lookup=lookup, table_text=note).failures == []
    # Strong copyleft is allowed only inside the iit closure.
    with pytest.raises(LookupError):
        audit.audit(["iit"], lookup=world(), table_text=note)
    assert audit.main(["--extras", "iit"], lookup=world()) == 2
    with pytest.raises(ValueError):
        audit.audit(["nonexistent"], lookup=lookup, table_text=note)


def test_table_lists_every_direct_dependency():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_data = data["project"]
    direct = set()
    for item in project_data["dependencies"]:
        direct.add(re.split(r"[<>=!~ \[;@]", item, maxsplit=1)[0])
    for items in project_data["optional-dependencies"].values():
        for item in items:
            name = re.split(r"[<>=!~ \[;@]", item, maxsplit=1)[0]
            if name != "multi-theory-consciousness":
                direct.add(name)
    table = (ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    rows = audit.table_names(table)
    missing = sorted(n for n in direct if audit.normalize(n) not in rows)
    assert missing == []
    assert audit.GPL_NOTE.search(table)
    assert not re.search(r"\b\d+\.\d+\.\d+\.\d+\b", table)
    assert "Apache-2.0" in table.split("##")[0]
