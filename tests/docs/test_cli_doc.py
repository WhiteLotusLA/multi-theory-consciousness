"""The README's command table lists exactly the ``mtc`` subcommands and flags.

The leaf commands and their options come from ``mtc.cli.main.build_parser()``;
``mtc assess`` passes its arguments through, so its options come from
``mtc.cli.assess.build_parser()``. A subcommand or a flag that the parser has
and the table does not, or the other way round, fails the test.
"""

import argparse
import re
from pathlib import Path

from mtc.cli import assess
from mtc.cli.main import build_parser

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
FLAG = re.compile(r"--[a-z][a-z-]*")


def _options(parser: argparse.ArgumentParser):
    found = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            continue
        for option in action.option_strings:
            if option.startswith("--") and option != "--help":
                found.add(option)
    return found


def _leaves(parser: argparse.ArgumentParser, prefix=()):
    leaves = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                leaves.update(_leaves(sub, prefix + (name,)))
    if not leaves and prefix:
        leaves[" ".join(prefix)] = parser
    return leaves


def _table_rows():
    text = README.read_text(encoding="utf-8")
    section = text.split("## Command line", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        match = re.fullmatch(r"`mtc ([a-z -]+)`", cells[0]) if cells else None
        if match and len(cells) == 3:
            rows[match.group(1)] = set(FLAG.findall(cells[1]))
    return section, rows


def test_cli_doc_matches_parser():
    parser = build_parser()
    leaves = _leaves(parser)
    expected = {}
    for name, sub in leaves.items():
        expected[name] = _options(assess.build_parser() if name == "assess" else sub)
    section, rows = _table_rows()
    assert set(rows) == set(expected), (
        f"missing from the README: {sorted(set(expected) - set(rows))}; "
        f"not a subcommand: {sorted(set(rows) - set(expected))}"
    )
    for name, options in expected.items():
        assert rows[name] == options, (
            f"mtc {name}: missing {sorted(options - rows[name])}, "
            f"unknown {sorted(rows[name] - options)}"
        )
    for option in _options(parser):
        assert f"`{option}" in section, f"global option {option} not documented"
