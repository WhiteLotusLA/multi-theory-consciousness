"""mtc assess on an install without the fep extra: one plain line on standard
error, no traceback, the usage exit code (2); any other ImportError still
propagates."""

import pytest

from mtc.cli import assess
from mtc.cli.missing_extra import PREFIX
from mtc.consciousness.active_inference import PYMDP_MISSING_MESSAGE


def _raise(exc):
    def _build(overrides):
        raise exc

    return _build


def test_assess_without_pymdp_prints_plain_message(monkeypatch, capsys):
    monkeypatch.setattr(
        assess,
        "_build_standalone_modules",
        _raise(ImportError(PYMDP_MISSING_MESSAGE)),
    )
    code = assess.main(["--quick"])
    captured = capsys.readouterr()
    assert code == 2
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert lines[0].startswith(PREFIX)
    assert "fep" in lines[0]
    assert "pip install" in lines[0]
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_assess_without_pymdp_module_itself_prints_plain_message(monkeypatch, capsys):
    """A failed import of pymdp itself (the module name set) is the same case."""
    monkeypatch.setattr(
        assess,
        "_build_standalone_modules",
        _raise(ImportError("No module named 'pymdp'", name="pymdp")),
    )
    assert assess.main(["--quick"]) == 2
    captured = capsys.readouterr()
    assert captured.err.strip().startswith(PREFIX)
    assert "Traceback" not in captured.err


def test_other_import_errors_still_propagate(monkeypatch):
    monkeypatch.setattr(
        assess,
        "_build_standalone_modules",
        _raise(ImportError("No module named 'somethingelse'", name="somethingelse")),
    )
    with pytest.raises(ImportError, match="somethingelse"):
        assess.main(["--quick"])
