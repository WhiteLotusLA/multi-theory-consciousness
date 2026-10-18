"""The one plain line a command prints when the fep extra is not installed."""

from mtc.cli import missing_extra
from mtc.cli.missing_extra import explain_missing_extra


def _module_message() -> str:
    from mtc.consciousness.active_inference import PYMDP_MISSING_MESSAGE

    return PYMDP_MISSING_MESSAGE


def test_the_active_inference_message_is_explained_in_one_line():
    line = explain_missing_extra(ImportError(_module_message()))
    assert line is not None
    assert line.startswith("MISSING EXTRA:")
    assert "\n" not in line
    assert "fep extra" in line
    assert "pip install 'multi-theory-consciousness[fep]'" in line
    assert "Traceback" not in line


def test_a_missing_pymdp_module_is_explained():
    for name in ("pymdp", "pymdp.legacy", "pymdp.legacy.agent"):
        exc = ModuleNotFoundError(f"No module named '{name}'", name=name)
        line = explain_missing_extra(exc)
        assert line is not None, name
        assert line.startswith("MISSING EXTRA:"), name
        assert line == missing_extra.LINE, name
        assert line == explain_missing_extra(ImportError(_module_message())), name


def test_other_import_errors_are_not_explained():
    for exc in (
        ImportError("No module named 'fastapi'", name="fastapi"),
        ModuleNotFoundError("No module named 'pymdpx'", name="pymdpx"),
        ModuleNotFoundError("No module named 'torch'", name="torch"),
        ImportError("cannot import name 'thing' from 'mtc.core'"),
        ImportError(),
    ):
        assert explain_missing_extra(exc) is None, repr(exc)


def test_an_error_that_is_not_an_import_error_is_not_explained():
    assert explain_missing_extra(ValueError(_module_message())) is None


def test_the_line_is_ascii_and_has_no_dash_characters():
    line = explain_missing_extra(ImportError(_module_message()))
    assert line is not None and line.isascii()
    assert chr(0x2014) not in line and chr(0x2013) not in line


def test_the_extra_the_line_names_is_the_one_doctor_blocks_on():
    from mtc.cli import doctor

    assert missing_extra.EXTRA == "fep"
    assert doctor.BLOCKING_EXTRAS == (missing_extra.EXTRA,)
    assert doctor.EXTRA_MODULES[missing_extra.EXTRA] == ("pymdp",)
    line = explain_missing_extra(ImportError(_module_message()))
    assert f"[{missing_extra.EXTRA}]" in str(line)


def test_unbuildable_workspace_is_named_only_when_active_inference_is_absent(
    monkeypatch,
):
    import pytest

    pytest.importorskip("pymdp", reason="the fep extra is not installed")
    from mtc.consciousness import active_inference

    if active_inference.Agent is None or active_inference.pymdp_utils is None:
        pytest.skip("pymdp is installed but active inference could not load it")
    assert missing_extra.workspace_extra_line() is None
    monkeypatch.setattr(active_inference, "Agent", None)
    assert missing_extra.workspace_extra_line() == explain_missing_extra(
        ImportError(_module_message())
    )
    monkeypatch.undo()
    monkeypatch.setattr(active_inference, "pymdp_utils", None)
    line = missing_extra.workspace_extra_line()
    assert line is not None and line.startswith("MISSING EXTRA:")


def test_an_error_naming_another_module_does_not_load_active_inference(monkeypatch):
    """The active inference module is heavy to import: an error that names
    some other module (a missing server library, say) is answered without
    loading it."""
    loaded = []
    real = missing_extra.importlib.import_module

    def watching(name, *args, **kwargs):
        loaded.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(missing_extra.importlib, "import_module", watching)
    exc = ModuleNotFoundError("No module named 'uvicorn'", name="uvicorn")
    assert explain_missing_extra(exc) is None
    assert loaded == []
    # The message-only error, which names no module, still loads it to compare.
    assert explain_missing_extra(ImportError(_module_message())) is not None
    assert loaded == [missing_extra.ACTIVE_INFERENCE_MODULE]
