"""Explicit logger setup: the mtc namespace, formatters, files and helpers."""

import importlib.util
import json
import logging
import os
import subprocess
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mtc.core import logger as mtc_logger
from mtc.core.config import Settings
from mtc.core.logger import (
    JSONFormatter,
    PlainFormatter,
    get_logger,
    log_consciousness_metric,
    log_emotional_state,
    log_safety_event,
    neutralize_pyphi_logging,
    reset_logging,
    setup_logging,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_logging():
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("MTC_") or key == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _flush(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()
    for name in mtc_logger.SPECIALTY_FILES:
        for handler in logging.getLogger(name).handlers:
            handler.flush()


def _records(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_setup_configures_mtc_logger_not_root(clean_env):
    root = logging.getLogger()
    before = list(root.handlers)
    level_before = root.level
    logger = setup_logging(Settings(), log_directory=clean_env / "logs")
    assert logger.name == "mtc"
    assert logger.propagate is False
    assert logger.handlers
    assert root.handlers == before
    assert root.level == level_before
    child = logging.getLogger("mtc.memory.example")
    assert child.parent is not None
    assert child.getEffectiveLevel() == logger.level


def test_setup_is_idempotent(clean_env):
    settings = Settings()
    first = setup_logging(settings, log_directory=clean_env / "logs")
    count = len(first.handlers)
    again = setup_logging(settings, log_directory=clean_env / "logs")
    assert again is first
    assert len(first.handlers) == count
    forced = setup_logging(settings, log_directory=clean_env / "logs", force=True)
    assert len(forced.handlers) == count


def test_import_configures_nothing(tmp_path):
    code = (
        "import logging, mtc.core.logger as m;"
        "lg = logging.getLogger('mtc');"
        "assert lg.handlers == [], lg.handlers;"
        "assert logging.getLogger().handlers == [];"
        "import os; assert not os.path.exists('logs')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYPHI_WELCOME_OFF": "yes"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord(
        "mtc.x", logging.INFO, __file__, 10, "hello %s", ("w",), None
    )
    record.session = "s-1"
    record.score = 0.5
    data = json.loads(JSONFormatter().format(record))
    assert data["message"] == "hello w"
    assert data["session"] == "s-1" and data["score"] == 0.5
    assert data["level"] == "INFO" and data["logger"] == "mtc.x"
    stamp = datetime.fromisoformat(data["timestamp"])
    assert stamp.utcoffset() == timedelta(0)
    for standard in ("args", "msg", "levelno", "created"):
        assert standard not in data


def test_plain_formatter_is_ascii_and_neutral():
    record = logging.LogRecord(
        "mtc.x",
        logging.WARNING,
        __file__,
        10,
        "caf" + chr(0xE9) + " " + chr(0x2192) + " done",
        (),
        None,
    )
    line = PlainFormatter().format(record)
    assert line.isascii()
    assert "WARNING" in line and "[mtc.x]" in line
    assert "\n" not in line


def test_rotation_follows_settings(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_LOG_ROTATION_SIZE_MB", "3")
    monkeypatch.setenv("MTC_LOG_ROTATION_BACKUPS", "2")
    logger = setup_logging(Settings(), log_directory=clean_env / "logs")
    rotating = [h for h in logger.handlers if hasattr(h, "maxBytes")]
    assert rotating and all(h.maxBytes == 3 * 1024 * 1024 for h in rotating)
    assert all(h.backupCount == 2 for h in rotating)


def test_level_and_format_come_from_settings(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("MTC_LOG_FORMAT", "plain")
    logger = setup_logging(Settings(), log_directory=clean_env / "logs")
    assert logger.level == logging.WARNING
    file_handler = next(h for h in logger.handlers if hasattr(h, "maxBytes"))
    assert isinstance(file_handler.formatter, PlainFormatter)
    monkeypatch.setenv("MTC_LOG_FORMAT", "themed")
    with pytest.raises(ValueError):
        setup_logging(Settings(), log_directory=clean_env / "logs", force=True)


def test_pytest_run_never_writes_production_log(clean_env, monkeypatch):
    monkeypatch.setenv("MTC_LOG_DIRECTORY", str(clean_env / "var"))
    logger = setup_logging(Settings())
    logger.info("a line")
    _flush(logger)
    names = sorted(p.name for p in (clean_env / "var").iterdir())
    assert names == ["mtc-test.log"]
    for specialty in mtc_logger.SPECIALTY_FILES:
        assert logging.getLogger(specialty).handlers == []


def test_directory_argument_writes_the_main_log(clean_env):
    logger = setup_logging(Settings(), log_directory=clean_env / "explicit")
    logger.info("a line")
    _flush(logger)
    names = sorted(p.name for p in (clean_env / "explicit").iterdir())
    assert names == [
        "consciousness_metrics.jsonl",
        "emotional_history.jsonl",
        "mtc.log",
        "safety_audit.jsonl",
    ]


def test_specialty_helpers_write_jsonl_with_aware_utc(clean_env):
    directory = clean_env / "logs"
    logger = setup_logging(Settings(), log_directory=directory)
    log_safety_event({"action": "probe", "decision": "allow", "level": "WARNING"})
    log_consciousness_metric("coverage", 0.5, theory="gwt")
    log_emotional_state({"emotion": "calm", "valence": 0.2})
    _flush(logger)
    safety = _records(directory / "safety_audit.jsonl")
    metrics = _records(directory / "consciousness_metrics.jsonl")
    emotion = _records(directory / "emotional_history.jsonl")
    assert safety[0]["decision"] == "allow"
    assert metrics[0]["metric"] == "coverage" and metrics[0]["theory"] == "gwt"
    assert emotion[0]["emotion"] == "calm"
    for record in (safety[0], metrics[0], emotion[0]):
        assert datetime.fromisoformat(record["timestamp"]).utcoffset() == timedelta(0)
    # the specialty records stay out of the main log
    assert "probe" not in (directory / "mtc.log").read_text()


def test_metric_helper_accepts_none_value(clean_env):
    directory = clean_env / "logs"
    logger = setup_logging(Settings(), log_directory=directory)
    log_consciousness_metric("unmeasured", None)
    log_consciousness_metric("broken", float("nan"), nested={"x": float("inf")})
    _flush(logger)
    records = _records(directory / "consciousness_metrics.jsonl")
    assert records[0]["value"] is None
    assert records[1]["value"] is None and records[1]["nested"] == {"x": None}


def test_helpers_are_silent_without_setup(clean_env):
    log_safety_event({"action": "probe"})
    log_consciousness_metric("x", 1.0)
    log_emotional_state({"emotion": "calm"})
    assert not (clean_env / "logs").exists()


def test_get_logger_namespaces_names():
    assert get_logger("memory.thing").name == "mtc.memory.thing"
    assert get_logger("mtc.memory.thing").name == "mtc.memory.thing"
    assert get_logger("mtc").name == "mtc"


def _fake_pyphi_conf(monkeypatch):
    """A module shaped like pyphi.conf: Option objects holding on_change hooks."""
    calls = []

    class Option:
        def __init__(self, on_change):
            self.on_change = on_change

    def reconfigure(obj, opt):
        calls.append("reconfigured")

    class PyphiConfig:
        LOG_FILE = Option(reconfigure)
        LOG_FILE_LEVEL = Option(reconfigure)
        LOG_STDOUT_LEVEL = Option(reconfigure)
        OTHER = Option(reconfigure)

        def __init__(self):
            self.seen = []

        def _callback(self, obj):
            self.seen.append("config-wide")

        @classmethod
        def options(cls):
            return {k: v for k, v in cls.__dict__.items() if isinstance(v, Option)}

    module = types.ModuleType("pyphi.conf")
    module.PyphiConfig = PyphiConfig
    monkeypatch.setitem(sys.modules, "pyphi.conf", module)
    return PyphiConfig, calls


def test_neutralizer_patches_loaded_pyphi_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyphi.conf", None)
    assert neutralize_pyphi_logging() is False

    config_class, calls = _fake_pyphi_conf(monkeypatch)
    assert neutralize_pyphi_logging() is True
    config = config_class()
    for name in ("LOG_FILE", "LOG_FILE_LEVEL", "LOG_STDOUT_LEVEL"):
        getattr(config_class, name).on_change(config, None)
    assert calls == []
    assert config.seen == ["config-wide"] * 3
    # an unrelated option keeps its own hook
    config_class.OTHER.on_change(config, None)
    assert calls == ["reconfigured"]
    # calling it again is harmless
    assert neutralize_pyphi_logging() is True

    bare = types.ModuleType("pyphi.conf")
    monkeypatch.setitem(sys.modules, "pyphi.conf", bare)
    assert neutralize_pyphi_logging() is False


def test_neutralizer_keeps_root_handlers_on_real_pyphi():
    """Against the installed PyPhi: changing a LOG_* option swaps no handler."""
    if importlib.util.find_spec("pyphi") is None:
        pytest.skip("PyPhi is not installed")
    code = (
        "import logging, warnings;"
        "warnings.simplefilter('ignore');"
        "import pyphi;"
        "from mtc.core.logger import neutralize_pyphi_logging;"
        "assert neutralize_pyphi_logging() is True;"
        "root = logging.getLogger();"
        "root.addHandler(logging.NullHandler());"
        "before = list(root.handlers);"
        "pyphi.config.LOG_STDOUT_LEVEL = pyphi.config.LOG_STDOUT_LEVEL;"
        "pyphi.config.LOG_FILE_LEVEL = pyphi.config.LOG_FILE_LEVEL;"
        "pyphi.config.LOG_FILE = pyphi.config.LOG_FILE;"
        "assert list(root.handlers) == before, (before, root.handlers);"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYPHI_WELCOME_OFF": "yes"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


def test_logger_never_imports_pyphi(tmp_path):
    code = (
        "import sys;"
        "from mtc.core.config import Settings;"
        "from mtc.core.logger import setup_logging, neutralize_pyphi_logging;"
        "import pathlib;"
        f"setup_logging(Settings(), log_directory=pathlib.Path({str(tmp_path)!r}));"
        "assert neutralize_pyphi_logging() is False;"
        "assert 'pyphi' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYPHI_WELCOME_OFF": "yes"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
