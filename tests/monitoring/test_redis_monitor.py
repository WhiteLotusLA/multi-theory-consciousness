"""Redis monitor script: modes, configured database, failure exit, ASCII output."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtc.devtools.privacy_audit import load_allowlist, load_denylist, run_audit
from mtc.memory.protocols import StoreUnavailable
from mtc.memory.redis_manager import RedisManager

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ops" / "redis_monitor.py"


def _load():
    spec = importlib.util.spec_from_file_location("redis_monitor_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


monitor = _load()


class FakeClient:
    def __init__(self, keyspace=None):
        self.keyspace = keyspace or {}
        self.sections = []

    async def info(self, section):
        self.sections.append(section)
        if section == "stats":
            return {"total_commands_processed": 4242, "instantaneous_ops_per_sec": 7}
        if section == "keyspace":
            return self.keyspace
        return {}


class FakeManager:
    def __init__(self, metrics=None, db=0, keyspace=None, connect_error=None):
        self.settings = SimpleNamespace(REDIS_DB=db)
        self._client = FakeClient(keyspace)
        self.metrics = metrics or {
            "total_reads": 10,
            "total_writes": 4,
            "cache_hits": 8,
            "cache_misses": 2,
            "errors": 0,
            "redis_memory_mb": 1.5,
            "redis_peak_memory_mb": 2.25,
        }
        self.connect_error = connect_error
        self.disconnected = False

    async def connect(self):
        if self.connect_error is not None:
            raise self.connect_error
        return True

    async def get_metrics(self):
        return dict(self.metrics)

    async def disconnect(self):
        self.disconnected = True


def test_quick_report_renders_metrics(capsys):
    manager = FakeManager()
    assert monitor.main(["quick"], manager=manager) == 0
    out = capsys.readouterr().out
    assert "Redis: reachable" in out
    assert "Memory: 1.50 MB" in out
    assert "Cache hit rate: 80.0% (good)" in out
    assert "Errors: 0" in out
    assert manager.disconnected is True


def test_snapshot_report_renders_counters_and_server_stats(capsys):
    manager = FakeManager(keyspace={"db0": {"keys": 12, "expires": 5}})
    assert monitor.main([], manager=manager) == 0
    out = capsys.readouterr().out
    assert "Total reads: 10" in out
    assert "Peak: 2.25 MB" in out
    assert "Total commands: 4242" in out
    assert "Operations per second: 7" in out
    assert "Keys: 12" in out and "Expires: 5" in out
    assert out.count("report 1") == 1


def test_keyspace_uses_configured_db(capsys):
    keyspace = {"db0": {"keys": 1, "expires": 1}, "db3": {"keys": 99, "expires": 7}}
    manager = FakeManager(db=3, keyspace=keyspace)
    assert monitor.main(["snapshot"], manager=manager) == 0
    out = capsys.readouterr().out
    assert "Keyspace (db3):" in out
    assert "Keys: 99" in out and "Keys: 1\n" not in out
    other = FakeManager(db=5, keyspace=keyspace)
    monitor.main(["snapshot"], manager=other)
    assert "no keys reported" in capsys.readouterr().out


def test_unmeasured_values_read_na_not_zero(capsys):
    metrics = {
        "total_reads": 0,
        "total_writes": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "errors": 0,
        "redis_memory_mb": None,
        "redis_peak_memory_mb": None,
    }
    assert monitor.main(["snapshot"], manager=FakeManager(metrics=metrics)) == 0
    out = capsys.readouterr().out
    assert "Cache hit rate: n/a" in out
    assert "Current: n/a" in out
    assert "Memory: no data" in out


def test_unreachable_redis_exits_nonzero(capsys):
    manager = FakeManager(connect_error=StoreUnavailable("Redis did not answer"))
    assert monitor.main(["quick"], manager=manager) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Redis did not answer" in captured.err


def test_missing_client_package_exits_nonzero_naming_the_extra(capsys):
    error = ImportError("the 'redis' extra is required: pip install mtc[redis]")
    assert monitor.main(["snapshot"], manager=FakeManager(connect_error=error)) == 1
    assert "redis" in capsys.readouterr().err


def test_real_manager_with_dead_client_exits_nonzero(capsys):
    class DeadClient:
        async def ping(self):
            raise ConnectionError("refused")

    manager = RedisManager(client=DeadClient())
    assert monitor.main(["quick"], manager=manager) == 1
    assert "refused" in capsys.readouterr().err


def test_watch_repeats_until_iterations_end(capsys, monkeypatch):
    pauses = []

    async def fake_pause(seconds):
        pauses.append(seconds)

    monkeypatch.setattr(monitor, "_pause", fake_pause)
    manager = FakeManager()
    assert monitor.main(["watch", "--iterations", "3", "--interval", "2"], manager) == 0
    out = capsys.readouterr().out
    assert [f"report {n}" in out for n in (1, 2, 3)] == [True, True, True]
    assert pauses == [2.0, 2.0]


def test_watch_stops_cleanly_on_interrupt(capsys, monkeypatch):
    async def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(monitor, "_pause", interrupt)
    assert monitor.main(["watch"], FakeManager()) == 0
    assert "Monitoring stopped" in capsys.readouterr().out


def test_invalid_arguments_are_rejected():
    with pytest.raises(SystemExit):
        monitor.main(["bogus"], FakeManager())
    with pytest.raises(SystemExit):
        monitor.main(["watch", "--interval", "-1"], FakeManager())


def test_output_is_ascii_and_neutral(capsys, tmp_path):
    manager = FakeManager(keyspace={"db0": {"keys": 3, "expires": 1}})
    monitor.main(["snapshot"], manager=manager)
    monitor.main(["quick"], manager=manager)
    out = capsys.readouterr().out
    out.encode("ascii")
    assert "\x1b" not in out
    source = SCRIPT.read_text(encoding="utf-8")
    source.encode("ascii")
    for word in ("Author", "Session", "Created"):
        assert word not in source and word not in out
    captured = tmp_path / "output.txt"
    captured.write_text(out, encoding="utf-8")
    denylist = load_denylist(ROOT / "config" / "privacy" / "denylist.txt")
    allowlist = load_allowlist(ROOT / "config" / "privacy" / "allowlist.txt")
    assert run_audit([SCRIPT], denylist, allowlist, root=ROOT) == []
    assert run_audit([captured], denylist, allowlist) == []


def test_script_holds_no_endpoint_literal():
    source = SCRIPT.read_text(encoding="utf-8")
    for literal in ("localhost", "127.0.0.1", "6379", "redis://"):
        assert literal not in source
