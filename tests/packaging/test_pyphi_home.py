"""The PyPhi import happens inside <DATA_DIRECTORY>/pyphi with the shipped
config copied there once, the working directory is restored, nothing lands in
the caller's directory, and the shipped config holds presentation keys only.
A Settings object configured before the first import decides whose
DATA_DIRECTORY that is; once the import has run, configuring moves nothing."""

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mtc.consciousness import pyphi_bridge
from mtc.core import config as config_module
from mtc.core.config import Settings

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "config" / "pyphi_config.yml"
PRESENTATION_KEYS = {
    "WELCOME_OFF",
    "PROGRESS_BARS",
    "LOG_STDOUT_LEVEL",
    "LOG_FILE_LEVEL",
    "LOG_FILE",
    "PARALLEL",
}
STORE_KEYS = (
    "MTC_POSTGRES_ENABLED",
    "MTC_MONGODB_ENABLED",
    "MTC_REDIS_ENABLED",
    "MTC_QDRANT_ENABLED",
    "MTC_NEO4J_ENABLED",
)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


FAKE_PYPHI = """
import os
from pathlib import Path

IMPORT_CWD = os.getcwd()
Path("__pyphi_cache__").mkdir(exist_ok=True)
Path("__pyphi_cache__", "marker").write_text("cache")
Path("pyphi.log").write_text("log")
CONFIG_SEEN = Path("pyphi_config.yml").exists()
"""


@pytest.fixture
def fake_pyphi(tmp_path, monkeypatch):
    packages = tmp_path / "fakes"
    package = packages / "pyphi"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(FAKE_PYPHI, encoding="utf-8")
    (package / "new_big_phi.py").write_text("LOADED = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(packages))
    for name in list(sys.modules):
        if name == "pyphi" or name.startswith("pyphi."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(pyphi_bridge, "_pyphi_module", None)
    monkeypatch.setattr(pyphi_bridge, "_pyphi_checked", False)
    # No recorded Settings from another test, and none left behind.
    monkeypatch.setattr(pyphi_bridge, "_home_settings", None)
    monkeypatch.setattr(pyphi_bridge, "_home_in_use", None)
    settings = Settings(
        _env_file=None,
        PROJECT_ROOT=str(tmp_path / "project"),
        DATA_DIRECTORY="state",
    )
    monkeypatch.setattr(config_module, "_settings", settings)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    yield {"home": tmp_path / "project" / "state" / "pyphi", "work": work}
    for name in list(sys.modules):
        if name == "pyphi" or name.startswith("pyphi."):
            sys.modules.pop(name, None)


def test_import_happens_in_pyphi_home_and_cwd_is_restored(fake_pyphi):
    module = pyphi_bridge._load_pyphi()
    assert module is not None
    assert Path(module.IMPORT_CWD).resolve() == fake_pyphi["home"].resolve()
    assert module.CONFIG_SEEN is True
    assert Path(os.getcwd()).resolve() == fake_pyphi["work"].resolve()
    assert (fake_pyphi["home"] / "__pyphi_cache__" / "marker").is_file()
    assert (fake_pyphi["home"] / "pyphi.log").is_file()
    # Later calls reuse the module and never change directory again.
    assert pyphi_bridge._load_pyphi() is module
    assert pyphi_bridge.pyphi_available() is True


def test_config_copied_once_and_kept(fake_pyphi):
    home = pyphi_bridge.prepare_pyphi_home()
    copy = home / "pyphi_config.yml"
    assert copy.read_text(encoding="utf-8") == SHIPPED.read_text(encoding="utf-8")
    copy.write_text("WELCOME_OFF: true\nPROGRESS_BARS: true\n", encoding="utf-8")
    assert pyphi_bridge.prepare_pyphi_home() == home
    assert "PROGRESS_BARS: true" in copy.read_text(encoding="utf-8")


def test_shipped_pyphi_config_has_only_presentation_keys():
    loaded = yaml.safe_load(SHIPPED.read_text(encoding="utf-8"))
    assert set(loaded) <= PRESENTATION_KEYS
    assert loaded["WELCOME_OFF"] is True
    assert loaded["PROGRESS_BARS"] is False
    assert loaded["LOG_STDOUT_LEVEL"] == "WARNING"
    assert loaded["LOG_FILE_LEVEL"] is None
    assert loaded["PARALLEL"] is False
    spec = importlib.util.find_spec("pyphi")
    if spec is not None and spec.submodule_search_locations:
        conf = Path(list(spec.submodule_search_locations)[0]) / "conf.py"
        source = conf.read_text(encoding="utf-8")
        for key in loaded:
            assert f"    {key} = Option(" in source, key


def test_nothing_created_in_the_original_cwd(fake_pyphi):
    before = sorted(p.name for p in fake_pyphi["work"].iterdir())
    pyphi_bridge._load_pyphi()
    after = sorted(p.name for p in fake_pyphi["work"].iterdir())
    assert before == after == []


def test_home_failure_imports_in_place_and_says_so(fake_pyphi, monkeypatch, caplog):
    def refuse(settings=None):
        raise PermissionError("read-only data directory")

    monkeypatch.setattr(pyphi_bridge, "prepare_pyphi_home", refuse)
    with caplog.at_level("WARNING"):
        module = pyphi_bridge._load_pyphi()
    assert Path(module.IMPORT_CWD).resolve() == fake_pyphi["work"].resolve()
    assert any("PyPhi home" in r.getMessage() for r in caplog.records)
    for leftover in ("__pyphi_cache__", "pyphi.log"):
        target = fake_pyphi["work"] / leftover
        if target.is_dir():
            (target / "marker").unlink()
            target.rmdir()
        else:
            target.unlink()


def test_configured_settings_decide_the_home(fake_pyphi, tmp_path):
    # The process Settings (the fixture's) put the home under
    # <tmp>/project/state; a runtime's own Settings point elsewhere, and a
    # relative DATA_DIRECTORY joins their PROJECT_ROOT, not the process one.
    own = Settings(
        _env_file=None,
        PROJECT_ROOT=str(tmp_path / "elsewhere"),
        DATA_DIRECTORY="runtime-data",
    )
    home = tmp_path / "elsewhere" / "runtime-data" / "pyphi"
    assert pyphi_bridge.configure_pyphi_home(own) == home
    # Recorded only: nothing imported and nothing created yet.
    assert "pyphi" not in sys.modules
    assert not (tmp_path / "elsewhere").exists()

    module = pyphi_bridge._load_pyphi()
    assert module is not None
    assert Path(module.IMPORT_CWD).resolve() == home.resolve()
    assert module.CONFIG_SEEN is True
    copy = home / "pyphi_config.yml"
    assert copy.read_text(encoding="utf-8") == SHIPPED.read_text(encoding="utf-8")
    assert (home / "__pyphi_cache__" / "marker").is_file()
    assert (home / "pyphi.log").is_file()
    # The working directory is restored and empty, and nothing exists under
    # the process Settings' project root.
    assert Path(os.getcwd()).resolve() == fake_pyphi["work"].resolve()
    assert sorted(fake_pyphi["work"].iterdir()) == []
    assert not (tmp_path / "project").exists()


def test_configure_after_the_import_does_not_move_the_home(fake_pyphi, tmp_path):
    first = Settings(
        _env_file=None, PROJECT_ROOT=str(tmp_path / "first"), DATA_DIRECTORY="data"
    )
    later = Settings(
        _env_file=None, PROJECT_ROOT=str(tmp_path / "later"), DATA_DIRECTORY="data"
    )
    home = (tmp_path / "first" / "data" / "pyphi").resolve()
    pyphi_bridge.configure_pyphi_home(first)
    module = pyphi_bridge._load_pyphi()
    assert Path(module.IMPORT_CWD).resolve() == home

    # The first import decided: a later call changes nothing and returns the
    # home in use; nothing is imported again or created for the later Settings.
    in_use = pyphi_bridge.configure_pyphi_home(later)
    assert in_use is not None and in_use.resolve() == home
    assert pyphi_bridge._load_pyphi() is module
    assert Path(module.IMPORT_CWD).resolve() == home
    assert not (tmp_path / "later").exists()
    assert sorted(p.name for p in home.iterdir()) == [
        "__pyphi_cache__",
        "pyphi.log",
        "pyphi_config.yml",
    ]
    assert sorted(fake_pyphi["work"].iterdir()) == []


def test_a_later_configure_before_the_import_replaces_the_record(fake_pyphi, tmp_path):
    own = Settings(
        _env_file=None, PROJECT_ROOT=str(tmp_path / "own"), DATA_DIRECTORY="data"
    )
    assert pyphi_bridge.configure_pyphi_home(own) == tmp_path / "own" / "data" / "pyphi"
    # None records the process Settings again, so the import follows them.
    assert pyphi_bridge.configure_pyphi_home(None) == fake_pyphi["home"]
    module = pyphi_bridge._load_pyphi()
    assert Path(module.IMPORT_CWD).resolve() == fake_pyphi["home"].resolve()
    assert not (tmp_path / "own").exists()


@pytest.mark.slow
def test_real_pyphi_import_leaves_cwd_clean(tmp_path):
    pytest.importorskip("pyphi")
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(ROOT),
        "PYPHI_WELCOME_OFF": "yes",
        "MTC_DEVICE": "cpu",
        "MTC_LLM_PROVIDER": "mock",
        "MTC_PROJECT_ROOT": str(tmp_path / "project"),
        "MTC_DATA_DIRECTORY": "state",
        "MTC_API_PORT": str(_unused_port()),
    }
    env.update({key: "false" for key in STORE_KEYS})
    code = (
        "from mtc.consciousness.pyphi_bridge import pyphi_available\n"
        "import os\n"
        "print(pyphi_available(), os.getcwd())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    available, cwd = result.stdout.strip().splitlines()[-1].split(" ", 1)
    assert available == "True"
    assert Path(cwd).resolve() == work.resolve()
    assert sorted(work.iterdir()) == []
    home = tmp_path / "project" / "state" / "pyphi"
    assert (home / "pyphi_config.yml").is_file()
    assert (home / "__pyphi_cache__").is_dir()
