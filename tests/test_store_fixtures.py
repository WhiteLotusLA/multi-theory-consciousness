"""The throwaway-store fixtures: naming, teardown guards and opt-in skips.

Nothing here opens a connection to any server.
"""

import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests import conftest as fixtures

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_NAMES = (
    "throwaway_postgres",
    "throwaway_mongo",
    "throwaway_qdrant",
    "throwaway_redis",
    "throwaway_neo4j",
)


def test_throwaway_names_carry_the_test_prefix(monkeypatch):
    monkeypatch.delenv(fixtures.PREFIX_VARIABLE, raising=False)
    name = fixtures.throwaway_name()
    assert re.fullmatch(r"mtc_test_[0-9a-f]{8}", name)
    assert fixtures.assert_throwaway(name) == name
    assert fixtures.throwaway_name() != name
    monkeypatch.setenv(fixtures.PREFIX_VARIABLE, "lab_check")
    custom = fixtures.throwaway_name()
    assert re.fullmatch(r"lab_check_[0-9a-f]{8}", custom)
    assert fixtures.assert_throwaway(custom) == custom
    assert fixtures.assert_throwaway_prefix(f"{custom}:") == f"{custom}:"


@pytest.mark.parametrize(
    "bad",
    [
        "postgres",
        "mtc",
        "mtc_consciousness",
        "mtc_test",
        "mtc_test_",
        "mtc_test_zzzzzzzz",
        "mtc_test_0123456",
        "mtc_test_0123456789",
        "xmtc_test_01234567",
        'mtc_test_01234567"; DROP DATABASE x; --',
        "mtc-test-01234567",
    ],
)
def test_teardown_refuses_unprefixed_names(monkeypatch, bad):
    monkeypatch.delenv(fixtures.PREFIX_VARIABLE, raising=False)
    with pytest.raises(AssertionError):
        fixtures.assert_throwaway(bad)
    for prefix in ("mtc:", "mtc_test:", bad, f"{bad}:"):
        with pytest.raises(AssertionError):
            fixtures.assert_throwaway_prefix(prefix)


def test_neo4j_names_use_dashes(monkeypatch):
    monkeypatch.delenv(fixtures.PREFIX_VARIABLE, raising=False)
    name = fixtures.throwaway_name(dashes=True)
    assert re.fullmatch(r"mtc-test-[0-9a-f]{8}", name)
    assert fixtures.assert_throwaway(name, dashes=True) == name
    with pytest.raises(AssertionError):
        fixtures.assert_throwaway(name)  # the underscore form is refused
    with pytest.raises(AssertionError):
        fixtures.assert_throwaway("mtc_test_01234567", dashes=True)


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        return iter(self._rows)


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self, **kwargs):
        return _FakeSession(self._rows)


def test_neo4j_fixture_skips_without_database_creation():
    community = _FakeDriver([{"edition": "community"}])
    assert fixtures.neo4j_can_create_databases(community) is False
    with pytest.raises(pytest.skip.Exception) as info:
        fixtures.require_neo4j_database_creation(community)
    assert str(info.value) == fixtures.NEO4J_COMMUNITY_REASON
    assert "community edition" in fixtures.NEO4J_COMMUNITY_REASON

    enterprise = _FakeDriver([{"edition": "enterprise"}])
    assert fixtures.neo4j_can_create_databases(enterprise) is True
    fixtures.require_neo4j_database_creation(enterprise)

    class _Broken:
        def session(self, **kwargs):
            raise RuntimeError("no server")

    with pytest.raises(pytest.skip.Exception):
        fixtures.require_neo4j_database_creation(_Broken())


@contextmanager
def _without_opt_in(monkeypatch):
    monkeypatch.delenv(fixtures.OPT_IN_VARIABLE, raising=False)
    yield


def test_opt_in_gate_names_the_variable(monkeypatch):
    with _without_opt_in(monkeypatch):
        with pytest.raises(pytest.skip.Exception) as info:
            fixtures.require_opt_in()
    assert fixtures.OPT_IN_VARIABLE in str(info.value)
    monkeypatch.setenv(fixtures.OPT_IN_VARIABLE, "0")
    with pytest.raises(pytest.skip.Exception):
        fixtures.require_opt_in()
    monkeypatch.setenv(fixtures.OPT_IN_VARIABLE, "1")
    fixtures.require_opt_in()


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_probe(name, request):
    """Runs only inside the subprocess below, where the opt-in is unset."""
    if os.environ.get("MTC_FIXTURE_PROBE") != "1":
        pytest.skip("runs in the subprocess of test_fixtures_skip_without_opt_in")
    request.getfixturevalue(name)
    pytest.fail(f"{name} did not skip without the opt-in")


def test_fixtures_skip_without_opt_in():
    env = {k: v for k, v in os.environ.items() if k != fixtures.OPT_IN_VARIABLE}
    env["MTC_FIXTURE_PROBE"] = "1"
    env["PYPHI_WELCOME_OFF"] = "yes"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_store_fixtures.py",
            "-k",
            "test_fixture_probe",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "5 skipped" in out, out
    assert "failed" not in out, out
    # pytest folds identical skip reasons into one "SKIPPED [5]" line
    assert "SKIPPED [5] " in out and fixtures.OPT_IN_REASON in out, out


DB_PROBE_SOURCE = """
import pytest


@pytest.mark.db
def test_store_test_written_without_a_fixture():
    raise AssertionError("a db test ran without the opt-in")
"""


def test_db_tests_skip_at_collection_without_opt_in(tmp_path):
    probe = tmp_path / "test_db_probe.py"
    probe.write_text(DB_PROBE_SOURCE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != fixtures.OPT_IN_VARIABLE}
    env["PYPHI_WELCOME_OFF"] = "yes"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(ROOT / "pyproject.toml"),
            "--rootdir",
            str(ROOT),
            "-p",
            "tests.conftest",
            "-p",
            "no:cacheprovider",
            "-q",
            "-rs",
            str(probe),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "1 skipped" in out, out
    assert "failed" not in out and "error" not in out, out
    assert fixtures.OPT_IN_REASON in out, out
