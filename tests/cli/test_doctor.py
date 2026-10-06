"""``mtc doctor``: nothing reads as healthy unless it was measured."""

import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Dict, List

import pytest

from mtc.cli import doctor
from mtc.cli import main as cli
from mtc.consciousness import pyphi_bridge
from mtc.core import config as config_module
from mtc.core.checks import CheckResult, CheckStatus, exit_code_for, summarize
from mtc.core.config import Settings
from mtc.core.envfile import read_env, upsert_env
from mtc.core.logger import reset_logging
from tests.cli.test_http_client import fake_api, unused_port
from tests.onboarding.test_wizard_noninteractive import (
    ENV_NAME,
    make_project,
    run,
    write_answers,
)

EXPECTED_NAMES = [
    "python",
    "settings",
    "env_file",
    "persona",
    "identity_consistency",
    "speakers",
    "api_key",
    "birth_date",
    "store:postgres",
    "store:mongodb",
    "store:qdrant",
    "store:neo4j",
    "store:redis",
    "brain",
    "provider_state",
    "embedding_model",
    "pyphi",
    "extras",
    "ctm",
    "service_manager",
    "llm_backend",
    "timezone",
]
REQUIRED = {
    "python",
    "settings",
    "api_key",
    "store:postgres",
    "store:mongodb",
    "store:qdrant",
    "store:neo4j",
    "store:redis",
    "brain",
    "timezone",
}


class FakeHub:
    """A stand-in for the model hub module: a cache that may be filled."""

    def __init__(self, cached: bool = False, fail: bool = False) -> None:
        self.cached = cached
        self.fail = fail
        self.calls: List[tuple] = []

    def snapshot_download(self, repo_id, local_files_only=False, **kwargs):
        self.calls.append((repo_id, bool(local_files_only)))
        if local_files_only:
            if not self.cached:
                raise FileNotFoundError("not in the cache")
            return "/cache/" + repo_id
        if self.fail:
            raise ConnectionError("no route")
        self.cached = True
        return "/cache/" + repo_id


@pytest.fixture
def environment(monkeypatch, tmp_path):
    """A wizard-made project, with the optional parts replaced by stand-ins:
    no real PyPhi import, a fake model hub and every extra reported present.
    The Settings each doctor run records in the PyPhi bridge for its first
    import are dropped afterwards, so no other test inherits them."""
    project = make_project(monkeypatch, tmp_path)
    answers = write_answers(tmp_path, api={"port": unused_port()})
    assert run(project, answers)[0] == 0
    monkeypatch.chdir(project)
    monkeypatch.setattr(config_module, "_settings", None)
    for variable in doctor.OFFLINE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    hub = FakeHub(cached=True)
    absent: set = set()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(doctor, "_has_module", lambda name: name not in absent)
    monkeypatch.setattr(pyphi_bridge, "pyphi_available", lambda: False)
    monkeypatch.setattr(pyphi_bridge, "_home_settings", None)
    state = types.SimpleNamespace(
        project=project, hub=hub, absent=absent, tmp=tmp_path, monkeypatch=monkeypatch
    )
    yield state
    config_module._settings = None
    reset_logging()


def settings_for(state, **overrides) -> Settings:
    return Settings(
        _env_file=str(state.project / ENV_NAME),
        PROJECT_ROOT=str(state.project),
        **overrides,
    )


def by_name(results: List[CheckResult]) -> Dict[str, CheckResult]:
    return {result.name: result for result in results}


def fake_probe(monkeypatch, outcomes: Dict[str, CheckResult]) -> List[str]:
    import mtc.db.verify as verify

    probed: List[str] = []

    async def probe_store(store, settings=None, **kwargs):
        probed.append(store)
        return outcomes[store]

    monkeypatch.setattr(verify, "probe_store", probe_store)
    return probed


def test_mock_in_memory_install_passes(environment):
    results = doctor.run_doctor(settings_for(environment))
    assert [result.name for result in results] == EXPECTED_NAMES
    assert [name for name, _required in doctor.CHECKS] == EXPECTED_NAMES
    assert {result.name for result in results if result.required} == REQUIRED - {
        "brain"
    }
    assert exit_code_for(results) == 0
    found = by_name(results)
    for store in ("postgres", "mongodb", "qdrant", "neo4j", "redis"):
        # Disabled is skipped, never ok.
        assert found[f"store:{store}"].status is CheckStatus.SKIPPED
    assert found["brain"].status is CheckStatus.OK
    assert "labelled test text" in found["brain"].detail
    assert found["env_file"].status is CheckStatus.OK
    assert found["persona"].status is CheckStatus.OK
    assert found["persona"].data == {"source": "configured"}
    assert found["identity_consistency"].status is CheckStatus.OK
    assert found["speakers"].status is CheckStatus.OK
    assert found["speakers"].data["by_role"] == {
        "creator": 1,
        "companion": 0,
        "guest": 0,
    }
    assert found["api_key"].status is CheckStatus.OK
    assert found["api_key"].data == {"api_key": "set", "auth_enabled": True}
    # Not bootstrapped yet: the install date is unknown, and says so.
    assert found["birth_date"].status is CheckStatus.INSUFFICIENT_DATA
    assert "bootstrap records it" in found["birth_date"].detail
    assert found["provider_state"].status is CheckStatus.SKIPPED
    assert found["embedding_model"].status is CheckStatus.OK
    assert found["pyphi"].status is CheckStatus.INSUFFICIENT_DATA
    for phrase in ("iit extra", "3.12", "GPL"):
        assert phrase in found["pyphi"].detail
    assert found["extras"].status is CheckStatus.OK
    assert found["service_manager"].detail == "SERVICE_MANAGER=none"
    assert found["llm_backend"].status is CheckStatus.OK
    assert found["timezone"].detail == "TIMEZONE=UTC"
    counts = summarize(results)
    assert counts["failed"] == 0 and counts["unreachable"] == 0
    assert counts["skipped"] >= 6

    # With the install date recorded the check is measured.
    dated = doctor.run_doctor(settings_for(environment, SYSTEM_BIRTH_DATE="2000-01-01"))
    assert by_name(dated)["birth_date"].status is CheckStatus.OK


def test_doctor_reports_unreachable_store_as_unreachable(environment, capsys):
    upsert_env(
        environment.project / ENV_NAME,
        {"MTC_POSTGRES_ENABLED": "true", "MTC_QDRANT_ENABLED": "true"},
    )
    probed = fake_probe(
        environment.monkeypatch,
        {
            "postgres": CheckResult(
                "store:postgres", CheckStatus.UNREACHABLE, "ConnectionRefusedError"
            ),
            "qdrant": CheckResult("store:qdrant", CheckStatus.OK, "answered"),
        },
    )
    results = doctor.run_doctor(settings_for(environment))
    found = by_name(results)
    assert probed == ["postgres", "qdrant"]
    assert found["store:postgres"].status is CheckStatus.UNREACHABLE
    assert found["store:postgres"].required is True
    assert found["store:qdrant"].status is CheckStatus.OK
    assert found["store:redis"].status is CheckStatus.SKIPPED
    assert exit_code_for(results) == 3

    # The command reports the same thing with the same exit code.
    assert cli.main(["doctor"]) == 3
    text = capsys.readouterr().out
    assert re.search(r"\[unreachable\]\s+store:postgres", text)
    assert "unreachable=1" in text

    # A missing client extra is failed and names the extra; a probe that
    # breaks is failed too, never silently fine.
    fake_probe(
        environment.monkeypatch,
        {
            "postgres": CheckResult(
                "store:postgres",
                CheckStatus.FAILED,
                "asyncpg is not installed (install the 'postgres' extra)",
                data={"missing_extra": "postgres"},
            ),
            "qdrant": CheckResult("store:qdrant", CheckStatus.OK, "answered"),
        },
    )
    missing = by_name(doctor.run_doctor(settings_for(environment)))
    assert missing["store:postgres"].status is CheckStatus.FAILED
    assert "postgres" in missing["store:postgres"].detail

    import mtc.db.verify as verify

    async def broken(store, settings=None, **kwargs):
        raise RuntimeError("driver exploded")

    environment.monkeypatch.setattr(verify, "probe_store", broken)
    crashed = doctor.run_doctor(settings_for(environment))
    assert by_name(crashed)["store:postgres"].status is CheckStatus.FAILED
    assert exit_code_for(crashed) == 1


def test_api_key_empty_with_auth_fails(environment, capsys):
    results = doctor.run_doctor(settings_for(environment, API_KEY=""))
    key = by_name(results)["api_key"]
    assert key.status is CheckStatus.FAILED and key.required is True
    assert key.data == {"api_key": "empty", "auth_enabled": True}
    assert "refuse every request" in key.detail
    assert exit_code_for(results) == 1

    # Through the command: the key removed from the environment file.
    upsert_env(environment.project / ENV_NAME, {"MTC_API_KEY": ""})
    assert cli.main(["doctor"]) == 1
    assert re.search(r"\[failed\]\s+api_key", capsys.readouterr().out)

    # Authentication switched off is reported, not blessed.
    off = by_name(
        doctor.run_doctor(settings_for(environment, API_KEY="", API_AUTH_ENABLED=False))
    )
    assert off["api_key"].status is CheckStatus.WARNING
    assert "authentication is off" in off["api_key"].detail


def test_doctor_missing_embeddings_extra_is_insufficient_data(environment):
    environment.absent.add("sentence_transformers")
    results = doctor.run_doctor(settings_for(environment))
    found = by_name(results)
    model = found["embedding_model"]
    assert model.status is CheckStatus.INSUFFICIENT_DATA
    assert model.required is False
    assert "embeddings extra" in model.detail
    assert model.data["missing_extra"] == "embeddings"
    assert environment.hub.calls == []
    # Without an embedder no thought can surface, and the ctm check says so.
    assert found["ctm"].status is CheckStatus.WARNING
    assert found["ctm"].data["predicted_quiet_reason"] == "no_embedder"
    assert exit_code_for(results) == 0
    assert exit_code_for(results, strict=True) == 1

    # A missing neural extra is named and stays advisory: the substrates are
    # left out and the rest runs. No indicator is named from a guess.
    environment.absent.add("torch")
    results = doctor.run_doctor(settings_for(environment))
    extras = by_name(results)["extras"]
    assert extras.status is CheckStatus.INSUFFICIENT_DATA
    assert extras.required is False
    assert extras.data["missing_extras"] == ["neural"]
    assert extras.data["missing_modules"] == {"neural": ["torch"]}
    assert extras.data["blocking_extras"] == []
    assert "neural extra missing" in extras.detail
    assert f"the Needs column of {doctor.MAPPING_DOCUMENT}" in extras.detail
    assert "names this extra in no row" in extras.detail
    assert "no indicator is named here" in extras.detail
    assert "fep extra" not in extras.detail
    assert exit_code_for(results) == 0


def test_missing_fep_extra_is_a_required_failure(environment, capsys):
    """Without pymdp the workspace cannot be built, so nothing runs: doctor
    says so and does not exit 0."""
    environment.absent.add("pymdp")
    results = doctor.run_doctor(settings_for(environment))
    extras = by_name(results)["extras"]
    assert extras.status is CheckStatus.FAILED
    assert extras.required is True
    assert extras.data["missing_extras"] == ["fep"]
    assert extras.data["blocking_extras"] == ["fep"]
    assert "fep extra missing" in extras.detail
    for phrase in (
        "workspace builds it unconditionally",
        "mtc assess",
        "mtc serve",
        "demo",
        "none of the indicators is measured",
    ):
        assert phrase in extras.detail, phrase
    assert "neural extra" not in extras.detail
    assert exit_code_for(results) == 1
    # Every other check of this install is healthy or advisory: the exit code
    # comes from the missing extra alone.
    others = [result for result in results if result.name != "extras"]
    assert exit_code_for(others) == 0

    assert cli.main(["doctor", "--no-network"]) == 1
    text = capsys.readouterr().out
    line = next(row for row in text.splitlines() if " extras" in row)
    assert "[failed]" in line and "(advisory)" not in line

    # Both missing: both are named, the blocking one first.
    environment.absent.add("torch")
    both = by_name(doctor.run_doctor(settings_for(environment)))["extras"]
    assert both.status is CheckStatus.FAILED and both.required is True
    assert both.data["missing_extras"] == ["fep", "neural"]
    assert both.detail.index("fep extra missing") < both.detail.index(
        "neural extra missing"
    )


def test_fep_failure_matches_the_workspace_build(monkeypatch):
    """What doctor says about a missing fep extra is what the code does: the
    workspace both runners and the runtime build raises without pymdp. If
    this stops raising, the extras check has to change with it."""
    from mtc.consciousness import active_inference
    from mtc.core.runner_support import build_workspace

    assert doctor.BLOCKING_EXTRAS == ("fep",)
    assert doctor.EXTRA_MODULES["fep"] == ("pymdp",)
    monkeypatch.setattr(active_inference, "Agent", None)
    monkeypatch.setattr(active_inference, "pymdp_utils", None)
    with pytest.raises(ImportError) as caught:
        build_workspace({}, None)
    assert "pymdp" in str(caught.value)


def _needs_column() -> Dict[str, str]:
    """Indicator name -> its Needs cell, from the committed mapping document."""
    document = Path(__file__).resolve().parents[2] / doctor.MAPPING_DOCUMENT
    needs: Dict[str, str] = {}
    for line in document.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        indicator = re.fullmatch(r"`([a-z_]+)`", cells[0])
        if indicator is not None and len(cells) == 6:
            needs[indicator.group(1)] = cells[5]
    return needs


def test_extras_statements_agree_with_the_mapping_document(environment):
    """What the extras check says about indicators and extras is what the
    committed mapping document holds: its Needs column names the fep extra
    in a row and the neural extra in none. If a row comes to name the neural
    extra, the neural detail has to name that indicator and this fails."""
    needs = _needs_column()
    named: Dict[str, List[str]] = {}
    for indicator, cell in needs.items():
        for extra in re.findall(r"`?([a-z]+)`? extra", cell):
            named.setdefault(extra, []).append(indicator)
    # The document does map an indicator to an extra, so neither the module
    # text nor a detail may say that no committed table does.
    assert named.get("fep"), "no Needs cell names the fep extra any more"
    assert "neural" not in named, (
        f"the Needs column names the neural extra for {named.get('neural')}: "
        "the neural detail in mtc/cli/doctor.py has to name them"
    )
    module_text = " ".join(doctor.__doc__.split())
    assert f"The Needs column of ``{doctor.MAPPING_DOCUMENT}``" in module_text
    assert "names the ``fep`` extra in a row" in module_text
    assert "the ``neural`` extra in none" in module_text
    for text in (module_text, *doctor.EXTRA_EFFECTS.values()):
        assert "no committed table" not in text.lower()

    environment.absent.update({"torch", "pymdp"})
    extras = by_name(doctor.run_doctor(settings_for(environment)))["extras"]
    assert extras.data["missing_extras"] == ["fep", "neural"]
    # Neither detail names a single indicator: fep stops all of them and the
    # document ties none to the neural extra.
    assert len(needs) >= 30
    for indicator in needs:
        assert indicator not in extras.detail, indicator


def test_embedding_model_absent_is_warning_with_fetch_hint(environment):
    environment.hub.cached = False
    result = by_name(doctor.run_doctor(settings_for(environment)))["embedding_model"]
    assert result.status is CheckStatus.WARNING
    assert "--fetch-models" in result.detail
    assert "not in the local model cache" in result.detail
    # Looked up locally only: nothing was downloaded.
    repo = Settings.model_fields["EMBEDDING_MODEL"].default
    assert environment.hub.calls == [(repo, True)]
    # A short model name is looked up under the library's organisation.
    environment.hub.calls.clear()
    doctor.run_doctor(settings_for(environment, EMBEDDING_MODEL="example-model"))
    assert environment.hub.calls == [("sentence-transformers/example-model", True)]


def test_fetch_models_downloads_through_the_hub(environment):
    environment.hub.cached = False
    repo = Settings.model_fields["EMBEDDING_MODEL"].default
    result = by_name(doctor.run_doctor(settings_for(environment), fetch_models=True))[
        "embedding_model"
    ]
    assert result.status is CheckStatus.OK
    assert result.data["fetched"] is True
    assert environment.hub.calls == [(repo, True), (repo, False), (repo, True)]
    # Already cached: nothing is fetched again.
    environment.hub.calls.clear()
    again = by_name(doctor.run_doctor(settings_for(environment), fetch_models=True))
    assert again["embedding_model"].status is CheckStatus.OK
    assert environment.hub.calls == [(repo, True)]
    # A download that fails is a warning, never a pass.
    environment.hub.cached = False
    environment.hub.fail = True
    failed = by_name(doctor.run_doctor(settings_for(environment), fetch_models=True))
    assert failed["embedding_model"].status is CheckStatus.WARNING
    assert "ConnectionError" in failed["embedding_model"].detail


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_fetch_refused_in_offline_mode(environment, variable):
    environment.hub.cached = False
    environment.monkeypatch.setenv(variable, "1")
    repo = Settings.model_fields["EMBEDDING_MODEL"].default
    result = by_name(doctor.run_doctor(settings_for(environment), fetch_models=True))[
        "embedding_model"
    ]
    assert result.status is CheckStatus.WARNING
    assert "refused" in result.detail and variable in result.detail
    assert environment.hub.calls == [(repo, True)]
    # --no-network refuses the download as well.
    environment.monkeypatch.delenv(variable)
    environment.hub.calls.clear()
    offline = by_name(
        doctor.run_doctor(settings_for(environment), network=False, fetch_models=True)
    )["embedding_model"]
    assert offline.status is CheckStatus.WARNING
    assert "--no-network" in offline.detail
    assert environment.hub.calls == [(repo, True)]


def test_no_network_skips_probes_as_insufficient_data(environment):
    import mtc.db.verify as verify

    async def forbidden(store, settings=None, **kwargs):
        raise AssertionError("a store was probed with the network off")

    environment.monkeypatch.setattr(verify, "probe_store", forbidden)
    cfg = settings_for(
        environment, POSTGRES_ENABLED=True, REDIS_ENABLED=True, LLM_PROVIDER="local"
    )
    results = doctor.run_doctor(cfg, network=False)
    found = by_name(results)
    for name in ("store:postgres", "store:redis", "brain"):
        assert found[name].status is CheckStatus.INSUFFICIENT_DATA, name
        assert "--no-network" in found[name].detail
        assert found[name].required is True
    assert found["store:mongodb"].status is CheckStatus.SKIPPED
    # Not measured is not a failure by default, and is one under --strict.
    assert exit_code_for(results) == 0
    assert exit_code_for(results, strict=True) == 1


def test_identity_consistency_warnings(environment):
    aligned = by_name(doctor.run_doctor(settings_for(environment)))
    assert aligned["identity_consistency"].status is CheckStatus.OK

    drifted = by_name(
        doctor.run_doctor(
            settings_for(
                environment,
                SYSTEM_NAME="Otheragent",
                CREATOR_NAME="Operator",
                COMPANION_TYPE="mentor",
                CORE_VALUES="thrift",
            )
        )
    )["identity_consistency"]
    assert drifted.status is CheckStatus.WARNING
    assert drifted.required is False
    assert drifted.data["differing"] == ["name", "creator", "type", "core_values"]
    # Which fields differ is said; the values are not.
    for value in ("Otheragent", "Operator", "Alice", "Tessera", "thrift"):
        assert value not in drifted.detail

    one = by_name(doctor.run_doctor(settings_for(environment, COMPANION_TYPE="child")))
    assert one["identity_consistency"].data["differing"] == ["type"]

    (environment.project / "config" / "persona.yaml").unlink()
    absent = by_name(doctor.run_doctor(settings_for(environment)))
    assert absent["identity_consistency"].status is CheckStatus.SKIPPED
    assert absent["persona"].status is CheckStatus.WARNING
    assert absent["persona"].data["source"] != "configured"

    (environment.project / "config" / "persona.yaml").write_text("- not a mapping\n")
    broken = by_name(doctor.run_doctor(settings_for(environment)))
    assert broken["persona"].status is CheckStatus.FAILED
    assert broken["identity_consistency"].status is CheckStatus.INSUFFICIENT_DATA


def test_doctor_never_prints_secrets_or_speaker_names(environment, capsys):
    env_path = environment.project / ENV_NAME
    upsert_env(
        env_path,
        {
            "MTC_POSTGRES_ENABLED": "true",
            "MTC_POSTGRES_PASSWORD": "store-password-value-1",
            "MTC_LLM_PROVIDER": "anthropic",
            "MTC_ANTHROPIC_MODEL": "example-model",
            "ANTHROPIC_API_KEY": "anthropic-key-value-1",
        },
    )
    fake_probe(
        environment.monkeypatch,
        {"postgres": CheckResult("store:postgres", CheckStatus.OK, "answered")},
    )
    env = read_env(env_path)
    secrets_held = [
        env["MTC_API_KEY"],
        env["MTC_SPEAKER_TOKENS"].split(":", 1)[1],
        "store-password-value-1",
        "anthropic-key-value-1",
    ]
    names = ["Alice", "Tessera"]

    assert cli.main(["doctor"]) == 0
    text = capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 0
    as_json = capsys.readouterr()
    shown = text.out + text.err + as_json.out + as_json.err
    for value in secrets_held:
        assert (value in shown) is False
    for name in names:
        assert name not in shown
    assert re.search(r"[0-9a-f]{32,}", shown) is None
    # What is said about a secret is whether it is set.
    payload = json.loads(as_json.out)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["api_key"]["data"]["api_key"] == "set"
    assert checks["brain"]["data"] == {
        "provider": "anthropic",
        "api_key": "set",
        "model": "set",
    }
    # Configured but never exercised: unverified, because no paid call is made.
    assert checks["brain"]["status"] == "insufficient_data"
    assert "unverified" in checks["brain"]["detail"]
    assert checks["speakers"]["data"]["entries"] == 1


def test_strict_mode(environment, capsys):
    # The mock install has advisory gaps (no install date, no PyPhi).
    assert cli.main(["doctor", "--no-network"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--no-network", "--strict"]) == 1
    text = capsys.readouterr().out
    assert "insufficient_data=" in text and "(advisory)" in text
    results = doctor.run_doctor(settings_for(environment))
    advisory_gaps = [
        result.name
        for result in results
        if not result.required
        and result.status in (CheckStatus.WARNING, CheckStatus.INSUFFICIENT_DATA)
    ]
    assert "birth_date" in advisory_gaps and "pyphi" in advisory_gaps
    assert exit_code_for(results) == 0
    assert exit_code_for(results, strict=True) == 1


def test_json_output_shape(environment, capsys):
    assert cli.main(["doctor", "--json", "--no-network"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert sorted(payload) == ["checks", "exit_code", "summary", "version"]
    assert payload["exit_code"] == 0
    assert [check["name"] for check in payload["checks"]] == EXPECTED_NAMES
    for check in payload["checks"]:
        assert sorted(check) == ["data", "detail", "name", "required", "status"]
        assert check["status"] in {status.value for status in CheckStatus}
        assert check["required"] is (check["name"] in REQUIRED - {"brain"})
    summary = payload["summary"]
    assert summary["total"] == len(EXPECTED_NAMES)
    assert sum(summary[status.value] for status in CheckStatus) == summary["total"]

    # Settings that do not load are a result, with every other check unmeasured.
    upsert_env(environment.project / ENV_NAME, {"MTC_WAKE_HOUR": "99"})
    assert cli.main(["doctor", "--json"]) == 1
    broken = json.loads(capsys.readouterr().out)
    statuses = {check["name"]: check["status"] for check in broken["checks"]}
    assert [check["name"] for check in broken["checks"]] == EXPECTED_NAMES
    assert statuses["settings"] == "failed"
    assert statuses["python"] == "ok"
    assert set(statuses.values()) == {"ok", "failed", "insufficient_data"}
    settings_check = broken["checks"][1]
    assert "WAKE_HOUR" in settings_check["detail"]
    assert "99" not in settings_check["detail"]


def test_provider_state_mismatch_warning(environment):
    state_path = environment.project / "data" / "provider_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"active_provider": "local"}), encoding="utf-8")
    mismatch = by_name(doctor.run_doctor(settings_for(environment)))["provider_state"]
    assert mismatch.status is CheckStatus.WARNING
    assert mismatch.data == {"saved": "local", "configured": "mock"}
    assert "LLM_PROVIDER" in mismatch.detail

    state_path.write_text(json.dumps({"active_provider": "mock"}), encoding="utf-8")
    match = by_name(doctor.run_doctor(settings_for(environment)))["provider_state"]
    assert match.status is CheckStatus.OK

    state_path.write_text("{ not json", encoding="utf-8")
    unreadable = by_name(doctor.run_doctor(settings_for(environment)))
    assert unreadable["provider_state"].status is CheckStatus.WARNING
    state_path.write_text(json.dumps({"saved_at": "x"}), encoding="utf-8")
    empty = by_name(doctor.run_doctor(settings_for(environment)))
    assert empty["provider_state"].status is CheckStatus.SKIPPED


def test_pyphi_check_uses_the_prepared_home(environment):
    # Undo the fixture's stand-in: this test drives the bridge's own loader
    # with a fake PyPhi import, so nothing real is imported.
    environment.monkeypatch.undo()
    monkeypatch = pytest.MonkeyPatch()
    try:
        for key in list(os.environ):
            if key.startswith("MTC_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("MTC_DEVICE", "cpu")
        work = environment.tmp / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        monkeypatch.setattr(config_module, "_settings", None)
        monkeypatch.delitem(sys.modules, "pyphi", raising=False)
        monkeypatch.setattr(pyphi_bridge, "_pyphi_checked", False)
        monkeypatch.setattr(pyphi_bridge, "_pyphi_module", None)
        imported_in: List[Path] = []
        fake = types.ModuleType("pyphi")
        fake.__version__ = "0.0.test"

        def fake_import():
            # Where PyPhi would write its cache and its log.
            imported_in.append(Path.cwd().resolve())
            (Path.cwd() / "__pyphi_cache__").mkdir()
            return fake

        monkeypatch.setattr(pyphi_bridge, "_import_pyphi", fake_import)

        result = doctor._check_pyphi()
        assert result.status is CheckStatus.OK
        assert result.required is False
        assert result.data == {"version": "0.0.test"}
        # The import ran once, inside the prepared home under the data
        # directory, where the shipped presentation config was copied ...
        home = (work / "data" / "pyphi").resolve()
        assert imported_in == [home]
        assert (home / "pyphi_config.yml").is_file()
        assert (home / "__pyphi_cache__").is_dir()
        assert pyphi_bridge.pyphi_home().resolve() == home
        # ... and the working directory is where it was, with nothing of
        # PyPhi's in it.
        assert Path.cwd().resolve() == work.resolve()
        assert sorted(path.name for path in work.iterdir()) == ["data"]
        # Asking again imports nothing more.
        assert doctor._check_pyphi().status is CheckStatus.OK
        assert len(imported_in) == 1

        # An import that fails reads insufficient_data, with the reason.
        monkeypatch.setattr(pyphi_bridge, "_pyphi_checked", False)
        monkeypatch.setattr(pyphi_bridge, "_pyphi_module", None)

        def broken_import():
            raise ImportError("No module named 'pyphi'")

        monkeypatch.setattr(pyphi_bridge, "_import_pyphi", broken_import)
        absent = doctor._check_pyphi()
        assert absent.status is CheckStatus.INSUFFICIENT_DATA
        assert absent.data == {"missing_extra": "iit"}
        assert Path.cwd().resolve() == work.resolve()
    finally:
        monkeypatch.undo()
        pyphi_bridge._pyphi_checked = False
        pyphi_bridge._pyphi_module = None
        config_module._settings = None


def test_pyphi_home_follows_the_checked_settings(environment):
    """``run_doctor(settings)`` records that Settings object in the PyPhi
    bridge, so the first import lands under its DATA_DIRECTORY and not under
    the process Settings' data directory."""
    monkeypatch = environment.monkeypatch
    monkeypatch.delitem(sys.modules, "pyphi", raising=False)
    monkeypatch.setattr(pyphi_bridge, "_pyphi_checked", False)
    monkeypatch.setattr(pyphi_bridge, "_pyphi_module", None)
    monkeypatch.setattr(pyphi_bridge, "_home_in_use", None)
    imported_in: List[Path] = []
    fake = types.ModuleType("pyphi")

    def fake_import():
        imported_in.append(Path.cwd().resolve())
        return fake

    monkeypatch.setattr(pyphi_bridge, "_import_pyphi", fake_import)
    process_home = pyphi_bridge.pyphi_home().resolve()
    checked = environment.tmp / "checked"
    cfg = settings_for(environment, DATA_DIRECTORY=str(checked / "data"))
    home = (checked / "data" / "pyphi").resolve()
    assert home != process_home

    # The fixture's stand-in for pyphi_available keeps the check from
    # importing; the run records the Settings and creates nothing.
    doctor.run_doctor(cfg, network=False)
    assert imported_in == []
    assert not home.exists()

    assert pyphi_bridge.require_pyphi() is fake
    assert imported_in == [home]
    assert (home / "pyphi_config.yml").is_file()
    assert not process_home.exists()
    assert Path.cwd().resolve() == environment.project.resolve()


def test_env_file_mode_warning(environment):
    env_path = environment.project / ENV_NAME
    assert by_name(doctor.run_doctor(settings_for(environment)))["env_file"].data == {
        "mode": "0600"
    }
    env_path.chmod(0o644)
    loose = by_name(doctor.run_doctor(settings_for(environment)))["env_file"]
    assert loose.status is CheckStatus.WARNING
    assert loose.required is False
    assert "0644" in loose.detail and "0600" in loose.detail
    env_path.chmod(0o600)
    cfg = settings_for(environment)
    env_path.unlink()
    gone = by_name(doctor.run_doctor(cfg))["env_file"]
    assert gone.status is CheckStatus.WARNING
    assert "mtc init" in gone.detail


def test_local_brain_is_probed_on_its_health_paths(environment):
    pytest.importorskip("httpx", reason="the llm extra (httpx) is not installed")
    with fake_api() as server:
        server.reply("GET", "/health", 200, {"status": "ok"})
        server.reply("GET", "/v1/models", 200, {"data": [{"id": "example-model"}]})
        cfg = settings_for(
            environment,
            LLM_PROVIDER="local",
            LLM_HOST="127.0.0.1",
            LLM_PORT=server.port,
            LLM_MODEL="example-model",
        )
        healthy = by_name(doctor.run_doctor(cfg))["brain"]
        assert healthy.status is CheckStatus.OK
        assert healthy.data["served_models"] == ["example-model"]
        other = by_name(
            doctor.run_doctor(cfg.model_copy(update={"LLM_MODEL": "another-model"}))
        )["brain"]
        assert other.status is CheckStatus.WARNING
        assert "does not include LLM_MODEL" in other.detail
        # A server that answers without being healthy is failed, not down.
        server.reply("GET", "/health", 503, {"status": "loading"})
        server.reply("GET", "/v1/models", 503, {})
        loading = doctor.run_doctor(cfg)
        assert by_name(loading)["brain"].status is CheckStatus.FAILED
        assert exit_code_for(loading) == 1
    # Nothing listening: unreachable, exit 3.
    down = doctor.run_doctor(cfg)
    assert by_name(down)["brain"].status is CheckStatus.UNREACHABLE
    assert by_name(down)["brain"].required is True
    assert exit_code_for(down) == 3
    # Without the HTTP client extra the check names it.
    environment.absent.add("httpx")
    missing = by_name(doctor.run_doctor(cfg))["brain"]
    assert missing.status is CheckStatus.FAILED
    assert missing.data["missing_extra"] == "llm"


def test_persona_and_speakers_load_errors_quote_no_file_content(environment):
    """A loader's message can quote the offending line of the persona or the
    speakers file, which holds real names: doctor reports the exception type,
    the line and the documented field it names, never the message itself."""
    (environment.project / "config" / "persona.yaml").write_text(
        "name: Quotedname: Hiddenpart\n", encoding="utf-8"
    )
    from mtc.pipeline.speaker_trust import SpeakerTrust

    def leaky(cls, settings=None):
        raise ValueError("speakers[0] role is wrong near: name: Quotedname Hiddenpart")

    environment.monkeypatch.setattr(SpeakerTrust, "from_settings", classmethod(leaky))
    found = by_name(doctor.run_doctor(settings_for(environment), network=False))
    persona, speakers = found["persona"], found["speakers"]
    assert persona.status is CheckStatus.FAILED
    assert speakers.status is CheckStatus.FAILED
    for result in (persona, speakers):
        assert "Quotedname" not in result.detail and "Hiddenpart" not in result.detail
        assert "ValueError" in result.detail
    assert "line 1" in persona.detail
    assert "fields named: name, role" in speakers.detail
