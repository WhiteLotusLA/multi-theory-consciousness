"""
Doctor
======

``mtc doctor``: one honest look at an install. Every check reports with the
shared vocabulary of ``mtc.core.checks`` and nothing reads as healthy unless
it was measured: a disabled store is ``skipped``, a store or model server
that refuses the connection is ``unreachable``, a missing optional extra and
a probe that was not allowed to run are ``insufficient_data`` with the
reason, an extra the install cannot run without is ``failed``, and an empty
API key with authentication on is ``failed``.

Checks, in order, with whether each is required (a required ``failed`` exits
1, a required ``unreachable`` exits 3; advisory checks only count under
``--strict``):

- ``python`` (required): Python 3.11 or later.
- ``settings`` (required): the Settings load. When they do not, every later
  check is ``insufficient_data``.
- ``env_file`` (advisory): the project's environment file exists with mode
  0600.
- ``persona`` (advisory): where the prompt's persona comes from.
- ``identity_consistency`` (advisory): ``SYSTEM_NAME``, ``CREATOR_NAME``,
  ``COMPANION_TYPE`` and ``CORE_VALUES`` against the persona file.
- ``speakers`` (advisory): configured speakers counted by role, never named.
- ``api_key`` (required): ``API_KEY`` is set while ``API_AUTH_ENABLED``.
- ``birth_date`` (advisory): ``SYSTEM_BIRTH_DATE``; the bootstrap records it.
- ``store:<name>`` (required), one per store, through
  ``mtc.db.verify.probe_store``.
- ``brain`` (required for ``local`` and ``anthropic``): the local server's
  health paths and served model ids; for ``anthropic`` that the key and the
  model are set, unverified because no paid request is made; ``mock`` is ok.
- ``provider_state`` (advisory): ``PROVIDER_STATE_PATH`` against
  ``LLM_PROVIDER``.
- ``embedding_model`` (advisory): ``EMBEDDING_MODEL`` in the local model
  cache, looked up without network: a snapshot with its configuration and
  weights the embedder loads, which is what a fetch and the embedder's own
  first use both leave. ``fetch_models`` downloads it: one set of
  weights, the tokenizer, the configuration and the repository's small
  files (about 90 MB for the default model), never the exports of the same
  weights for other runtimes that the model's repository also holds. The
  result counts the files fetched and the files left on the hub, and does
  not claim that the embedder reads every fetched file.
- ``pyphi`` (advisory): PyPhi through the bridge's lazy loader, with its home
  under the checked Settings' ``DATA_DIRECTORY``
  (``pyphi_bridge.configure_pyphi_home``; the first import in a process
  decides).
- ``extras`` (advisory for ``neural``, required for ``fep``): a missing
  ``neural`` extra is ``insufficient_data``, because the substrates it
  installs are left out and the rest runs. A missing ``fep`` extra is a
  required ``failed``: the workspace builds the active inference module
  unconditionally and that module raises without pymdp, so ``mtc assess``,
  ``mtc serve`` and the demo stop at start (each with the one
  ``MISSING EXTRA`` line of ``mtc.cli.missing_extra`` and exit code 2) and
  no indicator is measured. The Needs column of
  ``docs/BUTLIN_2025_MAPPING.md`` says what each indicator reads: it names
  the ``fep`` extra in a row and active inference in others, and the
  ``neural`` extra in none. The ``fep`` detail therefore names the
  indicators that read active inference (``FEP_INDICATORS``), the ``neural``
  detail points at that column and names no indicator, and a test compares
  both statements with the document.
- ``ctm`` (advisory): continuous thought's config, seed and persona sources
  and the quiet reason they predict.
- ``service_manager`` and ``llm_backend`` (advisory): reported as set.
- ``timezone`` (required): ``TIMEZONE`` is an IANA zone.

A secret is reported as ``set`` or ``empty`` and never shown; speaker and
creator names are never printed, and a persona or speakers file that does not
load is reported by the exception type, the line and the documented fields
the message names, never by the message (a parser's message can quote the
file, which holds names). Files written: ``fetch_models`` downloads the
embedder's files into the model cache, and the ``pyphi`` check, which imports
PyPhi through the bridge when it is installed, creates
``<DATA_DIRECTORY>/pyphi`` of the checked
Settings on the first import (a copy of the shipped PyPhi configuration,
PyPhi's cache and its log), also under ``--no-network``; nothing else is
written. Heavy modules (the prompt assembler, the provider, the store checks)
are imported inside the check that needs them.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import logging
import os
import re
import stat
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

from mtc.core.checks import CheckResult, CheckStatus
from mtc.core.config import Settings, csv_list
from mtc.core.paths import project_root, resolve_path

logger = logging.getLogger(__name__)

MINIMUM_PYTHON = (3, 11)
PYPHI_MINIMUM_PYTHON = (3, 12)
ENV_FILE_NAME = ".env"
MODELS_PATH = "v1/models"
MODELS_TIMEOUT_S = 3.0
OFFLINE_VARIABLES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
EMBEDDING_ORGANISATION = "sentence-transformers"
# What a fetch leaves on the hub. A model repository holds the same weights
# several times: exports for other runtimes, in their own directories or with
# their own suffixes, which the sentence embedder never loads, and often both
# weight formats of the embedder's own framework, of which the loader takes
# safetensors when a module directory has it.
EXPORT_DIRECTORIES = ("onnx", "openvino")
EXPORT_SUFFIXES = (".onnx", ".h5", ".ot", ".msgpack")
SAFETENSORS_WEIGHTS = ("model.safetensors", "model-*-of-*.safetensors")
PICKLED_WEIGHTS = ("pytorch_model.bin", "pytorch_model-*-of-*.bin")
# The files a cached snapshot is looked up by. Newer versions of the hub
# library refuse a snapshot that lacks a file the lookup's patterns select,
# and neither a fetch here nor the embedder's first use downloads the whole
# repository, so the lookup names the configuration only and the weights are
# then looked for on disk.
SNAPSHOT_MARKERS = ("modules.json", "config.json")
STORE_NAMES = ("postgres", "mongodb", "qdrant", "neo4j", "redis")
_STORE_ENABLE_FIELDS = {
    "postgres": "POSTGRES_ENABLED",
    "mongodb": "MONGODB_ENABLED",
    "qdrant": "QDRANT_ENABLED",
    "neo4j": "NEO4J_ENABLED",
    "redis": "REDIS_ENABLED",
}
# The modules each extra installs and what its absence does. The workspace
# builds the active inference module unconditionally and that module raises
# without pymdp, so an install without the fep extra cannot run at all; the
# neural substrates are left out when their modules are absent and the rest
# runs. MAPPING_DOCUMENT is the committed table of what each indicator reads
# (its Needs column); it names the neural extra in no row, so the neural
# effect names no indicator.
MAPPING_DOCUMENT = "docs/BUTLIN_2025_MAPPING.md"
# The indicators whose Needs cell in MAPPING_DOCUMENT names the fep extra or
# active inference, in document order; a test compares this list with the
# document.
FEP_INDICATORS: Sequence[str] = (
    "sparse_smooth_coding",
    "online_reportable_learning",
    "prediction_error_minimization",
    "hierarchical_prediction",
    "temporal_prediction_depth",
    "agent_environment_coupling",
    "complex_agency",
    "agency",
    "fixed_repertoire",
)
EXTRA_MODULES: Dict[str, Sequence[str]] = {
    "neural": ("torch", "snntorch", "reservoirpy"),
    "fep": ("pymdp",),
}
EXTRA_EFFECTS: Dict[str, str] = {
    "neural": "the torch-backed neural substrates (the spiking and liquid "
    "networks) cannot run, so what they feed is unmeasured; the Needs column "
    f"of {MAPPING_DOCUMENT} says what each indicator reads and names this "
    "extra in no row, so no indicator is named here and mtc assess lists "
    "each indicator's status",
    "fep": "active inference (pymdp) cannot be built and the workspace "
    "builds it unconditionally, so mtc assess, mtc serve and the full "
    "pipeline demo stop at start with one MISSING EXTRA line and none of the "
    "indicators is measured until the fep extra is installed; the Needs "
    f"column of {MAPPING_DOCUMENT} names this extra or active inference for "
    "the indicators that read it: " + ", ".join(FEP_INDICATORS),
}
# Extras nothing runs without: missing, the check is a required failure.
BLOCKING_EXTRAS: Sequence[str] = ("fep",)

# Every check name in report order, with whether it is required.
CHECKS: Sequence[tuple] = (
    ("python", True),
    ("settings", True),
    ("env_file", False),
    ("persona", False),
    ("identity_consistency", False),
    ("speakers", False),
    ("api_key", True),
    ("birth_date", False),
    *((f"store:{name}", True) for name in STORE_NAMES),
    ("brain", True),
    ("provider_state", False),
    ("embedding_model", False),
    ("pyphi", False),
    ("extras", False),
    ("ctm", False),
    ("service_manager", False),
    ("llm_backend", False),
    ("timezone", True),
)

OK = CheckStatus.OK
WARNING = CheckStatus.WARNING
FAILED = CheckStatus.FAILED
UNREACHABLE = CheckStatus.UNREACHABLE
SKIPPED = CheckStatus.SKIPPED
INSUFFICIENT = CheckStatus.INSUFFICIENT_DATA


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _settings_fields(exc: Exception) -> str:
    """The fields a Settings error names, never their values."""
    errors = getattr(exc, "errors", None)
    names: List[str] = []
    if callable(errors):
        try:
            names = sorted({str(e["loc"][0]) for e in errors() if e.get("loc")})
        except Exception:  # noqa: BLE001 - a description must not fail
            names = []
    return ", ".join(names)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_python() -> CheckResult:
    version = sys.version_info
    shown = f"{version[0]}.{version[1]}.{version[2]}"
    if tuple(version[:2]) < MINIMUM_PYTHON:
        return CheckResult(
            "python",
            FAILED,
            f"Python {shown}; MTC needs {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or later",
        )
    return CheckResult("python", OK, f"Python {shown}", data={"version": shown})


def _check_env_file(cfg: Settings) -> CheckResult:
    path = project_root(cfg) / ENV_FILE_NAME
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return CheckResult(
            "env_file",
            WARNING,
            "no environment file in the project directory; mtc init writes one",
            required=False,
        )
    if mode & 0o077:
        return CheckResult(
            "env_file",
            WARNING,
            f"mode {mode:04o} lets other users read the secrets; set it to 0600",
            required=False,
            data={"mode": f"{mode:04o}"},
        )
    return CheckResult(
        "env_file",
        OK,
        f"present, mode {mode:04o}",
        required=False,
        data={"mode": f"{mode:04o}"},
    )


def _load_error(exc: BaseException, fields: Sequence[str]) -> str:
    """An exception reported without its message: the type, the YAML line when
    the parser gave one, and the documented fields the message names."""
    parts = [type(exc).__name__]
    current: Optional[BaseException] = exc
    while current is not None:
        mark = getattr(current, "problem_mark", None) or getattr(
            current, "context_mark", None
        )
        line = getattr(mark, "line", None)
        if isinstance(line, int):
            parts.append(f"line {line + 1}")
            break
        current = current.__cause__
    message = str(exc)
    named = [f for f in fields if re.search(rf"\b{re.escape(f)}\b", message)]
    if named:
        parts.append("fields named: " + ", ".join(named))
    return "; ".join(parts)


def _check_persona(cfg: Settings) -> CheckResult:
    from mtc.brain.prompt import PromptAssembler
    from mtc.core.persona import PERSONA_KEYS

    try:
        source = PromptAssembler.from_settings(cfg).persona_source
    except ValueError as exc:
        return CheckResult(
            "persona",
            FAILED,
            f"the persona does not load ({_load_error(exc, tuple(PERSONA_KEYS))}); "
            "fix PERSONA_PATH or run mtc init --force",
            required=False,
        )
    data = {"source": source}
    if source == "configured":
        return CheckResult(
            "persona", OK, "source: configured (PERSONA_PATH)", False, data
        )
    return CheckResult(
        "persona",
        WARNING,
        f"source: {source}; no persona is configured, so the neutral text "
        "applies (mtc init writes one)",
        required=False,
        data=data,
    )


def _check_identity(cfg: Settings) -> CheckResult:
    from mtc.core.persona import load_persona

    name = "identity_consistency"
    try:
        persona = load_persona(settings=cfg)
    except ValueError:
        return CheckResult(
            name, INSUFFICIENT, "not compared: the persona does not load", False
        )
    if persona is None:
        return CheckResult(name, SKIPPED, "no persona file to compare", required=False)
    creator = persona.get("creator") if isinstance(persona.get("creator"), dict) else {}
    raw_values = persona.get("core_values")
    if isinstance(raw_values, str):
        persona_values: Optional[List[str]] = csv_list(raw_values)
    elif isinstance(raw_values, (list, tuple)):
        persona_values = [str(item).strip() for item in raw_values]
    else:
        persona_values = None
    pairs = (
        ("name", str(persona.get("name") or "").strip() or None, cfg.SYSTEM_NAME),
        ("creator", str(creator.get("name") or "").strip() or None, cfg.CREATOR_NAME),
        ("type", persona.get("companion_type"), cfg.COMPANION_TYPE),
        ("core_values", persona_values, csv_list(cfg.CORE_VALUES)),
    )
    differing = [
        label for label, mine, theirs in pairs if mine is not None and mine != theirs
    ]
    if differing:
        return CheckResult(
            name,
            WARNING,
            "the persona file and the settings differ on: "
            + ", ".join(differing)
            + " (prompts use the persona; mtc init --force writes both)",
            required=False,
            data={"differing": differing},
        )
    return CheckResult(
        name, OK, "name, creator, type and core values agree", required=False
    )


def _check_speakers(cfg: Settings) -> CheckResult:
    from mtc.pipeline.speaker_trust import _ENTRY_FIELDS, SpeakerTrust

    try:
        status = SpeakerTrust.from_settings(cfg).status()
    except ValueError as exc:
        return CheckResult(
            "speakers",
            FAILED,
            f"the speakers file does not load ({_load_error(exc, _ENTRY_FIELDS)}); "
            "fix SPEAKERS_PATH or run mtc init --force",
            False,
        )
    by_role = dict(status.get("by_role") or {})
    data = {
        "entries": int(status.get("entries", 0)),
        "by_role": by_role,
        "allowlisted": int(status.get("allowlisted", 0)),
    }
    if not data["entries"]:
        return CheckResult(
            "speakers",
            WARNING,
            "no speaker is configured: every speaker gets the default trust",
            required=False,
            data=data,
        )
    counts = ", ".join(f"{count} {role}" for role, count in by_role.items())
    return CheckResult(
        "speakers",
        OK,
        f"{counts}; {data['allowlisted']} on the trusted list",
        required=False,
        data=data,
    )


def _check_api_key(cfg: Settings) -> CheckResult:
    key = "set" if cfg.API_KEY else "empty"
    data = {"api_key": key, "auth_enabled": bool(cfg.API_AUTH_ENABLED)}
    if cfg.API_AUTH_ENABLED and not cfg.API_KEY:
        return CheckResult(
            "api_key",
            FAILED,
            "API_KEY is empty while API_AUTH_ENABLED is true: the server would "
            "refuse every request (mtc init generates a key)",
            data=data,
        )
    if not cfg.API_AUTH_ENABLED:
        return CheckResult(
            "api_key",
            WARNING,
            f"authentication is off (API_AUTH_ENABLED=false); API_KEY is {key}",
            data=data,
        )
    return CheckResult("api_key", OK, "API_KEY is set; authentication is on", data=data)


def _check_birth_date(cfg: Settings) -> CheckResult:
    if cfg.SYSTEM_BIRTH_DATE:
        return CheckResult(
            "birth_date",
            OK,
            f"install date {cfg.SYSTEM_BIRTH_DATE}",
            required=False,
            data={"date": cfg.SYSTEM_BIRTH_DATE},
        )
    return CheckResult(
        "birth_date",
        INSUFFICIENT,
        "SYSTEM_BIRTH_DATE is not set; the bootstrap records it on first init "
        "(mtc db init), and no age is computed until then",
        required=False,
    )


def _check_stores(cfg: Settings, network: bool) -> List[CheckResult]:
    import asyncio

    results: List[CheckResult] = []
    for store in STORE_NAMES:
        name = f"store:{store}"
        if not getattr(cfg, _STORE_ENABLE_FIELDS[store]):
            results.append(
                CheckResult(
                    name, SKIPPED, "disabled; this data stays in process memory"
                )
            )
            continue
        if not network:
            results.append(
                CheckResult(name, INSUFFICIENT, "enabled; not probed (--no-network)")
            )
            continue
        from mtc.db import verify as db_verify

        try:
            result = asyncio.run(db_verify.probe_store(store, cfg))
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            result = CheckResult(
                name, FAILED, f"the probe failed: {type(exc).__name__}"
            )
        result.name = name
        result.required = True
        results.append(result)
    return results


def _served_models(base_url: str) -> Optional[List[str]]:
    """Model ids the local server lists, or None when it does not say."""
    import json
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(
            f"{base_url}/{MODELS_PATH}", timeout=MODELS_TIMEOUT_S
        ) as reply:
            if int(reply.status) != 200:
                return None
            document = json.loads(reply.read(1024 * 1024).decode("utf-8"))
    except (OSError, ValueError):
        return None
    entries = document.get("data") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        return None
    return [
        str(item["id"]) for item in entries if isinstance(item, dict) and "id" in item
    ]


def _check_brain(cfg: Settings, network: bool) -> CheckResult:
    provider = cfg.LLM_PROVIDER
    data: Dict[str, Any] = {"provider": provider}
    if provider == "mock":
        return CheckResult(
            "brain",
            OK,
            "mock provider: every reply is labelled test text; no model is used",
            required=False,
            data=data,
        )
    if provider == "anthropic":
        data.update(
            api_key="set" if cfg.ANTHROPIC_API_KEY else "empty",
            model="set" if cfg.ANTHROPIC_MODEL else "empty",
        )
        missing = [
            key
            for key, value in (
                ("ANTHROPIC_API_KEY", cfg.ANTHROPIC_API_KEY),
                ("ANTHROPIC_MODEL", cfg.ANTHROPIC_MODEL),
            )
            if not value
        ]
        if missing:
            return CheckResult(
                "brain",
                FAILED,
                f"anthropic provider: {', '.join(missing)} not set",
                data=data,
            )
        if not _has_module("anthropic"):
            data["missing_extra"] = "anthropic"
            return CheckResult(
                "brain",
                FAILED,
                "anthropic provider: the anthropic extra is not installed",
                data=data,
            )
        return CheckResult(
            "brain",
            INSUFFICIENT,
            "anthropic provider: the key and the model are set but unverified; "
            "no paid request is made here, so the first message is the check",
            data=data,
        )
    if not network:
        return CheckResult(
            "brain",
            INSUFFICIENT,
            "local provider: not probed (--no-network)",
            data=data,
        )
    if not _has_module("httpx"):
        data["missing_extra"] = "llm"
        return CheckResult(
            "brain",
            FAILED,
            "local provider: the llm extra (httpx) is not installed",
            data=data,
        )
    import asyncio

    from mtc.brain.providers.local_chat import LocalChatCompletionsProvider

    async def probe() -> Any:
        local = LocalChatCompletionsProvider.from_settings(cfg)
        try:
            return await local.health_check()
        finally:
            closer = getattr(local, "aclose", None) or getattr(local, "close", None)
            if callable(closer):
                outcome = closer()
                if hasattr(outcome, "__await__"):
                    await outcome

    health = asyncio.run(probe())
    detail = str(getattr(health, "detail", ""))
    if not getattr(health, "ok", False):
        # A status code in the detail means a server answered, unhealthy.
        if re.search(r":\s*\d{3}\b", detail):
            return CheckResult(
                "brain",
                FAILED,
                f"local provider: the server answered but is not healthy ({detail})",
                data=data,
            )
        return CheckResult(
            "brain",
            UNREACHABLE,
            f"local provider: no model server answered ({detail}); start it "
            "(scripts/ops/start_llm.sh) or wait for the model to load",
            data=data,
        )
    served = _served_models(cfg.LLM_BASE_URL or cfg.LLM_URL)
    data["served_models"] = served
    if served is None:
        return CheckResult(
            "brain",
            OK,
            f"local provider: {detail}; the server does not list its models",
            data=data,
        )
    listing = ", ".join(served) if served else "none"
    if cfg.LLM_MODEL and cfg.LLM_MODEL not in served:
        return CheckResult(
            "brain",
            WARNING,
            f"local provider: {detail}; serving {listing}, which does not "
            "include LLM_MODEL",
            data=data,
        )
    return CheckResult(
        "brain", OK, f"local provider: {detail}; serving {listing}", data=data
    )


def _check_provider_state(cfg: Settings) -> CheckResult:
    import json

    name = "provider_state"
    path = resolve_path(cfg.PROVIDER_STATE_PATH, settings=cfg)
    if not path.is_file():
        return CheckResult(
            name, SKIPPED, "no saved provider choice yet; LLM_PROVIDER applies", False
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        saved = record.get("active_provider") if isinstance(record, dict) else None
    except (OSError, ValueError):
        return CheckResult(
            name, WARNING, "the provider state file is unreadable", required=False
        )
    data = {"saved": saved, "configured": cfg.LLM_PROVIDER}
    if saved is None:
        return CheckResult(
            name, SKIPPED, "the state file names no provider", required=False, data=data
        )
    if saved != cfg.LLM_PROVIDER:
        return CheckResult(
            name,
            WARNING,
            f"the state file records {saved!r} but LLM_PROVIDER is "
            f"{cfg.LLM_PROVIDER!r}; the server follows LLM_PROVIDER at start",
            required=False,
            data=data,
        )
    return CheckResult(
        name, OK, f"the saved choice matches LLM_PROVIDER ({saved})", False, data
    )


def _offline_variable() -> Optional[str]:
    for variable in OFFLINE_VARIABLES:
        if os.environ.get(variable, "").strip().lower() in ("1", "true", "yes", "on"):
            return variable
    return None


def _embedding_repo(model: str) -> str:
    return model if "/" in model else f"{EMBEDDING_ORGANISATION}/{model}"


def _named(path: str, patterns: Sequence[str]) -> bool:
    name = path.rpartition("/")[2]
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _embedding_files(paths: Sequence[str]) -> List[str]:
    """Return the files of a model repository worth fetching for the embedder.

    Left out: every file under an export directory or with an export suffix
    (the same weights for other runtimes), and a module directory's pickled
    weights when that directory also holds safetensors weights, which the
    loader prefers. A directory whose only weights are pickled keeps them.
    Everything else stays: the tokenizer and configuration files are small
    and their names vary by model. The result is therefore a superset of
    what the embedder loads: it also holds the repository's small files that
    the embedder never reads (the readme, the attributes file, training
    scripts and data configurations), so the filter can only fetch too much,
    never too little.
    """

    with_safetensors = {
        path.rpartition("/")[0] for path in paths if _named(path, SAFETENSORS_WEIGHTS)
    }
    kept = []
    for path in paths:
        if any(part in EXPORT_DIRECTORIES for part in path.split("/")[:-1]):
            continue
        if path.lower().endswith(EXPORT_SUFFIXES):
            continue
        if (
            _named(path, PICKLED_WEIGHTS)
            and path.rpartition("/")[0] in with_safetensors
        ):
            continue
        kept.append(path)
    return sorted(kept)


def _literal_pattern(path: str) -> str:
    """Return the pattern that matches exactly ``path`` and nothing else."""
    return re.sub(r"([*?\[])", r"[\1]", path)


def _holds_weights(folder: str) -> bool:
    """Tell whether a snapshot folder holds weights the embedder loads."""
    for directory, subdirectories, names in os.walk(folder):
        subdirectories[:] = [
            entry for entry in subdirectories if entry not in EXPORT_DIRECTORIES
        ]
        for name in names:
            if not _named(name, SAFETENSORS_WEIGHTS + PICKLED_WEIGHTS):
                continue
            # A snapshot entry is a link into the cache's blobs: the blob
            # has to be there too.
            if os.path.exists(os.path.join(directory, name)):
                return True
    return False


def _embedding_cached(hub: Any, repo: str) -> bool:
    """Tell, without network, whether the model cache holds ``repo``.

    Present means a snapshot with its configuration (``SNAPSHOT_MARKERS``)
    and at least one weight file the embedder loads. Any miss, an incomplete
    snapshot included, reads as absent.
    """
    try:
        folder = hub.snapshot_download(
            repo, local_files_only=True, allow_patterns=list(SNAPSHOT_MARKERS)
        )
    except Exception:  # noqa: BLE001 - any miss means it is not cached
        return False
    return _holds_weights(str(folder))


def _check_embedding_model(
    cfg: Settings, network: bool, fetch_models: bool
) -> CheckResult:
    name = "embedding_model"
    model = str(cfg.EMBEDDING_MODEL)
    data: Dict[str, Any] = {"model": model}
    if not _has_module("sentence_transformers"):
        data["missing_extra"] = "embeddings"
        return CheckResult(
            name,
            INSUFFICIENT,
            "the embeddings extra (sentence-transformers) is not installed: "
            "semantic recall and thought novelty are unmeasured without it",
            required=False,
            data=data,
        )
    local = resolve_path(model, settings=cfg)
    if local.is_dir():
        return CheckResult(name, OK, "a local model directory", False, data)
    import huggingface_hub

    repo = _embedding_repo(model)

    if _embedding_cached(huggingface_hub, repo):
        return CheckResult(name, OK, f"{repo} is in the local model cache", False, data)
    if not fetch_models:
        return CheckResult(
            name,
            WARNING,
            f"{repo} is not in the local model cache; the first use would "
            "download it (mtc doctor --fetch-models fetches it now)",
            required=False,
            data=data,
        )
    offline = _offline_variable()
    if offline is not None or not network:
        why = f"{offline} is set" if offline is not None else "--no-network was given"
        return CheckResult(
            name,
            WARNING,
            f"{repo} is not in the local model cache, and the fetch was "
            f"refused: {why}",
            required=False,
            data=data,
        )
    # The repository is listed first, so that the exports for other runtimes
    # and the duplicate weights are not requested: a download without a file
    # filter takes every export in the repository, about ten times the size
    # for the default model.
    try:
        listed = [str(path) for path in huggingface_hub.list_repo_files(repo)]
        wanted = _embedding_files(listed)
        if wanted:
            huggingface_hub.snapshot_download(
                repo, allow_patterns=[_literal_pattern(path) for path in wanted]
            )
    except Exception as exc:  # noqa: BLE001 - reported, with the type only
        return CheckResult(
            name,
            WARNING,
            f"{repo} could not be downloaded ({type(exc).__name__})",
            required=False,
            data=data,
        )
    if not wanted:
        return CheckResult(
            name,
            WARNING,
            f"the hub lists no file of {repo} that the sentence embedder "
            "loads; nothing was downloaded",
            required=False,
            data=data,
        )
    if _embedding_cached(huggingface_hub, repo):
        left_out = len(listed) - len(wanted)
        data["fetched"] = True
        data["fetched_files"] = len(wanted)
        data["left_out_files"] = left_out
        return CheckResult(
            name,
            OK,
            f"{repo} was downloaded to the cache ({len(wanted)} files "
            "fetched: the weights, the tokenizer, the configuration and the "
            f"repository's small files; {left_out} files of other runtimes "
            "and duplicate weights were left on the hub)",
            required=False,
            data=data,
        )
    return CheckResult(
        name,
        WARNING,
        f"{repo} was requested but is still not in the cache",
        required=False,
        data=data,
    )


def _check_pyphi(cfg: Optional[Settings] = None) -> CheckResult:
    from mtc.consciousness import pyphi_bridge

    # The first import puts PyPhi's home under the checked Settings'
    # DATA_DIRECTORY (None: the process Settings).
    pyphi_bridge.configure_pyphi_home(cfg)
    if pyphi_bridge.pyphi_available():
        module = pyphi_bridge.require_pyphi()
        version = str(getattr(module, "__version__", "") or "")
        if not version:
            from importlib import metadata

            try:
                version = metadata.version("pyphi")
            except metadata.PackageNotFoundError:
                version = "unknown"
        return CheckResult(
            "pyphi",
            OK,
            f"PyPhi {version} imports through the bridge",
            required=False,
            data={"version": version},
        )
    floor = f"{PYPHI_MINIMUM_PYTHON[0]}.{PYPHI_MINIMUM_PYTHON[1]}"
    reason = "PyPhi does not import"
    if tuple(sys.version_info[:2]) < PYPHI_MINIMUM_PYTHON:
        reason += f" (this is Python {sys.version_info[0]}.{sys.version_info[1]})"
    return CheckResult(
        "pyphi",
        INSUFFICIENT,
        f"{reason}: exact IIT Phi is unmeasured. It comes with the iit extra, "
        f"which needs Python {floor} or later; PyPhi is licensed under the "
        "GPL (version 3 or later) and is never bundled",
        required=False,
        data={"missing_extra": "iit"},
    )


def _check_extras() -> CheckResult:
    missing: Dict[str, List[str]] = {}
    for extra, modules in EXTRA_MODULES.items():
        absent = [module for module in modules if not _has_module(module)]
        if absent:
            missing[extra] = absent
    if not missing:
        return CheckResult(
            "extras", OK, "the neural and fep extras are installed", required=False
        )
    blocking = sorted(extra for extra in missing if extra in BLOCKING_EXTRAS)
    ordered = blocking + [extra for extra in missing if extra not in blocking]
    parts = [f"{extra} extra missing: {EXTRA_EFFECTS[extra]}" for extra in ordered]
    data: Dict[str, Any] = {
        "missing_extras": sorted(missing),
        "missing_modules": missing,
        "blocking_extras": blocking,
    }
    if "fep" in missing:
        data["fep_indicators"] = list(FEP_INDICATORS)
    return CheckResult(
        "extras",
        FAILED if blocking else INSUFFICIENT,
        "; ".join(parts),
        required=bool(blocking),
        data=data,
    )


def _check_ctm(cfg: Settings) -> CheckResult:
    if not cfg.CTM_ENABLED:
        return CheckResult(
            "ctm", SKIPPED, "continuous thought is off (CTM_ENABLED=false)", False
        )
    from mtc.consciousness.continuous.config import CTMConfig
    from mtc.consciousness.continuous.persona import CTMPersona
    from mtc.consciousness.continuous.seeds import SeedLibrary

    try:
        config = CTMConfig.load(settings=cfg)
        seeds = SeedLibrary.from_config_dir(settings=cfg)
        persona = CTMPersona.load(settings=cfg)
    except ValueError as exc:
        return CheckResult(
            "ctm",
            FAILED,
            f"the continuous thought configuration does not load "
            f"({type(exc).__name__})",
            required=False,
        )
    if seeds.source == "missing":
        quiet: Optional[str] = "no_seeds"
    elif not _has_module("sentence_transformers"):
        quiet = "no_embedder"
    else:
        quiet = None
    data = {
        "config_source": config.source,
        "seed_source": seeds.source,
        "seed_packs": len(seeds.packs),
        "persona_source": persona.source,
        "store": cfg.CTM_STORE,
        "predicted_quiet_reason": quiet,
    }
    detail = (
        f"config: {config.source}; seeds: {seeds.source} ({len(seeds.packs)} "
        f"pack(s)); persona: {persona.source}; store: {cfg.CTM_STORE}"
    )
    if quiet is not None:
        return CheckResult(
            "ctm",
            WARNING,
            f"{detail}; thoughts will stay quiet ({quiet}): none can surface "
            "until that is fixed",
            required=False,
            data=data,
        )
    return CheckResult(
        "ctm", OK, f"{detail}; no quiet reason predicted", required=False, data=data
    )


def _check_service_manager(cfg: Settings) -> CheckResult:
    return CheckResult(
        "service_manager",
        OK,
        f"SERVICE_MANAGER={cfg.SERVICE_MANAGER}",
        required=False,
        data={"value": cfg.SERVICE_MANAGER},
    )


def _check_llm_backend(cfg: Settings) -> CheckResult:
    backend = cfg.LLM_BACKEND
    detail = f"LLM_BACKEND={backend}" if backend else "LLM_BACKEND is empty"
    if not backend and cfg.LLM_PROVIDER == "local":
        detail += ": the stack scripts will not start a model server"
    return CheckResult(
        "llm_backend", OK, detail, required=False, data={"value": backend}
    )


def _check_timezone(cfg: Settings) -> CheckResult:
    from zoneinfo import ZoneInfo

    try:
        ZoneInfo(cfg.TIMEZONE)
    except (KeyError, ValueError, OSError):
        return CheckResult("timezone", FAILED, "TIMEZONE is not an IANA zone name")
    return CheckResult(
        "timezone", OK, f"TIMEZONE={cfg.TIMEZONE}", data={"value": cfg.TIMEZONE}
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _guarded(name: str, required: bool, check: Callable[[], Any]) -> List[CheckResult]:
    """Run one check; an unexpected error is a failed check, never a crash."""
    try:
        outcome = check()
    except Exception as exc:  # noqa: BLE001 - the report must be complete
        logger.debug("doctor check %s raised %s", name, type(exc).__name__)
        return [
            CheckResult(
                name,
                FAILED,
                f"the check itself failed ({type(exc).__name__})",
                required=required,
            )
        ]
    return list(outcome) if isinstance(outcome, list) else [outcome]


def run_doctor(
    settings: Optional[Settings] = None,
    *,
    network: bool = True,
    fetch_models: bool = False,
) -> List[CheckResult]:
    """Run every check and return the results in report order.

    ``settings`` None loads them here, so a load failure is itself a result.
    With ``network`` false no store or model server is probed and nothing is
    downloaded; those checks read ``insufficient_data``.
    """
    results: List[CheckResult] = _guarded("python", True, _check_python)
    cfg = settings
    if cfg is None:
        try:
            cfg = Settings()
        except Exception as exc:  # noqa: BLE001 - named by field, never by value
            fields = _settings_fields(exc)
            detail = "the settings do not load"
            if fields:
                detail += f" (fields: {fields})"
            results.append(CheckResult("settings", FAILED, detail))
            for name, required in CHECKS[2:]:
                results.append(
                    CheckResult(
                        name,
                        INSUFFICIENT,
                        "not run: the settings did not load",
                        required=required,
                    )
                )
            return results
    results.append(CheckResult("settings", OK, "the settings load"))
    active: Settings = cfg
    plan: Sequence[tuple] = (
        ("env_file", False, lambda: _check_env_file(active)),
        ("persona", False, lambda: _check_persona(active)),
        ("identity_consistency", False, lambda: _check_identity(active)),
        ("speakers", False, lambda: _check_speakers(active)),
        ("api_key", True, lambda: _check_api_key(active)),
        ("birth_date", False, lambda: _check_birth_date(active)),
        ("stores", True, lambda: _check_stores(active, network)),
        ("brain", True, lambda: _check_brain(active, network)),
        ("provider_state", False, lambda: _check_provider_state(active)),
        (
            "embedding_model",
            False,
            lambda: _check_embedding_model(active, network, fetch_models),
        ),
        ("pyphi", False, lambda: _check_pyphi(active)),
        ("extras", False, _check_extras),
        ("ctm", False, lambda: _check_ctm(active)),
        ("service_manager", False, lambda: _check_service_manager(active)),
        ("llm_backend", False, lambda: _check_llm_backend(active)),
        ("timezone", True, lambda: _check_timezone(active)),
    )
    for name, required, check in plan:
        results.extend(_guarded(name, required, check))
    return results
