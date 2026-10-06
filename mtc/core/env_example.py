"""Render .env.example and docs/CONFIGURATION.md from the Settings model.

The committed files are generated artifacts; tests fail when they drift.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Tuple, Type

from pydantic import AliasChoices
from pydantic_settings import BaseSettings

# fmt: off
SECTIONS: List[Tuple[str, Tuple[str, ...]]] = [
    ("System", ("SYSTEM_NAME", "CREATOR_NAME", "SERVICE_HOST", "ENVIRONMENT", "DEBUG",
                "LOG_", "PROJECT_ROOT", "MODEL_DIRECTORY", "DATA_DIRECTORY",
                "TEMP_DIRECTORY")),
    ("Identity and locale", ("CREATOR_RELATIONSHIP", "PERSONA_PATH", "CORE_VALUES",
                             "SYSTEM_BIRTH_DATE", "COMPANION_TYPE",
                             "DEVELOPMENTAL_STAGE_ENABLED", "TIMEZONE",
                             "WAKE_HOUR", "SLEEP_HOUR", "LOCATION_")),
    ("Paths", ("AGENT_SPACE_DIR", "IDENTITY_STATEMENT_PATH", "SHADOW_LOG_DIR",
               "RESULTS_DIR", "STATE_DIR")),
    ("Databases", ("DB_PREFIX", "POSTGRES_", "DB_POOL_", "MONGODB_", "REDIS_",
                   "QDRANT_", "NEO4J_", "VECTOR_", "STORE_")),
    ("LLM", ("LLM_", "MODEL_", "CONTEXT_SIZE", "BATCH_SIZE", "TEMPERATURE",
             "TOP_", "MIN_P", "REPEAT_PENALTY", "THINKING_MODE", "GPU_LAYERS",
             "THREADS", "USE_M", "ANTHROPIC_")),
    ("Brain and prompt", ("GENERATION_", "THINKING_LOOP_", "PROVIDER_STATE_PATH",
                          "VISION_ENABLED", "HISTORY_TOKEN_BUDGET",
                          "CHARS_PER_TOKEN", "PROMPT_", "TRANSITION_")),
    ("Narration", ("NARRATE_",)),
    # Before the consciousness section, whose SELF_ prefix would otherwise
    # capture SELF_ASSESSMENT_INTERVAL_S (the first matching section wins).
    ("Runtime", ("PIPELINE_PRESET", "SLEEP_SCHEDULER_", "AUTONOMOUS_LIFE_",
                 "PHI_IDLE_", "SELF_ASSESSMENT_", "THINKING_HEARTBEAT_S")),
    ("Neural substrates", ("DEVICE", "SNN_", "LSM_", "HTM_")),
    ("Speakers and trust", ("DEFAULT_SPEAKER", "DEFAULT_EMOTION", "SPEAKER",
                            "TRUST")),
    ("Consciousness modules", ("CONSCIOUSNESS_CYCLE", "CONSCIOUSNESS_ENABLED",
                               "META_", "SELF_", "GWT_", "AST_", "HOT_", "FEP_",
                               "IIT_", "RPT_", "BLT_", "ATTENTION_GRAB_")),
    ("Continuous thought", ("CTM_", "SENSITIVE_MEMORY_")),
    ("Assessment", ("ASSESSMENT_", "ABLATION_", "LONGITUDINAL_", "NOISE_",
                    "TRACK_INDICATORS", "METRICS_", "CONSCIOUSNESS_SCORE",
                    "CONSCIOUSNESS_PASSING", "DCM_")),
    ("Memory", ("WORKING_MEMORY_", "EPISODIC_", "SEMANTIC_", "EMBEDDING_MODEL",
                "RELATIONSHIP_KEYWORDS", "RAG_", "PROTECTED_NAMES", "NIGHT_")),
    ("Emotion and learning", ("OPENNESS", "CONSCIENTIOUSNESS", "EXTRAVERSION",
                              "AGREEABLENESS", "NEUROTICISM", "EMOTION",
                              "LEARNING_", "NOVELTY_", "PATTERN_")),
    ("Safety and API", ("SAFETY_", "CONSENT_", "EMERGENCY_", "TRANSPARENCY_",
                        "MANIPULATION_THRESHOLD", "DECEPTION_THRESHOLD",
                        "VALUE_ALIGNMENT_MINIMUM", "ALERT_NOTIFIERS", "API_",
                        "WEBSOCKET_", "PROMETHEUS_")),
    ("Operations", ("BACKUP_", "SERVICE_MANAGER")),
]
# fmt: on

SECRET_MARKERS = ("PASSWORD", "API_KEY", "SECRET", "TOKEN")


def _section_for(name: str) -> str:
    for title, prefixes in SECTIONS:
        if any(name == p or name.startswith(p) for p in prefixes):
            return title
    return "Other"


def _env_name(settings_cls: Type[BaseSettings], name: str, info) -> str:
    alias = info.validation_alias
    if isinstance(alias, AliasChoices):
        return str(alias.choices[0])
    if isinstance(alias, str):
        return alias
    prefix = settings_cls.model_config.get("env_prefix", "")
    return f"{prefix}{name}"


def _render_default(name: str, default) -> str:
    # A marker in the name hides a text value only: numeric keys such as
    # HISTORY_TOKEN_BUDGET are not secrets and must render their default.
    if isinstance(default, str) and any(m in name for m in SECRET_MARKERS):
        return ""
    if default is None:
        return ""
    if isinstance(default, bool):
        return "true" if default else "false"
    if isinstance(default, (dict, list)):
        return json.dumps(default, sort_keys=True)
    return str(default)


def _default(info):
    return info.get_default(call_default_factory=True)


def _grouped(settings_cls: Type[BaseSettings]) -> Iterable[Tuple[str, list]]:
    groups: dict = {title: [] for title, _ in SECTIONS}
    groups["Other"] = []
    for name, info in settings_cls.model_fields.items():
        groups[_section_for(name)].append((name, info))
    for title, items in groups.items():
        if items:
            yield title, items


def render_env_example(settings_cls: Type[BaseSettings]) -> str:
    lines = [
        "# Multi-Theory Consciousness Framework configuration",
        "# GENERATED from mtc.core.config.Settings by scripts/gen_env_example.py.",
        "# Edit the Settings model, then run: python scripts/gen_env_example.py",
        "# Copy to .env and fill in the empty secrets.",
        "",
    ]
    for title, items in _grouped(settings_cls):
        lines.append(f"# === {title} ===")
        for name, info in items:
            if info.description:
                lines.append(f"# {info.description}")
            lines.append(
                f"{_env_name(settings_cls, name, info)}="
                f"{_render_default(name, _default(info))}"
            )
        lines.append("")
    return "\n".join(lines)


def render_configuration_md(settings_cls: Type[BaseSettings]) -> str:
    lines = [
        "# Configuration",
        "",
        "GENERATED from `mtc.core.config.Settings` by `scripts/gen_env_example.py`."
        " Do not edit by hand; edit the model and regenerate.",
        "",
        "Every key is read through `get_settings()`. Values come from the process"
        " environment or a `.env` file in the working directory. Secrets render"
        " empty here and must be supplied by the operator.",
        "",
    ]
    for title, items in _grouped(settings_cls):
        lines += [
            f"## {title}",
            "",
            "| Variable | Default | Description |",
            "|---|---|---|",
        ]
        for name, info in items:
            default = _render_default(name, _default(info)) or "(empty)"
            desc = (info.description or "").replace("|", "/")
            lines.append(
                f"| `{_env_name(settings_cls, name, info)}` | `{default}` | {desc} |"
            )
        lines.append("")
    return "\n".join(lines)
