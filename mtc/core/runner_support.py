"""
Shared construction and warmup helpers for the experiment and assessment runners.

Both runners build the same workspace from the same layered overrides and give
the attention schema the same shift types during warmup. Keeping that in one
place means a preset switches every instance in a run the same way. Everything
here is a pure function of its arguments: no I/O, no random draws.
"""

from dataclasses import fields
from typing import Any, Dict, Optional

from mtc.consciousness.attention_schema import AttentionShiftType
from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    TemporalWorkspaceConfig,
)
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.consciousness.phi_tracker import PhiTracker
from mtc.pipeline.cycle_augmenter import CycleAugmenter, CycleAugmenterConfig

# Warmup profiles of runner.warmup_profile
WARMUP_PROFILE_V04 = "v04"
WARMUP_PROFILE_V05 = "v05"
WARMUP_PROFILES = (WARMUP_PROFILE_V05, WARMUP_PROFILE_V04)

# fep.* behaviour switches forwarded to the workspace's own inference module.
# The generative-model geometry and learning rates stay as the workspace sets
# them; a runner's separate module takes every fep.* key.
FEP_BEHAVIOUR_SWITCHES = (
    "hpp_hidden_dim_multiplier",
    "hierarchy_observation",
    "meta_accuracy_method",
    "drive_model",
    "drive_decay_floor",
    "legacy_double_decay",
)

# gwt.* behaviour switches that are EnhancedGlobalWorkspace constructor fields.
GWT_SWITCHES = (
    "competition_mode",
    "thalamic_gate_enabled",
    "theory_interactions_enabled",
    "legacy_coverage_ratio",
    "body_metrics_enabled",
    "precision_feedback_enabled",
    "self_prediction_enabled",
)

# ast.* keys that are AttentionSchemaModule constructor fields; the workspace
# builds its own schema from them through attention_schema_kwargs.
AST_SWITCHES = (
    "voluntary_boost",
    "focus_threshold",
    "hyperfocus_threshold",
    "capture_salience_threshold",
    "provenance_mode",
    "prediction_window_seconds",
    "history_size",
)

# cycle.* switches of the CycleAugmenter around each workspace cycle. The
# runners pin every one of them off unless the overrides set it.
CYCLE_SWITCHES = tuple(f.name for f in fields(CycleAugmenterConfig))

# gwt.* keys that belong to TemporalWorkspaceConfig, not to the workspace
_TEMPORAL_FIELDS = frozenset(f.name for f in fields(TemporalWorkspaceConfig))

# The rotation v05 assigns over four cycles: a chosen shift, a pulled one, a
# goal-driven one, a pulled one
_V05_ROTATION = (
    AttentionShiftType.VOLUNTARY,
    AttentionShiftType.CAPTURED,
    AttentionShiftType.GOAL_DRIVEN,
    AttentionShiftType.CAPTURED,
)


def attention_shift_type(cycle: int, profile: str) -> AttentionShiftType:
    """Shift type a warmup assigns to the attention focus on a cycle.

    v04 only ever reported captured shifts. v05 rotates VOLUNTARY, CAPTURED,
    GOAL_DRIVEN, CAPTURED so typed provenance has real chosen shifts to count.
    Deterministic: it draws nothing from the random stream.
    """
    if profile == WARMUP_PROFILE_V04:
        return AttentionShiftType.CAPTURED
    if profile == WARMUP_PROFILE_V05:
        return _V05_ROTATION[cycle % len(_V05_ROTATION)]
    raise ValueError(f"profile must be one of {WARMUP_PROFILES}, got {profile!r}")


def build_phi_tracker(
    phi_overrides: Dict[str, Any], min_transitions: int
) -> PhiTracker:
    """PhiTracker from the phi.* keys (scale, warmup_reference).

    ``min_transitions`` is the number of recorded transitions before the
    tracker reports a Phi; a short warmup lowers it to what its analyses need.
    """
    kwargs: Dict[str, Any] = {"min_transitions": min_transitions}
    if "scale" in phi_overrides:
        kwargs["phi_scale"] = phi_overrides["scale"]
    if "warmup_reference" in phi_overrides:
        kwargs["phi_warmup_reference"] = phi_overrides["warmup_reference"]
    return PhiTracker(**kwargs)


def build_cycle_augmenter(
    overrides: Dict[str, Any],
    workspace: EnhancedGlobalWorkspace,
    pad_model: Optional[PADAffectModel] = None,
    *,
    pinned_off: bool = True,
) -> CycleAugmenter:
    """The cycle augmenter a runner wraps around each workspace cycle.

    With ``pinned_off`` (the runners' setting) every ``cycle.*`` switch is off
    unless the layered overrides set it, so a seeded run's numbers move only
    when a researcher opts in; ``pinned_off=False`` starts from the live
    pipeline's defaults. ``pad_model`` is the PAD model the state's coordinate
    is recorded through when ``cycle.pad_from_state`` is on.
    """
    cycle_ov = {
        k: v for k, v in overrides.get("cycle", {}).items() if k in CYCLE_SWITCHES
    }
    config = CycleAugmenterConfig.from_overrides(cycle_ov, pinned_off=pinned_off)
    return CycleAugmenter(config, workspace, pad_model=pad_model)


def build_workspace(
    overrides: Dict[str, Any],
    phi_tracker: Optional[PhiTracker],
    *,
    body_metrics_default: bool = True,
) -> EnhancedGlobalWorkspace:
    """The workspace both runners use, built from the layered overrides.

    The workspace builds its own inference, beautiful loop, damasio,
    metacognition and attention schema modules; the fep, blt, damasio, hot and
    ast overrides reach them through its constructor so a preset switches every
    instance in the run. The gwt keys fill the workspace's own fields, its
    behaviour switches and its temporal config.

    ``body_metrics_default`` is the value of ``body_metrics_enabled`` when the
    overrides do not set it. Machine metrics and the previous cycle's duration
    reach the protoself when it is on, so a seeded run passes False.
    """
    gwt_ov = dict(overrides.get("gwt", {}))

    temporal_ov: Dict[str, Any] = {
        k: gwt_ov[k] for k in _TEMPORAL_FIELDS if k in gwt_ov
    }
    nested = gwt_ov.get("temporal")
    if isinstance(nested, dict):
        temporal_ov.update(nested)
    temporal_config = TemporalWorkspaceConfig(**temporal_ov) if temporal_ov else None

    switches: Dict[str, Any] = {"body_metrics_enabled": body_metrics_default}
    switches.update({k: gwt_ov[k] for k in GWT_SWITCHES if k in gwt_ov})

    fep_forward = {
        k: v for k, v in overrides.get("fep", {}).items() if k in FEP_BEHAVIOUR_SWITCHES
    }
    ast_forward = {
        k: v for k, v in overrides.get("ast", {}).items() if k in AST_SWITCHES
    }

    return EnhancedGlobalWorkspace(
        capacity=gwt_ov.get("capacity", 7),
        ignition_threshold=gwt_ov.get("ignition_threshold", 0.3),
        amplification_factor=gwt_ov.get("amplification_factor", 2.0),
        temporal_config=temporal_config,
        active_inference_config=(
            EnhancedGlobalWorkspace.default_active_inference_config(**fep_forward)
        ),
        beautiful_loop_kwargs=dict(overrides.get("blt", {})),
        damasio_kwargs=dict(overrides.get("damasio", {})),
        metacognition_kwargs=dict(overrides.get("hot", {})),
        attention_schema_kwargs=ast_forward,
        phi_tracker=phi_tracker,
        **switches,
    )
