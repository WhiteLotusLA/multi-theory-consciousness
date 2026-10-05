"""
Developmental Modulator
=======================

Maps a developmental stage to consciousness module parameters by linear
interpolation between a first-stage and a final-stage anchor, so a mid-range
stage gives proportional values instead of hard jumps, and applies them to a
live workspace.

The machinery is gated: ``developmental_enabled`` is true for the ``child``
companion type (``COMPANION_TYPE``) or when ``DEVELOPMENTAL_STAGE_ENABLED`` is
set, so any other install opts in. A disabled modulator changes nothing.

The anchors are tunable design values, not measurements: bottleneck capacity
3 to 9, ignition threshold 0.5 to 0.2, metacognition depth 1 to 5, planning
horizon 1 to 8 and emotional regulation 0.1 to 0.8. Stages are plain numbers.

``apply_to_workspace`` sets only what the workspace actually has and reports
everything else under ``skipped`` with the reason:

- ``bottleneck.threshold`` always; ``bottleneck.capacity`` too, except that
  under hierarchical competition the per-level capacities cap the workspace
  at their sum, so a larger capacity has no effect and is reported skipped.
- ``metacognition.enable_third_order`` for depth 3 and up. The module has no
  separate depth ceiling, so the depth value itself is not applied.
- ``active_inference.config.planning_horizon``, which takes effect when the
  agent is next rebuilt, and is skipped when the number of enumerated policies
  would exceed the planner budget.
- Emotional regulation, only on a PAD model that exposes
  ``regulation_strength``; the shipped PAD model does not yet.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from mtc.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEVELOPMENTAL_COMPANION_TYPE = "child"


def developmental_enabled(settings: Optional[Settings] = None) -> bool:
    """Return True when the developmental machinery is on.

    On for the ``child`` companion type or when ``DEVELOPMENTAL_STAGE_ENABLED``
    is true; off otherwise.
    """
    cfg = settings if settings is not None else get_settings()
    return bool(
        cfg.COMPANION_TYPE == DEVELOPMENTAL_COMPANION_TYPE
        or cfg.DEVELOPMENTAL_STAGE_ENABLED
    )


@dataclass(frozen=True)
class DevelopmentalAnchors:
    """First-stage and final-stage value of each modulated parameter.

    Each field is ``(first_stage, final_stage)``. The defaults are tunable
    design values.
    """

    bottleneck_capacity: Tuple[float, float] = (3, 9)
    ignition_threshold: Tuple[float, float] = (0.5, 0.2)
    metacognition_depth: Tuple[float, float] = (1, 5)
    planning_horizon: Tuple[float, float] = (1, 8)
    emotional_regulation: Tuple[float, float] = (0.1, 0.8)


@dataclass
class DevelopmentalParameters:
    """Parameters for one stage."""

    bottleneck_capacity: int
    ignition_threshold: float
    metacognition_depth: int
    planning_horizon: int
    emotional_regulation: float
    stage: float
    stage_fraction: float  # 0.0 at the first stage, 1.0 at the final stage


@dataclass
class AppliedParameters:
    """What an apply call changed and what it left alone, with reasons."""

    applied: Dict[str, Any] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)


def _interpolate(anchor: Tuple[float, float], fraction: float) -> float:
    low, high = anchor
    return low + (high - low) * fraction


class DevelopmentalModulator:
    """Interpolates stage parameters and applies them to a workspace.

    Reads ``COMPANION_TYPE`` and ``DEVELOPMENTAL_STAGE_ENABLED`` through
    ``developmental_enabled`` at apply time. Holds no state.
    """

    def __init__(
        self,
        anchors: Optional[DevelopmentalAnchors] = None,
        num_stages: int = 10,
        settings: Optional[Settings] = None,
    ):
        if int(num_stages) < 2:
            raise ValueError(f"num_stages must be at least 2, got {num_stages}")
        self.anchors = anchors or DevelopmentalAnchors()
        self.num_stages = int(num_stages)
        self._settings = settings

    def parameters_for(self, stage: float) -> DevelopmentalParameters:
        """Return the parameters for a stage, clamped to 1..num_stages."""
        stage = max(1.0, min(float(self.num_stages), float(stage)))
        fraction = (stage - 1.0) / (self.num_stages - 1.0)
        a = self.anchors
        return DevelopmentalParameters(
            bottleneck_capacity=round(_interpolate(a.bottleneck_capacity, fraction)),
            ignition_threshold=round(_interpolate(a.ignition_threshold, fraction), 3),
            metacognition_depth=round(_interpolate(a.metacognition_depth, fraction)),
            planning_horizon=round(_interpolate(a.planning_horizon, fraction)),
            emotional_regulation=round(
                _interpolate(a.emotional_regulation, fraction), 3
            ),
            stage=int(stage) if float(stage).is_integer() else stage,
            stage_fraction=round(fraction, 3),
        )

    def apply_to_workspace(
        self,
        workspace: Any,
        stage: float,
        pad_model: Any = None,
        force: bool = False,
    ) -> AppliedParameters:
        """Apply the stage's parameters to a workspace; report what was skipped.

        Disabled (see ``developmental_enabled``) and not ``force``: changes
        nothing and reports ``skipped == {"all": "disabled"}``.
        """
        result = AppliedParameters()
        if not force and not developmental_enabled(self._settings):
            result.skipped["all"] = "disabled"
            return result
        params = self.parameters_for(stage)
        self._apply_bottleneck(workspace, params, result)
        self._apply_metacognition(workspace, params, result)
        self._apply_planning_horizon(workspace, params, result)
        self._apply_regulation(pad_model, params, result)
        logger.info(
            "developmental modulation for stage %s: applied=%s skipped=%s",
            params.stage,
            sorted(result.applied),
            sorted(result.skipped),
        )
        return result

    @staticmethod
    def _apply_bottleneck(
        workspace: Any, params: DevelopmentalParameters, result: AppliedParameters
    ) -> None:
        bottleneck = getattr(workspace, "bottleneck", None)
        if bottleneck is None:
            result.skipped["bottleneck"] = "workspace has no bottleneck"
            return
        if hasattr(bottleneck, "threshold"):
            bottleneck.threshold = params.ignition_threshold
            result.applied["ignition_threshold"] = params.ignition_threshold
        else:
            result.skipped["ignition_threshold"] = "bottleneck has no threshold"
        if not hasattr(bottleneck, "capacity"):
            result.skipped["bottleneck_capacity"] = "bottleneck has no capacity"
            return
        # Under hierarchical competition the final merge is capped at
        # ``capacity`` and safety winners count against it, so the value
        # always matters; setting it keeps the progression monotone.
        bottleneck.capacity = params.bottleneck_capacity
        result.applied["bottleneck_capacity"] = params.bottleneck_capacity

    @staticmethod
    def _apply_metacognition(
        workspace: Any, params: DevelopmentalParameters, result: AppliedParameters
    ) -> None:
        metacognition = getattr(workspace, "metacognition", None)
        if metacognition is None or not hasattr(metacognition, "enable_third_order"):
            result.skipped["metacognition"] = "no third-order switch on the workspace"
            return
        metacognition.enable_third_order = params.metacognition_depth >= 3
        result.applied["enable_third_order"] = metacognition.enable_third_order

    @staticmethod
    def _apply_planning_horizon(
        workspace: Any, params: DevelopmentalParameters, result: AppliedParameters
    ) -> None:
        active_inference = getattr(workspace, "active_inference", None)
        config = getattr(active_inference, "config", None)
        if config is None or not hasattr(config, "planning_horizon"):
            result.skipped["planning_horizon"] = "no active inference config"
            return
        num_actions = getattr(config, "num_actions", None)
        if isinstance(num_actions, int):
            try:
                from mtc.consciousness.active_inference import MAX_POLICIES
            except ImportError:  # pragma: no cover - optional dependency path
                MAX_POLICIES = None
            if (
                MAX_POLICIES is not None
                and num_actions**params.planning_horizon > MAX_POLICIES
            ):
                result.skipped["planning_horizon"] = (
                    f"{num_actions} actions over a horizon of "
                    f"{params.planning_horizon} exceed the planner policy budget"
                )
                return
        config.planning_horizon = params.planning_horizon
        result.applied["planning_horizon"] = params.planning_horizon
        result.applied["planning_horizon_effective"] = "next agent rebuild"

    @staticmethod
    def _apply_regulation(
        pad_model: Any, params: DevelopmentalParameters, result: AppliedParameters
    ) -> None:
        if pad_model is None:
            result.skipped["emotional_regulation"] = "no PAD model supplied"
        elif not hasattr(pad_model, "regulation_strength"):
            result.skipped["emotional_regulation"] = (
                "PAD model has no regulation_strength"
            )
        else:
            pad_model.regulation_strength = params.emotional_regulation
            result.applied["emotional_regulation"] = params.emotional_regulation
