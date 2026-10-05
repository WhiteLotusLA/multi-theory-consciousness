"""
Narration Flags
===============

One reader for the ``NARRATE_*`` Settings family. Every flag defaults to false
in Settings, so the live pipeline narrates nothing about its inner state unless
an operator opts in; the module keyword arguments keep their own True defaults
for direct research use, and the composition root passes these values.

Config keys read (at call time): ``NARRATE_SOMATIC``, ``NARRATE_AFFECT_LINE``,
``NARRATE_NARRATIVE_IDENTITY``, ``NARRATE_BODY_DRIVES``, ``NARRATE_BINDING``,
``NARRATE_SELF_MODEL``, ``NARRATE_FREE_ENERGY``, ``NARRATE_THOUGHT_MOOD``,
``NARRATE_SPEAKER_MODEL``, ``NARRATE_INNER_STATE`` and ``NARRATE_ALL``, the
documented one-line experiment profile that turns all ten on. No state is
persisted and no private data is held.
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Tuple

from mtc.core.config import Settings, get_settings

NARRATION_FLAG_KEYS: Tuple[str, ...] = (
    "NARRATE_SOMATIC",
    "NARRATE_AFFECT_LINE",
    "NARRATE_NARRATIVE_IDENTITY",
    "NARRATE_BODY_DRIVES",
    "NARRATE_BINDING",
    "NARRATE_SELF_MODEL",
    "NARRATE_FREE_ENERGY",
    "NARRATE_THOUGHT_MOOD",
    "NARRATE_SPEAKER_MODEL",
    "NARRATE_INNER_STATE",
)


@dataclass(frozen=True)
class NarrationFlags:
    """Which inner-state narrations the live pipeline renders.

    Effect map (field, Settings key, what it gates):

    - ``somatic`` (``NARRATE_SOMATIC``): the somatic segment of
      ``DamasioLayers``.
    - ``affect_line`` (``NARRATE_AFFECT_LINE``):
      ``PADAffectModel.narrate_affect_line``.
    - ``narrative_identity`` (``NARRATE_NARRATIVE_IDENTITY``): the
      narrative-identity segment of ``DamasioLayers``.
    - ``body_drives`` (``NARRATE_BODY_DRIVES``): the body-drives segment of
      ``DamasioLayers``, ``ActiveInferenceConfig.narrate_body_drives`` and the
      orchestrator's body-state and circadian lines.
    - ``binding`` (``NARRATE_BINDING``): the orchestrator's binding line and
      the prompt's neural-state section.
    - ``self_model`` (``NARRATE_SELF_MODEL``): the ``MetacognitionModule`` and
      ``BeautifulLoop`` self-model reports.
    - ``free_energy`` (``NARRATE_FREE_ENERGY``):
      ``ActiveInferenceConfig.narrate_free_energy`` and the orchestrator's
      surprise line.
    - ``thought_mood`` (``NARRATE_THOUGHT_MOOD``): the prompt's thought-mood
      stamps.
    - ``speaker_model`` (``NARRATE_SPEAKER_MODEL``):
      ``TheoryOfMind.narrate_speaker_model``.
    - ``inner_state`` (``NARRATE_INNER_STATE``): the orchestrator's
      self-awareness layer.

    ``NARRATE_ALL`` true turns every flag on (``all_on``).
    """

    somatic: bool = False
    affect_line: bool = False
    narrative_identity: bool = False
    body_drives: bool = False
    binding: bool = False
    self_model: bool = False
    free_energy: bool = False
    thought_mood: bool = False
    speaker_model: bool = False
    inner_state: bool = False

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "NarrationFlags":
        """Read the flags from Settings now; ``NARRATE_ALL`` turns all on."""
        active = settings if settings is not None else get_settings()
        if bool(getattr(active, "NARRATE_ALL", False)):
            return cls.all_on()
        values = {
            field.name: bool(getattr(active, key, False))
            for field, key in zip(fields(cls), NARRATION_FLAG_KEYS)
        }
        return cls(**values)

    @classmethod
    def all_on(cls) -> "NarrationFlags":
        """The experiment profile: every narration on."""
        return cls(**{field.name: True for field in fields(cls)})

    def as_dict(self) -> Dict[str, bool]:
        """The flags by field name, in declaration order."""
        return dict(asdict(self))
