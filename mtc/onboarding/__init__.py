"""Onboarding: the setup wizard, its answers schema and the store seeding.

``mtc.onboarding.schema`` holds the persona, speakers and answers models,
``mtc.onboarding.wizard`` the eleven-step setup (``run_init``) and
``mtc.onboarding.seed_db`` the install date and the store seeds. The names
below are loaded on first use, so importing the package stays cheap.
"""

from typing import Any, List

_EXPORTS = {
    "CompanionType": "mtc.onboarding.schema",
    "OnboardingAnswers": "mtc.onboarding.schema",
    "PersonaConfig": "mtc.onboarding.schema",
    "SpeakerConfig": "mtc.onboarding.schema",
    "SpeakersFile": "mtc.onboarding.schema",
    "export_json_schemas": "mtc.onboarding.schema",
    "apply_plan": "mtc.onboarding.wizard",
    "collect_interactive": "mtc.onboarding.wizard",
    "load_answers": "mtc.onboarding.wizard",
    "plan_writes": "mtc.onboarding.wizard",
    "run_init": "mtc.onboarding.wizard",
}

__all__: List[str] = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
