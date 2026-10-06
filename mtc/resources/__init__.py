"""
Packaged Resources
==================

Generated copies of the shipped runtime defaults, so a non-editable install
reads them from any working directory. Every file below this package is
written by ``scripts/sync_resources.py`` from its source in the repository and
is never edited by hand; ``mtc.core.resources.RESOURCE_MANIFEST`` names each
source and its reader:

- ``config/ctm/ctm.yaml``, ``config/ctm/seeds/default.yaml`` and
  ``config/ctm/persona.example.yaml`` (continuous thought);
- ``config/persona.example.yaml`` and ``config/speakers.example.yaml``;
- ``config/transition_prompts.json`` and ``config/dreams.yaml``;
- ``config/pyphi_config.yml`` (the PyPhi home's presentation config);
- ``data/dcm/bundled_spec.json`` and ``data/dcm/baselines/``;
- ``experiments/param_schema.toml`` and ``experiments/presets/``.

None of them holds private data.
"""
