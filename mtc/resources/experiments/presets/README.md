# Experiment presets

A preset is a TOML file of parameter overrides that the runners layer on top
of the module defaults. Presets use the same keys and sections as
`experiments/params.toml` and are validated against
`experiments/param_schema.toml`, so a typo or an out-of-range value fails the
run before any module is built.

## Layering

1. Module defaults (the new v0.5.0 behaviour).
2. `--preset NAME` loads `experiments/presets/NAME.toml` on top of the defaults.
3. `--params PATH` (default `experiments/params.toml`) loads on top of the
   preset, so a value in the params file wins over the same key in the preset.

Both runners accept the two flags:

```bash
python3 scripts/run_experiment.py --preset legacy_v04
python3 scripts/run_assessment.py --preset legacy_v04 --full
python3 scripts/run_experiment.py --params path/to/other_params.toml
```

## legacy_v04

Reproduces the published v0.4.0 behaviour so the v0.4.0 numbers stay
reproducible. It sets every behaviour switch to its v0.4.0 value. The `[fep]`
numeric values (`planning_horizon`, `num_actions`, `num_hidden_states`,
`learning_rate_A`, `learning_rate_B`) and `[blt].field_evidencing_threshold`
are copied verbatim from the `experiments/params.toml` that was applied when
the v0.4.0 baseline (`tests/fixtures/baseline_v040_experiment.txt`) was
captured. `tests/test_legacy_preset.py` checks that the preset still lands on
that baseline.

The preset also sets an `[assessment]` block (`indicator_formulas`,
`missing_signal_policy`, `irreducibility_method`, `causal_emergence_threshold`,
`phi_source`) and three tables: `[pad]` (`coherence_method = "v04_variance"`),
`[causal_emergence]` (`tpm_row_order = "little_endian"`) and `[rpt]`
(`stdp_plasticity_enabled = false`). `assessment.phi_normalization`,
`assessment.phi_fixed_reference`, `pad.coherence_window` and
`pad.coherence_min_history` have no preset value: the first two default to the
v0.4.0 divide-by-5.0 behaviour and the PAD tuning keys are inert under
`v04_variance`.

The v0.5.0 release added eleven switch rows, each with its v0.4.0 value in
the preset: `assessment.noise_baseline_matching = "v04_any"`,
`runner.assess_blt_source = "standalone"`, `gwt.legacy_broadcast_receivers =
true`, `gwt.legacy_sustain_rate = true`, `gwt.refractory_clock = "per_cycle"`
(also the default), `gwt.recency_clock = "v04_wall"` (the wall-clock recency
bonus of v0.4.0; the default is the simulated clock), a `[hot]` table
(`overall_confidence_source =
"v04_constant"`, `introspection_counting = "v04_per_hot"` and an explicit
`introspection_cooldown = 1.0`, the wall-clock gate the runners otherwise pin
at 0.0), `blt.binding_evidence = "v04_salience_posterior"`,
`blt.strange_loop_test = "v04_confidence_match"` and `damasio.first_delta =
"v04_zero"`. One dependency: `blt.loop_quality_formula = "v04"` needs
`blt.binding_evidence = "v04_salience_posterior"` (the v04 formula reads a
binding quality only that evidence measures); a params file that sets the
formula alone fails at construction with a message naming both keys.
`tests/test_legacy_preset.py` checks that every schema row has a preset
value, an unchanged v0.4.0 default or a recorded reason.

## Adding a preset

1. Copy an existing preset to `experiments/presets/<name>.toml`.
2. Keep only the keys that differ from the defaults; every key must exist in
   `experiments/param_schema.toml`.
3. Add a one-line comment at the top saying what the preset is for.
4. Run `python3 scripts/run_experiment.py --preset <name>` once to confirm the
   file validates.
