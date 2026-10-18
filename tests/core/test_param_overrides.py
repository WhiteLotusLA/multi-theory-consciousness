import pytest


class TestParamOverrides:
    def _write_toml(self, tmp_path, filename, content):
        filepath = tmp_path / filename
        filepath.write_text(content)
        return str(filepath)

    def test_load_empty_overrides(self, tmp_path):
        from mtc.core.param_overrides import load_overrides

        schema_path = self._write_toml(
            tmp_path,
            "schema.toml",
            '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n',
        )
        params_path = self._write_toml(tmp_path, "params.toml", "# empty\n")
        result = load_overrides(params_path, schema_path)
        assert result == {}

    def test_load_valid_override(self, tmp_path):
        from mtc.core.param_overrides import load_overrides

        schema_path = self._write_toml(
            tmp_path,
            "schema.toml",
            '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n',
        )
        params_path = self._write_toml(tmp_path, "params.toml", "[gwt]\ncapacity = 5\n")
        result = load_overrides(params_path, schema_path)
        assert result == {"gwt": {"capacity": 5}}

    def test_reject_out_of_bounds(self, tmp_path):
        from mtc.core.param_overrides import load_overrides

        schema_path = self._write_toml(
            tmp_path,
            "schema.toml",
            '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n',
        )
        params_path = self._write_toml(
            tmp_path, "params.toml", "[gwt]\ncapacity = 50\n"
        )
        with pytest.raises(ValueError, match="out of bounds"):
            load_overrides(params_path, schema_path)

    def test_reject_unknown_key(self, tmp_path):
        from mtc.core.param_overrides import load_overrides

        schema_path = self._write_toml(
            tmp_path,
            "schema.toml",
            '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n',
        )
        params_path = self._write_toml(
            tmp_path, "params.toml", "[gwt]\nnonexistent_param = 42\n"
        )
        with pytest.raises(ValueError, match="not in schema"):
            load_overrides(params_path, schema_path)

    def test_reject_wrong_type(self, tmp_path):
        from mtc.core.param_overrides import load_overrides

        schema_path = self._write_toml(
            tmp_path,
            "schema.toml",
            '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n',
        )
        params_path = self._write_toml(
            tmp_path, "params.toml", "[gwt]\ncapacity = 5.5\n"
        )
        with pytest.raises(ValueError, match="type"):
            load_overrides(params_path, schema_path)

    def test_nested_drives(self, tmp_path):
        from mtc.core.param_overrides import load_overrides

        schema_path = self._write_toml(
            tmp_path,
            "schema.toml",
            '[fep.drives.curiosity.optimal_level]\ntype = "float"\ndefault = 0.5\nmin = 0.0\nmax = 1.0\ndescription = "Curiosity target level"\n',
        )
        params_path = self._write_toml(
            tmp_path, "params.toml", "[fep.drives.curiosity]\noptimal_level = 0.7\n"
        )
        result = load_overrides(params_path, schema_path)
        assert result["fep"]["drives"]["curiosity"]["optimal_level"] == 0.7

    def test_reject_value_outside_choices(self, tmp_path):
        from mtc.core.param_overrides import load_overrides

        schema_path = self._write_toml(
            tmp_path,
            "schema.toml",
            '[fep.drive_model]\ntype = "str"\ndefault = "f4"\n'
            'choices = ["f4", "v04"]\ndescription = "Drive model"\n',
        )
        params_path = self._write_toml(
            tmp_path, "params.toml", '[fep]\ndrive_model = "bogus"\n'
        )
        with pytest.raises(ValueError, match="not one of"):
            load_overrides(params_path, schema_path)

    def test_accept_value_inside_choices(self, tmp_path):
        from mtc.core.param_overrides import load_overrides

        schema_path = self._write_toml(
            tmp_path,
            "schema.toml",
            '[fep.drive_model]\ntype = "str"\ndefault = "f4"\n'
            'choices = ["f4", "v04"]\ndescription = "Drive model"\n',
        )
        params_path = self._write_toml(
            tmp_path, "params.toml", '[fep]\ndrive_model = "v04"\n'
        )
        assert load_overrides(params_path, schema_path) == {
            "fep": {"drive_model": "v04"}
        }

    def test_real_schema_choices_are_registered(self):
        """Every string switch in the shipped schema lists its choices."""
        import tomllib
        from pathlib import Path

        from mtc.core.param_overrides import _flatten_schema

        root = Path(__file__).resolve().parents[2]
        with open(root / "experiments" / "param_schema.toml", "rb") as f:
            flat = _flatten_schema(tomllib.load(f))
        for key in (
            "fep.world_model",
            "fep.meta_accuracy_method",
            "fep.drive_model",
            "blt.loop_quality_formula",
            "damasio.circadian",
        ):
            assert flat[key]["type"] == "str"
            assert flat[key]["default"] in flat[key]["choices"]

    def test_preset_layers_under_params(self, tmp_path):
        from mtc.core.param_overrides import load_layered_overrides

        schema_path = self._write_toml(
            tmp_path,
            "schema.toml",
            '[fep.planning_horizon]\ntype = "int"\ndefault = 5\nmin = 1\n'
            'max = 10\ndescription = "Horizon"\n'
            '[fep.num_actions]\ntype = "int"\ndefault = 4\nmin = 2\n'
            'max = 16\ndescription = "Actions"\n',
        )
        (tmp_path / "presets").mkdir()
        self._write_toml(
            tmp_path / "presets",
            "demo.toml",
            "[fep]\nplanning_horizon = 10\nnum_actions = 2\n",
        )
        params_path = self._write_toml(
            tmp_path, "params.toml", "[fep]\nplanning_horizon = 3\n"
        )
        result = load_layered_overrides(
            params_path,
            schema_path,
            preset="demo",
            presets_dir=str(tmp_path / "presets"),
        )
        assert result == {"fep": {"planning_horizon": 3, "num_actions": 2}}

    def test_unknown_preset_is_rejected(self, tmp_path):
        from mtc.core.param_overrides import load_layered_overrides

        schema_path = self._write_toml(tmp_path, "schema.toml", "")
        params_path = self._write_toml(tmp_path, "params.toml", "")
        (tmp_path / "presets").mkdir()
        with pytest.raises(ValueError, match="not found"):
            load_layered_overrides(
                params_path,
                schema_path,
                preset="nope",
                presets_dir=str(tmp_path / "presets"),
            )
        with pytest.raises(ValueError, match="Invalid preset name"):
            load_layered_overrides(
                params_path,
                schema_path,
                preset="../schema",
                presets_dir=str(tmp_path / "presets"),
            )

    def test_shipped_legacy_preset_validates(self):
        from pathlib import Path

        from mtc.core.param_overrides import load_layered_overrides

        root = Path(__file__).resolve().parents[2]
        result = load_layered_overrides(
            str(root / "experiments" / "params.toml"),
            str(root / "experiments" / "param_schema.toml"),
            preset="legacy_v04",
        )
        assert result["fep"]["drive_model"] == "v04"
        assert result["fep"]["legacy_double_decay"] is True
        assert result["blt"]["loop_quality_formula"] == "v04"
        assert result["damasio"]["circadian"] == "v04_sawtooth"
        # The ten switch rows added for v0.5.0 (section 7.1 of the fix plan)
        assert result["assessment"]["noise_baseline_matching"] == "v04_any"
        assert result["runner"]["assess_blt_source"] == "standalone"
        assert result["gwt"]["legacy_broadcast_receivers"] is True
        assert result["gwt"]["legacy_sustain_rate"] is True
        assert result["gwt"]["refractory_clock"] == "per_cycle"
        assert result["hot"]["overall_confidence_source"] == "v04_constant"
        assert result["hot"]["introspection_counting"] == "v04_per_hot"
        assert result["hot"]["introspection_cooldown"] == 1.0
        assert result["blt"]["binding_evidence"] == "v04_salience_posterior"
        assert result["blt"]["strange_loop_test"] == "v04_confidence_match"
        assert result["damasio"]["first_delta"] == "v04_zero"

    # The ten rows of section 7.1: (row, legacy value, v0.5.0 default)
    NEW_SWITCH_ROWS = (
        ("assessment.noise_baseline_matching", "v04_any", "producer"),
        ("runner.assess_blt_source", "standalone", "workspace"),
        ("gwt.legacy_broadcast_receivers", True, False),
        ("gwt.legacy_sustain_rate", True, False),
        ("gwt.refractory_clock", "per_cycle", "per_cycle"),
        ("hot.overall_confidence_source", "v04_constant", "hot_mean"),
        ("hot.introspection_counting", "v04_per_hot", "episodes"),
        ("blt.binding_evidence", "v04_salience_posterior", "candidate_vectors"),
        ("blt.strange_loop_test", "v04_confidence_match", "independent_calibration"),
        ("damasio.first_delta", "v04_zero", "unmeasured"),
    )

    def test_new_switch_rows_declare_both_values(self):
        import tomllib
        from pathlib import Path

        from mtc.core.param_overrides import _flatten_schema

        root = Path(__file__).resolve().parents[2]
        with open(root / "experiments" / "param_schema.toml", "rb") as f:
            flat = _flatten_schema(tomllib.load(f))
        for key, legacy, default in self.NEW_SWITCH_ROWS:
            row = flat[key]
            assert row["default"] == default, key
            assert row["description"], key
            if isinstance(legacy, bool):
                assert row["type"] == "bool", key
            else:
                assert row["type"] == "str", key
                assert {legacy, default} <= set(row["choices"]), key
                if legacy != default:
                    assert set(row["choices"]) == {legacy, default}, key
        # The refractory clock keeps its v0.4.0 arithmetic as the default
        # (amendment A8) and offers the elapsed clock as the other choice
        assert set(flat["gwt.refractory_clock"]["choices"]) == {"per_cycle", "elapsed"}
        assert "machine independent" in flat["gwt.refractory_clock"]["description"]

    def test_new_switch_rows_reject_a_bad_value(self, tmp_path):
        from pathlib import Path

        from mtc.core.param_overrides import load_overrides

        root = Path(__file__).resolve().parents[2]
        schema = str(root / "experiments" / "param_schema.toml")
        for key, legacy, default in self.NEW_SWITCH_ROWS:
            section, name = key.rsplit(".", 1)
            if isinstance(legacy, bool):
                bad = self._write_toml(
                    tmp_path, f"bad_{name}.toml", f'[{section}]\n{name} = "yes"\n'
                )
                with pytest.raises(ValueError, match="type"):
                    load_overrides(bad, schema)
                good_value = "true" if legacy else "false"
            else:
                bad = self._write_toml(
                    tmp_path, f"bad_{name}.toml", f'[{section}]\n{name} = "bogus"\n'
                )
                with pytest.raises(ValueError, match="not one of"):
                    load_overrides(bad, schema)
                good_value = f'"{legacy}"'
            good = self._write_toml(
                tmp_path, f"good_{name}.toml", f"[{section}]\n{name} = {good_value}\n"
            )
            assert load_overrides(good, schema)[section][name] == legacy

    def test_introspection_cooldown_accepts_the_runner_pin(self, tmp_path):
        """The schema minimum of hot.introspection_cooldown is 0.0: the value
        both runners pin; a negative cooldown is still rejected."""
        from pathlib import Path

        from mtc.core.param_overrides import load_overrides

        root = Path(__file__).resolve().parents[2]
        schema = str(root / "experiments" / "param_schema.toml")
        pinned = self._write_toml(
            tmp_path, "pinned.toml", "[hot]\nintrospection_cooldown = 0.0\n"
        )
        assert load_overrides(pinned, schema)["hot"]["introspection_cooldown"] == 0.0
        negative = self._write_toml(
            tmp_path, "negative.toml", "[hot]\nintrospection_cooldown = -0.1\n"
        )
        with pytest.raises(ValueError, match="out of bounds"):
            load_overrides(negative, schema)

    def test_single_level_loop_is_rejected_by_the_schema(self, tmp_path):
        """blt.num_levels needs at least two levels: precision balance is
        undefined for one (R6-17)."""
        from pathlib import Path

        from mtc.core.param_overrides import load_overrides

        root = Path(__file__).resolve().parents[2]
        schema = str(root / "experiments" / "param_schema.toml")
        one = self._write_toml(tmp_path, "one.toml", "[blt]\nnum_levels = 1\n")
        with pytest.raises(ValueError, match="out of bounds"):
            load_overrides(one, schema)
        two = self._write_toml(tmp_path, "two.toml", "[blt]\nnum_levels = 2\n")
        assert load_overrides(two, schema)["blt"]["num_levels"] == 2

    def test_voluntary_boost_row_says_it_has_no_effect(self):
        """ast.voluntary_boost is stored and has no effect in v0.5.0 (R2-N2):
        its schema description says so instead of promising a boost."""
        import tomllib
        from pathlib import Path

        from mtc.core.param_overrides import _flatten_schema

        root = Path(__file__).resolve().parents[2]
        with open(root / "experiments" / "param_schema.toml", "rb") as f:
            flat = _flatten_schema(tomllib.load(f))
        description = flat["ast.voluntary_boost"]["description"]
        assert "no effect" in description
        assert "v0.5.0" in description
        assert "applied" not in description.lower()

    def test_phase3_switches_validate(self, tmp_path):
        """The preset loads its Phase 3 values and each string switch rejects
        a bad choice."""
        from pathlib import Path

        from mtc.core.param_overrides import load_layered_overrides, load_overrides

        root = Path(__file__).resolve().parents[2]
        schema = str(root / "experiments" / "param_schema.toml")
        preset = load_layered_overrides(
            str(root / "experiments" / "params.toml"), schema, preset="legacy_v04"
        )
        assert preset["assessment"]["indicator_formulas"] == "v04"
        assert preset["assessment"]["missing_signal_policy"] == "v04_constants"
        assert preset["assessment"]["irreducibility_method"] == "spectral_entropy"
        assert preset["assessment"]["causal_emergence_threshold"] == 0.0
        assert preset["assessment"]["phi_source"] == "calculator"
        assert preset["pad"]["coherence_method"] == "v04_variance"
        assert preset["causal_emergence"]["tpm_row_order"] == "little_endian"
        assert preset["rpt"]["stdp_plasticity_enabled"] is False

        for section, key in (
            ("assessment", "indicator_formulas"),
            ("assessment", "missing_signal_policy"),
            ("assessment", "irreducibility_method"),
            ("assessment", "phi_source"),
            ("assessment", "phi_normalization"),
            ("pad", "coherence_method"),
            ("causal_emergence", "tpm_row_order"),
        ):
            bad = self._write_toml(
                tmp_path, f"bad_{key}.toml", f'[{section}]\n{key} = "bogus"\n'
            )
            with pytest.raises(ValueError, match="not one of"):
                load_overrides(bad, schema)

        for section, key, value in (
            ("assessment", "causal_emergence_threshold", 1.5),
            ("assessment", "phi_fixed_reference", 0.0),
            ("pad", "coherence_window", 2),
            ("pad", "coherence_min_history", 100),
        ):
            bad = self._write_toml(
                tmp_path, f"range_{key}.toml", f"[{section}]\n{key} = {value}\n"
            )
            with pytest.raises(ValueError, match="out of bounds"):
                load_overrides(bad, schema)

        good = self._write_toml(
            tmp_path,
            "good.toml",
            '[assessment]\nphi_normalization = "percentile"\n'
            "[rpt]\nstdp_plasticity_enabled = true\n",
        )
        assert load_overrides(good, schema)["assessment"]["phi_normalization"] == (
            "percentile"
        )
