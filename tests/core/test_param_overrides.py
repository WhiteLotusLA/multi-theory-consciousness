import pytest
from pathlib import Path


class TestParamOverrides:
    def _write_toml(self, tmp_path, filename, content):
        filepath = tmp_path / filename
        filepath.write_text(content)
        return str(filepath)

    def test_load_empty_overrides(self, tmp_path):
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n')
        params_path = self._write_toml(tmp_path, "params.toml", "# empty\n")
        result = load_overrides(params_path, schema_path)
        assert result == {}

    def test_load_valid_override(self, tmp_path):
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n')
        params_path = self._write_toml(tmp_path, "params.toml", "[gwt]\ncapacity = 5\n")
        result = load_overrides(params_path, schema_path)
        assert result == {"gwt": {"capacity": 5}}

    def test_reject_out_of_bounds(self, tmp_path):
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n')
        params_path = self._write_toml(tmp_path, "params.toml", "[gwt]\ncapacity = 50\n")
        with pytest.raises(ValueError, match="out of bounds"):
            load_overrides(params_path, schema_path)

    def test_reject_unknown_key(self, tmp_path):
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n')
        params_path = self._write_toml(tmp_path, "params.toml", "[gwt]\nnonexistent_param = 42\n")
        with pytest.raises(ValueError, match="not in schema"):
            load_overrides(params_path, schema_path)

    def test_reject_wrong_type(self, tmp_path):
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", '[gwt.capacity]\ntype = "int"\ndefault = 7\nmin = 1\nmax = 20\ndescription = "Workspace capacity"\n')
        params_path = self._write_toml(tmp_path, "params.toml", "[gwt]\ncapacity = 5.5\n")
        with pytest.raises(ValueError, match="type"):
            load_overrides(params_path, schema_path)

    def test_nested_drives(self, tmp_path):
        from mtc.core.param_overrides import load_overrides
        schema_path = self._write_toml(tmp_path, "schema.toml", '[fep.drives.curiosity.optimal_level]\ntype = "float"\ndefault = 0.5\nmin = 0.0\nmax = 1.0\ndescription = "Curiosity target level"\n')
        params_path = self._write_toml(tmp_path, "params.toml", "[fep.drives.curiosity]\noptimal_level = 0.7\n")
        result = load_overrides(params_path, schema_path)
        assert result["fep"]["drives"]["curiosity"]["optimal_level"] == 0.7
