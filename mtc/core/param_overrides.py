"""
Parameter override layer for AutoConsciousness experiments.

Loads experiments/params.toml, validates against experiments/param_schema.toml,
returns a nested dict of validated overrides. Used by run_experiment.py to
thread overrides into module constructors.
"""

import tomllib
from pathlib import Path
from typing import Any


_TYPE_MAP = {
    "int": int,
    "float": (int, float),
    "bool": bool,
    "str": str,
}


def _flatten_schema(schema: dict, prefix: str = "") -> dict[str, dict]:
    """Flatten nested TOML schema into dot-separated keys."""
    flat = {}
    for key, value in schema.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "type" in value:
            flat[full_key] = value
        elif isinstance(value, dict):
            flat.update(_flatten_schema(value, full_key))
    return flat


def _flatten_params(params: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested TOML params into dot-separated keys."""
    flat = {}
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _unflatten(flat: dict[str, Any]) -> dict:
    """Convert dot-separated keys back to nested dict."""
    result = {}
    for dotted_key, value in flat.items():
        parts = dotted_key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


def load_overrides(params_path: str, schema_path: str) -> dict:
    """Load and validate parameter overrides.

    Args:
        params_path: Path to params.toml (agent-edited overrides)
        schema_path: Path to param_schema.toml (bounds and types)

    Returns:
        Nested dict of validated overrides (only keys present in params.toml)

    Raises:
        ValueError: If a param is not in schema, wrong type, or out of bounds
    """
    with open(schema_path, "rb") as f:
        raw_schema = tomllib.load(f)
    with open(params_path, "rb") as f:
        raw_params = tomllib.load(f)

    schema = _flatten_schema(raw_schema)
    params = _flatten_params(raw_params)

    if not params:
        return {}

    validated = {}
    for dotted_key, value in params.items():
        if dotted_key not in schema:
            raise ValueError(
                f"Parameter '{dotted_key}' not in schema. "
                f"Valid keys: {sorted(schema.keys())}"
            )

        spec = schema[dotted_key]
        expected_type = spec["type"]
        py_types = _TYPE_MAP.get(expected_type)
        if py_types is None:
            raise ValueError(f"Unknown type '{expected_type}' for '{dotted_key}'")

        if not isinstance(value, py_types):
            raise ValueError(
                f"Parameter '{dotted_key}' has type {type(value).__name__}, "
                f"expected {expected_type}"
            )

        if expected_type == "float" and isinstance(value, int):
            value = float(value)

        if "min" in spec and value < spec["min"]:
            raise ValueError(
                f"Parameter '{dotted_key}' = {value} out of bounds "
                f"(min: {spec['min']})"
            )
        if "max" in spec and value > spec["max"]:
            raise ValueError(
                f"Parameter '{dotted_key}' = {value} out of bounds "
                f"(max: {spec['max']})"
            )

        validated[dotted_key] = value

    return _unflatten(validated)
