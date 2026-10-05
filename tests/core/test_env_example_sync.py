"""The committed .env.example and docs/CONFIGURATION.md are generated artifacts."""

from pathlib import Path

from mtc.core.config import Settings
from mtc.core.env_example import render_configuration_md, render_env_example

ROOT = Path(__file__).resolve().parents[2]


def test_env_example_matches_settings():
    assert (ROOT / ".env.example").read_text() == render_env_example(Settings)


def test_configuration_doc_matches_settings():
    assert (ROOT / "docs" / "CONFIGURATION.md").read_text() == render_configuration_md(
        Settings
    )


def test_every_rendered_key_is_a_settings_field():
    rendered = render_env_example(Settings)
    keys = [
        line.split("=", 1)[0]
        for line in rendered.splitlines()
        if line and not line.startswith("#")
    ]
    fields = Settings.model_fields
    for key in keys:
        name = key[len("MTC_") :] if key.startswith("MTC_") else key
        assert name in fields, key


def test_secrets_render_empty():
    for line in render_env_example(Settings).splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0]
        if "PASSWORD" in key or "API_KEY" in key:
            assert line.endswith("="), line
