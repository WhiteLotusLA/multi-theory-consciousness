"""The committed JSON schemas are generated from the onboarding models."""

import importlib.util
import json
from pathlib import Path

from mtc.onboarding.schema import export_json_schemas

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "docs" / "schemas"
SCRIPT = ROOT / "scripts" / "gen_schemas.py"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_schemas_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_schemas_match_models():
    documents = export_json_schemas()
    assert sorted(documents) == [
        "onboarding-answers.schema.json",
        "persona.schema.json",
    ]
    committed_names = sorted(path.name for path in SCHEMA_DIR.glob("*.json"))
    assert committed_names == sorted(documents)
    for name, document in documents.items():
        text = (SCHEMA_DIR / name).read_text(encoding="utf-8")
        assert json.loads(text) == json.loads(json.dumps(document)), name
        assert text.endswith("}\n"), name
        assert text == json.dumps(document, indent=2, sort_keys=True) + "\n", name
    persona = documents["persona.schema.json"]
    assert persona["title"] == "PersonaConfig"
    assert "core_memory" not in persona["required"]
    answers = documents["onboarding-answers.schema.json"]
    assert answers["properties"]["acknowledged"]["const"] is True
    assert "acknowledged" in answers["required"]


def test_gen_schemas_check_mode(tmp_path, capsys):
    generator = _generator()
    assert generator.main(["--check"]) == 0

    target = tmp_path / "schemas"
    # Nothing written yet: every file is drift, and the check writes nothing.
    assert generator.main(["--check", "--dir", str(target)]) == 1
    assert not target.exists()
    captured = capsys.readouterr()
    assert "DRIFT: persona.schema.json" in captured.out
    assert "regenerate" in captured.err.lower()
    assert "pydantic" in captured.err

    assert generator.main(["--dir", str(target)]) == 0
    assert generator.main(["--check", "--dir", str(target)]) == 0

    # Formatting alone is not drift (parsed JSON is compared) ...
    persona = target / "persona.schema.json"
    document = json.loads(persona.read_text(encoding="utf-8"))
    persona.write_text(json.dumps(document), encoding="utf-8")
    assert generator.main(["--check", "--dir", str(target)]) == 0
    # ... a changed schema is.
    document["title"] = "SomethingElse"
    persona.write_text(json.dumps(document), encoding="utf-8")
    assert generator.main(["--check", "--dir", str(target)]) == 1
    persona.write_text("{ not json", encoding="utf-8")
    assert generator.main(["--check", "--dir", str(target)]) == 1
