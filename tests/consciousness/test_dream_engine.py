"""Dream engine: YAML profiles and templates, selection, comfort bias, prompts."""

import random
import re
import string
from pathlib import Path

import pytest

from mtc.consciousness.dream_engine import (
    DEFAULT_CONFIG_PATH,
    DreamContext,
    DreamEngine,
    DreamPrompt,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "dreams.yaml"

MINIMAL_YAML = """
selection:
  history_size: 5
  repeat_window: 2
  repeat_penalty: 0.5
  comforting_types: [rest]
  comfort_multiplier: 4.0
profiles:
  default:
    parameters: {temperature: 0.5, max_tokens: 120}
    weights: {drift: 1.0}
  bands:
    - stages: [1, 10]
      parameters: {temperature: 0.6, max_tokens: 300}
      weights: {drift: 0.5, rest: 0.5}
templates:
  drift: "Drift dream about {hypothetical}."
  rest: "Rest dream."
fillers:
  hypothetical: "an open question"
  hypothetical_from_topic: "revisit {topic}"
grounding:
  topics: "Topics: {topics}."
  question: "Question: {question}."
  emotion: "Undertone: {emotion}."
textures:
  night: {token_scale: 1.0, min_tokens: 32, suffix: "Night texture."}
  nap: {token_scale: 0.5, min_tokens: 32, suffix: "Nap texture."}
system_prompt: "System dream prompt."
closing_instruction: "Closing line."
"""

# Phrasing that addresses the dreamer as a child or casts a caretaker.
NON_NEUTRAL = (
    "toddler",
    "child",
    "kid",
    "baby",
    "innocent",
    "little one",
    "loves you",
    "who loves",
    "mother",
    "father",
    "parent",
    "sweetheart",
    "good girl",
    "good boy",
    "tuck",
    "lullaby",
)


def _engine(**kwargs) -> DreamEngine:
    kwargs.setdefault("rng", random.Random(11))
    return DreamEngine(**kwargs)


def _all_text(engine: DreamEngine) -> list:
    cfg = engine.config
    texts = list(cfg.templates.values())
    texts += [cfg.system_prompt, cfg.closing_instruction]
    texts += [texture.suffix for texture in cfg.textures.values()]
    texts += list(cfg.fillers.values()) + list(cfg.grounding.values())
    return texts


def test_profiles_and_templates_load_from_yaml(tmp_path):
    assert DEFAULT_CONFIG_PATH == REPO_CONFIG
    shipped = _engine()
    cfg = shipped.config
    assert len(cfg.bands) == 4
    assert [(band.low, band.high) for band in cfg.bands] == [
        (1, 3),
        (4, 6),
        (7, 8),
        (9, 10),
    ]
    used = set(cfg.default.weights)
    for band in cfg.bands:
        used |= set(band.weights)
    used |= set(cfg.comforting_types)
    assert used <= set(cfg.templates)
    assert set(cfg.textures) == {"night", "nap"}

    path = tmp_path / "dreams.yaml"
    path.write_text(MINIMAL_YAML, encoding="utf-8")
    custom = _engine(config_path=path)
    assert set(custom.config.templates) == {"drift", "rest"}
    assert custom.select_dream_type() == "drift"
    prompt = custom.build_prompt(DreamContext([], None, None, mode="night"))
    assert prompt.system_prompt == "System dream prompt."
    assert prompt.max_tokens == 120


def test_missing_template_is_rejected(tmp_path):
    broken = MINIMAL_YAML.replace('  rest: "Rest dream."\n', "")
    path = tmp_path / "dreams.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="rest"):
        DreamEngine(config_path=path)


def test_default_profile_without_stage():
    engine = _engine()
    assert engine.stage is None
    default_types = set(engine.config.default.weights)
    seen = {engine.select_dream_type() for _ in range(200)}
    assert seen <= default_types
    assert len(seen) > 1
    params = engine.parameters()
    assert params == {
        "temperature": engine.config.default.temperature,
        "max_tokens": engine.config.default.max_tokens,
    }


@pytest.mark.parametrize(
    "stage,band_index", [(1, 0), (2, 0), (5, 1), (8, 2), (9, 3), (10, 3)]
)
def test_stage_selects_band_profile(stage, band_index):
    engine = _engine(stage=stage)
    band = engine.config.bands[band_index]
    seen = {engine.select_dream_type() for _ in range(200)}
    assert seen <= set(band.weights)
    assert engine.parameters()["max_tokens"] == band.max_tokens
    assert engine.parameters()["temperature"] == band.temperature


def test_stage_is_clamped_and_settable():
    engine = _engine(stage=0)
    assert engine.stage == 1
    engine.set_stage(42)
    assert engine.stage == 10
    engine.set_stage(None)
    assert engine.stage is None


def test_repeat_avoidance_halves_recent_types():
    engine = _engine(stage=8)
    band = engine.config.bands[2]
    assert engine.type_weights() == band.weights
    history = [
        "abstract_integration",
        "symbolic_processing",
        "creative_synthesis",
        "meta_awareness",
        "emotional_resolution",
    ]
    engine.from_state_dict(
        {"scalars": {}, "arrays": {}, "histories": {"dream_types": history}}
    )
    assert engine.config.repeat_window == 3
    recent = set(history[-3:])
    weights = engine.type_weights()
    for name, weight in band.weights.items():
        factor = engine.config.repeat_penalty if name in recent else 1.0
        assert weights[name] == pytest.approx(weight * factor), name
    assert engine.config.repeat_penalty == 0.5


def test_selection_appends_one_history_entry():
    engine = _engine(stage=4)
    engine.select_dream_type()
    engine.select_dream_type(comfort_bias=True)
    assert len(engine.recent_types) == 2


def test_explicit_type_does_not_roll():
    engine = _engine(stage=10)
    prompt = engine.build_prompt(
        DreamContext([], None, None, mode="night"), dream_type="lucid"
    )
    assert prompt.dream_type == "lucid"
    assert "lucid" in prompt.prompt.lower()
    assert engine.recent_types == []


def test_comfort_bias_off_by_default():
    engine = _engine(stage=8)
    assert engine.comfort_bias is False
    weights = engine.type_weights()
    assert "comfort_dream" not in weights
    assert weights == engine.config.bands[2].weights


def test_comfort_bias_forces_comforting_candidates():
    engine = _engine(stage=8, comfort_bias=True)
    band = engine.config.bands[2]
    assert "comfort_dream" not in band.weights
    weights = engine.type_weights()
    multiplier = engine.config.comfort_multiplier
    mean = sum(band.weights.values()) / len(band.weights)
    assert weights["comfort_dream"] == pytest.approx(mean * multiplier)
    assert weights["emotional_resolution"] == pytest.approx(
        band.weights["emotional_resolution"] * multiplier
    )
    chosen = [engine.select_dream_type() for _ in range(200)]
    comforting = set(engine.config.comforting_types)
    assert sum(1 for name in chosen if name in comforting) > 120
    assert any(name not in comforting for name in chosen)
    assert "comfort_dream" in chosen

    plain = _engine(stage=8)
    assert "comfort_dream" in plain.type_weights(comfort_bias=True)


def test_prompt_grounded_in_context():
    engine = _engine(stage=5)
    context = DreamContext(
        recent_topics=["tide pools", "a chess endgame", "rain on glass", "maps"],
        dominant_emotion="curious",
        unresolved_question="how memory decides what to keep",
        mode="night",
    )
    prompt = engine.build_prompt(context, dream_type="what_if")
    assert isinstance(prompt, DreamPrompt)
    assert "tide pools, a chess endgame, rain on glass" in prompt.prompt
    assert "maps" not in prompt.prompt
    assert "how memory decides what to keep" in prompt.prompt
    assert "Emotional undertone: curious." in prompt.prompt
    assert prompt.prompt.rstrip().endswith(engine.config.closing_instruction)

    bare = engine.build_prompt(
        DreamContext([], None, None, mode="night"), dream_type="exploration"
    )
    assert "exploration" not in engine.config.emotion_types
    assert "Recent experiences" not in bare.prompt
    assert "Something unresolved" not in bare.prompt
    assert "Emotional undertone" not in bare.prompt
    assert "feeling that has no name" not in bare.prompt
    assert "{" not in bare.prompt


def test_nap_and_night_textures_differ():
    engine = _engine(stage=7)
    night = engine.build_prompt(
        DreamContext([], None, None, mode="night"), dream_type="memory_consolidation"
    )
    nap = engine.build_prompt(
        DreamContext([], None, None, mode="nap"), dream_type="memory_consolidation"
    )
    assert night.max_tokens == engine.config.bands[2].max_tokens
    assert nap.max_tokens < night.max_tokens
    assert engine.config.textures["nap"].suffix in nap.prompt
    assert engine.config.textures["night"].suffix in night.prompt
    assert engine.config.textures["nap"].suffix not in night.prompt
    early_nap = _engine(stage=2).build_prompt(
        DreamContext([], None, None, mode="nap"), dream_type="comfort_dream"
    )
    late_nap = _engine(stage=9).build_prompt(
        DreamContext([], None, None, mode="nap"), dream_type="lucid"
    )
    assert early_nap.max_tokens < late_nap.max_tokens
    with pytest.raises(ValueError):
        DreamContext([], None, None, mode="siesta")


def test_seeded_rng_is_deterministic():
    first = DreamEngine(stage=6, rng=random.Random(5))
    second = DreamEngine(stage=6, rng=random.Random(5))
    assert [first.select_dream_type() for _ in range(30)] == [
        second.select_dream_type() for _ in range(30)
    ]


def test_state_dict_round_trip():
    engine = _engine(stage=3)
    for _ in range(4):
        engine.select_dream_type()
    state = engine.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    restored = _engine(stage=3)
    restored.from_state_dict(state)
    assert restored.recent_types == engine.recent_types
    assert restored.type_weights() == engine.type_weights()


def test_templates_are_neutral():
    engine = _engine()
    placeholders = {"dominant_emotion", "hypothetical"}
    for name, template in engine.config.templates.items():
        fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
        assert fields <= placeholders, name
    for text in _all_text(engine):
        lowered = text.lower()
        for phrase in NON_NEUTRAL:
            assert not re.search(rf"\b{re.escape(phrase)}", lowered), (phrase, text)
        assert "\u2013" not in text and "\u2014" not in text
        assert text.isascii()
    raw = REPO_CONFIG.read_text(encoding="utf-8")
    assert "\u2013" not in raw and "\u2014" not in raw


def _profiles(engine: DreamEngine):
    """Every profile of the config, each as a stage to pass (None: default)."""
    return [None] + [band.low for band in engine.config.bands]


def _emotion_types(engine: DreamEngine) -> set:
    return {
        name
        for name, template in engine.config.templates.items()
        if "{dominant_emotion}" in template
    }


def test_context_without_emotion_never_draws_an_emotion_type():
    engine = _engine()
    needing = _emotion_types(engine)
    assert needing == {"emotional_echo", "emotional_processing", "emotional_resolution"}
    assert set(engine.config.emotion_types) == needing
    for stage in _profiles(engine):
        for gentle in (None, True):
            bare = _engine(stage=stage, rng=random.Random(7))
            drawn = {
                bare.build_prompt(
                    DreamContext([], None, None, mode="night"), comfort_bias=gentle
                ).dream_type
                for _ in range(2000)
            }
            assert not drawn & needing, (stage, gentle, drawn & needing)
            assert len(drawn) > 1, (stage, gentle)
    # with a measured emotion they are drawn again
    seen = set()
    for stage in _profiles(engine):
        felt = _engine(stage=stage, rng=random.Random(7))
        for _ in range(2000):
            prompt = felt.build_prompt(DreamContext([], "curious", None, mode="night"))
            seen.add(prompt.dream_type)
            if prompt.dream_type in needing:
                assert "curious" in prompt.prompt
    assert needing <= seen


def test_explicit_emotion_type_without_emotion_is_refused():
    engine = _engine(stage=5)
    with pytest.raises(ValueError, match="emotional_processing.*measured emotion"):
        engine.build_prompt(
            DreamContext([], None, None, mode="night"),
            dream_type="emotional_processing",
        )
    with pytest.raises(ValueError, match="measured emotion"):
        engine.build_prompt(
            DreamContext([], "   ", None, mode="night"), dream_type="emotional_echo"
        )
    prompt = engine.build_prompt(
        DreamContext([], "uneasy", None, mode="night"),
        dream_type="emotional_processing",
    )
    assert "an unsettled feeling: uneasy." in prompt.prompt


def test_draw_with_no_type_left_is_refused(tmp_path):
    only_feelings = MINIMAL_YAML.replace(
        '  drift: "Drift dream about {hypothetical}."',
        '  drift: "Drift dream through {dominant_emotion}."',
    ).replace('  rest: "Rest dream."', '  rest: "Rest in {dominant_emotion}."')
    path = tmp_path / "dreams.yaml"
    path.write_text(only_feelings, encoding="utf-8")
    engine = _engine(config_path=path)
    with pytest.raises(ValueError, match="profiles.default"):
        engine.build_prompt(DreamContext([], None, None, mode="night"))
    engine.set_stage(4)
    with pytest.raises(ValueError, match="stages 1-10"):
        engine.build_prompt(DreamContext([], None, None, mode="night"))
    assert engine.recent_types == []  # a refused draw records nothing
    felt = engine.build_prompt(DreamContext([], "calm", None, mode="night"))
    assert felt.dream_type in {"drift", "rest"}


def test_dominant_emotion_filler_is_refused(tmp_path):
    raw = REPO_CONFIG.read_text(encoding="utf-8")
    assert "dominant_emotion:" not in raw  # the shipped file carries no filler
    assert "feeling that has no name" not in raw
    filled = MINIMAL_YAML.replace(
        "fillers:\n", 'fillers:\n  dominant_emotion: "an unnamed feeling"\n'
    )
    path = tmp_path / "dreams.yaml"
    path.write_text(filled, encoding="utf-8")
    with pytest.raises(ValueError, match="fillers.dominant_emotion"):
        DreamEngine(config_path=path)
