"""Emotion extractor: raw values from state and drives, relative rules."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from mtc.consciousness.emotion_extractor import (
    DEFAULT_LABELS,
    EmotionAveragesProvider,
    EmotionExtractor,
    EmotionVocabulary,
    InMemoryEmotionAverages,
)


def make_state(valence=0.0, arousal=0.5, vitality=0.5, depth=0, markers=None):
    damasio = {
        "protoself": {
            "primordial_feelings": {
                "pleasure_pain": valence,
                "arousal": arousal,
                "vitality": vitality,
            }
        },
        "core": {"somatic_markers": markers or []},
    }
    return SimpleNamespace(
        damasio_state=damasio, protoself_stability=0.7, epistemic_depth=depth
    )


def make_drives(**urgencies):
    names = (
        "attention_budget",
        "curiosity",
        "social_connection",
        "coherence",
        "safety",
    )

    def provider():
        return {
            name: {
                "current_level": 0.5,
                "optimal_level": 0.5,
                "deviation": 0.0,
                "urgency": urgencies.get(name, 0.0),
            }
            for name in names
        }

    return provider


async def prime(avg, session, n, **dims):
    for _ in range(n):
        await avg.record(session, dict(dims))


@pytest.mark.asyncio
async def test_cold_start_uses_neutral_prior_and_says_so():
    ext = EmotionExtractor(min_baseline_samples=20)
    out = await ext.extract(make_state(valence=0.4, arousal=0.8, vitality=0.9))
    assert out.baseline == "prior"
    assert out.label is not None
    # one sample recorded after this call; scaled by 0 samples seen before it
    assert out.confidence == pytest.approx(0.0)
    avg = InMemoryEmotionAverages()
    await prime(avg, "default", 10, valence=0.0, arousal=0.5, vitality=0.5)
    ext = EmotionExtractor(averages=avg, min_baseline_samples=20)
    out = await ext.extract(make_state(valence=0.4, arousal=0.8, vitality=0.9))
    assert out.baseline == "prior"
    assert 0.0 < out.confidence <= 0.5


@pytest.mark.asyncio
async def test_running_baseline_after_min_samples():
    avg = InMemoryEmotionAverages()
    await prime(avg, "default", 20, valence=0.0, arousal=0.5, vitality=0.5)
    ext = EmotionExtractor(averages=avg, min_baseline_samples=20)
    out = await ext.extract(make_state(valence=0.4, arousal=0.8, vitality=0.9))
    assert out.baseline == "running"
    assert out.label in DEFAULT_LABELS


@pytest.mark.asyncio
async def test_relative_rules_use_running_average():
    state = make_state(valence=-0.3, arousal=0.2, vitality=0.4)
    # Prior valence mean 0.0 with spread 0.25: -0.3 is far below -> sad.
    cold = await EmotionExtractor().extract(state)
    assert cold.label == "sad"
    # A habitually negative running baseline makes the same value unremarkable.
    avg = InMemoryEmotionAverages()
    for v in (-0.35, -0.25, -0.3, -0.32, -0.28) * 4:
        await avg.record("default", {"valence": v, "arousal": 0.2, "vitality": 0.4})
    warm = await EmotionExtractor(averages=avg).extract(state)
    assert warm.baseline == "running"
    assert warm.label != "sad"


@pytest.mark.asyncio
async def test_missing_damasio_fields_are_excluded_not_defaulted():
    avg = InMemoryEmotionAverages()
    state = SimpleNamespace(
        damasio_state=None, protoself_stability=0.5, epistemic_depth=0
    )
    ext = EmotionExtractor(averages=avg)
    out = await ext.extract(state)
    assert out.raw["valence"] is None
    assert out.raw["arousal"] is None
    assert out.raw["vitality"] is None
    # No measured dimension at all: no label and zero confidence.
    assert out.label is None
    assert out.confidence == 0.0
    # Nothing was recorded for the missing dimensions.
    assert await avg.get_averages("default") is None or "valence" not in (
        await avg.get_averages("default")
    )
    # A negative-valence rule cannot fire without valence, even with drives.
    out = await EmotionExtractor(drives=make_drives(curiosity=0.8)).extract(state)
    assert out.label == "curious"


@pytest.mark.asyncio
async def test_drive_urgency_shapes_label():
    state = make_state(valence=0.0, arousal=0.5, vitality=0.5)
    out = await EmotionExtractor(drives=make_drives(curiosity=0.8)).extract(state)
    assert out.label in ("curious", "fascinated")
    out = await EmotionExtractor(drives=make_drives(safety=0.7)).extract(state)
    assert out.label == "uneasy"
    out = await EmotionExtractor(drives=make_drives(social_connection=0.9)).extract(
        state
    )
    assert out.label == "loving"


@pytest.mark.asyncio
async def test_drive_read_failure_is_warned_not_zeroed(caplog):
    def broken():
        raise RuntimeError("down")

    with caplog.at_level("WARNING"):
        out = await EmotionExtractor(drives=broken).extract(make_state())
    assert out.raw["drive_curiosity"] is None
    assert "drive" in caplog.text.lower()


@pytest.mark.asyncio
async def test_vocabulary_configurable():
    vocab = EmotionVocabulary(labels=("sad", "neutral"))
    out = await EmotionExtractor(
        vocabulary=vocab, drives=make_drives(curiosity=0.9)
    ).extract(make_state())
    assert out.label == "neutral"  # curious is outside the vocabulary
    none_vocab = EmotionVocabulary(labels=("sad",))
    out = await EmotionExtractor(vocabulary=none_vocab).extract(make_state())
    assert out.label is None


@pytest.mark.asyncio
async def test_no_subtitles_by_default():
    out = await EmotionExtractor().extract(make_state(valence=0.4, arousal=0.8))
    assert out.subtitle is None
    assert EmotionVocabulary(labels=("a",)).subtitles == {}
    vocab = EmotionVocabulary(labels=("neutral",), subtitles={"neutral": "plain"})
    out = await EmotionExtractor(vocabulary=vocab).extract(make_state())
    assert out.subtitle == "plain"


def test_default_vocabulary_has_twenty_labels():
    assert len(DEFAULT_LABELS) == 20
    assert len(set(DEFAULT_LABELS)) == 20


@pytest.mark.asyncio
async def test_in_memory_averages_contract():
    avg = InMemoryEmotionAverages()
    assert isinstance(avg, EmotionAveragesProvider)
    assert await avg.get_averages("s") is None
    assert await avg.sample_count("s") == 0
    await avg.record("s", {"valence": 0.2})
    await avg.record("s", {"valence": 0.6, "arousal": 0.5})
    got = await avg.get_averages("s")
    assert got["valence"] == pytest.approx(0.4)
    assert got["valence_std"] > 0
    assert got["arousal"] == pytest.approx(0.5)
    assert await avg.sample_count("s") == 2
    assert await avg.sample_count("other") == 0


@pytest.mark.asyncio
async def test_state_dict_round_trip():
    ext = EmotionExtractor()
    await ext.extract(make_state(valence=0.3), session_id="a")
    await ext.extract(make_state(valence=0.5), session_id="a")
    state = ext.to_state_dict()
    fresh = EmotionExtractor()
    fresh.from_state_dict(state)
    assert await fresh._averages.get_averages("a") == await ext._averages.get_averages(
        "a"
    )
    assert await fresh._averages.sample_count("a") == 2


def test_module_imports_no_redis():
    code = (
        "import sys; import mtc.consciousness.emotion_extractor; "
        "sys.exit(1 if 'redis' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.asyncio
async def test_empty_samples_are_not_counted_and_never_gain_confidence():
    avg = InMemoryEmotionAverages()
    state = SimpleNamespace(
        damasio_state=None, protoself_stability=0.5, epistemic_depth=0
    )
    ext = EmotionExtractor(averages=avg, min_baseline_samples=20)
    for _ in range(30):
        out = await ext.extract(state)
        assert out.label is None
        assert out.confidence == 0.0
        assert out.baseline == "prior"
    assert await avg.sample_count("default") == 0
    await avg.record("default", {"valence": None})
    assert await avg.sample_count("default") == 0


@pytest.mark.asyncio
async def test_fallback_label_has_zero_confidence():
    """No rule fired: the fallback label carries no confidence and no runner-up."""
    avg = InMemoryEmotionAverages()
    await prime(avg, "default", 25, valence=0.0, arousal=0.5, vitality=0.5)
    ext = EmotionExtractor(averages=avg)
    out = await ext.extract(make_state(valence=0.0, arousal=0.5, vitality=0.5))
    assert out.baseline == "running"
    assert out.label == "neutral"
    assert out.secondary is None
    assert out.confidence == 0.0


@pytest.mark.asyncio
async def test_single_rule_margin_keeps_the_fallback_as_runner_up():
    """One rule at 0.45 against the fallback at 0.1: margin (0.45 - 0.1) / 0.45."""
    avg = InMemoryEmotionAverages()
    await prime(avg, "default", 25, valence=-0.2, arousal=0.5, vitality=0.5)
    ext = EmotionExtractor(averages=avg)
    out = await ext.extract(make_state(valence=-0.1, arousal=0.5, vitality=0.6))
    assert out.baseline == "running"
    assert out.label == "calm"
    assert out.secondary == "neutral"
    assert out.confidence == pytest.approx((0.45 - 0.1) / 0.45)
    assert out.confidence == pytest.approx(0.778, abs=1e-3)


@pytest.mark.asyncio
async def test_somatic_intensity_is_the_mean_of_the_measured_intensities():
    markers = [
        {"valence": 0.2, "intensity": None},
        {"valence": 0.4, "intensity": 0.6},
        {"valence": -0.3, "intensity": float("nan")},
        {"valence": 0.5, "intensity": 0.2},
    ]
    ext = EmotionExtractor()
    raw = ext._extract_raw(make_state(markers=markers))
    assert raw["somatic_intensity_avg"] == pytest.approx(0.4)
    assert raw["somatic_valence_avg"] == pytest.approx(0.8 / 4)
    out = await ext.extract(make_state(markers=markers))
    assert out.raw["somatic_intensity_avg"] == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_somatic_intensity_is_none_when_no_marker_has_one():
    markers = [
        {"valence": 0.2, "intensity": None},
        {"valence": -0.4, "intensity": None},
    ]
    ext = EmotionExtractor()
    raw = ext._extract_raw(make_state(markers=markers))
    assert raw["somatic_intensity_avg"] is None
    assert raw["somatic_valence_avg"] == pytest.approx(-0.1)
    out = await ext.extract(make_state(valence=0.4, arousal=0.8, markers=markers))
    assert out.raw["somatic_intensity_avg"] is None
    assert isinstance(out.label, str) and out.label
