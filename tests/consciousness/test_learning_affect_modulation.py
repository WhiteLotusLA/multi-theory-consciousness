"""Affect-modulated learning rate: inverted-U arousal, significance, clamps."""

from types import SimpleNamespace

import pytest

from mtc.consciousness.learning.affect_modulation import AffectModulatedRate
from mtc.consciousness.pad_affect import PADState


def pad(pleasure=0.0, arousal=0.6, dominance=1.0) -> PADState:
    return PADState(pleasure=pleasure, arousal=arousal, dominance=dominance)


def test_rate_peaks_near_arousal_point_six():
    modulator = AffectModulatedRate(base_rate=0.02)
    at_peak = modulator.rate(pad(arousal=0.6))
    assert at_peak == pytest.approx(0.02)
    for arousal in (0.1, 0.3, 0.9, 1.0):
        assert modulator.rate(pad(arousal=arousal)) < at_peak
    near = modulator.rate(pad(arousal=0.55))
    far = modulator.rate(pad(arousal=0.2))
    assert near > far


def test_valence_magnitude_boosts_rate():
    modulator = AffectModulatedRate(base_rate=0.02)
    flat = modulator.rate(pad(pleasure=0.0))
    assert modulator.rate(pad(pleasure=0.8)) > flat
    assert modulator.rate(pad(pleasure=-0.8)) == pytest.approx(
        modulator.rate(pad(pleasure=0.8))
    )
    assert modulator.rate(pad(pleasure=1.0)) == pytest.approx(0.02 * 1.6)


def test_rate_clamped_to_bounds():
    base = 0.02
    modulator = AffectModulatedRate(base_rate=base)
    assert AffectModulatedRate.MIN_MULT == 0.3 and AffectModulatedRate.MAX_MULT == 3.0
    grid = [
        modulator.rate(pad(p, a, d))
        for p in (-1.0, 0.0, 1.0)
        for a in (0.0, 0.3, 0.6, 1.0)
        for d in (0.0, 1.0)
    ]
    assert min(grid) == pytest.approx(base * 0.3)
    assert max(grid) <= base * 3.0 + 1e-12
    # out-of-range inputs are clamped before the curve is applied
    wild = modulator.rate(pad(pleasure=9.0, arousal=0.6, dominance=1.0))
    assert wild == pytest.approx(modulator.rate(pad(pleasure=1.0)))


def test_retention_clamped():
    modulator = AffectModulatedRate(base_rate=0.02)
    assert modulator.retention(pad(pleasure=0.0, arousal=0.0)) == pytest.approx(0.5)
    assert modulator.retention(pad(pleasure=1.0, arousal=1.0)) == pytest.approx(2.0)
    assert modulator.retention(pad(pleasure=-5.0, arousal=7.0)) == pytest.approx(2.0)
    mid = modulator.retention(pad(pleasure=0.5, arousal=0.5))
    assert 0.5 < mid < 2.0


def test_base_rate_defaults_to_settings(monkeypatch):
    from mtc.consciousness.learning import affect_modulation

    monkeypatch.setattr(
        affect_modulation, "get_settings", lambda: SimpleNamespace(LEARNING_RATE=0.5)
    )
    modulator = AffectModulatedRate()
    assert modulator.rate(pad(arousal=0.6)) == pytest.approx(0.5)
    monkeypatch.setattr(
        affect_modulation, "get_settings", lambda: SimpleNamespace(LEARNING_RATE=0.1)
    )
    assert modulator.rate(pad(arousal=0.6)) == pytest.approx(0.1)


def test_missing_affect_leaves_rate_and_retention_unmodulated():
    modulator = AffectModulatedRate(base_rate=0.04)
    assert modulator.rate(None) == pytest.approx(0.04)
    assert modulator.retention(None) == pytest.approx(1.0)
