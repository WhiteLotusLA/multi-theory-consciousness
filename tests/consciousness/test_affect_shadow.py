"""Affect shadow taps: write estimates and PAD context, never raise."""

import json
import re

from mtc.consciousness.affect_estimator import AffectEstimate
from mtc.consciousness.affect_shadow import (
    SIGNAL_ID,
    log_thought_affect_shadow,
    log_turn_affect_shadow,
    parse_annotation,
)
from mtc.consciousness.pad_affect import PADState
from mtc.validation.shadow_log import ShadowSignalLogger


class FakeEstimator:
    def __init__(self):
        self.calls = []

    def estimate(self, text, source_kind="response", annotation=None):
        self.calls.append((text, source_kind, annotation))
        if not text:
            return None
        return AffectEstimate(
            valence=0.25,
            arousal=-0.5,
            confidence=0.6,
            source_kind=source_kind,
            computed_from=[f"text:{source_kind}", "anchors:v1"],
        )


class PlainEstimator:
    """An estimator without an annotation parameter."""

    def estimate(self, text, source_kind="response"):
        return AffectEstimate(0.0, 0.0, 0.3, source_kind, ["x"])


def read(tmp_path):
    path = tmp_path / f"{SIGNAL_ID}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_turn_tap_writes_estimates_and_pad(tmp_path):
    log = ShadowSignalLogger(SIGNAL_ID, tmp_path)
    pad = PADState(pleasure=0.1, arousal=0.6, dominance=0.4)
    log_turn_affect_shadow(
        "a reply", "a question", pad, estimator=FakeEstimator(), logger=log
    )
    rows = read(tmp_path)
    assert [r["context"]["source_kind"] for r in rows] == ["response", "received"]
    first = rows[0]
    assert first["values"] == {"valence": 0.25, "arousal": -0.5}
    assert first["confidence"] == 0.6
    assert first["context"]["pad_pleasure"] == 0.1
    assert first["context"]["pad_dominance"] == 0.4
    assert "anchors:v1" in first["computed_from"]
    assert first["signal_id"] == "affect_estimator_v1"


def test_turn_tap_reads_mapping_pad_and_skips_empty_received(tmp_path):
    log = ShadowSignalLogger(SIGNAL_ID, tmp_path)
    log_turn_affect_shadow(
        "a reply",
        None,
        {"pleasure": 0.2, "arousal": 0.3, "dominance": 0.9},
        estimator=FakeEstimator(),
        logger=log,
    )
    rows = read(tmp_path)
    assert len(rows) == 1
    assert rows[0]["context"]["pad_dominance"] == 0.9


def test_thought_tap_writes_ids(tmp_path):
    log = ShadowSignalLogger(SIGNAL_ID, tmp_path)
    log_thought_affect_shadow(
        "a musing", "t-1", "reflection", estimator=FakeEstimator(), logger=log
    )
    (row,) = read(tmp_path)
    assert row["context"] == {
        "source_kind": "thought",
        "thought_id": "t-1",
        "thought_type": "reflection",
    }


def test_empty_thought_writes_nothing(tmp_path):
    log = ShadowSignalLogger(SIGNAL_ID, tmp_path)
    log_thought_affect_shadow("", estimator=FakeEstimator(), logger=log)
    assert read(tmp_path) == []


def test_taps_never_raise(tmp_path, caplog):
    class Exploding:
        def estimate(self, *args, **kwargs):
            raise RuntimeError("boom")

    log = ShadowSignalLogger(SIGNAL_ID, tmp_path)
    with caplog.at_level("WARNING"):
        log_turn_affect_shadow("x", "y", None, estimator=Exploding(), logger=log)
        log_thought_affect_shadow("x", estimator=Exploding(), logger=log)
    assert "affect shadow failed" in caplog.text
    assert read(tmp_path) == []


def test_annotation_parsing_off_by_default(tmp_path):
    log = ShadowSignalLogger(SIGNAL_ID, tmp_path)
    est = FakeEstimator()
    text = "[tone: warm] hello"
    log_turn_affect_shadow("r", text, estimator=est, logger=log)
    assert est.calls[1] == (text, "received", None)

    est = FakeEstimator()
    pattern = re.compile(r"\[tone: (\w+)\]")
    log_turn_affect_shadow(
        "r", text, estimator=est, logger=log, annotation_pattern=pattern
    )
    assert est.calls[1] == (text, "received", "warm")
    rows = read(tmp_path)
    assert rows[-1]["context"]["annotation"] == "warm"

    est = FakeEstimator()
    log_turn_affect_shadow(
        "r",
        text,
        annotation="given",
        estimator=est,
        logger=log,
        annotation_pattern=pattern,
    )
    assert est.calls[1][2] == "given"


def test_estimator_without_annotation_parameter_still_works(tmp_path):
    log = ShadowSignalLogger(SIGNAL_ID, tmp_path)
    log_turn_affect_shadow("r", "q", estimator=PlainEstimator(), logger=log)
    assert len(read(tmp_path)) == 2


def test_parse_annotation_helper():
    pattern = re.compile(r"<(\w+)>")
    assert parse_annotation("hi <Calm>", pattern) == "calm"
    assert parse_annotation("hi", pattern) is None
    assert parse_annotation("hi <Calm>", None) is None
    assert parse_annotation(None, pattern) is None
