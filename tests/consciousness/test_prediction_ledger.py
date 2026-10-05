"""
Tests for the prediction ledger.

The ledger records a concrete forecast before its outcome exists and scores
it later on an absolute L1 tolerance fixed at predict time, so the accuracy
it reports is measured rather than asserted.
"""

import json

import numpy as np

from mtc.consciousness.prediction_ledger import PredictionLedger


def _forecast():
    return np.array([0.6, 0.3, 0.1])


def test_predict_returns_id_and_is_unscored():
    ledger = PredictionLedger()
    pid = ledger.predict(_forecast(), tolerance=0.5)

    assert isinstance(pid, str) and pid
    assert ledger.is_pending(pid)
    assert ledger.scored_count == 0
    assert ledger.measured_accuracy() is None


def test_score_within_tolerance_is_hit():
    ledger = PredictionLedger()
    pid = ledger.predict(_forecast(), tolerance=0.5)

    # L1 distance 0.2 is inside the 0.5 tolerance.
    verdict = ledger.score(pid, np.array([0.5, 0.4, 0.1]))

    assert verdict is True
    assert not ledger.is_pending(pid)
    assert ledger.scored_count == 1
    assert np.isclose(ledger.last_scored.abs_error, 0.2)


def test_score_outside_tolerance_is_miss():
    ledger = PredictionLedger()
    pid = ledger.predict(_forecast(), tolerance=0.5)

    # L1 distance 1.2 is outside the tolerance.
    verdict = ledger.score(pid, np.array([0.0, 0.0, 1.0]))

    assert verdict is False
    assert ledger.scored_count == 1


def test_score_unknown_or_repeated_id_is_none():
    ledger = PredictionLedger()
    pid = ledger.predict(_forecast(), tolerance=0.5)

    assert ledger.score("missing", _forecast()) is None
    assert ledger.score(pid, _forecast()) is True
    # A prediction is scored once; a second score has no pending entry.
    assert ledger.score(pid, _forecast()) is None


def test_score_shape_mismatch_is_unscored():
    ledger = PredictionLedger()
    pid = ledger.predict(_forecast(), tolerance=0.5)

    assert ledger.score(pid, np.array([1.0, 0.0])) is None
    assert ledger.scored_count == 0
    assert not ledger.is_pending(pid)


def test_tolerance_is_fixed_at_predict_time():
    ledger = PredictionLedger()
    loose = ledger.predict(_forecast(), tolerance=2.0)
    tight = ledger.predict(_forecast(), tolerance=0.01)
    observed = np.array([0.0, 0.0, 1.0])

    assert ledger.score(loose, observed) is True
    assert ledger.score(tight, observed) is False


def test_measured_accuracy_none_until_twenty():
    ledger = PredictionLedger()
    for i in range(19):
        pid = ledger.predict(_forecast(), tolerance=0.5)
        ledger.score(pid, _forecast())
        assert ledger.measured_accuracy() is None, f"answered after {i + 1}"

    # The twentieth outcome makes the rate answerable; all hits so far.
    pid = ledger.predict(_forecast(), tolerance=0.5)
    ledger.score(pid, _forecast())
    assert ledger.measured_accuracy() == 1.0

    for _ in range(20):
        pid = ledger.predict(_forecast(), tolerance=0.5)
        ledger.score(pid, np.array([0.0, 0.0, 1.0]))
    assert ledger.measured_accuracy() == 0.5
    assert ledger.measured_accuracy(window=20) == 0.0


def test_ledger_is_bounded():
    ledger = PredictionLedger(max_pending=200, max_scored=1000)

    for _ in range(5000):
        ledger.predict(_forecast(), tolerance=0.5)
    assert ledger.pending_count == 200

    for _ in range(5000):
        pid = ledger.predict(_forecast(), tolerance=0.5)
        ledger.score(pid, _forecast())
    assert ledger.scored_count == 1000
    assert ledger.pending_count <= 200


def test_calibration_data_summarises_scored_outcomes():
    ledger = PredictionLedger()
    hit = ledger.predict(_forecast(), tolerance=0.5)
    miss = ledger.predict(_forecast(), tolerance=0.5)
    ledger.score(hit, _forecast())
    ledger.score(miss, np.array([0.0, 0.0, 1.0]))

    data = ledger.calibration_data()

    assert data["scored"] == 2
    assert data["hits"] == 1
    assert data["measured_accuracy"] is None  # below the 20 outcome minimum
    assert len(data["records"]) == 2
    assert {r["was_accurate"] for r in data["records"]} == {True, False}
    json.dumps(data)  # export is JSON-safe


def test_state_dict_round_trip():
    ledger = PredictionLedger()
    for i in range(30):
        pid = ledger.predict(_forecast(), tolerance=0.5, context={"index": i})
        observed = _forecast() if i % 3 else np.array([0.0, 0.0, 1.0])
        ledger.score(pid, observed)
    pending_ids = [ledger.predict(_forecast(), tolerance=0.5) for _ in range(3)]

    state = ledger.to_state_dict()
    assert set(state) == {"scalars", "arrays", "histories"}
    # Scalars and histories survive a JSON round trip.
    state = {
        "scalars": json.loads(json.dumps(state["scalars"])),
        "arrays": state["arrays"],
        "histories": json.loads(json.dumps(state["histories"])),
    }

    restored = PredictionLedger()
    restored.from_state_dict(state)

    assert restored.measured_accuracy() == ledger.measured_accuracy()
    assert restored.pending_ids() == pending_ids
    assert restored.scored_count == ledger.scored_count
    # A restored pending prediction is still scoreable.
    assert restored.score(pending_ids[0], _forecast()) is True
