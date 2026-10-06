"""Request diagnostics: duplicate receipts and responses are counted, never acted on."""

import pytest

from mtc.api.diagnostics import RequestObserver


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_duplicate_request_ids_counted():
    observer = RequestObserver(ttl_s=60.0, clock=Clock())
    assert observer.observe_receipt("req-1") == 1
    assert observer.observe_receipt("req-1") == 2
    assert observer.observe_receipt("req-2") == 1
    assert observer.observe_response("req-1") == 1
    assert observer.observe_response("req-1") == 2
    report = observer.duplicates()
    assert report["duplicate_receipts"] == 1
    assert report["duplicate_responses"] == 1
    assert report["tracked"] == 2


def test_observe_only_never_blocks():
    observer = RequestObserver(clock=Clock())
    # Many repeats of one id, a missing id and an empty id: every call returns
    # a count and nothing raises or refuses.
    counts = [observer.observe_receipt("same") for _ in range(5)]
    assert counts == [1, 2, 3, 4, 5]
    assert observer.observe_receipt(None) == 0
    assert observer.observe_receipt("") == 0
    assert observer.observe_response(None) == 0
    report = observer.duplicates()
    assert report["duplicate_receipts"] == 4
    assert report["untracked"] == 2
    # A response for an id never received is still counted, not rejected.
    assert observer.observe_response("never-received") == 1


def test_ttl_expiry():
    clock = Clock()
    observer = RequestObserver(ttl_s=10.0, clock=clock)
    assert observer.observe_receipt("req-1") == 1
    clock.now += 5.0
    assert observer.observe_receipt("req-1") == 2
    clock.now += 6.0  # 11 s after the first sighting: the entry has expired
    assert observer.observe_receipt("req-1") == 1
    clock.now += 11.0
    assert observer.duplicates()["tracked"] == 0
    # The totals survive expiry.
    assert observer.duplicates()["duplicate_receipts"] == 1


def test_ttl_must_be_positive():
    with pytest.raises(ValueError, match="ttl_s"):
        RequestObserver(ttl_s=0.0)
