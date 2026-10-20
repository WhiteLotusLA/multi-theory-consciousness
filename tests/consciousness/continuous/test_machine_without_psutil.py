"""The continuous thought machine builds without psutil (the neural extra's
process statistics) and reports the resource check as unavailable with the
reason, instead of raising while it primes the reading at construction."""

import sys

from tests.consciousness.continuous.test_machine_cadence import make_machine


def test_machine_builds_without_psutil_and_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    machine = make_machine(["one thought"])
    resource = machine.get_status()["resource"]
    assert resource["available"] is False
    assert resource["reason"] == "psutil is not installed"
    assert resource["checks"] == 0
