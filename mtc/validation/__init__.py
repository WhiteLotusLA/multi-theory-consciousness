"""
Validation
==========

The acceptance-gate harness for shadow signals: ``mtc.validation.shadow_log``
records a signal in shadow (measured, never narrated) and
``mtc.validation.gates`` judges the record with the movement, content-coupling,
range-honesty and null-beating gates; ``scripts/validation/run_gates.py`` runs
them from a spec. Each module is imported by its own path, so importing the
shadow log does not load the statistics stack.
"""
