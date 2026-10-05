"""
Safety
======

The safety layer: the trust compass (``mtc.safety.trust_compass``), the safety
monitor, its audit store and the safety integration. Each module is imported
by its own path; this package re-exports nothing, so importing it loads no
store client and no model.

No module here grants, scales or seeds trust by a speaker's name. The trusted
speaker check lives in the conversation pipeline's ``speaker_trust`` module
and reaches these modules only as an injected trust lookup.
"""
