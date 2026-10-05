"""
Monitoring
==========

Operational monitors that watch the framework itself: the alert manager with
its console and file notifiers, the alert store, and the wellbeing monitor.
The package re-exports nothing; import from the submodules. No monitor here is
wired into the experiment runners, the workspace cycle or the assessment.
"""
