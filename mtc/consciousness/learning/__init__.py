"""
Experiential Learning
=====================

System-level learning from lived experience: conversations, emotional
trajectories and their outcomes. It adds pattern detection across experiences
and behavioral adaptation on top of the low-level learning that already
happens inside the neural substrates.

Flow:

    ExperienceTracker -> PatternLearner -> BehavioralAdapter
    (captures outcomes)  (finds patterns)  (modulates the pipeline)

``AffectModulatedRate`` scales the learning rate from the affect state and
``LearningSystem`` wires the parts behind a ``LearningStore`` protocol (an
in-memory store by default). Nothing here is wired into the runners.
"""

from mtc.consciousness.learning.affect_modulation import AffectModulatedRate
from mtc.consciousness.learning.behavioral_adapter import BehavioralAdapter
from mtc.consciousness.learning.experience_tracker import ExperienceTracker
from mtc.consciousness.learning.learning_system import LearningSystem
from mtc.consciousness.learning.pattern_learner import PatternLearner

__all__ = [
    "AffectModulatedRate",
    "ExperienceTracker",
    "PatternLearner",
    "BehavioralAdapter",
    "LearningSystem",
]
