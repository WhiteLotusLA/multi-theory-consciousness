"""The stub noise baselines are pinned value for value.

The noise stubs draw fresh random numbers on every statistics call, so one
added, removed or reordered call to ``get_statistics``,
``get_loop_statistics``, ``get_oscillatory_binding_stats`` or ``get_latest``
inside any indicator body moves most of the 30 baselines, the legacy ones
included. The values below were captured from the tree with the seeds set as
in ``_baselines`` and are compared exactly. Mutation check recorded when the
pin was written: one extra ``rpt.get_statistics()`` call in one v05 indicator
body changed 28 of the 30 v05 baselines.
"""

import asyncio
import random

import numpy as np

from mtc.assessment.assessment import ConsciousnessAssessment

V04_BASELINES = {
    "affect_coherence": 0.10728469315063642,
    "agency": 0.4,
    "agent_environment_coupling": 0.04371281292110204,
    "algorithmic_recurrence": 0.1492604971610962,
    "attention_control": 0.04680208782049181,
    "attention_schema": 0.13469909927467943,
    "bayesian_binding_quality": 0.1910923333167342,
    "causal_emergence": 0.5462045899696207,
    "complex_agency": 0.25,
    "embodiment": 0.4,
    "epistemic_depth": 0.08962098537119476,
    "flexible_responsiveness": 0.3,
    "genuine_implementation": 0.2,
    "global_broadcast": 0.018850620152952226,
    "global_ignition_nuanced": 0.41200138925107355,
    "hierarchical_prediction": 0.16269389761847344,
    "higher_order_representations": 0.27999999999999997,
    "ignition_dynamics": 0.3642108025585163,
    "integrated_information": 0.0,
    "irreducibility": 0.7891272260041104,
    "local_recurrence": 0.09865355100431715,
    "metacognition": 0.42820718933810886,
    "online_reportable_learning": 0.25,
    "organized_percepts": 0.3,
    "oscillatory_binding_coherence": 0.0,
    "prediction_error_minimization": 0.05663258196320449,
    "recurrent_processing": 0.3867318360110229,
    "sequential_module_querying": 0.4699999999999999,
    "sparse_smooth_coding": 0.019225010146329248,
    "temporal_prediction_depth": 0.11011899155214344,
}

V05_BASELINES = {
    "affect_coherence": 0.19724801375800655,
    "agency": 0.21103225966340214,
    "agent_environment_coupling": 0.04331370849898476,
    "algorithmic_recurrence": 0.13718843329101565,
    "attention_control": 0.09211011857461857,
    "attention_schema": 0.1316336623707054,
    "bayesian_binding_quality": 0.0856904471237722,
    "causal_emergence": 0.4682530589405219,
    "complex_agency": 0.36910243777619556,
    "embodiment": 0.11348733365172695,
    "epistemic_depth": 0.10596871548641687,
    "flexible_responsiveness": 0.6026666666666667,
    "genuine_implementation": 0.13333333333333333,
    "global_broadcast": 0.16458540168129115,
    "global_ignition_nuanced": 0.2746152403712997,
    "hierarchical_prediction": 0.24878144998801965,
    "higher_order_representations": 0.43,
    "ignition_dynamics": 0.3575383498434354,
    "integrated_information": 0.49218572815055167,
    "irreducibility": 0.7891272260041104,
    "local_recurrence": 0.16569021569197784,
    "metacognition": 0.24116150883009171,
    "online_reportable_learning": 0.09,
    "organized_percepts": 0.1758430161217371,
    "oscillatory_binding_coherence": 0.38436960184523444,
    "prediction_error_minimization": 0.20859661159930237,
    "recurrent_processing": 0.2530964050607594,
    "sequential_module_querying": 0.5335425951573204,
    "sparse_smooth_coding": 0.1074946596432647,
    "temporal_prediction_depth": 0.06539250171911784,
}


def _baselines(formulas, policy):
    random.seed(42)
    np.random.seed(42)
    assessment = ConsciousnessAssessment(
        indicator_formulas=formulas, missing_signal_policy=policy
    )
    return asyncio.run(assessment.run_noise_baseline(iterations=5))


def _differences(expected, actual):
    names = sorted(set(expected) | set(actual))
    return {
        name: (expected.get(name), actual.get(name))
        for name in names
        if expected.get(name) != actual.get(name)
    }


def test_stub_noise_baselines_are_unchanged():
    legacy = _baselines("v04", "v04_constants")
    assert _differences(V04_BASELINES, legacy) == {}
    current = _baselines("v05", "insufficient_data")
    assert _differences(V05_BASELINES, current) == {}
