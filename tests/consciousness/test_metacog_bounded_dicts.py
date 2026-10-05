"""The metacognition by-id dicts must stay bounded with their deques.

A process that runs for days registers a first-order state every cycle.
The deques evict old entries; the lookup dicts must evict the same ids,
or they grow for the lifetime of the process.
"""

import pytest

from mtc.consciousness.metacognition import (
    FirstOrderStateType,
    MetacognitionModule,
    MetaType,
)


def _module(first_order: int = 5, hots: int = 4) -> MetacognitionModule:
    return MetacognitionModule(
        hot_generation_threshold=0.0,
        max_first_order_buffer=first_order,
        max_hot_buffer=hots,
    )


def _register(meta: MetacognitionModule, index: int):
    return meta.register_first_order_state(
        content=f"synthetic content {index}",
        content_summary=f"synthetic state {index}",
        state_type=FirstOrderStateType.THOUGHT,
        source_module="test",
        confidence=0.8,
        evidence_strength=0.7,
    )


@pytest.mark.asyncio
async def test_hot_by_id_dicts_evict_with_deques():
    meta = _module(first_order=5, hots=4)
    for i in range(40):
        state = _register(meta, i)
        await meta.generate_hot(state, meta_type=MetaType.AWARENESS, voluntary=True)
    for i in range(10):
        await meta.introspect_consciousness_level(0.5 + i * 0.01)

    assert len(meta.higher_order_thoughts) == 4
    assert len(meta.hots_by_id) == len(meta.higher_order_thoughts)
    assert set(meta.hots_by_id) == {h.id for h in meta.higher_order_thoughts}
    assert len(meta.first_order_by_id) == len(meta.first_order_states) == 5


def test_first_order_dict_matches_deque_after_overflow():
    meta = _module(first_order=5)
    for i in range(50):
        _register(meta, i)

    assert len(meta.first_order_states) == 5
    assert set(meta.first_order_by_id) == {s.id for s in meta.first_order_states}


@pytest.mark.asyncio
async def test_lookup_by_id_still_works_for_retained():
    meta = _module(first_order=3, hots=3)
    states = [_register(meta, i) for i in range(6)]
    hots = []
    for state in states[-3:]:
        hots.append(
            await meta.generate_hot(state, meta_type=MetaType.AWARENESS, voluntary=True)
        )

    for state in states[:3]:
        assert state.id not in meta.first_order_by_id
    for state in states[3:]:
        assert meta.first_order_by_id[state.id] is state
    for hot in meta.higher_order_thoughts:
        assert meta.hots_by_id[hot.id] is hot
