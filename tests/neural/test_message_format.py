"""Tests for the neural message envelope and its payload types."""

from datetime import datetime, timezone

import numpy as np
import pytest

from mtc.neural.protocols.message_format import (
    CreativeVector,
    EmotionalState,
    MessagePriority,
    NeuralMessage,
    NeuralMessageType,
    SpikeTrain,
    TemporalPattern,
)


class TestNeuralMessage:
    def test_defaults(self):
        msg = NeuralMessage(
            source="snn", target="htm", message_type=NeuralMessageType.SPIKE_TRAIN
        )

        assert msg.priority == MessagePriority.NORMAL
        assert msg.payload is None
        assert msg.requires_ack is False
        assert msg.metadata == {}
        assert len(msg.message_id) == 12
        assert msg.sequence_number >= 0

    def test_timestamp_is_aware_utc(self):
        before = datetime.now(timezone.utc)
        msg = NeuralMessage(
            source="snn", target="htm", message_type=NeuralMessageType.CONTROL
        )
        after = datetime.now(timezone.utc)

        assert msg.timestamp.tzinfo is not None
        assert msg.timestamp.utcoffset().total_seconds() == 0
        assert before <= msg.timestamp <= after

    def test_sequence_numbers_increase(self):
        first = NeuralMessage(
            source="a", target="b", message_type=NeuralMessageType.CONTROL
        )
        second = NeuralMessage(
            source="a", target="b", message_type=NeuralMessageType.CONTROL
        )

        assert second.sequence_number > first.sequence_number

    def test_explicit_fields_and_payload(self):
        spikes = SpikeTrain(
            neuron_ids=[1, 2, 3], spike_times=[0.01, 0.02, 0.03], strengths=[1, 1, 1]
        )
        msg = NeuralMessage(
            source="snn",
            target="ctm",
            message_type=NeuralMessageType.SPIKE_TRAIN,
            priority=MessagePriority.CRITICAL,
            payload=spikes,
            requires_ack=True,
            metadata={"origin": "unit-test"},
            sequence_number=41,
        )

        assert msg.payload is spikes
        assert msg.requires_ack is True
        assert msg.metadata == {"origin": "unit-test"}
        assert msg.sequence_number == 41

    def test_priority_sort_puts_critical_first(self):
        priorities = [
            MessagePriority.LOW,
            MessagePriority.CRITICAL,
            MessagePriority.NORMAL,
            MessagePriority.HIGH,
        ]
        messages = [
            NeuralMessage(
                source="a",
                target="b",
                message_type=NeuralMessageType.STATE_UPDATE,
                priority=p,
            )
            for p in priorities
        ]

        ordered = sorted(messages, key=lambda m: m.priority.value)

        assert [m.priority for m in ordered] == [
            MessagePriority.CRITICAL,
            MessagePriority.HIGH,
            MessagePriority.NORMAL,
            MessagePriority.LOW,
        ]


class TestSpikeTrain:
    def test_rate_and_numpy_view(self):
        spikes = SpikeTrain(
            neuron_ids=[4, 4, 7],
            spike_times=[0.01, 0.04, 0.09],
            strengths=[0.5, 0.75, 1.0],
        )

        # Three spikes over the 90 ms they span
        assert spikes.get_spike_rate(window_ms=100.0) == pytest.approx(3 / 0.09)
        array = spikes.to_numpy()
        assert array.shape == (3, 3)
        np.testing.assert_allclose(array[1], [4.0, 0.04, 0.75])

    def test_empty_train_has_zero_rate(self):
        assert (
            SpikeTrain(neuron_ids=[], spike_times=[], strengths=[]).get_spike_rate()
            == 0.0
        )


class TestEmotionalState:
    def test_defaults_and_vector(self):
        state = EmotionalState(valence=0.2, arousal=0.4)

        assert state.dominance == 0.5
        assert state.intensity == 0.5
        np.testing.assert_allclose(state.to_vector(), [0.2, 0.4, 0.5, 0.5])

    def test_distance(self):
        origin = EmotionalState(valence=0.0, arousal=0.0, dominance=0.0, intensity=0.0)
        shifted = EmotionalState(valence=0.0, arousal=0.6, dominance=0.8, intensity=0.0)

        assert origin.distance_to(shifted) == pytest.approx(1.0)
        assert origin.distance_to(origin) == pytest.approx(0.0)

    def test_named_emotions(self):
        state = EmotionalState(
            valence=0.7,
            arousal=0.3,
            primary_emotion="calm",
            secondary_emotions=["interest"],
        )

        assert state.primary_emotion == "calm"
        assert state.secondary_emotions == ["interest"]


class TestCreativeVector:
    def test_identical_vectors_score_one(self):
        values = np.linspace(-1.0, 1.0, 16).astype(np.float32)
        a = CreativeVector(dimensions=16, values=values)
        b = CreativeVector(dimensions=16, values=values.copy())

        assert a.similarity_to(b) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_half(self):
        a = CreativeVector(dimensions=3, values=np.array([1.0, 0.0, 0.0]))
        b = CreativeVector(dimensions=3, values=np.array([0.0, 1.0, 0.0]))

        assert a.similarity_to(b) == pytest.approx(0.5)

    def test_opposite_vectors_score_zero(self):
        a = CreativeVector(dimensions=2, values=np.array([1.0, 2.0]))
        b = CreativeVector(dimensions=2, values=np.array([-1.0, -2.0]))

        assert a.similarity_to(b) == pytest.approx(0.0)

    def test_scores_and_tags(self):
        vector = CreativeVector(
            dimensions=4,
            values=np.zeros(4, dtype=np.float32),
            novelty_score=0.3,
            coherence_score=0.6,
            tags=["alpha", "beta"],
        )

        assert vector.novelty_score == 0.3
        assert vector.coherence_score == 0.6
        assert vector.tags == ["alpha", "beta"]


class TestTemporalPattern:
    def test_exact_match_and_wrong_shape(self):
        data = np.ones((3, 4), dtype=np.float32)
        pattern = TemporalPattern(
            sequence_length=3,
            pattern_data=data,
            confidence_scores=np.ones(3, dtype=np.float32),
            pattern_id="seq_a",
            category="synthetic",
        )

        assert pattern.match_score(data) == pytest.approx(1.0)
        assert pattern.match_score(np.ones((2, 4))) == 0.0
        assert pattern.pattern_id == "seq_a"
        assert pattern.category == "synthetic"

    def test_random_match_in_unit_interval(self):
        rng = np.random.default_rng(0)
        pattern = TemporalPattern(
            sequence_length=5,
            pattern_data=rng.random((5, 8)),
            confidence_scores=rng.random(5).astype(np.float32),
        )

        assert 0.0 <= pattern.match_score(rng.random((5, 8))) <= 1.0


def test_enum_values():
    assert [t.value for t in NeuralMessageType] == [
        "spike_train",
        "prediction",
        "creative_output",
        "emotional_state",
        "temporal_pattern",
        "state_update",
        "control",
        "synchronization",
    ]
    assert [p.value for p in MessagePriority] == [0, 1, 2, 3]
