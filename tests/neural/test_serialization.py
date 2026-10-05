"""Tests for JSON and binary serialization of neural messages, including the
timezone handling of the binary timestamp."""

import json
import os
import struct
import time
from datetime import datetime, timedelta, timezone

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
from mtc.neural.protocols.serialization import NeuralSerializer, SerializationFormat

# 2000-01-01T00:00:00Z in microseconds since the Unix epoch
Y2K_US = 946_684_800_000_000
# Offset of the 8-byte timestamp in the binary header: magic, version, type,
# priority, then the 12-byte message id
TIMESTAMP_OFFSET = 4 + 1 + 1 + 1 + 12


@pytest.fixture
def serializer():
    return NeuralSerializer()


@pytest.fixture
def non_utc_host(monkeypatch):
    """Run with a host timezone well away from UTC (POSIX only)."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available on this platform")
    monkeypatch.setenv("TZ", "Etc/GMT+5")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _round_trip(serializer, message, fmt):
    return serializer.deserialize(serializer.serialize(message, fmt), fmt)


class TestJSON:
    def test_envelope_round_trip(self, serializer):
        original = NeuralMessage(
            source="snn",
            target="htm",
            message_type=NeuralMessageType.SPIKE_TRAIN,
            priority=MessagePriority.HIGH,
            requires_ack=True,
            metadata={"key": "value"},
        )

        text = serializer.serialize(original, SerializationFormat.JSON)
        assert isinstance(text, str)
        assert json.loads(text)["source"] == "snn"

        restored = serializer.deserialize(text, SerializationFormat.JSON)
        assert restored.source == "snn"
        assert restored.target == "htm"
        assert restored.priority == MessagePriority.HIGH
        assert restored.message_type == NeuralMessageType.SPIKE_TRAIN
        assert restored.requires_ack is True
        assert restored.metadata == {"key": "value"}
        assert restored.message_id == original.message_id
        assert restored.timestamp == original.timestamp
        assert restored.payload is None

    def test_naive_json_timestamp_decodes_as_utc(self, serializer):
        message = NeuralMessage(
            source="a",
            target="b",
            message_type=NeuralMessageType.CONTROL,
            timestamp=datetime(2000, 1, 1, 12, 0, 0),
        )

        restored = _round_trip(serializer, message, SerializationFormat.JSON)

        assert restored.timestamp == datetime(2000, 1, 1, 12, tzinfo=timezone.utc)

    def test_spike_train_payload(self, serializer):
        payload = SpikeTrain(
            neuron_ids=[1, 2, 3],
            spike_times=[0.01, 0.02, 0.03],
            strengths=[0.8, 0.9, 0.7],
        )
        message = NeuralMessage(
            source="snn",
            target="htm",
            message_type=NeuralMessageType.SPIKE_TRAIN,
            payload=payload,
        )

        restored = _round_trip(serializer, message, SerializationFormat.JSON).payload

        assert isinstance(restored, SpikeTrain)
        assert restored.neuron_ids == [1, 2, 3]
        assert restored.strengths == [0.8, 0.9, 0.7]

    def test_emotional_state_payload(self, serializer):
        payload = EmotionalState(
            valence=0.4,
            arousal=0.6,
            dominance=0.3,
            primary_emotion="interest",
            secondary_emotions=["calm"],
            intensity=0.9,
        )
        message = NeuralMessage(
            source="snn",
            target="hub",
            message_type=NeuralMessageType.EMOTIONAL_STATE,
            payload=payload,
        )

        restored = _round_trip(serializer, message, SerializationFormat.JSON).payload

        assert isinstance(restored, EmotionalState)
        np.testing.assert_allclose(restored.to_vector(), payload.to_vector())
        assert restored.primary_emotion == "interest"
        assert restored.secondary_emotions == ["calm"]

    def test_creative_vector_payload(self, serializer):
        values = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
        payload = CreativeVector(
            dimensions=4, values=values, novelty_score=0.7, tags=["a", "b"]
        )
        message = NeuralMessage(
            source="lsm",
            target="hub",
            message_type=NeuralMessageType.CREATIVE_OUTPUT,
            payload=payload,
        )

        restored = _round_trip(serializer, message, SerializationFormat.JSON).payload

        assert isinstance(restored, CreativeVector)
        np.testing.assert_allclose(restored.values, values)
        assert restored.novelty_score == pytest.approx(0.7)
        assert restored.tags == ["a", "b"]

    def test_temporal_pattern_payload(self, serializer):
        payload = TemporalPattern(
            sequence_length=2,
            pattern_data=np.array([[1.0, 0.0], [0.0, 1.0]]),
            confidence_scores=np.array([0.9, 0.8], dtype=np.float32),
            pattern_id="seq_b",
            category="synthetic",
        )
        message = NeuralMessage(
            source="htm",
            target="hub",
            message_type=NeuralMessageType.TEMPORAL_PATTERN,
            payload=payload,
        )

        restored = _round_trip(serializer, message, SerializationFormat.JSON).payload

        assert isinstance(restored, TemporalPattern)
        np.testing.assert_allclose(restored.pattern_data, payload.pattern_data)
        assert restored.pattern_id == "seq_b"


class TestBinary:
    def test_envelope_round_trip(self, serializer):
        original = NeuralMessage(
            source="snn",
            target="htm",
            message_type=NeuralMessageType.PREDICTION,
            priority=MessagePriority.CRITICAL,
            sequence_number=7,
        )

        data = serializer.serialize(original, SerializationFormat.BINARY)
        assert isinstance(data, bytes)
        assert data[:4] == b"SNMP"

        restored = serializer.deserialize(data, SerializationFormat.BINARY)
        assert restored.source == "snn"
        assert restored.target == "htm"
        assert restored.message_type == NeuralMessageType.PREDICTION
        assert restored.priority == MessagePriority.CRITICAL
        assert restored.sequence_number == 7
        assert restored.message_id == original.message_id

    def test_payload_round_trip(self, serializer):
        payload = SpikeTrain(
            neuron_ids=[10, 20], spike_times=[0.1, 0.2], strengths=[0.5, 0.6]
        )
        message = NeuralMessage(
            source="snn",
            target="htm",
            message_type=NeuralMessageType.SPIKE_TRAIN,
            payload=payload,
        )

        restored = _round_trip(serializer, message, SerializationFormat.BINARY).payload

        assert isinstance(restored, SpikeTrain)
        assert restored.neuron_ids == [10, 20]

    def test_binary_round_trip_preserves_aware_utc(self, serializer):
        stamp = datetime(2000, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
        message = NeuralMessage(
            source="a",
            target="b",
            message_type=NeuralMessageType.CONTROL,
            timestamp=stamp,
        )

        restored = _round_trip(serializer, message, SerializationFormat.BINARY)

        assert restored.timestamp == stamp
        assert restored.timestamp.utcoffset() == timedelta(0)

    def test_binary_round_trip_converts_other_offsets_to_utc(self, serializer):
        plus_two = timezone(timedelta(hours=2))
        stamp = datetime(2000, 1, 1, 2, 0, 0, tzinfo=plus_two)
        message = NeuralMessage(
            source="a",
            target="b",
            message_type=NeuralMessageType.CONTROL,
            timestamp=stamp,
        )

        restored = _round_trip(serializer, message, SerializationFormat.BINARY)

        assert restored.timestamp == datetime(2000, 1, 1, tzinfo=timezone.utc)
        assert restored.timestamp.utcoffset() == timedelta(0)

    def test_binary_encode_has_no_host_offset(self, serializer, non_utc_host):
        assert time.timezone != 0, os.environ.get("TZ")
        aware = NeuralMessage(
            source="a",
            target="b",
            message_type=NeuralMessageType.CONTROL,
            timestamp=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        naive = NeuralMessage(
            source="a",
            target="b",
            message_type=NeuralMessageType.CONTROL,
            timestamp=datetime(2000, 1, 1),
        )

        for message in (aware, naive):
            data = serializer.serialize(message, SerializationFormat.BINARY)
            (ts_us,) = struct.unpack_from("<Q", data, TIMESTAMP_OFFSET)
            assert ts_us == Y2K_US

        restored = _round_trip(serializer, naive, SerializationFormat.BINARY)
        assert restored.timestamp == datetime(2000, 1, 1, tzinfo=timezone.utc)

    def test_binary_is_smaller_than_json_for_spike_trains(self, serializer):
        message = NeuralMessage(
            source="snn_primary",
            target="htm_main",
            message_type=NeuralMessageType.SPIKE_TRAIN,
            payload=SpikeTrain(
                neuron_ids=list(range(50)),
                spike_times=[i * 0.001 for i in range(50)],
                strengths=[0.8] * 50,
            ),
        )

        json_size = len(
            serializer.serialize(message, SerializationFormat.JSON).encode()
        )
        binary_size = len(serializer.serialize(message, SerializationFormat.BINARY))

        assert binary_size < json_size

    def test_invalid_magic_is_rejected(self, serializer):
        with pytest.raises(ValueError, match="Invalid magic"):
            serializer.deserialize(b"XXXX" + b"\x00" * 40, SerializationFormat.BINARY)


def test_protobuf_is_not_implemented(serializer):
    message = NeuralMessage(
        source="a", target="b", message_type=NeuralMessageType.CONTROL
    )

    with pytest.raises(NotImplementedError):
        serializer.serialize(message, SerializationFormat.PROTOBUF)
    with pytest.raises(NotImplementedError):
        serializer.deserialize(b"", SerializationFormat.PROTOBUF)


def test_format_values():
    assert [f.value for f in SerializationFormat] == ["binary", "json", "protobuf"]
