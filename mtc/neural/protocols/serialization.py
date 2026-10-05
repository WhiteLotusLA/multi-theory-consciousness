"""
Neural Serializer -- Efficient Message Encoding/Decoding
==========================================================

Serializes NeuralMessage objects for persistence, IPC, and debugging.
Supports multiple formats:

- BINARY: compact struct-based encoding for high-throughput paths
- JSON: human-readable encoding for debugging and logging
- PROTOBUF: placeholder for future gRPC integration

The binary header is fixed-size; payloads are JSON-encoded inside it.

Timestamps: the binary format stores microseconds since the Unix epoch in
UTC. A naive timestamp is read as UTC (``datetime.timestamp`` would read it as
host local time and shift it by the host's UTC offset), and decoding returns
a timezone-aware UTC datetime. JSON carries ISO 8601 text; a naive value
decodes as UTC.
"""

from enum import Enum
import struct
import json
import numpy as np
import logging
from datetime import datetime, timedelta, timezone

from mtc.neural.protocols.message_format import (
    NeuralMessage,
    NeuralMessageType,
    MessagePriority,
    SpikeTrain,
    EmotionalState,
    CreativeVector,
    TemporalPattern,
)

logger = logging.getLogger(__name__)

# Binary format magic bytes for validation
_MAGIC = b"SNMP"  # Substrate Neural Message Protocol
_VERSION = 1


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """Return an aware datetime; a naive value is taken to be UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _epoch_microseconds(value: datetime) -> int:
    """Microseconds since the Unix epoch, independent of the host timezone."""
    delta = _as_utc(value) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


class SerializationFormat(Enum):
    """Supported serialization formats."""

    BINARY = "binary"
    JSON = "json"
    PROTOBUF = "protobuf"


class NeuralSerializer:
    """
    Serializes and deserializes NeuralMessage objects.

    Binary format structure:
      [4B magic][1B version][1B msg_type][1B priority]
      [12B msg_id][8B timestamp_us][4B seq_num]
      [1B source_len][source_bytes][1B target_len][target_bytes]
      [1B has_payload][payload_bytes...]
    """

    def serialize(
        self,
        message: NeuralMessage,
        fmt: SerializationFormat = SerializationFormat.JSON,
    ) -> bytes | str:
        """
        Serialize a NeuralMessage.

        Returns bytes for BINARY/PROTOBUF, str for JSON.
        """
        if fmt == SerializationFormat.BINARY:
            return self._serialize_binary(message)
        elif fmt == SerializationFormat.JSON:
            return self._serialize_json(message)
        elif fmt == SerializationFormat.PROTOBUF:
            raise NotImplementedError("Protobuf serialization not yet implemented")
        raise ValueError(f"Unknown format: {fmt}")

    def deserialize(
        self,
        data: bytes | str,
        fmt: SerializationFormat = SerializationFormat.JSON,
    ) -> NeuralMessage:
        """
        Deserialize bytes/str back to a NeuralMessage.
        """
        if fmt == SerializationFormat.BINARY:
            return self._deserialize_binary(data)
        elif fmt == SerializationFormat.JSON:
            return self._deserialize_json(data)
        elif fmt == SerializationFormat.PROTOBUF:
            raise NotImplementedError("Protobuf deserialization not yet implemented")
        raise ValueError(f"Unknown format: {fmt}")

    # --- JSON ---

    def _serialize_json(self, message: NeuralMessage) -> str:
        """Serialize to JSON string."""
        obj = {
            "source": message.source,
            "target": message.target,
            "message_type": message.message_type.value,
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
            "sequence_number": message.sequence_number,
            "requires_ack": message.requires_ack,
            "message_id": message.message_id,
            "metadata": message.metadata,
        }

        if message.payload is not None:
            obj["payload"] = self._payload_to_dict(message.payload)
            obj["payload_type"] = type(message.payload).__name__

        return json.dumps(obj)

    def _deserialize_json(self, data: str) -> NeuralMessage:
        """Deserialize from JSON string."""
        obj = json.loads(data)

        payload = None
        if "payload" in obj and "payload_type" in obj:
            payload = self._dict_to_payload(obj["payload"], obj["payload_type"])

        return NeuralMessage(
            source=obj["source"],
            target=obj["target"],
            message_type=NeuralMessageType(obj["message_type"]),
            priority=MessagePriority(obj["priority"]),
            timestamp=_as_utc(datetime.fromisoformat(obj["timestamp"])),
            sequence_number=obj["sequence_number"],
            requires_ack=obj.get("requires_ack", False),
            message_id=obj.get("message_id", ""),
            metadata=obj.get("metadata", {}),
            payload=payload,
        )

    # --- Binary ---

    def _serialize_binary(self, message: NeuralMessage) -> bytes:
        """Serialize to compact binary format."""
        parts = []

        # Header
        parts.append(_MAGIC)
        parts.append(struct.pack("B", _VERSION))
        parts.append(
            struct.pack("B", list(NeuralMessageType).index(message.message_type))
        )
        parts.append(struct.pack("B", message.priority.value))

        # Message ID (padded/truncated to 12 bytes)
        mid = message.message_id.encode("utf-8")[:12].ljust(12, b"\x00")
        parts.append(mid)

        # Timestamp as microseconds since the epoch, in UTC
        ts_us = _epoch_microseconds(message.timestamp)
        parts.append(struct.pack("<Q", ts_us))

        # Sequence number
        parts.append(struct.pack("<I", message.sequence_number))

        # Source string
        src = message.source.encode("utf-8")
        parts.append(struct.pack("B", len(src)))
        parts.append(src)

        # Target string
        tgt = message.target.encode("utf-8")
        parts.append(struct.pack("B", len(tgt)))
        parts.append(tgt)

        # Payload
        if message.payload is not None:
            payload_bytes = self._serialize_payload_binary(message.payload)
            parts.append(struct.pack("B", 1))  # has_payload
            parts.append(struct.pack("<I", len(payload_bytes)))
            parts.append(payload_bytes)
        else:
            parts.append(struct.pack("B", 0))  # no payload

        return b"".join(parts)

    def _deserialize_binary(self, data: bytes) -> NeuralMessage:
        """Deserialize from binary format."""
        offset = 0

        # Magic
        magic = data[offset : offset + 4]
        offset += 4
        if magic != _MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic!r}")

        # Version
        struct.unpack_from("B", data, offset)[0]
        offset += 1

        # Message type
        type_idx = struct.unpack_from("B", data, offset)[0]
        offset += 1
        msg_type = list(NeuralMessageType)[type_idx]

        # Priority
        priority_val = struct.unpack_from("B", data, offset)[0]
        offset += 1
        priority = MessagePriority(priority_val)

        # Message ID
        mid = data[offset : offset + 12].rstrip(b"\x00").decode("utf-8")
        offset += 12

        # Timestamp
        ts_us = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        # Exact integer arithmetic; the result is aware UTC
        timestamp = _EPOCH + timedelta(microseconds=ts_us)

        # Sequence number
        seq = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        # Source
        src_len = struct.unpack_from("B", data, offset)[0]
        offset += 1
        source = data[offset : offset + src_len].decode("utf-8")
        offset += src_len

        # Target
        tgt_len = struct.unpack_from("B", data, offset)[0]
        offset += 1
        target = data[offset : offset + tgt_len].decode("utf-8")
        offset += tgt_len

        # Payload
        has_payload = struct.unpack_from("B", data, offset)[0]
        offset += 1
        payload = None
        if has_payload:
            payload_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            payload_bytes = data[offset : offset + payload_len]
            payload = self._deserialize_payload_binary(payload_bytes)

        return NeuralMessage(
            source=source,
            target=target,
            message_type=msg_type,
            priority=priority,
            timestamp=timestamp,
            sequence_number=seq,
            message_id=mid,
            payload=payload,
        )

    # --- Payload helpers ---

    def _payload_to_dict(self, payload) -> dict:
        """Convert a payload object to a JSON-serializable dict."""
        if isinstance(payload, SpikeTrain):
            return {
                "neuron_ids": payload.neuron_ids,
                "spike_times": payload.spike_times,
                "strengths": payload.strengths,
            }
        elif isinstance(payload, EmotionalState):
            return {
                "valence": payload.valence,
                "arousal": payload.arousal,
                "dominance": payload.dominance,
                "primary_emotion": payload.primary_emotion,
                "secondary_emotions": payload.secondary_emotions,
                "intensity": payload.intensity,
            }
        elif isinstance(payload, CreativeVector):
            return {
                "dimensions": payload.dimensions,
                "values": payload.values.tolist(),
                "novelty_score": payload.novelty_score,
                "coherence_score": payload.coherence_score,
                "tags": payload.tags,
            }
        elif isinstance(payload, TemporalPattern):
            return {
                "sequence_length": payload.sequence_length,
                "pattern_data": payload.pattern_data.tolist(),
                "confidence_scores": payload.confidence_scores.tolist(),
                "pattern_id": payload.pattern_id,
                "category": payload.category,
            }
        return {}

    def _dict_to_payload(self, d: dict, payload_type: str):
        """Reconstruct a payload object from a dict."""
        if payload_type == "SpikeTrain":
            return SpikeTrain(
                neuron_ids=d["neuron_ids"],
                spike_times=d["spike_times"],
                strengths=d["strengths"],
            )
        elif payload_type == "EmotionalState":
            return EmotionalState(
                valence=d["valence"],
                arousal=d["arousal"],
                dominance=d.get("dominance", 0.5),
                primary_emotion=d.get("primary_emotion"),
                secondary_emotions=d.get("secondary_emotions", []),
                intensity=d.get("intensity", 0.5),
            )
        elif payload_type == "CreativeVector":
            return CreativeVector(
                dimensions=d["dimensions"],
                values=np.array(d["values"], dtype=np.float32),
                novelty_score=d.get("novelty_score", 0.0),
                coherence_score=d.get("coherence_score", 0.0),
                tags=d.get("tags", []),
            )
        elif payload_type == "TemporalPattern":
            return TemporalPattern(
                sequence_length=d["sequence_length"],
                pattern_data=np.array(d["pattern_data"]),
                confidence_scores=np.array(d["confidence_scores"], dtype=np.float32),
                pattern_id=d.get("pattern_id", ""),
                category=d.get("category", ""),
            )
        return None

    def _serialize_payload_binary(self, payload) -> bytes:
        """Serialize payload to binary (uses JSON internally for now)."""
        d = self._payload_to_dict(payload)
        d["_type"] = type(payload).__name__
        return json.dumps(d).encode("utf-8")

    def _deserialize_payload_binary(self, data: bytes):
        """Deserialize payload from binary."""
        d = json.loads(data.decode("utf-8"))
        ptype = d.pop("_type", None)
        if ptype:
            return self._dict_to_payload(d, ptype)
        return None
