"""Orchestrator candidates: salience rules, the trust bonus, the converters."""

import asyncio
import secrets
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtc.consciousness.enhanced_global_workspace import (
    EnhancedGlobalWorkspace,
    WorkspaceCandidateSource,
)
from mtc.consciousness.pad_affect import PADAffectModel
from mtc.core.config import Settings
from mtc.core.narration import NarrationFlags
from mtc.memory.thought_gist import ThoughtGist
from mtc.pipeline.consciousness_orchestrator import ConsciousnessOrchestrator
from mtc.pipeline.context_types import RawConversationContext
from mtc.pipeline.cycle_augmenter import CycleAugmenter, CycleAugmenterConfig
from mtc.pipeline.speaker_trust import SpeakerEntry, SpeakerTrust

from datetime import datetime, timezone

T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)


def _trust(cfg, **env):
    return SpeakerTrust(
        [SpeakerEntry(name="Alice", role="creator", trust=0.9, token_env="TOKEN_A")],
        settings=cfg,
    )


def _orchestrator(cfg=None, trust=None, workspace=None, **kwargs):
    cfg = cfg or Settings(DEBUG=False)
    workspace = workspace or EnhancedGlobalWorkspace()
    augmenter = CycleAugmenter(
        CycleAugmenterConfig.all_off(), workspace, pad_model=PADAffectModel()
    )
    trust = trust or _trust(cfg)
    return ConsciousnessOrchestrator(
        workspace,
        augmenter=augmenter,
        narration=NarrationFlags(),
        speaker_trust=trust,
        settings=cfg,
        **kwargs,
    )


def _raw(orchestrator, message, **fields):
    identity = fields.pop("identity", None) or orchestrator.speaker_trust.verify(
        "Alice", None
    )
    return RawConversationContext(message=message, identity=identity, **fields)


def test_message_candidate_salience_rules():
    orchestrator = _orchestrator()
    plain = orchestrator._message_to_candidate(_raw(orchestrator, "a short note"))
    question = orchestrator._message_to_candidate(_raw(orchestrator, "a short note?"))
    assert plain.content_type == "message"
    assert plain.source is WorkspaceCandidateSource.SENSORY
    assert question.activation_level > plain.activation_level
    assert question.goal_relevance > plain.goal_relevance
    assert plain.metadata["is_question"] is False and question.metadata["is_question"]

    calm = orchestrator._message_to_candidate(_raw(orchestrator, "the report is ready"))
    emotional = orchestrator._message_to_candidate(
        _raw(orchestrator, "I am so happy and excited, truly grateful")
    )
    assert calm.emotional_salience == 0.0
    assert emotional.emotional_salience == pytest.approx(0.6)
    assert emotional.metadata["emotion_words"] == 3

    short = orchestrator._message_to_candidate(_raw(orchestrator, "yes"))
    long = orchestrator._message_to_candidate(_raw(orchestrator, "word " * 60))
    assert long.activation_level > short.activation_level
    assert long.activation_level == pytest.approx(0.5)  # saturates at 50 words
    assert short.recency_bonus == 1.0


def test_trust_bonus_only_for_trusted_identity(monkeypatch):
    token = secrets.token_hex(16)
    monkeypatch.setenv("TOKEN_A", token)
    cfg = Settings(DEBUG=False, TRUSTED_SPEAKERS="Alice")
    trust = _trust(cfg)
    orchestrator = _orchestrator(cfg, trust)
    message = "could you tell me about the weather?"
    spoofed = orchestrator._message_to_candidate(
        _raw(orchestrator, message, identity=trust.verify("Alice", None))
    )
    creator_name = orchestrator._message_to_candidate(
        _raw(orchestrator, message, identity=trust.verify(cfg.CREATOR_NAME, None))
    )
    guest = orchestrator._message_to_candidate(
        _raw(orchestrator, message, identity=trust.verify("Guest", None))
    )
    verified = orchestrator._message_to_candidate(
        _raw(orchestrator, message, identity=trust.verify("Alice", token))
    )
    assert (
        spoofed.activation_level
        == creator_name.activation_level
        == guest.activation_level
    )
    assert verified.activation_level == pytest.approx(
        spoofed.activation_level + cfg.TRUST_ACTIVATION_BONUS
    )
    assert spoofed.metadata["trust_bonus"] == 0.0
    assert verified.metadata["trust_bonus"] == pytest.approx(cfg.TRUST_ACTIVATION_BONUS)
    # The candidate carries no speaker name
    assert "Alice" not in repr(verified.metadata)


def test_memory_concept_thought_candidates():
    orchestrator = _orchestrator()
    memory = orchestrator._memory_to_candidate(
        {"content": "a stored line", "score": 0.8, "emotional_weight": 0.4}
    )
    assert memory.source is WorkspaceCandidateSource.MEMORY
    assert memory.activation_level == pytest.approx(0.8)
    assert memory.emotional_salience == pytest.approx(0.4)
    assert memory.summary == "a stored line"
    assert memory.metadata["scored"] is True
    text_match = orchestrator._memory_to_candidate(
        {"text": "a text match", "similarity_score": None, "relevance": 0.6}
    )
    assert text_match.activation_level == pytest.approx(0.6)
    unscored = orchestrator._memory_to_candidate({"content": "no score"})
    assert unscored.activation_level == pytest.approx(0.5)
    assert unscored.metadata["scored"] is False

    concept = orchestrator._concept_to_candidate(
        {"name": "weather", "importance": 0.5, "connections": 4}
    )
    assert concept.source is WorkspaceCandidateSource.CONCEPT
    assert concept.summary == "weather"
    assert concept.activation_level == pytest.approx(0.5 * 0.8 + 0.2)
    assert concept.goal_relevance == pytest.approx(0.5)
    many = orchestrator._concept_to_candidate({"name": "x", "connection_count": 40})
    assert many.activation_level == pytest.approx(0.4 + 0.3)  # factor capped

    thought = orchestrator._thought_to_candidate(
        {
            "content": "a passing thought",
            "thought_type": "reflection",
            "emotional_tone": "curiosity",
        }
    )
    assert thought.source is WorkspaceCandidateSource.CTM
    assert thought.activation_level == pytest.approx(0.5)
    assert thought.emotional_salience == pytest.approx(0.3)
    assert thought.metadata["thought_type"] == "reflection"
    scored = orchestrator._thought_to_candidate(
        {"content": "x", "salience_score": 0.9, "intensity": 0.8}
    )
    assert scored.activation_level == pytest.approx(0.9)
    assert scored.emotional_salience == pytest.approx(0.4)


def test_gist_candidate_reads_thought_gist_fields():
    orchestrator = _orchestrator()
    gist = ThoughtGist(
        gist_id="gist_1",
        created_at=T0,
        summary="a recurring pattern of thought",
        source_count=4,
        source_thought_ids=["a", "b", "c", "d"],
        dominant_emotion="curiosity",
        importance_score=0.8,
        retention_strength=0.5,
    )
    candidate = orchestrator._gist_to_candidate(gist)
    assert candidate.source is WorkspaceCandidateSource.GIST
    assert candidate.summary == "a recurring pattern of thought"
    assert candidate.activation_level == pytest.approx(0.4)
    assert candidate.emotional_salience == pytest.approx(0.3)
    assert candidate.metadata["source_count"] == 4
    assert candidate.metadata["dominant_emotion"] == "curiosity"
    assert candidate.metadata["importance_measured"] is True

    # The surfacing dicts carry the same field names
    from_dict = orchestrator._gist_to_candidate(
        {
            "gist_id": "gist_2",
            "summary": "another pattern",
            "source_count": 2,
            "dominant_emotion": None,
            "importance_score": None,
            "retention_strength": None,
        }
    )
    assert from_dict.activation_level == pytest.approx(0.5)
    assert from_dict.emotional_salience == 0.0
    assert from_dict.metadata["importance_measured"] is False
    # The keys the reference read are never consulted
    stale = orchestrator._gist_to_candidate(
        {"gist_text": "stale key", "pattern_strength": 0.9, "summary": "real"}
    )
    assert stale.summary == "real"
    assert stale.activation_level == pytest.approx(0.5)


def test_self_assessment_candidate_injected_once():
    orchestrator = _orchestrator()
    raw = _raw(orchestrator, "hello")
    assert all(
        c.content_type != "self_assessment" for c in orchestrator._build_candidates(raw)
    )
    comparison = {
        "dimension_divergences": {"agency": {"divergence": 0.3}},
        "notable_divergences": ["agency"],
    }
    orchestrator.record_self_assessment(SimpleNamespace(parsed_count=3), comparison)
    assert orchestrator.status()["self_assessment_pending"] is True
    first = [
        c
        for c in orchestrator._build_candidates(raw)
        if c.content_type == "self_assessment"
    ]
    assert len(first) == 1
    assert first[0].source is WorkspaceCandidateSource.INTERNAL
    assert first[0].metadata["notable_divergences"] == 1
    assert "1 notable" in first[0].summary
    second = [
        c
        for c in orchestrator._build_candidates(raw)
        if c.content_type == "self_assessment"
    ]
    assert second == []
    assert orchestrator.status()["self_assessment_pending"] is False


class _PublicNeural:
    """Only the neural orchestrator's public surface; no private member."""

    def __init__(self):
        self.latest_neural_signals = None
        self.latest_substrate_states = None
        self.state = SimpleNamespace(current_experience=None)
        self.published = []

    def publish_retrieved_memories(self, memories):
        self.published.append(memories)


def test_retrieved_memories_published_to_integration():
    neural = _PublicNeural()
    orchestrator = _orchestrator(neural_orchestrator=neural)
    memories = [{"content": "one", "score": 0.7}, {"content": "two", "score": 0.4}]
    raw = _raw(orchestrator, "what did we say?", retrieved_memories=memories)
    context = asyncio.run(orchestrator.process(raw))
    assert context.cycle_ok is True
    assert neural.published == [memories]
    assert neural.published[0] is not memories  # copied, not aliased
    memories.append({"content": "three"})
    assert len(neural.published[0]) == 2
    assert orchestrator.status()["failures"] == {}
    # A neural orchestrator without the public hook is a counted failure,
    # never a write into its private members
    bare = SimpleNamespace(
        latest_neural_signals=None,
        latest_substrate_states=None,
        state=SimpleNamespace(current_experience=None),
    )
    unwired = _orchestrator(neural_orchestrator=bare)
    asyncio.run(unwired.process(_raw(unwired, "hello", retrieved_memories=memories)))
    assert unwired.status()["failures"] == {"publish_retrieval": 1}
    assert vars(bare).keys() == {
        "latest_neural_signals",
        "latest_substrate_states",
        "state",
    }


def test_pipeline_reads_no_private_neural_member():
    import mtc.pipeline as pipeline

    root = Path(pipeline.__file__).parent
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for private in ("_last_retrieved_memories", "_verification_neural_data"):
            assert private not in source, (path.name, private)
