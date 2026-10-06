"""Runtime conversation pipeline: speaker trust, context types, the cycle
augmenter, the consciousness orchestrator, the turn helpers, the context
assembly, the plugin protocol, the conversation pipeline facade and the
runtime composition root.

The public names below are resolved on first access (PEP 562), so importing
the package loads no store client, model client or web framework; each module
is imported by its own path when one of its names is read.
"""

from importlib import import_module
from typing import Any, Dict

_EXPORTS: Dict[str, str] = {
    "SpeakerEntry": "mtc.pipeline.speaker_trust",
    "SpeakerIdentity": "mtc.pipeline.speaker_trust",
    "SpeakerTrust": "mtc.pipeline.speaker_trust",
    "ABSENT_REASONS": "mtc.pipeline.context_types",
    "ModulatedContext": "mtc.pipeline.context_types",
    "RawConversationContext": "mtc.pipeline.context_types",
    "TheorySections": "mtc.pipeline.context_types",
    "TurnResult": "mtc.pipeline.context_types",
    "CycleAugmenter": "mtc.pipeline.cycle_augmenter",
    "CycleAugmenterConfig": "mtc.pipeline.cycle_augmenter",
    "ConsciousnessOrchestrator": "mtc.pipeline.consciousness_orchestrator",
    "OrchestratorConfig": "mtc.pipeline.consciousness_orchestrator",
    "apply_narration": "mtc.pipeline.consciousness_orchestrator",
    "Session": "mtc.pipeline.session_state",
    "SessionManager": "mtc.pipeline.session_state",
    "MessagePersistence": "mtc.pipeline.message_persistence",
    "build_turn_snapshot": "mtc.pipeline.message_persistence",
    "PostTurnConfig": "mtc.pipeline.post_turn",
    "PostTurnIntegrator": "mtc.pipeline.post_turn",
    "TurnRecord": "mtc.pipeline.post_turn",
    "ContextSummarizer": "mtc.pipeline.context_summarizer",
    "render_summary": "mtc.pipeline.context_summarizer",
    "BrainWindow": "mtc.pipeline.context_assembly",
    "GatheredContext": "mtc.pipeline.context_assembly",
    "build_brain_window": "mtc.pipeline.context_assembly",
    "gather_context": "mtc.pipeline.context_assembly",
    "PluginContext": "mtc.pipeline.plugins",
    "PluginOutcome": "mtc.pipeline.plugins",
    "PluginRegistry": "mtc.pipeline.plugins",
    "ResponsePlugin": "mtc.pipeline.plugins",
    "ToolOutcome": "mtc.pipeline.plugins",
    "TurnPlugin": "mtc.pipeline.plugins",
    "ConversationPipeline": "mtc.pipeline.conversation_pipeline",
    "GatedBrain": "mtc.pipeline.conversation_pipeline",
    "LLMFlight": "mtc.pipeline.conversation_pipeline",
    "PipelineConfig": "mtc.pipeline.conversation_pipeline",
    "BackgroundTask": "mtc.pipeline.runtime",
    "HUB_NAMES": "mtc.pipeline.runtime",
    "NOT_REGISTERED": "mtc.pipeline.runtime",
    "Runtime": "mtc.pipeline.runtime",
    "StoreBuilders": "mtc.pipeline.runtime",
    "build_runtime": "mtc.pipeline.runtime",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_EXPORTS))
