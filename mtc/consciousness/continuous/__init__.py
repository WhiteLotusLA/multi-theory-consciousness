"""Continuous thought: background thoughts, their stores and their surfacing.

The package exports the three names the neural orchestrator's guarded import
expects (``ThoughtGenerator``, ``ThoughtStream``, ``ContinuousThoughtMachine``),
the data types, stores and helpers around them, and ``build_thought_machine``,
which assembles a machine from Settings and the CTM config files.

``build_thought_machine`` reads ``CTM_CONFIG_DIR`` and ``CTM_PROFILE`` (through
``CTMConfig.load`` and ``SeedLibrary.from_config_dir``), ``CTM_STORE``
(``memory`` builds an ``InMemoryThoughtStore`` and an in-memory gist store;
``mongo`` builds a ``MongoThoughtStore`` over the given or a new
``NeuralDataManager``, which also serves as the gist store), and the sensitive
memory keys (``MemoryWeightRule.from_settings``). A passed store is used as is.
A brain, when given, is wrapped in an ``LLMThoughtEnhancer`` with the CTM
persona (``CTMPersona.load``), the mood gate read from the passed
Settings (``NARRATE_THOUGHT_MOOD`` or ``NARRATE_ALL``) and the same Settings
for its ``DEBUG`` read; without one no thought
gets a model-written form. The novelty filter always exists; without an embedder it reports
``none`` and every thought stays quiet, and a non-semantic embedder needs
``allow_non_semantic=True`` before its thoughts may surface. Importing the
package loads no model library, HTTP client or store client.
"""

import random
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from mtc.consciousness.continuous.buffer import ThoughtBuffer, ThoughtStream
from mtc.consciousness.continuous.config import CTMConfig
from mtc.consciousness.continuous.enhancer import LLMThoughtEnhancer
from mtc.consciousness.continuous.generator import (
    BridgeEmotionSource,
    EmotionSource,
    MemoryManagerSource,
    MemorySource,
    MemoryWeightRule,
    PadEmotionSource,
    ThoughtGenerator,
)
from mtc.consciousness.continuous.machine import ContinuousThoughtMachine, TickResult
from mtc.consciousness.continuous.novelty import NoveltyFilter, NoveltyResult
from mtc.consciousness.continuous.persona import CTMPersona
from mtc.consciousness.continuous.schemas import (
    ThoughtIntensity,
    ThoughtRecord,
    ThoughtType,
)
from mtc.consciousness.continuous.seeds import SeedLibrary
from mtc.consciousness.continuous.store import (
    InMemoryThoughtStore,
    MongoThoughtStore,
    ThoughtStore,
)
from mtc.core.config import Settings, get_settings
from mtc.core.narration import NarrationFlags

__all__ = [
    "BridgeEmotionSource",
    "CTMConfig",
    "CTMPersona",
    "ContinuousThoughtMachine",
    "EmotionSource",
    "InMemoryThoughtStore",
    "LLMThoughtEnhancer",
    "MemoryManagerSource",
    "MemorySource",
    "MemoryWeightRule",
    "MongoThoughtStore",
    "NoveltyFilter",
    "NoveltyResult",
    "PadEmotionSource",
    "SeedLibrary",
    "ThoughtBuffer",
    "ThoughtGenerator",
    "ThoughtIntensity",
    "ThoughtRecord",
    "ThoughtStore",
    "ThoughtStream",
    "ThoughtType",
    "TickResult",
    "build_thought_machine",
]


def build_thought_machine(
    *,
    settings: Optional[Settings] = None,
    brain: Any = None,
    store: Any = None,
    gist_store: Any = None,
    embedder: Any = None,
    memory_source: Optional[MemorySource] = None,
    emotion_source: Optional[EmotionSource] = None,
    contention_guard: Optional[Callable[[], bool]] = None,
    level_provider: Optional[Callable[[], Awaitable[Optional[float]]]] = None,
    broadcaster: Optional[Callable[[Dict[str, Any]], Any]] = None,
    profile: Optional[str] = None,
    vector_search: Any = None,
    allow_non_semantic: bool = False,
    data_manager: Any = None,
    config_dir: Any = None,
    clock: Optional[Callable[[], datetime]] = None,
    rng: Optional[random.Random] = None,
) -> ContinuousThoughtMachine:
    """Assemble a thought machine from Settings and the CTM files.

    See the module docstring for what each argument selects. ``profile`` and
    ``config_dir`` override ``CTM_PROFILE`` and ``CTM_CONFIG_DIR``; ``clock``
    (aware UTC) and ``rng`` are shared by the generator, the machine and the
    stores built here.
    """
    cfg = settings if settings is not None else get_settings()
    config = CTMConfig.load(config_dir, profile, settings=cfg)
    seeds = SeedLibrary.from_config_dir(config_dir, settings=cfg)
    if store is None:
        if cfg.CTM_STORE == "mongo":
            if data_manager is None:
                from mtc.neural.mongodb_schemas import NeuralDataManager

                data_manager = NeuralDataManager(settings=cfg)
            store = MongoThoughtStore(
                data_manager, cfg, clock, surfacing=config.surfacing
            )
            if gist_store is None:
                gist_store = data_manager
        else:
            store = InMemoryThoughtStore(clock=clock)
    if gist_store is None:
        from mtc.memory.inmemory import InMemoryThoughtGistStore

        gist_store = InMemoryThoughtGistStore()
    enhancer = None
    if brain is not None:
        persona = CTMPersona.load(config_dir, settings=cfg)
        enhancer = LLMThoughtEnhancer(
            brain,
            persona,
            config,
            now=clock,
            narrate_mood=NarrationFlags.from_settings(cfg).thought_mood,
            settings=cfg,
        )
    novelty = NoveltyFilter(
        embedder,
        cache_size=config.novelty.cache,
        dedup_threshold=config.novelty.dedup,
        express_threshold=config.novelty.express,
        allow_non_semantic=allow_non_semantic,
    )
    shared_rng = rng if rng is not None else random.Random()
    generator = ThoughtGenerator(
        seeds,
        config,
        memory_source=memory_source,
        emotion_source=emotion_source,
        weight_rules=MemoryWeightRule.from_settings(cfg),
        rng=shared_rng,
        clock=clock,
        settings=cfg,
    )
    buffer = ThoughtBuffer(
        max_buffer_size=config.buffer_size, theme_keywords=config.theme_keywords
    )
    return ContinuousThoughtMachine(
        generator,
        buffer,
        store,
        gist_store=gist_store,
        novelty=novelty,
        enhancer=enhancer,
        config=config,
        broadcaster=broadcaster,
        contention_guard=contention_guard,
        consciousness_level_provider=level_provider,
        vector_search=vector_search,
        clock=clock,
        rng=shared_rng,
        settings=cfg,
    )
