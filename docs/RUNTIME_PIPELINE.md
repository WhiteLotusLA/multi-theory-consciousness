# Runtime pipeline

This page follows one conversation turn and one consciousness cycle through
the live runtime, in the order the code runs them. The composition root is
`mtc/pipeline/runtime.py` (`build_runtime`); the turn is
`ConversationPipeline.send_message` in `mtc/pipeline/conversation_pipeline.py`;
the cycle is `ConsciousnessOrchestrator.process` in
`mtc/pipeline/consciousness_orchestrator.py`; the post-turn hooks are
`PostTurnIntegrator.run` in `mtc/pipeline/post_turn.py`.

## One turn

`send_message(message, speaker, speaker_token)` runs under the pipeline's turn
lock, one turn at a time:

1. **Verify the speaker.** `SpeakerTrust.verify` decides who is speaking from
   the token, never from the name, and `use_identity` makes that identity
   current for every trust lookup of the turn. A name with no valid token is a
   bracketed claim with the default trust.
2. **Route the session.** The session key is `SpeakerTrust.session_key`. A new
   session is rehydrated from the conversation repository only for an
   authenticated speaker or the anonymous default; a claimed name starts
   empty, so it never reads an authenticated speaker's history.
3. **Note activity.** The sleep scheduler is told a conversation is happening
   and the autonomous-life loop is paused.
4. **Safety gate.** Unless the identity is trusted, the message goes through
   `SafetySystemsIntegration.pre_message_check` with the trust source's safety
   role. A deny ends the turn `blocked` and stores nothing; a check that could
   not run is counted and the turn continues.
5. **Store the message.** The user entry is appended, the oldest history is
   summarized and trimmed (quotes are pinned verbatim), and the user turn is
   persisted.
6. **Turn plugins.** A registered plugin may answer (`reply`, stored under its
   name with the `tool` role, never as the system's words) or add a tool turn.
   No plugin ships.
7. **Gather context.** Core memory, recalled memories (with echoes of the
   current window removed), surfaced thoughts, thought gists and the coherence
   package are gathered at the same time, each failure isolated and named. The
   theory-of-mind speaker block stays empty unless the speaker-model narration
   flag is on.
8. **The consciousness cycle** (below).
9. **Learning.** The learning system's directives are appended and its
   temperature delta is added inside the orchestrator's bounds; drive
   adjustments are reported, never applied.
10. **The brain.** The brain window is built and `think_with_context` is
    called through the foreground side of the single-flight gate with the five
    theory sections, bounded by `GENERATION_TIMEOUT_S`.
11. **Response plugins.** Each hop is a tool turn and one more bounded brain
    call, up to the registry's cap.
12. **Store the reply.** Reasoning is split from the spoken text, the
    assistant turn is persisted with its provenance snapshot, the post-turn
    hooks run (a failure there never discards the reply), the loop resumes and
    a `TurnResult` is returned.

## One cycle

`ConsciousnessOrchestrator.process(raw)` runs these named steps:

1. `publish_retrieval`: the retrieved memories are published to the neural
   orchestrator.
2. `heartbeat`: the baseline is marked due when the heartbeat (900 seconds)
   has passed.
3. `goals`: the intrinsic goals are read.
4. `candidates`: workspace candidates are built from the message, memories,
   thoughts and gists; the message candidate is always first.
5. `augment_before` (`CycleAugmenter.before_cycle`): urgent drives become
   HOMEOSTATIC candidates, the last prospective simulation becomes an
   INTERNAL candidate, cross-modal binding adds bound percepts, and a
   pre-attentive cap keeps the strongest 20.
6. `cycle`: `EnhancedGlobalWorkspace.process_consciousness_cycle` with the
   neural orchestrator's latest signals: the thalamic gate, hierarchical
   competition, ignition and broadcast, the attention schema, metacognition,
   active inference, the beautiful loop and the Damasio layers. The workspace
   records into the PhiTracker.
7. `rpt` and `causal_emergence`: run every 50 cycles each.
8. `augment_after` (`CycleAugmenter.after_cycle`): the PAD coordinate is
   computed and recorded on the state, the gate's arousal is fed, and the
   prospective simulator may prepare one candidate for the next cycle.
9. `reality_monitor`, `event_triggers` (an ignition spike, a streak of high
   integration or a high loop quality marks the baseline due),
   `attention_observer`, `developmental` (only when development is enabled)
   and `persistence` (the state hub's cycle hook).
10. `modulation`: the modulated context is built in four layers.

### The four modulation layers

1. **Narrative.** Workspace focus and ignition are always reported; every
   other line is gated by a `NarrationFlags` field, and every `NARRATE_*` key
   is false by default.
2. **Temperature.** A generation temperature from the cycle, inside 0.5 and
   1.2 (`OrchestratorConfig`).
3. **Directives.** Behavioural permissions in neutral second person.
4. **Self-awareness.** Gated by the inner-state flag; an unmeasured baseline is
   reported as unmeasured.

### The five theory sections

`TheorySections` (`mtc/pipeline/context_types.py`) carries five prompt slots:
`global_workspace_context` (a computed workspace line), `attention_guidance`,
`metacognitive_insights` (the attention schema's and metacognition's reports
under the self-model flag, a neutral measured summary otherwise),
`active_inference_context` (under the free-energy or body-drive flag) and
`neural_state` (the substrate block, under the binding flag). Phi appears in
no layer and no section: the integration proxy is telemetry, read through the
API only.

## Stores each step touches

With every enable key false (the default) each of these is the in-memory
implementation. With a store enabled:

| Step | PostgreSQL | MongoDB | Qdrant | Neo4j | Redis |
|---|---|---|---|---|---|
| route the session (rehydrate) | `messages` | | | | |
| safety gate | `safety_audit` | `alerts` (when an alert fires) | | | |
| store the message | `conversations`, `messages` | `conversation_turns`, `conversation_sessions` | | | `wm:conversation:{session}` |
| gather context | `core_memory`, `episodic_memory` | `thought_streams`, `thought_gists` (with `CTM_STORE=mongo`) | the memory collection | concepts | |
| store the reply | `messages` | `conversation_turns` (with the provenance snapshot) | | | `wm:conversation:{session}` |
| post-turn `neural_experience` | | neural telemetry, when the neural orchestrator's data manager stores substrate states | | | |
| post-turn `emotion` | `emotional_history` | | | | |
| post-turn `learning` | | `learning_experiences`, `learning_beliefs` | | | |
| post-turn `theory_of_mind` | | `speaker_models` | | | |
| post-turn `episodic_memory` | `episodic_memory` | | a point per memory | | |
| post-turn `concepts` | | | | `Concept`, `Experience` and `MENTIONED_IN` | |
| post-turn `baseline` | `consciousness_snapshots`, `consciousness_events` | | | | |

The privacy auditor keeps its audit log in process memory in the live
runtime; the `privacy_audit_log` collection is registered and created by the
bootstrap for callers that pass a MongoDB audit store. Intrinsic goals,
attention grabs and dreams use `intrinsic_goals`, `attention_grabs` and
`dreams` when MongoDB is enabled. See [DATABASES.md](DATABASES.md).

## Post-turn hooks

`PostTurnIntegrator.run` runs one hook per step, in `STEP_ORDER`, each
isolated:

| Step | What it does | Cadence |
|---|---|---|
| `verification` | coherence and emotional-authenticity tests on the assembled state | every turn |
| `temporal_verification` | appends the state to the temporal samples; runs the temporal-consistency test | every 10 turns |
| `neural_experience` | one experience through the neural orchestrator with the winner's or the PAD emotions; skipped when neither exists | every turn |
| `emotion` | the emotion extractor's label with the PAD coordinate, one emotional-history row | every turn |
| `learning` | the learning system records the turn | every turn |
| `theory_of_mind` | the verified speaker's model is updated | every turn |
| `episodic_memory` | one episodic memory of the exchange | every turn |
| `concepts` | concepts of the message and the spoken reply (never the speaker's name) | every turn |
| `homeostasis` | the drives see a conversation or exploring activity | every turn |
| `safety_log` | the safety integration's post-message log | every turn |
| `affect_shadow` | the affect shadow tap (shadow only) | every turn |
| `identity_marker` | an identity marker with the values this turn supplies | every 20 turns |
| `baseline` | the orchestrator's baseline assessment when it is due | when due |

The cadences are `PostTurnConfig` (`temporal_every_n` 10,
`identity_marker_every_n` 20).

## Concurrency

- **The turn lock.** One turn at a time; turns from the API and every
  websocket connection wait on the same lock. The API's assessment refresh
  takes the same lock, so an assessment never runs inside a turn.
- **The single-flight brain gate.** `LLMFlight` lets one caller use the brain
  at a time. A turn (foreground) waits for a background holder to finish; a
  background user (the thought enhancer, dreams, the autonomous-life loop,
  memory consolidation, self-assessment) waits while a turn holds the gate or
  waits for it, so the conversation always has priority.
- **Background yielding.** `flight.contended(window)` tells the thought
  machine that a turn is active, waiting, or ended within the CTM contention
  window (120 seconds by default); the machine then keeps its template thought
  and does not call the brain.

## Honest failures

A `NoAnswerError` (empty, reasoning-only or looping reply), a provider or
pipeline timeout and `BrainUnavailableError` end the turn with `success`
false and an `error_code` of `no_answer`, `generation_timeout` or
`brain_unavailable`. No assistant row is written and no text is composed in
the system's place. A turn its caller cancels (the API's first-token and stall
bounds, a client that left) re-raises the cancellation after counting it.

## Absent signals

Every optional `ConsciousnessState` field left None in a cycle is named in
`absent_signals` with one reason from `ABSENT_REASONS`
(`mtc/pipeline/context_types.py`): `warming_up`, `cadence_not_reached`,
`no_producer`, `no_input` or `disabled`. The PAD inputs that were measured are
listed in `pad_sources_measured`; an input not listed contributed its declared
neutral default.

## Loud failure counters and DEBUG

Every step of the pipeline, the orchestrator and the post-turn hooks counts
its failures by name, logs the first and every hundredth at WARNING (the step
and the error type, never the text) and reports the counts in `status()`.
With `MTC_DEBUG=true` a step failure raises instead, which is the setting for
development and tests.

## The provenance snapshot

`build_turn_snapshot` (`mtc/pipeline/message_persistence.py`) records, with
each assistant turn, what the system was told about itself (narrative, theory
sections, directives, self-awareness), the generation parameters, the
computed state with its PAD coordinate, the integration proxy labelled as
such, the measured PAD sources, the absent signals and the provider id. It is
stored in the turn's `metadata.consciousness_snapshot`.

## Background tasks

`build_runtime` starts these `BackgroundTask`s:

| Task | Cadence | Condition |
|---|---|---|
| continuous thought machine | its own cadence | `CTM_ENABLED` |
| sleep scheduler | its own clock | `SLEEP_SCHEDULER_ENABLED` |
| autonomous-life loop | its own clock | `AUTONOMOUS_LIFE_ENABLED` (false by default) |
| baseline heartbeat and wellbeing sampler | every 60 seconds | always |
| Phi idle sampler | every `PHI_IDLE_SAMPLE_S` | when above 0 (0 by default) |
| self-assessment | every `SELF_ASSESSMENT_INTERVAL_S` | when above 0 (0 by default) |
| full state save | every 300 seconds | always |
| session pruning | hourly | always |

## Start and stop

`Runtime.start` restores the persistence hub (`STATE_DIR`) and starts the
tasks. `Runtime.stop` stops the tasks, ends the open conversations, saves the
hub, then shuts down the neural orchestrator, the memory stack, the store
clients and the brain, in that order. The API server's lifespan calls them;
see [API.md](API.md). `scripts/ops/stop_stack.sh` sends SIGTERM and waits a
grace period, because the server saves state during this shutdown; see
[OPERATIONS.md](OPERATIONS.md).
