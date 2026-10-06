# Continuous thought

The continuous thought machine (`mtc/consciousness/continuous/machine.py`,
`ContinuousThoughtMachine`) generates short background thoughts between
conversation turns, measures how new each one is, lets only novel ones
surface into the conversation prompt, stores them, and now and then writes a
gist of the recent window. It is on by default (`CTM_ENABLED`) and attached to
the neural orchestrator by the runtime.

The thoughts are template text from seed packs, optionally rewritten by the
model when the brain is free. They are not evidence of an inner life; see
[HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md).

## Configuration

`CTMConfig.load` (`mtc/consciousness/continuous/config.py`) reads `ctm.yaml`
from `CTM_CONFIG_DIR` (default `config/ctm`, relative to `PROJECT_ROOT`) and
applies the profile named by `CTM_PROFILE` (`default`). A missing file means
the built-in defaults, which equal the shipped `config/ctm/ctm.yaml`; an
unknown key or an out-of-range value raises `CTMConfigError` naming the file.

### Cadence

Rates are thoughts per second for each consciousness state:

| State | Rate | About one thought every |
|---|---|---|
| `awake` | 0.008 | 125 seconds |
| `drowsy` | 0.005 | 200 seconds |
| `dreaming` | 0.003 | 333 seconds |
| `deep_sleep` | 0.002 | 500 seconds |

The state comes from the sleep scheduler (`set_consciousness_state`). Each
delay is jittered by 0.3 of its length, with a floor of `min_sleep_s` (1
second).

### Rate limit

At most `max_thoughts_per_minute` (2) thoughts are generated in any 60
seconds. At the limit a tick does nothing (`rate_limited`) and the loop backs
off for 10 seconds.

### Level modulation

The measured consciousness level, read through the runtime's level cache (at
most once every 300 seconds), scales the cadence by `base + slope * level`
(0.4 plus 1.2 times the level), refreshed every 5 thoughts. With no level the
cadence is left unmodulated and `get_status()["modulation"]` says why
(`no_provider`, `unavailable` or `disabled`); no level is assumed.

### The CPU guard

Every 10 thoughts the process CPU is sampled without blocking (psutil,
optional). Above `resource.cpu_percent_limit` (5 percent) the cadence is
halved; it is restored once a later sample is under the limit.

### The demo profile

`CTM_PROFILE=demo` overlays `profiles.demo` from `ctm.yaml` (or the built-in
overlay): awake rate 0.5 per second, 30 thoughts per minute, a contention
window of 0 seconds and a gist every 5 thoughts. It exists for short runs such
as `examples/continuous_thought.py`.

## One tick

1. While paused nothing happens (`paused`).
2. The rate limit (above).
3. The level is refreshed on the first tick and every 5 thoughts.
4. The generator (`generator.py`) makes a thought from a strategy (`memory`,
   `emotion`, `association`, `question` or `spontaneous` while awake;
   `consolidation`, `creative`, `emotion` or `spontaneous` while dreaming),
   weighted by `strategy_weights` and `dream_strategy_weights`. With no seed
   text it misses (`no_seeds`).
5. **Novelty.** `NoveltyFilter` (`novelty.py`) compares the thought's
   embedding with the last 50 thoughts. At a cosine similarity of 0.85 or more
   it is a near-duplicate; a unique thought may surface only when its novelty
   (one minus the similarity) is at least 0.5. Whether a thought may surface
   comes only from this measurement; the generator's per-strategy
   `express_probability` sets a provisional flag that the machine replaces.
6. **The model-written form.** A surfacing thought is rewritten by the
   enhancer (`enhancer.py`) only when the contention guard says the brain is
   free (below). A miss keeps the template thought and is counted. A model
   line that is a near-duplicate of a recent thought is dropped in favour of
   the template thought and counted as `llm_duplicates`.
7. **Dedup on write.** A near-duplicate is not stored; the earlier thought's
   access count goes up instead. A unique thought gets its memory
   associations (with vector search), the current burst's `stream_id` (a new
   burst after 300 seconds of silence or a change of state) and is inserted;
   its parent's child link is recorded by `thought_id`.
8. Only a stored thought joins the buffer; an expressed one is broadcast as a
   `spontaneous_thought` websocket frame to clients that presented the API
   key.
9. **Gists** (below).
10. **The CPU guard** (above).

`inject_thought` stores an external thought, such as a dream from the sleep
scheduler, unexpressed, even while paused.

## Embedders and why a fresh machine is quiet

Novelty needs an embedder. The runtime builds the sentence embedder
(`EMBEDDING_MODEL`, the `embeddings` extra) when its package imports; the model
loads on first use. Without an embedder every thought stays quiet
(`no_embedder`). The hash embedder (`HashEmbedder` in `mtc/core/embeddings.py`)
measures word overlap, not meaning: it still catches near-identical texts, but
a thought may surface with it only when the caller passes the explicit
non-semantic opt-in (`allow_non_semantic=True`); otherwise the reason is
`non_semantic_embedder`.

## Quiet reasons

`get_status()["quiet_reason"]` and the API's `quiet_reason` name why the last
tick produced no expressed, model-formed thought (`QUIET_REASONS` in
`machine.py`):

| Reason | Meaning |
|---|---|
| `no_embedder` | no embedder, so novelty cannot be measured and nothing surfaces |
| `non_semantic_embedder` | only a non-semantic embedder, and no opt-in |
| `contended` | the brain is busy with a conversation or was within the contention window, so no model-written form was made |
| `paused` | the machine is paused (the sleep scheduler pauses the thought stream outside the awake state) |
| `rate_limited` | the per-minute limit was reached |
| `no_seeds` | the generator had no seed text for its strategy |

None means the last tick expressed a thought or nothing was quiet about it.

## The contention guard

The runtime gives the machine `flight.contended(contention_window_s)` as its
guard: true while a turn holds or waits for the brain, and for
`contention_window_s` (120 seconds) after a turn ends. A machine built with no
guard counts as contended, so the model is never taken from a conversation by
default. The demo profile sets the window to 0.

## Enhancer misses

`LLMThoughtEnhancer` returns a thought or nothing. A missing brain, a missing
context value, a `NoAnswerError`, any other brain failure, an empty or
reasoning-only reply and a reply that starts with one of
`llm.placeholder_prefixes` are misses. Misses store nothing in the model's
place; `misses`, `miss_reasons` and `last_miss_reason` count and name them.
The emotional state reaches the enhancer's prompt only when
`NARRATE_THOUGHT_MOOD` is on.

## Gists

Every `gist.every` stored thoughts (50; 5 in the demo profile), outside a
60-second cooldown and only when at least half of the window is new, the
machine composes a gist from the window's own thought texts (no model call).
The gist carries `source_count` (the thoughts it summarizes), the starting
retention 0.8, and a `dominant_emotion`: the emotion that most often scored
above 0.3 in the window, or None when no thought carried a measured emotion.
Gists surface into the prompt by retention, most important first; the memory
pruner decays their retention.

## Stores

`CTM_STORE` picks the thought store (`mtc/consciousness/continuous/store.py`):

- `memory` (the default): `InMemoryThoughtStore` and an in-memory gist store,
  held in process memory.
- `mongo`: `MongoThoughtStore` writes the `thought_streams` collection through
  the neural data manager, and gists go to `thought_gists` through the same
  manager. It needs `MONGODB_ENABLED`; `CTM_STORE=mongo` without it is a
  configuration error naming both keys, and a MongoDB build that failed falls
  back to memory with the error in the store report.

Every stored thought has a persisted `thought_id` (indexed in MongoDB);
children are linked by `thought_id`, never by the burst's `stream_id`. A
stored document the store cannot read honestly is skipped and counted in
`status()["skipped_unreadable"]`.

Dream thoughts never surface: thoughts whose type starts with an entry of
`surfacing.excluded_type_prefixes` (`dream`) are excluded by type.

## Authoring seed packs

Seed packs (`seeds.py`) are YAML mappings of short templates and word lists.
The shipped `config/ctm/seeds/default.yaml` is neutral. An operator adds packs
as `config/ctm/seeds.d/*.yaml` (gitignored), read in name order after the
default; per key, a list extends the lists before it and a scalar overrides.
The onboarding wizard writes one pack for the chosen companion type as
`config/ctm/seeds.d/10_<pack>.yaml`.

Keys: the templates `memory_templates`, `emotion_templates`,
`association_templates`, `questions`, `spontaneous_lines` and the four dream
families; plain entries `concept_pairs`, `creative_concepts` and
`memory_query_terms`; the slot word lists `reflection`, `feeling`, `emotion`,
`descriptor`, `assessment` and `observation`; and `description`.

A template names a slot as `{name}`. A slot word list fills it with a random
entry; a context slot (`{memory}`, `{current_emotion}`, `{first}`,
`{second}`, `{concept}`) is filled by the generator with real material. A
template naming any other slot is refused when the pack loads, naming the
file, the key and the slot, so a template no caller could fill never ships.

## Authoring the thought persona

`CTMPersona` (`persona.py`) holds the enhancer's system prompt, one
instruction per thought category and the labels that introduce each piece of
context. Resolution order: `config/ctm/persona.yaml` (gitignored, written by
the wizard), else the shipped `config/ctm/persona.example.yaml`, else the
built-in neutral text; `source` says which was used. File keys: `name` (empty
means `SYSTEM_NAME`), `system_prompt` (may use `{name}` only),
`category_prompts` and `prompt_headers`. Any placeholder other than `{name}`,
which includes every creator or speaker field, is refused at load time:
thought prompts never refer to the creator.

## Routes and the CLI

- `POST /api/ctm/start` (optional `{"state": ...}` with `awake`, `drowsy`,
  `dreaming` or `deep_sleep`), `POST /api/ctm/stop`, `GET /api/ctm/status`
  (`available`, `running`, `quiet_reason`, `status`). With `CTM_ENABLED`
  false the start and stop routes answer 503.
- `mtc ctm start [--state STATE]`, `mtc ctm stop` and `mtc ctm status` call
  those routes on the running server.
- `GET /api/consciousness/status` carries the machine's status under `ctm`.

See [API.md](API.md).
