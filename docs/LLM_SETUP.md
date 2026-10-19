# LLM setup

The brain (`mtc/brain/brain.py`, `LLMBrain`) talks to exactly one provider,
chosen by `LLM_PROVIDER`:

| Provider | `LLM_PROVIDER` | Extra | Data leaves the machine |
|---|---|---|---|
| A local chat-completions server | `local` (the default) | `llm` (httpx) | no, when the server runs on this machine |
| Anthropic | `anthropic` | `anthropic` | yes: the system prompt and the conversation go to an external API |
| Mock | `mock` | none | no |

The brain runs the provider's health check at start. An unhealthy provider
leaves the brain degraded: a call raises `BrainUnavailableError`, the API
answers 503 for a turn (`brain_unavailable`), and no other provider is tried
in its place. A degraded brain re-checks the provider on the call path when a
call arrives, at most once every 5 seconds (`HEALTH_RECHECK_S` in
`mtc/brain/brain.py`, measured on the brain's own monotonic clock), so turns
succeed again once the provider recovers, without a restart; a healthy
brain is never re-probed. A re-check whose health probe raises any error
counts as unhealthy, so the call still raises `BrainUnavailableError` and
the turn answers 503 (`brain_unavailable`). The brain's `status()` also records the outcome
of the last real provider call (`last_call_ok`: None before one, true when
a completion came back, false on a provider error or timeout; and
`last_call_at`) and `health_recheck_s`; these are reported, never used to
refuse a call. `/health` reads them (see [API.md](API.md)). The active
choice is recorded in the provider state file (`PROVIDER_STATE_PATH`,
default `./data/provider_state.json`) for status pages and the CLI; it never
changes which provider runs.

## The mock provider

`LLM_PROVIDER=mock` (`mtc/brain/providers/mock_provider.py`) opens no socket
and needs no model. Every reply is labelled:

```
[mock] reply to: <the first 60 characters of the message> (<8 hex digits>)
```

The hex digits are the start of the SHA-256 digest of the message, so replies
are identical across runs. Mock replies are test text, not a model's answer;
`mtc init`'s example answers, the demo, the examples and the tests use it.

## A local chat-completions server

`LocalChatCompletionsProvider` (`mtc/brain/providers/local_chat.py`) speaks
the common chat-completions HTTP format that most local inference servers
expose. The settings it reads:

| Key | Default | Meaning |
|---|---|---|
| `LLM_HOST` and `LLM_PORT` | `localhost` and 8080 | the server's address |
| `LLM_BASE_URL` | derived from host and port | the base URL; set it to reach a server elsewhere |
| `LLM_COMPLETIONS_PATH` | `v1/chat/completions` | the request path under the base URL (a leading slash is optional) |
| `LLM_HEALTH_PATHS` | `/health,/v1/models` | the paths the health check tries, in order |
| `LLM_MODEL` | empty | the model id to send; empty sends no model field |
| `LLM_TEMPLATE_FAMILY` | `generic` | the request shape (below) |
| `LLM_CONNECT_TIMEOUT_S` and `LLM_REQUEST_TIMEOUT_S` | 10.0 and 600.0 | client timeouts |
| `LLM_EXTRA_BODY` | empty | a mapping merged last into every request body |
| `CONTEXT_SIZE` | 32768 | the context window the brain budgets against |

A streamed call waits for the server's next bytes at least
`GENERATION_TIMEOUT_S` plus 60 seconds, so a long silent prefill is not cut
short inside the generation bound that protects it.

### Running the server

Any server that answers the chat-completions format on `LLM_PORT` works.
`scripts/ops/start_llm.sh` launches one of two, chosen by `LLM_BACKEND`:

- `mlx_lm` (Apple Silicon): `python3 -m mlx_lm.server` from the `mlx-lm`
  package.
- `llama_cpp` (elsewhere): `llama-server` from llama.cpp.

The script reads its settings through `python3 -m mtc.core.env_export`, so
it uses exactly the values Settings resolves:

- `mlx_lm`: `python3 -m mlx_lm.server --model MODEL_PATH --host LLM_HOST
  --port LLM_PORT`. The server has no flag for the other settings, so they
  are not passed. `MODEL_PATH` must be a model directory.
- `llama_cpp`: `llama-server -m MODEL_PATH --host LLM_HOST --port LLM_PORT
  -c CONTEXT_SIZE -b BATCH_SIZE`, plus `-ngl GPU_LAYERS` (left out at -1, so
  the server's own default applies), `-t THREADS` (left out at 0) and the
  memory-mapping options for `USE_MMAP` and `USE_MLOCK` in the form the
  installed `llama-server` accepts (the script reads its help text and uses
  `--load-mode` when the build has it, else `--no-mmap` and `--mlock`).
  `MODEL_PATH` must be a model file.

`USE_MLOCK` defaults false because locking a whole model in memory fails
without raised memory-lock limits. An empty `LLM_BACKEND`, a missing model
path or a missing server program exits 2 with a plain message; a port
already answering health exits 0. `--dry-run` prints the command without
running it; `--background` starts it detached with its PID under
`<DATA_DIRECTORY>/run/llm.pid` and its log at `<LOG_DIRECTORY>/llm.log`, and
`--wait SECONDS` then waits for a health path to answer. The script's
tests run it against stub server programs; no model was served on the
development machine for this documentation.

Ollama also serves a chat-completions endpoint at `/v1/chat/completions` on
its own port (its project's release notes document the endpoint); pointing
`LLM_PORT` or `LLM_BASE_URL` at it is an option that was not tested with MTC.

### Memory footprints

A model server needs memory for the weights and for the key-value cache,
which grows with `CONTEXT_SIZE`. As approximate guides only, not
measurements made for this release: about 20 GB for a small model, and about
60 GB for a 27B model at 8-bit quantization. `CONTEXT_SIZE` defaults to 32768 because a
128k context allocates a cache most machines cannot hold; the brain's history
budget is bounded by `HISTORY_TOKEN_BUDGET` either way.

### Cache growth on cache-heavy servers

Some servers keep prompt caches between requests, and the stable prompt
prefix the prompt assembler produces is built to reuse them. On a server that
keeps a cache per conversation, memory use grows with the number of
conversations and can approach the machine's limit over a long run. Watch the
server's memory, cap its cache with the server's own option where it has one,
and restart it in a quiet period when needed (`scripts/ops/stop_stack.sh`
then `start_stack.sh`; see [OPERATIONS.md](OPERATIONS.md)).

## Template families

`LLM_TEMPLATE_FAMILY` (`mtc/brain/families.py`) shapes the request for the
model behind the server:

- `generic` (the default): only the standard fields (`model` when set,
  `messages`, `temperature`, `top_p`, `max_tokens`, `stop`, `stream`,
  `frequency_penalty`). Most installs run a non-reasoning instruct model
  behind a server that may reject unknown fields, so this is the safe choice.
  Reasoning is still read when the server emits it (a `reasoning_content` or
  `reasoning` field, or inline think tags).
- `qwen`: a reasoning model with think tags. Thinking is switched off through
  `chat_template_kwargs` (`enable_thinking` false) when a call does not ask
  for it, the end-of-turn token is added as a stop sequence while thinking,
  and the extended sampling fields (`top_k`, `min_p`, `repetition_penalty`)
  are sent.
- `gemma`: a reasoning model whose template marks thought with channel-style
  tags; its end-of-turn tokens are added as stop sequences while thinking and
  the extended sampling fields are sent. Its tag pairs and stop tokens were
  written without a served template to check them against and are
  unverified; check them against the template your server serves before
  relying on this family, and use `generic` otherwise.

Every family's tags are normalized to `<think>` and `</think>`
(`mtc/brain/thinking.py`).

## Thinking mode

`THINKING_MODE` (true) lets calls ask for reasoning. The reply's reasoning is
kept apart from its spoken text: only the spoken text is stored as the
system's words, and the API returns reasoning separately (`thinking`). A
thinking loop (the same reasoning repeating) is detected and retried up to
`THINKING_LOOP_MAX_RETRIES` times with a raised penalty; a loop that outlives
the retries, an empty reply and a reasoning-only reply raise `NoAnswerError`,
and no reply is stored.

## The Anthropic provider

`LLM_PROVIDER=anthropic` (`mtc/brain/providers/anthropic_provider.py`) needs
the `anthropic` extra, `ANTHROPIC_API_KEY` (read from `ANTHROPIC_API_KEY` or
`MTC_ANTHROPIC_API_KEY`) and `ANTHROPIC_MODEL`; both are required. Using it
sends the system prompt, which carries the persona and recalled memories, and
the conversation to an external API.

- `ANTHROPIC_MAX_TOKENS` (8192) is the completion budget of a direct
  provider call that sets none; conversation turns use `LLM_MAX_TOKENS`, as
  every provider's turns do.
- `ANTHROPIC_THINKING_BUDGET` 0 asks for adaptive thinking when a caller wants
  reasoning (0 is not off); a positive value sends a fixed budget of at least
  1024 tokens and below `ANTHROPIC_MAX_TOKENS`, for models that still accept
  one.
- `ANTHROPIC_SAMPLING_ENABLED` (false) gates the sampling parameters: current
  models reject `temperature`, `top_p` and `top_k`, so none is sent unless it
  is true.
- The health check never calls the API: the provider reads unverified until a
  real call succeeds, so `mtc doctor` makes no paid call.
- Image input is refused: vision is not part of this release.

The DCM evaluator (`mtc/assessment/dcm_evaluator.py`) completes through the
same provider layer when it runs in its Anthropic mode.

## Changing the model

`ModelTransitionProtocol` (`mtc/brain/model_transition.py`) is a tool a
researcher runs around a model change; it is never part of the automatic
cycle.

1. **Baseline.** Before the switch, capture the assessment summary, the
   identity markers an identity source supplies, and the brain's answers to
   the generic probes in `config/transition_prompts.json`. The baseline is
   written to `latest_baseline.json` under `TRANSITION_HISTORY_DIR`
   (`results/model_transitions`).
2. **Swap.** Change `LLM_MODEL`, `MODEL_PATH` or the provider and restart the
   model server and `mtc serve`.
3. **Evaluate.** `evaluate_post_transition` repeats the three and compares
   identity drift, personality drift, score drift and response similarity
   (the Jaccard overlap of the answers). The report is written as
   `transition_<timestamp>.json`.
4. **Decide.** The recommendation is `rollback` when any drift reaches twice
   its threshold or three alerts fire, `review` when any threshold is
   exceeded, `proceed` otherwise, and `insufficient_data` when either
   assessment has no overall score. A probe the brain does not answer is
   excluded and named in `skipped_prompts`.
5. **Roll back** by restoring the previous model setting and restarting.

The probe answers are the system's own words and stay in that private
directory.
