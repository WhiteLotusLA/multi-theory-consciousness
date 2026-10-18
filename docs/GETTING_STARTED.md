# Getting Started

MTC installs as a Python package with one command, `mtc`. The setup wizard
(`mtc init`) writes the configuration; everything else reads it. Start with
the first path below: its commands need no database, no model server and no
network after the install, and they show the whole pipeline running. One
qualifier: `mtc serve` with `mtc chat` on an install with the `all` extra
loads the sentence embedder, and the model hub library makes a request on
that first load even with the model cached (and downloads the model when it
is not); set `HF_HUB_OFFLINE=1` once the model is cached (`mtc doctor
--fetch-models` fetches it) to keep that path off the network too.

Read [HONEST_LIMITATIONS.md](HONEST_LIMITATIONS.md) first: MTC is a research
architecture, not a consciousness detector.

## Prerequisites

- Python 3.11 or later (3.12 or later for the optional `iit` extra).
- Git and pip.
- For the store paths: Docker with Compose v2, or Homebrew (macOS) or your
  distribution's packages (Linux).

## Path 1: in memory, with the mock provider

No database and no model: every store stays in process memory and the mock
provider answers with labelled test text. This is the path run on the
development machine.

```bash
git clone https://github.com/WhiteLotusLA/multi-theory-consciousness.git
cd multi-theory-consciousness
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

mtc init --print-example-answers > answers.yaml
mtc init --from answers.yaml --yes --no-probe
mtc doctor
mtc assess
python examples/full_pipeline_demo.py
```

- `pip install -e ".[all]"` installs the core with every extra except `iit`:
  the neural substrates, active inference, the store clients, the model
  clients, the API server, the embedder and the research tools. A bare
  `pip install -e .` installs the core alone, which imports and configures
  but cannot build the active inference module. The minimum install that can
  run the assessment is the `fep` extra (pymdp): on a bare install
  `mtc assess`, `mtc serve`, the demo and the examples that build a
  workspace each stop with one line on stderr that names the extra
  (`MISSING EXTRA: the fep extra is not installed. ...`), with no traceback,
  and exit with code 2. `mtc doctor` names the extras that are missing and
  the indicators that read active inference.
- `mtc init --print-example-answers` prints a synthetic answers file (the
  companion `Tessera`, the creator `Alice`, the `friend` type, every store
  disabled, the `mock` provider). Edit it, or run `mtc init` without flags
  for the interactive wizard. The wizard writes `.env` (mode 0600, with a
  generated API key), the persona, the speakers file and the identity
  statement, all gitignored. See [ONBOARDING.md](ONBOARDING.md).
- `mtc doctor` checks the install and says what it could not measure
  ([FIRST_RUN.md](FIRST_RUN.md) explains a fresh install's warnings).
- `mtc assess` (the quick mode, its default) assesses freshly built modules
  without a warmup; most indicators read `insufficient_data` until modules
  have run, and the report says so. `mtc assess --full` warms the modules up
  first.
- The full pipeline demo runs conversation turns through the whole runtime on
  the mock provider and prints a summary.

## Path 2: the stores with Docker Compose

The Compose file was reviewed against official documentation and not run on
the development machine; the path run on the development machine is the
in-memory install with the mock provider (Path 1).

```bash
pip install -e ".[all]"
mtc init                                  # enable the stores, service manager docker
make db-up DB_SERVICES="postgres mongo qdrant"
mtc db init
mtc db check
mtc doctor --fetch-models                 # the embedding model, once
```

See [SETUP_DOCKER.md](SETUP_DOCKER.md).

## Path 3: the stores as host services (Homebrew or apt)

```bash
pip install -e ".[all]"
mtc init                                  # enable the stores, service manager brew
scripts/db/init_postgres.sh               # the PostgreSQL role and database
mtc db init
mtc db check
scripts/ops/start_stack.sh                # stores, model server and API
```

See [SETUP_MACOS.md](SETUP_MACOS.md) for the formulas, the Qdrant binary and
the Linux notes.

## A model

The mock provider is the default the example answers choose. For a real
model, run a chat-completions server on `LLM_PORT` (mlx_lm on Apple Silicon,
llama.cpp elsewhere; `scripts/ops/start_llm.sh` launches either) and choose
the `local` provider in `mtc init`, or choose the `anthropic` provider, which
sends prompts to an external service. See [LLM_SETUP.md](LLM_SETUP.md).

## Configuration

Every setting is a field of `Settings` in `mtc/core/config.py`, read from
the `MTC_`-prefixed environment or `.env`. [CONFIGURATION.md](CONFIGURATION.md)
lists every key with its default; `.env.example` is generated from the same
source. A project with no `.env` works: every setting has a default (and the
API refuses every request until an `API_KEY` exists, which `mtc init`
generates).

## The API and chat

```bash
mtc serve                         # the API on 127.0.0.1:8200 by default
mtc chat                          # an interactive chat through the server
mtc chat --message "hello"        # one message
mtc ctm status                    # the continuous thought machine
```

`mtc serve` binds loopback and requires the API key; `mtc chat` reads the key
and the creator's speaker token from the project's settings. See
[API.md](API.md) and [SECURITY.md](SECURITY.md).

## Tests

```bash
pip install -e ".[neural,fep,llm,anthropic,api,databases,dev]"
PYPHI_WELCOME_OFF=yes timeout 1500 python3 -m pytest tests -q -p no:cacheprovider -m "not slow"
```

These are the extras the CI tests job installs: the core with `dev` alone
cannot run the suite, because the workspace builds active inference (the
`fep` extra) and many tests need the `neural`, `api` and `llm` extras.
The `all` extra covers them too. The suite needs no network, store
or model. See [TESTING.md](TESTING.md).

## Verification

The fresh-machine verification of this release (a clean clone, a
non-editable install, the commands of Path 1, `mtc serve` and `mtc chat` on a
free port) is pending; this note will record the date, the Python version and
the path it tested when its report is in. Until then: the `mtc` commands of
Path 1 (`init --print-example-answers`, `init --from ... --yes --no-probe`,
`doctor` with `--no-network`, `assess`) and the demo were run from a
temporary project directory on the development machine with the mock
provider, through the same entry point the console script calls
(`python3 -m mtc.cli.main`), and each exited 0; the clone and the
`pip install` were not run there; the commands of Paths 2 and 3 were
reviewed, not run.

## Next

- [FIRST_RUN.md](FIRST_RUN.md): what a fresh install measures and why.
- [ARCHITECTURE.md](ARCHITECTURE.md) and [RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md): how the parts fit.
- [CONSCIOUSNESS_THEORIES.md](CONSCIOUSNESS_THEORIES.md): the eight theories and the assessment method.
- [OPERATIONS.md](OPERATIONS.md): running the stack day to day.
