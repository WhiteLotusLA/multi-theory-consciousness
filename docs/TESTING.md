# Testing

The suite is hermetic by default: it needs no network, no store, no model
server and no accelerator, and it refuses to reach any of them by accident.
This page explains how to run it, what each marker means, and the rules a new
test follows.

## A bounded run

Run the suite with a time bound and without the cache plugin, the way the
gates do:

```bash
PYPHI_WELCOME_OFF=yes timeout 1500 python3 -m pytest tests -q -p no:cacheprovider -m "not slow"
```

`PYPHI_WELCOME_OFF=yes` silences PyPhi's banner when the `iit` extra is
installed. `pyproject.toml` also sets a per-test timeout (pytest-timeout,
900 seconds) and `--strict-markers`, so a misspelled marker is an error.

The Makefile has two shortcuts:

```bash
make test        # the whole suite
make test-fast   # -m "not slow and not db and not live"
```

## Markers

| Marker | Meaning | Default |
|---|---|---|
| `slow` | wall-clock heavy or subprocess tests: the example runs, the wheel build, the baseline runners | runs unless deselected |
| `db` | needs a real store; uses the throwaway fixtures | skipped at collection without `MTC_TEST_STORES=1` |
| `live` | talks to a running stack | never run by default |
| `gpu` | needs MPS or CUDA | runs unless deselected; deselect on a machine whose accelerator is shared |
| `benchmark` | performance measurement, not correctness | runs unless deselected |
| `security` | security-relevant behaviour | runs |

A machine whose accelerator serves something else should add `not gpu` to
every marker expression.

## The hermetic guard

`tests/conftest.py` sets `MTC_DEVICE=cpu` for the session (an exported value
still wins) and installs a network guard: an in-process connection to a
default service port from `Settings` (the API and model server ports always,
the store ports unless `MTC_TEST_STORES=1`) is refused before it leaves the
process, fails the test that made it and is listed at the end of the run. A
test that needs a server starts its own on an ephemeral loopback port (bind
port 0 and read the port back). Child processes are outside the guard, so a
test that starts one gives it an explicit environment: `MTC_DEVICE=cpu`, the
mock provider, every store disabled and an unused API port.
`tests/test_hermetic_guard.py` tests the guard itself.

## Throwaway stores

The `db` tests run against real servers only through five fixtures in
`tests/conftest.py` (`throwaway_postgres`, `throwaway_mongo`,
`throwaway_qdrant`, `throwaway_redis`, `throwaway_neo4j`). They exist because
the machine running the suite may run stores that serve something else: each
fixture creates a name from `MTC_TEST_DB_PREFIX` (default `mtc_test`) plus
eight random hex digits, asserts the name matches that pattern before it
creates it and again before it drops it, and touches nothing else. No fixture
flushes a Redis database or writes to a default graph, database, collection or
key space. Neo4j skips on the community edition, which cannot create a
separate database.

```bash
MTC_TEST_STORES=1 PYPHI_WELCOME_OFF=yes timeout 1500 python3 -m pytest tests -q -p no:cacheprovider -m db -rs
```

Run that only against servers you own; `-rs` lists every skip with its
reason. The optional CI workflow `.github/workflows/db-stores.yml` runs the
`db` tests the same way against service containers (PostgreSQL 17, MongoDB 8,
Redis 7 and Qdrant; Neo4j is not in it), on manual dispatch and weekly; the
per-push CI runs the hermetic suite.

## Synthetic fixtures

Fixtures use synthetic data only: the speaker names `Alice`, `Operator` and
`Guest`, the sample companion `Tessera`, a system name from `SYSTEM_NAME` or a
synthetic value, and fixed instants on 2000-01-01. Where a calendar day
matters, tests are parametrized over `UTC`, `Europe/Berlin` and `Asia/Tokyo`.
Never use a real person's name in any form, split or encoded; the privacy
audit assembles literal pieces and catches it.

## The mock provider

Tests that need a brain use `MockProvider` (deterministic, labelled `[mock]`
replies, scriptable failures) or the fake brain of the brain protocol; HTTP
clients are tested with `httpx.MockTransport` or a stdlib server on an
ephemeral port. No test calls a real model.

## Baseline tests

- `tests/test_legacy_preset.py`: the `legacy_v04` preset reproduces the
  recorded v0.4.0 experiment-runner output
  (`tests/fixtures/baseline_v040_experiment.txt`) to 0.01 on the overall score
  with an equal passing count.
- `tests/test_v050_baseline.py`: the v0.5.0 defaults reproduce
  `tests/fixtures/baseline_v050_experiment.txt` and
  `tests/fixtures/baseline_v050_assessment.json` at the same tolerance.
- `tests/test_run_assessment.py`: the assessment runner and its flags.

A change that moves one of these numbers is a measurement change; record it in
[MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md) with the switch that moves it.

## The wheel test

`tests/packaging/test_wheel_install.py` (slow) builds a wheel, installs it
without dependencies into a virtual environment that sees the installed
packages, and from a directory outside the repository checks that the
package, the console script, every shipped resource, `mtc init`,
`mtc doctor` and the full pipeline demo work from the installed copy. It
needs no network.

## Documentation drift tests

`tests/docs/` compares the documents with the code: the indicator counts, the
CLI table with the parser, the API page with the application's routes, the
databases page with the schema registry, the knowledge-graph page with the
models and constraints, the first-run thresholds with their constants, the
architecture tree with the packages, the indicator mapping with the indicator
configurations, the onboarding type table with the shipped type defaults, the
relative links, and the reviewed-not-run statements of the setup pages. A
document that drifts from the code fails a test.

## Generated artifacts

Some files are generated and checked, never edited by hand. Each generator has
a `--check` mode, and `make env-check` runs all of them:

| Artifact | Generator |
|---|---|
| `.env.example`, `docs/CONFIGURATION.md` | `python3 scripts/gen_env_example.py` |
| `docs/schemas/*.json` | `python3 scripts/gen_schemas.py` |
| `migrations/postgres/*.sql` | `python3 -m mtc.db.migrations --write` |
| `mtc/resources/**` | `python3 scripts/sync_resources.py` |

`make generate` runs the four generators. After changing a Settings field, a
schema model, a registry artifact or a shipped default, regenerate and commit
the result with the change.

## The privacy audit

`make privacy-audit` audits every tracked file; the pre-commit hook
(`make hooks`) audits staged changes. See [PRIVACY_AUDIT.md](PRIVACY_AUDIT.md).

## Wall-clock tests

Code that waits takes an injected clock, so tests never sleep for real time.
Tests that must use wall time (subprocesses, servers, the example runs) are
marked `slow` and bound every wait.

## Adding a test

1. Write the test first, named for the property it pins, in the directory
   that mirrors the module (`tests/<package>/test_<module>.py`).
2. Make it able to fail: an assertion that cannot fail (`or True`,
   `assert True`, a body that is only `pass`, a condition that is always
   true) proves nothing, and the gates sweep for them.
3. Use synthetic data and fixed instants; no network, store or accelerator
   unless the test is marked for it.
4. Mark it `slow` when it starts a subprocess or takes wall time, `db` when it
   needs a real store (and use the throwaway fixtures), `gpu` when it needs an
   accelerator.
5. Never weaken, skip or delete an existing test to make a change pass, and
   never widen a tolerance; a `pytest.skip` always states its reason.
