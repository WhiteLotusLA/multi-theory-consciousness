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

The baseline tests compare every metric line and every report field, through
the pure comparators of `tests/baseline_compare.py`:

- An experiment-runner run is compared as the block the runner prints after
  its `---` line: every metric line, as the printed string, so the
  comparison holds at the runner's print resolution. Only the two timing
  lines (`processing_ms`, `total_seconds`) are left out.
- A seeded assessment report (the JSON `mtc assess --json` writes) is
  compared field by field: statuses, counts, keys, strings and booleans
  exactly, floats within 1e-9. Only the session id, the timestamp and the
  timing fields are left out. A top-level mapping the report gained after a
  fixture was captured (`failures`) reads as empty when the fixture lacks
  it.
- When a comparison fails, the message lists every differing line or field
  and then how the running environment differs from the one recorded in
  `tests/fixtures/baseline_v050_env.txt`, so an environment change is told
  apart from a code change.

The tests:

- `tests/test_legacy_preset.py`: the `legacy_v04` preset prints every metric
  line of the recorded v0.4.0 experiment-runner output
  (`tests/fixtures/baseline_v040_experiment.txt`) again as the same string,
  plus the three lines added since v0.4.0 and no other line; the seeded
  assessment under the preset reproduces every field of the legacy
  assessment fixture (`tests/fixtures/baseline_legacy_assessment.json`, the
  report of the published legacy run) except the three wall-clock readouts
  named below; every row of the parameter schema has
  a legacy value, a v0.4.0 default or a recorded reason.
- `tests/test_v050_baseline.py`: the v0.5.0 defaults reproduce
  `tests/fixtures/baseline_v050_experiment.txt` line for line and
  `tests/fixtures/baseline_v050_assessment.json` field for field.
- `tests/test_run_experiment.py` and `tests/test_run_assessment.py`: two runs
  of one seed agree over every line and every field, and the runners' flags,
  ablation printing and state-directory restore work.

Three evidence values of the seeded assessment, `hots_generated` and
`coverage_ratio` of `higher_order_representations` and `introspections` of
`metacognition`, follow the machine's speed while the introspection cooldown
is a wall-clock gate, and under the legacy preset no score, status,
confidence, count or printed line depends on them; the comparison of a run
that keeps the wall-clock cooldown (the legacy preset) leaves exactly these
three keys out (`WALL_CLOCK_EVIDENCE_KEYS`), and the legacy seeded assessment
run at clock speeds 1, 1 and 25 differs in exactly those three readouts and
otherwise only in its timings, timestamps and session id. What makes the
v0.5.0 fixtures comparable in full is the runner pin, not that allowance:
both runners pin `hot.introspection_cooldown` at 0.0 unless the overrides
set it, so the
default seeded assessment reads 14, 2.8 and 3 for those three values on
every machine (under the defaults the 1.0 s gate would also move the
metacognition score). The pin removes the gate, not every clock read: the
attention bottleneck's recency bonus (0.1 x exp(-0.1 x a candidate's age in
seconds) in every composite salience), while it aged candidates on the wall
clock, moved `genuine_implementation.evidence.avg_loop_quality` by 1.2e-9 when
every clock ran 25 times faster, and moved no score, status or count. The
bonus now ages candidates on the workspace's simulated clock
(`gwt.recency_clock`, default `simulated`; `v04_wall` restores the v0.4.0
wall-clock age), so under the defaults the seeded assessment reads no wall
clock into a score or an evidence value: two runs of one seed at normal
speed and one with every clock 25 times faster differ only in their timings,
timestamps and session id. The measurements are recorded under "What the
cooldown pin removes" in [MIGRATION_FROM_V04.md](MIGRATION_FROM_V04.md), and
the 1e-9 bound was not widened for it.

The baseline tests start the runners as subprocesses and are marked `slow`
(one runner run takes one to two minutes on the development machine). Three noise tests in `tests/assessment/test_recalibration.py`
(`test_noise_normalized_below_thresholds`,
`test_no_perfect_scores_after_normalization`,
`test_overall_score_near_zero_on_noise`) drive the experiment runner's
modules in process and are not marked `slow`: they take about a minute
each, so they run in every default run of the suite.

The per-push CI deselects the `slow` tests, so the CI job `published-numbers`
(`.github/workflows/ci.yml`) runs the tests that hold the published numbers:
the four baseline and runner files above, `tests/assessment/test_phi_exact_path.py`
and `tests/assessment/test_dcm_benchmark.py`, with `-m "not db and not gpu"`,
on pull requests to main, nightly and by hand. It runs on the platform the
fixtures were recorded on (`macos-14`, Python 3.14, a test pins the version
against the environment file) with numpy, scipy, torch, snntorch and
reservoirpy pinned to the versions `tests/fixtures/baseline_v050_env.txt`
records, read from that file at run time, and installs the `neural`, `fep`,
`research`, `iit` and `dev` extras. Regenerating the fixtures and that file
moves the job with them.

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
