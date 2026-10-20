# Developer bring-up

A development checkout with every extra, the hooks and the generated
artifacts. For a user install, start with [GETTING_STARTED.md](GETTING_STARTED.md).

## Clone and install

```bash
git clone https://github.com/WhiteLotusLA/multi-theory-consciousness.git
cd multi-theory-consciousness
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

`all` holds every extra except `iit`. On Python 3.12 or later, add exact IIT
4.0 Phi with `pip install -e ".[iit]"` (PyPhi from a pinned commit; GPLv3 or
later, never part of `all`, see [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)).
`python3 scripts/license_audit.py --extras all` checks the licences of what
is installed.

Two platform notes from the v0.5.0 release's matrices (fresh virtual
environments on macOS arm64, 2026-10-19). The `graphillion` wheel PyPhi pulls
in for Python 3.12 is linked against Homebrew's gcc runtime and fails to load
on a machine without it (`libstdc++.6.dylib` not found); rebuild it from
source after installing the `iit` extra, which was measured with
graphillion 2.1:
`pip install --no-binary graphillion --force-reinstall --no-deps graphillion`.
The `research` extra's pymc on Python 3.11 resolves a pytensor whose C
backend does not compile on the newest Xcode (`ld: library 'd64' not found`);
`PYTENSOR_FLAGS=cxx=` runs its slower Python backend, and Python 3.12 or
later resolves a pytensor that compiles.

## Hooks and a first test run

```bash
make hooks        # the pre-commit privacy audit of staged changes
make test-fast    # the suite without slow, db and live tests
```

On a machine whose accelerator serves something else, run with the CPU and
without the `gpu` marker:

```bash
MTC_DEVICE=cpu PYPHI_WELCOME_OFF=yes timeout 1500 python3 -m pytest tests -q -p no:cacheprovider -m "not slow and not gpu"
```

The privacy audit needs the private denylist: put it in
`config/privacy/denylist.local.txt` (gitignored) or name it with
`PRIVACY_DENYLIST_EXTRA`. See [PRIVACY_AUDIT.md](PRIVACY_AUDIT.md) and
[TESTING.md](TESTING.md).

## Generated artifacts

Never edit these by hand; regenerate and commit them with the change that
caused them:

| Artifact | Generator | Check |
|---|---|---|
| `.env.example`, `docs/CONFIGURATION.md` | `python3 scripts/gen_env_example.py` | `--check` |
| `docs/schemas/persona.schema.json`, `docs/schemas/onboarding-answers.schema.json` | `python3 scripts/gen_schemas.py` | `--check` |
| `migrations/postgres/*.sql` | `python3 -m mtc.db.migrations --write` | `--check` |
| `mtc/resources/**` | `python3 scripts/sync_resources.py` | `--check` |

`make generate` runs the four generators and `make env-check` the four
checks; CI runs the checks on every push.

## Tests against real stores

The `db` tests run only with `MTC_TEST_STORES=1`, against throwaway names
(`MTC_TEST_DB_PREFIX`, default `mtc_test`, plus eight hex digits) that the
fixtures create and drop. Use servers you own; the fixtures touch nothing
else, and Neo4j skips on the community edition. See [TESTING.md](TESTING.md).

## Adding a Settings key

1. Add the field to `Settings` in `mtc/core/config.py` with a type, a default
   and a description (the description becomes the generated documentation).
   Host, port and URI defaults live only here.
2. Give it a reader in the same change: the module that reads it names it in
   its docstring and reads it at call time.
3. If its prefix belongs to no section of `SECTIONS` in
   `mtc/core/env_example.py`, add it to one (no key may land under Other).
4. Regenerate with `python3 scripts/gen_env_example.py` and commit
   `.env.example` and `docs/CONFIGURATION.md`.

## Adding a store artifact

1. Define the artifact as a module constant with the naming convention
   (`*_SCHEMA_SQL`, `*_INDEXES`, `COLLECTION_PARAMS`, `PAYLOAD_INDEXES`,
   `CONSTRAINTS`, `KEY_FAMILIES`); statements are `CREATE ... IF NOT EXISTS`
   and hold no drop, password, seed row or extension.
2. Register its dotted path in `SCHEMA_ARTIFACTS` in
   `mtc/core/schema_registry.py` (a test fails on an unregistered constant).
3. For PostgreSQL, assign it to a file in `MIGRATION_FILES`
   (`mtc/db/migrations.py`) and run `python3 -m mtc.db.migrations --write`.
4. `mtc db init` and `mtc db check` pick it up from the registry; document it
   in [DATABASES.md](DATABASES.md), where a drift test checks it.

## Adding a subcommand

1. Add the parser in `build_parser()` in `mtc/cli/main.py` and a handler that
   imports its heavy modules inside the handler (importing `mtc.cli.main`
   loads no store client, web framework or HTTP client, and a test checks it).
2. Return the shared exit codes: 0, 1, 2, 3 and 130.
3. Report checks with `mtc.core.checks.CheckStatus`; a check that did not run
   is never `ok`.
4. Add the row to the command table in the [README](../README.md); a drift
   test compares the table with the parser.

## Documents

Documents are written from the code. A count, route, flag, registry name or
threshold is read from the code, the drift tests in `tests/docs` compare
them, and anything not run is marked reviewed, not run.
