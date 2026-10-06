# Contributing to the Multi-Theory Consciousness Framework

Thank you for your interest in contributing. This project sits at the intersection of computer science, neuroscience and philosophy of mind, and welcomes contributions from all three communities and from anyone willing to engage with the material seriously.

## Before You Begin

Read [docs/HONEST_LIMITATIONS.md](docs/HONEST_LIMITATIONS.md). Understanding what this framework cannot do is as important as understanding what it can do. Then set up a development checkout with [docs/DEVELOPER_BRINGUP.md](docs/DEVELOPER_BRINGUP.md).

## Reporting Issues

Open an issue on GitHub for:
- **Bugs**: unexpected behavior, crashes, incorrect measurements
- **Theory concerns**: an implementation that misrepresents the theory it claims to model
- **Documentation gaps**: unclear explanations, missing context, a document that disagrees with the code
- **Feature requests**: new theories, indicators or neural substrates

Include your Python version and OS, the steps to reproduce, the expected and actual behavior, and the relevant output of `mtc doctor --json` (it reports secrets as `set` or `empty`, never their values). Report security problems privately; see [docs/SECURITY.md](docs/SECURITY.md).

## Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Write the test first, then the code (see [docs/TESTING.md](docs/TESTING.md)).
3. Run a bounded test run, the linters and the privacy audit:
   ```bash
   PYPHI_WELCOME_OFF=yes timeout 1500 python3 -m pytest tests -q -p no:cacheprovider -m "not slow"
   make lint
   make privacy-audit
   make env-check
   ```
4. Open the PR with what the change does, why, which theories or modules it affects, and its trade-offs. A change that moves an assessment number names the switch that moves it and records the before and after values in [docs/MIGRATION_FROM_V04.md](docs/MIGRATION_FROM_V04.md).

Reviews check correctness against the theory, tests that can fail, documentation, style, and honesty: code and documents must not claim more than they measure.

## Rules every change follows

### The privacy audit

The audit (`scripts/privacy_audit.py`, [docs/PRIVACY_AUDIT.md](docs/PRIVACY_AUDIT.md)) fails closed on personal data, identity residue and build artifacts. Install the pre-commit hook with `make hooks`; CI runs the audit on every push and on every line a pull request adds. Fix a hit by rewriting the line; add a narrow, commented allowlist rule only when the hit is legitimate and cannot be rewritten.

### Never copy from the reference implementation

MTC is a public framework written for this repository. Prompts, persona text, seed packs, templates, SQL, scripts and documents are written fresh from the framework's own code and design; nothing is copied from a private implementation and edited down.

### Synthetic fixtures

Tests and examples use synthetic data only: the speaker names `Alice`, `Operator` and `Guest`, the sample companion `Tessera`, a system name from `SYSTEM_NAME` or a synthetic value, and fixed instants on 2000-01-01. Never write a real person's name, in any form, in code, comments, tests, configuration or documents; published scientific citations in the theory documents are the exception.

### No dashes, no emoji

Write no em dash or en dash anywhere (use commas, colons, parentheses or a plain hyphen), and no emoji. Python source is ASCII only. The privacy audit checks both.

### Bounded test runs

Every test run has a time bound (`timeout 1500` around pytest, plus the per-test timeout in `pyproject.toml`). A test never sleeps for real time when an injected clock can do, never reaches the network or a store unless it is marked for it, and never claims an accelerator unless marked `gpu`.

### Tests that can fail

Never weaken, skip, xfail or delete a test to make a change pass, and never widen a tolerance. An assertion that cannot fail (`or True`, `assert True`, a body that is only `pass`) is not a test.

### Generated artifacts

`.env.example`, `docs/CONFIGURATION.md`, `docs/schemas/*.json`, `migrations/postgres/*.sql` and `mtc/resources/**` are generated, never edited by hand. `make generate` regenerates them; `make env-check` checks them, and so does CI.

### Every Settings key has a reader

Configuration lives in `mtc/core/config.py`. A new key needs a reader in the same change, a description (it becomes the generated documentation), and a regenerated `.env.example`. Defaults for hosts, ports and URIs live only there: no other module writes a host, port or URI literal. Store clients are imported lazily through `mtc.memory.clients.require_client`.

### Identity is configuration

Names, relationships, the companion type and the persona come from configuration the operator writes (`mtc init`), never from code. A name never grants trust: only `mtc/pipeline/speaker_trust.py` reads the trust settings.

### Documents are written from the code

Every count, route, flag, registry name and threshold in a document is read from the code, and the drift tests in `tests/docs` compare them. Mark anything that was not run as reviewed, not run, and claim no speed, size or test count that a run did not produce.

## Code Style

- **Formatter**: black, line length 88
- **Linter**: flake8, max line length 88
- **Type hints** on public functions
- **Docstrings** in the third person, naming the module's config keys, the files and stores it writes and the private data it touches; no author lines or dates
- **Logging** through `logging.getLogger(__name__)`, ASCII-only messages, never a secret, a message text or a name

## Consciousness Module Guidelines

### Before changing a theory module

1. **Read the original papers.** Each module lists its theoretical basis. Do not change an implementation on intuition alone.
2. **Understand the indicator mapping.** Each module feeds specific indicators (see [docs/BUTLIN_2025_MAPPING.md](docs/BUTLIN_2025_MAPPING.md)). A change that alters a measurement goes behind a switch whose v0.4.0 value lands in the `legacy_v04` preset.
3. **Do not stub.** Every method computes a meaningful value or reports that it could not (`insufficient_data`, None); a constant never stands in for a missing measurement.
4. **Document simplifications.** These are computational approximations; say what was simplified and why.

### Adding a theory or an indicator

1. Add the module under `mtc/consciousness/` (or a subpackage).
2. Add the indicator to `ConsciousnessAssessment.indicator_configs` in `mtc/assessment/assessment.py` with its theory, threshold, weight and metadata, and its `_v05_` measurement method.
3. Write tests for a passing case, a failing case and the `insufficient_data` case.
4. Update [docs/CONSCIOUSNESS_THEORIES.md](docs/CONSCIOUSNESS_THEORIES.md), [docs/BUTLIN_2025_MAPPING.md](docs/BUTLIN_2025_MAPPING.md) and the README; the drift tests fail until the counts agree.

## Commit Messages

```
feat(consciousness): add RPT local recurrence measurement

Measures local recurrence from the SNN's feedback statistics. The
threshold of 0.3 sits above the noise baseline of the runner's warmup.
```

Prefixes: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.

## Questions?

Open a GitHub Discussion or an issue.
