# Privacy audit

MTC is a public repository ported from a private reference implementation.
The privacy audit keeps identity residue, personal data and build artifacts
out of the tree and out of the history. It fails closed: one hit is a non-zero
exit, and a non-zero exit blocks the commit, the CI run and the phase exit.

## What it checks

- **Tokens.** Two denylists feed one check. `config/privacy/denylist.txt` is
  tracked and holds tokens that are public by design (the reference system's
  name and identifier prefixes, its private directory and database names, its
  vocabulary, model-name patterns, home-path prefixes, the origin date). The
  private extra list holds names of real people and animals and never enters
  the repository: it is read from the file named by `$PRIVACY_DENYLIST_EXTRA`
  (the audit stops if that file is missing), or from the gitignored
  `config/privacy/denylist.local.txt`. Alphabetic tokens match whole words,
  case-insensitive; other tokens match as substrings. Private hits print as
  `private#N` so the output never repeats a name.
- **Patterns.** Email addresses, phone numbers, decimal coordinate pairs,
  absolute home paths, non-loopback IPv4 addresses, key-like strings (hyphenated
  `sk-` provider keys too), em and en dashes, and emoji in `.py` files.
- **Artifacts.** Paths with a `__pycache__`, `CMakeFiles`, `build`,
  `node_modules`, `.DS_Store` or `.egg-info` part; `.pyc`, `.pyo`, `.so`,
  `.dylib`, `.dump` and `.jsonl` files; anything under `agent_space/`,
  `results/` or `logs/`; anything under `data/` except `data/dcm/`.

Only files with an unlisted suffix are skipped. A file that is unreadable, not
UTF-8 or holds a NUL byte is still scanned and also gets an `encoding` hit.

## The allowlist

`config/privacy/allowlist.txt` holds `glob kind regex` lines. The glob matches
the repository-relative path (`dir/**` covers every file below `dir`), the
kind is one audit kind or `*`, and the regex is searched in the reported
match. A rule is justified only when the hit is legitimate and cannot be
rewritten away: license and package attribution, text in a published paper,
the auditor's own patterns and self-test fixtures, a numeric argument that
looks like a coordinate. Keep rules narrow (one file, anchored regex) and give
each a comment. Never allowlist a private token. A malformed line stops the
audit.

## Where it runs

- **Locally:** `make privacy-audit` audits every tracked file;
  `python3 scripts/privacy_audit.py --paths a.py b.md` audits named files.
- **Through the CLI:** `mtc privacy-audit` runs the same audit with the
  repository's lists under `config/privacy` (it exits 2, naming the missing
  list, outside a checkout) and takes the same modes.
- **Pre-commit:** `make hooks` installs a hook that runs `--staged`, which
  audits the staged blobs rather than the working copy.
- **CI:** the `privacy-audit` job writes the private list from the repository
  secret `PRIVACY_DENYLIST_EXTRA` (one token per line) to a temporary file and
  audits the full tree. On pull requests it also runs
  `--patch-from origin/<base>`, which audits every line the branch's commits
  added, file by file, so a hit added and later removed still fails. The job
  fails when the secret is missing, except on fork pull requests, which get no
  secrets and run the public checks with a warning.

## Runtime mode

`python3 scripts/privacy_audit.py --runtime DIR [DIR ...]` audits
directories that need not be inside the repository, such as a project's log
and results directories after a run. Only the artifact rule is off there:
runtime logs legitimately hold `.jsonl` and `.log` files (rotated ones
included), which the artifact rule exists to keep out of the tracked tree,
not out of a log directory, so they are scanned as text instead of flagged.
Every token, pattern and encoding check still applies. Paths print as
`<directory name>/<relative path>`, allowlist rules apply to that path, a
path that is not a directory is an error (never a clean result), and the
audit prints how many files it scanned and how many it skipped (files it
does not read as text, such as `.npz`, `.pkl` or `.db`, counted by suffix)
before its verdict: a skipped file was not audited. The mode cannot
be combined with `--paths`, `--staged` or `--patch-from`
(`run_runtime_audit` in `mtc/devtools/privacy_audit.py`).

The CI install smoke (`.github/workflows/ci.yml`, job `install-smoke`) sets
up a fresh project with `mtc init` from the example answers, runs the
examples (the demo and the continuous-thought example write their state under
`runtime/`) and an assessment into `results/`, then runs the runtime mode over
`runtime` and `results`, so a run that writes a name, a secret-shaped string
or a dash into its own state or results fails CI.

## The wizard's outputs are gitignored

Everything `mtc init` writes (`.env`, `config/persona.yaml`,
`config/speakers.yaml`, the identity statement under `agent_space/`,
`config/ctm/persona.yaml` and the seed packs under `config/ctm/seeds.d/`) is
gitignored, and a test checks every path the wizard writes with
`git check-ignore`. Those files hold the real names an operator chose and
must never be committed; the audit's denylists do not know those names.

## Reading a hit

Each hit prints as `path:line:kind:match`:

```
docs/plan.md:7:token:private#2
pkg/__pycache__/m.cpython-311.pyc:0:artifact:m.cpython-311.pyc
```
Line `0` marks a path-level artifact hit; in `--patch-from` mode lines count
added lines per file.

## What to do on a hit

1. Rewrite the line: "the reference implementation" for the reference
   system's name, delete personal names, make paths repository-relative,
   replace dashes with commas, colons or hyphens.
2. Remove artifacts from the index and extend `.gitignore`.
3. Only if the hit is legitimate and cannot be rewritten, add a narrow,
   commented allowlist rule.

Never copy identity-laden material from the reference implementation and edit
it down: prompts, seeds, persona text and init SQL are rewritten from scratch,
and the reference's values never touch the working tree or the history.
