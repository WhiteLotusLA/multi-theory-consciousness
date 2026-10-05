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
- **Pre-commit:** `make hooks` installs a hook that runs `--staged`, which
  audits the staged blobs rather than the working copy.
- **CI:** the `privacy-audit` job writes the private list from the repository
  secret `PRIVACY_DENYLIST_EXTRA` (one token per line) to a temporary file and
  audits the full tree. On pull requests it also runs
  `--patch-from origin/<base>`, which audits every line the branch's commits
  added, file by file, so a hit added and later removed still fails. The job
  fails when the secret is missing, except on fork pull requests, which get no
  secrets and run the public checks with a warning.

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
