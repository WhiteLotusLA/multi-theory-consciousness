# Privacy lists

These files feed `scripts/privacy_audit.py`. See `docs/PRIVACY_AUDIT.md` for
the full gate.

## The two denylists

- `denylist.txt` (tracked): tokens that are public by design, such as the
  reference system's name and identifier prefixes, its private directory and
  database names, its vocabulary, model-name patterns, home-path prefixes and
  the origin date.
- The private extra list (never tracked): names of real people and animals.
  Publishing them in a guard list would publish the very thing the guard
  protects, so they live outside the repository.

Format for both: one token per line; blank lines and lines starting with `#`
are ignored. Purely alphabetic tokens (letters, apostrophes, spaces) match as
whole words, case-insensitive. Any other token matches as a substring.

## Where the private list comes from

1. `$PRIVACY_DENYLIST_EXTRA`, when set, is the path to the private file. If the
   variable is set and the file is missing, the audit stops with an error.
2. Otherwise `config/privacy/denylist.local.txt`, which `.gitignore` excludes.

In CI the job writes the private list from the repository secret
`PRIVACY_DENYLIST_EXTRA` (one token per line) to a temporary file and points
the variable at it. Hits on private tokens print as `private#N` (the 1-based
line position in that list), never as the token itself.

## The allowlist

`allowlist.txt` lines are `glob kind regex`. The glob is matched against the
repository-relative path (`dir/**` covers every file below `dir`), the kind is
one audit kind or `*`, and the regex is searched in the reported match. A
malformed line or an unknown kind stops the audit. Every rule needs a reason;
never allowlist a private token.
