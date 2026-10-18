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
are ignored. A purely alphabetic token (letters, apostrophes, spaces) matches,
case-insensitively, where it starts after anything that is not a letter (the
start of a line, a space, punctuation, an underscore or a digit), or at a
camelCase boundary: a lower-case letter, then the token with its first letter
capitalised and not followed by another lower-case letter (`myToken`,
`myTokenBot`). Inside a longer word after a letter it is not a hit
(`xstokeny`, `XSTOKEN`); a suffixed form after a boundary is (`Tokens`,
`Token2`). Any other token matches as a substring. Tokens are also matched in
hidden forms: split literals in Python, escapes, and a folded view without
zero-width characters or Cyrillic and Greek lookalikes (kind `assembled-token`);
an assembled value is folded the same way. A literal-only format the audit
refuses to evaluate (a width or precision of 10000 or more inside a conversion
specifier, written or taken from an operand, or an operand-supplied one that
is not a number) is an `encoding` hit, never dropped; digits, stars and
braces outside a conversion specifier are plain text. In patch mode a merge
commit counts the lines its resolution adds.

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
repository-relative path (`dir/**` covers every file below `dir`); a member of
an archive is reported, and matched, as `<archive>!<member>`, so a glob such
as `paper/submission.tar.gz!main.tex` names one member. The kind is one audit
kind (`nonascii` and `assembled-token` included) or `*`, and the regex is
searched in the reported match. A malformed line or an unknown kind stops the
audit. Every rule needs a reason; never allowlist a private token.
