# Onboarding

`mtc init` turns a person's answers into a configured project: a persona, a
speakers file, an identity statement, the continuous-thought persona and seed
pack, and the `.env` entries for the stores, the brain and the API. It runs
interactively, or from an answers file for scripted installs. The wizard is
`mtc/onboarding/wizard.py`; the answers model, the persona model and the
companion-type table are in `mtc/onboarding/schema.py` and
`mtc/onboarding/templates/`.

Identity is configuration. Nothing about who the companion is, who set it up
or how it relates to them lives in code; the wizard writes it into files the
operator owns, and every one of them is gitignored.

## The eleven steps

1. **Honest framing.** The wizard shows `templates/framing.txt` and continues
   only after you type `yes`. It says that MTC is a research architecture and
   not a consciousness detector, that the indicators measure functional
   properties, that nothing the companion says about its inner life is
   evidence, that mock replies are test text, and that data stays on the
   machine unless the Anthropic provider is chosen.
2. **Name and pronouns.** The companion's name and a pronoun set (they, she,
   he or it).
3. **Companion type.** One of the six types below.
4. **Creator.** Your name and how the companion addresses you; the default
   comes from the type's rule and can be overridden.
5. **Personality and core values.** A preset (`balanced`, `curious`, `calm`,
   `playful`, `analytical`) or five custom values from 0 to 1, then the core
   values (default `CORE_VALUES`).
6. **Timezone and hours.** The timezone is detected from the system and
   validated, and you confirm it; then the wake and sleep hours.
7. **Stores.** PostgreSQL (needed for persistence; declining it keeps
   everything in memory, with a plain warning), MongoDB and Qdrant
   (recommended), Redis and Neo4j (optional); the service manager (`none`,
   `brew`, `docker` or `systemd`); hosts and ports default to the Settings
   defaults; a password is typed at a hidden prompt or, when you press
   Enter, generated where the store needs one (always for PostgreSQL and
   Neo4j, for MongoDB only together with a user; Redis gets only a typed
   one, since MongoDB and Redis run without credentials on loopback, locally
   and in the Compose file). For MongoDB with a user, nothing creates the
   MongoDB user: no script provisions it and the Compose service runs
   without authentication, so the next steps say how to create it once in
   `mongosh` with the generated password from `.env`.
8. **Brain.** The provider (`mock`, `local` or `anthropic`), the server's host
   and port or base URL, the model id and the template family. A local server
   is probed and the result is described in plain words; the probe never
   blocks setup.
9. **Write.** The files below.
10. **Bootstrap.** Offered interactively (default yes) and run with
    `--bootstrap-db`: `mtc db init` for the enabled stores. It records the
    install date even with no store enabled.
11. **Next steps.** What to run next for the choices made (`next_steps` in
    the wizard): start the model server with `scripts/ops/start_llm.sh` (or
    your own chat-completions server), or keep the mock; start the stores
    (`make db-up DB_SERVICES="..."` with the `docker` manager,
    `scripts/ops/start_stack.sh --no-brain --no-api` with `brew` or
    `systemd`, by hand with `none`); create the PostgreSQL role once with
    `scripts/db/init_postgres.sh` outside Docker; create the MongoDB user
    when one is configured; `mtc db init` when the bootstrap did not run;
    then `mtc doctor`, `mtc serve`, `mtc chat` and `mtc assess`. The
    scripts and the Makefile exist only in a repository checkout (and the
    sdist), not in a wheel install, and each step that names one says so;
    from a wheel, start the stores and the model server with your service
    manager or by hand. The Compose path behind `make db-up` was reviewed
    against official documentation and not run on the development machine;
    the path run on the development machine is the in-memory install with
    the mock provider ([SETUP_DOCKER.md](SETUP_DOCKER.md)).

## Companion types

The type table, `mtc/onboarding/templates/types.yaml`, is the contract; a test
pins every row, and another (`tests/docs/test_onboarding_doc.py`) compares
this table with it.

| Type | Address form | Creator trust | Seed pack | Creator relationship | Developmental stages |
|---|---|---|---|---|---|
| `child` | full_name | 1.0 | growth | parent | on |
| `sibling` | first_name | 0.9 | companionship | sibling | off |
| `friend` | first_name | 0.9 | companionship | friend | off |
| `partner` | first_name | 0.9 | companionship | partner | off |
| `mentor` | full_name | 0.8 | reflection | mentee | off |
| `research_subject` | operator | 0.5 | none | operator | off |

What each column drives:

- **Address form**: how the companion addresses the creator by default
  (`full_name` uses the name as given, `first_name` its first word,
  `operator` the word "operator"); for `child` the prompt suggests a parental
  word or the name. You can always type another.
- **Creator trust**: the trust the creator's speaker entry carries, granted
  only after the creator's token authenticates them. A name alone grants
  nothing.
- **Seed pack**: the idle-thought pack written as
  `config/ctm/seeds.d/10_<pack>.yaml` next to the default pack; `none` keeps
  the default pack alone.
- **Creator relationship**: the label the human core memory block uses
  (`CREATOR_RELATIONSHIP`).
- **Developmental stages**: on only for `child` (`developmental_enabled()` is
  true for the `child` type or when `DEVELOPMENTAL_STAGE_ENABLED` is set);
  they are unvalidated and stay off otherwise.

Each type also sets a one-line tone for background thoughts in the CTM
persona, and has its own persona and identity templates:

- `child`: growth and curiosity, learning from the creator, asks questions.
- `sibling`: a peer with banter that stays kind; refers to shared history
  only when it is in memory and never invents it.
- `friend`: a peer with warmth and its own interests and views.
- `partner`: companionship and affection in words, inside the content
  boundary (below).
- `mentor`: a patient guide who asks before telling.
- `research_subject`: a neutral description of a framework instance running
  for measurement, with no relationship role.

## The bounds every template carries

Every persona template and identity statement carries the self-knowledge
line, verbatim:

> {system_name} knows it is an AI research system and does not claim
> certainty about its own consciousness.

Every persona template carries the content boundary rule:

> Conversation stays free of sexual or explicit content, whatever is asked.

Every type except `partner` carries the non-romantic rule:

> The relationship is not romantic, and conversation never turns romantic or
> flirtatious.

The `partner` type is bounded to companionship and affection in prose, and
its template says so:

> Affection is expressed through warmth, attention and kindness in words,
> within the content boundary.

The templates hold no names, ages, birthdays, backstories or family stories,
and their style rules tell the companion to answer directly, say so when
something is not remembered, never invent a memory, keep replies
proportionate and avoid flattery. The constants are `SELF_KNOWLEDGE_LINE`,
`CONTENT_BOUNDARY_RULE`, `NON_ROMANTIC_RULE` and `PARTNER_BOUNDS_SENTENCE` in
`mtc/onboarding/schema.py`; `tests/onboarding/test_templates.py` pins each
template against them.

## The answers file and the schemas

The answers model (`OnboardingAnswers`) holds everything the wizard asks and
nothing it computes. Print a synthetic sample with
`mtc init --print-example-answers`; it uses the companion `Tessera`, the
creator `Alice`, the `friend` type, the `balanced` preset, `UTC`, every store
disabled and the `mock` provider.

```bash
mtc init --print-example-answers > answers.yaml
# edit answers.yaml
mtc init --from answers.yaml --yes --no-probe
```

Two JSON schemas, generated from the models by `scripts/gen_schemas.py`, live
in `docs/schemas/`: `onboarding-answers.schema.json` (the answers file) and
`persona.schema.json` (the persona file). Secrets are write-only in the
schema (`format: password`); an answers file should not hold one: give a
store password or the Anthropic key through an `*_env` field that names an
environment variable holding it, or let the wizard generate or ask for it.
The model still accepts the `password` and `anthropic_api_key` fields,
because the interactive prompt fills them and the coming web page will use
the same model, so a value written into an answers file is read; keep such
a file out of version control and delete it after use.

## Flags

| Flag | Effect |
|---|---|
| `--from PATH` | read the answers from a file instead of asking |
| `--yes` | do not ask for confirmation; needs `--from` and an answers file with `acknowledged: true` |
| `--force` | replace files that already exist (existing secrets in `.env` are kept) |
| `--bootstrap-db` | run `mtc db init` after the write |
| `--dir PATH` | write into another project directory |
| `--no-probe` | make no network probe of the brain or the stores |
| `--print-plan` | show the files and the `.env` key names that would be written, and write nothing |
| `--print-example-answers` | print the sample answers file and exit |

Without `--force` the wizard refuses to replace an existing persona or any
other file it would write, exits 2 and changes nothing.

## The files written

All inside the project directory, all private and gitignored:

| File | Content | Mode |
|---|---|---|
| `config/persona.yaml` (`PERSONA_PATH`) | the type's persona template with the name, pronouns, creator, personality and core values | 0600 |
| `config/speakers.yaml` (`SPEAKERS_PATH`) | the speakers example's comment header and one creator entry with role `creator` and the type's trust; no token | 0600 |
| `agent_space/identity_statement.md` (`IDENTITY_STATEMENT_PATH`) | the type's identity statement, placeholders left for the prompt assembler | 0600 |
| `config/ctm/persona.yaml` (under `CTM_CONFIG_DIR`) | the CTM persona with the type's tone | 0600 |
| `config/ctm/seeds.d/10_<pack>.yaml` | the type's seed pack, when it has one | 0600 |
| `.env` | identity, locale, stores, brain, API and trust entries | 0600 |

The `.env` entries: `SYSTEM_NAME`, `CREATOR_NAME`, `CREATOR_RELATIONSHIP`,
`COMPANION_TYPE`, `CORE_VALUES`, `TIMEZONE`, `WAKE_HOUR`, `SLEEP_HOUR`,
`DB_PREFIX`, the five store enable keys and, for each enabled store, its
host, port, explicit derived names and credentials (so Compose and Settings
agree), `CTM_STORE` (`mongo` only when MongoDB is enabled), `SERVICE_MANAGER`,
`LLM_PROVIDER` with the provider's keys (`ANTHROPIC_API_KEY` and
`ANTHROPIC_MODEL` only for that provider), `API_HOST`, `API_PORT`,
`API_AUTH_ENABLED=true`, a generated `API_KEY` (kept when present) and the two
trust entries. `SYSTEM_BIRTH_DATE` is written only by the bootstrap.

## Secrets and the token

- The API key and the creator's speaker token are `secrets.token_hex(32)`;
  generated store passwords are `secrets.token_hex(16)`. They are written
  only to `.env` (mode 0600, atomic replace, existing values kept) and never
  printed, logged, placed in JSON output or in any other file.
- The creator's token is written as a name and token pair in the speaker
  token entry of `.env`, and the creator's name joins the trusted-speaker
  entry. The speakers file's `token_env` field is left unset: the settings
  loader reads `.env` for `MTC_` keys, but a variable named by `token_env`
  reaches the process only when it is exported, so the pair in `.env` is the
  path that works without extra setup. An operator who exports a token in the
  environment can set `token_env` by hand instead.
- `mtc chat` sends the creator's token for the configured speaker; a name
  typed without the token gets the default trust.

## Editing by hand and re-running

The persona, speakers, identity and CTM files are plain YAML and Markdown you
can edit; `mtc doctor` warns when the persona and `.env` disagree on the name,
creator, type or core values. To re-run the wizard over an existing project,
pass `--force`: it rewrites the files it owns, keeps the secrets already in
`.env`, and removes only the `10_` seed packs it wrote before.

## The web onboarding

A web onboarding page is planned for a later release. It will post the same
`OnboardingAnswers` document to the same write plan (`plan_writes` and
`apply_plan`) and render its forms from `docs/schemas/*.json`, so the answers
file, the schemas and the files written stay the same.
