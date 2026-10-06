# Third-party licences

Multi-Theory Consciousness (MTC) is licensed under Apache-2.0. It vendors no
third-party code: every dependency below is installed by pip from its own
distribution, and the optional ones only when you ask for their extra.

## Direct dependencies

Every direct dependency of the core install and of every extra, with the
licence its distribution declares. Rows marked "not installed here" give the
licence the project publishes; the licence audit confirms them wherever the
extra is installed (the CI licence job installs `.[all]`).

| Package | Extra | Licence | Copyleft |
|---|---|---|---|
| numpy | core | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | no |
| scipy | core | BSD-3-Clause | no |
| pydantic | core | MIT | no |
| pydantic-settings | core | MIT | no |
| python-dotenv | core | BSD-3-Clause | no |
| PyYAML | core | MIT | no |
| snntorch | neural | MIT | no |
| torch | neural | BSD-3-Clause | no |
| reservoirpy | neural | MIT | no |
| psutil | neural | BSD-3-Clause | no |
| inferactively-pymdp | fep | MIT | no |
| pyphi | iit | GPL-3.0-or-later | yes (strong; see below) |
| sentence-transformers | embeddings | Apache-2.0 (not installed here) | no |
| huggingface_hub | embeddings | Apache-2.0 | no |
| asyncpg | postgres | Apache-2.0 | no |
| pymongo | mongo | Apache-2.0 | no |
| redis | redis | MIT | no |
| neo4j | neo4j | Apache-2.0 (not installed here) | no |
| qdrant-client | qdrant | Apache-2.0 | no |
| httpx | llm, api | BSD-3-Clause | no |
| anthropic | anthropic | MIT (not installed here) | no |
| fastapi | api | MIT | no |
| uvicorn | api | BSD-3-Clause | no |
| pymc | research | Apache-2.0 (not installed here) | no |
| requests | research | Apache-2.0 | no |
| pytest | dev | MIT | no |
| pytest-asyncio | dev | Apache-2.0 | no |
| pytest-timeout | dev | MIT | no |
| black | dev | MIT | no |
| flake8 | dev | MIT | no |
| packaging | dev | Apache-2.0 or BSD-2-Clause | no |

The `databases` extra is the union of `postgres`, `mongo`, `redis`, `neo4j`
and `qdrant`; `assessment` is an alias of `research` plus `anthropic`; `all`
is every extra except `iit`.

## PyPhi (the iit extra)

PyPhi computes exact IIT 4.0 Phi. It is licensed under the GNU General Public
License, version 3 or later (GPLv3 or later).

- It is installed only through the `iit` extra (Python 3.12 or later), from a
  pinned commit of its own repository, and never through `all`.
- MTC never vendors it and never imports it at module import:
  `mtc.consciousness.pyphi_bridge` loads it lazily, only when it is installed.
- A user who installs the `iit` extra runs a GPLv3 combination on that
  machine.
- Redistributing a bundle that contains PyPhi (an image, an archive, a frozen
  application) falls under the GPLv3 for the whole bundle.
- MTC itself stays Apache-2.0; without the extra, the PhiTracker reports Phi
  as unavailable instead of estimating it.

## Weak copyleft and unlabelled transitive dependencies

Distributions the licence audit found in the installed closure of the extras
on the development machine that are weak copyleft or declare no licence in
their metadata. None is modified or vendored; each is used as an installed
library.

| Package | Licence | Pulled in by | Note |
|---|---|---|---|
| certifi | MPL-2.0 | requests, httpx, pymongo | weak copyleft; the CA bundle is used unmodified |
| tqdm | MPL-2.0 and MIT | huggingface_hub, reservoirpy, pyphi | weak copyleft; used unmodified |
| pathspec | MPL-2.0 | black (dev) | weak copyleft; development tool only |
| nose | LGPL | inferactively-pymdp releases before 1.0 | weak copyleft; a test dependency those releases declare |
| matplotlib-inline | BSD-3-Clause | ipython, through inferactively-pymdp releases before 1.0 | the metadata declares no licence field; the licence comes from its description |

docutils, also pulled in by the documentation dependencies of
inferactively-pymdp releases before 1.0, declares public-domain, BSD and GPL
classifiers for different parts of its sources; MTC does not import it.

## Re-running the audit

Install the project with the extras to audit, then run the audit from the
repository:

```bash
pip install ".[all]"
python scripts/license_audit.py --extras all
pip install ".[iit]"
python scripts/license_audit.py --extras iit
```

It exits 1 when strong copyleft appears in the closure of any extra except
`iit`, when a weak-copyleft or unlabelled distribution has no row in this file,
when a needed distribution is not installed, or when an `iit` run finds PyPhi
and this file lacks the PyPhi section; it exits 2 when the project is not
installed. `--json` prints every distribution with its licence and the extras
that bring it.
