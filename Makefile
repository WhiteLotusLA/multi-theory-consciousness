.PHONY: db-up db-init db-verify db-down demo api test test-fast privacy-audit env-check generate lint hooks

# The Compose services db-up starts; name the ones the setup enabled, for
# example: make db-up DB_SERVICES="postgres mongo qdrant redis"
DB_SERVICES ?= postgres mongo qdrant
COMPOSE = docker compose --env-file .env -f docker/docker-compose.yml

# Start the stores in containers and wait until they are healthy. The Compose
# file was reviewed, not run, on the development machine (see docker/README.md).
db-up:
	$(COMPOSE) up -d --wait $(DB_SERVICES)

# Idempotent: creates what is missing and records the install date.
db-init:
	mtc db init

db-verify:
	mtc db check

# Stops and removes the containers; the named volumes, and the data in them,
# stay. Removing a volume is never done from here.
db-down:
	$(COMPOSE) down

# The whole pipeline with the mock provider and in-memory stores.
demo:
	PYPHI_WELCOME_OFF=yes python3 examples/full_pipeline_demo.py

api:
	mtc serve

test:
	PYPHI_WELCOME_OFF=yes python3 -m pytest tests -q

test-fast:
	PYPHI_WELCOME_OFF=yes python3 -m pytest tests -q -m "not slow and not db and not live"

privacy-audit:
	python3 scripts/privacy_audit.py

# The generated artifacts match their sources.
env-check:
	python3 scripts/gen_env_example.py --check
	python3 scripts/gen_schemas.py --check
	PYPHI_WELCOME_OFF=yes python3 -m mtc.db.migrations --check
	python3 scripts/sync_resources.py --check

# Rewrite the generated artifacts (never edit them by hand).
generate:
	python3 scripts/gen_env_example.py
	python3 scripts/gen_schemas.py
	PYPHI_WELCOME_OFF=yes python3 -m mtc.db.migrations --write
	python3 scripts/sync_resources.py

lint:
	black --check --line-length 88 mtc tests scripts examples
	flake8 mtc tests scripts examples
	for script in $$(git ls-files 'scripts/*.sh'); do bash -n "$$script" || exit 1; done
	if command -v shellcheck >/dev/null 2>&1; then shellcheck $$(git ls-files 'scripts/*.sh'); else echo "shellcheck is not installed; only bash -n ran"; fi

hooks:
	bash scripts/install_git_hooks.sh
