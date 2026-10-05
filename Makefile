.PHONY: test test-fast privacy-audit env-check lint hooks

test:
	PYPHI_WELCOME_OFF=yes python3 -m pytest tests -q

test-fast:
	PYPHI_WELCOME_OFF=yes python3 -m pytest tests -q -m "not slow and not db and not live"

privacy-audit:
	python3 scripts/privacy_audit.py

env-check:
	python3 scripts/gen_env_example.py --check

lint:
	black --check --line-length 88 mtc tests scripts
	flake8 mtc tests scripts

hooks:
	bash scripts/install_git_hooks.sh
