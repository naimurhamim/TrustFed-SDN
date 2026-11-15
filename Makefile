ifeq ($(OS),Windows_NT)
	PYTHON := ".venv\Scripts\python.exe"
	PIP := ".venv\Scripts\pip.exe"
else
	PYTHON := .venv/bin/python
	PIP := .venv/bin/pip
endif

.PHONY: help install status lint format test check probe smoke-testbed testbed-up testbed-down notify demo reproduce reproduce-quick

help:
	@echo TrustFed-SDN Makefile targets:
	@echo   install          Install editable package and dev dependencies into .venv
	@echo   status           Compare GitHub sync, packages, datasets, and team checklist
	@echo   lint             Run ruff and black --check
	@echo   format           Format code with ruff --fix and black
	@echo   test             Run pytest test suite
	@echo   check            Run lint + pytest + package import smoke check
	@echo   probe            Run hardware and environment probe script
	@echo   smoke-testbed    Run Linux Mininet + OVS + RYU smoke test (Linux root required)

install:
	$(PIP) install -e ".[dev]"

status:
	$(PYTHON) scripts/team_status.py

lint:
	$(PYTHON) -m ruff check trustfed scripts tests testbed
	$(PYTHON) -m black --check trustfed scripts tests testbed

format:
	$(PYTHON) -m black trustfed scripts tests testbed
	$(PYTHON) -m ruff check --fix trustfed scripts tests testbed

test:
	$(PYTHON) -m pytest

check: lint test
	$(PYTHON) -c "import trustfed; print('Import check OK: trustfed', trustfed.__version__)"

probe:
	$(PYTHON) scripts/probe_hardware.py

smoke-testbed:
	bash scripts/smoke_test_linux.sh

testbed-up:
	sudo bash scripts/run_step07_testbed.sh

testbed-down:
	sudo mn -c

notify:
	$(PYTHON) scripts/notify_telegram.py --attach ../PRITOM_ANTIGRAVITY_SETUP.md --attach docs/TEAM_CHECKLIST.md

demo:
	$(PYTHON) scripts/run_step08_live.py --demo

reproduce:
	$(PYTHON) scripts/run_step09_full_evaluation.py

reproduce-quick:
	$(PYTHON) scripts/run_step09_full_evaluation.py --quick

