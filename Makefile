ifeq ($(OS),Windows_NT)
	PYTHON := ".venv\Scripts\python.exe"
	PIP := ".venv\Scripts\pip.exe"
else
	PYTHON := .venv/bin/python
	PIP := .venv/bin/pip
endif

.PHONY: help install lint format test check probe smoke-testbed testbed-up testbed-down demo reproduce reproduce-quick

help:
	@echo TrustFed-SDN Makefile targets:
	@echo   install          Install editable package and dev dependencies into .venv
	@echo   lint             Run ruff and black --check
	@echo   format           Format code with ruff --fix and black
	@echo   test             Run pytest test suite
	@echo   check            Run lint + pytest + package import smoke check
	@echo   probe            Run hardware and environment probe script
	@echo   smoke-testbed    Run Linux Mininet + OVS + RYU smoke test (Linux root required)

install:
	$(PIP) install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check trustfed scripts tests
	$(PYTHON) -m black --check trustfed scripts tests

format:
	$(PYTHON) -m ruff check --fix trustfed scripts tests
	$(PYTHON) -m black trustfed scripts tests

test:
	$(PYTHON) -m pytest

check: lint test
	$(PYTHON) -c "import trustfed; print('Import check OK: trustfed', trustfed.__version__)"

probe:
	$(PYTHON) scripts/probe_hardware.py

smoke-testbed:
	bash scripts/smoke_test_linux.sh

testbed-up:
	@echo "Implemented in Step 07 (see docs/steps/STEP_07_sdn_testbed.md)"

testbed-down:
	@echo "Implemented in Step 07 (see docs/steps/STEP_07_sdn_testbed.md)"

demo:
	@echo "Implemented in Step 08 (see docs/steps/STEP_08_live_integration.md)"

reproduce:
	@echo "Implemented in Step 09 (see docs/steps/STEP_09_full_evaluation.md)"

reproduce-quick:
	@echo "Implemented in Step 09 (see docs/steps/STEP_09_full_evaluation.md)"

