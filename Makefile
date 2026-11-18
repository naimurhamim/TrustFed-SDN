ifeq ($(OS),Windows_NT)
	PYTHON := ".venv\Scripts\python.exe"
	PIP := ".venv\Scripts\pip.exe"
else
	PYTHON := .venv/bin/python
	PIP := .venv/bin/pip
endif

.PHONY: help install status lint format test check probe smoke-testbed testbed-up testbed-down notify demo studio reproduce reproduce-quick qr site-preview

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
	@echo   qr               Regenerate and verify SVG/PNG/PDF QR codes via scripts/make_qr.py
	@echo   site-preview     Serve ../07_Website/trustfed-sdn-site locally at http://127.0.0.1:8080

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

qr:
	$(PYTHON) scripts/make_qr.py

studio:
	$(PYTHON) demo/studio_server.py

site-preview:
	$(PYTHON) -m http.server 8080 --directory ../07_Website/trustfed-sdn-site

