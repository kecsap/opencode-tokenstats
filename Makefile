PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: help test run run-api doctor doctor-api warmup build install-wheel clean-dist

help:
	@printf "Targets:\n"
	@printf "  make test           Run test suite\n"
	@printf "  make run            Run CLI locally (no install), local mode\n"
	@printf "  make run-api        Run CLI locally (no install), api mode\n"
	@printf "  make doctor         Run doctor locally (no install), local mode\n"
	@printf "  make doctor-api     Run doctor locally (no install), api mode\n"
	@printf "  make warmup         Warm tokenizer cache locally\n"
	@printf "  make build          Build wheel/sdist into dist/\n"
	@printf "  make install-wheel  Install built wheel from dist/\n"
	@printf "  make clean-dist     Remove dist/ build artifacts\n"

test:
	pytest -q

run:
	PYTHONPATH=src $(PYTHON) -m opencode_tokenstats.cli --mode local --help

run-api:
	PYTHONPATH=src $(PYTHON) -m opencode_tokenstats.cli --mode api --help

doctor:
	PYTHONPATH=src $(PYTHON) -m opencode_tokenstats.cli --mode local doctor

doctor-api:
	PYTHONPATH=src $(PYTHON) -m opencode_tokenstats.cli --mode api doctor

warmup:
	PYTHONPATH=src $(PYTHON) -m opencode_tokenstats.cli tokenizer-warmup

build:
	$(PIP) install build
	$(PYTHON) -m build

install-wheel:
	$(PIP) install dist/*.whl

clean-dist:
	rm -rf dist build *.egg-info
