CONFIG ?= configs/baseline.yaml
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
DEPS_STAMP ?= $(VENV)/.deps.stamp
CATEGORY_COUNTS ?= 4,6,8
START_SEED ?= 0
SEED_COUNT ?= 30

.PHONY: setup features split train stability compare-categories test

setup: $(DEPS_STAMP)

$(PYTHON):
	python -m venv $(VENV)

$(DEPS_STAMP): requirements.txt | $(PYTHON)
	$(PIP) install -r requirements.txt
	touch $(DEPS_STAMP)

features: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.features --config $(CONFIG)

split: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.split --config $(CONFIG)

train: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train --config $(CONFIG)

stability: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.stability --config $(CONFIG) --start-seed $(START_SEED) --seed-count $(SEED_COUNT)

compare-categories: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.category_comparison --config $(CONFIG) --category-counts $(CATEGORY_COUNTS) --start-seed $(START_SEED) --seed-count $(SEED_COUNT)

test: setup
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests
