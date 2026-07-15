SHARED_CONFIG ?= configs/shared.yaml
CLASSICAL_MODELS_CONFIG ?= configs/classical_models.yaml
CNN_MASK_MODELS_CONFIG ?= configs/cnn_mask_models.yaml
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
DEPS_STAMP ?= $(VENV)/.deps.stamp
CATEGORY_COUNTS ?= 4,6,8
START_SEED ?= 0
SEED_COUNT ?= 30
MODELS ?=
DEVICE ?= auto

.PHONY: setup features split train clean stability compare-categories analyze-features test

setup: $(DEPS_STAMP)

$(PYTHON):
	python -m venv $(VENV)

$(DEPS_STAMP): requirements.txt | $(PYTHON)
	$(PIP) install -r requirements.txt
	touch $(DEPS_STAMP)

features: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.features --shared-config $(SHARED_CONFIG)

split: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.split --shared-config $(SHARED_CONFIG)

train: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_pipeline --shared-config $(SHARED_CONFIG) --classical-models-config $(CLASSICAL_MODELS_CONFIG) --cnn-mask-models-config $(CNN_MASK_MODELS_CONFIG) --device $(DEVICE)

clean: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.clean_train --shared-config $(SHARED_CONFIG) --models $(MODELS)

stability: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.stability --shared-config $(SHARED_CONFIG) --models-config $(CLASSICAL_MODELS_CONFIG) --start-seed $(START_SEED) --seed-count $(SEED_COUNT)

compare-categories: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.category_comparison --shared-config $(SHARED_CONFIG) --models-config $(CLASSICAL_MODELS_CONFIG) --category-counts $(CATEGORY_COUNTS) --start-seed $(START_SEED) --seed-count $(SEED_COUNT)

analyze-features: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.feature_analysis --shared-config $(SHARED_CONFIG) --models-config $(CLASSICAL_MODELS_CONFIG) --start-seed $(START_SEED) --seed-count $(SEED_COUNT)

test: setup
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests
