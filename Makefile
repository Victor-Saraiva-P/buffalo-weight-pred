CONFIG ?= configs/baseline.yaml
PYTHON ?= python

.PHONY: features test

features:
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.features --config $(CONFIG)

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests
