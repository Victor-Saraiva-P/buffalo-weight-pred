CONFIG ?= configs/baseline.yaml
PYTHON ?= .venv/bin/python

.PHONY: features split train test

features:
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.features --config $(CONFIG)

split:
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.split --config $(CONFIG)

train:
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train --config $(CONFIG)

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests
