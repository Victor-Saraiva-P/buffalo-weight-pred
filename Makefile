REPORT_CONFIG ?= configs/report.yaml
FEATURE_CONTRACT ?=
FEATURE_REPORT ?=
APPROACH_CONTRACT ?=
APPROACH_REPORT ?=
DIAGNOSTIC_CONTRACT ?=
DIAGNOSTIC_REPORT ?=
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
DRY_RUN ?= false
TRAIN_DRY_RUN =
ifeq ($(DRY_RUN),true)
TRAIN_DRY_RUN = --dry-run
endif

.PHONY: setup reproduce inputs feature-selection confirm-features baselines compare-baselines confirm-approach tuning diagnostics-descriptive diagnostics-learning diagnostics-sensitivity confirm-diagnostics report-clean test

setup: $(PYTHON)
	$(PYTHON) main.py setup

reproduce: setup
	$(PYTHON) main.py reproduce --config $(REPORT_CONFIG) $(TRAIN_DRY_RUN)

inputs: setup
	$(PYTHON) main.py inputs --config $(REPORT_CONFIG)

feature-selection: setup
	$(PYTHON) main.py feature-selection --config $(REPORT_CONFIG)

confirm-features: setup
	$(PYTHON) main.py confirm-features --config $(REPORT_CONFIG) --contract $(FEATURE_CONTRACT) --report $(FEATURE_REPORT)

baselines: setup
	$(PYTHON) main.py baselines --config $(REPORT_CONFIG)

compare-baselines: setup
	$(PYTHON) main.py compare-baselines --config $(REPORT_CONFIG)

confirm-approach: setup
	$(PYTHON) main.py confirm-approach --config $(REPORT_CONFIG) --contract $(APPROACH_CONTRACT) --report $(APPROACH_REPORT)

tuning: setup
	$(PYTHON) main.py tuning --config $(REPORT_CONFIG)

diagnostics-descriptive: setup
	$(PYTHON) main.py diagnostics-descriptive --config $(REPORT_CONFIG)

diagnostics-learning: setup
	$(PYTHON) main.py diagnostics-learning --config $(REPORT_CONFIG)

diagnostics-sensitivity: setup
	$(PYTHON) main.py diagnostics-sensitivity --config $(REPORT_CONFIG)

confirm-diagnostics: setup
	$(PYTHON) main.py confirm-diagnostics --config $(REPORT_CONFIG) --contract $(DIAGNOSTIC_CONTRACT) --report $(DIAGNOSTIC_REPORT)

report-clean: setup
	$(PYTHON) main.py clean $(STAGE) --config $(REPORT_CONFIG)

$(PYTHON):
	python -m venv $(VENV)

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests


