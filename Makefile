SHARED_CONFIG ?= configs/shared.yaml
CLASSICAL_MODELS_CONFIG ?= configs/classical_models.yaml
CNN_MASK_MODELS_CONFIG ?= configs/cnn_mask_models.yaml
MASK_CLASSICAL_EXPERIMENTS_CONFIG ?= configs/mask_classical_experiments.yaml
FUSION_EXPERIMENTS_CONFIG ?= configs/pca_feature_fusion_experiments.yaml
ALLOMETRIC_EXPERIMENTS_CONFIG ?= configs/allometric_mask_experiments.yaml
GEOMETRY_CHANNELS_EXPERIMENTS_CONFIG ?= configs/cnn_geometry_channels_experiments.yaml
TARGET_TRANSFORM_EXPERIMENTS_CONFIG ?= configs/target_transform_experiments.yaml
FUSION_TUNING_EXPERIMENTS_CONFIG ?= configs/fusion_tuning_experiments.yaml
CANONICAL_FUSION_EXPERIMENTS_CONFIG ?= configs/canonical_fusion_experiments.yaml
HEAVY_WEIGHTING_EXPERIMENTS_CONFIG ?= configs/heavy_weighting_experiments.yaml
CALIBRATION_PREDICTIONS ?= generated/train/dual_pca24_canonical16/predictions.csv
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
DEPS_STAMP ?= $(VENV)/.deps.stamp
CATEGORY_COUNTS ?= 4,6,8
START_SEED ?= 0
SEED_COUNT ?= 30
MODELS ?=
DEVICE ?= auto
DRY_RUN ?= false
TRAIN_DRY_RUN =
ifeq ($(DRY_RUN),true)
TRAIN_DRY_RUN = --dry-run
endif
ENSEMBLE_MODELS ?= dual_pca24_canonical16,tuned_96_pca24,fusion_original_stretch_log,fusion_original_stretch_cube_root,geometry_resnet18_pretrained_last_block,hist_gradient_boosting_baseline

.PHONY: setup features split train train-mask-experiments train-fusion-experiments train-allometric-experiments train-geometry-channel-experiments train-target-transform-experiments train-fusion-tuning train-canonical-fusion train-heavy-weighting calibrate clean stability compare-categories analyze-features ensemble diagnostics test

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
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_pipeline --shared-config $(SHARED_CONFIG) --classical-models-config $(CLASSICAL_MODELS_CONFIG) --cnn-mask-models-config $(CNN_MASK_MODELS_CONFIG) --device $(DEVICE) $(TRAIN_DRY_RUN)

train-mask-experiments: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_cnn_mask --shared-config $(SHARED_CONFIG) --models-config $(MASK_CLASSICAL_EXPERIMENTS_CONFIG) --device $(DEVICE) $(TRAIN_DRY_RUN)

train-fusion-experiments: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_classical --shared-config $(SHARED_CONFIG) --models-config $(FUSION_EXPERIMENTS_CONFIG) $(TRAIN_DRY_RUN)

train-allometric-experiments: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_classical --shared-config $(SHARED_CONFIG) --models-config $(ALLOMETRIC_EXPERIMENTS_CONFIG) $(TRAIN_DRY_RUN)

train-geometry-channel-experiments: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_cnn_mask --shared-config $(SHARED_CONFIG) --models-config $(GEOMETRY_CHANNELS_EXPERIMENTS_CONFIG) --device $(DEVICE) $(TRAIN_DRY_RUN)

train-target-transform-experiments: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_classical --shared-config $(SHARED_CONFIG) --models-config $(TARGET_TRANSFORM_EXPERIMENTS_CONFIG) $(TRAIN_DRY_RUN)

train-fusion-tuning: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_classical --shared-config $(SHARED_CONFIG) --models-config $(FUSION_TUNING_EXPERIMENTS_CONFIG) $(TRAIN_DRY_RUN)

train-canonical-fusion: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_classical --shared-config $(SHARED_CONFIG) --models-config $(CANONICAL_FUSION_EXPERIMENTS_CONFIG) $(TRAIN_DRY_RUN)

train-heavy-weighting: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.train_classical --shared-config $(SHARED_CONFIG) --models-config $(HEAVY_WEIGHTING_EXPERIMENTS_CONFIG) $(TRAIN_DRY_RUN)

calibrate: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.prediction_calibration --predictions $(CALIBRATION_PREDICTIONS)

clean: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.clean_train --shared-config $(SHARED_CONFIG) --models $(MODELS)

stability: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.stability --shared-config $(SHARED_CONFIG) --models-config $(CLASSICAL_MODELS_CONFIG) --start-seed $(START_SEED) --seed-count $(SEED_COUNT)

compare-categories: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.category_comparison --shared-config $(SHARED_CONFIG) --models-config $(CLASSICAL_MODELS_CONFIG) --category-counts $(CATEGORY_COUNTS) --start-seed $(START_SEED) --seed-count $(SEED_COUNT)

analyze-features: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.feature_analysis --shared-config $(SHARED_CONFIG) --models-config $(CLASSICAL_MODELS_CONFIG) --start-seed $(START_SEED) --seed-count $(SEED_COUNT)

ensemble: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.ensemble_oof --models $(ENSEMBLE_MODELS)

diagnostics: setup
	PYTHONPATH=src $(PYTHON) -m buffalo_weight.diagnostics --shared-config $(SHARED_CONFIG)

test: setup
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests
