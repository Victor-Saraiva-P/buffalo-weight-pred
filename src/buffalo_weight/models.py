from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Protocol, TYPE_CHECKING

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor

from buffalo_weight.target_transform import inverse_target, transform_target

if TYPE_CHECKING:
    from xgboost import XGBRegressor


MODEL_CONFIG_PATTERN = re.compile(r"^[a-z0-9_]+$")
RANDOM_FOREST_MODEL = "random_forest"
EXTRA_TREES_MODEL = "extra_trees"
HIST_GRADIENT_BOOSTING_MODEL = "hist_gradient_boosting"
XGBOOST_MODEL = "xgboost"
CNN_MASK_MODEL = "cnn_mask"
PCA_SVR_MASK_MODEL = "pca_svr_mask"
MASK_FEATURE_MODEL = "mask_feature"
PRETRAINED_MASK_EMBEDDING_MODEL = "pretrained_mask_embedding"
PCA_FEATURE_FUSION_MODEL = "pca_feature_fusion"
MASK_PREDICTION_MODELS = frozenset(
    {CNN_MASK_MODEL, MASK_FEATURE_MODEL, PCA_SVR_MASK_MODEL, PRETRAINED_MASK_EMBEDDING_MODEL}
)
FEATURE_FUSION_MODELS = frozenset({PCA_FEATURE_FUSION_MODEL})
ModelParam = bool | float | int | str


class ClassicalRegressor(Protocol):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> object: ...

    def predict(self, x_validation: np.ndarray) -> np.ndarray: ...


class _XgboostRegressor:
    def __init__(self, model: XGBRegressor) -> None:
        self.model = model

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> object:
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_validation: np.ndarray) -> np.ndarray:
        from xgboost import DMatrix

        return self.model.get_booster().predict(DMatrix(x_validation))


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    params: dict[str, ModelParam]


ALLOWED_PARAMS = {
    RANDOM_FOREST_MODEL: {
        "n_estimators",
        "random_state",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "target_transform",
    },
    EXTRA_TREES_MODEL: {
        "n_estimators",
        "random_state",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "target_transform",
    },
    HIST_GRADIENT_BOOSTING_MODEL: {
        "random_state",
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "target_transform",
    },
    XGBOOST_MODEL: {
        "n_estimators",
        "random_state",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
    },
    CNN_MASK_MODEL: {
        "epochs",
        "batch_size",
        "learning_rate",
        "image_size",
        "weight_decay",
        "random_state",
        "patience",
        "augment",
        "validation_fraction",
        "resize_mode",
        "architecture",
        "pretrained",
        "fine_tune_mode",
        "input_representation",
    },
    PCA_SVR_MASK_MODEL: {
        "image_size",
        "n_components",
        "random_state",
        "c",
        "epsilon",
        "gamma",
        "resize_mode",
    },
    MASK_FEATURE_MODEL: {
        "image_size",
        "resize_mode",
        "representation",
        "estimator",
        "n_components",
        "random_state",
        "alpha",
        "c",
        "epsilon",
        "gamma",
        "n_estimators",
        "min_samples_leaf",
        "max_features",
    },
    PRETRAINED_MASK_EMBEDDING_MODEL: {
        "image_size",
        "resize_mode",
        "architecture",
        "estimator",
        "n_components",
        "random_state",
        "alpha",
        "c",
        "epsilon",
        "gamma",
        "batch_size",
    },
    PCA_FEATURE_FUSION_MODEL: {
        "image_size",
        "resize_mode",
        "n_components",
        "random_state",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "target_transform",
        "target_power",
        "canonical_components",
        "canonical_resize_mode",
        "heavy_sample_weight",
        "heavy_quantile",
    },
}
REQUIRED_PARAMS = {
    RANDOM_FOREST_MODEL: {"n_estimators", "random_state"},
    EXTRA_TREES_MODEL: {"n_estimators", "random_state"},
    HIST_GRADIENT_BOOSTING_MODEL: {"random_state"},
    XGBOOST_MODEL: {"n_estimators", "random_state"},
    CNN_MASK_MODEL: {"epochs", "batch_size", "learning_rate", "image_size", "random_state"},
    PCA_SVR_MASK_MODEL: {"image_size", "n_components", "random_state"},
    MASK_FEATURE_MODEL: {"image_size", "representation", "estimator", "random_state"},
    PRETRAINED_MASK_EMBEDDING_MODEL: {
        "image_size",
        "architecture",
        "estimator",
        "random_state",
    },
    PCA_FEATURE_FUSION_MODEL: {"image_size", "n_components", "n_estimators", "random_state"},
}


def model_param_values(params: dict[object, object], config_name: str) -> dict[str, ModelParam]:
    parsed_params = {}
    for key, value in params.items():
        if not isinstance(value, (bool, float, int, str)):
            raise ValueError(
                f"config training.model_configs.{config_name}.params.{key} was {value!r}; expected a scalar"
            )
        parsed_params[str(key)] = value
    return parsed_params


def model_configs_map(training: dict[object, object]) -> dict[object, object]:
    raw_configs = training.get("model_configs")
    if isinstance(raw_configs, dict) and raw_configs:
        return raw_configs
    raise ValueError(f"config training.model_configs was {raw_configs!r}; expected a non-empty map")


def validate_model_config_name(config_name: str) -> None:
    if MODEL_CONFIG_PATTERN.fullmatch(config_name):
        return
    raise ValueError(
        f"config training.model_configs.{config_name} must use only lowercase letters, numbers, and underscores"
    )


def model_name(raw_config: dict[object, object], config_name: str) -> str:
    model = raw_config.get("model")
    if not isinstance(model, str):
        raise ValueError(f"config training.model_configs.{config_name}.model was {model!r}; expected a string")
    if model in ALLOWED_PARAMS:
        return model
    raise ValueError(f"unsupported model was {model!r}; expected one of {sorted(ALLOWED_PARAMS)}")


def raw_model_params(raw_config: dict[object, object], config_name: str) -> dict[object, object]:
    params = raw_config.get("params", {})
    if isinstance(params, dict):
        return params
    raise ValueError(f"config training.model_configs.{config_name}.params was {params!r}; expected a map")


def validate_model_params(config_name: str, model: str, params: dict[object, object]) -> None:
    unknown = sorted(str(param) for param in params if str(param) not in ALLOWED_PARAMS[model])
    if unknown:
        raise ValueError(f"unsupported params for {config_name} ({model}): {', '.join(unknown)}")
    missing = sorted(REQUIRED_PARAMS[model] - {str(param) for param in params})
    if missing:
        raise ValueError(f"missing params for {config_name} ({model}): {', '.join(missing)}")


def parse_model_config(config_name: str, raw_config: object) -> ModelConfig:
    validate_model_config_name(config_name)
    if not isinstance(raw_config, dict):
        raise ValueError(f"config training.model_configs.{config_name} was {raw_config!r}; expected a map")
    model = model_name(raw_config, config_name)
    params = raw_model_params(raw_config, config_name)
    validate_model_params(config_name, model, params)
    return ModelConfig(config_name, model, model_param_values(params, config_name))


def parse_model_configs(training: dict[object, object]) -> list[ModelConfig]:
    return [
        parse_model_config(str(name), raw_config)
        for name, raw_config in model_configs_map(training).items()
    ]


def validate_unique_model_configs(configs: list[ModelConfig]) -> None:
    names = [config.name for config in configs]
    if len(names) == len(set(names)):
        return
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    raise ValueError(f"duplicate model configuration names were {duplicates!r}; expected unique names")


def xgboost_compute_params(cuda_available: bool, cuda_build: bool) -> dict[str, str]:
    """Select XGBoost's accelerated histogram backend, preferring CUDA when usable.

    Example: ``xgboost_compute_params(True, True)`` selects the CUDA device.
    """
    device = "cuda" if cuda_available and cuda_build else "cpu"
    return {"device": device, "tree_method": "hist"}


def build_model(config: ModelConfig) -> ClassicalRegressor:
    builders = {
        RANDOM_FOREST_MODEL: _build_random_forest,
        EXTRA_TREES_MODEL: _build_extra_trees,
        HIST_GRADIENT_BOOSTING_MODEL: _build_hist_gradient_boosting,
        XGBOOST_MODEL: _build_xgboost,
    }
    if config.model in builders:
        return builders[config.model](config)
    if config.model in MASK_PREDICTION_MODELS | FEATURE_FUSION_MODELS:
        raise ValueError(f"{config.model} must be trained from mask rows, not feature arrays")
    raise ValueError(f"unsupported model: {config.model}")


def _build_sklearn_model(
    config: ModelConfig, constructor: Callable[..., ClassicalRegressor]
) -> ClassicalRegressor:
    params = dict(config.params)
    target_transform = str(params.pop("target_transform", "identity"))
    return _target_regressor(constructor(**params), target_transform)


def _build_random_forest(config: ModelConfig) -> ClassicalRegressor:
    return _build_sklearn_model(config, RandomForestRegressor)


def _build_extra_trees(config: ModelConfig) -> ClassicalRegressor:
    return _build_sklearn_model(config, ExtraTreesRegressor)


def _build_hist_gradient_boosting(config: ModelConfig) -> ClassicalRegressor:
    return _build_sklearn_model(config, HistGradientBoostingRegressor)


def _build_xgboost(config: ModelConfig) -> ClassicalRegressor:
    try:
        import torch
        import xgboost
    except ImportError as error:
        raise ValueError("xgboost and torch dependencies are required for model xgboost") from error
    compute_params = xgboost_compute_params(
        torch.cuda.is_available(), bool(xgboost.build_info().get("USE_CUDA", False))
    )
    model = xgboost.XGBRegressor(**config.params, **compute_params, objective="reg:squarederror")
    return _XgboostRegressor(model)


def _target_regressor(regressor: ClassicalRegressor, transform: str) -> ClassicalRegressor:
    if transform == "identity":
        return regressor
    transform_target(np.asarray([1.0]), transform)
    return TransformedTargetRegressor(
        regressor=regressor,
        func=lambda values: transform_target(values, transform),
        inverse_func=lambda values: inverse_target(values, transform),
        check_inverse=False,
    )
