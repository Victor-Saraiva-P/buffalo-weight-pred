from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol, TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestRegressor

if TYPE_CHECKING:
    from xgboost import XGBRegressor


MODEL_CONFIG_PATTERN = re.compile(r"^[a-z0-9_]+$")
RANDOM_FOREST_MODEL = "random_forest"
XGBOOST_MODEL = "xgboost"
CNN_MASK_MODEL = "cnn_mask"
PCA_SVR_MASK_MODEL = "pca_svr_mask"
MASK_PREDICTION_MODELS = frozenset({CNN_MASK_MODEL, PCA_SVR_MASK_MODEL})
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
}
REQUIRED_PARAMS = {
    RANDOM_FOREST_MODEL: {"n_estimators", "random_state"},
    XGBOOST_MODEL: {"n_estimators", "random_state"},
    CNN_MASK_MODEL: {"epochs", "batch_size", "learning_rate", "image_size", "random_state"},
    PCA_SVR_MASK_MODEL: {"image_size", "n_components", "random_state"},
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


def xgboost_compute_params(cuda_available: bool, cuda_build: bool) -> dict[str, str]:
    """Select XGBoost's accelerated histogram backend, preferring CUDA when usable.

    Example: ``xgboost_compute_params(True, True)`` selects the CUDA device.
    """
    device = "cuda" if cuda_available and cuda_build else "cpu"
    return {"device": device, "tree_method": "hist"}


def build_model(config: ModelConfig) -> ClassicalRegressor:
    if config.model == RANDOM_FOREST_MODEL:
        return RandomForestRegressor(**config.params)
    if config.model == XGBOOST_MODEL:
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
    if config.model in MASK_PREDICTION_MODELS:
        raise ValueError(f"{config.model} must be trained from mask rows, not feature arrays")
    raise ValueError(f"unsupported model: {config.model}")
