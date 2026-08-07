from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Protocol

import numpy as np
from sklearn.ensemble import RandomForestRegressor


MODEL_CONFIG_PATTERN = re.compile(r"^[a-z0-9_]+$")
RANDOM_FOREST_MODEL = "random_forest"
CNN_MASK_MODEL = "cnn_mask"
MASK_PREDICTION_MODELS = frozenset({CNN_MASK_MODEL})
FEATURE_FUSION_MODELS: frozenset[str] = frozenset()
ModelParam = bool | float | int | str


class ClassicalRegressor(Protocol):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> object: ...

    def predict(self, x_validation: np.ndarray) -> np.ndarray: ...


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
}
REQUIRED_PARAMS = {
    RANDOM_FOREST_MODEL: {"n_estimators", "random_state"},
    CNN_MASK_MODEL: {"epochs", "batch_size", "learning_rate", "image_size", "random_state"},
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
    """Reject duplicate names before they can share one artifact directory."""
    names = [config.name for config in configs]
    if len(names) == len(set(names)):
        return
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    raise ValueError(f"duplicate model configuration names were {duplicates!r}; expected unique names")


def build_model(config: ModelConfig) -> ClassicalRegressor:
    builders = {
        RANDOM_FOREST_MODEL: _build_random_forest,
    }
    if config.model in builders:
        return builders[config.model](config)
    if config.model in MASK_PREDICTION_MODELS | FEATURE_FUSION_MODELS:
        raise ValueError(f"{config.model} must be trained from mask rows, not feature arrays")
    raise ValueError(f"unsupported model: {config.model}")


def _build_random_forest(config: ModelConfig) -> ClassicalRegressor:
    """Build a Random Forest regressor instance from configuration parameters.

    Example: ``_build_random_forest(config)`` returns a configured RandomForestRegressor.
    """
    model_params = dict(config.params)
    n_estimators = int(model_params.get("n_estimators", 100))
    random_state = int(model_params.get("random_state", 42))
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        **{key: value for key, value in model_params.items() if key not in {"n_estimators", "random_state"}},
    )


