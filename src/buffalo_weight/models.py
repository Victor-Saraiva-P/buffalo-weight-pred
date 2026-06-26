from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from sklearn.ensemble import RandomForestRegressor


MODEL_CONFIG_PATTERN = re.compile(r"^[a-z0-9_]+$")
RANDOM_FOREST_MODEL = "random_forest"
XGBOOST_MODEL = "xgboost"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    params: dict[str, Any]


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
}
REQUIRED_PARAMS = {
    RANDOM_FOREST_MODEL: {"n_estimators", "random_state"},
    XGBOOST_MODEL: {"n_estimators", "random_state"},
}


def parse_model_configs(training: dict[object, object]) -> list[ModelConfig]:
    raw_configs = training.get("model_configs")
    if not isinstance(raw_configs, dict) or not raw_configs:
        raise ValueError("config training.model_configs must be a non-empty map")

    configs = []
    for name, raw_config in raw_configs.items():
        config_name = str(name)
        if not MODEL_CONFIG_PATTERN.fullmatch(config_name):
            raise ValueError(
                f"config training.model_configs.{config_name} must use only lowercase letters, numbers, and underscores"
            )
        if not isinstance(raw_config, dict):
            raise ValueError(f"config training.model_configs.{config_name} must be a map")

        model = raw_config.get("model")
        if not isinstance(model, str):
            raise ValueError(f"config training.model_configs.{config_name}.model must be a string")
        if model not in ALLOWED_PARAMS:
            raise ValueError(f"unsupported model: {model}")

        params = raw_config.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"config training.model_configs.{config_name}.params must be a map")

        allowed = ALLOWED_PARAMS[model]
        unknown = sorted(str(param) for param in params if str(param) not in allowed)
        if unknown:
            raise ValueError(
                f"unsupported params for {config_name} ({model}): {', '.join(unknown)}"
            )
        missing = sorted(REQUIRED_PARAMS[model] - {str(param) for param in params})
        if missing:
            raise ValueError(f"missing params for {config_name} ({model}): {', '.join(missing)}")

        configs.append(ModelConfig(config_name, model, {str(key): value for key, value in params.items()}))

    return configs


def build_model(config: ModelConfig):
    if config.model == RANDOM_FOREST_MODEL:
        return RandomForestRegressor(**config.params)
    if config.model == XGBOOST_MODEL:
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ValueError("xgboost dependency is required for model xgboost") from error
        return XGBRegressor(**config.params, objective="reg:squarederror")
    raise ValueError(f"unsupported model: {config.model}")
