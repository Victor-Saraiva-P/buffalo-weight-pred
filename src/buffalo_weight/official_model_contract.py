"""Model policy for the reproducible report environment."""

from __future__ import annotations

from buffalo_weight.models import ModelConfig, XGBOOST_MODEL


def validate_official_model_configs(configs: list[ModelConfig]) -> None:
    """Reject optional models; for example, XGBoost needs a non-official environment."""
    optional = [(config.name, config.model) for config in configs if config.model == XGBOOST_MODEL]
    if not optional:
        return
    raise ValueError(
        f"official model configs included {optional!r}; expected no xgboost models because "
        "the official environment contains only the eight approved dependencies"
    )
