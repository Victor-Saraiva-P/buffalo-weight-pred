"""Typed contracts and pre-registered tuning configuration variations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from buffalo_weight.compact_cnn_types import CompactCnnRecipe
from buffalo_weight.dense_feature_adapter import DenseTrainingRecipe
from buffalo_weight.resnet_baseline_adapter import ResNetTrainingRecipe

TuningRunStatus = Literal["absent", "obsolete", "reusable", "rebuilt", "baseline_maintained"]

ALLOWED_APPROACHES = ("random_forest", "dense_feature_network", "compact_cnn", "resnet18")


@dataclass(frozen=True)
class TuningVariation:
    """Represent one pre-registered tuning variation.

    Example: ``variation.name`` provides a unique identifier within the stage.
    """

    name: str
    approach: str
    baseline_configuration: str
    recipe: dict[str, object] | DenseTrainingRecipe | CompactCnnRecipe | ResNetTrainingRecipe


RF_VARIATIONS = (
    TuningVariation(
        name="rf_tuning_estimators",
        approach="random_forest",
        baseline_configuration="random_forest_baseline",
        recipe={
            "n_estimators": 50, "criterion": "squared_error", "bootstrap": True,
            "max_depth": None, "min_samples_leaf": 2, "min_samples_split": 4,
            "max_features": 0.6, "random_state": 44,
        },
    ),
    TuningVariation(
        name="rf_tuning_depth",
        approach="random_forest",
        baseline_configuration="random_forest_baseline",
        recipe={
            "n_estimators": 30, "criterion": "squared_error", "bootstrap": True,
            "max_depth": 15, "min_samples_leaf": 2, "min_samples_split": 4,
            "max_features": 0.7, "random_state": 44,
        },
    ),
    TuningVariation(
        name="rf_tuning_features",
        approach="random_forest",
        baseline_configuration="random_forest_baseline",
        recipe={
            "n_estimators": 40, "criterion": "squared_error", "bootstrap": True,
            "max_depth": None, "min_samples_leaf": 3, "min_samples_split": 6,
            "max_features": "sqrt", "random_state": 44,
        },
    ),
)

DENSE_VARIATIONS = (
    TuningVariation(
        name="dense_tuning_architecture",
        approach="dense_feature_network",
        baseline_configuration="dense",
        recipe=DenseTrainingRecipe(hidden_layers=(128, 64), dropout=0.15, learning_rate=0.001),
    ),
    TuningVariation(
        name="dense_tuning_lr",
        approach="dense_feature_network",
        baseline_configuration="dense",
        recipe=DenseTrainingRecipe(hidden_layers=(64, 32), dropout=0.20, learning_rate=0.0005),
    ),
    TuningVariation(
        name="dense_tuning_regularization",
        approach="dense_feature_network",
        baseline_configuration="dense",
        recipe=DenseTrainingRecipe(hidden_layers=(64, 32), dropout=0.30, weight_decay=0.0005),
    ),
)

COMPACT_CNN_VARIATIONS = (
    TuningVariation(
        name="compact_tuning_lr",
        approach="compact_cnn",
        baseline_configuration="compact_cnn",
        recipe=CompactCnnRecipe(learning_rate=0.0005),
    ),
    TuningVariation(
        name="compact_tuning_patience",
        approach="compact_cnn",
        baseline_configuration="compact_cnn",
        recipe=CompactCnnRecipe(max_epochs=500, patience=60),
    ),
    TuningVariation(
        name="compact_tuning_augmentation",
        approach="compact_cnn",
        baseline_configuration="compact_cnn",
        recipe=CompactCnnRecipe(translation_fraction=0.10),
    ),
)

RESNET18_VARIATIONS = (
    TuningVariation(
        name="resnet_tuning_lr",
        approach="resnet18",
        baseline_configuration="resnet18_pretrained_partial",
        recipe=ResNetTrainingRecipe(
            warmup_learning_rate=0.0005, layer4_learning_rate=0.00005, head_learning_rate=0.00025,
        ),
    ),
    TuningVariation(
        name="resnet_tuning_warmup",
        approach="resnet18",
        baseline_configuration="resnet18_pretrained_partial",
        recipe=ResNetTrainingRecipe(warmup_epochs=30, max_partial_epochs=200, patience=30),
    ),
    TuningVariation(
        name="resnet_tuning_regularization",
        approach="resnet18",
        baseline_configuration="resnet18_pretrained_partial",
        recipe=ResNetTrainingRecipe(weight_decay=0.0005, gradient_clip=2.5),
    ),
)

_REGISTRY: dict[str, tuple[TuningVariation, ...]] = {
    "random_forest": RF_VARIATIONS,
    "dense_feature_network": DENSE_VARIATIONS,
    "compact_cnn": COMPACT_CNN_VARIATIONS,
    "resnet18": RESNET18_VARIATIONS,
}


def validate_pre_registered_approach(approach: str) -> None:
    """Validate that the approach name is supported by pre-registration.

    Example: ``validate_pre_registered_approach("random_forest")`` succeeds.
    """
    if approach not in ALLOWED_APPROACHES:
        raise ValueError(
            f"approach was {approach!r}; expected one of: {', '.join(ALLOWED_APPROACHES)}"
        )


def get_pre_registered_variations(
    approach: str, budget: int = 3,
) -> tuple[TuningVariation, ...]:
    """Retrieve pre-registered variations within the allowed budget.

    Example: ``get_pre_registered_variations("random_forest", 3)`` returns registered RF variations.
    """
    validate_pre_registered_approach(approach)
    if budget <= 0:
        return ()
    all_variations = _REGISTRY[approach]
    capped = all_variations[:budget]
    return capped
