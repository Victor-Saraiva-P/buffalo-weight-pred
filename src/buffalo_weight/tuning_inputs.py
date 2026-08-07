"""Validation and gate inspection for configuration tuning inputs."""

from __future__ import annotations

from buffalo_weight.approach_confirmation import require_approach_gate
from buffalo_weight.feature_confirmation_manifest import validate_frozen_feature_contract
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.tuning_types import TuningVariation, get_pre_registered_variations


def validate_tuning_gate_and_contract(
    contract: ReportContract,
) -> tuple[str, str, int, tuple[str, ...] | None, tuple[TuningVariation, ...]]:
    """Validate approach gate, feature contract, and retrieve pre-registered variations.

    Example: returns ``(approach, baseline, budget, frozen_features, variations)``.
    """
    approach, baseline_config, budget = require_approach_gate(contract)
    frozen_features = _frozen_features_if_required(contract, approach)
    variations = get_pre_registered_variations(approach, budget)
    return approach, baseline_config, budget, frozen_features, variations


def _frozen_features_if_required(
    contract: ReportContract, approach: str,
) -> tuple[str, ...] | None:
    if approach not in ("random_forest", "dense_feature_network"):
        return None
    features = validate_frozen_feature_contract(contract)
    return features
