"""Observed and structural redundancy for the candidate feature universe."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.stats import spearmanr

from buffalo_weight.feature_evaluation import FeatureSample
from buffalo_weight.feature_selection_contract import REDUNDANCY_FAMILIES

AREA_BIJECTION_FEATURES = frozenset(("area", "equivalent_diameter", "area_power_1_5"))


@dataclass(frozen=True)
class FeatureRedundancy:
    feature_a: str
    feature_b: str
    structural_relation: str
    pearson: float | None
    spearman: float | None
    removal_group: str


def calculate_feature_redundancy(
    samples: list[FeatureSample], feature_names: tuple[str, ...]
) -> list[FeatureRedundancy]:
    """Calculate every pair; for example, 26 features produce 325 ordered rows."""
    return [_redundancy_row(samples, first, second)
            for first, second in itertools.combinations(feature_names, 2)]


def _redundancy_row(
    samples: list[FeatureSample], feature_a: str, feature_b: str
) -> FeatureRedundancy:
    first = np.asarray([sample.feature_values[feature_a] for sample in samples])
    second = np.asarray([sample.feature_values[feature_b] for sample in samples])
    pearson = _finite_correlation(cast(float, np.corrcoef(first, second)[0, 1]))
    spearman = _finite_correlation(cast(float, spearmanr(first, second).statistic))
    return FeatureRedundancy(feature_a, feature_b, _structural_relation(feature_a, feature_b),
                             pearson, spearman, _removal_group(feature_a, feature_b))


def _structural_relation(feature_a: str, feature_b: str) -> str:
    if {feature_a, feature_b}.issubset(AREA_BIJECTION_FEATURES):
        return "area_bijection"
    return "none"


def _removal_group(feature_a: str, feature_b: str) -> str:
    for group, members in REDUNDANCY_FAMILIES.items():
        if feature_a in members and feature_b in members:
            return group
    return "none"


def _finite_correlation(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
