from __future__ import annotations

from buffalo_weight.feature_evaluation import FeatureEvidence, FeatureSample, RemovalGroup
from buffalo_weight.feature_selection_rules import classify_mae_delta, permutation_seed


class FixedFeatureEvidenceRunner:
    """Return complete deterministic evidence without scientific model training."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []

    def evaluate(
        self, samples: list[FeatureSample], feature_names: tuple[str, ...],
        removal_groups: tuple[RemovalGroup, ...], permutation_count: int, split_seed: int,
    ) -> list[FeatureEvidence]:
        self.calls.append((len(samples), feature_names, tuple(group.name for group in removal_groups)))
        rows: list[FeatureEvidence] = []
        folds = sorted({sample.fold for sample in samples})
        scopes = [("fold", fold, sum(sample.fold == fold for sample in samples)) for fold in folds]
        scopes.append(("oof", None, len(samples)))
        for baseline in ("random_forest", "dense"):
            rows.extend(self._baseline_rows(feature_names, removal_groups, scopes, baseline,
                                            permutation_count, split_seed))
        return rows

    def _baseline_rows(
        self, features: tuple[str, ...], groups: tuple[RemovalGroup, ...],
        scopes: list[tuple[str, int | None, int]], baseline: str,
        permutation_count: int, split_seed: int,
    ) -> list[FeatureEvidence]:
        rows: list[FeatureEvidence] = []
        for scope, fold, count in scopes:
            rows.extend(self._isolated_rows(features, baseline, scope, fold, count))
            rows.extend(self._removal_rows(features, groups, baseline, scope, fold, count))
            rows.extend(self._permutation_rows(features, baseline, scope, fold, count,
                                               permutation_count, split_seed))
        return rows

    @staticmethod
    def _isolated_rows(
        features: tuple[str, ...], baseline: str, scope: str,
        fold: int | None, count: int,
    ) -> list[FeatureEvidence]:
        return [FeatureEvidence("isolated", baseline, feature, scope, fold, None, None,
                                count, None, 12.0, None, None) for feature in features]

    def _removal_rows(
        self, features: tuple[str, ...], groups: tuple[RemovalGroup, ...], baseline: str,
        scope: str, fold: int | None, count: int,
    ) -> list[FeatureEvidence]:
        targets = [*features, *(group.name for group in groups)]
        return [self._delta_row("removal", baseline, target, scope, fold, count,
                                self._removal_delta(target, baseline)) for target in targets]

    @staticmethod
    def _removal_delta(target: str, baseline: str) -> float:
        if target == "area":
            return -1.2 if baseline == "random_forest" else 0.5
        if target == "perimeter":
            return -1.2 if baseline == "random_forest" else 1.2
        return 0.0

    def _permutation_rows(
        self, features: tuple[str, ...], baseline: str, scope: str,
        fold: int | None, count: int, permutation_count: int, split_seed: int,
    ) -> list[FeatureEvidence]:
        rows: list[FeatureEvidence] = []
        for feature in features:
            for repetition in range(permutation_count):
                seed = None if fold is None else permutation_seed(
                    split_seed, fold, feature, repetition
                )
                rows.append(self._delta_row("permutation", baseline, feature, scope, fold,
                                            count, 0.1 * (repetition + 1), repetition, seed))
        return rows

    @staticmethod
    def _delta_row(
        experiment: str, baseline: str, target: str, scope: str,
        fold: int | None, count: int, delta: float,
        repetition: int | None = None, seed: int | None = None,
    ) -> FeatureEvidence:
        return FeatureEvidence(experiment, baseline, target, scope, fold, repetition, seed,
                               count, 10.0, 10.0 + delta, delta,
                               classify_mae_delta(delta))
