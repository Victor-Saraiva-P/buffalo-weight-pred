"""Completeness checks for comparative feature evidence."""

from __future__ import annotations

from buffalo_weight.feature_evaluation import FeatureEvidence, FeatureSample, RemovalGroup
from buffalo_weight.feature_selection_rules import permutation_seed


def validate_feature_evidence(
    evidence: list[FeatureEvidence], samples: list[FeatureSample],
    features: tuple[str, ...], groups: tuple[RemovalGroup, ...],
    permutation_count: int, split_seed: int,
) -> None:
    """Validate full experiment coverage; for example, missing neutral results are rejected."""
    folds = sorted({sample.fold for sample in samples})
    targets = (*features, *(group.name for group in groups))
    expected = 2 * (len(folds) + 1) * (
        len(features) + len(targets) + len(features) * permutation_count
    )
    if len(evidence) != expected:
        raise ValueError(f"feature evidence rows were {len(evidence)}; expected exactly {expected}")
    _validate_evidence_keys(evidence)
    _validate_fold_membership(evidence, folds)
    _validate_permutation_seeds(evidence, split_seed)


def _validate_evidence_keys(evidence: list[FeatureEvidence]) -> None:
    keys = [(row.experiment, row.baseline, row.target, row.scope, row.fold, row.repetition)
            for row in evidence]
    if len(keys) != len(set(keys)):
        raise ValueError("feature evidence keys were duplicated; expected unique experiment rows")
    allowed = {"isolated", "removal", "permutation"}
    invalid = sorted({row.experiment for row in evidence} - allowed)
    if invalid:
        raise ValueError(f"feature experiments were {invalid!r}; expected only {sorted(allowed)!r}")


def _validate_fold_membership(evidence: list[FeatureEvidence], folds: list[int]) -> None:
    invalid = [row.fold for row in evidence if
               (row.scope == "fold" and row.fold not in folds)
               or (row.scope == "oof" and row.fold is not None)]
    if invalid:
        raise ValueError(f"evidence folds were {invalid!r}; expected folds {folds!r} or null OOF")


def _validate_permutation_seeds(evidence: list[FeatureEvidence], split_seed: int) -> None:
    for row in evidence:
        if row.experiment != "permutation":
            continue
        expected = None if row.scope == "oof" else permutation_seed(
            split_seed, _required_int(row.fold, "fold"), row.target,
            _required_int(row.repetition, "repetition"),
        )
        if row.permutation_seed != expected:
            raise ValueError(
                f"permutation seed was {row.permutation_seed!r}; expected {expected!r} "
                f"for {row.target}/{row.scope}"
            )


def _required_int(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"permutation {name} was null; expected an integer for fold evidence")
    return value
