"""Completeness and artifact checks for comparative feature evidence."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from buffalo_weight.csv_io import csv_columns
from buffalo_weight.feature_evaluation import FeatureEvidence, FeatureSample, RemovalGroup
from buffalo_weight.feature_selection_manifest import expected_csv_schemas
from buffalo_weight.feature_selection_contract import REMOVAL_GROUPS, STRUCTURAL_RELATIONS
from buffalo_weight.feature_selection_rules import classify_mae_delta, permutation_seed
from buffalo_weight.feature_selection_types import (
    EVIDENCE_EFFECTS,
    EVIDENCE_SCOPES,
    FEATURE_BASELINES,
    FEATURE_EXPERIMENTS,
    canonical_evidence_sort_key,
)
from buffalo_weight.png_artifact import read_png_artifact_spec

SIX_DECIMALS = re.compile(r"^-?\d+\.\d{6}$")


def validate_feature_evidence(
    evidence: list[FeatureEvidence], samples: list[FeatureSample],
    features: tuple[str, ...], groups: tuple[RemovalGroup, ...],
    permutation_count: int, split_seed: int,
) -> None:
    """Validate experiment coverage; for example, missing neutral results are rejected."""
    folds = sorted({sample.fold for sample in samples})
    expected_keys = _expected_evidence_keys(features, groups, folds, permutation_count)
    actual_keys = {_evidence_key(row) for row in evidence}
    if len(evidence) != len(actual_keys) or actual_keys != expected_keys:
        missing, extra = sorted(expected_keys - actual_keys), sorted(actual_keys - expected_keys)
        raise ValueError(
            f"feature evidence keys had duplicates={len(evidence) != len(actual_keys)}, "
            f"missing={missing!r}, extra={extra!r}; expected exactly the canonical experiment keys"
        )
    fold_sizes = {fold: sum(sample.fold == fold for sample in samples) for fold in folds}
    for row in evidence:
        _validate_evidence_values(row, fold_sizes, len(samples), split_seed)


def validate_feature_selection_artifacts(
    output_dir: Path, features: tuple[str, ...]
) -> None:
    """Validate public files; for example, schemas, ordering and 300 DPI are mandatory."""
    validate_feature_selection_evidence_files(output_dir, features)
    provisional_contract = json.loads((output_dir / "shared_feature_contract.json").read_text())
    if (provisional_contract.get("status"), provisional_contract.get("selected_features"),
            provisional_contract.get("human_decision")) != ("provisional", None, None):
        raise ValueError(
            f"shared feature contract gate was {provisional_contract!r}; "
            "expected provisional status with null selection and human decision"
        )


def validate_feature_selection_evidence_files(
    output_dir: Path, features: tuple[str, ...]
) -> None:
    """Validate shared evidence; for example, promotion checks copied CSVs and figures."""
    _validate_csv_schemas(output_dir)
    _validate_evidence_csv(output_dir / "feature_predictive_evidence.csv")
    _validate_redundancy_csv(output_dir / "feature_redundancy.csv", features)
    _validate_figures(output_dir)


def _expected_evidence_keys(
    features: tuple[str, ...], groups: tuple[RemovalGroup, ...],
    folds: list[int], permutation_count: int,
) -> set[tuple[object, ...]]:
    keys: set[tuple[object, ...]] = set()
    targets = (*features, *(group.name for group in groups))
    scopes: list[tuple[str, int | None]] = [("fold", fold) for fold in folds] + [("oof", None)]
    for baseline in ("random_forest", "dense"):
        for scope, fold in scopes:
            keys.update(("isolated", baseline, target, scope, fold, None) for target in features)
            keys.update(("removal", baseline, target, scope, fold, None) for target in targets)
            keys.update(("permutation", baseline, target, scope, fold, repetition)
                        for target in features for repetition in range(permutation_count))
    return keys


def _evidence_key(row: FeatureEvidence) -> tuple[object, ...]:
    return (
        row.experiment, row.baseline, row.target, row.scope, row.fold, row.repetition,
    )


def _validate_evidence_values(
    row: FeatureEvidence, fold_sizes: dict[int, int], total_count: int, split_seed: int,
) -> None:
    expected_count = total_count if row.scope == "oof" else fold_sizes.get(row.fold or -1)
    if row.n != expected_count or not math.isfinite(row.result_mae_kg):
        raise ValueError(
            f"evidence n/result was {row.n}/{row.result_mae_kg} for {_evidence_key(row)!r}; "
            f"expected n={expected_count} and finite result MAE"
        )
    if row.experiment == "isolated":
        _validate_isolated_values(row)
        return
    _validate_delta_values(row)
    if row.experiment == "permutation":
        _validate_permutation_seed(row, split_seed)
    elif row.permutation_seed is not None:
        raise ValueError(
            f"removal permutation seed was {row.permutation_seed!r}; expected null"
        )


def _validate_isolated_values(row: FeatureEvidence) -> None:
    values = (row.reference_mae_kg, row.delta_mae_kg, row.effect, row.repetition,
              row.permutation_seed)
    if values != (None, None, None, None, None):
        raise ValueError(f"isolated evidence fields were {values!r}; expected all optional fields null")


def _validate_delta_values(row: FeatureEvidence) -> None:
    if row.reference_mae_kg is None or row.delta_mae_kg is None or row.effect is None:
        raise ValueError(f"delta evidence was {row!r}; expected reference, delta and effect")
    if not math.isfinite(row.reference_mae_kg) or not math.isfinite(row.delta_mae_kg):
        raise ValueError(
            f"delta evidence values were {row.reference_mae_kg}/{row.delta_mae_kg}; "
            "expected finite kilograms"
        )
    calculated = row.result_mae_kg - row.reference_mae_kg
    expected_effect = classify_mae_delta(row.delta_mae_kg)
    if not math.isclose(row.delta_mae_kg, calculated, abs_tol=1e-9) or row.effect != expected_effect:
        raise ValueError(
            f"delta/effect was {row.delta_mae_kg}/{row.effect}; "
            f"expected {calculated}/{expected_effect} from result minus reference"
        )


def _validate_permutation_seed(row: FeatureEvidence, split_seed: int) -> None:
    expected = None if row.scope == "oof" else permutation_seed(
        split_seed, _required_int(row.fold, "fold"), row.target,
        _required_int(row.repetition, "repetition"),
    )
    if row.permutation_seed != expected:
        raise ValueError(
            f"permutation seed was {row.permutation_seed!r}; expected {expected!r} "
            f"for {row.target}/{row.scope}"
        )


def _validate_csv_schemas(output_dir: Path) -> None:
    for name, expected in expected_csv_schemas().items():
        actual = csv_columns(output_dir / name)
        if actual != expected:
            raise ValueError(f"selection columns were {actual!r} for {name}; expected {expected!r}")


def _validate_evidence_csv(path: Path) -> None:
    rows = _read_csv(path)
    keys = [_csv_evidence_key(row) for row in rows]
    if keys != sorted(keys):
        raise ValueError(f"evidence ordering began {keys[:3]!r}; expected canonical sorted keys")
    invalid = [row for row in rows if not _valid_csv_evidence_row(row)]
    if invalid:
        raise ValueError(f"evidence CSV rows included {invalid[:1]!r}; expected canonical types/nullability")


def _validate_redundancy_csv(path: Path, features: tuple[str, ...]) -> None:
    rows = _read_csv(path)
    expected = [(first, second) for index, first in enumerate(features)
                for second in features[index + 1 :]]
    actual = [(row["feature_a"], row["feature_b"]) for row in rows]
    if actual != expected:
        raise ValueError(f"redundancy pair order began {actual[:3]!r}; expected {expected[:3]!r}")
    invalid = [row for row in rows if not _valid_redundancy_row(row)]
    if invalid:
        raise ValueError(
            f"redundancy CSV rows included {invalid[:1]!r}; expected canonical enums/correlations"
        )


def _valid_redundancy_row(row: dict[str, str]) -> bool:
    allowed_groups = {group.name for group in REMOVAL_GROUPS} | {"none"}
    allowed_relations = {name for name, _ in STRUCTURAL_RELATIONS}
    relation_parts = row["structural_relation"].split("|")
    relation_valid = relation_parts == ["none"] or (
        "none" not in relation_parts and set(relation_parts).issubset(allowed_relations)
    )
    correlations_valid = all(_valid_correlation(row[name]) for name in ("pearson", "spearman"))
    return row["removal_group"] in allowed_groups and relation_valid and correlations_valid


def _valid_correlation(value: str) -> bool:
    if value == "":
        return True
    if not SIX_DECIMALS.fullmatch(value):
        return False
    numeric_value = float(value)
    return -1.0 <= numeric_value <= 1.0


def _valid_csv_evidence_row(row: dict[str, str]) -> bool:
    if not _valid_csv_base_fields(row):
        return False
    if row["experiment"] == "isolated":
        return not any(row[name] for name in (
            "reference_mae_kg", "delta_mae_kg", "effect", "repetition", "permutation_seed"
        ))
    numeric = all(SIX_DECIMALS.fullmatch(row[name]) for name in
                  ("reference_mae_kg", "delta_mae_kg"))
    if not numeric or row["effect"] not in EVIDENCE_EFFECTS:
        return False
    return _valid_csv_experiment_fields(row)


def _valid_csv_base_fields(row: dict[str, str]) -> bool:
    allowed = row["experiment"] in FEATURE_EXPERIMENTS
    allowed = allowed and row["baseline"] in FEATURE_BASELINES
    allowed = allowed and row["scope"] in EVIDENCE_SCOPES and bool(row["target"])
    allowed = allowed and row["n"].isdigit() and bool(SIX_DECIMALS.fullmatch(row["result_mae_kg"]))
    fold_valid = row["fold"].isdigit() if row["scope"] == "fold" else row["fold"] == ""
    return allowed and fold_valid


def _valid_csv_experiment_fields(row: dict[str, str]) -> bool:
    if row["experiment"] == "removal":
        return row["repetition"] == row["permutation_seed"] == ""
    repetition_valid = row["repetition"].isdigit()
    seed_valid = row["permutation_seed"].isdigit() if row["scope"] == "fold" else (
        row["permutation_seed"] == ""
    )
    return repetition_valid and seed_valid


def _csv_evidence_key(row: dict[str, str]) -> tuple[object, ...]:
    key = canonical_evidence_sort_key(
        row["experiment"], row["baseline"], row["target"], row["scope"],
        int(row["fold"] or 0), int(row["repetition"] or 0),
    )
    return key


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    return rows


def _validate_figures(output_dir: Path) -> None:
    for name in ("redundancy_heatmap.png", "removal_heatmap.png", "permutation_effects.png"):
        specification = read_png_artifact_spec(output_dir / name)
        actual = (specification.width_px, specification.height_px, specification.dpi)
        if actual != (2400, 1800, 300):
            raise ValueError(f"figure shape/DPI was {actual!r} for {name}; expected (2400, 1800, 300)")


def _required_int(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"permutation {name} was null; expected an integer for fold evidence")
    return value
