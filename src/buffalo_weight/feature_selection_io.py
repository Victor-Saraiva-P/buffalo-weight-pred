"""CSV boundaries for feature-selection inputs and tidy evidence."""

from __future__ import annotations

import csv
from pathlib import Path

from buffalo_weight.csv_io import format_csv_number
from buffalo_weight.feature_evaluation import FeatureEvidence, FeatureSample
from buffalo_weight.feature_redundancy import FeatureRedundancy
from buffalo_weight.feature_selection_contract import EVIDENCE_COLUMNS, REDUNDANCY_COLUMNS
from buffalo_weight.input_schema import FEATURE_COLUMNS, SPLIT_COLUMNS


def load_feature_samples(
    inputs_dir: Path, feature_names: tuple[str, ...]
) -> list[FeatureSample]:
    """Join public input artifacts; for example, each mask receives its canonical fold."""
    feature_rows = _read_expected_csv(inputs_dir / "feature_index.csv", FEATURE_COLUMNS)
    split_rows = _read_expected_csv(inputs_dir / "canonical_split.csv", SPLIT_COLUMNS)
    split_by_name = {row["file_name"]: row for row in split_rows}
    feature_names_in_rows = {row["file_name"] for row in feature_rows}
    if set(split_by_name) != feature_names_in_rows:
        raise ValueError(
            f"feature/split names were {feature_names_in_rows!r}/{set(split_by_name)!r}; "
            "expected identical file_name sets"
        )
    return [_sample(row, split_by_name[row["file_name"]], feature_names) for row in feature_rows]


def write_feature_evidence(path: Path, evidence: list[FeatureEvidence]) -> None:
    """Write deterministic tidy evidence; for example, fold rows precede OOF summaries."""
    rows = [_evidence_record(row) for row in sorted(evidence, key=_evidence_sort_key)]
    _write_csv(path, EVIDENCE_COLUMNS, rows)


def write_feature_redundancy(path: Path, rows: list[FeatureRedundancy]) -> None:
    """Write all feature pairs; for example, nullable correlations become empty fields."""
    records = [{
        "feature_a": row.feature_a, "feature_b": row.feature_b,
        "structural_relation": row.structural_relation,
        "pearson": _optional_number(row.pearson), "spearman": _optional_number(row.spearman),
        "removal_group": row.removal_group,
    } for row in rows]
    _write_csv(path, REDUNDANCY_COLUMNS, records)


def _sample(
    feature_row: dict[str, str], split_row: dict[str, str], feature_names: tuple[str, ...]
) -> FeatureSample:
    name = feature_row["file_name"]
    try:
        values = {feature: float(feature_row[feature]) for feature in feature_names}
        return FeatureSample(name, int(split_row["fold"]), split_row["weight_category"],
                             float(feature_row["weight_kg"]), values)
    except (KeyError, ValueError) as error:
        raise ValueError(f"feature row was {name!r}; expected finite numeric features and fold") from error


def _read_expected_csv(path: Path, expected: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if fields != expected:
        raise ValueError(f"CSV columns were {fields!r} at {path}; expected exactly {expected!r}")
    return rows


def _evidence_record(row: FeatureEvidence) -> dict[str, str]:
    return {
        "experiment": row.experiment, "baseline": row.baseline, "target": row.target,
        "scope": row.scope, "fold": _optional_int(row.fold),
        "repetition": _optional_int(row.repetition),
        "permutation_seed": _optional_int(row.permutation_seed), "n": str(row.n),
        "reference_mae_kg": _optional_number(row.reference_mae_kg),
        "result_mae_kg": format_csv_number(row.result_mae_kg),
        "delta_mae_kg": _optional_number(row.delta_mae_kg), "effect": row.effect or "",
    }


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _optional_number(value: float | None) -> str:
    if value is None:
        return ""
    formatted = format_csv_number(value)
    return formatted


def _optional_int(value: int | None) -> str:
    if value is None:
        return ""
    formatted = str(value)
    return formatted


def _evidence_sort_key(row: FeatureEvidence) -> tuple[object, ...]:
    scope_rank = 0 if row.scope == "fold" else 1
    key = (row.experiment, row.baseline, row.target, scope_rank,
           row.fold or 0, row.repetition or 0)
    return key
