"""Canonical weight categories and evaluation folds."""

from __future__ import annotations

from collections import Counter

from sklearn.model_selection import StratifiedKFold

from buffalo_weight.curated_inputs import ValidMask


def canonical_split_rows(
    masks: list[ValidMask], category_count: int, fold_count: int, seed: int
) -> list[dict[str, str]]:
    """Build the canonical split; for example, ``canonical_split_rows(masks, 10, 5, 42)``."""
    categories = _weight_categories(masks, category_count)
    folds = _stratified_folds(masks, categories, fold_count, seed)
    rows = [_split_row(mask, categories[mask.file_name], folds[mask.file_name]) for mask in masks]
    _validate_distribution(rows, category_count, fold_count)
    return rows


def _weight_categories(masks: list[ValidMask], count: int) -> dict[str, str]:
    ordered = sorted(masks, key=lambda mask: (mask.weight_kg, mask.file_name))
    total = len(ordered)
    return {
        mask.file_name: f"B{min(index * count // total + 1, count)}"
        for index, mask in enumerate(ordered)
    }


def _stratified_folds(
    masks: list[ValidMask], categories: dict[str, str], count: int, seed: int
) -> dict[str, int]:
    labels = [categories[mask.file_name] for mask in masks]
    splitter = StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
    assignments: dict[str, int] = {}
    for fold, (_, reserved) in enumerate(splitter.split(labels, labels), start=1):
        for index in reserved:
            assignments[masks[int(index)].file_name] = fold
    return assignments


def _split_row(mask: ValidMask, category: str, fold: int) -> dict[str, str]:
    return {
        "file_name": mask.file_name,
        "farm": mask.farm,
        "weight_kg": _format_number(mask.weight_kg),
        "weight_category": category,
        "fold": str(fold),
    }


def _validate_distribution(rows: list[dict[str, str]], categories: int, folds: int) -> None:
    category_counts = Counter(row["weight_category"] for row in rows)
    fold_counts = Counter(row["fold"] for row in rows)
    _validate_balanced_counts(category_counts, len(rows), categories, "category")
    _validate_balanced_counts(fold_counts, len(rows), folds, "fold")
    for fold in range(1, folds + 1):
        within = Counter(row["weight_category"] for row in rows if row["fold"] == str(fold))
        _validate_fold_categories(within, category_counts, folds, fold)


def _validate_balanced_counts(
    counts: Counter[str], total: int, group_count: int, label: str
) -> None:
    expected = {total // group_count, -(-total // group_count)}
    if len(counts) != group_count or not set(counts.values()) <= expected:
        raise ValueError(f"{label} counts were {dict(counts)}; expected balanced {group_count} groups")


def _validate_fold_categories(
    actual: Counter[str], totals: Counter[str], folds: int, fold: int
) -> None:
    for category, total in totals.items():
        expected = {total // folds, -(-total // folds)}
        if actual[category] not in expected:
            raise ValueError(
                f"fold {fold} category {category} count was {actual[category]}; expected {sorted(expected)}"
            )


def _format_number(value: float) -> str:
    return f"{value:.6f}"
