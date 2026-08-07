"""Subset generation for controlled learning curves evaluation.

Reference: GitHub Issue #25.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class StratifiedFoldSample(Protocol):
    """Protocol for samples used in stratified fold subsampling."""

    @property
    def file_name(self) -> str:
        ...

    @property
    def weight_category(self) -> str:
        ...

    @property
    def fold(self) -> int:
        ...


SampleT = TypeVar("SampleT", bound=StratifiedFoldSample)


def generate_nested_subsets(
    samples: list[SampleT],
    outer_fold: int,
    seed: int = 45,
) -> dict[float, list[SampleT]]:
    """Generate nested, stratified, deterministic subsets (50%, 75%, 100%) for outer training.

    Example: ``generate_nested_subsets(samples, outer_fold=1, seed=45)`` returns {0.5: [...], 0.75: [...], 1.0: [...]}.
    """
    outer_train = sorted(
        [s for s in samples if s.fold != outer_fold],
        key=lambda s: s.file_name,
    )
    if not outer_train:
        raise ValueError(f"outer fold {outer_fold} produced 0 training samples from {len(samples)} inputs")

    total_count = len(outer_train)
    count_75 = round(total_count * 0.75)
    count_50 = round(total_count * 0.50)

    subset_75 = _stratified_sample(outer_train, count_75, seed)
    subset_50 = _stratified_sample(subset_75, count_50, seed)

    return {
        0.50: subset_50,
        0.75: subset_75,
        1.00: outer_train,
    }


def _stratified_sample(
    items: list[SampleT],
    target_count: int,
    seed: int,
) -> list[SampleT]:
    if target_count >= len(items):
        return list(items)

    labels = [s.weight_category for s in items]
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=target_count, random_state=seed)
    dummy_x = np.zeros(len(items))
    train_idx, _ = next(splitter.split(dummy_x, labels))
    selected = [items[int(i)] for i in sorted(train_idx)]
    return selected
