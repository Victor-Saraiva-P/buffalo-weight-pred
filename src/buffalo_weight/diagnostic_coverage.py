"""Characterize sample coverage and cross-tabulations for valid masks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


from collections.abc import Callable


@dataclass(frozen=True)
class DiagnosticCoverageSample:
    """Single sample record for diagnostic coverage analysis."""

    file_name: str
    farm: str
    weight_category: str
    resolution: str
    weight_kg: float


@dataclass(frozen=True)
class DiagnosticCoverageSummary:
    """Summary of coverage counts and cross-tabulations across valid masks."""

    sample_count: int
    category_counts: dict[str, int]
    farm_counts: dict[str, int]
    resolution_counts: dict[str, int]
    crosstab_category_farm: list[dict[str, str | int]]
    crosstab_farm_resolution: list[dict[str, str | int]]
    crosstab_category_resolution: list[dict[str, str | int]]


def compute_sample_coverage(
    samples: list[DiagnosticCoverageSample],
) -> DiagnosticCoverageSummary:
    """Compute sample counts and cross-tabulations across categories, farms, and resolutions.

    Example: ``compute_sample_coverage(samples)`` returns a ``DiagnosticCoverageSummary``.
    """
    if not samples:
        raise ValueError(f"samples were {samples!r}; expected non-empty list of coverage samples")
    cat_counts = dict(Counter(s.weight_category for s in samples))
    farm_counts = dict(Counter(s.farm for s in samples))
    res_counts = dict(Counter(s.resolution for s in samples))
    return DiagnosticCoverageSummary(
        sample_count=len(samples),
        category_counts=cat_counts,
        farm_counts=farm_counts,
        resolution_counts=res_counts,
        crosstab_category_farm=_build_pair_crosstab(
            samples, "weight_category", lambda s: s.weight_category, "farm", lambda s: s.farm,
        ),
        crosstab_farm_resolution=_build_pair_crosstab(
            samples, "farm", lambda s: s.farm, "resolution", lambda s: s.resolution,
        ),
        crosstab_category_resolution=_build_pair_crosstab(
            samples, "weight_category", lambda s: s.weight_category, "resolution", lambda s: s.resolution,
        ),
    )


def _build_pair_crosstab(
    samples: list[DiagnosticCoverageSample],
    key1_name: str,
    key1_func: Callable[[DiagnosticCoverageSample], str],
    key2_name: str,
    key2_func: Callable[[DiagnosticCoverageSample], str],
) -> list[dict[str, str | int]]:
    pairs = Counter((key1_func(s), key2_func(s)) for s in samples)
    return [
        {key1_name: v1, key2_name: v2, "n": n}
        for (v1, v2), n in sorted(pairs.items())
    ]
