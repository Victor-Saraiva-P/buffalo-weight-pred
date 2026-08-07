"""Types and data structures for controlled sensitivity diagnostic slice.

Reference: GitHub Issue #26.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SENSITIVITY_PERTURBATION_KINDS = (
    "scale_shrink",
    "scale_grow",
    "shift_up",
    "shift_down",
    "shift_left",
    "shift_right",
    "contraction",
    "expansion",
)

PerturbationKind = Literal[
    "scale_shrink",
    "scale_grow",
    "shift_up",
    "shift_down",
    "shift_left",
    "shift_right",
    "contraction",
    "expansion",
]

MorphologyStatus = Literal["eligible", "rejected"]


@dataclass(frozen=True)
class MorphologyEligibility:
    """Eligibility result for contraction+expansion pair on a single mask.

    Example: ``MorphologyEligibility("img01", "eligible", "")`` marks an eligible mask.
    """

    file_name: str
    status: MorphologyStatus
    rejection_reason: str


@dataclass(frozen=True)
class SensitivityPerturbationRecord:
    """Per-mask, per-perturbation, per-configuration sensitivity record.

    Example: ``SensitivityPerturbationRecord("rf", "baseline", "img01", "scale_shrink", ...)``
    """

    configuration: str
    evaluation_scope: str
    file_name: str
    perturbation: PerturbationKind
    status: MorphologyStatus
    rejection_reason: str
    original_prediction_kg: float
    perturbed_prediction_kg: float
    delta_kg: float


@dataclass(frozen=True)
class SensitivitySlice:
    """Complete diagnostic slice for controlled sensitivity analysis.

    Example: ``SensitivitySlice(records, eligibilities)`` holds all results.
    """

    records: tuple[SensitivityPerturbationRecord, ...]
    eligibilities: tuple[MorphologyEligibility, ...]
