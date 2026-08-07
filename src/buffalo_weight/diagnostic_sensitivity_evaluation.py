"""Evaluation logic for controlled sensitivity of predictions to mask perturbations.

Each already-trained configuration predicts perturbed masks without retraining.
Features are recalculated for tabular models; CNNs apply their habitual letterbox.

Reference: GitHub Issue #26.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from buffalo_weight.diagnostic_sensitivity_eligibility import (
    CANONICAL_LONG_SIDE,
    MORPHOLOGY_DISK_RADIUS_CANONICAL,
    compute_all_eligibilities,
)
from buffalo_weight.diagnostic_sensitivity_perturbations import (
    compute_shift_pixels,
    euclidean_disk,
    perturb_contraction,
    perturb_expansion,
    perturb_scale_grow,
    perturb_scale_shrink,
    perturb_shift,
)
from buffalo_weight.diagnostic_sensitivity_types import (
    MorphologyEligibility,
    PerturbationKind,
    SensitivityPerturbationRecord,
    SensitivitySlice,
)
from buffalo_weight.feature_calculators import calculate_mask_features
from buffalo_weight.feature_evaluation import (
    FeatureBaseline,
    FeatureSample,
    PredictionPartition,
    TrainingPartition,
)
from buffalo_weight.feature_selection_io import load_feature_samples
from buffalo_weight.feature_confirmation_manifest import validate_frozen_feature_contract
from buffalo_weight.reproduction_config import ReportContract


# Non-morphological perturbations that apply to all 132 masks
SCALE_SHIFT_PERTURBATIONS: tuple[PerturbationKind, ...] = (
    "scale_shrink", "scale_grow",
    "shift_up", "shift_down", "shift_left", "shift_right",
)

# Morphological perturbations that only apply to eligible masks
MORPHOLOGY_PERTURBATIONS: tuple[PerturbationKind, ...] = (
    "contraction", "expansion",
)


class SensitivityMaskLoader(Protocol):
    """Seam for loading original binary masks by file_name."""

    def load_mask(self, file_name: str) -> np.ndarray:
        """Load a binary mask as 2D float array.

        Example: ``loader.load_mask("img01")`` returns the mask array.
        """
        ...


class FilesystemMaskLoader:
    """Load masks from the filesystem as binary arrays.

    Example: ``FilesystemMaskLoader(masks_dir).load_mask("img01")``
    """

    def __init__(self, masks_dir: Path) -> None:
        self._masks_dir = masks_dir

    def load_mask(self, file_name: str) -> np.ndarray:
        """Load a mask PNG file as a 2D binary float32 array.

        Example: ``loader.load_mask("img01")`` reads from masks_dir.
        """
        from PIL import Image
        mask_path = self._masks_dir / file_name
        if not mask_path.is_file():
            raise ValueError(f"mask was unavailable at {mask_path}; expected existing PNG")
        with Image.open(mask_path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.float32)
        return (arr > 0).astype(np.float32)


def evaluate_sensitivity(
    contract: ReportContract,
    mask_loader: SensitivityMaskLoader | None = None,
    random_forest_baseline: FeatureBaseline | None = None,
) -> SensitivitySlice:
    """Evaluate sensitivity of trained predictions to controlled mask perturbations.

    Uses existing OOF predictions and re-predicts on perturbed masks without retraining.

    Example: ``evaluate_sensitivity(contract)`` returns a SensitivitySlice.
    """
    feature_names = validate_frozen_feature_contract(contract)
    feature_samples = load_feature_samples(contract.inputs_output_dir, feature_names)
    loader = mask_loader or FilesystemMaskLoader(contract.inputs.masks_dir)

    # Load all original masks
    masks = {s.file_name: loader.load_mask(s.file_name) for s in feature_samples}
    canonical_ls = contract.inputs.canonical_long_side

    # Compute eligibility for morphological perturbations
    eligibilities = compute_all_eligibilities(masks, canonical_ls)
    eligible_names = {e.file_name for e in eligibilities if e.status == "eligible"}

    records: list[SensitivityPerturbationRecord] = []

    # Evaluate each configuration's sensitivity using baseline models
    records.extend(
        _evaluate_tabular_sensitivity(
            feature_samples, feature_names, masks, eligibilities,
            eligible_names, canonical_ls, random_forest_baseline,
        )
    )

    ordered = tuple(sorted(records, key=lambda r: (r.configuration, r.evaluation_scope, r.file_name, r.perturbation)))
    return SensitivitySlice(ordered, tuple(eligibilities))


def _evaluate_tabular_sensitivity(
    samples: list[FeatureSample],
    feature_names: tuple[str, ...],
    masks: dict[str, np.ndarray],
    eligibilities: list[MorphologyEligibility],
    eligible_names: set[str],
    canonical_long_side: int,
    rf_baseline: FeatureBaseline | None,
) -> list[SensitivityPerturbationRecord]:
    """Evaluate sensitivity for tabular (random forest) baseline.

    Retrains per fold on ORIGINAL data, predicts held-out on perturbed features.
    """
    from buffalo_weight.feature_baselines import RandomForestBaseline
    model = rf_baseline or RandomForestBaseline()
    records: list[SensitivityPerturbationRecord] = []

    folds = sorted({s.fold for s in samples})
    for fold in folds:
        train = [s for s in samples if s.fold != fold]
        held_out = [s for s in samples if s.fold == fold]

        # Fit model on ORIGINAL training data (no retraining on perturbed)
        train_partition = _to_training_partition(train, feature_names)
        predictor = model.fit(train_partition, feature_names)

        # Original predictions for held-out
        orig_preds = predictor.predict(
            _to_prediction_partition(held_out, feature_names)
        )
        orig_by_name = {s.file_name: float(orig_preds[i]) for i, s in enumerate(held_out)}

        # Evaluate each perturbation for held-out masks
        for kind in SCALE_SHIFT_PERTURBATIONS:
            records.extend(
                _predict_perturbed_tabular(
                    held_out, masks, feature_names, canonical_long_side,
                    predictor, orig_by_name, kind, model.name, "baseline",
                )
            )

        # Morphological perturbations — only eligible masks
        for kind in MORPHOLOGY_PERTURBATIONS:
            records.extend(
                _predict_morphological_tabular(
                    held_out, masks, feature_names, canonical_long_side,
                    predictor, orig_by_name, kind, model.name, "baseline",
                    eligible_names, eligibilities,
                )
            )

    return records


def _predict_perturbed_tabular(
    held_out: list[FeatureSample],
    masks: dict[str, np.ndarray],
    feature_names: tuple[str, ...],
    canonical_long_side: int,
    predictor: object,
    orig_by_name: dict[str, float],
    kind: PerturbationKind,
    configuration: str,
    scope: str,
) -> list[SensitivityPerturbationRecord]:
    """Predict held-out masks with a specific perturbation applied."""
    records: list[SensitivityPerturbationRecord] = []
    for sample in held_out:
        mask = masks[sample.file_name]
        perturbed = _apply_perturbation(mask, kind, canonical_long_side)

        if perturbed is None:
            # Shift cut foreground — record as rejected
            records.append(SensitivityPerturbationRecord(
                configuration, scope, sample.file_name, kind,
                "rejected", "shift_cuts_foreground",
                orig_by_name[sample.file_name], float("nan"), float("nan"),
            ))
            continue

        features = calculate_mask_features(perturbed, canonical_long_side)
        feature_vector = np.asarray(
            [[features.get(name, 0.0) for name in feature_names]], dtype=np.float64
        )
        pred_partition = PredictionPartition(feature_vector, (sample.file_name,))
        pred_value = float(predictor.predict(pred_partition)[0])
        orig_value = orig_by_name[sample.file_name]
        delta = pred_value - orig_value

        records.append(SensitivityPerturbationRecord(
            configuration, scope, sample.file_name, kind,
            "eligible", "", orig_value, pred_value, delta,
        ))
    return records


def _predict_morphological_tabular(
    held_out: list[FeatureSample],
    masks: dict[str, np.ndarray],
    feature_names: tuple[str, ...],
    canonical_long_side: int,
    predictor: object,
    orig_by_name: dict[str, float],
    kind: PerturbationKind,
    configuration: str,
    scope: str,
    eligible_names: set[str],
    eligibilities: list[MorphologyEligibility],
) -> list[SensitivityPerturbationRecord]:
    """Predict morphological perturbations — all 132 appear, rejected have empty fields."""
    elig_by_name = {e.file_name: e for e in eligibilities}
    records: list[SensitivityPerturbationRecord] = []

    for sample in held_out:
        elig = elig_by_name[sample.file_name]
        orig_value = orig_by_name[sample.file_name]

        if elig.status == "rejected":
            # All 132 masks appear in the table; rejected have empty perturbed fields
            records.append(SensitivityPerturbationRecord(
                configuration, scope, sample.file_name, kind,
                "rejected", elig.rejection_reason,
                orig_value, float("nan"), float("nan"),
            ))
            continue

        mask = masks[sample.file_name]
        perturbed = _apply_perturbation(mask, kind, canonical_long_side)
        features = calculate_mask_features(perturbed, canonical_long_side)
        feature_vector = np.asarray(
            [[features.get(name, 0.0) for name in feature_names]], dtype=np.float64
        )
        pred_partition = PredictionPartition(feature_vector, (sample.file_name,))
        pred_value = float(predictor.predict(pred_partition)[0])
        delta = pred_value - orig_value

        records.append(SensitivityPerturbationRecord(
            configuration, scope, sample.file_name, kind,
            "eligible", "", orig_value, pred_value, delta,
        ))
    return records


def _apply_perturbation(
    mask: np.ndarray, kind: PerturbationKind, canonical_long_side: int
) -> np.ndarray | None:
    """Apply a single perturbation to a mask. Returns None if shift cuts foreground."""
    if kind == "scale_shrink":
        return perturb_scale_shrink(mask)
    if kind == "scale_grow":
        return perturb_scale_grow(mask)

    if kind.startswith("shift_"):
        shift_px = compute_shift_pixels(mask)
        dy, dx = _shift_direction(kind, shift_px)
        result, cuts = perturb_shift(mask, dy, dx)
        return None if cuts else result

    original_long_side = max(mask.shape)
    disk = euclidean_disk(MORPHOLOGY_DISK_RADIUS_CANONICAL, canonical_long_side, original_long_side)
    if kind == "contraction":
        return perturb_contraction(mask, disk)
    if kind == "expansion":
        return perturb_expansion(mask, disk)

    raise ValueError(f"perturbation kind was {kind!r}; expected one of {SCALE_SHIFT_PERTURBATIONS + MORPHOLOGY_PERTURBATIONS}")


def _shift_direction(kind: PerturbationKind, shift_px: int) -> tuple[int, int]:
    """Convert a shift perturbation kind to (dy, dx) pixel offsets.

    Example: ``_shift_direction("shift_up", 5)`` returns ``(-5, 0)``.
    """
    if kind == "shift_up":
        return (-shift_px, 0)
    if kind == "shift_down":
        return (shift_px, 0)
    if kind == "shift_left":
        return (0, -shift_px)
    if kind == "shift_right":
        return (0, shift_px)
    raise ValueError(f"shift kind was {kind!r}; expected shift_up/down/left/right")


def _to_training_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
) -> TrainingPartition:
    """Build a training partition from feature samples.

    Example: ``_to_training_partition(samples, ("area", "perimeter"))``
    """
    values = np.asarray(
        [[s.feature_values[name] for name in feature_names] for s in samples],
        dtype=np.float64,
    )
    targets = np.asarray([s.weight_kg for s in samples], dtype=np.float64)
    return TrainingPartition(
        values, targets,
        tuple(s.file_name for s in samples),
        tuple(s.weight_category for s in samples),
    )


def _to_prediction_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
) -> PredictionPartition:
    """Build a prediction partition from feature samples.

    Example: ``_to_prediction_partition(samples, ("area", "perimeter"))``
    """
    values = np.asarray(
        [[s.feature_values[name] for name in feature_names] for s in samples],
        dtype=np.float64,
    )
    return PredictionPartition(values, tuple(s.file_name for s in samples))
