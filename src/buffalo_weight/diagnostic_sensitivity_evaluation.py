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
from buffalo_weight.feature_confirmation_manifest import validate_frozen_feature_contract
from buffalo_weight.feature_evaluation import (
    FeatureBaseline,
    FeatureSample,
    PredictionPartition,
    TrainingPartition,
)
from buffalo_weight.feature_selection_io import load_feature_samples
from buffalo_weight.reproduction_config import ReportContract


SCALE_SHIFT_PERTURBATIONS: tuple[PerturbationKind, ...] = (
    "scale_shrink", "scale_grow",
    "shift_up", "shift_down", "shift_left", "shift_right",
)

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
        """Load a mask PNG file as a 2D binary float32 array."""
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

    Example: ``evaluate_sensitivity(contract)`` returns a SensitivitySlice.
    """
    names = validate_frozen_feature_contract(contract)
    samples = load_feature_samples(contract.inputs_output_dir, names)
    loader = mask_loader or FilesystemMaskLoader(contract.inputs.masks_dir)
    masks = {s.file_name: loader.load_mask(s.file_name) for s in samples}
    canonical_ls = contract.inputs.canonical_long_side
    eligibilities = compute_all_eligibilities(masks, canonical_ls)
    eligible_names = {e.file_name for e in eligibilities if e.status == "eligible"}
    records = _evaluate_all_tabular_folds(samples, names, masks, eligibilities, eligible_names, canonical_ls, random_forest_baseline)
    ordered = tuple(sorted(records, key=lambda r: (r.configuration, r.evaluation_scope, r.file_name, r.perturbation)))
    return SensitivitySlice(ordered, tuple(eligibilities))


def _evaluate_all_tabular_folds(
    samples: list[FeatureSample], names: tuple[str, ...], masks: dict[str, np.ndarray],
    eligibilities: list[MorphologyEligibility], eligible_names: set[str],
    canonical_ls: int, rf_baseline: FeatureBaseline | None,
) -> list[SensitivityPerturbationRecord]:
    from buffalo_weight.feature_baselines import RandomForestBaseline
    model = rf_baseline or RandomForestBaseline()
    records: list[SensitivityPerturbationRecord] = []
    for fold in sorted({s.fold for s in samples}):
        records.extend(_evaluate_single_fold(samples, fold, names, masks, eligibilities, eligible_names, canonical_ls, model))
    return records


def _evaluate_single_fold(
    samples: list[FeatureSample], fold: int, names: tuple[str, ...], masks: dict[str, np.ndarray],
    eligibilities: list[MorphologyEligibility], eligible_names: set[str],
    canonical_ls: int, model: FeatureBaseline,
) -> list[SensitivityPerturbationRecord]:
    train = [s for s in samples if s.fold != fold]
    held_out = [s for s in samples if s.fold == fold]
    predictor = model.fit(_to_training_partition(train, names), names)
    orig_preds = predictor.predict(_to_prediction_partition(held_out, names))
    orig_by_name = {s.file_name: float(orig_preds[i]) for i, s in enumerate(held_out)}
    records: list[SensitivityPerturbationRecord] = []
    for kind in SCALE_SHIFT_PERTURBATIONS:
        records.extend(_predict_perturbed_tabular(held_out, masks, names, canonical_ls, predictor, orig_by_name, kind, model.name, "baseline"))
    for kind in MORPHOLOGY_PERTURBATIONS:
        records.extend(_predict_morphological_tabular(held_out, masks, names, canonical_ls, predictor, orig_by_name, kind, model.name, "baseline", eligibilities))
    return records


def _predict_perturbed_tabular(
    held_out: list[FeatureSample], masks: dict[str, np.ndarray], names: tuple[str, ...],
    canonical_ls: int, predictor: object, orig_by_name: dict[str, float],
    kind: PerturbationKind, config: str, scope: str,
) -> list[SensitivityPerturbationRecord]:
    records: list[SensitivityPerturbationRecord] = []
    for sample in held_out:
        mask = masks[sample.file_name]
        perturbed = _apply_perturbation(mask, kind, canonical_ls)
        if perturbed is None:
            records.append(SensitivityPerturbationRecord(config, scope, sample.file_name, kind, "rejected", "shift_cuts_foreground", orig_by_name[sample.file_name], float("nan"), float("nan")))
            continue
        records.append(_predict_single_sample_record(sample.file_name, perturbed, names, canonical_ls, predictor, orig_by_name[sample.file_name], kind, config, scope))
    return records


def _predict_morphological_tabular(
    held_out: list[FeatureSample], masks: dict[str, np.ndarray], names: tuple[str, ...],
    canonical_ls: int, predictor: object, orig_by_name: dict[str, float],
    kind: PerturbationKind, config: str, scope: str, eligibilities: list[MorphologyEligibility],
) -> list[SensitivityPerturbationRecord]:
    elig_by_name = {e.file_name: e for e in eligibilities}
    records: list[SensitivityPerturbationRecord] = []
    for sample in held_out:
        elig = elig_by_name[sample.file_name]
        orig_val = orig_by_name[sample.file_name]
        if elig.status == "rejected":
            records.append(SensitivityPerturbationRecord(config, scope, sample.file_name, kind, "rejected", elig.rejection_reason, orig_val, float("nan"), float("nan")))
            continue
        mask = masks[sample.file_name]
        perturbed = _apply_perturbation(mask, kind, canonical_ls)
        records.append(_predict_single_sample_record(sample.file_name, perturbed, names, canonical_ls, predictor, orig_val, kind, config, scope))
    return records


def _predict_single_sample_record(
    file_name: str, perturbed: np.ndarray, names: tuple[str, ...], canonical_ls: int,
    predictor: object, orig_value: float, kind: PerturbationKind, config: str, scope: str,
) -> SensitivityPerturbationRecord:
    features = calculate_mask_features(perturbed, canonical_ls)
    vec = np.asarray([[features.get(name, 0.0) for name in names]], dtype=np.float64)
    pred_val = float(predictor.predict(PredictionPartition(vec, (file_name,)))[0])
    return SensitivityPerturbationRecord(config, scope, file_name, kind, "eligible", "", orig_value, pred_val, pred_val - orig_value)


def _apply_perturbation(
    mask: np.ndarray, kind: PerturbationKind, canonical_long_side: int,
) -> np.ndarray | None:
    if kind == "scale_shrink":
        return perturb_scale_shrink(mask)
    if kind == "scale_grow":
        return perturb_scale_grow(mask)
    if kind.startswith("shift_"):
        shift_px = compute_shift_pixels(mask)
        dy, dx = _shift_direction(kind, shift_px)
        result, cuts = perturb_shift(mask, dy, dx)
        return None if cuts else result
    return _apply_morphological_perturbation(mask, kind, canonical_long_side)


def _apply_morphological_perturbation(
    mask: np.ndarray, kind: PerturbationKind, canonical_long_side: int,
) -> np.ndarray:
    disk = euclidean_disk(MORPHOLOGY_DISK_RADIUS_CANONICAL, canonical_long_side, max(mask.shape))
    if kind == "contraction":
        return perturb_contraction(mask, disk)
    if kind == "expansion":
        return perturb_expansion(mask, disk)
    raise ValueError(f"perturbation kind was {kind!r}; expected valid perturbation")


def _shift_direction(kind: PerturbationKind, shift_px: int) -> tuple[int, int]:
    shifts = {"shift_up": (-shift_px, 0), "shift_down": (shift_px, 0), "shift_left": (0, -shift_px), "shift_right": (0, shift_px)}
    if kind not in shifts:
        raise ValueError(f"shift kind was {kind!r}; expected shift_up/down/left/right")
    return shifts[kind]


def _to_training_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
) -> TrainingPartition:
    values = np.asarray([[s.feature_values[name] for name in feature_names] for s in samples], dtype=np.float64)
    targets = np.asarray([s.weight_kg for s in samples], dtype=np.float64)
    return TrainingPartition(values, targets, tuple(s.file_name for s in samples), tuple(s.weight_category for s in samples))


def _to_prediction_partition(
    samples: list[FeatureSample], feature_names: tuple[str, ...],
) -> PredictionPartition:
    values = np.asarray([[s.feature_values[name] for name in feature_names] for s in samples], dtype=np.float64)
    return PredictionPartition(values, tuple(s.file_name for s in samples))
