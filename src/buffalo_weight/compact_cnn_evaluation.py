"""Outer-fold evaluation for the compact CNN baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedShuffleSplit

from buffalo_weight.compact_cnn_adapter import (
    COMPACT_CNN_RECIPE,
    CompactCnnRecipe,
    CompactCnnTargetScale,
    CompactCnnTrainingAdapter,
    MaskBatch,
)


@dataclass(frozen=True)
class CompactCnnSample:
    file_name: str
    farm: str
    weight_kg: float
    weight_category: str
    fold: int
    mask_path: Path


@dataclass(frozen=True)
class CompactCnnPrediction:
    file_name: str
    farm: str
    weight_category: str
    fold: int
    observed_weight_kg: float
    predicted_weight_kg: float


@dataclass(frozen=True)
class CompactCnnMetric:
    scope: str
    group: str
    fold: int | None
    sample_count: int
    mae_kg: float
    rmse_kg: float
    bias_kg: float
    r2: float


@dataclass(frozen=True)
class CompactCnnTrainingAudit:
    selection_ids: tuple[str, ...]
    stopping_ids: tuple[str, ...]
    refit_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    selected_epochs: int


@dataclass(frozen=True)
class CompactCnnEvaluation:
    predictions: tuple[CompactCnnPrediction, ...]
    metrics: tuple[CompactCnnMetric, ...]
    training_audits: tuple[CompactCnnTrainingAudit, ...]


def evaluate_compact_cnn(
    samples: list[CompactCnnSample], adapter: CompactCnnTrainingAdapter,
    recipe: CompactCnnRecipe = COMPACT_CNN_RECIPE,
) -> CompactCnnEvaluation:
    """Produce OOF evidence; for example, every configured fold is held out once."""
    predictions: list[CompactCnnPrediction] = []
    audits: list[CompactCnnTrainingAudit] = []
    for fold in sorted({sample.fold for sample in samples}):
        fold_predictions, audit = _evaluate_fold(samples, fold, adapter, recipe)
        predictions.extend(fold_predictions)
        audits.append(audit)
    ordered = tuple(sorted(predictions, key=lambda row: row.file_name))
    _validate_oof_predictions(samples, ordered)
    return CompactCnnEvaluation(ordered, tuple(compact_cnn_metrics(ordered)), tuple(audits))


def compact_cnn_metrics(
    predictions: tuple[CompactCnnPrediction, ...],
) -> list[CompactCnnMetric]:
    """Summarize grouped OOF predictions; for example, global MAE is not fold-averaged."""
    rows = [_metric_row("fold", f"fold_{fold}", fold,
                        tuple(row for row in predictions if row.fold == fold))
            for fold in sorted({row.fold for row in predictions})]
    rows.append(_metric_row("oof", "all", None, predictions))
    for category in ("B1", "B10"):
        grouped = tuple(row for row in predictions if row.weight_category == category)
        if grouped:
            rows.append(_metric_row("category", category, None, grouped))
    return rows


def load_compact_cnn_samples(split_path: Path, masks_dir: Path) -> list[CompactCnnSample]:
    """Load canonical sample metadata; for example, file names resolve directly to PNGs."""
    import csv

    with split_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    samples = [_sample_from_row(row, masks_dir) for row in rows]
    if not samples:
        raise ValueError(f"canonical split at {split_path} had 0 rows; expected labelled masks")
    return samples


def load_letterboxed_mask(path: Path, image_size: int = 224) -> NDArray[np.float32]:
    """Read one binary channel; for example, a 2×4 mask becomes centered letterbox input."""
    with Image.open(path) as source:
        image = source.convert("L")
    values = np.unique(np.asarray(image))
    if not set(int(value) for value in values) <= {0, 255}:
        raise ValueError(f"mask values at {path} were {values.tolist()!r}; expected only 0/255")
    resized = _nearest_letterbox(image, image_size)
    binary = (np.asarray(resized, dtype=np.uint8) > 0).astype(np.float32)
    return np.asarray(binary[None, :, :], dtype=np.float32)


def fit_compact_target_scale(targets_kg: NDArray[np.float64]) -> CompactCnnTargetScale:
    """Fit permitted labels; for example, a constant target uses a unit scale."""
    mean, deviation = float(np.mean(targets_kg)), float(np.std(targets_kg))
    return CompactCnnTargetScale(mean, deviation if deviation != 0.0 else 1.0)


def _evaluate_fold(
    samples: list[CompactCnnSample], fold: int, adapter: CompactCnnTrainingAdapter,
    recipe: CompactCnnRecipe,
) -> tuple[list[CompactCnnPrediction], CompactCnnTrainingAudit]:
    external_train = [sample for sample in samples if sample.fold != fold]
    held_out = [sample for sample in samples if sample.fold == fold]
    selection, stopping = _inner_samples(external_train, recipe.inner_seed)
    selection_batch, stopping_batch = _mask_batch(selection), _mask_batch(stopping)
    inner_scale = fit_compact_target_scale(selection_batch.targets_kg)
    epochs = adapter.select_epoch_count(selection_batch, stopping_batch, inner_scale, recipe)
    refit_batch = _mask_batch(external_train)
    predictor = adapter.fit_epochs(
        refit_batch, fit_compact_target_scale(refit_batch.targets_kg), epochs, recipe,
    )
    held_out_batch = _mask_batch(held_out)
    values = predictor.predict_kg(held_out_batch)
    predictions = [_prediction(sample, float(value))
                   for sample, value in zip(held_out, values, strict=True)]
    audit = _training_audit(selection, stopping, external_train, held_out, epochs)
    return predictions, audit


def _inner_samples(
    samples: list[CompactCnnSample], seed: int,
) -> tuple[list[CompactCnnSample], list[CompactCnnSample]]:
    strata = [sample.weight_category for sample in samples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    selection, stopping = next(splitter.split(np.zeros(len(samples)), strata))
    return ([samples[int(index)] for index in selection],
            [samples[int(index)] for index in stopping])


def _mask_batch(samples: list[CompactCnnSample]) -> MaskBatch:
    pixels = np.stack([load_letterboxed_mask(sample.mask_path) for sample in samples])
    targets: NDArray[np.float64] = np.asarray(
        [sample.weight_kg for sample in samples], dtype=np.float64,
    )
    return MaskBatch(pixels, targets, tuple(sample.file_name for sample in samples),
                     tuple(sample.weight_category for sample in samples))


def _sample_from_row(row: dict[str, str], masks_dir: Path) -> CompactCnnSample:
    required = ("file_name", "farm", "weight_kg", "weight_category", "fold")
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(f"canonical split row was {row!r}; expected non-empty fields {required!r}")
    file_name = row["file_name"]
    path = masks_dir / file_name
    if not path.is_file():
        raise ValueError(f"mask path was {path}; expected an indexed PNG file")
    return CompactCnnSample(file_name, row["farm"], float(row["weight_kg"]),
                            row["weight_category"], int(row["fold"]), path)


def _nearest_letterbox(image: Image.Image, image_size: int) -> Image.Image:
    scale = min(image_size / image.width, image_size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.NEAREST)
    canvas = Image.new("L", (image_size, image_size), 0)
    canvas.paste(resized, ((image_size - width) // 2, (image_size - height) // 2))
    return canvas


def _prediction(sample: CompactCnnSample, predicted_kg: float) -> CompactCnnPrediction:
    return CompactCnnPrediction(sample.file_name, sample.farm, sample.weight_category,
                                sample.fold, sample.weight_kg, predicted_kg)


def _metric_row(
    scope: str, group: str, fold: int | None,
    predictions: tuple[CompactCnnPrediction, ...],
) -> CompactCnnMetric:
    observed = np.asarray([row.observed_weight_kg for row in predictions])
    predicted = np.asarray([row.predicted_weight_kg for row in predictions])
    residual = predicted - observed
    r2 = float(r2_score(observed, predicted)) if len(observed) > 1 else 0.0
    return CompactCnnMetric(scope, group, fold, len(observed),
                            float(np.mean(np.abs(residual))),
                            float(np.sqrt(np.mean(residual ** 2))),
                            float(np.mean(residual)), r2)


def _training_audit(
    selection: list[CompactCnnSample], stopping: list[CompactCnnSample],
    refit: list[CompactCnnSample], held_out: list[CompactCnnSample], epochs: int,
) -> CompactCnnTrainingAudit:
    identifiers = lambda rows: tuple(sample.file_name for sample in rows)
    return CompactCnnTrainingAudit(identifiers(selection), identifiers(stopping),
                                   identifiers(refit), identifiers(held_out), epochs)


def _validate_oof_predictions(
    samples: list[CompactCnnSample], predictions: tuple[CompactCnnPrediction, ...],
) -> None:
    expected = sorted(sample.file_name for sample in samples)
    actual = [row.file_name for row in predictions]
    if actual != expected:
        raise ValueError(f"OOF prediction names were {actual!r}; expected exactly {expected!r}")
