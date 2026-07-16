from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from buffalo_weight.canonical_mask import load_canonical_masks
from buffalo_weight.cnn_mask import find_mask_path, load_masks
from buffalo_weight.split import assign_folds, assign_weight_categories
from buffalo_weight.train import format_metric


QUALITY_FIELDS = [
    "file_name", "width", "height", "resolution", "foreground_ratio", "component_count",
    "largest_component_fraction", "hole_ratio", "touches_border",
    "pixel_hash",
]
SHAPE_COLUMNS = [
    "solidity", "circularity", "aspect_ratio", "extent", "convexity",
    "hu_moment_1", "hu_moment_2", "middle_to_end_ratio", "centroid_x_offset", "centroid_y_ratio",
]


def audit_masks(masks_dir: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Measure segmentation morphology; for example, ``audit_masks(path, rows)``."""
    return [_audit_mask(masks_dir, row["file_name"]) for row in rows]


def _audit_mask(masks_dir: Path, file_name: str) -> dict[str, str]:
    path = find_mask_path(masks_dir, file_name)
    mask = np.asarray(Image.open(path).convert("L")) > 0
    labeled, component_count = label(mask)
    component_sizes = np.bincount(labeled.reshape(-1))[1:]
    area = int(mask.sum())
    holes = int(binary_fill_holes(mask).sum()) - area
    touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    return {
        "file_name": file_name,
        "width": str(mask.shape[1]),
        "height": str(mask.shape[0]),
        "resolution": f"{mask.shape[1]}x{mask.shape[0]}",
        "foreground_ratio": format_metric(area / mask.size),
        "component_count": str(component_count),
        "largest_component_fraction": format_metric(float(component_sizes.max() / area)),
        "hole_ratio": format_metric(holes / area),
        "touches_border": str(int(touches)),
        "pixel_hash": hashlib.sha256(mask.tobytes()).hexdigest(),
    }


def quality_error_associations(
    audit_rows: list[dict[str, str]], absolute_error_by_name: dict[str, float]
) -> list[dict[str, str]]:
    """Associate mask morphology with OOF error; for example, ``quality_error_associations(rows, errors)``."""
    metrics = ("foreground_ratio", "component_count", "largest_component_fraction", "hole_ratio", "touches_border")
    errors = np.asarray([absolute_error_by_name[row["file_name"]] for row in audit_rows])
    results = []
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in audit_rows])
        correlation, p_value = spearmanr(values, errors)
        results.append({
            "metric": metric,
            "spearman_abs_error": format_metric(float(correlation)),
            "p_value": format_metric(float(p_value)),
        })
    return results


def _feature_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    return np.asarray([[float(row[column].replace(",", ".")) for column in columns] for row in rows])


def classification_test(matrix: np.ndarray, labels: list[str]) -> dict[str, str]:
    """Cross-validate domain classification; for example, ``classification_test(features, farms)``."""
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    label_array = np.asarray(labels)
    scores = []
    for train, validation in splitter.split(matrix, labels):
        model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=42)
        model.fit(matrix[train], label_array[train])
        scores.append(balanced_accuracy_score(label_array[validation], model.predict(matrix[validation])))
    return {"balanced_accuracy_mean": format_metric(float(np.mean(scores))), "balanced_accuracy_std": format_metric(float(np.std(scores)))}


def domain_tests(rows: list[dict[str, str]], feature_columns: list[str], audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Test farm and resolution signals; for example, ``domain_tests(rows, columns, audit)``."""
    farms = [row["farm"] for row in rows]
    weights = np.asarray([float(row["weight"]) for row in rows])
    resolutions = {row["file_name"]: row["resolution"] for row in audit_rows}
    resolution_labels = ["high" if resolutions[row["file_name"]] == "4032x3024" else "low" for row in rows]
    common = (weights >= 92) & (weights <= 265)
    tests = [
        _classification_row("farm_all_features", classification_test(_feature_matrix(rows, feature_columns), farms), len(rows)),
        _classification_row("farm_weight_only", classification_test(weights[:, None], farms), len(rows)),
        _classification_row("resolution_all_features", classification_test(_feature_matrix(rows, feature_columns), resolution_labels), len(rows)),
    ]
    common_rows = [row for row, selected in zip(rows, common, strict=True) if selected]
    common_farms = [farm for farm, selected in zip(farms, common, strict=True) if selected]
    tests.append(_classification_row("farm_common_support_shape", classification_test(_feature_matrix(common_rows, SHAPE_COLUMNS), common_farms), len(common_rows)))
    tests.extend(_farm_transfer_rows(rows, feature_columns, common))
    return tests


def _classification_row(name: str, scores: dict[str, str], count: int) -> dict[str, str]:
    return {"test": name, "n": str(count), "metric": "balanced_accuracy", **scores}


def _farm_transfer_rows(rows: list[dict[str, str]], columns: list[str], common: np.ndarray) -> list[dict[str, str]]:
    matrix = _feature_matrix(rows, columns)
    weights = np.asarray([float(row["weight"]) for row in rows])
    farms = np.asarray([row["farm"] for row in rows])
    results = []
    for train_farm, test_farm in (("Faco", "Manezinho"), ("Manezinho", "Faco")):
        train = (farms == train_farm) & common
        test = (farms == test_farm) & common
        model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=42)
        model.fit(matrix[train], weights[train])
        mae = mean_absolute_error(weights[test], model.predict(matrix[test]))
        results.append({"test": f"transfer_{train_farm}_to_{test_farm}", "n": str(test.sum()), "metric": "mae", "mae": format_metric(mae)})
    return results


def feature_weight_correlations(rows: list[dict[str, str]], feature: str) -> list[dict[str, str]]:
    """Correlate a mask feature and weight by farm; for example, ``feature_weight_correlations(rows, "area")``."""
    results = []
    for farm in sorted({row["farm"] for row in rows}):
        selected = [row for row in rows if row["farm"] == farm]
        values = np.asarray([float(row[feature]) for row in selected])
        weights = np.asarray([float(row["weight"]) for row in selected])
        correlation, p_value = pearsonr(values, weights)
        results.append({"farm": farm, "feature": feature, "n": str(len(selected)), "pearson": format_metric(float(correlation)), "p_value": format_metric(float(p_value))})
    return results


def nearest_visual_pairs(masks_dir: Path, rows: list[dict[str, str]], count: int = 20) -> tuple[list[dict[str, str]], np.ndarray]:
    """Find similar canonical masks with different weights; for example, ``nearest_visual_pairs(path, rows)``."""
    masks = load_canonical_masks(masks_dir, rows, 64, "letterbox") > 0
    flat = masks.reshape(len(rows), -1).astype(np.float32)
    intersections = flat @ flat.T
    areas = flat.sum(axis=1)
    iou = intersections / np.maximum(areas[:, None] + areas[None, :] - intersections, 1)
    pairs = _rank_pairs(rows, iou, count)
    return pairs, masks.astype(np.float32)


def _rank_pairs(rows: list[dict[str, str]], iou: np.ndarray, count: int) -> list[dict[str, str]]:
    candidates = []
    for first in range(len(rows)):
        for second in range(first + 1, len(rows)):
            weight_difference = abs(float(rows[first]["weight"]) - float(rows[second]["weight"]))
            candidates.append((weight_difference * iou[first, second], first, second, weight_difference))
    ranked = sorted(candidates, reverse=True)[:count]
    return [
        {
            "file_name_a": rows[first]["file_name"], "file_name_b": rows[second]["file_name"],
            "weight_a": rows[first]["weight"], "weight_b": rows[second]["weight"],
            "weight_difference": format_metric(difference), "canonical_iou": format_metric(float(iou[first, second])),
        }
        for _, first, second, difference in ranked
    ]


def integrity_tests(
    rows: list[dict[str, str]], audit_rows: list[dict[str, str]], canonical_masks: np.ndarray
) -> list[dict[str, str]]:
    """Check exact and near duplicates; for example, ``integrity_tests(rows, audit, masks)``."""
    feature_columns = sorted(key for key in rows[0] if key not in {"file_name", "farm", "weight", "tag"})
    feature_vectors = [tuple(row[column] for column in feature_columns) for row in rows]
    hashes = [row["pixel_hash"] for row in audit_rows]
    flat = canonical_masks.reshape(len(rows), -1).astype(np.float32)
    intersections = flat @ flat.T
    areas = flat.sum(axis=1)
    iou = intersections / np.maximum(areas[:, None] + areas[None, :] - intersections, 1)
    upper = iou[np.triu_indices(len(rows), 1)]
    return [
        {"test": "duplicate_file_names", "value": str(len(rows) - len({row["file_name"] for row in rows}))},
        {"test": "duplicate_pixel_masks", "value": str(len(hashes) - len(set(hashes)))},
        {"test": "duplicate_feature_vectors", "value": str(len(feature_vectors) - len(set(feature_vectors)))},
        {"test": "canonical_pairs_iou_ge_095", "value": str(int((upper >= 0.95).sum()))},
        {"test": "maximum_canonical_iou", "value": format_metric(float(upper.max()))},
    ]


def scale_ablation_tests(rows: list[dict[str, str]], feature_columns: list[str]) -> list[dict[str, str]]:
    """Compare absolute scale and shape-only regressors; for example, ``scale_ablation_tests(rows, columns)``."""
    split_rows = [row.copy() for row in rows]
    assign_weight_categories(split_rows, 10)
    assign_folds(split_rows, 5, 42)
    configurations = {"all_features": feature_columns, "shape_only": SHAPE_COLUMNS, "area_only": ["area"]}
    results = []
    for name, columns in configurations.items():
        predictions, actual = [], []
        matrix = _feature_matrix(split_rows, columns)
        weights = np.asarray([float(row["weight"]) for row in split_rows])
        folds = np.asarray([int(row["fold"]) for row in split_rows])
        for fold in sorted(set(folds)):
            train, validation = folds != fold, folds == fold
            model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, random_state=42)
            model.fit(matrix[train], weights[train])
            predictions.extend(model.predict(matrix[validation]).tolist())
            actual.extend(weights[validation].tolist())
        results.append({"representation": name, "mae": format_metric(mean_absolute_error(actual, predictions))})
    return results


def nearest_full_representation_pairs(
    masks_dir: Path, rows: list[dict[str, str]], feature_columns: list[str], count: int = 20
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Find contradictions in the model input space; for example, ``nearest_full_representation_pairs(path, rows, columns)``."""
    raw = load_masks(masks_dir, rows, 96, "stretch").reshape(len(rows), -1)
    canonical = load_canonical_masks(masks_dir, rows, 96, "letterbox").reshape(len(rows), -1)
    raw_components = PCA(24, svd_solver="randomized", random_state=42).fit_transform(raw)
    canonical_components = PCA(16, svd_solver="randomized", random_state=42).fit_transform(canonical)
    representation = StandardScaler().fit_transform(
        np.column_stack((_feature_matrix(rows, feature_columns), raw_components, canonical_components))
    )
    distances, indexes = NearestNeighbors(n_neighbors=2).fit(representation).kneighbors(representation)
    pairs = _unique_neighbor_pairs(rows, distances[:, 1], indexes[:, 1])
    differences = np.asarray([float(row["weight_difference"]) for row in pairs])
    summary = {
        "mean_nearest_weight_difference": format_metric(float(differences.mean())),
        "median_nearest_weight_difference": format_metric(float(np.median(differences))),
        "p90_nearest_weight_difference": format_metric(float(np.quantile(differences, 0.9))),
    }
    ranked = sorted(pairs, key=lambda row: float(row["weight_difference"]) / max(float(row["distance"]), 1e-9), reverse=True)
    return ranked[:count], summary


def _unique_neighbor_pairs(
    rows: list[dict[str, str]], distances: np.ndarray, neighbors: np.ndarray
) -> list[dict[str, str]]:
    pairs = {}
    for first, second in enumerate(neighbors):
        low, high = sorted((first, int(second)))
        difference = abs(float(rows[low]["weight"]) - float(rows[high]["weight"]))
        pairs[low, high] = {
            "file_name_a": rows[low]["file_name"], "file_name_b": rows[high]["file_name"],
            "weight_a": rows[low]["weight"], "weight_b": rows[high]["weight"],
            "weight_difference": format_metric(difference), "distance": format_metric(float(distances[first])),
        }
    return list(pairs.values())
