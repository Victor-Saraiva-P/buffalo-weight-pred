from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
from contextlib import contextmanager
from typing import Iterator

from buffalo_weight.models import (
    CNN_MASK_MODEL,
    FEATURE_FUSION_MODELS,
    MASK_PREDICTION_MODELS,
    ModelConfig,
)


MANIFEST_FILE = "provenance.json"
COMPARISON_FILES = ("model_comparison.csv", "model_comparison.png")
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class TrainingEvidence:
    split_rows: list[dict[str, str]]
    feature_rows: list[dict[str, str]]
    feature_columns: list[str]
    masks_dir: Path | None
    device: str


@dataclass(frozen=True)
class ArtifactPlan:
    config: ModelConfig
    status: str
    reasons: tuple[str, ...]


def plan_artifacts(
    output_dir: Path, model_configs: list[ModelConfig], evidence: TrainingEvidence
) -> list[ArtifactPlan]:
    return [artifact_plan(output_dir, config, evidence) for config in model_configs]


def prepare_artifacts(
    output_dir: Path,
    model_configs: list[ModelConfig],
    evidence: TrainingEvidence,
    dry_run: bool = False,
) -> tuple[list[ArtifactPlan], list[ModelConfig]]:
    plans = plan_artifacts(output_dir, model_configs, evidence)
    if not dry_run and any(plan.status == "new" or plan.status == "stale" for plan in plans):
        for plan in plans:
            remove_stale_artifact(output_dir, plan)
        for filename in COMPARISON_FILES:
            (output_dir / filename).unlink(missing_ok=True)
    pending = [plan.config for plan in plans if plan.status != "reuse"]
    return plans, pending


def print_artifact_plan(plans: list[ArtifactPlan]) -> None:
    for plan in plans:
        reason = f" ({', '.join(plan.reasons)})" if plan.reasons else ""
        print(f"{plan.status}: {plan.config.name}{reason}")


@contextmanager
def training_lock(output_dir: Path) -> Iterator[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".train.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError(f"training output directory was locked by {lock_path}; expected one active run") from error
    try:
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def artifact_plan(output_dir: Path, config: ModelConfig, evidence: TrainingEvidence) -> ArtifactPlan:
    expected = expected_manifest(config, evidence)
    manifest_path = output_dir / config.name / MANIFEST_FILE
    if not manifest_path.is_file():
        return ArtifactPlan(config, "new", ("manifest missing",))
    actual = _read_json(manifest_path)
    reasons = manifest_differences(actual, expected, output_dir / config.name)
    return ArtifactPlan(config, "reuse" if not reasons else "stale", tuple(reasons))


def expected_manifest(config: ModelConfig, evidence: TrainingEvidence) -> dict[str, object]:
    source_names, source_hash = recipe_source(config)
    return {
        "manifest_version": MANIFEST_VERSION,
        "model_config": config.name,
        "model": config.model,
        "params": _canonical(config.params),
        "recipe_sources": source_names,
        "recipe_hash": source_hash,
        "dependency_versions": dependency_versions(config),
        "input_hash": input_hash(config, evidence),
        "device": _resolved_device(config, evidence.device),
    }


def manifest_differences(
    actual: dict[str, object], expected: dict[str, object], output_dir: Path
) -> list[str]:
    reasons = [
        key for key, value in expected.items() if key != "device" and actual.get(key) != value
    ]
    for filename in ("fold_metrics.csv", "predictions.csv"):
        path = output_dir / filename
        recorded = actual.get(f"{filename}_sha256")
        if not path.is_file() or recorded != file_hash(path):
            reasons.append(f"{filename} changed")
    return reasons


def remove_stale_artifact(output_dir: Path, plan: ArtifactPlan) -> None:
    if plan.status in {"new", "stale"}:
        shutil.rmtree(output_dir / plan.config.name, ignore_errors=True)


def write_manifest(output_dir: Path, config: ModelConfig, evidence: TrainingEvidence) -> None:
    manifest = expected_manifest(config, evidence)
    model_dir = output_dir / config.name
    manifest["fold_metrics.csv_sha256"] = file_hash(model_dir / "fold_metrics.csv")
    manifest["predictions.csv_sha256"] = file_hash(model_dir / "predictions.csv")
    _write_json(model_dir / MANIFEST_FILE, manifest)


def recipe_source(config: ModelConfig) -> tuple[list[str], str]:
    shared = _module_text("buffalo_weight.train")
    selected = _selected_source(config)
    sources = ["buffalo_weight.train:predict_fold_weights", *selected[0]]
    return sources, _digest([shared, selected[1]])


def dependency_versions(config: ModelConfig) -> dict[str, str | None]:
    names = {"numpy": "numpy", "scikit-learn": "scikit-learn"}
    if config.model in MASK_PREDICTION_MODELS or config.model in FEATURE_FUSION_MODELS:
        names["pillow"] = "pillow"
    if config.model == CNN_MASK_MODEL:
        names["torch"] = "torch"
        if config.params.get("architecture") in {"efficientnet_b0", "mobilenet_v3_small", "resnet18"}:
            names["torchvision"] = "torchvision"
        if config.params.get("input_representation") == "geometry_channels":
            names["scipy"] = "scipy"
    if config.model == "pretrained_mask_embedding":
        names["torchvision"] = "torchvision"
    if config.model == "xgboost":
        names["xgboost"] = "xgboost"
    if config.model == "pca_feature_fusion":
        names["scipy"] = "scipy"
    return {name: _package_version(module) for name, module in names.items()}


def _resolved_device(config: ModelConfig, requested: str) -> str:
    if requested != "auto" or config.model not in MASK_PREDICTION_MODELS:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def input_hash(config: ModelConfig, evidence: TrainingEvidence) -> str:
    rows = evidence.split_rows
    parts = [_canonical(_all_rows(rows))]
    if config.model not in MASK_PREDICTION_MODELS and config.model not in FEATURE_FUSION_MODELS:
        rows = evidence.feature_rows
    selected_rows = [_selected_row(row, evidence.feature_columns) for row in rows]
    parts.append(_canonical(selected_rows))
    if config.model in MASK_PREDICTION_MODELS or config.model in FEATURE_FUSION_MODELS:
        parts.append(_mask_hash(evidence.masks_dir, rows))
    return _digest(parts)


def _selected_row(row: dict[str, str], columns: list[str]) -> dict[str, str]:
    names = ["file_name", "weight", "weight_category", "fold", *columns]
    return {name: row.get(name, "") for name in dict.fromkeys(names)}


def _all_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(sorted(row.items())) for row in sorted(rows, key=lambda item: item.get("file_name", ""))]


def _mask_hash(masks_dir: Path | None, rows: list[dict[str, str]]) -> str:
    if masks_dir is None:
        return "no-masks-directory"
    values = []
    for row in sorted(rows, key=lambda item: item.get("file_name", "")):
        path = _find_mask(masks_dir, row.get("file_name", ""))
        values.append((row.get("file_name", ""), file_hash(path) if path else "missing"))
    return _digest([_canonical(values)])


def _find_mask(directory: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def _selected_source(config: ModelConfig) -> tuple[list[str], str]:
    if config.model == CNN_MASK_MODEL:
        architecture = str(config.params.get("architecture", "baseline"))
        from buffalo_weight.cnn_architectures import MASK_NETWORK_RECIPE_SYMBOLS

        functions = MASK_NETWORK_RECIPE_SYMBOLS.get(architecture, ("build_mask_network",))
        symbols = ["CnnMaskRegressor", "load_mask_inputs", "load_masks", "load_mask", "find_mask_path"]
        if config.params.get("input_representation") == "geometry_channels":
            symbols.append("geometry_channels")
        return _source_bundle(
            [
                *[("buffalo_weight.cnn_architectures", symbol) for symbol in functions],
                *[("buffalo_weight.cnn_mask", symbol) for symbol in symbols],
            ]
        )
    module = {
        "pca_feature_fusion": "pca_feature_fusion",
        "pca_svr_mask": "pca_svr_mask",
        "mask_feature": "mask_classical",
        "pretrained_mask_embedding": "pretrained_mask_embedding",
    }.get(config.model, "models")
    symbol = {
        "random_forest": "_build_random_forest",
        "extra_trees": "_build_extra_trees",
        "hist_gradient_boosting": "_build_hist_gradient_boosting",
        "xgboost": "_build_xgboost",
        "pca_feature_fusion": "PcaFeatureFusionRegressor",
        "pca_svr_mask": "PcaSvrMaskRegressor",
        "mask_feature": "MaskFeatureRegressor",
        "pretrained_mask_embedding": "PretrainedMaskEmbeddingRegressor",
    }.get(config.model, "build_model")
    pairs = [(f"buffalo_weight.{module}", symbol)]
    if module == "models":
        pairs.append(("buffalo_weight.models", "_target_regressor"))
    if config.model == "pca_feature_fusion":
        pairs.extend(
            [
                ("buffalo_weight.canonical_mask", "canonicalize_mask"),
                ("buffalo_weight.canonical_mask", "principal_axis_angle"),
                ("buffalo_weight.canonical_mask", "load_canonical_masks"),
                ("buffalo_weight.pca_feature_fusion", "_feature_value"),
                ("buffalo_weight.target_transform", "transform_target"),
                ("buffalo_weight.target_transform", "inverse_target"),
                ("buffalo_weight.target_transform", "transform_target_power"),
                ("buffalo_weight.target_transform", "inverse_target_power"),
            ]
        )
    if config.model in MASK_PREDICTION_MODELS:
        pairs.extend(
            [
                ("buffalo_weight.cnn_mask", "load_masks"),
                ("buffalo_weight.cnn_mask", "load_mask"),
                ("buffalo_weight.cnn_mask", "find_mask_path"),
            ]
        )
    return _source_bundle(pairs)


def _source_bundle(pairs: list[tuple[str, str]]) -> tuple[list[str], str]:
    names = [f"{module}:{symbol}" for module, symbol in pairs]
    source = [_source_text(module, symbol) for module, symbol in pairs]
    return names, _digest(source)


def _source_text(module_name: str, symbol_name: str) -> str:
    module = importlib.import_module(module_name)
    value = getattr(module, symbol_name)
    return inspect.getsource(value)


def _module_text(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return Path(str(module.__file__)).read_text()


def _package_version(module_name: str) -> str | None:
    try:
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(values: list[object]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode()).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    temp_path.replace(path)
