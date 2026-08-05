"""Freshness and integrity for the dense baseline configuration."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from buffalo_weight.dense_baseline_artifacts import (
    validate_dense_baseline_artifacts,
)
from buffalo_weight.dense_baseline_evaluation import (
    DenseBaselineEvaluation,
    DenseFoldAudit,
)
from buffalo_weight.dense_baseline_provenance import DenseBaselineProvenance
from buffalo_weight.feature_baselines import DENSE_BASELINE_RECIPE
from buffalo_weight.feature_selection_io import load_feature_samples
from buffalo_weight.feature_selection_manifest import artifact_output_records
from buffalo_weight.hashing import sha256_file
from buffalo_weight.reproduction_config import ReportContract

OUTPUT_FILES = ("predictions.csv", "fold_metrics.csv")
VALIDATIONS = (
    "schemas", "ordering", "sha256", "one_oof_prediction_per_valid_mask",
    "outer_fold_isolation", "full_outer_retrain", "deterministic_cuda",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def dense_baseline_identity(
    contract: ReportContract, selected_features: tuple[str, ...],
    provenance: DenseBaselineProvenance,
) -> dict[str, object]:
    """Fingerprint reusable work; for example, a changed feature contract invalidates it."""
    confirmed_dir = contract.confirmed_feature_selection_dir
    input_dir = contract.inputs_output_dir
    return {
        "manifest_version": 1, "stage": "baselines", "model_config": "dense",
        "selected_features": list(selected_features), "recipe": dense_recipe_record(),
        "recipe_sha256": provenance.dense_baseline_recipe_hash(),
        "scientific_environment": provenance.scientific_environment(),
        "inputs": _input_hashes(input_dir, confirmed_dir, selected_features),
    }


def dense_baseline_status(
    contract: ReportContract, selected_features: tuple[str, ...],
    provenance: DenseBaselineProvenance,
) -> str:
    """Classify the dense artifact; for example, a tampered CSV is obsolete."""
    output_dir = dense_baseline_output_dir(contract)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        manifest = json.loads(manifest_path.read_text())
        identity = dense_baseline_identity(contract, selected_features, provenance)
        if not isinstance(manifest, dict) or any(manifest.get(key) != value
                                                 for key, value in identity.items()):
            return "obsolete"
        validate_dense_baseline_manifest(manifest, output_dir, contract)
        return "reusable"
    except (OSError, ValueError, TypeError):
        return "obsolete"


def complete_dense_baseline_manifest(
    contract: ReportContract, output_dir: Path, identity: dict[str, object],
    evaluation: DenseBaselineEvaluation, execution: dict[str, object], source_commit: str,
) -> dict[str, object]:
    """Complete provenance last; for example, output hashes prove atomic completion."""
    manifest = identity.copy()
    manifest.update({
        "package_type": "reconstructible_model_configuration", "revision": 1,
        "status": "complete", "command": "python main.py baselines",
        "source_commit": source_commit,
        "execution": execution, "fold_training": _fold_training(evaluation.fold_audits),
        "outputs": artifact_output_records(output_dir, OUTPUT_FILES),
        "validations": list(VALIDATIONS),
    })
    return manifest


def validate_dense_baseline_manifest(
    manifest: dict[str, object], output_dir: Path, contract: ReportContract,
) -> None:
    """Validate completion; for example, a mismatched output hash rejects reuse."""
    _validate_fixed_fields(manifest)
    _validate_execution(manifest.get("execution"))
    _validate_source_commit(manifest.get("source_commit"))
    _validate_fold_training(manifest.get("fold_training"), contract)
    _validate_output_records(manifest, output_dir)
    selected = _validated_selected_features(manifest.get("selected_features"))
    samples = load_feature_samples(contract.inputs_output_dir, selected)
    epochs = _manifest_epochs(manifest["fold_training"])
    validate_dense_baseline_artifacts(output_dir, samples, contract.inputs.fold_count, epochs)


def _validate_fixed_fields(manifest: dict[str, object]) -> None:
    fixed = (manifest.get("package_type"), manifest.get("revision"), manifest.get("status"),
             manifest.get("command"), manifest.get("validations"))
    expected = ("reconstructible_model_configuration", 1, "complete",
                "python main.py baselines", list(VALIDATIONS))
    if fixed != expected:
        raise ValueError(f"dense manifest fixed fields were {fixed!r}; expected {expected!r}")


def _validate_output_records(manifest: dict[str, object], output_dir: Path) -> None:
    expected_outputs = artifact_output_records(output_dir, OUTPUT_FILES)
    if manifest.get("outputs") != expected_outputs:
        raise ValueError(
            f"dense manifest outputs were {manifest.get('outputs')!r}; "
            f"expected {expected_outputs!r}"
        )


def _validated_selected_features(candidate: object) -> tuple[str, ...]:
    selected = candidate
    if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected):
        raise ValueError(f"dense selected_features were {selected!r}; expected a string list")
    return tuple(selected)


def dense_recipe_record() -> dict[str, object]:
    """Expose the frozen recipe; for example, manifests make architecture auditable."""
    recipe = asdict(DENSE_BASELINE_RECIPE)
    recipe["hidden_layers"] = list(DENSE_BASELINE_RECIPE.hidden_layers)
    recipe.update({
        "activation": "relu", "initialization": "he_normal", "output": "linear",
        "batch_normalization": False, "optimizer": "adamw",
        "loss": "l1_standardized_target", "feature_standardization": "training_partition",
        "target_standardization": "training_partition",
    })
    return recipe


def dense_baseline_output_dir(contract: ReportContract) -> Path:
    """Locate the dense configuration; for example, future baselines use sibling paths."""
    return Path(contract.artifacts_root) / "baselines" / "dense"


def _input_hashes(
    inputs_dir: Path, confirmed_dir: Path, selected_features: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    feature_path = inputs_dir / "feature_index.csv"
    split_path = inputs_dir / "canonical_split.csv"
    contract_path = confirmed_dir / "shared_feature_contract.json"
    return {
        "selected_feature_rows": _selected_feature_record(feature_path, selected_features),
        "canonical_split.csv": _file_record(split_path),
        "shared_feature_contract.json": _file_record(contract_path),
    }


def _selected_feature_record(
    path: Path, selected_features: tuple[str, ...],
) -> dict[str, object]:
    columns = ("file_name", "weight_kg", *selected_features)
    with path.open(newline="", encoding="utf-8") as source:
        rows = [{name: row[name] for name in columns} for row in csv.DictReader(source)]
    encoded = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "row_count": len(rows),
            "schema": list(columns)}


def _file_record(path: Path) -> dict[str, object]:
    return {"sha256": sha256_file(path)}


def _fold_training(audits: tuple[DenseFoldAudit, ...]) -> list[dict[str, object]]:
    return [{
        "fold": audit.fold, "selected_epochs": audit.selected_epochs,
        "selection_ids_sha256": _ids_hash(audit.selection_ids),
        "stopping_ids_sha256": _ids_hash(audit.stopping_ids),
        "retrain_ids_sha256": _ids_hash(audit.retrain_ids),
        "held_out_ids_sha256": _ids_hash(audit.held_out_ids),
        "selection_count": len(audit.selection_ids), "stopping_count": len(audit.stopping_ids),
        "retrain_count": len(audit.retrain_ids), "held_out_count": len(audit.held_out_ids),
    } for audit in audits]


def _ids_hash(values: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _validate_source_commit(candidate: object) -> None:
    valid = isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{40}", candidate)
    if not valid:
        raise ValueError(f"dense source_commit was {candidate!r}; expected a 40-character hex SHA")


def _validate_execution(candidate: object) -> None:
    if not isinstance(candidate, dict):
        raise ValueError(f"dense execution was {candidate!r}; expected a mapping")
    actual = (candidate.get("device"), candidate.get("deterministic_algorithms"),
              candidate.get("cudnn_benchmark"))
    expected = ("cuda", True, False)
    if actual != expected:
        raise ValueError(f"dense execution policy was {actual!r}; expected {expected!r}")


def _validate_fold_training(candidate: object, contract: ReportContract) -> None:
    if not isinstance(candidate, list) or len(candidate) != contract.inputs.fold_count:
        count = len(candidate) if isinstance(candidate, list) else candidate
        raise ValueError(
            f"dense fold training records were {count!r}; expected {contract.inputs.fold_count}"
        )
    folds = [record.get("fold") for record in candidate if isinstance(record, dict)]
    expected_folds = list(range(1, contract.inputs.fold_count + 1))
    if folds != expected_folds or not all(_valid_fold_record(record, contract) for record in candidate):
        raise ValueError(
            f"dense fold training records were {candidate!r}; expected folds {expected_folds!r} "
            "with valid partition counts and SHA-256 hashes"
        )


def _valid_fold_record(candidate: object, contract: ReportContract) -> bool:
    if not isinstance(candidate, dict):
        return False
    selection = _positive_int(candidate.get("selection_count"))
    stopping = _positive_int(candidate.get("stopping_count"))
    retrain = _positive_int(candidate.get("retrain_count"))
    held_out = _positive_int(candidate.get("held_out_count"))
    if None in (selection, stopping, retrain, held_out):
        return False
    assert selection is not None and stopping is not None
    assert retrain is not None and held_out is not None
    count_valid = selection + stopping == retrain
    count_valid = count_valid and retrain + held_out == contract.inputs.expected_mask_count
    hashes = [value for key, value in candidate.items() if key.endswith("_sha256")]
    hash_valid = len(hashes) == 4 and all(isinstance(value, str) and SHA256.fullmatch(value)
                                          for value in hashes)
    epochs = candidate.get("selected_epochs")
    return count_valid and hash_valid and isinstance(epochs, int) and epochs > 0


def _positive_int(candidate: object) -> int | None:
    if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0:
        return None
    return candidate


def _manifest_epochs(candidate: object) -> dict[int, int]:
    if not isinstance(candidate, list):
        raise ValueError(f"dense fold_training was {candidate!r}; expected validated records")
    return {record["fold"]: record["selected_epochs"] for record in candidate
            if isinstance(record, dict)
            and isinstance(record.get("fold"), int)
            and isinstance(record.get("selected_epochs"), int)}
