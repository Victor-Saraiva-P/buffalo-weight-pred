"""Atomic, reusable execution of the ResNet-18 baseline configuration."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from buffalo_weight.environment_contract import (
    RESNET18_CACHE_PATH,
    RESNET18_SHA256,
    RESNET18_WEIGHT_NAME,
    RuntimeProbe,
)
from buffalo_weight.artifact_provenance import training_lock
from buffalo_weight.csv_io import csv_columns, csv_row_count
from buffalo_weight.curated_inputs import input_hashes
from buffalo_weight.feature_confirmation import require_baselines_gate
from buffalo_weight.hashing import sha256_file
from buffalo_weight.input_schema import SPLIT_COLUMNS
from buffalo_weight.reproduction_config import ReportContract
from buffalo_weight.resnet18_weights import require_offline_resnet18_weights
from buffalo_weight.resnet_baseline_adapter import (
    RESNET18_BASELINE_RECIPE,
    ResNet18BaselineAdapter,
)
from buffalo_weight.resnet_baseline_artifacts import (
    MODEL_CONFIG,
    load_resnet_samples,
    output_metadata,
    validate_output_metadata,
    validate_predictions,
    write_manifest,
    write_resnet_outputs,
)
from buffalo_weight.resnet_baseline_evaluation import (
    ResNetBaselineEvaluator,
    ResNetOofPrediction,
    ResNetSample,
    ResNetTrainingAdapter,
)
from buffalo_weight.resnet_baseline_provenance import (
    ResNetBaselineProvenance,
    SystemResNetBaselineProvenance,
)
from buffalo_weight.snapshot_io import (
    FilesystemSnapshotPublisher,
    SnapshotPublisher,
    clean_snapshot_stage,
)
from buffalo_weight.system_setup import default_runtime_probe, require_official_neural_runtime


class ResNetBaselineRunner(Protocol):
    """Expensive execution seam; for example, CLI tests inject deterministic predictions."""

    def preflight(self) -> None:
        """Validate CUDA and weights; for example, failure precedes mask loading."""
        ...

    def evaluate(self, samples: tuple[ResNetSample, ...]) -> list[ResNetOofPrediction]:
        """Produce OOF predictions; for example, one result is returned per mask."""
        ...

    def execution_metadata(self) -> dict[str, object]:
        """Describe execution; for example, record GPU identity without cache invalidation."""
        ...


class ScientificResNetBaselineRunner:
    """Compose official offline CUDA adapters; for example, the public CLI uses this."""

    def __init__(
        self, adapter: ResNetTrainingAdapter | None = None,
        weights_path: Path = RESNET18_CACHE_PATH, weights_sha256: str = RESNET18_SHA256,
        runtime_probe: RuntimeProbe | None = None,
    ) -> None:
        self._adapter = adapter
        self._weights_path = weights_path
        self._weights_sha256 = weights_sha256
        self._runtime_probe = runtime_probe

    def preflight(self) -> None:
        """Fail early; for example, missing weights instruct the setup command."""
        require_offline_resnet18_weights(self._weights_path, self._weights_sha256)
        require_official_neural_runtime(False, self._runtime_probe)

    def evaluate(self, samples: tuple[ResNetSample, ...]) -> list[ResNetOofPrediction]:
        """Evaluate canonical folds; for example, each refit reloads official weights."""
        adapter = self._adapter or ResNet18BaselineAdapter()
        evaluator = ResNetBaselineEvaluator(adapter, RESNET18_BASELINE_RECIPE.inner_seed)
        return evaluator.evaluate(samples)

    def execution_metadata(self) -> dict[str, object]:
        """Record CUDA audit fields; for example, hardware changes do not invalidate reuse."""
        probe = self._runtime_probe or default_runtime_probe()
        compute = probe.compute_environment()
        return {
            "device": "cuda", "deterministic": True, "official": True,
            "gpu_name": compute.gpu_name, "cuda_capability": compute.cuda_capability,
            "cuda_version": compute.cuda_version, "driver_version": compute.driver_version,
        }


def run_resnet_baseline_stage(
    contract: ReportContract, dry_run: bool = False,
    runner: ResNetBaselineRunner | None = None,
    provenance: ResNetBaselineProvenance | None = None,
    publisher: SnapshotPublisher | None = None,
) -> str:
    """Run or reuse the baseline; for example, dry-run never probes CUDA or weights."""
    require_baselines_gate(contract)
    _require_current_mask_inputs(contract)
    resolved_provenance = provenance or SystemResNetBaselineProvenance()
    status = resnet_baseline_status(contract, resolved_provenance)
    if dry_run or status == "reusable":
        return status
    resolved_runner = runner or ScientificResNetBaselineRunner()
    return _run_locked_rebuild(contract, resolved_runner, resolved_provenance,
                               publisher or FilesystemSnapshotPublisher())


def resnet_baseline_status(
    contract: ReportContract, provenance: ResNetBaselineProvenance
) -> str:
    """Inspect artifact freshness; for example, a changed predictions hash is obsolete."""
    output_dir = resnet_baseline_output_dir(contract)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return "absent"
    try:
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest, output_dir, contract, provenance)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "obsolete"
    return "reusable"


def resnet_baseline_output_dir(contract: ReportContract) -> Path:
    """Locate one configuration; for example, sibling baselines keep separate artifacts."""
    return contract.artifacts_root / "baselines" / MODEL_CONFIG


def _run_locked_rebuild(
    contract: ReportContract, runner: ResNetBaselineRunner,
    provenance: ResNetBaselineProvenance, publisher: SnapshotPublisher,
) -> str:
    with training_lock(contract.artifacts_root / "baselines"):
        status = resnet_baseline_status(contract, provenance)
        if status == "reusable":
            return status
        if status == "obsolete":
            clean_snapshot_stage(resnet_baseline_output_dir(contract))
        runner.preflight()
        _build_snapshot(contract, runner, provenance, publisher)
    return "rebuilt"


def _build_snapshot(
    contract: ReportContract, runner: ResNetBaselineRunner,
    provenance: ResNetBaselineProvenance, publisher: SnapshotPublisher,
) -> None:
    destination = resnet_baseline_output_dir(contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".resnet18-", dir=destination.parent))
    identity = _baseline_identity(contract, provenance)
    try:
        _write_snapshot(temporary, contract, runner, provenance, identity)
        publisher.publish(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_snapshot(
    output_dir: Path, contract: ReportContract, runner: ResNetBaselineRunner,
    provenance: ResNetBaselineProvenance, identity: dict[str, object],
) -> None:
    samples = load_resnet_samples(
        contract.inputs_output_dir / "canonical_split.csv", contract.inputs.masks_dir
    )
    predictions = runner.evaluate(samples)
    validate_predictions(predictions, samples)
    write_resnet_outputs(output_dir, predictions)
    if _baseline_identity(contract, provenance) != identity:
        raise ValueError("ResNet baseline identity changed; expected an unchanged evaluation")
    manifest = _complete_manifest(output_dir, identity, provenance, runner)
    write_manifest(output_dir / "manifest.json", manifest)


def _complete_manifest(
    output_dir: Path, identity: dict[str, object], provenance: ResNetBaselineProvenance,
    runner: ResNetBaselineRunner,
) -> dict[str, object]:
    source_commit = _attested_source_commit(provenance, identity["recipe_hash"])
    return {
        "schema_version": 1, "status": "complete", "stage": "baselines",
        "model_config": MODEL_CONFIG, "identity": identity,
        "source_commit": source_commit,
        "execution": runner.execution_metadata(), "outputs": output_metadata(output_dir),
    }


def _attested_source_commit(
    provenance: ResNetBaselineProvenance, recipe_hash: object
) -> str:
    commit = provenance.repository_commit()
    if provenance.recipe_hash_at_commit(commit) != recipe_hash:
        raise ValueError(
            f"baseline source commit was {commit!r}; expected committed recipe {recipe_hash!r}"
        )
    return commit


def _baseline_identity(
    contract: ReportContract, provenance: ResNetBaselineProvenance
) -> dict[str, object]:
    return {
        "recipe": asdict(RESNET18_BASELINE_RECIPE), "recipe_hash": provenance.recipe_hash(),
        "dependencies": provenance.dependency_versions(),
        "inputs": _input_hashes(contract),
        "weights": {"name": RESNET18_WEIGHT_NAME, "sha256": RESNET18_SHA256},
    }


def _input_hashes(contract: ReportContract) -> dict[str, str]:
    samples = load_resnet_samples(
        contract.inputs_output_dir / "canonical_split.csv", contract.inputs.masks_dir
    )
    return {
        "training_split": _training_split_hash(samples),
        "masks": _mask_collection_hash(samples),
    }


def _training_split_hash(samples: tuple[ResNetSample, ...]) -> str:
    rows = [
        (sample.file_name, sample.weight_category, sample.fold, sample.weight_kg)
        for sample in samples
    ]
    serialized = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _mask_collection_hash(samples: tuple[ResNetSample, ...]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.file_name.encode())
        digest.update(sha256_file(sample.mask_path).encode())
    return digest.hexdigest()


def _validate_manifest(
    manifest: object, output_dir: Path, contract: ReportContract,
    provenance: ResNetBaselineProvenance,
) -> None:
    if not isinstance(manifest, dict):
        raise ValueError(f"baseline manifest was {manifest!r}; expected a mapping")
    fixed = (manifest.get("schema_version"), manifest.get("status"),
             manifest.get("stage"), manifest.get("model_config"))
    expected = (1, "complete", "baselines", MODEL_CONFIG)
    if fixed != expected:
        raise ValueError(f"baseline manifest fields were {fixed!r}; expected {expected!r}")
    identity = _baseline_identity(contract, provenance)
    _validate_audit_fields(manifest, provenance, identity["recipe_hash"])
    if manifest.get("identity") != identity:
        raise ValueError(f"baseline identity was {manifest.get('identity')!r}; expected {identity!r}")
    validate_output_metadata(output_dir, manifest.get("outputs"),
                             contract.inputs.expected_mask_count, contract.inputs.fold_count)


def _validate_audit_fields(
    manifest: dict[str, object], provenance: ResNetBaselineProvenance,
    recipe_hash: object,
) -> None:
    commit = manifest.get("source_commit")
    execution = manifest.get("execution")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"baseline source commit was {commit!r}; expected a full Git SHA")
    if provenance.recipe_hash_at_commit(commit) != recipe_hash:
        raise ValueError(
            f"baseline source commit was {commit!r}; expected source matching {recipe_hash!r}"
        )
    if not isinstance(execution, dict):
        raise ValueError(f"baseline execution was {execution!r}; expected a mapping")
    required = (execution.get("device"), execution.get("deterministic"),
                execution.get("official"))
    if required != ("cuda", True, True):
        raise ValueError(
            f"baseline execution fields were {required!r}; expected ('cuda', True, True)"
        )


def _require_current_mask_inputs(contract: ReportContract) -> None:
    manifest_path = contract.inputs_output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"inputs manifest was unavailable at {manifest_path}; expected current mask inputs"
        ) from error
    _validate_mask_input_manifest(contract, manifest)


def _validate_mask_input_manifest(
    contract: ReportContract, manifest: object
) -> None:
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError(f"inputs manifest was {manifest!r}; expected complete mask inputs")
    if manifest.get("inputs") != input_hashes(contract.inputs):
        raise ValueError(
            f"inputs hashes were {manifest.get('inputs')!r}; expected current mask/index hashes"
        )
    outputs = manifest.get("outputs")
    record = outputs.get("canonical_split.csv") if isinstance(outputs, dict) else None
    _validate_split_record(contract, record)


def _validate_split_record(contract: ReportContract, record: object) -> None:
    split_path = contract.inputs_output_dir / "canonical_split.csv"
    expected = {
        "sha256": sha256_file(split_path), "rows": contract.inputs.expected_mask_count,
        "columns": SPLIT_COLUMNS,
    }
    actual = {
        "sha256": record.get("sha256"), "rows": record.get("rows"),
        "columns": record.get("columns"),
    } if isinstance(record, dict) else record
    if actual != expected or csv_row_count(split_path) != expected["rows"] or (
        csv_columns(split_path) != SPLIT_COLUMNS
    ):
        raise ValueError(f"canonical split record was {actual!r}; expected {expected!r}")
